import os
import json
import requests
import socket
import tempfile
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

from concurrent.futures import ThreadPoolExecutor, as_completed
from .vision_tool import process_page_visuals

import dotenv

dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")           # Serper.dev API Key
JINA_API_KEY = os.getenv("JINA_API_KEY")               # Jina API Key
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") # SiliconFlow API Key

CACHE_FILE = ".cache/search_cache.json"                # Local cache file path
_CACHE_LOCK = threading.RLock()
_JINA_READER_REACHABLE = None


def current_date_context() -> str:
    configured = os.getenv("PTAH_CURRENT_DATE", "").strip()
    if configured:
        return f"Current date: {configured}. Use this as today's date."
    now = datetime.now(ZoneInfo(os.getenv("PTAH_TIMEZONE", "Asia/Shanghai")))
    return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}, {now.tzinfo}). Use this as today's date."


def load_cache() -> dict:
    """Load local cache."""
    with _CACHE_LOCK:
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def save_cache(cache: dict):
    """Save cache to local file."""
    with _CACHE_LOCK:
        cache_dir = os.path.dirname(CACHE_FILE)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".search_cache.",
            suffix=".tmp",
            dir=cache_dir or ".",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, CACHE_FILE)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def update_cache_entry(query: str, payload: dict):
    """Merge one cache entry into the latest on-disk cache."""
    with _CACHE_LOCK:
        cache = load_cache()
        cache[query] = payload
        save_cache(cache)


def serper_search(query: str, api_key: str, top_k: int = 5):
    """
    Search Google using Serper.dev.
    Returns a list of: [{'title': ..., 'url': ..., 'snippet': ...}, ...]
    """
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }
    payload = {"q": query}

    print(f"🌐 Searching Google (via Serper) for: {query}")
          
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if not response.ok:
            raise RuntimeError(
                f"Serper API failed: {response.status_code} - {response.text[:200]}"
            )

        data = response.json()
        organic = data.get("organic", [])
        if not organic:
            print("⚠️ No organic search results found.")
            return []

        results = []
        for item in organic[:top_k]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        print(f"✅ Got {len(results)} search results.")
        return results

    except Exception as e:
        print(f"[ERROR] serper_search failed: {e}")
        return []


def fetch_page_content(url: str, jina_api_key: str = None) -> str:
    """Fetch main text content from a web page using direct Jina Reader markdown extraction."""
    try:
        headers = {
            "Authorization": f"Bearer {jina_api_key}" if jina_api_key else "",
            "X-Return-Format": "markdown",
            "X-With-Generated-Alt": "true"
        }
        resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=30)

        if resp.status_code == 200:
            return resp.text.strip()[:10000]
        else:
            return f"[Error] Jina error {resp.status_code}: {resp.text[:200]}"

    except Exception as e:
        return f"[Error] Failed fetching {url}: {e}"


def jina_reader_reachable(timeout: float = 3.0) -> bool:
    """Fast network probe for Jina Reader; avoids slow per-URL failures when blocked."""
    global _JINA_READER_REACHABLE
    if _JINA_READER_REACHABLE is not None:
        return _JINA_READER_REACHABLE

    try:
        with socket.create_connection(("r.jina.ai", 443), timeout=timeout):
            _JINA_READER_REACHABLE = True
            print("✅ Jina Reader reachable: https://r.jina.ai")
    except OSError as e:
        print(f"⚠️ Jina Reader unreachable from this host: {e}. Falling back to Serper snippets.")
        _JINA_READER_REACHABLE = False
    return _JINA_READER_REACHABLE


def build_search_result_with_fallback(item: dict, content: str = "") -> dict | None:
    """Return a grounded result even when Jina page extraction fails."""
    title = (item.get("title") or "").strip()
    url = (item.get("url") or item.get("link") or "").strip()
    snippet = (item.get("snippet") or "").strip()
    content = (content or "").strip()

    if not url:
        return None

    if not content or content.startswith("[Error]"):
        if not title and not snippet:
            return None
        content = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}\n\n"
            "[Fallback note] Jina Reader page extraction was unavailable or failed; "
            "this evidence card is grounded in the search result title, URL, and snippet."
        )

    return {
        "title": title,
        "url": url,
        "snippet": snippet,
        "content": content,
    }


def fetch_multiple_pages(urls, jina_api_key=None, max_workers=8):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(fetch_page_content, url, jina_api_key): url
            for url in urls
        }
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                results[url] = future.result()
            except Exception as exc:
                results[url] = f"[Error] {exc}"
    return results


def build_evidence_cards_prompts(query: str, raw_results: str):
    system_prompt = (
        "You are a rigorous research librarian.\n"
        "Your job: summarize web search results into per-page evidence cards.\n\n"
        "Rules:\n"
        "- Use ONLY the provided raw results. Do NOT add outside knowledge.\n"
        "- Do NOT speculate. If unclear, say 'Not specified'.\n"
        "- Each page must become ONE card.\n"
        "- Prefer extracting exact numbers (with units and time ranges) if present.\n"
        "- If a result looks like marketing/SEO without evidence, mark credibility Low.\n"
        "- Keep each card concise but information-dense.\n"
        "- If the page content is missing/too thin, reflect that in the summary and credibility.\n\n"
        "Output format (must follow exactly):\n"
        "CARD [i]\n"
        "Title: ...\n"
        "URL: ...\n"
        "Overall summary:\n"
        "(A paragraph summarizing what this page is mainly about and its main takeaways)\n"
        "Key claims:\n"
        "- (2-5 bullets)\n"
        "Data & evidence:\n"
        "- (0-5 bullets; include exact numbers, units, dates, sample sizes, metrics if present)\n"
        "Examples/cases:\n"
        "- (0-3 bullets)\n"
        "Key data & tables:\n"
        "- (Extract and preserve any tables, numeric lists, rankings, or parameter tables. ALWAYS provide a descriptive caption starting with 'Table: the table caption' before each table.)\n"
        "----\n"
    )

    user_prompt = f"""
Search Query:
{query}

Raw Search Results (each item includes title/url/snippet/content excerpt):
{raw_results}

Produce per-page evidence cards. Include as many cards as there are valid results.
Follow the output format EXACTLY.
/no_think
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages



def chat_completion(
    messages,
    api_key: str = "EMPTY",
    model: str | None = None,
    max_tokens: int = 4096,
):
    """
    Call Local vLLM OpenAI-compatible chat completion endpoint.
    """

    model = model or os.getenv("PTAH_LLM_MODEL_NAME", "models/Qwen3-32B")
    base_url = os.getenv("PTAH_LLM_BASE_URL", "http://localhost:8000/v1")
    url = f"{base_url}/chat/completions"

    if not api_key:
        api_key = "EMPTY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


    payload = {
        "model": model, 
        "messages": [{"role": "system", "content": current_date_context()}, *messages],
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "temperature": 0.7
    }

    try:
        resp = requests.post(url, headers=headers, json=payload)
        
        if not resp.ok:
            error_msg = f"[Error] vLLM API failed: {resp.status_code} - {resp.text}"
            print(error_msg)
            return error_msg
            
        data = resp.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return content.strip() if content else "[Error] Empty response from vLLM."

    except Exception as e:
        error_msg = f"[Error] vLLM request connection failed: {e}"
        print(error_msg)
        return error_msg


def summarize(query: str, results: list) -> str:
    """
    Build raw_formatted from valid results and ask Qwen3-32B to generate evidence cards.
    """
    raw_formatted = ""
    cnt = 0
    for i, item in enumerate(results, 1):
        content = item.get("content", "")
        if content and not str(content).startswith("[Error]"):
            raw_formatted += (
                f"[{i}] {item.get('title', '').strip()}\n"
                f"URL: {item.get('url', '').strip()}\n"
                f"Snippet: {item.get('snippet', '').strip()}\n"
                f"Content: {content}\n\n"
            )
            cnt += 1

    if not raw_formatted:
        return "No search results found. Please try a different query."

    messages = build_evidence_cards_prompts(query=query, raw_results=raw_formatted)
    summary = chat_completion(
        messages=messages,
        api_key=SILICONFLOW_API_KEY,
    )
    return summary


def search_and_summarize(
    query: str,
    top_k: int = 10,
    do_summarize: bool = True,
    cache_summaries: bool = True
):
    """
    Search + crawl web pages with caching.
    Additionally: call SiliconFlow(Qwen3-32B) to produce an evidence-cards summary.
    """
    cache = load_cache()

    # 1) Cache hit
    if query in cache and isinstance(cache[query], dict):
        cached = cache[query]
        cached_results = cached.get("results", [])
        if len(cached_results) >= top_k:
            print(f"📦 Using cached results for: {query} ({len(cached_results)} results)")
            # if user needs summary but cache doesn't have it, we can regenerate
            if do_summarize and not cached.get("summary_cards"):
                print("🧾 No cached summary found; regenerating summary...")
                summary_cards = summarize(query, cached_results[:top_k])
                cached["summary_cards"] = summary_cards
                if cache_summaries:
                    update_cache_entry(query, cached)
            return cached.get("results", [])[:top_k], cached.get("summary_cards", "")

    # 2) Call Serper
    search_results = serper_search(query, SERPER_API_KEY, top_k)
    urls = [item["url"] for item in search_results if item["url"]]

    if urls and jina_reader_reachable():
        print(f"📄 Fetching {len(urls)} pages...")
        page_texts = fetch_multiple_pages(urls, jina_api_key=JINA_API_KEY, max_workers=5)
    else:
        print(f"📄 Skipping Jina fetch for {len(urls)} pages; using Serper snippets.")
        page_texts = {}

    # 3) Combine results
    output_results = []
    for item in search_results:
        url = item["url"]
        result_item = build_search_result_with_fallback(item, page_texts.get(url, ""))
        if result_item:
            output_results.append(result_item)

    # 4) Summarize
    summary_cards = ""
    if do_summarize:
        print("🧾 Summarizing search results...")
        summary_cards = summarize(query, output_results)

    # 5) Save cache (results + optional summary)
    cache_payload = {"results": output_results}
    if cache_summaries:
        cache_payload["summary_cards"] = summary_cards
    update_cache_entry(query, cache_payload)
    print(f"💾 Cached {len(output_results)} results for: {query}")

    return output_results, summary_cards


def search_with_vision(
    query: str, 
    section_data: dict, 
    top_k: int = 5
) -> tuple:
    """
    Run cached multimodal search:
    1. Check cache:
       - Hit: reuse valid_results content and text_summary.
       - Miss: run search and page fetch.
    2. Run in parallel:
       - Task A: LLM summary when no cached summary exists.
       - Task B: VLM audit on content because visual requirements are section-specific.
    3. Write new search or summary results back to cache.
    
    Returns:
        (text_summary, final_visual_evidence, source_urls)
    """
    
    cache = load_cache()
    valid_results = []
    text_summary = ""
    is_cache_hit = False

    # --- Step 1: Check Cache ---
    if query in cache and isinstance(cache[query], dict):
        cached_payload = cache[query]
        cached_results = cached_payload.get("results", [])
        
        # Any cached result is enough to avoid repeated page fetching.
        if cached_results:
            print(f"📦 Using cached content for: {query}")
            # Limit cached results to top_k.
            valid_results = [
                item
                for item in (
                    build_search_result_with_fallback(res, res.get("content", ""))
                    for res in cached_results[:top_k]
                )
                if item
            ]
            text_summary = cached_payload.get("summary_cards", "")
            is_cache_hit = True

    # --- Step 2: Search & Fetch (If Cache Miss) ---
    if not is_cache_hit:
        # Fetch URLs from search.
        search_results = serper_search(query, SERPER_API_KEY, top_k)
        urls = [item["url"] for item in search_results if item["url"]]
        
        # Fetch page content in parallel.
        if urls and jina_reader_reachable():
            print(f"📄 Fetching {len(urls)} pages...")
            page_texts = fetch_multiple_pages(urls, jina_api_key=JINA_API_KEY, max_workers=5)
        else:
            print(f"📄 Skipping Jina fetch for {len(urls)} pages; using Serper snippets.")
            page_texts = {}
        
        # Build normalized result objects.
        for item in search_results:
            url = item["url"]
            result_item = build_search_result_with_fallback(item, page_texts.get(url, ""))
            if result_item:
                valid_results.append(result_item)

    # Extract URLs from valid results to return
    source_urls = [res["url"] for res in valid_results if "url" in res]

    if not valid_results:
        return "No valid content found.", [], []

    # --- Step 3: Parallel processing (conditional summary + visual audit) ---
    
    # Generate a summary only when none exists.
    need_summary = not bool(text_summary)
    
    print(f"⚡ Parallel Process: Summary [{'RUN' if need_summary else 'SKIP/CACHE'}] + Visual Audit [{len(valid_results)} pages]...")
    
    final_visual_evidence = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        # Task A: text summary, only when needed.
        future_summary = None
        if need_summary:
            future_summary = executor.submit(summarize, query, valid_results)
        
        # Task B: visual audit always runs because section requirements can change.
        future_visuals = []
        for res in valid_results:
            # Only pages with markdown image markers need visual auditing.
            if "![" in res.get("content", ""):
                f = executor.submit(
                    process_page_visuals, 
                    res["content"], 
                    section_data
                )
                future_visuals.append(f)
            
        # --- Step 4: Collect results ---
        
        # Collect text summary.
        if future_summary:
            try:
                text_summary = future_summary.result()
                print("✅ Text summary generated.")
                
                # Update cache for both new searches and missing summaries.
                cache_payload = {
                    "results": valid_results,
                    "summary_cards": text_summary
                }
                update_cache_entry(query, cache_payload)
                print(f"💾 Cache updated for: {query}")
                
            except Exception as e:
                text_summary = f"[Error in Summary] {e}"
                print(f"❌ Text summary failed: {e}")
                # Preserve fetched content even if summarization fails.
                if not is_cache_hit:
                    update_cache_entry(query, {"results": valid_results, "summary_cards": ""})
        elif is_cache_hit:
             print("✅ Used cached summary.")

        # Collect visual evidence.
        for f in as_completed(future_visuals):
            try:
                images = f.result()
                if images:
                    final_visual_evidence.extend(images)
            except Exception as e:
                print(f"❌ Visual audit thread failed: {e}")

    print(f"🏁 Multimodal Search Finished. Collected {len(final_visual_evidence)} valid images.")

    return text_summary, final_visual_evidence, source_urls


def clean_cache():
    """Clean cache by removing entries whose content begins with '[Error]'."""
    cache = load_cache()
    changed = False

    for query, payload in list(cache.items()):
        # Support both legacy list cache and new dict cache
        if isinstance(payload, list):
            results = payload
            cleaned_results = [
                item for item in results
                if not (isinstance(item, dict) and str(item.get("content", "")).startswith("[Error]"))
            ]
            if len(cleaned_results) != len(results):
                cache[query] = cleaned_results
                changed = True
        elif isinstance(payload, dict):
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue
            cleaned_results = [
                item for item in results
                if not (isinstance(item, dict) and str(item.get("content", "")).startswith("[Error]"))
            ]
            if len(cleaned_results) != len(results):
                payload["results"] = cleaned_results
                cache[query] = payload
                changed = True

    if changed:
        save_cache(cache)
        print("Cache cleaned and saved.")
    else:
        print("Cache is already clean. Nothing to remove.")


def get_page_from_cache(url: str) -> str:
    """Retrieve page content from cache if available."""
    cache = load_cache()
    for payload in cache.values():
        results = payload.get("results", [])
        for item in results:
            if item.get("url") == url:
                return item.get("content", "")
    return ""


if __name__ == "__main__":
    # query = "Renmin University of China"
    # top_k = 10

    # clean_cache()

    # results, summary = search_and_summarize(query, top_k=top_k, do_summarize=True)

    # print("\n" + "=" * 80)

    # print(summary)

    # url = "https://www.lawson.jp/en/ir/library/annual_report/2022/pdf/ar2022_P25-32.pdf"
    # print(get_page_from_cache(url))

    # sec_data = {
    #   "id": "1",
    #   "title": "Introduction to Java Web Development Foundations",
    #   "summary": "Explains the origins of Java Servlets and JSP as foundational technologies for server-side web development.",
    #   "key_points": [
    #     "Servlets provided raw HTTP request/response handling via Java classes",
    #     "JSP enabled dynamic content generation through scriptlets",
    #     "Both required manual resource management and thread safety implementation"
    #   ],
    #   "needs_further_research": [
    #     "Exact release dates of major Servlet API versions",
    #     "Performance benchmarks comparing early Servlet implementations"
    #   ],
    #   "suggested_search_queries": [
    #     "Java Servlet API version history",
    #     "Thread safety challenges in JSP development"
    #   ],
    #   "visual_requirements": [
    #     "Timeline diagram showing Servlet/JSP technology stack components"
    #   ]
    # }

    # search_with_vision(
    #     query="java servlet api version history",
    #     section_data=sec_data,
    #     top_k=5
    # )

    serper_search("DeepSeek R1 model", SERPER_API_KEY, top_k=5)
