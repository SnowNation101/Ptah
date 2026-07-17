from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from utils.common import (
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_SILICONFLOW_API_URL,
    DEFAULT_UNIAPI_BASE_URL,
    DEFAULT_UNIAPI_MODEL,
    call_judge_chat,
    extract_assistant_text,
    extract_json_object,
    load_jsonl,
    project_path,
    read_json,
    resolve_api_key,
    write_json,
)


CITATION_RE = re.compile(r"\[([A-Z]\d+)\]")
REFERENCE_RE = re.compile(r"^\s*-\s*\[\[([A-Z]\d+)\]\s*([^\]]+)\]\(([^)]+)\)", re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")

EXTRACT_SYSTEM_PROMPT = """
You are an information extraction assistant for factuality evaluation.

Extract all atomic factual claims from the candidate report that are explicitly associated with citations.
Each extracted item must preserve the nearby citation ids exactly as they appear, such as C1 or C12.
Only extract factual claims that can be checked against cited web sources.
Do not extract section titles, purely subjective opinions, or uncited claims.

Return strictly valid JSON:
{
  "claims": [
    {
      "claim": "one atomic factual claim",
      "citation_ids": ["C1"],
      "source_sentence": "the original sentence containing the citation"
    }
  ]
}
""".strip()

VALIDATE_SYSTEM_PROMPT = """
You are a strict fact-checking evaluator for DeepResearch Bench FACT.

Given a candidate claim, the citation ids attached to it, and the scraped contents of those cited sources,
decide whether the cited evidence supports the claim.

Labels:
- "supported": the reference fully or partially supports the statement.
- "unsupported": the reference is valid but does not support the statement.
- "unknown": the reference has no valid content or is unavailable.

Be conservative. Do not use outside knowledge. Use only the provided cited source contents.

Return strictly valid JSON:
{
  "result": "supported",
  "reason": "brief explanation grounded in the provided source text"
}
""".strip()


def report_text(report: dict[str, Any]) -> str:
    return report.get("clean_text") or report.get("text") or report.get("raw_text") or ""


def load_queries(path: str | Path) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in load_jsonl(path) if "id" in item}


def extract_references(text: str) -> dict[str, dict[str, str]]:
    refs: dict[str, dict[str, str]] = {}
    for match in REFERENCE_RE.finditer(text or ""):
        cid = match.group(1).strip()
        refs[cid] = {
            "id": cid,
            "title": match.group(2).strip(),
            "url": match.group(3).strip(),
        }
    return refs


def fallback_extract_claims(report_id: int, text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for sentence in SENTENCE_SPLIT_RE.split(text or ""):
        sentence = sentence.strip()
        if not sentence:
            continue
        citation_ids = CITATION_RE.findall(sentence)
        if not citation_ids:
            continue
        clean_claim = CITATION_RE.sub("", sentence).strip()
        if len(clean_claim) < 20:
            continue
        claims.append(
            {
                "report_id": report_id,
                "claim": clean_claim,
                "citation_ids": list(dict.fromkeys(citation_ids)),
                "source_sentence": sentence,
                "extract_method": "regex_fallback",
            }
        )
    return claims


def call_judge_json(
    *,
    provider: str,
    api_key: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    response = call_judge_chat(
        provider=provider,
        api_key=api_key,
        api_url=api_url,
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}} if provider in {"local", "vllm"} else None,
    )
    raw = extract_assistant_text(response)
    parsed = extract_json_object(raw)
    parsed["_raw_output"] = raw
    return parsed


def extract_claims_with_judge(
    *,
    report_id: int,
    question: str,
    report_text_value: str,
    provider: str,
    api_key: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    max_tokens: int,
    timeout: int,
) -> list[dict[str, Any]]:
    payload = {
        "question": question,
        "report": report_text_value,
    }
    try:
        parsed = call_judge_json(
            provider=provider,
            api_key=api_key,
            api_url=api_url,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        raw_claims = parsed.get("claims", [])
        if not isinstance(raw_claims, list):
            raise ValueError("claims is not a list")
        claims: list[dict[str, Any]] = []
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim", "")).strip()
            citation_ids = item.get("citation_ids", [])
            if not claim or not isinstance(citation_ids, list):
                continue
            normalized_ids = [str(cid).strip() for cid in citation_ids if str(cid).strip()]
            if not normalized_ids:
                continue
            claims.append(
                {
                    "report_id": report_id,
                    "claim": claim,
                    "citation_ids": list(dict.fromkeys(normalized_ids)),
                    "source_sentence": str(item.get("source_sentence", "")).strip(),
                    "extract_method": "judge",
                }
            )
        return claims or fallback_extract_claims(report_id, report_text_value)
    except Exception as exc:
        claims = fallback_extract_claims(report_id, report_text_value)
        for item in claims:
            item["extract_error"] = str(exc)
        return claims


def normalize_claim(text: str) -> str:
    text = CITATION_RE.sub("", text.lower())
    text = re.sub(r"\W+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def deduplicate_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for item in claims:
        key = (int(item["report_id"]), normalize_claim(item["claim"]))
        if key not in seen:
            seen[key] = item
            continue
        existing_ids = seen[key].setdefault("citation_ids", [])
        for cid in item.get("citation_ids", []):
            if cid not in existing_ids:
                existing_ids.append(cid)
    return list(seen.values())


def jina_reader_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme:
        url = f"https://{url}"
    return f"https://r.jina.ai/{url}"


def scrape_url(url: str, jina_api_key: str | None, timeout: int = 45) -> dict[str, Any]:
    headers = {"X-Return-Format": "markdown"}
    if jina_api_key:
        headers["Authorization"] = f"Bearer {jina_api_key}"
    try:
        response = requests.get(jina_reader_url(url), headers=headers, timeout=timeout)
        return {
            "url": url,
            "status_code": response.status_code,
            "ok": response.ok,
            "content": response.text[:20000] if response.ok else "",
            "error": "" if response.ok else response.text[:500],
        }
    except Exception as exc:
        return {"url": url, "status_code": None, "ok": False, "content": "", "error": str(exc)}


def scrape_sources(
    reports: dict[int, dict[str, Any]],
    max_workers: int,
    jina_api_key: str | None,
) -> dict[tuple[int, str], dict[str, Any]]:
    jobs: list[tuple[int, str, dict[str, str]]] = []
    for report_id, report in reports.items():
        refs = extract_references(report.get("text", "") or report.get("raw_text", "") or report_text(report))
        for cid, ref in refs.items():
            if ref.get("url"):
                jobs.append((report_id, cid, ref))

    scraped: dict[tuple[int, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        future_to_ref = {
            executor.submit(scrape_url, ref["url"], jina_api_key): (report_id, cid, ref)
            for report_id, cid, ref in jobs
        }
        for future in as_completed(future_to_ref):
            report_id, cid, ref = future_to_ref[future]
            payload = future.result()
            payload.update({"report_id": report_id, "citation_id": cid, "title": ref.get("title", "")})
            scraped[(report_id, cid)] = payload
    return scraped


def validate_claim(
    *,
    claim: dict[str, Any],
    scraped: dict[tuple[int, str], dict[str, Any]],
    provider: str,
    api_key: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    report_id = int(claim["report_id"])
    sources = []
    for cid in claim.get("citation_ids", []):
        src = scraped.get((report_id, cid))
        if src:
            sources.append(
                {
                    "citation_id": cid,
                    "title": src.get("title", ""),
                    "url": src.get("url", ""),
                    "content": src.get("content", ""),
                    "scrape_ok": src.get("ok", False),
                    "scrape_error": src.get("error", ""),
                }
            )
    payload = {
        "claim": claim.get("claim", ""),
        "citation_ids": claim.get("citation_ids", []),
        "sources": sources,
    }
    try:
        parsed = call_judge_json(
            provider=provider,
            api_key=api_key,
            api_url=api_url,
            base_url=base_url,
            model=model,
            messages=[
                {"role": "system", "content": VALIDATE_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            timeout=timeout,
        )
        label = str(parsed.get("result", parsed.get("label", ""))).strip().lower()
        if label not in {"supported", "unsupported", "unknown"}:
            label = "unknown"
        return {
            **claim,
            "sources": sources,
            "result": label,
            "reason": str(parsed.get("reason", "")).strip(),
            "raw_output": parsed.get("_raw_output", ""),
        }
    except Exception as exc:
        return {
            **claim,
            "sources": sources,
            "result": "unknown",
            "reason": f"Validation failed: {exc}",
            "raw_output": "",
        }


def build_official_records(validated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for item in validated:
        report_id = int(item["report_id"])
        record = grouped.setdefault(report_id, {"id": report_id, "citations": {}, "citations_deduped": {}})
        for source in item.get("sources", []):
            cid = source.get("citation_id", "")
            if not cid:
                continue
            citation = record["citations_deduped"].setdefault(
                cid,
                {
                    "url": source.get("url", ""),
                    "title": source.get("title", ""),
                    "validate_error": None,
                    "validate_res": [],
                },
            )
            citation["validate_res"].append(
                {
                    "idx": len(citation["validate_res"]) + 1,
                    "statement": item.get("claim", ""),
                    "result": item.get("result", "unknown"),
                    "reason": item.get("reason", ""),
                }
            )
            record["citations"][cid] = citation["url"]
    return [grouped[report_id] for report_id in sorted(grouped)]


def compute_stats(validated: list[dict[str, Any]]) -> dict[str, Any]:
    supported = sum(1 for item in validated if item.get("result") == "supported")
    unsupported = sum(1 for item in validated if item.get("result") == "unsupported")
    unknown = sum(1 for item in validated if item.get("result") == "unknown")
    total_citations = supported + unsupported
    report_ids = sorted({int(item["report_id"]) for item in validated})
    total_num = len(report_ids)
    return {
        "num_reports": total_num,
        "num_supported": supported,
        "num_unsupported": unsupported,
        "num_unknown": unknown,
        "total_citations": round(total_citations / total_num, 4) if total_num else None,
        "total_valid_citations": round(supported / total_num, 4) if total_num else None,
        "valid_rate": round(supported / total_citations, 4) if total_citations else None,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official-style DRB FACT evaluation for generated reports.")
    parser.add_argument("--query-file", default=str(project_path("data", "drb", "query.jsonl")))
    parser.add_argument("--report-dir", default=str(project_path("outputs", "drb")))
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-jsonl", default=str(project_path("eval_results", "dr_bench", "fact", "Ptah", "raw_results.jsonl")))
    parser.add_argument("--out-summary", default=str(project_path("eval_results", "dr_bench", "fact", "Ptah", "fact_summary.json")))
    parser.add_argument("--provider", default=DEFAULT_JUDGE_PROVIDER, choices=["uniapi", "siliconflow", "openai-compatible", "openai", "local", "vllm"])
    parser.add_argument("--model", default=DEFAULT_UNIAPI_MODEL)
    parser.add_argument("--api-url", default="", help=f"Full chat completions URL. SiliconFlow default: {DEFAULT_SILICONFLOW_API_URL}")
    parser.add_argument("--base-url", default=DEFAULT_UNIAPI_BASE_URL)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--jina-api-key-env", default="JINA_API_KEY")
    return parser


def main() -> int:
    load_dotenv()
    args = build_argparser().parse_args()

    api_key, used_api_key_env = resolve_api_key(args.api_key_env, args.provider)
    if not api_key and args.provider in {"local", "vllm"}:
        api_key = "EMPTY"
    if not api_key:
        print(f"Missing API key env var: {used_api_key_env}", file=sys.stderr)
        return 2

    queries = load_queries(args.query_file)
    ids = [i for i in sorted(queries) if i >= args.start_id]
    if args.end_id:
        ids = [i for i in ids if i <= args.end_id]
    if args.limit > 0:
        ids = ids[: args.limit]

    report_dir = Path(args.report_dir)
    reports: dict[int, dict[str, Any]] = {}
    for report_id in ids:
        report_path = report_dir / f"report_{report_id}.json"
        if report_path.exists():
            reports[report_id] = read_json(report_path)

    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_dir = out_jsonl.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Extract cited claims")
    extracted: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = []
        for report_id, report in reports.items():
            futures.append(
                executor.submit(
                    extract_claims_with_judge,
                    report_id=report_id,
                    question=queries.get(report_id, {}).get("prompt", ""),
                    report_text_value=report_text(report),
                    provider=args.provider,
                    api_key=api_key,
                    api_url=args.api_url or None,
                    base_url=args.base_url or None,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                )
            )
        for future in as_completed(futures):
            extracted.extend(future.result())
    write_jsonl(out_dir / "extracted_claims.jsonl", extracted)

    print("[2/5] Deduplicate claims")
    deduplicated = deduplicate_claims(extracted)
    write_jsonl(out_dir / "deduplicated_claims.jsonl", deduplicated)

    print("[3/5] Scrape cited sources")
    jina_api_key = os.environ.get(args.jina_api_key_env, "")
    scraped = scrape_sources(reports, max_workers=args.max_workers, jina_api_key=jina_api_key)
    write_jsonl(out_dir / "scraped_sources.jsonl", list(scraped.values()))

    print("[4/5] Validate claims")
    validated: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [
            executor.submit(
                validate_claim,
                claim=claim,
                scraped=scraped,
                provider=args.provider,
                api_key=api_key,
                api_url=args.api_url or None,
                base_url=args.base_url or None,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            for claim in deduplicated
        ]
        for future in as_completed(futures):
            validated.append(future.result())
    validated.sort(key=lambda item: (int(item["report_id"]), item.get("claim", "")))
    write_jsonl(out_dir / "validated_claims.jsonl", validated)
    official_records = build_official_records(validated)
    write_jsonl(out_jsonl, official_records)

    print("[5/5] Compute statistics")
    summary = compute_stats(validated)
    write_json(out_summary, summary)
    official_stat_path = out_summary.with_suffix(".txt")
    official_stat_path.write_text(
        "\n".join(
            [
                f"total_citations: {summary['total_citations']}",
                f"total_valid_citations: {summary['total_valid_citations']}",
                f"valid_rate: {summary['valid_rate']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
