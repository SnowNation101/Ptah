import os
import re
import json
import base64
import requests
import mimetypes
import hashlib
from io import BytesIO
from PIL import Image
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Runtime configuration
VLLM_VISION_URL = os.getenv("PTAH_VLM_CHAT_URL", "http://localhost:8001/v1/chat/completions")
MODEL_PATH = os.getenv("PTAH_VLM_MODEL_NAME", "models/Qwen3-VL-32B-Instruct")
IMAGE_CACHE_DIR = ".cache/images"

# Static image filters
MIN_WIDTH = 200
MIN_HEIGHT = 200
MAX_ASPECT_RATIO = 3.0   # Images wider than 3:1 are often separators or banners.
MIN_FILE_SIZE = 5 * 1024 # Very small files are usually decorative assets.
ALLOWED_MIMETYPES = ['image/jpeg', 'image/png', 'image/webp']

def is_valid_static_rule(content, content_type):
    """Apply size, aspect-ratio, and MIME-type filters before VLM auditing."""
    # Check file size.
    if len(content) < MIN_FILE_SIZE:
        return False, f"Too small ({len(content)//1024}KB)"

    # Check MIME type.
    if content_type not in ALLOWED_MIMETYPES:
        return False, f"Unsupported MIME type ({content_type})"

    # Check dimensions and aspect ratio.
    try:
        with Image.open(BytesIO(content)) as img:
            w, h = img.size
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                return False, f"Too small dimensions ({w}x{h})"
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > MAX_ASPECT_RATIO:
                return False, f"Extreme aspect ratio ({aspect_ratio:.1f}:1)"
            
            return True, f"Passed ({w}x{h})"
    except Exception as e:
        return False, f"Image parse failed: {e}"

def download_and_cache(img_url):
    """Download an image, apply static filters, and cache accepted files."""
    download_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    try:
        resp = requests.get(img_url, headers=download_headers, timeout=10)
        if not resp.ok: 
            return None

        content_type = resp.headers.get('Content-Type', '').split(';')[0].lower()
        
        # Apply static filters.
        is_valid, reason = is_valid_static_rule(resp.content, content_type)
        if not is_valid:
            return None

        img_id = hashlib.md5(img_url.encode()).hexdigest()
        
        # Infer file extension.
        ext = mimetypes.guess_extension(content_type)
        if not ext:
            parsed_path = urlparse(img_url).path
            ext = os.path.splitext(parsed_path)[1]
            if not ext:
                ext = ".jpg"
        if ext == '.jpe': ext = '.jpg'

        filename = f"{img_id}{ext}"
        local_path = os.path.join(IMAGE_CACHE_DIR, filename)

        os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(resp.content)
            
        print(f"   📥 Downloaded: {filename} ({content_type}) <- {img_url[:40]}...")
        return {"url": img_url, "local_path": local_path, "type": content_type}

    except Exception as e:
        return None


def audit_all_images(candidates, section_title, visual_requirements):
    if not candidates:
        return []

    count = len(candidates)
    print(f"🤖 Calling VLM to audit {count} images at once...")

    # Build the multimodal prompt.
    user_content = [
        {"type": "text", "text": f"Audit these {count} images for a research report.\n"}
    ]
    user_content.append({
        "type": "text", 
        "text": f"Section Title: {section_title}\nPlanner Requirements: {json.dumps(visual_requirements)}\n\n"
    })

    # Insert images and their alt text in order.
    for idx, img in enumerate(candidates):
        try:
            with open(img['local_path'], "rb") as f:
                base64_img = base64.b64encode(f.read()).decode('utf-8')
            
            alt_text_display = img.get('alt_text', 'No description provided.')
            
            user_content.append({
                "type": "text", 
                "text": f"--- Image ID {idx} ---\nContext/Description: {alt_text_display}\n"
            })
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img['type']};base64,{base64_img}"}
            })
        except Exception as e:
            print(f"⚠️ Failed to load image {idx} for auditing: {e}")
            continue

    user_content.append({"type": "text", "text": """
Task:
For each Image ID provided above, analyze its visual content AND the provided context description to decide whether to KEEP or DROP.

**Criteria for KEEP:**
1. **EVIDENCE**: The image contains objective facts, such as **statistical charts, data tables, experimental graphs, UI screenshots, or photos of specific physical entities**.
2. **ILLUSTRATION**: The image helps explain complex concepts (e.g., architecture diagrams, flowcharts).
3. **CONTEXT MATCH**: The image content aligns with its provided description (Alt text) and the section title.

**Criteria for DROP:**
- Generic stock photos or decorative gradients.
- Images that contradict their description or are low resolution.
- Navigational elements (icons, logos).

Output strictly only a JSON object with a "results" list:
```json
{
  "results": [
    {
      "id": 0,
      "decision": "KEEP",
      "reason": "Shows specific architecture mentioned in text"
    },
    ...
  ]
}
```
"""})

    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": user_content}],
        "enable_thinking": True,
        "max_tokens": 4096 
    }

    try:
        response = requests.post(VLLM_VISION_URL, json=payload, timeout=180) 
        if response.ok:
            content = response.json()['choices'][0]['message']['content']
            # Strip thinking output when present.
            if '</think>' in content:
                content = content.split('</think>')[-1]
            
            print(content[:200] + "...")

            # Parse JSON response.
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                results_list = data.get("results", [])
                
                final_kept_images = []
                for res in results_list:
                    img_idx = res.get("id")
                    decision = res.get("decision", "").upper()
                    
                    # Keep valid image IDs marked as KEEP.
                    if decision == "KEEP" and isinstance(img_idx, int) and img_idx < len(candidates):
                        # Preserve original candidate metadata.
                        merged_info = candidates[img_idx].copy()
                        
                        if "reason" in res:
                            merged_info["vlm_reason"] = res["reason"]


                        if "id" in merged_info: 
                            del merged_info["id"] 
                            
                        final_kept_images.append(merged_info)
                        
                return final_kept_images
    except Exception as e:
        print(f"❌ VLM audit failed: {e}")

    return []


def process_page_visuals(markdown_content, section_data):
    title = section_data.get("title", "")
    requirements = section_data.get("visual_requirements", [])

    # Extract image URLs and alt text from markdown.
    raw_matches = re.findall(r'!\[(.*?)\]\((https?://.*?)\)', markdown_content)

    if not raw_matches: 
        return []

    # Build URL -> cleaned alt text mapping.
    url_to_alt = {}
    unique_urls = []

    for raw_alt, url in raw_matches:
        # Remove generated prefixes such as "Image 5:".
        clean_alt = re.sub(r'^Image\s*\d+:\s*', '', raw_alt, flags=re.IGNORECASE).strip()
        
        if not clean_alt:
            clean_alt = "Image found in research material."
            
        url_to_alt[url] = clean_alt
        if url not in unique_urls:
            unique_urls.append(url)

    # Download candidates in parallel.
    downloaded_files = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_cache, url) for url in unique_urls]
        for f in as_completed(futures):
            res = f.result()
            if res: 
                downloaded_files.append(res)

    if not downloaded_files:
        print("No qualified image in this page.")
        return []

    # Attach alt text to downloaded image metadata.
    candidates = []
    for img_obj in downloaded_files:
        url = img_obj['url']
        if url in url_to_alt:
            img_obj['alt_text'] = url_to_alt[url]
            candidates.append(img_obj)

    print(f"{len(candidates)} images passed static filter. Sending ALL to VLM...")

    # Run VLM audit.
    final_selected = audit_all_images(candidates, title, requirements)

    print(f"✅ Selected {len(final_selected)}/{len(candidates)} images.")
    return final_selected
