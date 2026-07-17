from __future__ import annotations

import asyncio
import base64
import os
import re
import shutil
from pathlib import Path
from typing import Optional


def _file_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    match = re.fullmatch(r"```(?:html)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _looks_like_html(text: str) -> bool:
    lower = (text or "").lower()
    return "<html" in lower and "</html>" in lower and "<body" in lower and "</body>" in lower


def _original_image_srcs(html: str) -> list[str]:
    return re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", html or "", flags=re.IGNORECASE)


def _find_chromium_executable() -> Optional[str]:
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    candidates = [
        env_path,
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
        shutil.which("chrome"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


async def _screenshot_page(
    url: str,
    output_path: str,
    width: int,
    height: int,
    wait_until: str = "networkidle",
    extra_wait_ms: int = 500,
) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        executable_path = _find_chromium_executable()
        launch_kwargs = {"headless": True, "args": ["--no-sandbox"]}
        if executable_path:
            launch_kwargs["executable_path"] = executable_path
        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = await context.new_page()
        await page.goto(url, wait_until=wait_until, timeout=60_000)
        if extra_wait_ms > 0:
            await page.wait_for_timeout(extra_wait_ms)

        scroll_height = await page.evaluate("() => document.documentElement.scrollHeight")
        clip_height = min(height, max(1, scroll_height))
        await page.screenshot(
            path=output_path,
            clip={"x": 0, "y": 0, "width": width, "height": clip_height},
        )
        await browser.close()


def render_html_screenshot(
    html_path: str,
    output_path: str,
    width: int = 1000,
    height: int = 2000,
) -> str:
    url = Path(html_path).resolve().as_uri()
    asyncio.run(_screenshot_page(url, output_path, width=width, height=height))
    return output_path


def refine_html_with_screenshot(
    *,
    client,
    html: str,
    screenshot_path: str,
    report_title: str = "Final Report",
    max_tokens: int = 12000,
) -> str:
    image_url = _file_to_data_url(screenshot_path)
    prompt = f"""
You are the HTML Refine module in a multimodal deep-research report pipeline.

You receive:
1. The current complete HTML document.
2. A rendered screenshot of that HTML document.

Your task is to improve the rendered presentation quality while preserving the report content.

Optimize only:
- HTML structure when needed for better scanability.
- CSS layout, spacing, typography, visual hierarchy, figure placement, references styling, and readability.
- Density-legibility balance, informational saliency, visual encoding diversity, and visual ergonomics.

Strict preservation rules:
- Preserve all factual text, section headings, citations, reference items, URLs, and image meanings.
- Preserve every existing image `src` path exactly. Do not rename, remove, or invent image assets.
- Do not add external CSS, JavaScript, fonts, or network dependencies.
- Do not add new claims, citations, references, or explanatory content.
- Do not use Markdown fences.
- Return only one complete valid HTML document.
- The document must be self-contained except for existing relative image paths.

Report title: {report_title}

[CURRENT_HTML_BEGIN]
{html}
[CURRENT_HTML_END]
""".strip()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]
    response, _ = client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.8,
        enable_thinking=False,
    )
    refined = _strip_code_fence(response)

    if not _looks_like_html(refined):
        raise ValueError("HTML refiner did not return a complete HTML document.")

    missing_srcs = [src for src in _original_image_srcs(html) if src not in refined]
    if missing_srcs:
        raise ValueError(f"HTML refiner dropped image src paths: {missing_srcs[:5]}")

    return refined


def refine_html_report(
    *,
    client,
    input_html_path: str,
    output_html_path: str,
    cache_dir: str,
    report_title: str = "Final Report",
    screenshot_width: int = 1000,
    screenshot_height: int = 2000,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    screenshot_path = os.path.join(cache_dir, "html_refine_input.png")
    response_path = os.path.join(cache_dir, "html_refine_output.html")
    error_path = os.path.join(cache_dir, "html_refine_error.txt")

    try:
        render_html_screenshot(
            input_html_path,
            screenshot_path,
            width=screenshot_width,
            height=screenshot_height,
        )
        with open(input_html_path, "r", encoding="utf-8") as f:
            original_html = f.read()

        refined_html = refine_html_with_screenshot(
            client=client,
            html=original_html,
            screenshot_path=screenshot_path,
            report_title=report_title,
        )

        with open(response_path, "w", encoding="utf-8") as f:
            f.write(refined_html)
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(refined_html)
        return os.path.abspath(output_html_path)

    except Exception as exc:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(str(exc))
        if os.path.abspath(input_html_path) != os.path.abspath(output_html_path):
            shutil.copyfile(input_html_path, output_html_path)
        return os.path.abspath(output_html_path)
