from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from io import BytesIO
from typing import Any, Optional

from utils.html_refiner import refine_html_report
from utils.report_preview import ReportPreviewBuilder


def pil_to_base64(img, fmt: str = "PNG", as_data_uri: bool = True) -> str:
    if img is None:
        return ""

    save_img = img
    if fmt.upper() in ("JPEG", "JPG") and getattr(img, "mode", "") not in ("RGB",):
        save_img = img.convert("RGB")

    buf = BytesIO()
    save_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    if as_data_uri:
        mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
        return f"data:{mime};base64,{b64}"
    return b64


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_research_images(cache_dir: str, section_id: int) -> Optional[list[dict]]:
    path = os.path.join(cache_dir, f"research_images_section_{section_id}.json")
    if not os.path.exists(path):
        return None
    return read_json(path)


def open_research_images(research_images: Optional[list[dict]]) -> Optional[list]:
    if not research_images:
        return None
    from PIL import Image

    images = []
    for item in research_images:
        local_path = item.get("local_path")
        if local_path:
            images.append(Image.open(local_path))
    return images or None


def references_from_research(research_result: dict) -> list[dict]:
    return research_result.get("references", []) or research_result.get("citations", []) or []


def replace_section_citations(writer, research_result: dict, text: str) -> str:
    local_to_global = writer.ingest_references(references_from_research(research_result))
    return writer.replace_local_citations(text, local_to_global)


def citation_to_md(citations: list[dict]) -> str:
    lines = []
    for item in citations:
        sid = item.get("id", "").strip()
        title = item.get("title", "").strip()
        url = item.get("url", "").strip()
        if sid and title and url:
            lines.append(f"- [[{sid}] {title}]({url})")
    return "\n".join(lines)


def append_references(
    raw_text: str,
    processed_text: str,
    final_citations: list[dict],
    language: str = "en",
) -> tuple[str, str]:
    citations_title = "参考文献" if language == "zh" else "References"
    citations_md = citation_to_md(final_citations)
    refs_block = f"\n\n## {citations_title}\n\n{citations_md}"
    return raw_text + refs_block, processed_text + refs_block


def clean_failed_image_markers(text: str) -> str:
    return (text or "").replace("<fail_to_generate_image>", "")


def build_report_output(
    *,
    report_id: Optional[int],
    question: str,
    raw_text: str,
    text: str,
    images: list,
    clean_text: Optional[str] = None,
) -> dict:
    output = {
        "question": question,
        "raw_text": raw_text,
        "text": text,
        "images": [pil_to_base64(img) for img in images],
    }
    if report_id is not None:
        output["id"] = report_id
    if clean_text is not None:
        output["clean_text"] = clean_text
    return output


def build_preview_with_html_refine(
    *,
    report_title: str,
    report_text: str,
    report_imgs: list,
    final_citations: list[dict],
    cache_dir: str,
    html_refine_client,
    enable_html_refine: bool = True,
) -> str:
    preview = ReportPreviewBuilder(output_dir=cache_dir)
    draft_name = "report_before_html_refine.html" if enable_html_refine else "report.html"
    draft_path = preview.build(
        report_title=report_title,
        report_text=report_text,
        report_imgs=report_imgs,
        final_citations=final_citations,
        out_filename=draft_name,
    )

    if not enable_html_refine:
        return draft_path

    final_path = os.path.join(cache_dir, "report.html")
    return refine_html_report(
        client=html_refine_client,
        input_html_path=draft_path,
        output_html_path=final_path,
        cache_dir=cache_dir,
        report_title=report_title,
    )


def _research_one_section(
    *,
    client,
    reviewer_client,
    cache_dir: str,
    question: str,
    overview: str,
    section: dict,
) -> tuple[int, dict]:
    from agents.researcher import ResearcherAgent

    section_id = section["id"]
    researcher = ResearcherAgent(client, reviewer_client, cache_dir=cache_dir)
    research_prompt = researcher.create_section_prompt(
        question=question,
        overview=overview,
        section_outline=section,
    )
    result = researcher.run(research_prompt, section_id, section)
    if not isinstance(result, dict):
        result_path = os.path.join(cache_dir, f"research_output_section_{section_id}.json")
        result = read_json(result_path)
    return section_id, result


def research_sections(
    *,
    sections: list[dict],
    client,
    reviewer_client,
    cache_dir: str,
    question: str,
    overview: str,
    section_workers: int = 1,
) -> dict[int, dict]:
    if not sections:
        return {}

    max_workers = max(1, int(section_workers or 1))
    if max_workers == 1 or len(sections) == 1:
        results = {}
        for section in sections:
            section_id, result = _research_one_section(
                client=client,
                reviewer_client=reviewer_client,
                cache_dir=cache_dir,
                question=question,
                overview=overview,
                section=section,
            )
            results[section_id] = result
        return results

    print(f"[Research] Running {len(sections)} sections with {max_workers} workers...")
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_section = {
            executor.submit(
                _research_one_section,
                client=client,
                reviewer_client=reviewer_client,
                cache_dir=cache_dir,
                question=question,
                overview=overview,
                section=section,
            ): section
            for section in sections
        }
        for future in as_completed(future_to_section):
            section = future_to_section[future]
            section_id = section["id"]
            try:
                finished_id, result = future.result()
                results[finished_id] = result
                print(f"[Research] Section {finished_id} finished.")
            except Exception as exc:
                raise RuntimeError(f"Section {section_id} research failed: {exc}") from exc

    return results
