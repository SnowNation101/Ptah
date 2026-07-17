import os
import shutil

from core.model_client import ModelClient
from agents.planner import PlannerAgent
from agents.writer import WriterAgent
from utils.image_tools import ImageTools
from tasks.bench_utils import (
    build_preview_with_html_refine,
    build_report_output,
    load_research_images,
    open_research_images,
    read_json,
    replace_section_citations,
    research_sections,
    write_json,
    write_text,
)


CACHE_DIR = ".cache/custom/"


class CustomTask:
    def __init__(self, args):
        self.args = args
        self.client = ModelClient(args.model_name, args.base_url, args.api_key)
        self.reviewer_client = ModelClient(
            args.reviewer_model_name,
            args.reviewer_base_url,
            args.reviewer_api_key,
        )
        self.writer_client = ModelClient(
            os.getenv("PTAH_VLM_MODEL_NAME", "models/Qwen3-VL-32B-Instruct"),
            os.getenv("PTAH_VLM_BASE_URL", "http://localhost:8001/v1/"),
            os.getenv("PTAH_VLM_API_KEY", "EMPTY"),
        )
        self.planner = PlannerAgent(
            self.client,
            self.reviewer_client,
            cache_dir=CACHE_DIR,
        )
        self.tools = ImageTools()

    def run(self):
        question = self.args.question
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs("outputs/custom", exist_ok=True)

        self.planner.run(question)
        outline = read_json(os.path.join(CACHE_DIR, "outline_output.json"))
        overview = outline.get("overview", "")
        sections = outline.get("sections", [])

        research_results = research_sections(
            sections=sections,
            client=self.client,
            reviewer_client=self.reviewer_client,
            cache_dir=CACHE_DIR,
            question=question,
            overview=overview,
            section_workers=self.args.section_workers,
        )

        writer = WriterAgent(self.writer_client, self.tools, cache_dir=CACHE_DIR)

        all_text_raw = ""
        all_text_processed = ""
        all_images = []

        for section in sections:
            section_id = section["id"]
            research_result = research_results[section_id]
            research_images = load_research_images(CACHE_DIR, section_id)

            write_prompt = writer.create_section_prompt(
                question=question,
                overview=overview,
                research_output=research_result,
                web_images=research_images,
            )
            raw_text = writer.write_section(write_prompt, section_id)
            raw_text = replace_section_citations(writer, research_result, raw_text)
            all_text_raw += raw_text + "\n\n"

            section_text_processed, section_images = writer.gen_images(
                raw_text,
                open_research_images(research_images),
            )
            all_text_processed += section_text_processed + "\n\n"
            all_images.extend(section_images)

        conclusion_raw = writer.write_conclusion(
            question=question,
            overview=overview,
            prior_sections_text=all_text_raw,
            conclusion_title="Conclusion",
        )
        all_text_raw += conclusion_raw
        all_text_processed += conclusion_raw

        write_text(os.path.join(CACHE_DIR, "raw_report.txt"), all_text_raw)
        final_citations = writer.get_final_citations()

        output = build_report_output(
            report_id=None,
            question=question,
            raw_text=all_text_raw,
            text=all_text_processed,
            images=all_images,
        )
        write_json("outputs/custom/report.json", output)

        html_path = build_preview_with_html_refine(
            report_title=outline.get("report_title", "Final Report"),
            report_text=all_text_processed,
            report_imgs=all_images,
            final_citations=final_citations,
            cache_dir=CACHE_DIR,
            html_refine_client=self.writer_client,
        )
        shutil.copy2(html_path, "outputs/custom/report.html")
        print(f"[Preview] HTML generated: {html_path}")
        print("[Preview] HTML copied to: outputs/custom/report.html")
