import json
import os
import shutil
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from core.model_client import ModelClient
from agents.planner import PlannerAgent
from agents.writer import WriterAgent
from utils.image_tools import ImageTools
from tasks.bench_utils import (
    append_references,
    build_preview_with_html_refine,
    build_report_output,
    clean_failed_image_markers,
    load_research_images,
    open_research_images,
    read_json,
    replace_section_citations,
    research_sections,
    write_json,
    write_text,
)


CACHE_ROOT = ".cache/dc/"
START_QID = 92


class DCTask:
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
        self.tools = ImageTools()

    def run(self):
        dataset = pd.read_csv("data/dc/queries.csv", encoding="utf-8")
        if "query" not in dataset.columns:
            raise ValueError("CSV must contain a 'query' column.")

        os.makedirs(CACHE_ROOT, exist_ok=True)
        os.makedirs("outputs/dc", exist_ok=True)

        outputs = []
        start_qid = getattr(self.args, "start_id", None) or START_QID
        end_qid = getattr(self.args, "end_id", None)
        limit = getattr(self.args, "limit", None)
        enable_html_refine = not getattr(self.args, "disable_html_refine", False)
        processed = 0

        for i, row in enumerate(tqdm(dataset.itertuples(index=False), total=len(dataset))):
            q_id = i + 1
            if q_id < start_qid:
                continue
            if end_qid is not None and q_id > end_qid:
                continue
            if limit is not None and processed >= limit:
                break

            row_dict = row._asdict() if hasattr(row, "_asdict") else {}
            question = row_dict.get("query", "")
            print(f"\nProcessing DC Question {q_id}:\n{question}\n")

            cache_dir = os.path.join(CACHE_ROOT, f"question_{q_id}")
            os.makedirs(cache_dir, exist_ok=True)

            planner = PlannerAgent(
                self.client, self.reviewer_client, cache_dir=cache_dir
            )
            planner.run(question)
            outline = read_json(os.path.join(cache_dir, "outline_output.json"))
            overview = outline.get("overview", "")
            sections = outline.get("sections", [])

            writer = WriterAgent(self.writer_client, self.tools, cache_dir=cache_dir)
            research_results = research_sections(
                sections=sections,
                client=self.client,
                reviewer_client=self.reviewer_client,
                cache_dir=cache_dir,
                question=question,
                overview=overview,
                section_workers=self.args.section_workers,
            )

            all_text_raw = ""
            all_text_processed = ""
            all_images = []

            for section in sections:
                section_id = section["id"]
                research_result = research_results[section_id]
                research_images = load_research_images(cache_dir, section_id)

                write_prompt = writer.create_section_prompt(
                    question=question,
                    overview=overview,
                    research_output=research_result,
                    web_images=research_images,
                )
                raw_text_0 = writer.write_section(write_prompt, section_id)
                raw_text = writer.refine_section(raw_text_0, section_id)
                raw_text = replace_section_citations(writer, research_result, raw_text)
                if not raw_text.strip():
                    raise ValueError(f"Writer produced empty text for question {q_id}, section {section_id}.")
                all_text_raw += raw_text + "\n\n"

                opened_research_images = open_research_images(research_images)
                section_text_processed, section_images = writer.gen_images(
                    raw_text,
                    opened_research_images,
                )
                if not section_text_processed.strip():
                    print(f"[Writer] Empty processed text for section {section_id}; falling back to raw text.")
                    section_text_processed = raw_text
                if not section_images and opened_research_images:
                    print(f"[Writer] No generated images for section {section_id}; inserting one research image fallback.")
                    section_text_processed = section_text_processed.rstrip() + "\n\n<image>\n"
                    section_images = [opened_research_images[0]]
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

            all_text_processed = clean_failed_image_markers(all_text_processed)
            if all_images:
                before_image_text = all_text_processed
                before_images = list(all_images)
                all_text_processed, all_images = writer.refine_images(
                    all_text_processed, all_images
                )
                if before_images and not all_images:
                    print("[Writer] Image refinement deleted all images; restoring pre-refinement images.")
                    all_text_processed = before_image_text
                    all_images = before_images
            all_text_processed, all_images = writer.refine_all_text(
                question, all_text_processed, all_images
            )
            if not all_text_processed.strip():
                print("[Writer] Final text refinement returned empty text; falling back to pre-reference raw text.")
                all_text_processed = all_text_raw

            final_citations = writer.get_final_citations()
            clean_text = writer.remove_citations(all_text_processed)
            output_raw_text, output_text = append_references(
                all_text_raw,
                all_text_processed,
                final_citations,
                language=row_dict.get("language", "en"),
            )

            write_text(os.path.join(cache_dir, "raw_report.txt"), output_raw_text)

            output = build_report_output(
                report_id=q_id,
                question=question,
                raw_text=output_raw_text,
                text=output_text,
                clean_text=clean_text,
                images=all_images,
            )
            outputs.append(output)
            write_json(f"outputs/dc/report_{q_id}.json", output)

            html_path = build_preview_with_html_refine(
                report_title=outline.get("report_title", "Final Report"),
                report_text=all_text_processed,
                report_imgs=all_images,
                final_citations=final_citations,
                cache_dir=cache_dir,
                html_refine_client=self.writer_client,
                enable_html_refine=enable_html_refine,
            )
            output_html_path = f"outputs/dc/report_{q_id}.html"
            os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
            shutil.copy2(html_path, output_html_path)
            print(f"[Preview] HTML generated for ID {q_id}: {html_path}")
            print(f"[Preview] HTML copied to: {output_html_path}")
            processed += 1

        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        write_json(f"outputs/dc/dc_all_{timestamp}.json", outputs)
