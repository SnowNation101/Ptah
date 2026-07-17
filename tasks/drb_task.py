import json
import os
import shutil

from tqdm import tqdm

from core.model_client import ModelClient
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
    read_text,
    replace_section_citations,
    write_json,
    write_text,
)


class DRBTask:
    def __init__(self, args):
        self.args = args
        self.client = ModelClient(args.model_name, args.base_url, args.api_key)
        self.reviewer_client = ModelClient(
            args.reviewer_model_name, args.reviewer_base_url, args.reviewer_api_key
        )
        self.writer_client = ModelClient(
            os.getenv("PTAH_VLM_MODEL_NAME", "models/Qwen3-VL-32B-Instruct"),
            os.getenv("PTAH_VLM_BASE_URL", "http://localhost:8001/v1/"),
            os.getenv("PTAH_VLM_API_KEY", "EMPTY"),
        )
        self.tools = ImageTools()

    def run(self):
        with open("data/drb/query.jsonl", "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]

        os.makedirs("outputs/drb", exist_ok=True)

        start_id = getattr(self.args, "start_id", None)
        end_id = getattr(self.args, "end_id", None)
        limit = getattr(self.args, "limit", None)
        enable_html_refine = not getattr(self.args, "disable_html_refine", False)
        processed = 0

        for item in tqdm(dataset):
            question = item["prompt"]
            item_id = item["id"]
            if start_id is not None and item_id < start_id:
                continue
            if end_id is not None and item_id > end_id:
                continue
            if limit is not None and processed >= limit:
                break

            language = item.get("language", "en")
            print(f"\nProcessing DRB Question {item_id}:\n{question}\n")

            cache_dir = f".cache/drb/question_{item_id}"
            os.makedirs(cache_dir, exist_ok=True)

            # DRB currently runs from cached planner/research/writer outputs, then
            # applies test-time scaling and final rendering.
            outline = read_json(os.path.join(cache_dir, "outline_output.json"))
            writer = WriterAgent(self.writer_client, self.tools, cache_dir=cache_dir)

            all_text_raw = ""
            all_text_processed = ""
            all_images = []

            for section in outline.get("sections", []):
                section_id = section["id"]
                research_result = read_json(
                    os.path.join(cache_dir, f"research_output_section_{section_id}.json")
                )
                research_images = load_research_images(cache_dir, section_id)

                raw_text_0 = read_text(
                    os.path.join(cache_dir, f"writer_output_section_{section_id}.txt")
                )

                raw_text = writer.refine_section(raw_text_0, section_id)
                raw_text = replace_section_citations(writer, research_result, raw_text)
                all_text_raw += raw_text + "\n\n"

                section_text_processed, section_images = writer.gen_images(
                    raw_text,
                    open_research_images(research_images),
                )
                all_text_processed += section_text_processed + "\n\n"
                all_images.extend(section_images)

            conclusion_raw = read_text(os.path.join(cache_dir, "writer_output_conclusion.txt"))
            all_text_raw += conclusion_raw
            all_text_processed += conclusion_raw

            all_text_processed = clean_failed_image_markers(all_text_processed)
            if all_images:
                all_text_processed, all_images = writer.refine_images(
                    all_text_processed, all_images
                )
            all_text_processed, all_images = writer.refine_all_text(
                question, all_text_processed, all_images
            )

            final_citations = writer.get_final_citations()
            clean_text = writer.remove_citations(all_text_processed)
            output_raw_text, output_text = append_references(
                all_text_raw,
                all_text_processed,
                final_citations,
                language=language,
            )

            write_text(os.path.join(cache_dir, "raw_report.txt"), output_raw_text)

            output = build_report_output(
                report_id=item_id,
                question=question,
                raw_text=output_raw_text,
                text=output_text,
                clean_text=clean_text,
                images=all_images,
            )
            write_json(f"outputs/drb/report_{item_id}.json", output)

            html_path = build_preview_with_html_refine(
                report_title=outline.get("report_title", "Final Report"),
                report_text=all_text_processed,
                report_imgs=all_images,
                final_citations=final_citations,
                cache_dir=cache_dir,
                html_refine_client=self.writer_client,
                enable_html_refine=enable_html_refine,
            )
            output_html_path = f"outputs/drb/report_{item_id}.html"
            os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
            shutil.copy2(html_path, output_html_path)
            print(f"[Preview] HTML generated for ID {item_id}: {html_path}")
            print(f"[Preview] HTML copied to: {output_html_path}")
            processed += 1
