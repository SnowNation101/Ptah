from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from utils.common import (
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SILICONFLOW_API_URL,
    DEFAULT_UNIAPI_BASE_URL,
    DEFAULT_UNIAPI_MODEL,
    average_score_dict,
    call_judge_chat,
    ensure_data_url,
    extract_assistant_text,
    extract_json_object,
    project_path,
    read_json,
    read_text,
    resolve_api_key,
    write_json,
)


DEFAULT_REPORT_DIR = project_path("outputs", "drb")
DEFAULT_PROMPT_PATH = project_path("prompts", "icq_eval.txt")
DEFAULT_OUTPUT_PATH = project_path("eval_results", "icq_eval_results.json")


def build_interleaved_user_content(question: str, text: str, images: list[str]) -> list[dict[str, Any]]:
    parts = re.split(r"(<image>)", text or "")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Evaluate the following multimodal analytical report according to the system instructions.\n\n"
                f"Original research question:\n{question}\n\n"
                "Report content:\n"
            ),
        }
    ]

    image_idx = 0
    for part in parts:
        if part == "<image>":
            if image_idx < len(images):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": ensure_data_url(images[image_idx])},
                    }
                )
                image_idx += 1
            else:
                content.append(
                    {
                        "type": "text",
                        "text": "[Warning: Missing image for this <image> placeholder.]",
                    }
                )
        elif part.strip():
            content.append({"type": "text", "text": part})

    if image_idx < len(images):
        content.append(
            {
                "type": "text",
                "text": "\n[Additional images appended because there are more images than <image> placeholders.]\n",
            }
        )
        while image_idx < len(images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": ensure_data_url(images[image_idx])},
                }
            )
            image_idx += 1

    return content


def load_report(report_dir: Path, report_id: int) -> dict[str, Any]:
    report_path = report_dir / f"report_{report_id}.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    report_json = read_json(report_path)
    images = report_json.get("images", [])
    if not isinstance(images, list):
        raise TypeError(f"'images' must be a list in {report_path}")

    return {
        "report_id": report_id,
        "report_path": str(report_path),
        "question": report_json.get("question", ""),
        "text": report_json.get("clean_text") or report_json.get("text") or "",
        "images": images,
    }


def call_icq_eval(
    *,
    provider: str,
    api_key: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    system_prompt: str,
    question: str,
    text: str,
    images: list[str],
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    resp_json = call_judge_chat(
        provider=provider,
        api_key=api_key,
        api_url=api_url,
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_interleaved_user_content(question, text, images),
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
    )
    raw_output = extract_assistant_text(resp_json)
    return {
        "raw_output": raw_output,
        "parsed_output": extract_json_object(raw_output),
    }


def iter_report_ids(start_id: int, end_id: int, limit: int | None) -> list[int]:
    ids = list(range(start_id, end_id + 1))
    if limit is not None and limit > 0:
        ids = ids[:limit]
    return ids


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Image Content Quality (ICQ) for report_*.json files."
    )
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory containing report_*.json.")
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT_PATH), help="ICQ system prompt path.")
    parser.add_argument("--out-json", default=str(DEFAULT_OUTPUT_PATH), help="Output JSON path.")
    parser.add_argument("--start-id", type=int, default=1, help="First report id.")
    parser.add_argument("--end-id", type=int, default=10, help="Last report id.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of reports to evaluate.")
    parser.add_argument("--provider", default=DEFAULT_JUDGE_PROVIDER, choices=["uniapi", "siliconflow", "openai-compatible", "openai", "local", "vllm"], help="Judge provider.")
    parser.add_argument("--model", default=DEFAULT_UNIAPI_MODEL, help=f"Judge model. Legacy SiliconFlow model: {DEFAULT_JUDGE_MODEL}")
    parser.add_argument("--api-url", default="", help=f"Full chat completions URL. SiliconFlow default: {DEFAULT_SILICONFLOW_API_URL}")
    parser.add_argument("--base-url", default=DEFAULT_UNIAPI_BASE_URL, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key-env", default="", help="Environment variable containing API key. UniAPI tries UNIAPI_API_KEY then OPENAI_API_KEY by default.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature.")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Judge max_tokens.")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds.")
    parser.add_argument("--sleep", type=float, default=1.0, help="Sleep seconds between requests.")
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

    report_dir = Path(args.report_dir)
    system_prompt = read_text(args.prompt)
    report_ids = iter_report_ids(args.start_id, args.end_id, args.limit or None)

    all_results: list[dict[str, Any]] = []
    successful_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, Any]] = []

    for report_id in report_ids:
        print(f"\n{'=' * 20} Evaluating report_{report_id}.json {'=' * 20}")
        try:
            report = load_report(report_dir, report_id)
            eval_result = call_icq_eval(
                provider=args.provider,
                api_key=api_key,
                api_url=args.api_url or None,
                base_url=args.base_url or None,
                model=args.model,
                system_prompt=system_prompt,
                question=report["question"],
                text=report["text"],
                images=report["images"],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            record = {
                "report_id": report_id,
                "report_path": report["report_path"],
                "question": report["question"],
                "parsed_output": eval_result["parsed_output"],
                "raw_output": eval_result["raw_output"],
                "status": "success",
            }
            all_results.append(record)
            successful_results.append(record)
            print(json.dumps(eval_result["parsed_output"], ensure_ascii=False, indent=2))
        except Exception as e:
            record = {
                "report_id": report_id,
                "report_path": str(report_dir / f"report_{report_id}.json"),
                "status": "failed",
                "error": str(e),
            }
            all_results.append(record)
            failed_results.append(record)
            print(f"Failed on report_{report_id}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    average_scores = average_score_dict(
        [
            {"scores": item.get("parsed_output", {}).get("scores", {})}
            for item in successful_results
        ]
    )
    summary = {
        "evaluated_range": [args.start_id, args.end_id],
        "num_success": len(successful_results),
        "num_failed": len(failed_results),
        "average_scores": average_scores,
        "failed_reports": failed_results,
    }
    write_json(args.out_json, {"summary": summary, "results": all_results})

    print(f"\n{'=' * 20} Final Summary {'=' * 20}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved full results to: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
