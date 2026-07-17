from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from utils.common import (
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SILICONFLOW_API_URL,
    DEFAULT_UNIAPI_BASE_URL,
    DEFAULT_UNIAPI_MODEL,
    call_judge_chat,
    extract_assistant_text,
    project_path,
    resolve_api_key,
)

DEFAULT_PROMPT_PATH = project_path("prompts", "mpq_eval.txt")
DEFAULT_API_URL = DEFAULT_SILICONFLOW_API_URL
DEFAULT_MODEL = DEFAULT_UNIAPI_MODEL

DEFAULT_RETRY_INTERVAL_SECONDS = 10
DEFAULT_MAX_RETRIES = 10


class ModelOutputParseError(Exception):
    """Raised when model output cannot be parsed as the expected JSON."""


def read_text_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File does not exist: {p}")
    return p.read_text(encoding="utf-8")


def file_to_data_url(image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Screenshot file does not exist: {p}")

    suffix = p.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(suffix, "application/octet-stream")
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def call_judge_vision(
    api_key: str,
    provider: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    system_prompt: str,
    image_data_url: str,
    temperature: float = 0.2,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> dict[str, Any]:
    return call_judge_chat(
        provider=provider,
        api_key=api_key,
        api_url=api_url,
        base_url=base_url,
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Evaluate this webpage screenshot according to the system prompt and output only the structured JSON scoring result.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format={"type": "json_object"},
    )


def try_parse_json_block(text: str) -> dict[str, Any] | None:
    """
    Extract JSON from model output.
    Supports a full JSON response, a fenced JSON block, or a JSON object embedded in text.
    """
    text = text.strip()
    if not text:
        return None

    # Full response is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # JSON in a fenced code block.
    code_block_patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in code_block_patterns:
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            block = m.group(1)
            try:
                obj = json.loads(block)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # JSON object embedded in text.
    candidates = re.findall(r"\{.*\}", text, flags=re.DOTALL)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def extract_scores_from_json(obj: dict[str, Any]) -> dict[str, float]:
    """
    Extract numeric metric scores from the parsed JSON.
    Supports examples such as:
    {
      "scores": {"layout": 4, "readability": 3.5}
    }
    or:
    {
      "layout": 4,
      "readability": 3.5
    }
    """
    scores: dict[str, float] = {}

    if "scores" in obj and isinstance(obj["scores"], dict):
        source = obj["scores"]
    else:
        source = obj

    for k, v in source.items():
        if isinstance(v, (int, float)):
            scores[str(k)] = float(v)
        elif isinstance(v, str):
            vv = v.strip()
            if re.fullmatch(r"-?\d+(\.\d+)?", vv):
                scores[str(k)] = float(vv)

    return scores


def parse_scores_from_json_only(assistant_text: str) -> tuple[dict[str, Any], dict[str, float]]:
    """Require parseable JSON with numeric scores; otherwise trigger retry."""
    obj = try_parse_json_block(assistant_text)
    if obj is None:
        raise ModelOutputParseError("Could not parse JSON from model output.")

    scores = extract_scores_from_json(obj)
    if not scores:
        raise ModelOutputParseError("Parsed JSON but found no numeric score fields.")

    return obj, scores


def evaluate_one_image_with_retry(
    *,
    qid: int,
    image_path: Path,
    api_key: str,
    provider: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retry_interval_seconds: int,
    max_retries: int,
) -> dict[str, Any]:
    image_data_url = file_to_data_url(image_path)

    last_error = ""
    last_raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Attempt {attempt}/{max_retries}")

            resp_json = call_judge_vision(
                api_key=api_key,
                provider=provider,
                api_url=api_url,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                image_data_url=image_data_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )

            assistant_text = extract_assistant_text(resp_json)
            last_raw_text = assistant_text.strip()

            parsed_json, scores = parse_scores_from_json_only(assistant_text)

            return {
                "id": qid,
                "success": True,
                "scores": scores,
                "raw_text": assistant_text.strip(),
                "parsed_json": parsed_json,
                "image_path": str(image_path),
                "attempts": attempt,
                "error": "",
            }

        except (RuntimeError, ModelOutputParseError) as e:
            last_error = str(e)
            print(f"  Attempt {attempt} failed: {last_error}")

            if attempt < max_retries:
                print(f"  Retry after {retry_interval_seconds}s...")
                time.sleep(retry_interval_seconds)
            else:
                print("  Reached max retries, mark this question as failed.")

    return {
        "id": qid,
        "success": False,
        "scores": {},
        "raw_text": last_raw_text,
        "parsed_json": None,
        "image_path": str(image_path),
        "attempts": max_retries,
        "error": last_error,
    }


def extract_question_id(path: Path) -> int | None:
    m = re.fullmatch(r"question_(\d+)\.(png|jpg|jpeg|webp)", path.name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def collect_images(input_dir: str | Path) -> list[tuple[int, Path]]:
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(f"Input directory does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {p}")

    items: list[tuple[int, Path]] = []
    for file in p.iterdir():
        if not file.is_file():
            continue
        qid = extract_question_id(file)
        if qid is not None:
            items.append((qid, file))

    items.sort(key=lambda x: x[0])
    return items


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate question_*.png screenshots in a directory, write JSONL, and print metric averages."
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing screenshots, for example output_screenshots/drb/ptah",
    )
    parser.add_argument(
        "--out-jsonl",
        default="evaluation_results.jsonl",
        help="Output JSONL path. Default: evaluation_results.jsonl",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT_PATH,
        help=f"Vision judge prompt path. Default: {DEFAULT_PROMPT_PATH}",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_JUDGE_PROVIDER,
        choices=["uniapi", "siliconflow", "openai-compatible", "openai", "local", "vllm"],
        help="Judge provider.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name. Default: {DEFAULT_MODEL}. Legacy SiliconFlow model: {DEFAULT_JUDGE_MODEL}",
    )
    parser.add_argument(
        "--api-url",
        default="",
        help=f"Full Chat Completions URL. SiliconFlow default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_UNIAPI_BASE_URL,
        help=f"OpenAI-compatible base URL. Default: {DEFAULT_UNIAPI_BASE_URL}",
    )
    parser.add_argument(
        "--api-key-env",
        default="",
        help="API key environment variable. UniAPI tries UNIAPI_API_KEY then OPENAI_API_KEY by default.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Model temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Model max_tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit: process the first N images. Use 0 for all.",
    )
    parser.add_argument(
        "--retry-interval",
        type=int,
        default=DEFAULT_RETRY_INTERVAL_SECONDS,
        help=f"Retry interval in seconds. Default: {DEFAULT_RETRY_INTERVAL_SECONDS}",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Maximum retries per image. Default: {DEFAULT_MAX_RETRIES}",
    )
    return parser


def main() -> int:
    load_dotenv()
    parser = build_argparser()
    args = parser.parse_args()

    api_key, used_api_key_env = resolve_api_key(args.api_key_env, args.provider)
    if not api_key and args.provider in {"local", "vllm"}:
        api_key = "EMPTY"
    if not api_key:
        print(
            f"Error: environment variable {used_api_key_env} was not found. Configure it in .env or the environment.",
            file=sys.stderr,
        )
        return 2

    try:
        system_prompt = read_text_file(args.prompt)
        images = collect_images(args.input_dir)

        if not images:
            print("No screenshot files matching the naming rule were found, for example question_1.png.", file=sys.stderr)
            return 1

        if args.limit > 0:
            images = images[: args.limit]

        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        metric_sums: dict[str, float] = defaultdict(float)
        metric_counts: dict[str, int] = defaultdict(int)

        total = len(images)
        print(f"Found {total} screenshots. Starting evaluation...")
        print(
            f"Retry policy: retry every {args.retry_interval}s when a request fails "
            f"or model output cannot be parsed as JSON, up to {args.max_retries} attempts."
        )

        success_count = 0
        failed_count = 0

        with out_path.open("w", encoding="utf-8") as f:
            for idx, (qid, image_path) in enumerate(images, start=1):
                print(f"\n[{idx}/{total}] Processing question_{qid}: {image_path}")

                record = evaluate_one_image_with_retry(
                    qid=qid,
                    image_path=image_path,
                    api_key=api_key,
                    provider=args.provider,
                    api_url=args.api_url or None,
                    base_url=args.base_url or None,
                    model=args.model,
                    system_prompt=system_prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    retry_interval_seconds=args.retry_interval,
                    max_retries=args.max_retries,
                )

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if record["success"]:
                    success_count += 1
                    scores = record["scores"]
                    for metric, value in scores.items():
                        metric_sums[metric] += value
                        metric_counts[metric] += 1
                    print(f"  Extracted scores: {scores}")
                else:
                    failed_count += 1
                    print(f"  Failed after retries: {record['error']}")

        print(f"\nEvaluation results saved to: {out_path}")
        print(f"Success: {success_count}, failed: {failed_count}")

        print("\n=== Metric Averages ===")
        if not metric_sums:
            print("No metric scores were parsed successfully.")
        else:
            for metric in sorted(metric_sums.keys()):
                avg = metric_sums[metric] / metric_counts[metric]
                print(f"{metric}: {avg:.4f}  (n={metric_counts[metric]})")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Run failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
