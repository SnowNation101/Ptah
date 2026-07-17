from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from utils.common import (
    DEFAULT_JUDGE_PROVIDER,
    DEFAULT_SILICONFLOW_API_URL,
    DEFAULT_UNIAPI_BASE_URL,
    DEFAULT_UNIAPI_MODEL,
    call_judge_chat,
    ensure_data_url,
    extract_assistant_text,
    extract_json_object,
    project_path,
    read_json,
    resolve_api_key,
)


DIMENSIONS = [
    "instruction_following",
    "comprehensiveness",
    "completeness",
    "writing_quality",
]

DEFAULT_DC_BASELINE_CSV = "data/dc/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv"

OFFICIAL_PAIRWISE_SYSTEM_PROMPT = """
You are an expert evaluator for reports to a research question.

You'll be comparing two responses to a research question: report_a and report_b.
Evaluate both reports on these dimensions:
1. Instruction following: Evaluates response's fidelity to user specified instructions and constraints.
2. Comprehensiveness: Measures breadth and range of information covered in response, addressing the scope of user request.
3. Completeness: Measures the depth and thoroughness of information for topics addressed in the report.
4. Writing quality: Evaluates clarity, conciseness, logical organization and overall readability of the report.

For each dimension, indicate which report you prefer ("a" or "b") and provide a concise explanation for your choice.
Your explanations should cite specific examples to justify your preference and point out what can be improved in the other report.
Also provide a gap_score from 0 to 5, where 0 indicates similar quality and 5 is the maximum difference.

Be fair and objective. Do not be biased towards either report A or B.
The length of a report is not necessarily an indicator of quality; focus on substance and how well it meets the user's needs.

Return strictly valid JSON with exactly this structure:
{
  "instruction_following": {"preferred": "a", "gap_score": 0, "explanation": ""},
  "comprehensiveness": {"preferred": "a", "gap_score": 0, "explanation": ""},
  "completeness": {"preferred": "a", "gap_score": 0, "explanation": ""},
  "writing_quality": {"preferred": "a", "gap_score": 0, "explanation": ""}
}
""".strip()

def load_dc_input_rows(path: str | Path) -> dict[int, dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DC input CSV not found: {p}")

    rows: dict[int, dict[str, str]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = {"question", "baseline_answer"}
        missing = sorted(required - set(fieldnames))
        if missing:
            raise ValueError(f"DC input CSV missing required columns: {missing}; got {fieldnames}")
        for idx, row in enumerate(reader, start=1):
            rows[idx] = {
                "question": row.get("question", ""),
                "baseline_answer": row.get("baseline_answer", ""),
            }
    return rows


def report_text_with_placeholders(report: dict[str, Any]) -> str:
    return report.get("clean_text") or report.get("text") or report.get("raw_text") or ""


def replace_markdown_links_with_text(text: str, replacement: str = "") -> str:
    """Mirror the official DC preprocessing by removing markdown link URLs."""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", rf"\1{replacement}", text or "")


def build_multimodal_report_content(text: str, images: list[str], include_images: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    image_idx = 0
    parts = re.split(r"(<image>)", text or "")
    for part in parts:
        if part == "<image>":
            if include_images and image_idx < len(images):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": ensure_data_url(images[image_idx])},
                    }
                )
            elif not include_images:
                content.append({"type": "text", "text": "[Image omitted from judge input]"})
            else:
                content.append({"type": "text", "text": "[Missing image placeholder]"})
            image_idx += 1
        elif part.strip():
            content.append({"type": "text", "text": part})

    if include_images and image_idx < len(images):
        content.append({"type": "text", "text": "\n[Additional report images]\n"})
        while image_idx < len(images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": ensure_data_url(images[image_idx])},
                }
            )
            image_idx += 1
    return content


def build_pairwise_messages(
    question: str,
    baseline_answer: str,
    candidate_text: str,
    images: list[str],
    include_images: bool,
    flip: bool = False,
) -> list[dict[str, Any]]:
    baseline_answer = replace_markdown_links_with_text(baseline_answer)
    candidate_text = replace_markdown_links_with_text(candidate_text)
    report_a_label = "candidate" if flip else "baseline"
    report_b_label = "baseline" if flip else "candidate"
    report_a_text = candidate_text if flip else baseline_answer
    report_b_text = baseline_answer if flip else candidate_text

    content: list[dict[str, Any]] = [
        {"type": "text", "text": f"Research question:\n{question}\n\n"},
        {"type": "text", "text": f"report_a ({report_a_label}):\n"},
    ]
    if flip:
        content.extend(build_multimodal_report_content(report_a_text, images, include_images))
    else:
        content.append({"type": "text", "text": report_a_text})
    content.append({"type": "text", "text": f"\n\nreport_b ({report_b_label}):\n"})
    if flip:
        content.append({"type": "text", "text": report_b_text})
    else:
        content.extend(build_multimodal_report_content(report_b_text, images, include_images))

    return [
        {"role": "system", "content": OFFICIAL_PAIRWISE_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def call_judge(
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
    response_json = call_judge_chat(
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
    raw_text = extract_assistant_text(response_json)
    return {"raw_output": raw_text, "parsed_output": extract_json_object(raw_text)}


def validate_pairwise_output(obj: dict[str, Any], candidate_is_b: bool) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for dim in DIMENSIONS:
        item = obj.get(dim)
        if not isinstance(item, dict):
            raise ValueError(f"Missing pairwise result for {dim}.")
        preferred = str(item.get("preferred", "")).lower()
        if preferred not in {"a", "b"}:
            raise ValueError(f"Invalid preferred value for {dim}: {preferred}")
        gap_score = float(item.get("gap_score", 0))
        if gap_score < 0 or gap_score > 5:
            raise ValueError(f"Invalid gap_score for {dim}: {gap_score}")
        if gap_score == 0:
            candidate_delta = 0.0
            candidate_preferred = False
            outcome = "tie"
        else:
            candidate_preferred = preferred == ("b" if candidate_is_b else "a")
            candidate_delta = gap_score if candidate_preferred else -gap_score
            outcome = "win" if candidate_delta > 0 else "lose"
        results[dim] = {
            "preferred": preferred,
            "gap_score": gap_score,
            "candidate_delta": candidate_delta,
            "candidate_score": 5 + candidate_delta,
            "candidate_preferred": candidate_preferred,
            "outcome": outcome,
            "explanation": item.get("explanation", ""),
        }
    return results


def consensus_from_pairwise_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, float] = {}
    stats: dict[str, dict[str, Any]] = {}
    for dim in DIMENSIONS:
        deltas: list[float] = []
        explanations: list[str] = []
        wins = losses = ties = 0
        for trial in trials:
            original = trial["original"]["scores"][dim]
            flipped = trial["flipped"]["scores"][dim]
            for item in (original, flipped):
                delta = float(item["candidate_delta"])
                deltas.append(delta)
                if delta > 0:
                    wins += 1
                elif delta < 0:
                    losses += 1
                else:
                    ties += 1
                explanation = str(item.get("explanation", "")).strip()
                if explanation:
                    explanations.append(explanation)
        total = len(deltas)
        avg_delta = sum(deltas) / total if total else 0.0
        if wins > total / 2:
            result = "win"
        elif losses > total / 2:
            result = "lose"
        else:
            result = "tie"
        scores[dim] = round(5 + avg_delta, 4)
        stats[dim] = {
            "result": result,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round(wins / total, 4) if total else None,
            "lose_rate": round(losses / total, 4) if total else None,
            "tie_rate": round(ties / total, 4) if total else None,
            "net_winrate": round(wins / (wins + losses), 4) if wins + losses else 0.5,
            "avg_signed_gap": round(avg_delta, 4),
            "score": scores[dim],
            "explanations": explanations,
        }
    return {
        "scores": scores,
        "overall_score": round(sum(scores.values()) / len(DIMENSIONS), 4),
        "dimension_stats": stats,
    }


def evaluate_report(
    *,
    report_id: int,
    question: str,
    report_dir: Path,
    baseline_answer: str,
    include_images: bool,
    provider: str,
    api_key: str,
    api_url: str | None,
    base_url: str | None,
    model: str,
    max_tokens: int,
    timeout: int,
    num_trials: int,
    metric_num_workers: int,
) -> dict[str, Any]:
    report_path = report_dir / f"report_{report_id}.json"
    report = read_json(report_path)
    candidate_text = report_text_with_placeholders(report)
    images = report.get("images", [])
    if not isinstance(images, list):
        images = []

    if not baseline_answer:
        raise ValueError("DC pairwise evaluation requires a non-empty baseline_answer.")

    def run_pairwise_call(trial_idx: int, flip: bool) -> tuple[int, str, dict[str, Any]]:
        result = call_judge(
            provider=provider,
            api_key=api_key,
            api_url=api_url,
            base_url=base_url,
            model=model,
            messages=build_pairwise_messages(question, baseline_answer, candidate_text, images, include_images, flip=flip),
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return trial_idx, "flipped" if flip else "original", result

    trial_map: dict[int, dict[str, Any]] = {
        trial_idx: {"trial": trial_idx}
        for trial_idx in range(1, num_trials + 1)
    }
    jobs = [
        (trial_idx, flip)
        for trial_idx in range(1, num_trials + 1)
        for flip in (False, True)
    ]
    with ThreadPoolExecutor(max_workers=max(1, metric_num_workers)) as executor:
        futures = [executor.submit(run_pairwise_call, trial_idx, flip) for trial_idx, flip in jobs]
        for future in as_completed(futures):
            trial_idx, order_key, result = future.result()
            if order_key == "original":
                trial_map[trial_idx]["original"] = {
                    "candidate_position": "b",
                    "scores": validate_pairwise_output(result["parsed_output"], candidate_is_b=True),
                    "raw_output": result["raw_output"],
                    "parsed_output": result["parsed_output"],
                }
            else:
                trial_map[trial_idx]["flipped"] = {
                    "candidate_position": "a",
                    "scores": validate_pairwise_output(result["parsed_output"], candidate_is_b=False),
                    "raw_output": result["raw_output"],
                    "parsed_output": result["parsed_output"],
                }

    trials = [trial_map[trial_idx] for trial_idx in range(1, num_trials + 1)]
    consensus = consensus_from_pairwise_trials(trials)
    return {
        "id": report_id,
        "question": question,
        "mode": "pairwise",
        "num_trials": num_trials,
        "report_path": str(report_path),
        "scores": consensus["scores"],
        "overall_score": consensus["overall_score"],
        "dimension_stats": consensus["dimension_stats"],
        "trials": trials,
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in records if r.get("status") == "success"]
    summary: dict[str, Any] = {
        "num_total": len(records),
        "num_success": len(ok),
        "num_failed": len(records) - len(ok),
    }
    for dim in DIMENSIONS:
        values = [float(r["scores"][dim]) for r in ok if dim in r.get("scores", {})]
        summary[dim] = round(sum(values) / len(values), 4) if values else None
        dim_stats = [r.get("dimension_stats", {}).get(dim, {}) for r in ok]
        for key in ("win_rate", "lose_rate", "tie_rate", "net_winrate", "avg_signed_gap"):
            stat_values = [float(s[key]) for s in dim_stats if isinstance(s, dict) and s.get(key) is not None]
            summary[f"{dim}_{key}"] = round(sum(stat_values) / len(stat_values), 4) if stat_values else None
        result_values = [s.get("result") for s in dim_stats if isinstance(s, dict)]
        total_results = len(result_values)
        summary[f"{dim}_result_win_rate"] = round(result_values.count("win") / total_results, 4) if total_results else None
        summary[f"{dim}_result_lose_rate"] = round(result_values.count("lose") / total_results, 4) if total_results else None
        summary[f"{dim}_result_tie_rate"] = round(result_values.count("tie") / total_results, 4) if total_results else None
    overall = [float(r["overall_score"]) for r in ok if "overall_score" in r]
    summary["overall_score"] = round(sum(overall) / len(overall), 4) if overall else None
    dimension_result_keys = [f"{dim}_result_win_rate" for dim in DIMENSIONS]
    win_rates = [summary[k] for k in dimension_result_keys if summary.get(k) is not None]
    summary["macro_result_win_rate"] = round(sum(win_rates) / len(win_rates), 4) if win_rates else None
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate DeepConsult/DC reports with Qwen3-VL judge.")
    parser.add_argument("--input-data", default=DEFAULT_DC_BASELINE_CSV, help="Official DC CSV with question and baseline_answer columns.")
    parser.add_argument("--report-dir", default=str(project_path("outputs", "dc")))
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=0, help="Inclusive end id. 0 means no explicit end.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--num-trials", type=int, default=3, help="Official DC pairwise trials per report.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel report evaluation workers.")
    parser.add_argument("--metric-num-workers", type=int, default=3, help="Parallel judge calls inside each pairwise metric.")
    parser.add_argument("--include-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-jsonl", default=str(project_path("eval_results", "dc", "dc_eval_results.jsonl")))
    parser.add_argument("--out-summary", default=str(project_path("eval_results", "dc", "dc_eval_summary.json")))
    parser.add_argument("--provider", default=DEFAULT_JUDGE_PROVIDER, choices=["uniapi", "siliconflow", "openai-compatible", "openai", "local", "vllm"])
    parser.add_argument("--model", default=DEFAULT_UNIAPI_MODEL)
    parser.add_argument("--api-url", default="", help=f"Full chat completions URL. SiliconFlow default: {DEFAULT_SILICONFLOW_API_URL}")
    parser.add_argument("--base-url", default=DEFAULT_UNIAPI_BASE_URL, help="OpenAI-compatible base URL, e.g. https://hk.uniapi.io/v1.")
    parser.add_argument("--api-key-env", default="", help="Environment variable containing API key. UniAPI tries UNIAPI_API_KEY then OPENAI_API_KEY by default.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=600)
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

    input_rows = load_dc_input_rows(args.input_data)
    report_dir = Path(args.report_dir)
    out_jsonl = Path(args.out_jsonl)
    out_summary = Path(args.out_summary)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    ids = [i for i in sorted(input_rows) if i >= args.start_id]
    if args.end_id:
        ids = [i for i in ids if i <= args.end_id]
    if args.limit > 0:
        ids = ids[: args.limit]

    missing_baselines = [i for i in ids if not input_rows[i].get("baseline_answer")]
    if missing_baselines:
        preview = ", ".join(str(i) for i in missing_baselines[:10])
        suffix = "..." if len(missing_baselines) > 10 else ""
        print(
            f"DC input CSV is missing baseline_answer for report ids: {preview}{suffix}",
            file=sys.stderr,
        )
        return 2

    records: list[dict[str, Any]] = []
    indexed_ids = {report_id: idx for idx, report_id in enumerate(ids, start=1)}

    def run_one(report_id: int) -> dict[str, Any]:
        question = input_rows[report_id]["question"]
        baseline_answer = input_rows[report_id]["baseline_answer"]
        print(f"[{indexed_ids[report_id]}/{len(ids)}] Evaluating DC report_{report_id}.json mode=pairwise")
        try:
            record = evaluate_report(
                report_id=report_id,
                question=question,
                report_dir=report_dir,
                baseline_answer=baseline_answer,
                include_images=args.include_images,
                provider=args.provider,
                api_key=api_key,
                api_url=args.api_url or None,
                base_url=args.base_url or None,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                num_trials=args.num_trials,
                metric_num_workers=args.metric_num_workers,
            )
            record["status"] = "success"
            print("  scores:", record["scores"], "overall:", record["overall_score"])
            return record
        except Exception as e:
            print("  failed:", e)
            return {
                "id": report_id,
                "question": question,
                "status": "failed",
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_to_id = {executor.submit(run_one, report_id): report_id for report_id in ids}
        for future in as_completed(future_to_id):
            records.append(future.result())

    records.sort(key=lambda item: int(item.get("id", 0)))
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = aggregate(records)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved jsonl: {out_jsonl}")
    print(f"Saved summary: {out_summary}")
    return 0 if summary["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
