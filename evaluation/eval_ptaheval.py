from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from utils.common import DEFAULT_JUDGE_PROVIDER, DEFAULT_UNIAPI_BASE_URL, DEFAULT_UNIAPI_MODEL, project_path


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PtahEval ICQ and MPQ for a report set.")
    parser.add_argument("--bench", choices=["dc", "drb", "custom"], default="drb")
    parser.add_argument("--report-dir", default="", help="Directory containing report_*.json.")
    parser.add_argument("--html-dir", default="", help="Directory containing report_*.html.")
    parser.add_argument("--screenshot-dir", default="", help="Directory for question_*.png screenshots.")
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", default=DEFAULT_JUDGE_PROVIDER, choices=["uniapi", "siliconflow", "openai-compatible", "openai", "local", "vllm"])
    parser.add_argument("--model", default=DEFAULT_UNIAPI_MODEL)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--base-url", default=DEFAULT_UNIAPI_BASE_URL)
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--out-dir", default="", help="Directory for PtahEval outputs.")
    parser.add_argument("--skip-icq", action="store_true")
    parser.add_argument("--skip-mpq", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1800)
    return parser


def main() -> int:
    load_dotenv()
    args = build_argparser().parse_args()

    report_dir = Path(args.report_dir) if args.report_dir else project_path("outputs", args.bench)
    html_dir = Path(args.html_dir) if args.html_dir else report_dir
    screenshot_dir = (
        Path(args.screenshot_dir)
        if args.screenshot_dir
        else project_path("output_screenshots", args.bench, "ptah")
    )
    out_dir = Path(args.out_dir) if args.out_dir else project_path("eval_results", args.bench, "ptaheval")
    out_dir.mkdir(parents=True, exist_ok=True)

    common_judge_args = [
        "--provider",
        args.provider,
        "--model",
        args.model,
        "--base-url",
        args.base_url,
    ]
    if args.api_url:
        common_judge_args.extend(["--api-url", args.api_url])
    if args.api_key_env:
        common_judge_args.extend(["--api-key-env", args.api_key_env])

    if not args.skip_mpq and not args.skip_render:
        render_cmd = [
            sys.executable,
            "-u",
            "evaluation/render_report_screenshots.py",
            "--html-dir",
            str(html_dir),
            "--output-dir",
            str(screenshot_dir),
            "--start-id",
            str(args.start_id),
            "--end-id",
            str(args.end_id),
            "--limit",
            str(args.limit),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
        ]
        if args.force_render:
            render_cmd.append("--force")
        run_command(render_cmd)

    if not args.skip_icq:
        icq_cmd = [
            sys.executable,
            "-u",
            "evaluation/eval_image.py",
            "--report-dir",
            str(report_dir),
            "--start-id",
            str(args.start_id),
            "--end-id",
            str(args.end_id),
            "--limit",
            str(args.limit),
            "--out-json",
            str(out_dir / "icq_eval_results.json"),
            *common_judge_args,
        ]
        run_command(icq_cmd)

    if not args.skip_mpq:
        mpq_cmd = [
            sys.executable,
            "-u",
            "evaluation/eval_screenshot.py",
            str(screenshot_dir),
            "--limit",
            str(args.limit),
            "--out-jsonl",
            str(out_dir / "mpq_eval_results.jsonl"),
            *common_judge_args,
        ]
        run_command(mpq_cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
