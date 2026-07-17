from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
from pathlib import Path


def find_chromium_executable() -> str | None:
    env_path = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
    candidates = [
        env_path,
        shutil.which("chromium"),
        shutil.which("chromium-browser"),
        shutil.which("google-chrome"),
        shutil.which("google-chrome-stable"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def report_id_from_html(path: Path) -> int | None:
    match = re.fullmatch(r"report_(\d+)\.html", path.name)
    if not match:
        return None
    return int(match.group(1))


async def render_one(html_path: Path, output_path: Path, width: int, height: int, extra_wait_ms: int) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        launch_kwargs = {"headless": True, "args": ["--no-sandbox"]}
        executable = find_chromium_executable()
        if executable:
            launch_kwargs["executable_path"] = executable
        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = await context.new_page()
        await page.goto(html_path.resolve().as_uri(), wait_until="networkidle", timeout=60_000)
        if extra_wait_ms > 0:
            await page.wait_for_timeout(extra_wait_ms)
        scroll_height = await page.evaluate("() => document.documentElement.scrollHeight")
        clip_height = min(height, max(1, scroll_height))
        await page.screenshot(
            path=str(output_path),
            clip={"x": 0, "y": 0, "width": width, "height": clip_height},
        )
        await browser.close()


async def render_all(args: argparse.Namespace) -> int:
    html_dir = Path(args.html_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: list[tuple[int, Path]] = []
    for path in html_dir.glob("report_*.html"):
        qid = report_id_from_html(path)
        if qid is not None and qid >= args.start_id:
            items.append((qid, path))
    items.sort(key=lambda item: item[0])
    if args.end_id:
        items = [item for item in items if item[0] <= args.end_id]
    if args.limit > 0:
        items = items[: args.limit]

    if not items:
        raise FileNotFoundError(f"No report_*.html files found in {html_dir}")

    for idx, (qid, html_path) in enumerate(items, start=1):
        out_path = output_dir / f"question_{qid}.png"
        if out_path.exists() and not args.force:
            print(f"[{idx}/{len(items)}] skip existing {out_path}")
            continue
        print(f"[{idx}/{len(items)}] render {html_path} -> {out_path}")
        await render_one(
            html_path=html_path,
            output_path=out_path,
            width=args.width,
            height=args.height,
            extra_wait_ms=args.extra_wait_ms,
        )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render report_*.html files into question_*.png screenshots for MPQ.")
    parser.add_argument("--html-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-id", type=int, default=1)
    parser.add_argument("--end-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=1800)
    parser.add_argument("--extra-wait-ms", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    return asyncio.run(render_all(build_argparser().parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
