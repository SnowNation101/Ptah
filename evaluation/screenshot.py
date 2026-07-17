import asyncio
from playwright.async_api import async_playwright

async def screenshot_page(
    url: str,
    output_path: str = "screenshot.png",
    width: int = 1920,
    height: int = 1080,
    y_offset: int = 0,
    wait_until: str = "networkidle",   # "load" / "domcontentloaded" / "networkidle"
    extra_wait_ms: int = 500,          # Extra wait time for client-side rendering.
):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page = await context.new_page()

        # Open the page.
        resp = await page.goto(url, wait_until=wait_until, timeout=60_000)
        if resp is None:
            print("Warning: no response object (possibly a file:// url or navigation issue).")
        else:
            print("HTTP status:", resp.status)

        # Optionally wait for SPA rendering to settle.
        if extra_wait_ms > 0:
            await page.wait_for_timeout(extra_wait_ms)

        # Read page height to keep the clip within bounds.
        scroll_height = await page.evaluate("() => document.documentElement.scrollHeight")
        y = max(0, min(y_offset, max(0, scroll_height - 1)))
        h = min(height, max(1, scroll_height - y))

        clip = {"x": 0, "y": y, "width": width, "height": h}

        await page.screenshot(path=output_path, clip=clip)
        await browser.close()

        print(f"Saved: {output_path}  (clip={clip}, page_height={scroll_height})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a web page and take a clipped screenshot.")
    parser.add_argument("url", help="Target URL, e.g. https://example.com or file://path/to/report.html")
    parser.add_argument("--out", default="screenshot.png", help="Output image path")
    parser.add_argument("--width", type=int, default=1920, help="Viewport width")
    parser.add_argument("--height", type=int, default=1080, help="Clip height (and viewport height)")
    parser.add_argument("--y", type=int, default=0, help="Vertical offset (px) from top")
    parser.add_argument("--wait", default="networkidle", choices=["load", "domcontentloaded", "networkidle"])
    parser.add_argument("--extra-wait-ms", type=int, default=500, help="Extra wait time after navigation")
    args = parser.parse_args()

    asyncio.run(
        screenshot_page(
            url=args.url,
            output_path=args.out,
            width=args.width,
            height=args.height,
            y_offset=args.y,
            wait_until=args.wait,
            extra_wait_ms=args.extra_wait_ms,
        )
    )
