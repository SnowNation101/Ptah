import os
import re
import html
from typing import Any, Dict, List, Optional, Tuple


class ReportPreviewBuilder:
    """
    Build a local HTML preview for a full report with interleaved images.

    New assumptions (single-report mode):
    - report_text is a single string that may contain one or more literal "<image>" placeholders.
    - report_imgs is a flat list of images (PIL.Image.Image), aligned with the placeholders in order.
    - final_citations is a unified list like:
      [{"id":"G1","title":"...","url":"..."}, ...]
    """

    def __init__(self, output_dir: str, assets_subdir: str = "report_assets"):
        self.output_dir = output_dir
        self.assets_dir = os.path.join(output_dir, assets_subdir)
        self.assets_subdir = assets_subdir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)


    def build(
        self,
        report_title: str,
        report_text: str,
        report_imgs: List[Any],
        final_citations: List[Dict[str, str]],
        out_filename: str = "report.html"
    ) -> str:
        """
        Create an HTML file in output_dir and return its absolute path.

        Args:
            report_title: Title shown at top.
            report_text: Entire report text (single string) that may include "<image>" placeholders.
            report_imgs: Flat list of images corresponding to placeholders.
            final_citations: Unified citation list [{"id","title","url"}, ...]
            out_filename: Output HTML filename.

        Returns:
            Absolute path to the generated HTML file.
        """
        body_parts: List[str] = [f"<h1>{html.escape(report_title)}</h1>"]

        frag, _ = self._render_text_with_images(
            text=report_text or "",
            imgs=report_imgs or [],
            img_start_idx=1
        )
        body_parts.append(f'<section class="section">{frag}</section>')
        body_parts.append('<hr class="sep"/>')

        body_parts.append(self._render_references(final_citations or []))

        html_doc = self._wrap_html(report_title, "\n".join(body_parts))

        out_path = os.path.join(self.output_dir, out_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_doc)

        return os.path.abspath(out_path)


    def _render_text_with_images(
        self,
        text: str,
        imgs: List[Any],
        img_start_idx: int
    ) -> Tuple[str, int]:
        """
        Split text on "<image>" and insert images between text parts.
        Returns (html_fragment, next_global_img_idx).
        """
        parts = (text or "").split("<image>")
        out: List[str] = []

        img_idx = img_start_idx
        used = 0

        for i, chunk in enumerate(parts):
            chunk = chunk.strip("\n")
            if chunk:
                out.append(self._text_to_html(chunk))

            # If not last chunk, there's an image placeholder position
            if i < len(parts) - 1:
                if used < len(imgs):
                    img_obj = imgs[used]
                    src, alt_or_reason = self._normalize_and_copy_image(img_obj, img_idx)
                    if src:
                        out.append(
                            f'<figure class="img">'
                            f'<img src="{html.escape(src)}" alt="{html.escape(alt_or_reason)}"/>'
                            f'</figure>'
                        )
                    else:
                        out.append(
                            f'<div class="img-missing">[Image missing: {html.escape(alt_or_reason)}]</div>'
                        )
                else:
                    out.append(
                        '<div class="img-missing">[Image missing: no image returned for this &lt;image&gt; placeholder]</div>'
                    )

                used += 1
                img_idx += 1

        return "\n".join(out), img_idx

    def _normalize_and_copy_image(self, img_obj: Any, idx: int) -> Tuple[Optional[str], str]:
        """
        Only supports PIL images (Image.Image / PngImageFile, etc.).
        Saves each image as PNG into assets_dir and returns a relative src.
        """
        from PIL import Image

        if not isinstance(img_obj, Image.Image):
            return (None, f"Expected PIL.Image.Image, got: {type(img_obj)}")

        # Ensure image data is loaded (avoid lazy-loading edge cases)
        try:
            img_obj.load()
        except Exception:
            pass

        # Normalize to a browser-friendly mode
        if img_obj.mode not in ("RGB", "RGBA"):
            try:
                img_obj = img_obj.convert("RGBA")
            except Exception:
                img_obj = img_obj.convert("RGB")

        dst_name = f"img_{idx:04d}.png"
        dst_path = os.path.join(self.assets_dir, dst_name)

        # Always write PNG for consistent preview behavior
        img_obj.save(dst_path, format="PNG")

        return (os.path.join(self.assets_subdir, dst_name), f"image_{idx}")


    def _text_to_html(self, text: str) -> str:
        """
        Minimal markdown-ish renderer:
        - #, ##, ### headings
        - paragraphs separated by blank lines
        - inline **bold**
        - keep inline [G#] citations unchanged
        """
        lines = (text or "").splitlines()
        html_lines: List[str] = []
        buf: List[str] = []

        def flush_paragraph():
            nonlocal buf
            if not buf:
                return
            paragraph = "\n".join(buf).strip()
            paragraph = html.escape(paragraph)
            paragraph = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", paragraph)
            paragraph = paragraph.replace("\n", "<br/>")
            html_lines.append(f"<p>{paragraph}</p>")
            buf = []

        for line in lines:
            raw = line.rstrip()
            if not raw.strip():
                flush_paragraph()
                continue

            if raw.startswith("### "):
                flush_paragraph()
                html_lines.append(f"<h3>{html.escape(raw[4:].strip())}</h3>")
                continue
            if raw.startswith("## "):
                flush_paragraph()
                html_lines.append(f"<h2>{html.escape(raw[3:].strip())}</h2>")
                continue
            if raw.startswith("# "):
                flush_paragraph()
                html_lines.append(f"<h1>{html.escape(raw[2:].strip())}</h1>")
                continue

            buf.append(raw)

        flush_paragraph()
        return "\n".join(html_lines)


    def _render_references(self, citations: List[Dict[str, str]]) -> str:
        items: List[str] = []
        for c in citations:
            cid = html.escape(c.get("id", ""))
            title = html.escape(c.get("title", ""))
            url = html.escape(c.get("url", ""))
            if url:
                items.append(
                    f'<li><span class="cid">[{cid}]</span> '
                    f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a></li>'
                )
            else:
                items.append(
                    f'<li><span class="cid">[{cid}]</span> {title}</li>'
                )

        return "<h2>References</h2><ul class='refs'>\n" + "\n".join(items) + "\n</ul>"


    def _wrap_html(self, title: str, body_html: str) -> str:
        css = """
        body {
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, "Noto Sans";
          margin: 28px;
          max-width: 980px;
        }
        .section { margin: 18px 0; }
        .sep { border: 0; border-top: 1px solid #ddd; margin: 28px 0; }
        p { line-height: 1.6; font-size: 16px; }
        figure.img { margin: 18px 0; }
        figure.img img {
          max-width: 100%;
          height: auto;
          border: 1px solid #eee;
          border-radius: 10px;
        }
        .img-missing {
          color: #a00;
          background: #fff3f3;
          padding: 10px;
          border: 1px solid #f1c0c0;
          border-radius: 8px;
          margin: 12px 0;
        }
        .refs { line-height: 1.5; }
        .cid { font-weight: 700; margin-right: 6px; }
        """

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>{css}</style>
</head>
<body>
{body_html}
</body>
</html>
""".strip()
