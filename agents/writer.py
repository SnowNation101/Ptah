from .base import BaseAgent
import os
import json
from utils.search import get_page_from_cache
from typing import List, Tuple, Optional, Dict
import re
from PIL.Image import Image
from utils.funcs import encode_image_to_base64
from utils.edit_qwen import SiliconFlowQwenEditClient

from dotenv import load_dotenv

load_dotenv()


class WriterAgent(BaseAgent):
    def __init__(self, client, tools, cache_dir: str = ".cache"):
        super().__init__(client, "prompts/writer.txt")
        self.tools = tools

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self._global_citation_map: Dict[
            Tuple[str, str], str
        ] = {}  # (title,url) -> "G#"
        self._next_global_citation_id: int = 1

        self.edit_client = SiliconFlowQwenEditClient()

    @staticmethod
    def _clean_model_text(text: str) -> str:
        return (text or "").split("</think>")[-1].strip()

    def create_section_prompt(
        self,
        question: str,
        overview: str,
        research_output: dict,
        web_images: Optional[list] = None,
    ) -> list:
        """
        Create the writing prompt for the Writer Agent in OpenAI multimodal format.
        Includes support for Tables and Local Image Base64 encoding.
        """

        section_id = research_output.get("section_id", "N/A")
        section_title = research_output.get("section_title", "Untitled Section")

        # ---- 1. Parse base data ----
        # Keep compatibility with legacy citation keys.
        references = (
            research_output.get("references", research_output.get("citations", []))
            or []
        )
        research_summary = research_output.get("research_summary", [])
        instructions_for_writer = research_output.get("instructions_for_writer", [])

        # Parse extracted tables.
        tables = research_output.get("tables", [])

        # ---- 2. Build source evidence cards ----
        source_blocks = []
        for ref in references:
            rid = ref.get("id", "")
            title = ref.get("title", "")
            url = ref.get("url", "")
            raw_content = get_page_from_cache(url) if url else ""

            block = f"### [{rid}] {title}\nURL: {url}\n\n[PAGE_CONTENT_BEGIN]\n{raw_content.strip() if raw_content else '(NO CONTENT)'}\n[PAGE_CONTENT_END]"
            source_blocks.append(block)

        sources_text = (
            "\n\n".join(source_blocks) if source_blocks else "(NO REFERENCES PROVIDED)"
        )

        # ---- 3. Serialize extracted data ----
        research_summary_json = json.dumps(
            research_summary, ensure_ascii=False, indent=2
        )
        instructions_text = (
            "\n".join(f"- {x}" for x in instructions_for_writer)
            if instructions_for_writer
            else "- (No instructions)"
        )
        references_json = json.dumps(references, ensure_ascii=False, indent=2)

        tables_text_list = []
        for t in tables:
            caption = t.get("table_caption", "Table")
            content = t.get("table", "")
            ref = t.get("ref", "")
            tables_text_list.append(f"**{caption}** (Source: [{ref}])\n{content}")
        tables_section_text = (
            "\n\n".join(tables_text_list)
            if tables_text_list
            else "(No tables provided)"
        )

        # ---- 4. Build the core text prompt ----
        text_prompt = f"""
# Global Writing Context
**Question:** {question}
**Overview:** {overview}

---
# Current Section: {section_title} (ID: {section_id})

# Evidence & Constraints
You MUST use ONLY the provided materials. Cite using [C#] inline.
If you use the images provided below, refer to them appropriately.

# Source Evidence Cards
{sources_text}

# Researcher Findings
{research_summary_json}

# Researcher Extracted Tables
Use these tables to support your writing or summarize them in text.
{tables_section_text}

# Writing Instructions
{instructions_text}

# Allowed Citations
{references_json}

# Output Requirements
- Output ONLY the prose for this section.
- Must start with the section title as a 2nd-level heading: "## {section_title}"
- Use 3rd-level headings ("###") for 2+ subsections.
- No bullet lists unless conceptually necessary.
- No meta commentary.
- Maintain an academic, neutral tone.
- Explicitly acknowledge evidence gaps if Researcher indicated missing evidence.
- If you need to present any table, DO NOT use Markdown syntax. You MUST use the code tool with matplotlib to render the table as an image.
- You MUST add at least 1 image; more images are encouraged.

Write the section now.
"""

        # ---- 5. Build the multimodal content list ----
        content = [{"type": "text", "text": text_prompt.strip()}]

        # ---- 6. Convert local images to base64 ----
        if web_images:
            content.append(
                {
                    "type": "text",
                    "text": "\n--- \n# Attached Visual Evidence (Web Images)\nBelow are real images found by the researcher. You can use them to support your writing:",
                }
            )

            for img in web_images:
                local_path = img.get("local_path")
                base64_image = (
                    encode_image_to_base64(local_path) if local_path else None
                )

                if base64_image:
                    img_description = f"Image Context: {img.get('alt_text', '')}. Analysis: {img.get('vlm_reason', '')}"
                    content.append({"type": "text", "text": img_description})

                    content.append(
                        {"type": "image_url", "image_url": {"url": base64_image}}
                    )
                else:
                    print(f"[Writer] Warning: Failed to load image from {local_path}")

        return content

    def write_section(self, section_prompt: list, section_id: int) -> str:
        """
        Generate RAW section text only (may include tool placeholders).
        No tool execution happens here.
        """
        print(f"[Writer] Writing (raw) for section {section_id}...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": section_prompt},
        ]

        response, reasoning = self.client.chat_completion(
            messages,
            max_tokens=8192,
            temperature=0.7,
            top_p=0.8,
            enable_thinking=False,
            repitition_penalty=1.0,
            presence_penalty=1.5,
        )

        response = self._clean_model_text(response)
        if not response:
            response = self._clean_model_text(reasoning)
        if not response:
            raise ValueError(f"Writer produced empty raw text for section {section_id}.")

        out_path = os.path.join(
            self.cache_dir, f"writer_output_section_{section_id}.txt"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response)

        return response

    def refine_section(self, raw_text: str, section_id: int) -> str:
        print(f"[Writer] Refining section {section_id}...")

        prompt = f"""You are an expert academic editor and formatting validator for multimodal research reports.

Your task is to revise and polish a section generated by a writing model.

You must preserve the meaning and structure while correcting formatting, citations, image tags, and code blocks.

IMPORTANT:
Think silently and output ONLY the final revised section.
Do NOT output explanations, analysis, or comments.

---

# Critical Fix Rules

You MUST apply the following corrections.

## 1. REMOVE ALL savefig CALLS (Mandatory)

Inside ANY `<imgen>` tag with `"source": "code"`:

DELETE every line containing any of the following:

plt.savefig  
savefig(  
fig.savefig  

These calls must NEVER appear in the final output.

Reason:
The rendering system automatically captures matplotlib output, so saving figures to files is unnecessary.

After removal, ensure the code still produces a visible matplotlib figure (e.g., plotting commands or `plt.show()`).

---

## 2. Image Tag Validation

Every image must use exactly this structure:

<imgen>{{"source": "<source_type>", "description": "<short descriptive title>", "params": {{<parameter_dict>}}}}</imgen>

Rules:

- JSON inside `<imgen>` must be valid.
- Use double quotes `"` only.
- `<imgen>` tags must NOT be wrapped in Markdown code blocks.
- Ensure both `<imgen>` and `</imgen>` exist.
- Do NOT remove images unless completely broken.
- Fix malformed JSON when necessary.

Required parameters by source:

"code" → `"params": {{"code": "python matplotlib code"}}`  
"search" → `"params": {{"query": "..."}}`  
"diffusion" → `"params": {{"prompt": "..."}}`  
"ref" → `"params": {{"img_index": <int>}}`  
"edit" → `"params": {{"img_index": <int>, "prompt": "..."}}`

---

## 3. Citation Normalization

All citations must follow exactly these formats:

[C1]  
[C2]  
[C1, C2]  
[C1, C3, C5]

Rules:

- Always include letter **C**
- Use comma + space between citations
- NEVER use `[1]`, `[C1,C2]`, `(C1)`, `[C 1]`
- Do NOT invent citations
- Only normalize formatting

---

## 4. Markdown Structure

The section MUST begin with a level-2 heading:

## Section Title

Subsections must use level-3 headings:

### Subsection Title

Do not change the section title unless formatting is incorrect.

---

## 5. Markdown Tables → Image Conversion

If Markdown tables appear (using `|` columns):

Convert the table into an image using:

`<imgen>` with `"source": "code"` and matplotlib.

Rules:

- Use the table data already present.
- Do NOT introduce new data.
- The generated code MUST NOT contain `savefig`.

If BOTH exist:
- a Markdown table
- an equivalent `<imgen>` table image

Then REMOVE the Markdown table and keep only the `<imgen>` image.

---

## 6. Language Polishing

Improve:

- clarity
- grammar
- academic tone
- readability
- logical flow

Constraints:

- Preserve meaning
- Do NOT remove images
- Do NOT remove citations
- Do NOT add new claims
- Do NOT introduce new references

---

# Output Requirements

Return ONLY the revised section.

The output must:

- Keep valid `<imgen>` tags
- Contain NO `savefig` calls
- Use normalized citations
- Maintain proper Markdown headings
- Contain no explanations

---

# Section Draft

{raw_text}
        """

        messages = [{"role": "user", "content": prompt}]

        response, reasoning = self.client.chat_completion(
            messages,
            max_tokens=8192,
            temperature=0.2,
            top_p=0.8,
            enable_thinking=False,
            repitition_penalty=1.0,
            presence_penalty=1.5,
        )

        response = self._clean_model_text(response)
        if not response:
            response = self._clean_model_text(reasoning)
        if not response:
            print(f"[Writer] Empty refined section {section_id}; falling back to raw text.")
            response = raw_text

        out_path = os.path.join(
            self.cache_dir, f"writer_refine_section_{section_id}.txt"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response)

        return response

    def create_conclusion_prompt(
        self,
        question: str,
        overview: str,
        full_body_text: str,
        conclusion_title: str = "Conclusion",
    ) -> str:
        """
        Prompt for conclusion writing. Explicitly forbids tool placeholders and new citations.
        """
        prompt = f"""
# Task: Write the Conclusion Section (No Tools)

You are writing the concluding chapter for an academic, neutral report.

**User's Original Question**
{question}

**Planner's Overall Report Overview**
{overview}

---

# Full Report Body (All Previous Sections)

Below is the complete report body (all sections before the conclusion). Read it carefully and write a conclusion that:
1) Synthesizes the central findings across sections,
2) States key takeaways with appropriate uncertainty,
3) Notes limitations and evidence gaps already mentioned,
4) Suggests concrete future directions (optional but preferred).

[REPORT_BODY_BEGIN]
{full_body_text}
[REPORT_BODY_END]

---

# Constraints (CRITICAL)

- You MUST NOT call or request any tools.
- You MUST NOT include any tool placeholders (e.g., do not output tags like <imgen>...</imgen>).
- You MUST NOT introduce new citations or new citation IDs.
  (You may reference prior claims at a high level, but do not add new [C#] markers.)
- Keep style consistent with the prior sections: academic, neutral, and concise.

---

# Output Requirements

- Output ONLY the conclusion section.
- Must start with a 2nd-level heading: "## {conclusion_title}"
- Prefer 2--4 short paragraphs.
- No meta commentary.

Write the conclusion now.
"""
        return prompt.strip()

    def write_conclusion(
        self,
        question: str,
        overview: str,
        prior_sections_text: str,
        conclusion_title: str = "Conclusion",
    ) -> str:
        """
        Write a conclusion based on all prior sections.
        No tool execution, no tool placeholders.
        """
        print("[Writer] Writing conclusion...")

        prompt = self.create_conclusion_prompt(
            question=question,
            overview=overview,
            full_body_text=prior_sections_text,
            conclusion_title=conclusion_title,
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        response, reasoning = self.client.chat_completion(
            messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=0.8,
            enable_thinking=False,
            repitition_penalty=1.0,
            presence_penalty=1.5,
        )
        response = self._clean_model_text(response)
        if not response:
            response = self._clean_model_text(reasoning)
        if not response:
            raise ValueError("Writer produced empty conclusion.")

        out_path = os.path.join(self.cache_dir, "writer_output_conclusion.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response)

        return response

    @staticmethod
    def make_interleaved_prompt(text: str, images: list[Image]) -> list:
        """
        Create a prompt that interleaves text and images for the refinement step.
        This is a placeholder implementation. The actual prompt design would depend on the specific refinement tasks.
        """
        encoded_images = [encode_image_to_base64(img) for img in images]
        text_segments = text.split("<image>")

        interleaved_content = []

        for i in range(len(text_segments)):
            if text_segments[i].strip():
                interleaved_content.append({"type": "text", "text": text_segments[i]})

            if i < len(encoded_images):
                interleaved_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_images[i],
                        },
                    }
                )

        return interleaved_content

    def refine_images(self, text: str, images: List[Image]) -> Tuple[str, List[Image]]:
        """
        Decide an action for each image in an interleaved text+image report.

        Input:
        - `text`: the original report text containing <image> placeholders
        - `images`: the images aligned with the <image> placeholders in `text`

        Model output protocol:
        - EXACTLY one action per input image, in the same order:
            1. <keep>
            2. <delete>
            3. <edit>...</edit>

        The model is NOT allowed to rewrite, polish, or otherwise modify the text.
        It only decides how each image should be handled based on the full text+image context.

        Return:
        - refined_text: original text with some <image> placeholders removed if deleted
        - refined_images: final image list aligned with the remaining <image> placeholders
        """

        num_images = len(images)

        prompt = f"""
You are an image-selection and image-editing planner for an interleaved report.

You will receive the FULL report as interleaved text and images.
The text provides the context for understanding each image.

Your task is to decide how to handle each image based on:
- how much information the image provides
- how relevant it is to the surrounding text
- whether it improves the reader's understanding

There are EXACTLY {num_images} input images in this report.

Your job:
- DO NOT rewrite, polish, summarize, or modify any text.
- DO NOT output the report text.
- ONLY output EXACTLY {num_images} image actions, one for each input image.
- The actions must be in the SAME ORDER as the input images appear in the report.

Important principle:
If an image does NOT provide meaningful information beyond the text, it SHOULD be deleted.

Prefer deleting images that are:
- redundant with the surrounding text
- decorative only
- screenshots without useful details
- blurry, low-quality, or unclear
- generic illustrations
- repeated or very similar to another image
- not referenced or explained in the text
- containing very little visual information

Allowed actions:

1. <keep>
Use this ONLY if the image clearly provides useful information that improves understanding.

2. <delete>
Use this if the image:
- adds little or no information
- is redundant with the text
- is decorative or generic
- is unclear or low-quality
- does not meaningfully help the reader

If you are unsure whether the image is useful, prefer <delete>.

3. <edit>YOUR_EDIT_INSTRUCTION</edit>
Use this if the image is useful but needs improvement to be more informative.

The edit instruction must be concrete, visual, and directly executable.

Good edit examples:
<edit>add a red arrow pointing to the ball near the center</edit>
<edit>crop the figure to focus on the chart area</edit>
<edit>increase contrast of the text labels</edit>
<edit>highlight the relevant object with a clear bounding box</edit>

Bad edit examples:
<edit>make it better</edit>
<edit>improve the image quality</edit>
<edit>make it more professional</edit>

Important rules:
- Output EXACTLY {num_images} actions.
- Output ONLY the actions, nothing else.
- Do not output explanations.
- Do not output bullet points.
- Do not output numbering.
- Do not output markdown fences.
- Do not output any text besides the action tags.
- Keep the output order exactly aligned with the input image order.
- Each line must contain exactly one action.

Final output format example:
<keep>
<edit>add a red arrow pointing to the ball</edit>
<delete>
""".strip()

        user_content = [
            {"type": "text", "text": prompt}
        ] + self.make_interleaved_prompt(text, images)

        messages = [
            {"role": "user", "content": user_content},
        ]

        response, _ = self.client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.2,
            top_p=0.8,
            enable_thinking=False,
            repitition_penalty=1.0,
            presence_penalty=1.5,
        )
        response = self._clean_model_text(response)

        with open(
            os.path.join(self.cache_dir, "writer_refine_all_img_txt.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(response)

        print(f"[Writer] Raw image refinement response:\n{response}\n")

        # ------------------------------------------------------------
        # Parse model output: only allow <keep>, <delete>, <edit>...</edit>
        # ------------------------------------------------------------
        action_pattern = re.compile(
            r"(<keep>|<delete>|<edit>.*?</edit>)",
            flags=re.DOTALL,
        )

        actions = [m.group(0).strip() for m in action_pattern.finditer(response)]

        if len(actions) != num_images:
            raise ValueError(
                f"Model output image action count mismatch: "
                f"expected {num_images}, got {len(actions)}.\n\nResponse:\n{response}"
            )

        # Extra strictness: reject any non-whitespace content outside valid actions
        leftover = action_pattern.sub("", response).strip()
        if leftover:
            raise ValueError(
                f"Model output contains invalid extra content outside action tags:\n{leftover}\n\nFull response:\n{response}"
            )

        # ------------------------------------------------------------
        # Apply actions to images, while preserving original text
        # ------------------------------------------------------------
        text_parts = re.split(r"(<image>)", text)

        num_placeholders = sum(1 for part in text_parts if part == "<image>")
        if num_placeholders != num_images:
            raise ValueError(
                f"Input text/image mismatch: text contains {num_placeholders} <image> placeholders, "
                f"but images list has {num_images} images."
            )

        refined_text_parts: List[str] = []
        refined_images: List[Image] = []
        image_idx = 0

        for part in text_parts:
            if part != "<image>":
                refined_text_parts.append(part)
                continue

            if image_idx >= num_images:
                raise ValueError(
                    f"More <image> placeholders than images. image_idx={image_idx}, total={num_images}"
                )

            action = actions[image_idx]
            current_image = images[image_idx]
            image_idx += 1

            if action == "<keep>":
                refined_text_parts.append("<image>")
                refined_images.append(current_image)

            elif action == "<delete>":
                # remove this image placeholder entirely
                pass

            else:
                m = re.fullmatch(r"<edit>(.*?)</edit>", action, flags=re.DOTALL)
                if not m:
                    raise ValueError(f"Malformed edit action: {action}")

                edit_prompt = m.group(1).strip()
                if not edit_prompt:
                    raise ValueError(f"Empty edit instruction for image #{image_idx}")

                edited_image = self.edit_client.edit_image(
                    image=current_image,
                    prompt=edit_prompt,
                )

                refined_text_parts.append("<image>")
                refined_images.append(edited_image)

        if image_idx != num_images:
            raise ValueError(
                f"Not all images were consumed: consumed={image_idx}, total={num_images}"
            )

        refined_text = "".join(refined_text_parts)
        refined_text = re.sub(r"\n{3,}", "\n\n", refined_text).strip()

        return refined_text, refined_images

    def refine_all_text(
        self, question: str, text: str, images: List[Image]
    ) -> Tuple[str, List[Image]]:
        """
        Refine only the text of an interleaved text+image report.

        Input:
        - question: the original user question / task that the report is answering
        - text: report text containing <image> placeholders
        - images: image list aligned with the <image> placeholders in text

        Behavior:
        - The model sees the report as interleaved text segments and images
        (because make_interleaved_prompt replaces <image> placeholders with actual image inputs)
        - The model may rewrite ONLY the text
        - The model must output the FULL report text, using <image> as placeholders
        at the original image positions
        - If the output contains the wrong number of <image> placeholders, retry up to 5 times

        Return:
        - refined_text: polished text with the same <image> placeholders
        - images: returned unchanged
        """

        print("[Writer] Refining all text...")

        num_images = len(images)
        num_placeholders = text.count("<image>")
        max_retries = 5

        if num_placeholders != num_images:
            raise ValueError(
                f"Input text/image mismatch: text contains {num_placeholders} <image> placeholders, "
                f"but images list has {num_images} images."
            )

        prompt = f"""
    You are a professional editor for an interleaved image-text report.

    You will receive:
    1. The original user question that the report is intended to answer.
    2. The FULL report as alternating text segments and images.

    Important:
    - In the input, you see actual images inserted between text segments.
    - You do NOT see the literal token <image> in the input.

    Original user question:
    {question}

    Your task:
    Refine ONLY the natural language text of the report so that the report answers the user's question more directly and more effectively.

    There are EXACTLY {num_images} images in this report.

    Output format requirements:
    - You must output the FULL refined report as plain text.
    - In your output, use the special token <image> to mark each image position.
    - Your output must contain EXACTLY {num_images} occurrences of <image>.
    - The <image> tokens must appear in the SAME order as the input images.
    - Each <image> must stay in the same position relative to the surrounding text content.
    - Do not add, remove, merge, split, rename, or move any <image> placeholder.
    - Do not output anything except the final refined report text.

    Editing requirements:
    - Refine ONLY the natural language text.
    - Do NOT modify any image.
    - Do NOT describe image editing instructions.
    - Do NOT add new facts not supported by the text or images.
    - Do NOT remove important information.
    - Do NOT change the overall meaning of the report.
    - Make the writing more rigorous, precise, formal, and well-structured.
    - Use a more formal, polished, and professional written style.
    - Reduce redundancy, repetition, and circular phrasing.
    - Avoid saying the same point repeatedly in slightly different ways.
    - Improve logical flow and transitions between paragraphs.
    - Make the introduction more directly aligned with the user's question.
    - Make the conclusion more directly answer the user's question and summarize the key findings clearly.
    - Especially in the introduction and conclusion, ensure the report stays on-topic and explicitly addresses the user's question.
    - Preserve useful details, but remove unnecessary verbosity.
    - Prefer concise and information-dense writing over repetitive elaboration.

    Your answer is invalid if:
    - it does not contain exactly {num_images} <image> tokens
    - it contains any explanation, note, bullet points, or markdown fences
    - it changes the order of image positions

    Example:
    If the input structure is:
    [text segment 1] [image 1] [text segment 2] [image 2] [text segment 3]

    Then your output structure must be:
    refined text segment 1<image>refined text segment 2<image>refined text segment 3
        """.strip()

        user_content = [
            {"type": "text", "text": prompt}
        ] + self.make_interleaved_prompt(text, images)

        messages = [
            {"role": "user", "content": user_content},
        ]

        last_response = None
        last_error = None

        for attempt in range(1, max_retries + 1):
            response, _ = self.client.chat_completion(
                messages,
                max_tokens=8192,
                temperature=0.3,
                top_p=0.8,
                enable_thinking=False,
                repitition_penalty=1.0,
                presence_penalty=1.1,
            )

            response = self._clean_model_text(response)
            last_response = response

            if hasattr(self, "cache_dir") and self.cache_dir:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(
                    os.path.join(
                        self.cache_dir, f"writer_refine_text_only_attempt_{attempt}.txt"
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(response)

            # ------------------------------------------------------------
            # Validate output
            # ------------------------------------------------------------
            output_num_placeholders = response.count("<image>")
            if output_num_placeholders != num_images:
                last_error = (
                    f"Attempt {attempt}: Model output placeholder count mismatch: "
                    f"expected {num_images}, got {output_num_placeholders}."
                )
                print(f"[Writer] {last_error} Retrying...")
                continue

            # Split input/output by <image> and compare segment counts
            input_parts = re.split(r"(<image>)", text)
            output_parts = re.split(r"(<image>)", response)

            input_placeholder_count = sum(1 for p in input_parts if p == "<image>")
            output_placeholder_count = sum(1 for p in output_parts if p == "<image>")

            if input_placeholder_count != output_placeholder_count:
                last_error = (
                    f"Attempt {attempt}: Placeholder count mismatch after split: "
                    f"input={input_placeholder_count}, output={output_placeholder_count}"
                )
                print(f"[Writer] {last_error} Retrying...")
                continue

            # Ensure placeholder slots are preserved syntactically
            corrupted = False
            for i, part in enumerate(output_parts):
                if i % 2 == 1 and part != "<image>":
                    last_error = f"Attempt {attempt}: Output placeholder corrupted at segment {i}: {part}"
                    print(f"[Writer] {last_error} Retrying...")
                    corrupted = True
                    break
            if corrupted:
                continue

            # Optional stricter structural check:
            # number of text segments should remain identical
            if len(input_parts) != len(output_parts):
                last_error = (
                    f"Attempt {attempt}: Output structure mismatch: "
                    f"input has {len(input_parts)} parts, output has {len(output_parts)} parts."
                )
                print(f"[Writer] {last_error} Retrying...")
                continue

            refined_text = response.strip()
            with open(
                os.path.join(self.cache_dir, "writer_final.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(refined_text)

            print(f"[Writer] Refining all text succeeded on attempt {attempt}.")
            return refined_text, images

        raise ValueError(
            f"Failed to generate valid refined text after {max_retries} attempts.\n"
            f"Last error: {last_error}\n\n"
            f"Last response:\n{last_response}"
        )

    def gen_images(
        self, input_text: str, input_images: list[Image] | None = None
    ) -> tuple[str, List[Image]]:
        """
        Run one global tool-execution pass to convert placeholders into images.
        Returns (processed_text, image_list).
        """
        print("[Writer] Executing interleaved tool calls...")

        processed_text, image_list, _ = self.tools.generate(input_text, input_images)

        return processed_text, image_list

    def ingest_references(self, references: List[dict]) -> Dict[str, str]:
        """
        Update global citation mapping using section-level local references.
        Returns local_id -> global_id mapping (e.g., "C1" -> "G3").
        """
        local_to_global: Dict[str, str] = {}

        if not references:
            return local_to_global

        for ref in references:
            local_id = ref.get("id", "")
            title = ref.get("title", "")
            url = ref.get("url", "")

            if not local_id or not title or not url:
                # Skip malformed reference entries
                continue

            key = (title, url)
            if key not in self._global_citation_map:
                gid = f"G{self._next_global_citation_id}"
                self._global_citation_map[key] = gid
                self._next_global_citation_id += 1
            else:
                gid = self._global_citation_map[key]

            local_to_global[local_id] = gid

        return local_to_global

    @staticmethod
    def replace_local_citations(text: str, mapping: Dict[str, str]) -> str:
        """
        Replace bracketed local citations like [C1], [C2, C3] into global ones [Gx].
        Only replaces tokens matching pattern C\\d+ within [...] groups.
        """
        if not text or not mapping:
            return text

        def repl(match):
            inside = match.group(1)

            def token_repl(m):
                local_id = m.group(0)
                return mapping.get(local_id, local_id)

            inside2 = re.sub(r"\bC\d+\b", token_repl, inside)
            return f"[{inside2}]"

        return re.sub(r"\[([^\]]*\bC\d+\b[^\]]*)\]", repl, text)

    def get_final_citations(self) -> List[dict]:
        """
        Return final global citations list sorted by numeric id.
        """
        final_citations = [
            {"id": gid, "title": title, "url": url}
            for (title, url), gid in self._global_citation_map.items()
        ]
        final_citations.sort(key=lambda x: int(x["id"][1:]))
        return final_citations

    @staticmethod
    def remove_citations(text: str) -> str:
        """
        Remove citation tokens like [C1], [C1, C2], [G3] from text.
        It removes the entire bracket group if it only contains citation tokens.
        """
        # Remove citation groups like [C1], [C1, C2], [G3]
        text = re.sub(r"\[(?:\s*[CG]\d+\s*(?:,\s*[CG]\d+\s*)*)\]", "", text)

        return text
