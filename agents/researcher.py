from .base import BaseAgent
from utils.search import search_with_vision
import re
import json
import os
from typing import List, Tuple, Optional, Any

BEGIN_SEARCH_TOKEN = "<search>"
END_SEARCH_TOKEN = "</search>"
BEGIN_SEARCH_RESULT_TOKEN = "<search_result>"
END_SEARCH_RESULT_TOKEN = "</search_result>"
BEGIN_OUTPUT_TOKEN = "<output>"
END_OUTPUT_TOKEN = "</output>"

STRING_ARRAY_KEYS = {
    "research_summary",
    "instructions_for_writer",
    "key_points",
    "needs_further_research",
    "suggested_search_queries",
    "visual_requirements",
    "strengths",
    "weakness",
    "weaknesses",
    "suggestions",
}


def loads_json_lenient(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = quote_bare_string_array_items(text)
        if repaired == text:
            raise
        return json.loads(repaired)


def quote_bare_string_array_items(text: str) -> str:
    """Repair the common Qwen JSON error: bare strings inside known string arrays."""
    lines = text.splitlines()
    repaired = []
    stack = []

    array_open_re = re.compile(r'^\s*"([^"]+)"\s*:\s*\[\s*$')
    bare_literal_re = re.compile(r"^(true|false|null|-?\d+(\.\d+)?)\s*,?$")

    for line in lines:
        stripped = line.strip()
        top = stack[-1] if stack else None

        if top and top.get("quote_bare") and stripped:
            trailing_comma = stripped.endswith(",")
            item = stripped[:-1].rstrip() if trailing_comma else stripped
            if (
                item
                and item not in {"]", "]}"}
                and not item.startswith(('"', "{", "[", "]"))
                and not bare_literal_re.match(stripped)
            ):
                indent = line[: len(line) - len(line.lstrip())]
                escaped = item.replace("\\", "\\\\").replace('"', '\\"')
                line = f'{indent}"{escaped}"{"," if trailing_comma else ""}'
                stripped = line.strip()

        repaired.append(line)

        match = array_open_re.match(stripped)
        if match:
            stack.append({"quote_bare": match.group(1) in STRING_ARRAY_KEYS})

        if stripped in {"]", "],"} and stack:
            stack.pop()

    return "\n".join(repaired)

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for citation-ready evidence for the assigned report section. "
            "Returns text evidence cards, selected visual evidence, and source URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused plain-text web search query for the assigned section."
                }
            },
            "required": ["query"]
        }
    }
}

class ResearcherAgent(BaseAgent):
    """
    ResearchAgent performs multi-step, planner-style research with:
    - iterative search + reasoning
    - rule-based grounding checks (URL must appear in tool trace)
    - reviewer-based rejection / revision loops
    - full trace grounding for reviewer evaluation
    """

    # URL regex
    _URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")

    def __init__(self, client, reviewer_client, cache_dir: str = ".cache"):
        super().__init__(client, "prompts/researcher.txt")

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.reviewer_client = reviewer_client
        with open("prompts/research_reviewer.txt", "r", encoding="utf-8") as f:
            self.reviewer_system_prompt = f.read()

        self.ground_urls = []
        self.images_from_webpage = []

    def create_section_prompt(
        self,
        question: str,
        overview: str,
        section_outline: dict
    ) -> str:
        """
        Create the user prompt for the Step 2 Researcher Agent.
        """
        key_points_str = "\n".join([f"- {kp}" for kp in section_outline.get("key_points", [])])
        needs_research_str = "\n".join([f"- {nr}" for nr in section_outline.get("needs_further_research", [])])
        queries_str = "\n".join([f"- {sq}" for sq in section_outline.get("suggested_search_queries", [])])

        sec_id = section_outline.get("id", "N/A")
        title = section_outline.get("title", "Untitled Section")
        summary = section_outline.get("summary", "")

        prompt = (
            "# Project Context\n"
            f"**User's Original Question:** {question}\n"
            f"**Report Overview:** {overview}\n"
            "\n-------------\n\n"
            "# Target Section Assignment\n"
            f"**Section ID:** {sec_id}\n"
            f"**Section Title:** {title}\n"
            f"**Section Summary:** {summary}\n\n"
            "# Research Instructions\n\n"
            "## 1. Verify & Expand (Key Points)\n"
            "The Planner has identified the following core concepts. Please verify their accuracy and provide detailed explanations, examples, or data to support them:\n"
            f"{key_points_str}\n\n"
            "## 2. Deep Dive (Needs Further Research)\n"
            "**CRITICAL:** The Planner identified gaps in our current knowledge. Your primary mission is to find answers to these specific questions:\n"
            f"{needs_research_str}\n\n"
            "## 3. Search Strategy\n"
            "Use the following queries as a starting point, but feel free to generate advanced queries based on your findings:\n"
            f"{queries_str}\n"
        ).strip()

        
        return prompt.strip()

    def run(
        self,
        section_prompt: str,
        section_id: int,
        section_data: dict,
        max_agent_steps: int = 10,
        max_review_rounds: int = 5,
        max_full_reruns: int = 1,   # Maximum full pipeline reruns after total parse failure.
    ):
        print(f"\n[Researcher] Start researching for section {section_id}")

        # Initial message template used to reset full reruns.
        init_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": section_prompt},
        ]

        final_result = None

        # Outer loop for full pipeline reruns.
        for full_run_idx in range(max_full_reruns + 1):
            print(f"\n[Researcher] === FULL RUN {full_run_idx + 1}/{max_full_reruns + 1} ===")

            # Reset grounding state, images, trace, and messages for each full run.
            self.ground_urls = []
            self.images_from_webpage = []
            all_rounds_trace = ""
            messages = list(init_messages)

            last_json_any: Optional[dict] = None
            last_json_grounded: Optional[dict] = None
            final_result = None

            for review_round in range(max_review_rounds):
                print(f"\n[Researcher] Start round {review_round + 1} for section {section_id}")

                # 1) Run one full research reasoning loop
                answer, current_round_trace, messages = self._run_research_once(
                    messages=messages,
                    section_data=section_data,
                    max_agent_steps=max_agent_steps,
                )

                # --- Update Global Trace ---
                all_rounds_trace += f"\n\n=== ROUND {review_round + 1} EXECUTION ===\n{current_round_trace}"

                # Include full_run_idx to avoid overwriting retry traces.
                self._save_text(
                    os.path.join(self.cache_dir, f"research_section_{section_id}_fullrun_{full_run_idx}_round_{review_round}.txt"),
                    all_rounds_trace
                )

                # 2) Parse final JSON output
                parsed = self._save_final_output(answer, section_id)
                if not isinstance(parsed, dict):
                    error_msg = (
                        "Reviewer rejected your output: invalid or missing JSON inside <output>...</output>. "
                        "Fix format only. Output MUST be valid JSON."
                    )
                    all_rounds_trace += f"\n\n=== ROUND {review_round + 1} SYSTEM FEEDBACK ===\n{error_msg}"
                    messages.append({"role": "user", "content": error_msg})
                    print(f"[Reviewer] Failed to parse JSON in round {review_round}. Retrying...")
                    continue

                last_json_any = parsed

                # 2.5) RULE-BASED GROUDING CHECK
                rule_review = self._rule_based_grounding_audit(json_output=parsed)

                if rule_review is not None and rule_review.get("decision") == "reject":
                    self._save_json(
                        os.path.join(self.cache_dir, f"research_reviewer_section_{section_id}_fullrun_{full_run_idx}_round_{review_round}.json"),
                        rule_review
                    )
                    feedback_msg = self._format_rule_reject_feedback(rule_review)
                    all_rounds_trace += f"\n\n=== ROUND {review_round + 1} AUTOMATED AUDIT FEEDBACK ===\n{feedback_msg}"
                    messages.append({"role": "user", "content": feedback_msg})
                    print(f"[Reviewer] Citation grounding check rejected round {review_round}. Retrying...")
                    continue

                # Passed rule-based audit
                last_json_grounded = parsed
                print("[Reviewer] Passed rule-based check.")

                # 3) Run reviewer on FULL HISTORICAL TRACE
                review_result = self._run_reviewer(
                    section_prompt=section_prompt,
                    full_history_trace=all_rounds_trace
                )

                self._save_json(
                    os.path.join(self.cache_dir, f"research_reviewer_section_{section_id}_fullrun_{full_run_idx}_round_{review_round}.json"),
                    review_result
                )

                decision = str(review_result.get("decision", "accept")).lower()
                if decision != "reject":
                    print(f"[Researcher] Section {section_id} accepted.")
                    final_result = parsed
                    break

                # 4) Construct structured feedback
                strengths = review_result.get("strengths", [])
                weaknesses = review_result.get("weakness", [])
                suggestions = review_result.get("suggestions", [])

                strengths_text = "\n".join(f"- {s}" for s in strengths) if strengths else "None"
                weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) if weaknesses else "None"
                suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"

                feedback_msg = (
                    "# REVIEWER DECISION: REJECTED\n\n"
                    "The Reviewer has identified critical issues in your previous output. You must revise your plan immediately.\n\n"
                    "## 1. Audit Findings\n"
                    "**[Strengths to Maintain]**\n"
                    f"{strengths_text}\n\n"
                    "**[Critical Weaknesses to Fix]**\n"
                    f"{weaknesses_text}\n\n"
                    "## 2. Directives for Revision\n"
                    "**[Actionable Suggestions]**\n"
                    f"{suggestions_text}\n\n"
                    "## 3. Execution Protocol (IMPORTANT)\n"
                    "1. **Analyze the Feedback:** Read the \"Weaknesses\" carefully. Do not ignore them.\n"
                    "2. **Re-Search if Necessary:** If the rejection was due to \"Lack of Evidence\", \"Hallucination\", or \"Vague Details\", you MUST call the web_search tool to gather ground truth. Do not just guess.\n"
                    "3. **Re-Think:** Use your <think> block to plan how to address these specific critiques.\n"
                    "4. **Format:** Output the corrected JSON inside <output>...</output>.\n"
                )

                all_rounds_trace += f"\n\n=== ROUND {review_round + 1} REVIEWER FEEDBACK ===\n{feedback_msg}"
                messages.append({"role": "user", "content": feedback_msg})

            # Finalize the current full run.
            if final_result is None:
                final_result = last_json_grounded if last_json_grounded is not None else last_json_any

            # If no JSON was parsed in the entire run, try a full reset.
            no_parseable_json_entire_run = (last_json_any is None)

            if no_parseable_json_entire_run:
                print(
                    f"[Researcher] FULL RUN {full_run_idx + 1}: No parseable JSON in any round."
                )
                # Retry with a full reset if allowed.
                if full_run_idx < max_full_reruns:
                    # Save a marker trace for debugging.
                    self._save_text(
                        os.path.join(self.cache_dir, f"research_section_{section_id}_fullrun_{full_run_idx}_NO_JSON.txt"),
                        all_rounds_trace
                    )
                    continue
                # No retry remains; the caller decides how to handle None.
            else:
                # Any parseable JSON prevents a full rerun, even if review rejected it.
                break

        # Save images collected during the final full run.
        if self.images_from_webpage:
            images_save_path = os.path.join(self.cache_dir, f"research_images_section_{section_id}.json")
            print(f"[Researcher] Saving {len(self.images_from_webpage)} collected images to {images_save_path}")
            self._save_json(images_save_path, self.images_from_webpage)
        else:
            print(f"[Researcher] No images collected for section {section_id}.")

        return final_result


    def _run_research_once(self, messages, section_data, max_agent_steps=20) -> Tuple[str, str, List[dict]]:
        """
        Run one full research reasoning loop with iterative search + reasoning.
        Returns the trace for THIS specific round only.
        """
        full_trace = ""
        assistant_output = ""
        seen_queries = set()

        for step in range(max_agent_steps):
            print(f"[Researcher] Step {step + 1}: Thinking...")

            msg = self.client.chat_completion_message(
                messages,
                repitition_penalty=1.1,
                presence_penalty=1.0,
                tools=[SEARCH_TOOL],
                tool_choice="auto",
            )
            content = msg.content or ""
            reasoning_content = getattr(msg, "reasoning_content", "")
            tool_calls = getattr(msg, "tool_calls", None) or []

            if reasoning_content:
                assistant_output = f"<think>{reasoning_content}</think>\n{content}"
            else:
                assistant_output = content

            if tool_calls:
                assistant_output += self._format_tool_calls_for_trace(tool_calls)

            full_trace += assistant_output

            if tool_calls:
                messages.append(self._assistant_message_from_tool_calls(msg))
                for tool_call in tool_calls:
                    search_query = self._extract_query_from_tool_call(tool_call)
                    print(f"[Researcher] Detected web_search tool call: {search_query}")

                    if not search_query:
                        result_text = "INVALID_TOOL_CALL: Missing required argument 'query'."
                    elif search_query in seen_queries:
                        result_text = (
                            "DUPLICATE_QUERY_BLOCKED\n"
                            f"You have already searched this exact query: '{search_query}'.\n"
                            "Please refine it (add constraints, specific keywords) and search again."
                        )

                    else:
                        seen_queries.add(search_query)

                        result_text, result_imgs, source_urls = search_with_vision(search_query, section_data, top_k=5)
                        for url in source_urls:
                            norm_url = self._normalize_url(url)
                            if norm_url:
                                self.ground_urls.append(norm_url)
                        
                        self.images_from_webpage.extend(result_imgs)

                    result_block = f"{BEGIN_SEARCH_RESULT_TOKEN}{result_text}{END_SEARCH_RESULT_TOKEN}"

                    full_trace += "\n" + result_block
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_block,
                    })

                continue

            messages.append({"role": "assistant", "content": assistant_output})
            print("[Researcher] No further search requested. Ending loop.")
            break

        return assistant_output, full_trace, messages

    @staticmethod
    def _assistant_message_from_tool_calls(msg):
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (getattr(msg, "tool_calls", None) or [])
            ],
        }

    @staticmethod
    def _extract_query_from_tool_call(tool_call) -> str:
        try:
            args = json.loads(tool_call.function.arguments or "{}")
            return str(args.get("query", "")).strip()
        except Exception:
            return ""

    @staticmethod
    def _format_tool_calls_for_trace(tool_calls) -> str:
        chunks = []
        for tool_call in tool_calls:
            chunks.append(
                "\n<tool_call>"
                + json.dumps(
                    {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                    ensure_ascii=False,
                )
                + "</tool_call>"
            )
        return "".join(chunks)

    @staticmethod
    def _normalize_url(url: str) -> str:
        if not isinstance(url, str):
            return ""
        u = url.strip()
        u = u.rstrip(").,;:'\"")
        if u.endswith("/"):
            u = u[:-1]
        return u

    def _extract_citation_urls(self, json_output: dict) -> List[str]:
        urls: List[str] = []
        refs = json_output.get("references", [])
        for c in refs:
            u = self._normalize_url(c.get("url", ""))
            if u:
                urls.append(u)
        return urls

    def _rule_based_grounding_audit(self, json_output: dict) -> Optional[dict]:
        """
        Rule-based check:
        1) Ensure actual searches occurred (check if self.ground_urls is not empty).
        2) Ensure every citation URL in final JSON exists in self.ground_urls.
        """
        if not self.ground_urls:
            return {
                "decision": "reject",
                "strengths": ["Produced parseable JSON."],
                "weakness": [
                    "No valid search results were gathered.",
                    "You cannot generate a grounded report without valid search sources."
                ],
                "suggestions": [
                    "Perform at least two web_search tool calls that return valid results.",
                    "Ensure your search queries are not returning empty results."
                ],
                "score": 0.0
            }

        citation_urls = self._extract_citation_urls(json_output)
        if not citation_urls:
            return None 

        valid_url_set = set(self.ground_urls)
        
        missing = []
        for u in citation_urls:
            if u not in valid_url_set:
                missing.append(u)

        if missing:
            return {
                "decision": "reject",
                "strengths": [
                    "Produced parseable JSON.",
                    f"Have gathered {len(self.ground_urls)} valid source URLs."
                ],
                "weakness": [
                    "Hallucinated Citations: The final JSON cites URLs that were NOT returned by the search tool.",
                    f"Missing/Invalid URLs (sample): {missing}"
                ],
                "suggestions": [
                    "You can ONLY cite URLs that are strictly listed in the web_search tool results.",
                    "Check your 'references' list. Remove any URL that you did not actually search.",
                    "If you need a specific source, perform a new web_search tool call to get it into your context."
                ],
                "score": 0.0
            }

        return None

    def _format_rule_reject_feedback(self, rule_review: dict) -> str:
        strengths = rule_review.get("strengths", [])
        weaknesses = rule_review.get("weakness", [])
        suggestions = rule_review.get("suggestions", [])

        strengths_text = "\n".join(f"- {s}" for s in strengths) if strengths else "None"
        weaknesses_text = "\n".join(f"- {w}" for w in weaknesses) if weaknesses else "None"
        suggestions_text = "\n".join(f"- {s}" for s in suggestions) if suggestions else "None"

        return f"""
# REVIEWER DECISION: REJECTED

Your output failed an automatic grounding check.

## 1. Audit Findings
**[Strengths to Maintain]**
{strengths_text}

**[Critical Weaknesses to Fix]**
{weaknesses_text}

## 2. Directives for Revision
**[Actionable Suggestions]**
{suggestions_text}

## 3. Execution Protocol (MANDATORY)
1. You MUST call the web_search tool to obtain real sources if any citation is missing from the tool trace.
2. After each tool result, decide what to search next.
3. Regenerate the final JSON in <output>...</output> and ensure every citation URL appears in web_search results.
""".strip()

    def _run_reviewer(self, section_prompt: str, full_history_trace: str, max_retries: int = 3) -> dict:
        print("\n[Reviewer] Reviewing research process and output...")

        review_input = (
            "Below is the Research Task and the FULL Execution History (including previous rounds and feedback).\n\n"
            "[Research Task]\n"
            f"{section_prompt}\n\n"
            "[FULL Execution History]\n"
            f"{full_history_trace}\n\n"
            "Evaluate grounding, citation integrity, and research adequacy based on the LATEST round.\n"
            "Check if the agent addressed previous feedback.\n"
            "Output valid JSON only."
        )

        messages = [
            {"role": "system", "content": self.reviewer_system_prompt},
            {"role": "user", "content": review_input},
        ]

        for attempt in range(max_retries):
            raw_output, _ = self.reviewer_client.chat_completion(
                messages, 
                enable_thinking=False
            )
            
            json_str = self._extract_json(raw_output)
            
            try:
                parsed_json = loads_json_lenient(json_str)
                return parsed_json
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Reviewer] JSON parsing failed (Attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({
                        "role": "user", 
                        "content": f"Your previous output was not a valid JSON. Error: {str(e)}. Please provide the evaluation in valid JSON format only."
                    })
                else:
                    print("[Reviewer] Max retries reached. Returning default reject.")
                    return {
                        "decision": "reject",
                        "strengths": ["Reviewer failed to provide valid JSON."],
                        "weakness": [],
                        "suggestions": []
                    }

    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text.strip()

    def _save_final_output(self, answer: str, section_id: int = None) -> Optional[dict]:
        save_path = os.path.join(self.cache_dir, f"research_output_section_{section_id}.json")

        start_idx = answer.rfind(BEGIN_OUTPUT_TOKEN)
        if start_idx == -1:
            return None

        content_start = start_idx + len(BEGIN_OUTPUT_TOKEN)
        end_idx = answer.find(END_OUTPUT_TOKEN, content_start)
        if end_idx == -1:
            return None

        if end_idx <= content_start:
            return None

        extracted = answer[content_start:end_idx].strip()
        
        try:
            js = loads_json_lenient(extracted)
            self._save_json(save_path, js)
            return js
        except Exception:
            return None

    @staticmethod
    def _save_text(path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _save_json(path: str, content: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
