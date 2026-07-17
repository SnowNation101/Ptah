from .base import BaseAgent
from utils.search import search_and_summarize
import re
import json
import os

BEGIN_SEARCH_TOKEN = "<search>"
END_SEARCH_TOKEN = "</search>"
BEGIN_SEARCH_RESULT_TOKEN = "<search_result>"
END_SEARCH_RESULT_TOKEN = "</search_result>"
BEGIN_IMGEN_TOKEN = "<imgen>"
END_IMGEN_TOKEN = "</imgen>"
BEGIN_OUTPUT_TOKEN = "<output>"
END_OUTPUT_TOKEN = "</output>"

STRING_ARRAY_KEYS = {
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
            "Search the web for evidence needed to plan the report. "
            "Returns concise evidence cards with page titles, URLs, claims, data, and examples."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A focused plain-text web search query."
                }
            },
            "required": ["query"]
        }
    }
}


class PlannerAgent(BaseAgent):
    """
    OutlineAgent performs multi-step reasoning with optional web searches.
    Each round of reasoning is reviewed. All process logs are saved to cache_dir.
    """

    def __init__(self, client, reviewer_client, cache_dir: str = ".cache"):
        super().__init__(client, "prompts/planner.txt")

        with open("prompts/outline_reviewer.txt", "r", encoding="utf-8") as f:
            self.reviewer_system_prompt = f.read().strip()
        self.reviewer_client = reviewer_client

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def run(self, question: str, max_agent_steps: int = 10, max_review_rounds: int = 3):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        last_valid_json = None

        for review_round in range(max_review_rounds + 1):

            answer, trace, messages = self._run_outline_once(
                messages=messages,
                max_agent_steps=max_agent_steps,
            )

            self._save_text(
                os.path.join(self.cache_dir, f"outline_round_{review_round}.txt"),
                trace
            )

            parsed = self._save_final_output(answer)
            if not isinstance(parsed, dict):
                print("[Reviewer]: Failed to parse valid JSON from output.")
                review_result = {
                    "decision": "reject",
                    "weakness": ["Failed to produce valid JSON in <output>...</output>."],
                    "suggestions": ["Ensure your final output is valid JSON enclosed within the <output> and </output>."]
                }
            else:
                last_valid_json = parsed
                print(f"[Planner] Completed round {review_round}, parsed JSON saved.")

                review_result = self._run_reviewer(question, trace)

            self._save_json(
                os.path.join(self.cache_dir, f"outline_reviewer_round_{review_round}.txt"),
                review_result
            )

            decision = review_result.get("decision", "reject").lower()

            if decision != "reject":
                return parsed

            if review_round >= max_review_rounds:
                return last_valid_json


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

            messages.append({
                "role": "user",
                "content": feedback_msg
            })

        return last_valid_json

    def _run_outline_once(self, messages, max_agent_steps=20):

        full_trace = ""

        assistant_output = ""
        seen_queries = set()

        for step in range(max_agent_steps):
            print(f"[Planner] Step {step + 1}: Thinking...")

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
                assistant_output = f"<think>{reasoning_content}</think>{content}"
            else:
                assistant_output = content

            if tool_calls:
                assistant_output += self._format_tool_calls_for_trace(tool_calls)

            print(f"[Planner] Assistant output:\n{assistant_output}\n")

            full_trace += assistant_output

            if tool_calls:
                messages.append(self._assistant_message_from_tool_calls(msg))
                for tool_call in tool_calls:
                    query_text = self._extract_query_from_tool_call(tool_call)
                    print(f"[Planner] Detected web_search tool call: {query_text}")

                    if not query_text:
                        result_text = "INVALID_TOOL_CALL: Missing required argument 'query'."
                    elif query_text in seen_queries:
                        print(f"[Planner] Duplicate search query detected, skipping: {query_text}")
                        result_text = (
                            "DUPLICATE_QUERY_BLOCKED\n"
                            f"You have already searched this exact query:\n"
                            f"- Query: {query_text}\n\n"
                            "Please use a different query or refine it (add constraints like year range, domain, "
                            "full framework name, or additional keywords) and search again."
                        )
                    else:
                        seen_queries.add(query_text)
                        _, result_text = search_and_summarize(query_text, top_k=5)

                    result_block = f"{BEGIN_SEARCH_RESULT_TOKEN}{result_text}{END_SEARCH_RESULT_TOKEN}"
                    full_trace += "\n" + result_block
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_block,
                    })

                continue

            print("[Planner] Done.")
            messages.append({"role": "assistant", "content": assistant_output})
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

    def _run_reviewer(self, question: str, full_trace: str, max_retries: int = 3) -> dict:
        print("[Reviewer] Reviewing planner output...")

        review_input = (
            "Below is the user question and the Planner output.\n\n"
            "[User Question]\n"
            f"{question}\n\n"
            "[Planner Output]\n"
            f"{full_trace}\n\n"
            "Please evaluate this and output your judgment in valid JSON format only."
        )

        messages = [
            {"role": "system", "content": self.reviewer_system_prompt},
            {"role": "user", "content": review_input},
        ]

        for attempt in range(max_retries):
            raw_output, _ = self.reviewer_client.chat_completion(
                messages, 
                enable_thinking=False,
            )
            print(f"[Reviewer] Raw output (Attempt {attempt + 1}):", raw_output)

            json_str = self._extract_json(raw_output)
            
            try:
                parsed_review = loads_json_lenient(json_str)
                return parsed_review
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Reviewer] JSON parsing failed: {e}")
                
                if attempt < max_retries - 1:
                    messages.append({"role": "assistant", "content": raw_output})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your previous response was not a valid JSON. "
                            f"Error: {str(e)}. "
                            "Please output only the JSON object, ensuring all quotes and braces are correct."
                        )
                    })
                else:
                    return {
                        "decision": "reject",
                        "weakness": ["Reviewer failed to output parseable JSON after multiple retries."],
                        "suggestions": ["Please try again; internal reviewer format error."]
                    }

        
    @staticmethod
    def _extract_json(text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start:end + 1]
        return text

    def _save_final_output(self, answer: str):
        """
        Strict mode:
        Extract JSON ONLY between the LAST BEGIN_OUTPUT_TOKEN and the FIRST END_OUTPUT_TOKEN after it.
        If tokens are missing/invalid, do not fallback to whole answer.
        Return parsed JSON (dict) or None if parsing failed.
        """
        save_path = os.path.join(self.cache_dir, "outline_output.json")

        start_idx = answer.rfind(BEGIN_OUTPUT_TOKEN)
        if start_idx == -1:
            self._save_text(save_path, f"Missing BEGIN_OUTPUT_TOKEN: {BEGIN_OUTPUT_TOKEN!r}")
            return None

        content_start = start_idx + len(BEGIN_OUTPUT_TOKEN)
        end_idx = answer.find(END_OUTPUT_TOKEN, content_start)
        if end_idx == -1:
            self._save_text(save_path, f"Missing END_OUTPUT_TOKEN after last BEGIN_OUTPUT_TOKEN: {END_OUTPUT_TOKEN!r}")
            return None

        if end_idx <= content_start:
            self._save_text(save_path, "Invalid token order: END_OUTPUT_TOKEN occurs before content start.")
            return None

        extracted = answer[content_start:end_idx].strip()

        try:
            js = loads_json_lenient(extracted)
            self._save_json(save_path, js)
            return js
        except Exception:
            # Save the extracted segment for debugging
            self._save_text(save_path, extracted)
            return None


    @staticmethod
    def _save_text(path, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _save_json(path, content):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
