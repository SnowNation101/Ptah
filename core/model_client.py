import os
import openai
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, List, Optional, Dict, Tuple


def current_date_context() -> str:
    configured = os.getenv("PTAH_CURRENT_DATE", "").strip()
    if configured:
        return (
            f"Current date: {configured}. "
            "Use this as today's date for all time-sensitive reasoning, searches, and report writing."
        )

    now = datetime.now(ZoneInfo(os.getenv("PTAH_TIMEZONE", "Asia/Shanghai")))
    return (
        f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}, {now.tzinfo}). "
        "Use this as today's date for all time-sensitive reasoning, searches, and report writing."
    )


def inject_current_date(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    date_context = current_date_context()
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        content = copied[0].get("content", "")
        if isinstance(content, str) and "Current date:" not in content:
            copied[0]["content"] = f"{date_context}\n\n{content}"
        return copied
    return [{"role": "system", "content": date_context}, *copied]

class ModelClient:
    def __init__(self, 
                 model_name: str, 
                 base_url: str, 
                 api_key: str = "EMPTY", 
                 **kwargs):
        """
        Initializes a universal model client.
        
        Args:
            model_name: The name of the model.
            base_url: The API base URL.
            api_key: The API Key. For local vLLM, this is usually "EMPTY".
            kwargs: Additional arguments for the openai.OpenAI client (e.g., timeout).
        """
        self.model_name = model_name
        
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )

    def chat_completion(self, 
                        messages: List[Dict[str, str]], 
                        max_tokens: int = 4096, 
                        stop: Optional[List[str]] = None,
                        temperature: float = 0.6,
                        top_p: float = 0.95,
                        enable_thinking: bool = True,
                        repitition_penalty: float = 1.0,
                        presence_penalty: float = 0.0,
                        ) -> Tuple[str, str]:
        """
        chat completion method.
        """
        msg = self.chat_completion_message(
            messages=messages,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
            repitition_penalty=repitition_penalty,
            presence_penalty=presence_penalty,
        )
        content = msg.content or ""
        reasoning_content = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        return content, reasoning_content

    def chat_completion_message(
                        self,
                        messages: List[Dict[str, Any]],
                        max_tokens: int = 4096,
                        stop: Optional[List[str]] = None,
                        temperature: float = 0.6,
                        top_p: float = 0.95,
                        enable_thinking: bool = True,
                        repitition_penalty: float = 1.0,
                        presence_penalty: float = 0.0,
                        tools: Optional[List[Dict[str, Any]]] = None,
                        tool_choice: Optional[Any] = None):
        """
        chat completion method that returns the raw assistant message.
        Use this when callers need native OpenAI-compatible tool_calls.
        """
        params = {
            "model": self.model_name,
            "messages": inject_current_date(messages),
            "max_tokens": max_tokens,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "repetition_penalty": repitition_penalty,
                "presence_penalty": presence_penalty,
                "include_stop_str_in_output": True,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                # "reasoning": True,
                # "show_reasoning": True,
            }
        }

        if tools is not None:
            params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
        
        wait = 5
        max_retries = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                msg = response.choices[0].message
                if not getattr(msg, "reasoning_content", None) and getattr(msg, "reasoning", None):
                    try:
                        setattr(msg, "reasoning_content", getattr(msg, "reasoning"))
                    except Exception:
                        pass
                return msg

            except openai.APIError as e:
                if attempt < max_retries - 1:
                    print(f"API Error: {e}. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise RuntimeError(f"API Provider Error: {e}")
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Model Client Error: {e}. Retrying in {wait} seconds...")
                    time.sleep(wait)
                    wait *= 2
                    continue
                raise RuntimeError(f"Model Client Error: {e}")
