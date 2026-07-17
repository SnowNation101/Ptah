from __future__ import annotations

import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "Qwen/Qwen3-VL-235B-A22B-Instruct"
DEFAULT_UNIAPI_BASE_URL = "https://hk.uniapi.io/v1"
DEFAULT_UNIAPI_MODEL = "qwen3-vl-235b-a22b-instruct"
DEFAULT_JUDGE_PROVIDER = "uniapi"


class EvalParseError(Exception):
    """Raised when a judge model response cannot be parsed into valid JSON."""


def current_date_context() -> str:
    configured = os.getenv("PTAH_CURRENT_DATE", "").strip()
    if configured:
        return f"Current date: {configured}. Use this as today's date for evaluation."
    now = datetime.now(ZoneInfo(os.getenv("PTAH_TIMEZONE", "Asia/Shanghai")))
    return f"Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}, {now.tzinfo}). Use this as today's date for evaluation."


def inject_current_date(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    date_context = current_date_context()
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        content = copied[0].get("content", "")
        if isinstance(content, str) and "Current date:" not in content:
            copied[0]["content"] = f"{date_context}\n\n{content}"
        return copied
    return [{"role": "system", "content": date_context}, *copied]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_data_url(img_b64: str, default_mime: str = "image/png") -> str:
    if img_b64.startswith("data:"):
        return img_b64
    return f"data:{default_mime};base64,{img_b64}"


def file_to_data_url(image_path: str | Path) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")

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


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise EvalParseError("Empty model output.")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    for pattern in (r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    start = text.find("{")
    if start != -1:
        level = 0
        for i, char in enumerate(text[start:]):
            if char == "{":
                level += 1
            elif char == "}":
                level -= 1
                if level == 0:
                    candidate = text[start : start + i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break

    raise EvalParseError("Failed to parse JSON object from model output.")


def extract_assistant_text(resp_json: dict[str, Any]) -> str:
    choices = resp_json.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def call_siliconflow_chat(
    *,
    api_key: str,
    api_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 300,
    response_format: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": inject_current_date(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if extra_body:
        payload.update(extra_body)

    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def normalize_provider(provider: str | None) -> str:
    provider = (provider or DEFAULT_JUDGE_PROVIDER).strip().lower()
    aliases = {
        "sf": "siliconflow",
        "openai_compatible": "openai-compatible",
        "openai-compatible": "openai-compatible",
        "openai": "openai-compatible",
        "vllm": "local",
    }
    return aliases.get(provider, provider)


def default_api_key_envs(provider: str | None) -> list[str]:
    provider = normalize_provider(provider)
    if provider == "uniapi":
        return ["UNIAPI_API_KEY", "OPENAI_API_KEY"]
    if provider == "siliconflow":
        return ["SILICONFLOW_API_KEY"]
    if provider == "openai-compatible":
        return ["OPENAI_API_KEY"]
    if provider == "local":
        return ["OPENAI_API_KEY", "SILICONFLOW_API_KEY"]
    return ["OPENAI_API_KEY"]


def resolve_api_key(api_key_env: str | None, provider: str | None) -> tuple[str, str]:
    candidates = [api_key_env] if api_key_env else default_api_key_envs(provider)
    for env_name in candidates:
        if not env_name:
            continue
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return "", candidates[0] or ""


def chat_completions_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def base_url_from_chat_url(api_url: str) -> str:
    api_url = api_url.rstrip("/")
    suffix = "/chat/completions"
    if api_url.endswith(suffix):
        return api_url[: -len(suffix)]
    return api_url


def call_openai_compatible_chat(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 300,
    response_format: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("openai package is required for UniAPI/OpenAI-compatible judge calls.") from e

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": inject_current_date(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_body:
        kwargs["extra_body"] = extra_body

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
    response = client.chat.completions.create(**kwargs)
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return json.loads(response.model_dump_json())


def call_judge_chat(
    *,
    provider: str | None,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    api_url: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    timeout: int = 300,
    response_format: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provider = normalize_provider(provider)
    if provider == "uniapi":
        return call_openai_compatible_chat(
            api_key=api_key,
            base_url=base_url or (base_url_from_chat_url(api_url) if api_url else DEFAULT_UNIAPI_BASE_URL),
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
        )
    if provider == "openai-compatible":
        if not base_url and not api_url:
            raise ValueError("--base-url or --api-url is required for openai-compatible provider.")
        return call_openai_compatible_chat(
            api_key=api_key,
            base_url=base_url or base_url_from_chat_url(str(api_url)),
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            extra_body=extra_body,
        )
    if provider == "siliconflow":
        target_api_url = api_url or DEFAULT_SILICONFLOW_API_URL
    else:
        target_api_url = api_url or chat_completions_url(base_url or DEFAULT_SILICONFLOW_API_URL)
    return call_siliconflow_chat(
        api_key=api_key,
        api_url=target_api_url,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        response_format=response_format,
        extra_body=extra_body,
    )


def average_score_dict(records: list[dict[str, Any]], scores_key: str = "scores") -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item in records:
        scores = item.get(scores_key, {})
        if not isinstance(scores, dict):
            continue
        for dim, value in scores.items():
            if isinstance(value, (int, float)):
                buckets.setdefault(dim, []).append(float(value))
    return {dim: round(sum(values) / len(values), 4) for dim, values in buckets.items() if values}
