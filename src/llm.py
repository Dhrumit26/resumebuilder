import json
import re
from typing import Optional

from openai import OpenAI

from .config import (
    GROK_API_KEY,
    GROK_BASE_URL,
    GROK_JUDGE_MODEL,
    GROK_MODEL,
    GROK_WRITER_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def _get_client() -> OpenAI:
    # timeout: a hung request must not stall the whole pipeline;
    # max_retries: the SDK retries transient network/5xx/429 errors with backoff
    if LLM_PROVIDER == "grok":
        if not GROK_API_KEY:
            raise ValueError("GROK_API_KEY is not set")
        return OpenAI(
            api_key=GROK_API_KEY,
            base_url=GROK_BASE_URL,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=2,
        )
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS, max_retries=2)


def _get_model(role: str = "writer") -> str:
    if LLM_PROVIDER == "grok":
        return GROK_JUDGE_MODEL if role == "judge" else GROK_WRITER_MODEL
    return OPENAI_MODEL


def call_llm(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    role: str = "writer",
) -> str:
    client = _get_client()
    kwargs = {
        "model": _get_model(role),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Empty response from LLM")
    return content.strip()


def _extract_json_text(raw: str) -> str:
    """Pull a JSON object/array out of markdown-wrapped or prose-prefixed model output."""
    text = raw.strip()

    # Strip ```json ... ``` or ``` ... ``` fences anywhere
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # Already looks like JSON
    if text.startswith("{") or text.startswith("["):
        return text

    # Find first balanced {...} or [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def call_llm_json(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    retries: int = 1,
    role: str = "judge",
) -> dict:
    last_error: Optional[Exception] = None
    current_prompt = prompt
    raw = ""

    for _attempt in range(retries + 1):
        raw = call_llm(current_prompt, temperature=temperature, max_tokens=max_tokens, role=role)
        try:
            cleaned = _extract_json_text(raw)
            data = json.loads(cleaned)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            return data
        except Exception as exc:
            last_error = exc
            current_prompt = (
                prompt
                + "\n\nIMPORTANT: Reply with ONLY a raw JSON object. "
                "No markdown, no code fences, no **Output:** prefix, no explanation."
            )

    preview = (raw[:240] + "...") if len(raw) > 240 else raw
    raise ValueError(
        f"Failed to parse LLM JSON ({type(last_error).__name__}: {last_error}). "
        f"Raw preview: {preview!r}"
    ) from last_error
