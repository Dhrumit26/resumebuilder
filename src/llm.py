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


MAX_JSON_TOKEN_CAP = 8000


class TruncatedCompletion(Exception):
    """The model stopped because it hit max_tokens, not because it finished.

    This is the real source of bullets that end mid-sentence. Before this
    existed, truncation was invisible here and downstream code tried to infer it
    from the bullet's trailing word — a guess that misfired on ordinary endings.
    """


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
    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise ValueError("Empty response from LLM")
    if getattr(choice, "finish_reason", None) == "length":
        raise TruncatedCompletion(
            f"hit the {max_tokens}-token cap mid-sentence (finish_reason=length)"
        )
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


_VALID_JSON_ESCAPES = set('"\\/bfnrtu')


def _fix_invalid_escapes(text: str) -> str:
    """Drop backslashes that JSON does not allow, so one stray \\% is survivable.

    Models asked for LaTeX-flavoured text sometimes emit "cut costs by 40\\%"
    inside a JSON string. That is an invalid escape and json.loads rejects the
    whole object — losing an entire section over one character.
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text) and text[i + 1] not in _VALID_JSON_ESCAPES:
            i += 1  # skip the backslash, keep the character it was escaping
            continue
        out.append(ch)
        i += 1
    return "".join(out)


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
    token_cap = max_tokens

    for _attempt in range(retries + 1):
        try:
            raw = call_llm(current_prompt, temperature=temperature, max_tokens=token_cap, role=role)
        except TruncatedCompletion as exc:
            # A JSON object cut mid-key is unparseable and unrecoverable, so raise the
            # ceiling and retry rather than losing the whole analysis. Without this a
            # truncated reviewer response returns None and the fix loop never runs.
            last_error = exc
            token_cap = min(token_cap * 2, MAX_JSON_TOKEN_CAP)
            continue
        try:
            cleaned = _extract_json_text(raw)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                data = json.loads(_fix_invalid_escapes(cleaned))
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


def call_web_research_json(prompt: str, max_tokens: int = 3000) -> dict:
    """Run one web-grounded research call through OpenAI Responses.

    The regular writer path remains provider-compatible Chat Completions. Web
    research is intentionally separate because tool syntax is provider-specific.
    Callers must catch failures and fall back to JD-only planning.
    """
    if LLM_PROVIDER != "openai":
        raise ValueError("Web research currently requires LLM_PROVIDER=openai")

    client = _get_client()
    response = client.responses.create(
        model=_get_model("writer"),
        input=prompt,
        tools=[{"type": "web_search_preview"}],
        max_output_tokens=min(max_tokens, MAX_JSON_TOKEN_CAP),
    )
    raw = getattr(response, "output_text", None)
    if not raw:
        raise ValueError("Empty web research response")
    cleaned = _extract_json_text(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = json.loads(_fix_invalid_escapes(cleaned))
    if not isinstance(data, dict):
        raise ValueError(f"Expected research JSON object, got {type(data).__name__}")
    return data
