"""LLM plumbing for the ALFWorld port.

Differences from the Jericho original:

* Two endpoints instead of one hardcoded OpenRouter URL -- an actor server and a
  separate evaluator server. The evaluator is a *different model family* so its
  step scores are not self-evaluations of the actor's own trajectory; in JitRL
  those scores become the advantage, so the bias would be injected straight into
  the policy.
* Thinking is disabled for the actor. The original's only cleanup was a markdown
  fence strip (jitrl_agent.py:657-671), so one unhandled <think> prefix turns
  every step of an episode into the default action, silently.
* Embeddings are gone. FAISS was only a recall filter -- final ranking is 100%
  Jaccard (cross_episode_memory.py:428-432) -- and it needed a second API key.
* BadRequestError no longer disappears. The original catches openai.APIError,
  of which BadRequestError is a subclass, so a context overflow burned 5x20s and
  returned {} -- which callers read as "no dense reward signal" with no error.
"""

import json
import os
import re
import time
from typing import Mapping

import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_ACTOR_BASE_URL = os.getenv("JITRL_ACTOR_BASE_URL", "http://localhost:8000/v1")
DEFAULT_EVAL_BASE_URL = os.getenv("JITRL_EVAL_BASE_URL", "http://localhost:8001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")   # vLLM ignores the value

_CLIENTS = {}


class TokenLimitExceededError(Exception):
    """Context window overflow -- retrying is pointless, surface it loudly."""


def get_client(base_url=None):
    url = base_url or DEFAULT_ACTOR_BASE_URL
    if url not in _CLIENTS:
        _CLIENTS[url] = OpenAI(api_key=API_KEY, base_url=url)
    return _CLIENTS[url]


# ── response cleanup ───────────────────────────────────────────────────────
_THINK_CLOSED = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_THINK_OPEN = re.compile(r"^\s*<think>.*", re.DOTALL)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def strip_think(text):
    """Remove a <think> block, including the unterminated case.

    Must run BEFORE the brace-scanning JSON strategies: a greedy `\\{.*\\}` would
    happily swallow braces that appear inside reasoning text.
    """
    if not text:
        return text
    out = _THINK_CLOSED.sub("", text)
    if out is text and "<think>" in text and "</think>" not in text:
        out = _THINK_OPEN.sub("", text)
    return out.strip()


def _fix_control_characters(s):
    """Escape raw control characters that appear inside JSON string literals."""
    out = []
    in_string = False
    escaped = False
    for ch in s:
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            out.append(ch)
            continue
        if in_string and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch])
            continue
        out.append(ch)
    return "".join(out)


def extract_json_from_response(text):
    """Best-effort JSON recovery. Returns a dict, or None.

    Strategies, in order: direct parse -> fence strip -> control-char repair ->
    outermost brace span. The <think> strip happens first so strategy 4 cannot
    latch onto reasoning text.
    """
    if not text:
        return None
    cleaned = strip_think(text).strip()

    for candidate in (cleaned,):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            pass

    unfenced = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", cleaned)).strip()
    try:
        return json.loads(unfenced)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return json.loads(_fix_control_characters(unfenced))
    except (json.JSONDecodeError, TypeError):
        pass

    m = re.search(r"\{.*\}", unfenced, re.DOTALL)
    if m:
        for attempt in (m.group(0), _fix_control_characters(m.group(0))):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, TypeError):
                pass
    return None


# ── chat ───────────────────────────────────────────────────────────────────
def _is_context_error(exc):
    msg = str(exc).lower()
    return any(k in msg for k in (
        "maximum context length", "context_length_exceeded", "longer than the maximum",
        "reduce the length", "max_model_len", "too long",
    ))


def chat_completion_with_retries(
    model: str,
    sys_prompt: str,
    prompt: str,
    max_retries: int = 5,
    retry_interval_sec: int = 5,
    top_logprobs: int = 0,
    response_format=None,
    base_url: str = None,
    enable_thinking: bool = False,
    reasoning_effort: str = None,
    extra_body: dict = None,
    **kwargs,
) -> Mapping:
    """Chat completion. Returns the raw response object, or {} after exhausting retries.

    top_logprobs is only sent when > 0 -- the Jericho original sent it
    unconditionally (openai_helpers.py:31-33 driven by jitrl_agent.py:646) even
    in verbalized mode, where the logprobs are never read.
    """
    client = get_client(base_url)

    body = dict(extra_body or {})
    if enable_thinking is False:
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort

    for n_remaining in range(max_retries, 0, -1):
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt},
                ],
                **kwargs,
            }
            if top_logprobs > 0:
                params["logprobs"] = True
                params["top_logprobs"] = top_logprobs
            if response_format:
                params["response_format"] = response_format
            if body:
                params["extra_body"] = body

            return client.chat.completions.create(**params)

        except openai.BadRequestError as e:
            # Do NOT retry a 400 -- the original swallowed these as APIError.
            if _is_context_error(e):
                raise TokenLimitExceededError(str(e)) from e
            print(f"[LLM] BadRequest (not retrying): {e}", flush=True)
            return {}

        except (openai.RateLimitError, openai.APIError, openai.OpenAIError,
                json.JSONDecodeError) as e:
            print(f"[LLM] {type(e).__name__}: {e}", flush=True)
            print(f"[LLM] retrying in {retry_interval_sec}s ({n_remaining - 1} left)", flush=True)
            time.sleep(retry_interval_sec)

    print("[LLM] exhausted retries, returning {}", flush=True)
    return {}


def chat_text_with_retries(model, sys_prompt, prompt, **kwargs):
    """Chat completion returning the message text (thinking already stripped)."""
    res = chat_completion_with_retries(model, sys_prompt, prompt, **kwargs)
    if res and hasattr(res, "choices") and res.choices:
        return strip_think(res.choices[0].message.content or "").strip()
    return ""


def response_text(res):
    if res and hasattr(res, "choices") and res.choices:
        return strip_think(res.choices[0].message.content or "").strip()
    return ""
