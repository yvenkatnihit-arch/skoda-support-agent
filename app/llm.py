"""Gemini clients: one chat model per graph node, plus the embeddings client.

Every chat call goes through LoggedLLM, which logs node, model, request size, elapsed time and token usage
(including thinking tokens), and retries transient errors with a growing wait. The SDK's own retries are
switched off so each attempt is visible in the logs and the timing is predictable.
"""

import json
import random
import time

import httpx
import structlog
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings, get_settings
from app.observability import current_stats

log = structlog.get_logger()

# Routing, tool use and summaries should be deterministic; customer-facing wording can vary a little.
NODE_TEMPERATURE = {"composer": 0.3, "clarify": 0.3}
SDK_SINGLE_ATTEMPT = 1  # langchain-google-genai: max_retries=1 means one attempt (0 means "SDK default of 5")
TRANSIENT_HTTP_CODES = {408, 429, 500, 502, 503, 504}
TRANSIENT_MARKERS = ("DEADLINE_EXCEEDED", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL")
MIN_PRO_THINKING_BUDGET = 128  # gemini-2.5-pro can't turn thinking off

# Visible output tokens per node, on top of the thinking budget. A cap stops a runaway generation in seconds
# instead of running into the server deadline (seen: json_schema output repeating "\n" forever -> 504).
NODE_MAX_OUTPUT_TOKENS = {
    "supervisor": 512, "clarify": 256, "summarize": 512,
    "car_info": 2048, "quotation": 2048, "locator": 2048, "service": 2048, "lead": 1024,
    "composer": 2048,  # longest real reply so far (price table + 3 showrooms) was ~850 tokens
}

AGENT_MAX_OUTPUT_TOKENS = 2048  # for agents added later that aren't listed above

# Structured output via function calling: json_schema mode intermittently degenerated on Gemini 2.5 Flash.
STRUCTURED_OUTPUT_METHOD = "function_calling"


class LLMUnavailableError(RuntimeError):
    """Gemini could not answer after all retries (or with a non-retryable error)."""

    def __init__(self, node: str, attempts: int, cause: Exception):
        self.node, self.attempts = node, attempts
        super().__init__(f"Gemini call for '{node}' failed after {attempts} attempt(s): "
                         f"{type(cause).__name__}: {str(cause)[:300]}")


def _api_key(s: Settings):
    if s.google_api_key is None:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to .env (see .env.example).")
    return s.google_api_key


# ---- thinking --------------------------------------------------------------------

def thinking_kwargs(model: str, budget: int) -> dict:
    """ChatGoogleGenerativeAI kwargs for a token budget: thinking_budget for 2.5, thinking_level for 3+."""
    name = model.removeprefix("models/")
    if name.startswith("gemini-3"):
        if budget < 0:
            return {}  # model default
        if budget == 0:
            return {"thinking_level": "minimal" if "flash" in name else "low"}
        return {"thinking_level": "low" if budget <= 1024 else "medium"}
    if "pro" in name and 0 <= budget < MIN_PRO_THINKING_BUDGET:
        log.warning("thinking budget raised to the model minimum", model=name, requested=budget,
                    used=MIN_PRO_THINKING_BUDGET)
        budget = MIN_PRO_THINKING_BUDGET
    return {"thinking_budget": budget}


# ---- retries ---------------------------------------------------------------------

def _error_chain(exc: BaseException):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def is_transient(exc: BaseException) -> bool:
    """Timeouts, rate limits and server errors are worth retrying; bad requests and auth errors are not."""
    for e in _error_chain(exc):
        if isinstance(e, (httpx.TimeoutException, httpx.TransportError, TimeoutError, ConnectionError)):
            return True
        code = getattr(e, "code", None) or getattr(e, "status_code", None)
        if isinstance(code, int) and code in TRANSIENT_HTTP_CODES:
            return True
        text = str(e)
        if any(marker in text for marker in TRANSIENT_MARKERS) or any(f"{c} " in text[:5] for c in TRANSIENT_HTTP_CODES):
            return True
    return False


def retry_delay(retry_number: int, s: Settings) -> float:
    """Exponential backoff with a little jitter: base, 2x base, 4x base ... capped at the max."""
    delay = min(s.llm_retry_base_delay_s * 2 ** (retry_number - 1), s.llm_retry_max_delay_s)
    return delay + random.uniform(0, delay * 0.25)


# ---- logging wrapper -------------------------------------------------------------

def request_chars(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", m)
        total += len(content) if isinstance(content, str) else len(json.dumps(content, default=str))
        for call in getattr(m, "tool_calls", None) or []:
            total += len(json.dumps(call.get("args", {}), default=str))
    return total


class LoggedLLM:
    """Wraps a chat runnable (plain, bind_tools or with_structured_output(include_raw=True))."""

    def __init__(self, runnable, node: str, model: str, settings: Settings | None = None, sleep=time.sleep):
        self.runnable, self.node, self.model = runnable, node, model
        self.settings = settings or get_settings()
        self._sleep = sleep

    def invoke(self, messages):
        s = self.settings
        chars = request_chars(messages)
        attempts = s.llm_max_retries + 1
        for attempt in range(1, attempts + 1):
            start = time.perf_counter()
            try:
                result = self.runnable.invoke(messages)
            except Exception as exc:  # noqa: BLE001 - classified below, re-raised as LLMUnavailableError
                elapsed = round(time.perf_counter() - start, 2)
                transient = is_transient(exc)
                log.warning("gemini call failed", node=self.node, model=self.model, attempt=attempt,
                            of=attempts, elapsed_s=elapsed, transient=transient,
                            error=f"{type(exc).__name__}: {str(exc)[:300]}")
                if not transient or attempt == attempts:
                    raise LLMUnavailableError(self.node, attempt, exc) from exc
                wait = retry_delay(attempt, s)
                log.info("gemini retry", node=self.node, wait_s=round(wait, 2), next_attempt=attempt + 1)
                if (stats := current_stats()) is not None:
                    stats.add_retry()
                self._sleep(wait)
                continue
            elapsed = round(time.perf_counter() - start, 2)
            raw, value = self._unwrap(result)
            usage = getattr(raw, "usage_metadata", None) or {}
            finish = (getattr(raw, "response_metadata", None) or {}).get("finish_reason")
            log.info("gemini call", node=self.node, model=self.model, attempt=attempt, elapsed_s=elapsed,
                     request_chars=chars, approx_request_tokens=chars // 4,
                     input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens"),
                     thinking_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
                     cached_input_tokens=(usage.get("input_token_details") or {}).get("cache_read"),
                     finish_reason=finish)
            if finish == "MAX_TOKENS":
                log.warning("gemini output cut at the token limit", node=self.node, model=self.model)
            if (stats := current_stats()) is not None:
                stats.add_gemini_call(elapsed, usage)
            return value
        raise AssertionError("unreachable")

    def _unwrap(self, result):
        """with_structured_output(include_raw=True) returns {raw, parsed, parsing_error}."""
        if isinstance(result, dict) and "raw" in result and "parsed" in result:
            if result.get("parsing_error") is not None or result["parsed"] is None:
                raise LLMUnavailableError(self.node, 1, ValueError(f"unparseable reply: {result.get('parsing_error')}"))
            return result["raw"], result["parsed"]
        return result, result


# ---- factories -------------------------------------------------------------------

def get_chat_model(node: str, settings: Settings | None = None) -> ChatGoogleGenerativeAI:
    """Gemini chat model for a node: <NODE>_MODEL (or DEFAULT_MODEL), timeout, thinking budget from .env."""
    s = settings or get_settings()
    model = s.model_for(node)
    budget = s.thinking_budget_for(node)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=_api_key(s),
        timeout=s.llm_timeout_s,
        max_retries=SDK_SINGLE_ATTEMPT,  # retries are done (and logged) by LoggedLLM
        temperature=NODE_TEMPERATURE.get(node, 0.0),
        max_output_tokens=NODE_MAX_OUTPUT_TOKENS.get(node, AGENT_MAX_OUTPUT_TOKENS) + max(budget, 0),  # thinking counts toward the limit
        **thinking_kwargs(model, budget),
    )


def logged(node: str, runnable, settings: Settings | None = None) -> LoggedLLM:
    s = settings or get_settings()
    return LoggedLLM(runnable, node, s.model_for(node), s)


def get_embeddings(settings: Settings | None = None) -> GoogleGenerativeAIEmbeddings:
    """Gemini embeddings client for the model and dimension configured in .env."""
    s = settings or get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=s.embedding_model,
        google_api_key=_api_key(s),
        output_dimensionality=s.embedding_dim,
    )
