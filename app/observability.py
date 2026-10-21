"""Observability, in two parts.

1. Turn logs (always on, local): every log line written during a turn carries `turn_id` and `thread_id`,
   and each turn ends with one "turn" summary line: nodes and their seconds, intents, agents, tools,
   Gemini calls/tokens, embeddings, guardrail result, status and total time. No message text is logged.

2. LangSmith tracing (optional, LANGSMITH_TRACING=true): each turn is sent to LangSmith as a trace you can
   click through (prompts, Gemini replies, tool arguments and results per node). Everything is passed
   through mask_phone before it leaves the machine (inputs, outputs, errors). Only our own tracer is used:
   the SDK's environment-variable auto-tracing is switched off so nothing is ever sent unmasked.
"""

import os
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from functools import lru_cache

import structlog

from app.config import Settings
from app.log import mask_value

log = structlog.get_logger()

# ---- 1. turn stats ---------------------------------------------------------------


@dataclass
class TurnStats:
    """Counters filled in by LoggedLLM and the vector store while a turn runs (thread-safe:
    tools can run in parallel)."""

    gemini_calls: int = 0
    gemini_retries: int = 0
    gemini_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_input_tokens: int = 0
    embedding_calls: int = 0
    embedding_cache_hits: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_gemini_call(self, seconds: float, usage: dict) -> None:
        with self._lock:
            self.gemini_calls += 1
            self.gemini_seconds += seconds
            self.input_tokens += usage.get("input_tokens") or 0
            self.output_tokens += usage.get("output_tokens") or 0
            self.thinking_tokens += (usage.get("output_token_details") or {}).get("reasoning") or 0
            self.cached_input_tokens += (usage.get("input_token_details") or {}).get("cache_read") or 0

    def add_retry(self) -> None:
        with self._lock:
            self.gemini_retries += 1

    def add_embedding(self, cached: bool) -> None:
        with self._lock:
            if cached:
                self.embedding_cache_hits += 1
            else:
                self.embedding_calls += 1

    def as_dict(self) -> dict:
        with self._lock:
            d = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        d["gemini_seconds"] = round(d["gemini_seconds"], 2)
        return d


_current_stats: ContextVar[TurnStats | None] = ContextVar("turn_stats", default=None)


def current_stats() -> TurnStats | None:
    """The running turn's counters, or None outside a turn (e.g. evals, scripts)."""
    return _current_stats.get()


@contextmanager
def turn_context(thread_id: str):
    """Give every log line in this turn a turn_id/thread_id and collect its stats.
    LangGraph copies context variables into its worker threads, so parallel tool calls are included."""
    turn_id = uuid.uuid4().hex[:8]
    stats = TurnStats()
    token = _current_stats.set(stats)
    structlog.contextvars.bind_contextvars(turn_id=turn_id, thread_id=thread_id)
    try:
        yield turn_id, stats
    finally:
        structlog.contextvars.unbind_contextvars("turn_id", "thread_id")
        _current_stats.reset(token)


# ---- 2. LangSmith ----------------------------------------------------------------

AUTO_TRACING_ENV_VARS = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")


def mask_trace(data):
    """LangSmith anonymizer: gets the JSON form of a run's inputs, outputs or error; masks phone numbers."""
    return mask_value(data)


def disable_auto_tracing() -> None:
    """Stop the LangSmith SDK from tracing on its own because of a shell variable: that path would skip
    our masking. Tracing happens only through tracing_callbacks()."""
    for name in AUTO_TRACING_ENV_VARS:
        if os.environ.get(name, "").strip().lower() not in ("", "false", "0"):
            log.warning("auto-tracing env var ignored; use LANGSMITH_TRACING in .env", variable=name)
        os.environ[name] = "false"


@lru_cache
def _client(api_key: str, endpoint: str | None):
    from langsmith import Client

    return Client(api_key=api_key, api_url=endpoint, anonymizer=mask_trace)


def langsmith_client(s: Settings):
    if s.langsmith_api_key is None or not s.langsmith_api_key.get_secret_value().strip():
        raise RuntimeError("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty. Add it to .env or set "
                           "LANGSMITH_TRACING=false.")
    return _client(s.langsmith_api_key.get_secret_value(), (s.langsmith_endpoint or "").strip() or None)


def tracing_callbacks(s: Settings) -> list:
    """[] when tracing is off; otherwise one LangSmith tracer whose client masks every payload."""
    if not s.langsmith_tracing:
        return []
    from langchain_core.tracers.langchain import LangChainTracer

    return [LangChainTracer(project_name=s.langsmith_project, client=langsmith_client(s))]


def flush_tracing(s: Settings) -> None:
    """Send any traces still queued (call before the process exits)."""
    if s.langsmith_tracing:
        try:
            langsmith_client(s).flush()
        except Exception:  # noqa: BLE001 - tracing must never break the app
            log.warning("could not flush LangSmith traces", exc_info=True)
