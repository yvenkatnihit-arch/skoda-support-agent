"""Application settings, loaded from environment variables and the local .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Graph nodes that call Gemini. Each can override DEFAULT_MODEL with <NODE>_MODEL in .env.
LLM_NODES = (
    "supervisor",
    "clarify",
    "car_info",
    "quotation",
    "locator",
    "service",
    "lead",
    "composer",
    "summarize",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Gemini
    google_api_key: SecretStr | None = None
    default_model: str = "gemini-2.5-flash"
    supervisor_model: str | None = None
    clarify_model: str | None = None
    car_info_model: str | None = None
    quotation_model: str | None = None
    locator_model: str | None = None
    service_model: str | None = None
    lead_model: str | None = None
    composer_model: str | None = None
    summarize_model: str | None = None
    llm_timeout_s: float = Field(default=60, gt=0)  # per attempt; also the server-side deadline
    llm_max_retries: int = Field(default=3, ge=0)  # retries after the first attempt (transient errors only)
    llm_retry_base_delay_s: float = Field(default=1.0, gt=0)  # wait before retry 1; doubles each retry
    llm_retry_max_delay_s: float = Field(default=8.0, gt=0)

    # Thinking budget (tokens) per node for Gemini 2.5: 0 = off, -1 = dynamic (model default).
    # Blank = the default in THINKING_BUDGET_DEFAULTS below.
    supervisor_thinking_budget: int | None = None
    clarify_thinking_budget: int | None = None
    car_info_thinking_budget: int | None = None
    quotation_thinking_budget: int | None = None
    locator_thinking_budget: int | None = None
    service_thinking_budget: int | None = None
    lead_thinking_budget: int | None = None
    composer_thinking_budget: int | None = None
    summarize_thinking_budget: int | None = None

    # Support tickets: what the confirmation promises the customer (a business promise: keep it accurate).
    support_response_promise: str = "within 24 hours"
    # Hand-off to a human. The number comes from the sample service FAQ: confirm it before going live.
    customer_care_number: str = "1800 2700 260"
    escalate_after_strikes: int = Field(default=2, ge=1)  # failed / stuck turns in a row before handing off

    # Graph limits
    max_tool_rounds: int = Field(default=5, ge=1, le=20)  # tool-call rounds per agent per question

    # Embeddings and retrieval (changing the embedding model or dimension requires rebuilding the index)
    embedding_model: str = "gemini-embedding-2"
    embedding_dim: int = Field(default=768, ge=128, le=3072)
    retrieval_top_k: int = Field(default=4, ge=1, le=10)
    # Tuned on the sample docs with gemini-embedding-2: off-topic questions score <= 0.58, relevant ones >= 0.67.
    retrieval_score_threshold: float = Field(default=0.62, ge=0, le=1)
    # hybrid = vector + BM25 keyword search fused with Reciprocal Rank Fusion; vector = embeddings only.
    retrieval_mode: Literal["hybrid", "vector"] = "hybrid"
    retrieval_candidates: int = Field(default=10, ge=1, le=50)  # results taken from each search before fusion
    retrieval_rrf_k: int = Field(default=60, ge=1)  # RRF constant: score = sum of 1 / (k + rank)

    # Ingestion chunking
    chunk_max_chars: int = Field(default=1500, ge=200)
    chunk_overlap_chars: int = Field(default=200, ge=0)

    # Geocoding (Nominatim usage policy requires an identifying User-Agent)
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    nominatim_user_agent: str = "skoda-support-agent/0.1"
    nominatim_timeout_s: float = Field(default=10, gt=0)

    # LangSmith (optional). Traces leave this machine, so they are phone-masked (app/observability.py).
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "skoda-agent"
    langsmith_endpoint: str | None = None  # blank = LangSmith's default (US); e.g. https://eu.api.smith.langchain.com

    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"

    @property
    def seed_dir(self) -> Path:
        return self.data_dir / "seed"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def conversations_dir(self) -> Path:
        return self.data_dir / "conversations"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def leads_file(self) -> Path:
        return self.data_dir / "leads.csv"

    @property
    def tickets_file(self) -> Path:
        return self.data_dir / "tickets.csv"

    def model_for(self, node: str) -> str:
        """Gemini model for a graph node: <NODE>_MODEL, else DEFAULT_MODEL. A newly added agent without its own
        field here simply uses DEFAULT_MODEL (add `<name>_model: str | None = None` to give it its own)."""
        return getattr(self, f"{node}_model", None) or self.default_model

    def thinking_budget_for(self, node: str) -> int:
        """<NODE>_THINKING_BUDGET, else the default for that node (agents not listed get the agent default)."""
        value = getattr(self, f"{node}_thinking_budget", None)
        if value is not None:
            return value
        return THINKING_BUDGET_DEFAULTS.get(node, AGENT_THINKING_BUDGET_DEFAULT)


# Nodes that only classify or extract don't think; agents get a small budget for tool choice.
# The composer only rewrites findings that already hold every figure, so 512 is enough: at 2048 it was the
# slowest and most expensive call (up to ~1,700 thinking tokens and ~11 s on multi-part replies).
AGENT_THINKING_BUDGET_DEFAULT = 512
THINKING_BUDGET_DEFAULTS = {
    "supervisor": 0, "clarify": 0, "summarize": 0,
    "car_info": 512, "quotation": 512, "locator": 512, "service": 512, "lead": 512,
    "composer": 512,
}


@lru_cache
def get_settings() -> Settings:
    return Settings()
