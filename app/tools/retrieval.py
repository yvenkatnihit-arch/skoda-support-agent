"""FAISS vector store over the ingested documents.

The index lives in data/index/ as three files written by ingestion/build_index.py:
  index.faiss    vectors only (row i = chunk i)
  chunks.json    text + metadata for each chunk, in the same order
  manifest.json  embedding model, dimension and sources used to build the index

Phase 3 adds the search_brochures and search_service_docs tools on top of VectorStore.
"""

import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import faiss
import numpy as np
import structlog
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.observability import current_stats
from app.tools.keyword_search import BM25Index

log = structlog.get_logger()

QUERY_CACHE_SIZE = 512  # query vectors kept in memory; repeated searches skip the embedding call
# Hybrid: keyword hits below half the best keyword score are ignored. On the sample docs 0.3 still let in
# single-word matches ("auto" in a test-drive FAQ for "Android Auto"); 0.5 keeps the real matches.
KEYWORD_MIN_RELATIVE_SCORE = 0.5

DocType = Literal["brochure", "specs", "faq", "service", "warranty"]
DOC_TYPES: tuple[str, ...] = ("brochure", "specs", "faq", "service", "warranty")
GENERAL_MODEL = "all"  # chunk applies to every model (e.g. warranty terms)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class IndexNotBuiltError(RuntimeError):
    pass


class IndexMismatchError(RuntimeError):
    pass


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    text: str = Field(min_length=1)
    model: str  # model id from models.json, or "all"
    doc_type: DocType
    source: str  # path relative to data/raw/
    page: int | None = None
    section: str = ""


@dataclass(frozen=True)
class SearchHit:
    score: float  # cosine similarity, higher is closer
    chunk: Chunk
    vector_rank: int | None = None  # 1-based rank in each search; None = that search didn't find it
    keyword_rank: int | None = None
    rrf: float | None = None  # fused Reciprocal Rank Fusion score (hybrid only)


# ---- embedding input format ----------------------------------------------------
# gemini-embedding-2 has no task_type parameter; it expects these prefixes instead.
# Older models (gemini-embedding-001) use task_type, which the LangChain wrapper sets.

def uses_task_prefixes(embedding_model: str) -> bool:
    return embedding_model.removeprefix("models/").startswith("gemini-embedding-2")


def document_input(chunk: Chunk, embedding_model: str) -> str:
    if uses_task_prefixes(embedding_model):
        return f"title: {chunk.section or 'none'} | text: {chunk.text}"
    return chunk.text


def query_input(query: str, embedding_model: str) -> str:
    if uses_task_prefixes(embedding_model):
        return f"task: search result | query: {query}"
    return query


def to_unit_vectors(vectors: list[list[float]], dim: int) -> np.ndarray:
    """Stack vectors as float32 and L2-normalise them, so inner product = cosine similarity."""
    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"expected vectors of dimension {dim}, got shape {arr.shape}")
    faiss.normalize_L2(arr)
    return arr


# ---- saving --------------------------------------------------------------------

def save_index(index_dir: Path, vectors: np.ndarray, chunks: list[Chunk], embedding_model: str,
               sources: list[dict]) -> dict:
    """Write index, chunks and manifest. Files are written to temp names first, then swapped in."""
    if len(chunks) != vectors.shape[0]:
        raise ValueError(f"{len(chunks)} chunks but {vectors.shape[0]} vectors")
    if any(c.id != i for i, c in enumerate(chunks)):
        raise ValueError("chunk ids must equal their position (0..n-1)")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    manifest = {
        "embedding_model": embedding_model,
        "dimension": int(vectors.shape[1]),
        "chunk_count": len(chunks),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": sources,
    }

    index_dir.mkdir(parents=True, exist_ok=True)
    tmp = {name: index_dir / f".{name}.tmp" for name in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE)}
    faiss.write_index(index, str(tmp[INDEX_FILE]))
    tmp[CHUNKS_FILE].write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=1), encoding="utf-8")
    tmp[MANIFEST_FILE].write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for name, path in tmp.items():
        os.replace(path, index_dir / name)
    return manifest


# ---- loading and search --------------------------------------------------------

class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: dict, embeddings: Embeddings):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest
        self.embeddings = embeddings
        self.embedding_model = manifest["embedding_model"]
        self.dimension = manifest["dimension"]
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_lock = threading.Lock()
        self.keyword_index = BM25Index([c.text for c in chunks])  # in memory, built from chunks.json

    @classmethod
    def load(cls, index_dir: Path, embeddings: Embeddings, expected_model: str,
             expected_dim: int) -> "VectorStore":
        """Load the index and refuse to start if it was built with a different embedding setup."""
        rebuild = "Run: python -m ingestion.build_index"
        paths = {name: index_dir / name for name in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE)}
        missing = [name for name, p in paths.items() if not p.exists()]
        if missing:
            raise IndexNotBuiltError(f"FAISS index not found in {index_dir} (missing {', '.join(missing)}). {rebuild}")

        manifest = json.loads(paths[MANIFEST_FILE].read_text(encoding="utf-8"))
        if manifest.get("embedding_model") != expected_model or manifest.get("dimension") != expected_dim:
            raise IndexMismatchError(
                f"Index was built with {manifest.get('embedding_model')} ({manifest.get('dimension')} dims) "
                f"but .env is set to {expected_model} ({expected_dim} dims). {rebuild}")

        index = faiss.read_index(str(paths[INDEX_FILE]))
        chunks = [Chunk.model_validate(c) for c in json.loads(paths[CHUNKS_FILE].read_text(encoding="utf-8"))]
        if index.ntotal != len(chunks) or index.d != expected_dim:
            raise IndexMismatchError(
                f"index.faiss has {index.ntotal} vectors of dim {index.d}, chunks.json has {len(chunks)} chunks. {rebuild}")
        if any(c.id != i for i, c in enumerate(chunks)):
            raise IndexMismatchError(f"chunks.json ids are out of order. {rebuild}")
        return cls(index, chunks, manifest, embeddings)

    def query_vector(self, query: str) -> np.ndarray:
        """Unit vector for a query, from the in-memory cache when the same query was embedded before.

        The key ignores case and extra spaces. The query text itself is not logged (it may be customer text).
        """
        key = " ".join(query.casefold().split())
        with self._query_cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                log.info("query embedding", cached=True)
                if (stats := current_stats()) is not None:
                    stats.add_embedding(cached=True)
                return cached
        start = time.perf_counter()
        vector = to_unit_vectors([self.embeddings.embed_query(query_input(query, self.embedding_model))],
                                 self.dimension)
        log.info("query embedding", cached=False, model=self.embedding_model,
                 elapsed_s=round(time.perf_counter() - start, 2))
        if (stats := current_stats()) is not None:
            stats.add_embedding(cached=False)
        with self._query_cache_lock:
            self._query_cache[key] = vector
            if len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector

    def _vector_hits(self, vector: np.ndarray, allowed: list[int], n: int, min_score: float) -> list[SearchHit]:
        """Top-n allowed chunks by cosine similarity. The filter runs inside FAISS (IDSelectorBatch)."""
        selector = faiss.IDSelectorBatch(np.asarray(allowed, dtype="int64"))
        scores, ids = self.index.search(vector, min(n, len(allowed)), params=faiss.SearchParameters(sel=selector))
        return [SearchHit(float(score), self.chunks[int(i)], vector_rank=rank)
                for rank, (score, i) in enumerate(zip(scores[0], ids[0]), 1) if i != -1 and score >= min_score]

    def search(
        self,
        query: str,
        k: int,
        *,
        models: list[str] | None = None,
        doc_types: list[str] | None = None,
        min_score: float = 0.0,
        mode: Literal["hybrid", "vector"] = "hybrid",
        candidates: int = 10,
        rrf_k: int = 60,
    ) -> list[SearchHit]:
        """Top-k chunks for a query, restricted to the given models / doc types.

        vector: chunks by cosine similarity; those below min_score are dropped.
        hybrid: the `candidates` best vector hits (>= min_score) and the `candidates` best BM25 keyword hits are
            merged with Reciprocal Rank Fusion: each chunk scores sum(1 / (rrf_k + rank)) over the lists it is in,
            so a chunk both searches rank high comes first. If no chunk reaches min_score in meaning, the
            question is off topic and nothing is returned, even if some words match.
        """
        if not query.strip():
            raise ValueError("query must not be empty")
        if k < 1:
            raise ValueError("k must be at least 1")
        allowed = [c.id for c in self.chunks
                   if (models is None or c.model in models)
                   and (doc_types is None or c.doc_type in doc_types)]
        if not allowed:
            return []

        vector = self.query_vector(query)
        if mode == "vector":
            return self._vector_hits(vector, allowed, k, min_score)

        vector_hits = self._vector_hits(vector, allowed, max(k, candidates), min_score)
        if not vector_hits:
            log.info("search", mode=mode, vector_hits=0, keyword_hits=0, returned=0)
            return []
        keyword = self.keyword_index.search(query, allowed, max(k, candidates))
        if keyword:  # drop words-only matches far weaker than the best one (e.g. a lone "price")
            floor = keyword[0][1] * KEYWORD_MIN_RELATIVE_SCORE
            keyword = [(i, s) for i, s in keyword if s >= floor]

        vector_rank = {h.chunk.id: h.vector_rank for h in vector_hits}
        keyword_rank = {i: rank for rank, (i, _) in enumerate(keyword, 1)}
        cosine = {h.chunk.id: h.score for h in vector_hits}
        for i in keyword_rank.keys() - cosine.keys():  # found by keywords only: still report its cosine
            cosine[i] = float(self.index.reconstruct(i) @ vector[0])

        def rrf(i: int) -> float:
            return sum(1 / (rrf_k + ranks[i]) for ranks in (vector_rank, keyword_rank) if i in ranks)

        fused = sorted(vector_rank.keys() | keyword_rank.keys(), key=lambda i: (-rrf(i), -cosine[i]))[:k]
        log.info("search", mode=mode, vector_hits=len(vector_rank), keyword_hits=len(keyword_rank),
                 in_both=len(vector_rank.keys() & keyword_rank.keys()), returned=len(fused),
                 keyword_only_returned=sum(i not in vector_rank for i in fused))
        return [SearchHit(cosine[i], self.chunks[i], vector_rank.get(i), keyword_rank.get(i), round(rrf(i), 5))
                for i in fused]


@lru_cache
def get_vector_store() -> VectorStore:
    from app.llm import get_embeddings

    s = get_settings()
    return VectorStore.load(s.index_dir, get_embeddings(s), s.embedding_model, s.embedding_dim)


# ---- tools ---------------------------------------------------------------------

NO_RESULTS = ("NO_RESULTS: nothing in the documents matches this question well enough. Tell the customer you "
              "don't have that information and suggest the Skoda website or a dealer. Do not guess.")


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, description=(
        "A short standalone search query (rewritten, not copied): replace 'it', 'its', 'that one' with what "
        "they refer to in the conversation, keep exact terms (ISOFIX, TPMS, 1.5 TSI, Android Auto) and add the "
        "common synonym, e.g. 'what about its boot?' -> 'boot space luggage capacity litres'. "
        "One topic per query; leave the model name out (pass it in model)."))
    model: str | None = Field(default=None, description="Skoda model the question is about, e.g. 'Kushaq'. "
                                                        "If the customer says 'it', use the model from the "
                                                        "conversation or known details. Leave empty only for "
                                                        "general questions.")


def _search(query: str, model: str | None, doc_types: list[str]) -> str:
    from app.tools.common import resolve_model, to_json

    models = [resolve_model(model).id, GENERAL_MODEL] if model else None
    s = get_settings()
    hits = get_vector_store().search(query, s.retrieval_top_k, models=models, doc_types=doc_types,
                                     min_score=s.retrieval_score_threshold, mode=s.retrieval_mode,
                                     candidates=s.retrieval_candidates, rrf_k=s.retrieval_rrf_k)
    if not hits:
        return NO_RESULTS
    return to_json({
        "instruction": "Answer only from these passages and cite the source. If they don't contain the answer, "
                       "say you don't have that information.",
        "results": [{"score": round(h.score, 3), "model": h.chunk.model, "doc_type": h.chunk.doc_type,
                     "source": h.chunk.source, "page": h.chunk.page, "section": h.chunk.section,
                     "text": h.chunk.text} for h in hits],
    })


def _search_tool(name: str, doc_types: list[str], description: str):
    from langchain_core.tools import tool

    from app.tools.common import returns_errors_as_text

    @returns_errors_as_text
    def run(query: str, model: str | None = None) -> str:
        try:
            return _search(query, model, doc_types)
        except (IndexNotBuiltError, IndexMismatchError) as exc:
            return f"ERROR: document search is unavailable ({exc}). Offer the customer a callback."

    run.__name__ = name
    return tool(name, args_schema=SearchArgs, description=description)(run)


search_brochures = _search_tool(
    "search_brochures", ["brochure", "specs", "faq"],
    "Search Skoda brochures, spec sheets and FAQs for features, specs, safety, variants, colours and "
    "comparisons. Pass the model to keep results about that car. Returns passages with sources.")

search_service_docs = _search_tool(
    "search_service_docs", ["service", "warranty", "faq"],
    "Search Skoda service, warranty, extended warranty, roadside assistance and service-booking documents. "
    "Returns passages with sources.")
