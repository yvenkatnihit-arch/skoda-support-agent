"""Rebuild the FAISS index from everything in data/raw/.

    python -m ingestion.build_index

Run it whenever you add, replace or delete documents, or change EMBEDDING_MODEL / EMBEDDING_DIM.
The index is always rebuilt from scratch.
"""

import hashlib
import sys
from collections import Counter
from pathlib import Path

import structlog
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.data.repository import get_repository
from app.log import configure_logging
from app.tools.retrieval import Chunk, document_input, save_index, to_unit_vectors
from ingestion.chunker import chunk_document
from ingestion.loaders import IngestionError, load_documents

log = structlog.get_logger()


def build_index(
    raw_dir: Path,
    index_dir: Path,
    embeddings: Embeddings,
    embedding_model: str,
    dim: int,
    model_ids: set[str],
    max_chars: int,
    overlap_chars: int,
) -> dict:
    docs = load_documents(raw_dir, model_ids)
    chunks: list[Chunk] = []
    per_source: Counter[str] = Counter()
    for doc in docs:
        for draft in chunk_document(doc, max_chars, overlap_chars):
            chunks.append(Chunk(id=len(chunks), text=draft.text, model=draft.model, doc_type=draft.doc_type,
                                source=draft.source, page=draft.page, section=draft.section))
            per_source[doc.source] += 1
    if not chunks:
        raise IngestionError("documents were loaded but produced no chunks")

    log.info("embedding chunks", chunks=len(chunks), model=embedding_model)
    vectors = embeddings.embed_documents([document_input(c, embedding_model) for c in chunks])
    if len(vectors) != len(chunks):
        raise IngestionError(f"embedding returned {len(vectors)} vectors for {len(chunks)} chunks")
    unit = to_unit_vectors(vectors, dim)

    sources = []
    for doc in docs:
        digest = hashlib.sha256((raw_dir / doc.source).read_bytes()).hexdigest()
        sources.append({"file": doc.source, "doc_type": doc.doc_type, "model": doc.model,
                        "chunks": per_source[doc.source], "sha256": digest})
    return save_index(index_dir, unit, chunks, embedding_model, sources)


def main() -> int:
    from app.llm import get_embeddings

    s = get_settings()
    configure_logging(s.log_level)
    try:
        manifest = build_index(
            raw_dir=s.raw_dir,
            index_dir=s.index_dir,
            embeddings=get_embeddings(s),
            embedding_model=s.embedding_model,
            dim=s.embedding_dim,
            model_ids={m.id for m in get_repository().list_models()},
            max_chars=s.chunk_max_chars,
            overlap_chars=s.chunk_overlap_chars,
        )
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"Built index: {manifest['chunk_count']} chunks, {manifest['embedding_model']} "
          f"({manifest['dimension']} dims) -> {s.index_dir}")
    for src in manifest["sources"]:
        print(f"  {src['chunks']:3} chunks  {src['doc_type']:9} {src['model']:11} {src['file']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
