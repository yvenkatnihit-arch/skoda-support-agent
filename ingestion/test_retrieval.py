"""Try the document search by hand and see which chunks come back.

    python -m ingestion.test_retrieval                      # run the preset questions (RETRIEVAL_MODE)
    python -m ingestion.test_retrieval --mode vector        # same questions, embeddings only, to compare
    python -m ingestion.test_retrieval "Does the Slavia have a sunroof?" --model slavia
    python -m ingestion.test_retrieval "warranty" --doc-type warranty --doc-type service
    python -m ingestion.test_retrieval "Who will win the IPL?" --all   # show scores below the threshold too

This is a manual check script, not a pytest test.
"""

import argparse

from app.config import get_settings
from app.tools.retrieval import DOC_TYPES, GENERAL_MODEL, get_vector_store

# (question, models filter, doc_types filter)
PRESETS = [
    ("What safety features does the Kushaq have?", ["kushaq", GENERAL_MODEL], ["brochure", "specs"]),
    ("Does the Kylaq have a sunroof?", ["kylaq", GENERAL_MODEL], ["brochure", "specs"]),
    ("Kodiaq seating capacity", ["kodiaq", GENERAL_MODEL], ["brochure", "specs"]),
    ("Slavia 1.5 TSI power and torque", ["slavia", GENERAL_MODEL], ["brochure", "specs"]),
    ("How long is the standard warranty?", None, ["warranty", "service"]),
    ("How often should I service my car?", None, ["service", "warranty"]),
    ("What is roadside assistance?", None, ["warranty", "service", "faq"]),
    ("Is there an electric Skoda in India?", None, None),
    ("Who will win the IPL?", None, None),  # off-topic: should score low
]


def show(question: str, models, doc_types, k: int, min_score: float, threshold: float, mode: str) -> None:
    s = get_settings()
    store = get_vector_store()
    hits = store.search(question, k, models=models, doc_types=doc_types, min_score=min_score, mode=mode,
                        candidates=s.retrieval_candidates, rrf_k=s.retrieval_rrf_k)
    filters = f"models={models or 'any'} doc_types={doc_types or 'any'}"
    print(f"\nQ: {question}\n   [{filters}]")
    if not hits:
        print("   (no hits)")
    for hit in hits:
        c = hit.chunk
        flag = "" if hit.score >= threshold else "  <- below threshold"
        page = f" p.{c.page}" if c.page else ""
        snippet = c.text.replace("\n", " ")[:110]
        ranks = f"  vector #{hit.vector_rank or '-'} keyword #{hit.keyword_rank or '-'}" if mode == "hybrid" else ""
        print(f"   {hit.score:.3f}  {c.model:10} {c.doc_type:9} {c.source}{page}{flag}{ranks}")
        print(f"          {snippet}")


def main() -> None:
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", nargs="*")
    parser.add_argument("--model", action="append", help="model id filter (repeatable); 'all' chunks are always included")
    parser.add_argument("--doc-type", action="append", choices=DOC_TYPES)
    parser.add_argument("-k", type=int, default=s.retrieval_top_k)
    parser.add_argument("--mode", choices=["hybrid", "vector"], default=s.retrieval_mode)
    parser.add_argument("--all", action="store_true", help="also show hits below RETRIEVAL_SCORE_THRESHOLD")
    args = parser.parse_args()

    min_score = 0.0 if args.all else s.retrieval_score_threshold
    store = get_vector_store()
    print(f"Index: {len(store.chunks)} chunks, {store.embedding_model}, mode {args.mode}, "
          f"threshold {s.retrieval_score_threshold}, k={args.k}")

    if args.questions:
        models = [*args.model, GENERAL_MODEL] if args.model else None
        for q in args.questions:
            show(q, models, args.doc_type, args.k, min_score, s.retrieval_score_threshold, args.mode)
    else:
        for q, models, doc_types in PRESETS:
            show(q, models, doc_types, 3, 0.0 if args.all else min_score, s.retrieval_score_threshold, args.mode)


if __name__ == "__main__":
    main()
