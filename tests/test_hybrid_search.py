"""Hybrid search: BM25 keyword search + vector search merged with Reciprocal Rank Fusion."""

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings

from app.tools.keyword_search import BM25Index, expand_query, tokenize
from app.tools.retrieval import Chunk, VectorStore, save_index, to_unit_vectors

TEXTS = [
    "Kushaq > Safety\n\nSix airbags, ESC and ISOFIX child seat mounts.",
    "Kushaq > Comfort\n\nVentilated front seats and an electric sunroof.",
    "Kushaq > Infotainment\n\n10-inch touchscreen with wireless Android Auto and Apple CarPlay.",
]
# Hand-made 4-d vectors, so we control what the "meaning" search finds.
DOC_VECTORS = [[1, 0, 0, 0], [0.8, 0.6, 0, 0], [0, 0, 1, 0]]


class FixedEmbeddings(Embeddings):
    def __init__(self, queries: dict[str, list[float]]):
        self.queries = queries

    def embed_documents(self, texts):
        raise AssertionError("not used")

    def embed_query(self, text):
        return self.queries[text]


@pytest.fixture
def make_store(tmp_path):
    chunks = [Chunk(id=i, text=t, model="kushaq", doc_type="brochure", source="brochure/kushaq.md",
                    section=t.split("\n")[0]) for i, t in enumerate(TEXTS)]
    save_index(tmp_path, to_unit_vectors(DOC_VECTORS, 4), chunks, "fake-model", [])

    def make(queries):
        return VectorStore.load(tmp_path, FixedEmbeddings(queries), "fake-model", 4)
    return make


# ---- keyword search --------------------------------------------------------------

def test_tokenize_folds_accents_plurals_and_drops_stopwords():
    assert tokenize("Does the Škoda Kushaq have airbags?") == ["kushaq", "airbag"]
    assert tokenize("Slavia 1.5 TSI power") == ["slavia", "1.5", "tsi", "power"]


def test_synonyms_are_added_at_half_weight():
    terms = expand_query("boot space")
    assert terms["boot"] == 1.0 and terms["space"] == 1.0
    assert terms["luggage"] == 0.5 and terms["cargo"] == 0.5


def test_bm25_ranks_the_chunk_with_the_exact_term_and_skips_non_matches():
    bm25 = BM25Index(TEXTS)
    assert bm25.search("ISOFIX", [0, 1, 2], k=5) == [(0, pytest.approx(bm25.score(0, {"isofix": 1.0})))]
    assert [i for i, _ in bm25.search("android auto carplay", [0, 1, 2], k=5)] == [2]
    assert bm25.search("ISOFIX", [1, 2], k=5) == []  # respects the model / doc-type filter


# ---- fusion ----------------------------------------------------------------------

def test_rrf_puts_the_chunk_both_searches_agree_on_first(make_store):
    # Meaning alone ranks Comfort (0.96) above Safety (0.6); only Safety contains "ISOFIX".
    store = make_store({"ISOFIX": [0.6, 0.8, 0, 0]})
    vector = store.search("ISOFIX", k=2, min_score=0.5, mode="vector")
    assert [h.chunk.id for h in vector] == [1, 0]
    hybrid = store.search("ISOFIX", k=2, min_score=0.5, mode="hybrid", rrf_k=60)
    assert [h.chunk.id for h in hybrid] == [0, 1]
    top = hybrid[0]
    assert (top.vector_rank, top.keyword_rank) == (2, 1)
    assert top.rrf == pytest.approx(1 / 62 + 1 / 61, abs=1e-5)
    assert top.score == pytest.approx(0.6, abs=1e-5)  # score stays the cosine similarity


def test_keyword_only_match_is_added_with_its_cosine(make_store):
    store = make_store({"android auto": [1, 0, 0, 0]})  # meaning points at Safety/Comfort, not Infotainment
    hits = store.search("android auto", k=3, min_score=0.5, mode="hybrid")
    infotainment = next(h for h in hits if h.chunk.id == 2)
    assert infotainment.vector_rank is None and infotainment.keyword_rank == 1
    assert infotainment.score == pytest.approx(0.0, abs=1e-6)


def test_off_topic_question_returns_nothing_even_if_words_match(make_store):
    store = make_store({"airbags in cricket": [0, 0, 0, 1]})  # no chunk close in meaning
    assert store.search("airbags in cricket", k=3, min_score=0.5, mode="hybrid") == []


def test_vector_mode_is_unchanged(make_store):
    store = make_store({"q": [1, 0, 0, 0]})
    hits = store.search("q", k=3, min_score=0.5, mode="vector")
    assert [h.chunk.id for h in hits] == [0, 1]
    assert all(h.keyword_rank is None and h.rrf is None for h in hits)
    assert np.isclose(hits[0].score, 1.0)
