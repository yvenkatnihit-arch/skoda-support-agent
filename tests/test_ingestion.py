import hashlib
import json
import math
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app.tools.retrieval import (
    GENERAL_MODEL,
    IndexMismatchError,
    IndexNotBuiltError,
    VectorStore,
    document_input,
    query_input,
)
from ingestion.build_index import build_index
from ingestion.chunker import chunk_document, parse_blocks
from ingestion.loaders import IngestionError, RawDocument, detect_model, load_documents

ROOT = Path(__file__).resolve().parent.parent
MODEL_IDS = {"kylaq", "kushaq", "slavia", "kodiaq", "octavia-rs", "enyaq"}
DIM = 64


class FakeEmbeddings(Embeddings):
    """Offline stand-in: bag-of-words hashed into DIM buckets, so shared words -> higher similarity."""

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _doc(text: str, model="kushaq", doc_type="brochure") -> RawDocument:
    return RawDocument(source=f"{doc_type}/{model}-x.md", doc_type=doc_type, model=model, pages=[(None, text)])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    _write(raw / "brochure" / "kushaq-brochure.md",
           "# Kushaq\n\n## Safety\n\nThe Kushaq has six airbags and electronic stability control.\n\n"
           "## Comfort\n\nPanoramic sunroof and ventilated seats.")
    _write(raw / "brochure" / "kylaq-brochure.md",
           "# Kylaq\n\n## Safety\n\nThe Kylaq has six airbags and a five star rating.")
    _write(raw / "warranty" / "skoda-warranty.md",
           "# Warranty\n\nStandard warranty is four years or one lakh kilometres.")
    _write(raw / "README.md", "not a document")
    return raw


@pytest.fixture
def index_dir(tmp_path: Path, raw_dir: Path) -> Path:
    out = tmp_path / "index"
    build_index(raw_dir, out, FakeEmbeddings(), "fake-model", DIM, MODEL_IDS, max_chars=400, overlap_chars=50)
    return out


def _store(index_dir: Path) -> VectorStore:
    return VectorStore.load(index_dir, FakeEmbeddings(), "fake-model", DIM)


# ---- loaders -------------------------------------------------------------------

def test_detect_model_uses_longest_model_prefix():
    assert detect_model("octavia-rs-brochure", MODEL_IDS) == "octavia-rs"
    assert detect_model("Kushaq_specs", MODEL_IDS) == "kushaq"
    assert detect_model("skoda-warranty", MODEL_IDS) == GENERAL_MODEL


def test_load_documents_tags_type_and_model_and_skips_root_readme(raw_dir):
    docs = {d.source: d for d in load_documents(raw_dir, MODEL_IDS)}
    assert set(docs) == {"brochure/kushaq-brochure.md", "brochure/kylaq-brochure.md", "warranty/skoda-warranty.md"}
    assert docs["brochure/kushaq-brochure.md"].model == "kushaq"
    assert docs["warranty/skoda-warranty.md"].model == GENERAL_MODEL
    assert docs["warranty/skoda-warranty.md"].doc_type == "warranty"


def test_file_in_unknown_folder_is_rejected(raw_dir):
    _write(raw_dir / "misc" / "notes.md", "text")
    with pytest.raises(IngestionError, match="misc/notes.md"):
        load_documents(raw_dir, MODEL_IDS)


def test_unsupported_file_type_is_rejected(raw_dir):
    _write(raw_dir / "brochure" / "kylaq.docx", "binary")
    with pytest.raises(IngestionError, match="unsupported file type"):
        load_documents(raw_dir, MODEL_IDS)


def test_markdown_comments_are_not_indexed(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "faq" / "faq.md", "# FAQ\n\n<!-- SAMPLE source note\nspanning lines -->\n\nReal answer.")
    [doc] = load_documents(raw, MODEL_IDS)
    chunks = chunk_document(doc, 500, 0)
    assert len(chunks) == 1 and "SAMPLE" not in chunks[0].text and "Real answer." in chunks[0].text


def test_html_loader_strips_scripts_and_keeps_headings_and_tables(tmp_path):
    raw = tmp_path / "raw"
    _write(raw / "faq" / "faq.html",
           "<html><head><script>track()</script></head><body><nav>Menu</nav>"
           "<h2>Warranty</h2><p>Four years cover.</p>"
           "<table><tr><th>Item</th><th>Cover</th></tr><tr><td>Parts</td><td>2 years</td></tr></table>"
           "</body></html>")
    [doc] = load_documents(raw, MODEL_IDS)
    text = doc.pages[0][1]
    assert "track()" not in text and "Menu" not in text
    assert "## Warranty" in text
    assert "| Item | Cover |\n| Parts | 2 years |" in text


# ---- chunker -------------------------------------------------------------------

def test_tables_are_never_split_even_when_long():
    rows = "\n".join(f"| row {i} | value {i} |" for i in range(60))  # far longer than max_chars
    chunks = chunk_document(_doc(f"# Specs\n\n{rows}"), max_chars=200, overlap_chars=20)
    tables = [c for c in chunks if "| row 0 |" in c.text]
    assert len(tables) == 1 and "| row 59 |" in tables[0].text


def test_chunks_do_not_cross_headings_and_carry_heading_path():
    text = "# Kushaq\n\n## Safety\n\nSix airbags.\n\n## Comfort\n\nVentilated seats."
    chunks = chunk_document(_doc(text), max_chars=500, overlap_chars=0)
    assert [c.section for c in chunks] == ["Kushaq > Safety", "Kushaq > Comfort"]
    assert chunks[0].text.startswith("Kushaq > Safety\n\n") and "Ventilated" not in chunks[0].text


def test_long_section_is_split_with_overlap():
    sentences = " ".join(f"Sentence number {i} about the car." for i in range(40))
    chunks = chunk_document(_doc(f"# Kylaq\n\n{sentences}"), max_chars=300, overlap_chars=60)
    assert len(chunks) > 1
    body = [c.text.split("\n\n", 1)[1] for c in chunks]
    assert all(len(b) <= 300 + 60 + 2 for b in body)
    # the start of each chunk repeats the end of the previous one
    first_words = body[1].split(".")[0]
    assert first_words in body[0]


def test_parse_blocks_recognises_headings_tables_and_lists():
    kinds = [b.kind for b in parse_blocks("# T\n\nPara one\ncontinues.\n\n- item\n\n| a | b |\n| 1 | 2 |")]
    assert kinds == ["heading", "text", "text", "table"]


def test_sample_documents_load_and_chunk():
    docs = load_documents(ROOT / "data" / "raw", MODEL_IDS)
    by_source = {d.source: d for d in docs}
    assert by_source["brochure/octavia-rs-brochure-sample.md"].model == "octavia-rs"
    assert by_source["warranty/skoda-warranty-sample.md"].model == GENERAL_MODEL
    chunks = [c for d in docs for c in chunk_document(d, 1500, 200)]
    kushaq_table = [c for c in chunks if c.source == "specs/kushaq-specs-sample.md" and "| 1.5 TSI DSG" in c.text]
    assert len(kushaq_table) == 1 and "| 1.0 TSI MT" in kushaq_table[0].text  # whole table in one chunk


# ---- embedding input format ----------------------------------------------------

def test_gemini_embedding_2_gets_task_prefixes():
    chunk = _store_chunk()
    assert query_input("sunroof?", "gemini-embedding-2") == "task: search result | query: sunroof?"
    assert document_input(chunk, "gemini-embedding-2").startswith("title: Kushaq > Safety | text: ")
    assert query_input("sunroof?", "gemini-embedding-001") == "sunroof?"
    assert document_input(chunk, "gemini-embedding-001") == chunk.text


def _store_chunk():
    from app.tools.retrieval import Chunk
    return Chunk(id=0, text="Six airbags", model="kushaq", doc_type="brochure", source="b.md", section="Kushaq > Safety")


# ---- index build, load and search ----------------------------------------------

def test_build_writes_three_files_and_manifest(index_dir):
    assert {p.name for p in index_dir.iterdir()} == {"index.faiss", "chunks.json", "manifest.json"}
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == "fake-model" and manifest["dimension"] == DIM
    assert manifest["chunk_count"] == len(json.loads((index_dir / "chunks.json").read_text(encoding="utf-8")))


def test_search_returns_most_similar_chunk_first(index_dir):
    hits = _store(index_dir).search("Kushaq airbags stability control", k=3, mode="vector")
    assert hits[0].chunk.model == "kushaq" and "airbags" in hits[0].chunk.text
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_repeated_query_is_embedded_once(index_dir):
    class Counting(FakeEmbeddings):
        calls = 0

        def embed_query(self, text):
            Counting.calls += 1
            return super().embed_query(text)

    store = VectorStore.load(index_dir, Counting(), "fake-model", DIM)
    first = store.search("Kushaq airbags", k=3)
    second = store.search("  kushaq   AIRBAGS ", k=3, models=["kushaq"])  # same query, other filters
    assert Counting.calls == 1
    assert second and second[0].chunk.id == first[0].chunk.id
    store.search("Kylaq sunroof", k=3)
    assert Counting.calls == 2


def test_model_filter_excludes_other_models(index_dir):
    hits = _store(index_dir).search("six airbags", k=10, models=["kylaq", GENERAL_MODEL])
    assert hits and {h.chunk.model for h in hits} <= {"kylaq", GENERAL_MODEL}


def test_doc_type_filter(index_dir):
    hits = _store(index_dir).search("warranty years", k=10, doc_types=["warranty"])
    assert hits and {h.chunk.doc_type for h in hits} == {"warranty"}


def test_min_score_drops_weak_matches(index_dir):
    store = _store(index_dir)
    assert store.search("cricket match tomorrow", k=5, min_score=0.5) == []
    assert store.search("cricket match tomorrow", k=5, min_score=0.0)  # without a threshold something comes back


def test_filter_with_no_matching_chunks_returns_empty(index_dir):
    assert _store(index_dir).search("airbags", k=5, models=["enyaq"]) == []


def test_load_refuses_a_different_embedding_model(index_dir):
    with pytest.raises(IndexMismatchError, match="rebuild|build_index"):
        VectorStore.load(index_dir, FakeEmbeddings(), "gemini-embedding-2", DIM)
    with pytest.raises(IndexMismatchError):
        VectorStore.load(index_dir, FakeEmbeddings(), "fake-model", 768)


def test_missing_index_says_how_to_build_it(tmp_path):
    with pytest.raises(IndexNotBuiltError, match="python -m ingestion.build_index"):
        VectorStore.load(tmp_path / "nothing", FakeEmbeddings(), "fake-model", DIM)
