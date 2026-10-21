"""Load raw documents from data/raw/.

Folder convention: data/raw/<doc_type>/<file>, where doc_type is brochure, specs, faq, service or warranty.
The model is read from the start of the file name (e.g. kushaq-brochure.pdf -> kushaq); files that
don't start with a model id apply to every model ("all").

Every loader returns text in a light Markdown form ('#' headings, '|' table rows) so the chunker
can treat all formats the same way.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.tools.retrieval import DOC_TYPES, GENERAL_MODEL

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES
IGNORED_ROOT_FILES = {"README.md", ".gitkeep"}


class IngestionError(Exception):
    pass


@dataclass
class RawDocument:
    source: str  # path relative to the raw dir, with forward slashes
    doc_type: str
    model: str
    pages: list[tuple[int | None, str]] = field(default_factory=list)  # (page number or None, text)


def detect_model(stem: str, model_ids: set[str]) -> str:
    """Longest leading run of '-'-separated tokens that is a model id (octavia-rs-brochure -> octavia-rs)."""
    tokens = stem.lower().replace("_", "-").split("-")
    for n in range(len(tokens), 0, -1):
        candidate = "-".join(tokens[:n])
        if candidate in model_ids:
            return candidate
    return GENERAL_MODEL


COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def load_text(path: Path) -> list[tuple[int | None, str]]:
    """Markdown/text. <!-- comments --> (e.g. source notes) are not indexed."""
    return [(None, COMMENT_RE.sub("", path.read_text(encoding="utf-8")))]


def load_pdf(path: Path) -> list[tuple[int | None, str]]:
    reader = PdfReader(path)
    pages = [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    if not any(text.strip() for _, text in pages):
        raise IngestionError(f"{path.name}: no extractable text (is it a scanned PDF?)")
    return pages


def load_html(path: Path) -> list[tuple[int | None, str]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "svg"]):
        tag.decompose()
    root = soup.body or soup
    blocks: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.name != "table" and el.find_parent("table"):
            continue  # table cells are emitted with their table
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                blocks.append("\n".join(rows))
            continue
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            blocks.append("#" * int(el.name[1]) + " " + text)
        elif el.name == "li":
            blocks.append("- " + text)
        else:
            blocks.append(text)
    if not blocks:
        raise IngestionError(f"{path.name}: no text found in HTML")
    return [(None, "\n\n".join(blocks))]


def load_document(path: Path, raw_dir: Path, model_ids: set[str]) -> RawDocument:
    rel = path.relative_to(raw_dir)
    if len(rel.parts) < 2 or rel.parts[0] not in DOC_TYPES:
        raise IngestionError(
            f"{rel.as_posix()}: put documents in a folder named after their type: "
            + ", ".join(f"data/raw/{d}/" for d in DOC_TYPES))
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        pages = load_pdf(path)
    elif suffix in HTML_SUFFIXES:
        pages = load_html(path)
    elif suffix in TEXT_SUFFIXES:
        pages = load_text(path)
    else:
        raise IngestionError(
            f"{rel.as_posix()}: unsupported file type '{suffix}' (supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})")
    return RawDocument(source=rel.as_posix(), doc_type=rel.parts[0],
                       model=detect_model(path.stem, model_ids), pages=pages)


def load_documents(raw_dir: Path, model_ids: set[str]) -> list[RawDocument]:
    if not raw_dir.is_dir():
        raise IngestionError(f"raw documents folder not found: {raw_dir}")
    docs = []
    for path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(raw_dir)
        if path.name == ".gitkeep" or (len(rel.parts) == 1 and path.name in IGNORED_ROOT_FILES):
            continue
        docs.append(load_document(path, raw_dir, model_ids))
    if not docs:
        raise IngestionError(f"no documents found under {raw_dir}")
    return docs
