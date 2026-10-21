"""Structure-aware chunking.

- Chunks never cross a heading, so each chunk stays on one topic.
- Tables (consecutive '|' lines) are never split: each table becomes its own chunk.
- Long sections are split at paragraph, then sentence, boundaries, with a small overlap.
- Each chunk starts with its heading path (e.g. "Skoda Kushaq > Safety") for context.
"""

import re
from dataclasses import dataclass

from ingestion.loaders import RawDocument

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class ChunkDraft:
    text: str
    model: str
    doc_type: str
    source: str
    page: int | None
    section: str


@dataclass
class Block:
    kind: str  # "heading", "table" or "text"
    text: str
    level: int = 0


def parse_blocks(text: str) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph():
        if paragraph:
            blocks.append(Block("text", " ".join(s.strip() for s in paragraph)))
            paragraph.clear()

    def flush_table():
        if table:
            blocks.append(Block("table", "\n".join(table)))
            table.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("|"):
            flush_paragraph()
            table.append(stripped)
            continue
        flush_table()
        heading = HEADING_RE.match(stripped)
        if heading:
            flush_paragraph()
            blocks.append(Block("heading", heading.group(2), level=len(heading.group(1))))
        elif not stripped:
            flush_paragraph()
        elif stripped.startswith(("- ", "* ")):
            flush_paragraph()  # list items stay separate paragraphs
            paragraph.append(stripped)
            flush_paragraph()
        else:
            paragraph.append(stripped)
    flush_paragraph()
    flush_table()
    return blocks


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split one over-long paragraph at sentence boundaries (hard-cut only if a sentence is too long)."""
    pieces, current = [], ""
    for sentence in SENTENCE_RE.split(text):
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _tail(text: str, overlap_chars: int) -> str:
    """Last ~overlap_chars of text, starting at a sentence or word boundary."""
    if overlap_chars <= 0 or len(text) <= overlap_chars:
        return text if overlap_chars > 0 else ""
    tail = text[-overlap_chars:]
    sentence = SENTENCE_RE.search(tail)
    if sentence:
        return tail[sentence.end():]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def chunk_document(doc: RawDocument, max_chars: int, overlap_chars: int) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    headings: list[tuple[int, str]] = []  # heading stack carries across PDF pages

    for page, text in doc.pages:
        body: list[str] = []
        only_overlap = False  # body holds just the overlap carried from the previous chunk

        def section() -> str:
            return " > ".join(h for _, h in headings)

        def emit(content: str):
            sec = section()
            full = f"{sec}\n\n{content}" if sec else content
            drafts.append(ChunkDraft(full, doc.model, doc.doc_type, doc.source, page, sec))

        def flush(carry_overlap: bool = False):
            nonlocal only_overlap
            if body and not only_overlap:
                content = "\n\n".join(body)
                emit(content)
                body.clear()
                tail = _tail(content, overlap_chars) if carry_overlap else ""
                if tail:
                    body.append(tail)
                only_overlap = bool(tail)
            elif not carry_overlap:
                body.clear()
                only_overlap = False

        for block in parse_blocks(text):
            if block.kind == "heading":
                flush()
                while headings and headings[-1][0] >= block.level:
                    headings.pop()
                headings.append((block.level, block.text))
            elif block.kind == "table":
                flush()
                emit(block.text)  # never split
            else:
                for piece in _split_long(block.text, max_chars):
                    if body and not only_overlap and len("\n\n".join(body)) + 2 + len(piece) > max_chars:
                        flush(carry_overlap=True)
                    body.append(piece)
                    only_overlap = False
        flush()
    return [d for d in drafts if d.text.strip()]
