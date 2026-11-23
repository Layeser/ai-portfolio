"""PDF ingestion and structure-aware chunking.

Chunks never cross a section boundary, are packed sentence by sentence up to
`chunk_size` characters with a one-sentence overlap, and carry their
provenance (document, department, section, page) for citations.

Each chunk also gets a *contextual header* ("<document title> > <section>")
that is indexed with the text: a chunk saying "it is capped at 35 euros" is
only retrievable if the model knows it belongs to "Business travel > Abroad".
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from ragassist.config import settings

SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Ý«])")


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    department: str
    title: str
    section: str
    page: int
    text: str

    @property
    def header(self) -> str:
        return f"{self.title} > {self.section}"

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    """(page number, line) pairs, with PDF line breaks inside paragraphs re-joined later."""
    lines = []
    for number, page in enumerate(PdfReader(str(path)).pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            if line.strip():
                lines.append((number, line.strip()))
    return lines


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    chunks, current = [], []
    for sentence in sentences:
        if current and len(" ".join(current + [sentence])) > size:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(path: Path, size: int = settings.chunk_size,
                   overlap: int = settings.chunk_overlap_sentences) -> list[Chunk]:
    lines = _read_pdf(path)
    if not lines:
        return []
    title_lines: list[str] = []  # a long title can wrap over several PDF lines
    sections: list[tuple[str, int, list[str]]] = []  # (heading, first page, lines)
    for page, line in lines:
        match = SECTION_RE.match(line)
        if match and len(line) < 80:
            sections.append((f"{match.group(1)}. {match.group(2)}", page, []))
        elif sections:
            sections[-1][2].append(line)
        else:
            title_lines.append(line)
    title = " ".join(title_lines) or path.stem

    chunks = []
    for heading, page, body in sections:
        text = re.sub(r"\s+", " ", " ".join(body)).strip()
        for i, piece in enumerate(_pack(SENTENCE_RE.split(text), size, overlap)):
            chunks.append(Chunk(
                chunk_id=f"{path.stem}#{heading.split('.')[0]}.{i}",
                doc=path.stem, department=path.parent.name, title=title,
                section=heading, page=page, text=piece,
            ))
    return chunks


def load_corpus(corpus_dir: Path = settings.corpus_dir) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.rglob("*.pdf")):
        chunks.extend(chunk_document(path))
    return chunks
