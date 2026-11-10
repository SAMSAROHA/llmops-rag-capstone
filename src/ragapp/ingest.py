"""Load .md / .txt / .pdf files, split them into overlapping chunks, index into Qdrant."""

from __future__ import annotations

import re
from pathlib import Path

from .vectorstore import Chunk, VectorStore

SUPPORTED = {".md", ".txt", ".pdf"}


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def split_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Greedy paragraph packing. Paragraphs longer than max_chars are split on
    sentence boundaries; each chunk starts with the tail of the previous one."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            units.append(para)
            continue
        current = ""
        for sentence in re.split(r"(?<=[.!?])\s+", para):
            while len(sentence) > max_chars:  # no sentence boundary at all
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if current and len(current) + 1 + len(sentence) > max_chars:
                units.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            units.append(current)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 2 + len(unit) > max_chars:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            current = f"{tail}\n\n{unit}" if tail else unit
        else:
            current = f"{current}\n\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


def load_chunks(root: Path, max_chars: int, overlap: int) -> list[Chunk]:
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    chunks: list[Chunk] = []
    for path in files:
        if path.suffix.lower() not in SUPPORTED:
            continue
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        for i, text in enumerate(split_text(read_file(path), max_chars, overlap)):
            chunks.append(Chunk(text=text, source=source, chunk_index=i))
    return chunks


def ingest(
    store: VectorStore, root: Path, max_chars: int, overlap: int, recreate: bool = False
) -> int:
    chunks = load_chunks(root, max_chars, overlap)
    if not chunks:
        raise FileNotFoundError(f"No .md/.txt/.pdf content found under {root}")
    store.ensure_collection(recreate=recreate)
    return store.upsert(chunks)
