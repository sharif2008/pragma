"""Text chunking for knowledge documents."""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts)


def load_document_text(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".pdf":
        return read_pdf_text(path)
    if suf in (".txt", ".md", ".json", ".csv"):
        return read_text_file(path)
    return read_text_file(path)


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks
