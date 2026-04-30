#!/usr/bin/env python3
"""
Upload local documents into the MySQL-backed knowledge base (same pipeline as POST /kb/upload).

Chunks are written to storage/vector_db and rows to knowledge_base_files so RAG (e.g. the CSV
trust-anchor benchmark with --use-rag) can retrieve them.

Requirements: DATABASE_URL, STORAGE_ROOT (optional), working MySQL; embedding model downloads on first index.

Usage (from backend/) - folder uploads (recursive .pdf .txt .md .json):

  python scripts\\upload_kb_files.py D:\\Projects\\ChainAgentVFL\\my_kb_folder
  python scripts\\upload_kb_files.py -d D:\\Projects\\ChainAgentVFL\\my_kb_folder
  python scripts\\upload_kb_files.py --dir D:\\Projects\\ChainAgentVFL\\my_kb_folder

Single files:

  python scripts\\upload_kb_files.py ..\\path\\guide.pdf
  python scripts\\upload_kb_files.py file1.pdf file2.md
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import UploadFile
from starlette.datastructures import Headers

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services import kb_service

_KB_SUFFIXES = {".pdf", ".txt", ".md", ".json"}


def _mime_for_path(path: Path) -> str:
    return {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
    }.get(path.suffix.lower(), "text/plain")


def _collect_files(paths: list[str], dirs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(str(p))
        if p.is_file():
            if p.suffix.lower() in _KB_SUFFIXES:
                out.append(p)
            else:
                raise ValueError(f"Unsupported type {p.suffix!r} for {p} (use {_KB_SUFFIXES})")
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in _KB_SUFFIXES:
                    out.append(f)
        else:
            raise ValueError(f"Not a file or directory: {p}")

    for raw in dirs:
        d = Path(raw).expanduser().resolve()
        if not d.is_dir():
            raise NotADirectoryError(str(d))
        for f in sorted(d.rglob("*")):
            if f.is_file() and f.suffix.lower() in _KB_SUFFIXES:
                out.append(f)

    # stable unique order
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in out:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(rp)
    return unique


def _upload_file_from_path(path: Path) -> UploadFile:
    data = path.read_bytes()
    return UploadFile(
        file=io.BytesIO(data),
        filename=path.name,
        headers=Headers({"content-type": _mime_for_path(path)}),
    )


async def _ingest_all(files: list[Path]) -> list[tuple[str, str, int]]:
    settings = get_settings()
    db = SessionLocal()
    results: list[tuple[str, str, int]] = []
    try:
        for path in files:
            uf = _upload_file_from_path(path)
            kb = await kb_service.ingest_kb_document(db, settings, uf)
            results.append((path.name, kb.public_id, kb.chunk_count))
            print(f"[ok] {path.name}  kb_public_id={kb.public_id}  chunks={kb.chunk_count}")
        return results
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Upload PDF/TXT/MD/JSON files into the knowledge base (DB + FAISS index under storage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Files and/or directories (directories are scanned recursively for supported extensions).",
    )
    ap.add_argument(
        "-d",
        "--dir",
        action="append",
        default=[],
        metavar="DIR",
        help="Folder path to scan recursively for .pdf .txt .md .json (repeatable). Same as passing that folder as a positional argument.",
    )
    args = ap.parse_args()

    if not args.paths and not args.dir:
        ap.error("Provide at least one file or directory (positional paths or --dir).")

    files = _collect_files(args.paths, args.dir)
    if not files:
        print("No matching files (.pdf .txt .md .json) found.", file=sys.stderr)
        return 1

    print(f"Ingesting {len(files)} file(s)…")
    asyncio.run(_ingest_all(files))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
