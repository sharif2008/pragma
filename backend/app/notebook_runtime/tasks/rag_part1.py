"""Standalone RAG Part 1: build FAISS vector store (no .ipynb). Run from repo root via scripts/rag_part1_build_vector_store.py."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings

from app.notebook_runtime.env import load_project_dotenv
from app.notebook_runtime.rag_index_build import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    TITLE_MODE_TRANSFORMER,
    build_documents_from_knowledge_base,
    expand_pdf_kb_items_with_sections,
    save_parent_store,
)
from app.notebook_runtime.rag_utils import faiss_from_documents_batched, save_vector_store

load_project_dotenv()


def _normalize_kb_record(
    item: dict[str, Any],
    source_file: str,
    doc_type: str,
    page: int | None = None,
) -> dict[str, Any]:
    out = dict(item)
    out["source_file"] = source_file
    out["doc_type"] = doc_type
    if page is not None:
        out["page"] = page
    return out


def load_pdf_file(file_path: Path) -> list[dict[str, Any]]:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    rows: list[dict[str, Any]] = []
    for i, doc in enumerate(docs):
        meta_page = doc.metadata.get("page")
        page_1based = int(meta_page) + 1 if meta_page is not None else i + 1
        rows.append(
            _normalize_kb_record(
                {
                    "title": f"{file_path.name} (page {page_1based})",
                    "text": doc.page_content or "",
                },
                file_path.name,
                "pdf",
                page=page_1based,
            )
        )
    return rows


def load_knowledge_base(knowledge_dir: Path, verbose: bool = True) -> list[dict[str, Any]]:
    kb: list[dict[str, Any]] = []
    knowledge_json = sorted(knowledge_dir.glob("*.json"))
    knowledge_pdf = sorted(knowledge_dir.glob("*.pdf"))
    if not knowledge_json and not knowledge_pdf:
        raise FileNotFoundError(f"No JSON or PDF files found in {knowledge_dir}")

    if verbose:
        print(f"Loading knowledge base from {knowledge_dir}:")

    for knowledge_file in knowledge_json:
        filename = knowledge_file.name
        try:
            with open(knowledge_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                n = 0
                for row in data:
                    if isinstance(row, dict) and "title" in row and "text" in row:
                        kb.append(_normalize_kb_record(row, filename, "json"))
                        n += 1
                if verbose:
                    print(f"  Loaded {n} document(s) from {filename}")
            elif isinstance(data, dict) and "title" in data and "text" in data:
                kb.append(_normalize_kb_record(data, filename, "json"))
                if verbose:
                    print(f"  Loaded 1 document from {filename}")
            elif verbose:
                print(f"  Skipped {filename} (unknown format)")
        except Exception as e:
            if verbose:
                print(f"  Error loading {filename}: {e}")

    for pdf_file in knowledge_pdf:
        try:
            docs = load_pdf_file(pdf_file)
            kb.extend(docs)
            if verbose:
                print(f"  Loaded {len(docs)} document(s) from {pdf_file.name} (PDF)")
        except Exception as e:
            if verbose:
                print(f"  Error loading {pdf_file.name}: {e}")

    if verbose:
        print(f"Total: {len(kb)} knowledge documents")
    return kb


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build RAG vector store (Part 1).")
    p.add_argument(
        "--knowledge-dir",
        type=Path,
        default=Path("RAG_docs/knowledge"),
        help="Directory with PDF/JSON knowledge files (relative to repo root unless absolute)",
    )
    p.add_argument(
        "--vector-store-dir",
        type=Path,
        default=Path("RAG_docs/vector_store"),
        help="Output directory for FAISS + manifest",
    )
    p.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    p.add_argument("--embed-batch", type=int, default=16)
    p.add_argument("--title-mode", default=TITLE_MODE_TRANSFORMER)
    p.add_argument("--max-llm-parents", type=int, default=120)
    args = p.parse_args(argv)

    try:
        import pypdf  # noqa: F401
    except ImportError as e:
        raise SystemExit("PDF loading requires pypdf. pip install pypdf") from e

    knowledge_dir = args.knowledge_dir.resolve()
    vector_store_dir = args.vector_store_dir.resolve()
    vector_store_dir.mkdir(parents=True, exist_ok=True)

    knowledge_base = load_knowledge_base(knowledge_dir)
    knowledge_expanded = expand_pdf_kb_items_with_sections(knowledge_base, knowledge_dir, verbose=True)

    embeddings = SentenceTransformerEmbeddings(
        model_name=args.embed_model,
        encode_kwargs={"batch_size": args.embed_batch},
    )
    rag_documents, parent_store = build_documents_from_knowledge_base(
        knowledge_expanded,
        embeddings,
        title_mode=args.title_mode,
        summarization_model=None,
        print_titles=True,
        max_llm_parents=args.max_llm_parents,
        verbose=True,
    )

    print(
        f"Embedding model={args.embed_model}, child chunk={CHILD_CHUNK_SIZE}/{CHILD_CHUNK_OVERLAP}, "
        f"{len(rag_documents)} FAISS vectors, {len(parent_store)} parents"
    )

    vector_store = faiss_from_documents_batched(rag_documents, embeddings)
    save_vector_store(
        vector_store,
        vector_store_dir,
        knowledge_expanded,
        embed_model=args.embed_model,
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        n_chunks=len(rag_documents),
        extra_manifest={
            "indexing": "parent_child_semantic",
            "n_parents": len(parent_store),
            "title_mode": args.title_mode,
            "max_llm_parents": args.max_llm_parents,
        },
    )
    ps_path = save_parent_store(vector_store_dir, parent_store)
    print(f"Saved vector store to {vector_store_dir}")
    print(f"Saved parent store to {ps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
