#!/usr/bin/env python3
"""Offline demo: chunk text, embed with SentenceTransformers, build FAISS (no API server)."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.config import get_settings
from app.rag.chunking import chunk_text
from app.rag.vector_store import FaissKnowledgeIndex


def main() -> None:
    settings = get_settings()
    text = (
        "DDoS attacks flood network resources. Mitigation includes rate limiting and upstream scrubbing. "
        "SOC playbooks require logging source IPs and alerting on threshold breaches."
    )
    chunks = [{"text": t, "source": "demo.txt"} for t in chunk_text(text, 120, 20)]
    out_dir = settings.storage_root / "vector_db" / "_demo_local_index"
    out_dir.mkdir(parents=True, exist_ok=True)
    store = FaissKnowledgeIndex(out_dir, settings.embedding_model)
    store.build_from_texts(chunks)
    store.load()
    hits = store.search("How to mitigate flooding?", top_k=2)
    print("Top hits:")
    for score, ch in hits:
        print(f"  {score:.4f} | {ch['text'][:80]}...")


if __name__ == "__main__":
    main()
