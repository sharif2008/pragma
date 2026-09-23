"""FAISS + SentenceTransformers vector store (per-KB directory)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Process-wide embedding models — opening a new FaissKnowledgeIndex must not reload weights.
_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Return a cached SentenceTransformer for ``model_name`` (load once per process)."""
    name = (model_name or "").strip() or "sentence-transformers/all-MiniLM-L6-v2"
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(name)
        if cached is not None:
            return cached
        logger.info("Loading embedding model (once): %s", name)
        model = SentenceTransformer(name)
        _MODEL_CACHE[name] = model
        return model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


class FaissKnowledgeIndex:
    """Stores chunk texts + FAISS IndexFlatIP on normalized embeddings (cosine via inner product)."""

    def __init__(self, index_dir: Path, model_name: str) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._chunks: list[dict[str, Any]] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = get_embedding_model(self.model_name)
        return self._model

    def _paths(self) -> tuple[Path, Path]:
        return self.index_dir / "index.faiss", self.index_dir / "meta.json"

    def build_from_texts(self, chunks: list[dict[str, Any]]) -> None:
        texts = [c["text"] for c in chunks]
        if not texts:
            self._chunks = []
            self._index = None
            self._save_empty()
            return
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb.astype("float32")
        emb = _normalize(emb)
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self._index = index
        self._chunks = chunks
        self._persist()

    def _persist(self) -> None:
        idx_path, meta_path = self._paths()
        if self._index is None:
            self._save_empty()
            return
        faiss.write_index(self._index, str(idx_path))
        meta_path.write_text(json.dumps({"chunks": self._chunks}, ensure_ascii=False), encoding="utf-8")

    def _save_empty(self) -> None:
        _, meta_path = self._paths()
        meta_path.write_text(json.dumps({"chunks": []}, ensure_ascii=False), encoding="utf-8")
        for p in self._paths():
            if p.suffix == ".faiss" and p.exists():
                p.unlink()

    def load(self) -> None:
        idx_path, meta_path = self._paths()
        if not meta_path.exists():
            self._chunks = []
            self._index = None
            return
        self._chunks = json.loads(meta_path.read_text(encoding="utf-8")).get("chunks", [])
        if idx_path.exists() and self._chunks:
            self._index = faiss.read_index(str(idx_path))
        else:
            self._index = None

    def search(self, query: str, top_k: int) -> list[tuple[float, dict[str, Any]]]:
        if not self._chunks or self._index is None:
            return []
        q = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
        q = _normalize(q)
        scores, indices = self._index.search(q, min(top_k, len(self._chunks)))
        out: list[tuple[float, dict[str, Any]]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            out.append((float(score), self._chunks[idx]))
        return out
