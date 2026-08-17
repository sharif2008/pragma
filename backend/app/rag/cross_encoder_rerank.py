"""Cross-encoder reranking for RAG chunk selection (query–passage relevance)."""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_LOCK = threading.Lock()
_MODEL_NAME: str | None = None
_ENCODER: Any = None


def get_cross_encoder(model_name: str = _DEFAULT_MODEL) -> Any | None:
    """Lazy-load and cache a sentence-transformers CrossEncoder. None if unavailable."""
    global _ENCODER, _MODEL_NAME
    with _LOCK:
        if _ENCODER is not None and _MODEL_NAME == model_name:
            return _ENCODER
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            logger.warning("CrossEncoder unavailable (sentence-transformers): %s", e)
            return None
        try:
            logger.info("Loading CrossEncoder reranker model=%s", model_name)
            _ENCODER = CrossEncoder(model_name)
            _MODEL_NAME = model_name
            logger.info("CrossEncoder reranker ready: %s", model_name)
            return _ENCODER
        except Exception as e:
            logger.warning("Failed to load CrossEncoder %s: %s", model_name, e)
            return None


def score_query_passages(
    query: str,
    passages: list[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[float] | None:
    """
    Return CrossEncoder relevance logits for (query, passage) pairs.
    None if the model cannot be loaded or passages is empty.
    """
    if not passages:
        return []
    q = (query or "").strip()
    if not q:
        return [0.0] * len(passages)
    ce = get_cross_encoder(model_name)
    if ce is None:
        return None
    pairs = [(q, (p or "")[:4000]) for p in passages]
    try:
        raw = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    except TypeError:
        raw = ce.predict(pairs)
    except Exception as e:
        logger.warning("CrossEncoder predict failed: %s", e)
        return None
    return [float(x) for x in raw]


def score_queries_passages_max(
    queries: list[str],
    passages: list[str],
    *,
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[float] | None:
    """
    For each passage, take the **max** CrossEncoder score across all non-empty queries.
    Falls back to single-query scoring when only one query is present.
    """
    if not passages:
        return []
    qs = [str(q).strip() for q in queries if str(q).strip()]
    if not qs:
        return [0.0] * len(passages)
    if len(qs) == 1:
        return score_query_passages(qs[0], passages, model_name=model_name, batch_size=batch_size)

    ce = get_cross_encoder(model_name)
    if ce is None:
        return None

    n = len(passages)
    max_scores = [-1e18] * n
    # Batch all (query, passage) pairs; reshape by query blocks.
    pairs: list[tuple[str, str]] = []
    for q in qs:
        for p in passages:
            pairs.append((q, (p or "")[:4000]))
    try:
        try:
            raw = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except TypeError:
            raw = ce.predict(pairs)
    except Exception as e:
        logger.warning("CrossEncoder multi-query predict failed: %s", e)
        return None

    for qi, _q in enumerate(qs):
        base = qi * n
        for pi in range(n):
            s = float(raw[base + pi])
            if s > max_scores[pi]:
                max_scores[pi] = s
    return max_scores


def normalize_scores(vals: list[float]) -> list[float]:
    """Min–max normalize to [0, 1] within the candidate set."""
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return [1.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]
