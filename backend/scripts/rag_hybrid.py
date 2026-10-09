"""Dense FAISS + BM25 over the same child ids → RRF → optional MMR → parents."""

from __future__ import annotations

import math
import re
from typing import Any

from scripts.reason import (
    _chunk_key,
    mmr_select,
    retrieve_child_chunks_for_query,
)

_TOKEN = re.compile(r"[a-z0-9]+")
DENSE_N = 80
BM25_N = 80
RRF_K = 60
MMR_LAMBDA = 0.5
FINAL_CHILDREN = 20
FINAL_PARENTS = 5


def _tok(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def _child_record(doc: Any) -> dict[str, Any]:
    meta = getattr(doc, "metadata", None) or {}
    return {
        "title": meta.get("retrieval_title") or meta.get("title", "Unknown"),
        "source_file": meta.get("source_file", ""),
        "parent_id": meta.get("parent_id"),
        "child_index": meta.get("child_index"),
        "chunk_text": getattr(doc, "page_content", None) or meta.get("text", "") or "",
    }


def _iter_store_docs(vector_store: Any) -> list[Any]:
    store = getattr(vector_store, "docstore", None)
    mapping = getattr(store, "_dict", None) if store is not None else None
    if isinstance(mapping, dict) and mapping:
        return list(mapping.values())
    return []


def build_bm25(vector_store: Any) -> dict[str, Any]:
    cached = getattr(vector_store, "_pragma_bm25", None)
    if cached is not None:
        return cached
    recs: list[dict[str, Any]] = []
    docs_tokens: list[list[str]] = []
    df: dict[str, int] = {}
    for doc in _iter_store_docs(vector_store):
        rec = _child_record(doc)
        toks = _tok(rec["chunk_text"])
        recs.append(rec)
        docs_tokens.append(toks)
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(recs)
    avgdl = (sum(len(t) for t in docs_tokens) / n) if n else 1.0
    idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    index = {"recs": recs, "tokens": docs_tokens, "idf": idf, "avgdl": avgdl, "n": n}
    setattr(vector_store, "_pragma_bm25", index)
    return index


def bm25_search(vector_store: Any, query: str, *, top_k: int = BM25_N) -> list[dict[str, Any]]:
    idx = build_bm25(vector_store)
    qtoks = _tok(query)
    if not qtoks or not idx["n"]:
        return []
    k1, b = 1.5, 0.75
    avgdl = float(idx["avgdl"] or 1.0)
    scores: list[tuple[float, int]] = []
    for i, toks in enumerate(idx["tokens"]):
        if not toks:
            continue
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks)
        s = 0.0
        for t in qtoks:
            if t not in tf:
                continue
            idf = float(idx["idf"].get(t, 0.0))
            freq = tf[t]
            s += idf * (freq * (k1 + 1.0)) / (freq + k1 * (1.0 - b + b * dl / avgdl))
        if s > 0.0:
            scores.append((s, i))
    scores.sort(reverse=True)
    out: list[dict[str, Any]] = []
    for s, i in scores[: int(top_k)]:
        d = dict(idx["recs"][i])
        d["bm25_score"] = float(s)
        d["vector_score"] = float(d.get("vector_score") or 0.0)
        out.append(d)
    return out


def dense_search(vector_store: Any, query: str, *, top_k: int = DENSE_N) -> list[dict[str, Any]]:
    return retrieve_child_chunks_for_query(
        vector_store,
        query,
        top_k=int(top_k),
        oversample_factor=1,
        balance_by_source_file=False,
    )


def rrf_fuse(lists: list[list[dict[str, Any]]], *, k: int = RRF_K) -> list[dict[str, Any]]:
    fused: dict[tuple[Any, ...], dict[str, Any]] = {}
    for hits in lists:
        for rank, d in enumerate(hits, start=1):
            key = _chunk_key(d)
            part = 1.0 / (float(k) + float(rank))
            prev = fused.get(key)
            if prev is None:
                rec = dict(d)
                rec["rrf"] = part
                rec["vector_score"] = float(d.get("vector_score") or 0.0)
                if d.get("bm25_score") is not None:
                    rec["bm25_score"] = float(d["bm25_score"])
                fused[key] = rec
            else:
                prev["rrf"] = float(prev.get("rrf") or 0.0) + part
                if d.get("vector_score") is not None:
                    prev["vector_score"] = max(float(prev.get("vector_score") or 0.0), float(d["vector_score"]))
                if d.get("bm25_score") is not None:
                    prev["bm25_score"] = max(float(prev.get("bm25_score") or 0.0), float(d["bm25_score"]))
    out = list(fused.values())
    out.sort(key=lambda x: float(x.get("rrf") or 0.0), reverse=True)
    for d in out:
        d["rerank_score"] = float(d.get("rrf") or 0.0)
    return out


def hybrid_children(
    vector_store: Any,
    query: str,
    *,
    rank: bool,
    dense_n: int = DENSE_N,
    bm25_n: int = BM25_N,
    final_children: int = FINAL_CHILDREN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dense = dense_search(vector_store, query, top_k=dense_n)
    lexical = bm25_search(vector_store, query, top_k=bm25_n)
    fused = rrf_fuse([dense, lexical])
    meta = {
        "dense_n": len(dense),
        "bm25_n": len(lexical),
        "rrf_n": len(fused),
        "pipeline": "faiss_bm25_rrf_mmr" if rank else "faiss_bm25_rrf",
    }
    if rank:
        picked = mmr_select(vector_store, query, fused, k=int(final_children), lambda_mult=MMR_LAMBDA)
    else:
        picked = fused[: int(final_children)]
    for c in picked:
        if c.get("rerank_score") is None:
            c["rerank_score"] = float(c.get("rrf") or c.get("vector_score") or 0.0)
    meta["final_children"] = len(picked)
    return picked, meta
