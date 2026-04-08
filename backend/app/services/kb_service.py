"""Knowledge base indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import FileKind, JobStatus, KnowledgeBaseFile, ManagedFile
from app.rag.chunking import chunk_text, load_document_text
from app.rag.vector_store import FaissKnowledgeIndex, _normalize
from app.services import file_service, prediction_service
from app.services.rag_templates_from_predictions import build_rag_templates_from_summary
from app.services.rag_templates_row_context import build_row_agent_templates
from app.utils.file_utils import remove_path

logger = logging.getLogger(__name__)

_RRF_K = 60


async def ingest_kb_document(
    db: Session,
    settings: Settings,
    upload: UploadFile,
) -> KnowledgeBaseFile:
    mf = await file_service.upload_file(db, settings, upload, FileKind.knowledge_doc, replace_public_id=None)

    src = file_service.resolved_path(settings, mf)
    text = load_document_text(src)
    chunks_raw = chunk_text(text, settings.rag_chunk_size, settings.rag_chunk_overlap)
    chunks = [{"text": t, "source": mf.original_name, "managed_file_public_id": mf.public_id} for t in chunks_raw]

    index_dir = settings.storage_root / "vector_db" / mf.public_id
    remove_path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    store = FaissKnowledgeIndex(index_dir, settings.embedding_model)
    store.build_from_texts(chunks)

    kb = KnowledgeBaseFile(
        managed_file_id=mf.id,
        vector_index_dir=str(index_dir.relative_to(settings.storage_root)),
        chunk_count=len(chunks_raw),
        embedding_model=settings.embedding_model,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info("KB indexed public_id=%s chunks=%s", kb.public_id, kb.chunk_count)
    return kb


def list_kb_files(db: Session) -> list[KnowledgeBaseFile]:
    return list(db.scalars(select(KnowledgeBaseFile).order_by(KnowledgeBaseFile.created_at.desc())).all())


def get_kb(db: Session, public_id: str) -> KnowledgeBaseFile:
    row = db.scalar(select(KnowledgeBaseFile).where(KnowledgeBaseFile.public_id == public_id))
    if not row:
        raise HTTPException(404, "Knowledge base entry not found")
    return row


def delete_kb(db: Session, settings: Settings, public_id: str) -> None:
    row = get_kb(db, public_id)
    vdir = settings.storage_root / row.vector_index_dir
    remove_path(vdir)
    mf = db.get(ManagedFile, row.managed_file_id)
    if mf:
        remove_path(file_service.resolved_path(settings, mf))
        db.delete(mf)
    db.delete(row)
    db.commit()


def _open_store(settings: Settings, kb: KnowledgeBaseFile) -> FaissKnowledgeIndex:
    path = settings.storage_root / kb.vector_index_dir
    store = FaissKnowledgeIndex(path, kb.embedding_model)
    store.load()
    return store


def query_kb(
    db: Session,
    settings: Settings,
    query: str,
    top_k: int,
    kb_public_ids: list[str] | None,
) -> list[tuple[float, dict, str]]:
    q = select(KnowledgeBaseFile)
    if kb_public_ids:
        q = q.where(KnowledgeBaseFile.public_id.in_(kb_public_ids))
    rows = list(db.scalars(q).all())
    if not rows:
        return []

    hits: list[tuple[float, dict, str]] = []
    for kb in rows:
        store = _open_store(settings, kb)
        for score, chunk in store.search(query, top_k):
            hits.append((score, chunk, kb.public_id))
    hits.sort(key=lambda x: x[0], reverse=True)
    return hits[:top_k]


def _chunk_fusion_key(kb_id: str, text: str) -> str:
    h = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{kb_id}:{h}"


def _norm_list(vals: list[float]) -> list[float]:
    if not vals:
        return []
    mn, mx = min(vals), max(vals)
    if mx <= mn:
        return [1.0] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


def _finalize_fused_pool_mmr(
    fused: dict[str, dict[str, Any]],
    rows: list[KnowledgeBaseFile],
    settings: Settings,
    queries: list[str],
    *,
    final_k: int,
    mmr_lambda: float,
    use_mmr: bool,
    pool_multiplier: int,
    meta: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Deduped fusion map → fusion rerank → pool → MMR (or top-k by rerank) → hit dicts.
    """
    if not fused or not rows:
        return [], meta

    items = list(fused.values())
    max_scores = [x["max_score"] for x in items]
    rrfs = [x["rrf"] for x in items]
    n_max = _norm_list(max_scores)
    n_rrf = _norm_list(rrfs)
    for i, it in enumerate(items):
        it["fusion_rerank"] = 0.55 * n_max[i] + 0.45 * n_rrf[i]

    items.sort(key=lambda x: x["fusion_rerank"], reverse=True)
    pool_n = min(max(final_k * pool_multiplier, final_k + 3), 80, len(items))
    pool = items[:pool_n]

    store0 = _open_store(settings, rows[0])
    model = store0.model
    texts = [p["chunk"].get("text") or "" for p in pool]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    emb = _normalize(emb)

    q_vecs = model.encode([q.strip() for q in queries], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_vecs = _normalize(q_vecs)
    q_centroid = _normalize(np.mean(q_vecs, axis=0, keepdims=True))
    sim_q = (emb @ q_centroid.T).flatten()

    selected: list[int] = []
    mmr_margins: list[float | None] = []

    if use_mmr:
        lam = max(0.0, min(1.0, mmr_lambda))
        remaining = set(range(len(pool)))
        while len(selected) < final_k and remaining:
            best_i: int | None = None
            best_mmr = -1e18
            for i in remaining:
                rel = float(sim_q[i])
                if not selected:
                    mmr = rel
                else:
                    div = max(float(np.dot(emb[i], emb[j])) for j in selected)
                    mmr = lam * rel - (1.0 - lam) * div
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_i = i
            assert best_i is not None
            selected.append(best_i)
            mmr_margins.append(best_mmr)
            remaining.remove(best_i)
    else:
        k = min(final_k, len(pool))
        selected = list(range(k))
        mmr_margins = [None] * k

    hits: list[dict[str, Any]] = []
    for idx, i in enumerate(selected):
        p = pool[i]
        ch = p["chunk"]
        mmr_val = mmr_margins[idx]
        hits.append(
            {
                "score": float(sim_q[i]),
                "text": ch.get("text", ""),
                "source": ch.get("source"),
                "kb_public_id": p["kb_id"],
                "rerank_score": float(p["fusion_rerank"]),
                "mmr_margin": float(mmr_val) if mmr_val is not None else None,
            }
        )
    meta["pool_size"] = pool_n
    meta["candidates_fused"] = len(fused)
    return hits, meta


def query_kb_multi_mmr(
    db: Session,
    settings: Settings,
    queries: list[str],
    *,
    final_k: int,
    per_query_k: int,
    mmr_lambda: float,
    kb_public_ids: list[str] | None,
    pool_multiplier: int = 4,
    use_mmr: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Multi-query retrieval: per-query FAISS hits, RRF + max-score fusion rerank, then MMR on the pool.
    All pooled texts are embedded with the first KB index's model (indices should share the same embedding space).
    """
    q = select(KnowledgeBaseFile)
    if kb_public_ids:
        q = q.where(KnowledgeBaseFile.public_id.in_(kb_public_ids))
    rows = list(db.scalars(q).all())
    meta: dict[str, Any] = {
        "queries_used": queries,
        "per_query_k": per_query_k,
        "final_k": final_k,
        "mmr_lambda": mmr_lambda,
        "fusion": "rrf_plus_max_score_then_mmr" if use_mmr else "rrf_plus_max_score_topk",
        "use_mmr": use_mmr,
        "pipeline": "single_request_per_query_faiss_then_fuse_mmr",
    }
    if not rows or not queries:
        return [], meta

    # fusion_key -> { chunk, kb_id, max_score, rrf }
    fused: dict[str, dict[str, Any]] = {}
    for q_idx, query in enumerate(queries):
        for kb in rows:
            store = _open_store(settings, kb)
            raw = store.search(query.strip(), per_query_k)
            for rank, (score, chunk) in enumerate(raw, start=1):
                text = chunk.get("text") or ""
                if not text.strip():
                    continue
                fk = _chunk_fusion_key(kb.public_id, text)
                rrf_part = 1.0 / (_RRF_K + rank)
                if fk not in fused:
                    fused[fk] = {"chunk": chunk, "kb_id": kb.public_id, "max_score": float(score), "rrf": rrf_part}
                else:
                    fused[fk]["max_score"] = max(fused[fk]["max_score"], float(score))
                    fused[fk]["rrf"] += rrf_part

    return _finalize_fused_pool_mmr(
        fused,
        rows,
        settings,
        queries,
        final_k=final_k,
        mmr_lambda=mmr_lambda,
        use_mmr=use_mmr,
        pool_multiplier=pool_multiplier,
        meta=meta,
    )


def fuse_per_query_hit_groups_mmr(
    db: Session,
    settings: Settings,
    queries: list[str],
    per_query_hits: list[list[dict[str, Any]]],
    *,
    final_k: int,
    mmr_lambda: float,
    kb_public_ids: list[str] | None,
    pool_multiplier: int = 4,
    use_mmr: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Each query was retrieved separately (e.g. repeated POST /kb/query). Merge hit lists with the same
    dedupe + RRF/max fusion + rerank + MMR as query_kb_multi_mmr. Rank in each list is 1-based list order.
    """
    q = select(KnowledgeBaseFile)
    if kb_public_ids:
        q = q.where(KnowledgeBaseFile.public_id.in_(kb_public_ids))
    rows = list(db.scalars(q).all())
    meta: dict[str, Any] = {
        "queries_used": queries,
        "final_k": final_k,
        "mmr_lambda": mmr_lambda,
        "fusion": "rrf_plus_max_score_then_mmr" if use_mmr else "rrf_plus_max_score_topk",
        "use_mmr": use_mmr,
        "pipeline": "sequential_kb_query_then_fuse_rerank_mmr",
        "per_query_hits_received": [len(g) for g in per_query_hits],
    }
    if not rows or not queries:
        return [], meta

    fused: dict[str, dict[str, Any]] = {}
    for q_idx, hits in enumerate(per_query_hits):
        if q_idx >= len(queries):
            break
        for rank, h in enumerate(hits, start=1):
            text = str(h.get("text") or "").strip()
            if not text:
                continue
            kb_id = str(h.get("kb_public_id") or "")
            fk = _chunk_fusion_key(kb_id, text)
            score = float(h.get("score") or 0.0)
            chunk = {"text": text, "source": h.get("source")}
            rrf_part = 1.0 / (_RRF_K + rank)
            if fk not in fused:
                fused[fk] = {"chunk": chunk, "kb_id": kb_id, "max_score": score, "rrf": rrf_part}
            else:
                fused[fk]["max_score"] = max(fused[fk]["max_score"], score)
                fused[fk]["rrf"] += rrf_part

    return _finalize_fused_pool_mmr(
        fused,
        rows,
        settings,
        queries,
        final_k=final_k,
        mmr_lambda=mmr_lambda,
        use_mmr=use_mmr,
        pool_multiplier=pool_multiplier,
        meta=meta,
    )


def prediction_job_rag_context(
    db: Session,
    settings: Settings,
    prediction_job_public_id: str,
    *,
    row_index: int | None = None,
) -> dict[str, Any]:
    """Templates for a chosen completed job; optional row-level SHAP-aware queries."""
    job = prediction_service.get_prediction_job(db, prediction_job_public_id)
    if job.status != JobStatus.completed:
        return {
            "prediction_job_public_id": job.public_id,
            "summary": None,
            "templates": [],
            "message": "Prediction job must be completed to build RAG templates.",
            "row_index": row_index,
            "row_context": None,
        }
    summary = prediction_service.load_prediction_summary(settings, job)
    templates = build_rag_templates_from_summary(summary)
    row_context: dict[str, Any] | None = None
    rj = job.results_json
    if row_index is not None:
        if not isinstance(rj, dict):
            return {
                "prediction_job_public_id": job.public_id,
                "summary": summary,
                "templates": [],
                "message": "Load prediction with include_results=true (or re-run prediction) to attach row-level results_json.",
                "row_index": row_index,
                "row_context": None,
            }
        rows = rj.get("rows")
        if not isinstance(rows, list) or not (0 <= row_index < len(rows)) or not isinstance(rows[row_index], dict):
            return {
                "prediction_job_public_id": job.public_id,
                "summary": summary,
                "templates": [],
                "message": f"Invalid row_index={row_index} for this job (rows={len(rows) if isinstance(rows, list) else 0}).",
                "row_index": row_index,
                "row_context": None,
            }
        row = rows[row_index]
        base = f"{summary.get('rows_flagged')} flagged / {summary.get('rows_total')} total"
        extra, row_context = build_row_agent_templates(
            job_public_id=job.public_id,
            row=row,
            base_summary_line=str(base),
        )
        templates = extra + templates

    return {
        "prediction_job_public_id": job.public_id,
        "summary": summary,
        "templates": templates,
        "message": None,
        "row_index": row_index,
        "row_context": row_context,
    }


def latest_prediction_rag_context(db: Session, settings: Settings) -> dict[str, Any]:
    """Latest completed prediction job + generated templates (empty if none)."""
    jobs = prediction_service.list_prediction_jobs(db, limit=40, offset=0)
    job = next((j for j in jobs if j.status == JobStatus.completed), None)
    if not job:
        return {
            "prediction_job_public_id": None,
            "summary": None,
            "templates": [],
            "message": "No completed prediction job found. Run a batch prediction first.",
            "row_index": None,
            "row_context": None,
        }
    summary = prediction_service.load_prediction_summary(settings, job)
    templates = build_rag_templates_from_summary(summary)
    return {
        "prediction_job_public_id": job.public_id,
        "summary": summary,
        "templates": templates,
        "message": None,
        "row_index": None,
        "row_context": None,
    }


def format_kb_hits_for_agent_context(hits: list[dict[str, Any]] | None) -> str | None:
    """Format ``query_kb_multi_mmr`` hits the same way as POST /agent/decide."""
    if not hits:
        return None
    lines: list[str] = []
    for h in hits:
        rr = h.get("rerank_score")
        rrs = f"{float(rr):.3f}" if rr is not None else "n/a"
        lines.append(f"- (sim={h['score']:.3f} rerank={rrs}) {h['text'][:800]}")
    return "\n\n".join(lines)


def default_rag_context_from_prediction_summary(
    db: Session,
    settings: Settings,
    summary: dict[str, Any],
) -> str | None:
    """
    Batch-level RAG string using the same defaults as POST /agent/decide when ``use_rag`` is true
    and ``kb_citations`` are not provided: first template's retrieval queries, multi-query MMR,
    ``final_k`` capped at 12, ``per_query_k=12``, ``mmr_lambda=0.55``, then ``query_kb`` fallback.
    """
    templates = build_rag_templates_from_summary(summary)
    queries: list[str] = []
    if templates and isinstance(templates[0], dict):
        rq = templates[0].get("retrieval_queries")
        if isinstance(rq, list):
            queries = [str(x) for x in rq if str(x).strip()]
    if not queries:
        queries = [
            "Security policy actions for network anomalies and attack classification outcomes. "
            f"Batch stats: flagged={summary.get('rows_flagged')}, total={summary.get('rows_total')}."
        ]
    raw_hits, _meta = query_kb_multi_mmr(
        db,
        settings,
        queries,
        final_k=min(max(settings.rag_top_k, 6), 12),
        per_query_k=12,
        mmr_lambda=0.55,
        kb_public_ids=None,
    )
    if raw_hits:
        return format_kb_hits_for_agent_context(raw_hits)
    q = (
        "Security policy actions for network anomalies and attack classification outcomes. "
        f"Batch stats: flagged={summary.get('rows_flagged')}, total={summary.get('rows_total')}."
    )
    hits = query_kb(db, settings, q, settings.rag_top_k, None)
    if hits:
        return "\n\n".join(f"- ({s:.3f}) {c.get('text', '')[:800]}" for s, c, _ in hits)
    return None
