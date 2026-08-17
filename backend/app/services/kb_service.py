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
from sqlalchemy.orm import Session, joinedload

from app.core.config import Settings
from app.models.domain import FileKind, JobStatus, KnowledgeBaseFile, ManagedFile
from app.rag.chunking import chunk_text, load_document_text
from app.rag.cross_encoder_rerank import normalize_scores, score_queries_passages_max, score_query_passages
from app.rag.vector_store import FaissKnowledgeIndex, _normalize
from app.services import file_service, prediction_service
from app.services.rag_templates_from_predictions import build_rag_templates_from_summary
from app.services.rag_templates_row_context import (
    build_row_agent_templates,
    build_templated_rag_retrieval_query,
)
from app.utils.file_utils import remove_path

logger = logging.getLogger(__name__)

# Pipeline run artifacts (traffic summaries, customer messages) are not KB documents.
_PIPELINE_ARTIFACT_PREFIXES = ("traffic_run_", "customer_message_")
_PIPELINE_ARTIFACT_MARKERS = ("_traffic_run_", "_customer_message_")

_RRF_K = 60
# Oversample FAISS hits before CrossEncoder so CE can pick better than bi-encoder top-k.
_CE_OVERSAMPLE = 4
_CE_POOL_CAP = 48


def is_pipeline_run_artifact_name(name: str | None) -> bool:
    """True for auto-generated run files that must not live in the knowledge base."""
    n = Path(name or "").name.lower()
    return n.startswith(_PIPELINE_ARTIFACT_PREFIXES) or any(m in n for m in _PIPELINE_ARTIFACT_MARKERS)


async def ingest_kb_document(
    db: Session,
    settings: Settings,
    upload: UploadFile,
) -> KnowledgeBaseFile:
    if is_pipeline_run_artifact_name(upload.filename):
        raise HTTPException(400, "Run artifacts are not knowledge documents; upload a PDF/TXT/MD/JSON guide instead")
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


def list_kb_files(db: Session) -> list[dict[str, Any]]:
    rows = list(
        db.scalars(
            select(KnowledgeBaseFile)
            .options(joinedload(KnowledgeBaseFile.managed_file))
            .order_by(KnowledgeBaseFile.created_at.desc())
        ).unique().all()
    )
    out: list[dict[str, Any]] = []
    for kb in rows:
        if _kb_is_pipeline_artifact(kb):
            continue
        mf = kb.managed_file
        out.append(
            {
                "id": kb.id,
                "public_id": kb.public_id,
                "managed_file_id": kb.managed_file_id,
                "vector_index_dir": kb.vector_index_dir,
                "chunk_count": kb.chunk_count,
                "embedding_model": kb.embedding_model,
                "created_at": kb.created_at,
                "original_name": mf.original_name if mf else None,
                "managed_file_public_id": mf.public_id if mf else None,
            }
        )
    return out


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


def purge_pipeline_run_kb_artifacts(db: Session, settings: Settings) -> dict[str, Any]:
    """Remove traffic_run_ / customer_message_ files from KB rows, vector indexes, and disk."""
    rows = list(
        db.scalars(
            select(KnowledgeBaseFile).options(joinedload(KnowledgeBaseFile.managed_file))
        )
        .unique()
        .all()
    )
    deleted_ids: list[str] = []
    for kb in rows:
        mf = kb.managed_file
        names = [mf.original_name if mf else "", mf.storage_path if mf else "", kb.vector_index_dir]
        if any(is_pipeline_run_artifact_name(n) for n in names):
            deleted_ids.append(kb.public_id)

    for pid in deleted_ids:
        delete_kb(db, settings, pid)

    orphan_files: list[str] = []
    knowledge_dir = settings.storage_root / "knowledge"
    if knowledge_dir.is_dir():
        for path in knowledge_dir.iterdir():
            if path.is_file() and is_pipeline_run_artifact_name(path.name):
                remove_path(path)
                orphan_files.append(path.name)

    return {"deleted_kb_public_ids": deleted_ids, "deleted_orphan_files": orphan_files}


def _kb_is_pipeline_artifact(kb: KnowledgeBaseFile) -> bool:
    mf = kb.managed_file
    names = [mf.original_name if mf else "", mf.storage_path if mf else ""]
    return any(is_pipeline_run_artifact_name(n) for n in names)


def _load_kb_rows_for_rag(db: Session, kb_public_ids: list[str] | None) -> list[KnowledgeBaseFile]:
    q = select(KnowledgeBaseFile).options(joinedload(KnowledgeBaseFile.managed_file))
    if kb_public_ids:
        q = q.where(KnowledgeBaseFile.public_id.in_(kb_public_ids))
    rows = list(db.scalars(q).unique().all())
    return [kb for kb in rows if not _kb_is_pipeline_artifact(kb)]


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
    rows = _load_kb_rows_for_rag(db, kb_public_ids)
    if not rows:
        return []

    fetch_k = min(max(top_k * _CE_OVERSAMPLE, top_k), 40) if settings.rag_use_cross_encoder else top_k
    hits: list[tuple[float, dict, str]] = []
    stores = [(_open_store(settings, kb), kb) for kb in rows]
    for store, kb in stores:
        for score, chunk in store.search(query, fetch_k):
            hits.append((score, chunk, kb.public_id))
    hits.sort(key=lambda x: x[0], reverse=True)
    # Dedupe by text hash across KBs, keep best vector score.
    seen: set[str] = set()
    deduped: list[tuple[float, dict, str]] = []
    for score, chunk, kb_id in hits:
        text = str(chunk.get("text") or "")
        fk = _chunk_fusion_key(kb_id, text)
        if fk in seen:
            continue
        seen.add(fk)
        deduped.append((score, chunk, kb_id))
    pool = deduped[: min(_CE_POOL_CAP, len(deduped))]

    if settings.rag_use_cross_encoder and pool:
        passages = [str(c.get("text") or "") for _, c, _ in pool]
        ce_raw = score_query_passages(
            query,
            passages,
            model_name=settings.rag_cross_encoder_model,
        )
        if ce_raw is not None and len(ce_raw) == len(pool):
            ranked = sorted(
                zip(ce_raw, pool),
                key=lambda x: x[0],
                reverse=True,
            )
            return [(float(ce), chunk, kb_id) for ce, (_, chunk, kb_id) in ranked[:top_k]]

    return pool[:top_k]


def query_kb_single(
    db: Session,
    settings: Settings,
    query: str,
    *,
    final_k: int = 10,
    kb_public_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Single FAISS (+ optional CrossEncoder) retrieval for one query string."""
    q = str(query or "").strip()
    if not q:
        return [], {"queries_used": [], "final_k": final_k, "fusion": "empty_query", "pipeline": "none"}
    raw = query_kb(db, settings, q, final_k, kb_public_ids)
    hits: list[dict[str, Any]] = []
    for score, chunk, kb_id in raw:
        sim = float(score)
        hits.append(
            {
                "score": sim,
                "text": str(chunk.get("text") or ""),
                "source": chunk.get("source"),
                "kb_public_id": kb_id,
                "rerank_score": sim if settings.rag_use_cross_encoder else None,
                "mmr_margin": None,
            }
        )
    meta: dict[str, Any] = {
        "queries_used": [q],
        "final_k": final_k,
        "fusion": "single_templated_query",
        "use_mmr": False,
        "pipeline": "templated_query_faiss_cross_encoder",
    }
    return hits, meta


def query_kb_templated_rag(
    db: Session,
    settings: Settings,
    *,
    summary: dict[str, Any],
    row: dict[str, Any] | None = None,
    final_k: int = 10,
    kb_public_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Agent/pipeline RAG: one query from static template + prediction values + SHAP contributions."""
    q = build_templated_rag_retrieval_query(summary, row)
    hits, meta = query_kb_single(db, settings, q, final_k=final_k, kb_public_ids=kb_public_ids)
    meta["template"] = "STATIC_RAG_RETRIEVAL_TEMPLATE"
    meta["row_index"] = row.get("row_index") if isinstance(row, dict) else None
    return hits, meta


def query_kb_static_rag(
    db: Session,
    settings: Settings,
    *,
    final_k: int = 10,
    kb_public_ids: list[str] | None = None,
    summary: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Backward-compatible wrapper — prefers templated query when summary is provided."""
    if summary is not None:
        return query_kb_templated_rag(
            db, settings, summary=summary, row=row, final_k=final_k, kb_public_ids=kb_public_ids
        )
    fallback = build_templated_rag_retrieval_query(
        {"rows_total": 0, "rows_flagged": 0, "head_json": []},
        None,
    )
    return query_kb_single(db, settings, fallback, final_k=final_k, kb_public_ids=kb_public_ids)


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
    Deduped fusion map → RRF/max pool → CrossEncoder (max over queries) → optional MMR → hit dicts.

    RRF+max only builds the candidate pool. Final order prefers raw CrossEncoder scores (CE-first).
    """
    if not fused or not rows:
        return [], meta

    items = list(fused.values())
    max_scores = [x["max_score"] for x in items]
    rrfs = [x["rrf"] for x in items]
    n_max = _norm_list(max_scores)
    n_rrf = _norm_list(rrfs)
    for i, it in enumerate(items):
        it["fusion_score"] = 0.55 * n_max[i] + 0.45 * n_rrf[i]
        it["fusion_rerank"] = it["fusion_score"]  # back-compat alias

    # Candidate pool by fusion only (retrieve stage).
    items.sort(key=lambda x: x["fusion_score"], reverse=True)
    pool_n = min(
        max(final_k * max(pool_multiplier, _CE_OVERSAMPLE), final_k + 8, 24),
        _CE_POOL_CAP,
        len(items),
    )
    pool = items[:pool_n]

    qs = [str(q).strip() for q in queries if str(q).strip()]
    ce_used = False
    if settings.rag_use_cross_encoder and pool and qs:
        passages = [str(p["chunk"].get("text") or "") for p in pool]
        ce_raw = score_queries_passages_max(
            qs,
            passages,
            model_name=settings.rag_cross_encoder_model,
        )
        if ce_raw is not None and len(ce_raw) == len(pool):
            ce_used = True
            for i, it in enumerate(pool):
                it["crossencoder_score"] = float(ce_raw[i])
                # CE-first: final ranking key is raw CE logit (higher = better).
                it["rerank_score"] = float(ce_raw[i])
            pool.sort(key=lambda x: x["rerank_score"], reverse=True)

    if not ce_used:
        for it in pool:
            it["rerank_score"] = float(it["fusion_score"])
            it["crossencoder_score"] = None

    store0 = _open_store(settings, rows[0])
    model = store0.model
    texts = [p["chunk"].get("text") or "" for p in pool]
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    emb = _normalize(emb)

    q_vecs = model.encode(qs if qs else [""], convert_to_numpy=True, show_progress_bar=False).astype("float32")
    q_vecs = _normalize(q_vecs)
    q_centroid = _normalize(np.mean(q_vecs, axis=0, keepdims=True))
    sim_q = (emb @ q_centroid.T).flatten()

    # MMR relevance: CE-first when available (normalize for stable λ scale), else bi-encoder sim.
    if ce_used:
        rel_scores = np.array(
            normalize_scores([float(p["rerank_score"]) for p in pool]),
            dtype="float32",
        )
    else:
        rel_scores = sim_q.astype("float32")

    selected: list[int] = []
    mmr_margins: list[float | None] = []

    if use_mmr:
        lam = max(0.0, min(1.0, mmr_lambda))
        remaining = set(range(len(pool)))
        while len(selected) < final_k and remaining:
            best_i: int | None = None
            best_mmr = -1e18
            for i in remaining:
                rel = float(rel_scores[i])
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
        # Pure CE-first (or fusion) top-k — no diversity pass.
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
                "rerank_score": float(p["rerank_score"]),
                "mmr_margin": float(mmr_val) if mmr_val is not None else None,
            }
        )
    meta["pool_size"] = pool_n
    meta["candidates_fused"] = len(fused)
    meta["cross_encoder"] = {
        "enabled": bool(settings.rag_use_cross_encoder),
        "used": ce_used,
        "model": settings.rag_cross_encoder_model if ce_used else None,
        "scoring": "max_over_queries" if ce_used else None,
        "final_order": "cross_encoder_first" if ce_used else "fusion_only",
    }
    fusion_label = "fusion_pool_ce_max_then_mmr" if (ce_used and use_mmr) else (
        "fusion_pool_ce_max_topk" if ce_used else (
            "rrf_plus_max_score_then_mmr" if use_mmr else "rrf_plus_max_score_topk"
        )
    )
    meta["fusion"] = fusion_label
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
    Multi-query retrieval: per-query FAISS hits, RRF + max-score fusion, CrossEncoder
    rerank, then optional MMR diversification on the CE-ranked pool.
    """
    rows = _load_kb_rows_for_rag(db, kb_public_ids)
    meta: dict[str, Any] = {
        "queries_used": queries,
        "per_query_k": per_query_k,
        "final_k": final_k,
        "mmr_lambda": mmr_lambda,
        "fusion": "pending",
        "use_mmr": use_mmr,
        "pipeline": "faiss_rrf_cross_encoder_mmr",
    }
    if not rows or not queries:
        return [], meta

    fused: dict[str, dict[str, Any]] = {}
    stores = [(kb, _open_store(settings, kb)) for kb in rows]
    for query in queries:
        qstr = query.strip()
        if not qstr:
            continue
        for kb, store in stores:
            raw = store.search(qstr, per_query_k)
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
    dedupe + RRF/max fusion + CrossEncoder rerank + MMR as query_kb_multi_mmr.
    """
    rows = _load_kb_rows_for_rag(db, kb_public_ids)
    meta: dict[str, Any] = {
        "queries_used": queries,
        "final_k": final_k,
        "mmr_lambda": mmr_lambda,
        "fusion": "pending",
        "use_mmr": use_mmr,
        "pipeline": "sequential_kb_query_then_fuse_cross_encoder_mmr",
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
    """Format KB retrieval hits the same way as POST /agent/decide."""
    if not hits:
        return None
    lines: list[str] = []
    for h in hits:
        rr = h.get("rerank_score")
        rrs = f"{float(rr):.3f}" if rr is not None else "n/a"
        src = str(h.get("source") or "").strip()
        src_bit = f" source={src}" if src else ""
        lines.append(f"- (sim={h['score']:.3f} rerank={rrs}{src_bit}) {h['text'][:800]}")
    return "\n\n".join(lines)


def default_rag_context_from_prediction_summary(
    db: Session,
    settings: Settings,
    summary: dict[str, Any],
) -> str | None:
    """Batch-level RAG via single templated query (prediction stats, no row SHAP)."""
    raw_hits, _meta = query_kb_templated_rag(db, settings, summary=summary, row=None, final_k=10)
    if raw_hits:
        return format_kb_hits_for_agent_context(raw_hits[:10])
    return None
