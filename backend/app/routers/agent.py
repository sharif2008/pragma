"""Agentic LLM decisions from prediction outputs + optional RAG."""

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import AgenticJob
from app.schemas.prediction import (
    AgenticDecideRequest,
    AgenticJobCreate,
    AgenticJobOut,
    AgenticPromptPreviewOut,
    AgenticReportOut,
)
from app.services import agent_service, kb_service, llm_service, prediction_service
from app.services.agentic_llm_prompt import (
    build_agentic_decide_user_prompt,
    build_sample_from_prediction_job,
    load_attack_agentic_config,
)
from app.services.rag_templates_from_predictions import build_rag_templates_from_summary

router = APIRouter(prefix="/agent", tags=["agent"])


def _load_report_artifact_json(settings: Settings, row: AgenticReport) -> dict | None:
    if not row.report_path:
        return None
    path = settings.storage_root / row.report_path
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError):
        return None


def _enrich_report_out_with_trust_json(
    settings: Settings, row: AgenticReport, out: AgenticReportOut
) -> AgenticReportOut:
    """Attach trust fields from on-disk report JSON when present (not stored in DB)."""
    if not row.report_path:
        return out
    path = settings.storage_root / row.report_path
    if not path.is_file():
        return out
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        updates: dict = {}
        if out.results_row_index is None:
            ri = data.get("results_row_index")
            if isinstance(ri, int) and ri >= 0:
                updates["results_row_index"] = ri
        if out.agentic_job_public_id is None:
            aj = data.get("agentic_job_public_id")
            if isinstance(aj, str) and aj.strip():
                updates["agentic_job_public_id"] = aj.strip()
        block = data.get("trust_chain")
        if isinstance(block, dict):
            tc = block.get("commitment_sha256")
            mode = block.get("mode")
            if isinstance(tc, str) and tc.strip():
                updates["trust_commitment"] = tc.strip()
                updates["trust_chain_mode"] = str(mode) if mode is not None else None
        if updates:
            return out.model_copy(update=updates)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return out


def _merge_feature_notes(body: AgenticDecideRequest) -> str | None:
    preset = body.agent_action_preset or "standard"
    base = (body.feature_notes or "").strip()
    extra = ""
    if preset == "containment_focus":
        extra = "Analyst preset: prioritize containment, isolation, and safe shutdown steps."
    elif preset == "fp_review":
        extra = "Analyst preset: emphasize false-positive checks, evidence quality, and baseline comparison."
    if not base and not extra:
        return None
    if base and extra:
        return f"{base}\n\n{extra}"
    return base or extra


def _build_rag_context(
    db: Session,
    settings: Settings,
    summary: dict,
    body: AgenticDecideRequest,
) -> str | None:
    if not body.use_rag:
        return None
    if body.kb_citations:
        lines: list[str] = []
        for i, h in enumerate(body.kb_citations, start=1):
            rs = f" rerank={h.rerank_score:.3f}" if h.rerank_score is not None else ""
            mm = f" mmr={h.mmr_margin:.3f}" if h.mmr_margin is not None else ""
            src = f" [{h.source}]" if h.source else ""
            lines.append(f"[{i}] sim={h.score:.3f}{rs}{mm}{src}\n{h.text[:900]}")
        return "\n\n".join(lines) if lines else None

    templates = build_rag_templates_from_summary(summary)
    queries = (
        templates[0]["retrieval_queries"]
        if templates
        else [
            "Security policy actions for network anomalies and attack classification outcomes. "
            f"Batch stats: flagged={summary.get('rows_flagged')}, total={summary.get('rows_total')}."
        ]
    )
    raw_hits, _meta = kb_service.query_kb_multi_mmr(
        db,
        settings,
        queries,
        final_k=min(max(settings.rag_top_k, 6), 12),
        per_query_k=12,
        mmr_lambda=0.55,
        kb_public_ids=None,
    )
    if not raw_hits:
        q = (
            "Security policy actions for network anomalies and attack classification outcomes. "
            f"Batch stats: flagged={summary.get('rows_flagged')}, total={summary.get('rows_total')}."
        )
        hits = kb_service.query_kb(db, settings, q, settings.rag_top_k, None)
        if hits:
            return "\n\n".join(f"- ({s:.3f}) {c.get('text', '')[:800]}" for s, c, _ in hits)
        return None
    lines = []
    for h in raw_hits:
        rr = h.get("rerank_score")
        rrs = f"{float(rr):.3f}" if rr is not None else "n/a"
        lines.append(f"- (sim={h['score']:.3f} rerank={rrs}) {h['text'][:800]}")
    return "\n\n".join(lines)


def _prepare_agent_decide_llm_inputs(
    db: Session,
    settings: Settings,
    body: AgenticDecideRequest,
):
    job = prediction_service.get_prediction_job(db, body.prediction_job_public_id)
    summary = prediction_service.load_prediction_summary(settings, job)
    rag_context = _build_rag_context(db, settings, summary, body)
    feature_notes = _merge_feature_notes(body)
    attack_actions_data, agentic_features_data = load_attack_agentic_config(verbose=False)
    sample_data = build_sample_from_prediction_job(
        job, summary, results_row_index=body.results_row_index
    )
    return job, sample_data, rag_context, feature_notes, attack_actions_data, agentic_features_data


@router.post("/decide/prompt-preview", response_model=AgenticPromptPreviewOut)
def agent_decide_prompt_preview(
    body: AgenticDecideRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticPromptPreviewOut:
    """Return the exact user prompt POST /agent/decide would send to the model (no LLM call)."""
    _job, sample_data, rag_context, feature_notes, attack_actions_data, agentic_features_data = (
        _prepare_agent_decide_llm_inputs(db, settings, body)
    )
    prompt = build_agentic_decide_user_prompt(
        sample_data,
        rag_context,
        attack_actions_data,
        agentic_features_data,
        include_knowledge_base=body.use_rag,
        feature_notes=feature_notes,
    )
    return AgenticPromptPreviewOut(prompt=prompt)


@router.get("/jobs", response_model=list[AgenticJobOut])
def list_agentic_jobs_api(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AgenticJobOut]:
    """List registered agentic jobs (newest activity first), joined with prediction batch status."""
    rows = agent_service.list_agentic_jobs(db, limit=limit, offset=offset)
    return [agent_service.agentic_job_out(db, r) for r in rows]


@router.post("/jobs", response_model=AgenticJobOut)
def create_agentic_job_api(
    body: AgenticJobCreate,
    db: Annotated[Session, Depends(get_db)],
) -> AgenticJobOut:
    """Persist an agentic job row (e.g. from RAG prep handoff) for the Agentic actions dropdown."""
    row = agent_service.create_agentic_job(
        db,
        prediction_job_public_id=body.prediction_job_public_id.strip(),
        results_row_index=body.results_row_index,
        label=body.label,
    )
    return agent_service.agentic_job_out(db, row)


@router.get("/reports", response_model=list[AgenticReportOut])
def list_agent_reports(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    agentic_job_public_id: str | None = Query(
        default=None,
        description="If set, return only reports linked to this agentic_jobs.public_id (404 if unknown).",
    ),
) -> list[AgenticReportOut]:
    """List agentic action records (newest first). Empty list when the job has no reports yet."""
    agentic_job_id: int | None = None
    if agentic_job_public_id and (aid := agentic_job_public_id.strip()):
        aj = db.scalar(select(AgenticJob).where(AgenticJob.public_id == aid))
        if not aj:
            raise HTTPException(404, "Agentic job not found")
        agentic_job_id = aj.id
    rows = agent_service.list_agentic_reports(
        db, limit=limit, offset=offset, agentic_job_id=agentic_job_id
    )
    return [agent_service.agentic_report_out(db, r) for r in rows]


@router.get("/reports/{public_id}", response_model=AgenticReportOut)
def get_agent_report(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticReportOut:
    """Single agentic report by public id."""
    row = agent_service.get_agentic_report(db, public_id)
    out = agent_service.agentic_report_out(db, row)
    out = _enrich_report_out_with_trust_json(settings, row, out)
    artifact = _load_report_artifact_json(settings, row)
    if artifact is not None:
        out = out.model_copy(update={"report_artifact": artifact})
    return out


@router.delete("/reports/{public_id}", status_code=204)
def delete_agent_report(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Remove an agentic report row and its on-disk JSON (if under storage_root)."""
    agent_service.delete_agentic_report(db, settings, public_id)


@router.post("/decide", response_model=AgenticReportOut)
async def agent_decide(
    body: AgenticDecideRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticReportOut:
    job, sample_data, rag_context, feature_notes, attack_actions_data, agentic_features_data = (
        _prepare_agent_decide_llm_inputs(db, settings, body)
    )

    linked_agentic_job = agent_service.resolve_agentic_job_for_decide(
        db,
        agentic_job_public_id=body.agentic_job_public_id,
        prediction_job_id=job.id,
        results_row_index=body.results_row_index,
    )
    agentic_job_id = linked_agentic_job.id if linked_agentic_job else None
    agentic_job_public_resolved = linked_agentic_job.public_id if linked_agentic_job else None

    decision = await llm_service.agent_decide(
        settings,
        sample_data,
        feature_notes,
        rag_context,
        attack_actions_data,
        agentic_features_data,
        use_rag=body.use_rag,
    )

    user_prompt = build_agentic_decide_user_prompt(
        sample_data,
        rag_context,
        attack_actions_data,
        agentic_features_data,
        include_knowledge_base=body.use_rag,
        feature_notes=feature_notes,
    )

    row, payload = agent_service.persist_agentic_report_from_decision(
        db,
        settings,
        job=job,
        results_row_index=body.results_row_index,
        agentic_job_id=agentic_job_id,
        agentic_job_public_id=agentic_job_public_resolved,
        sample_data=sample_data,
        user_prompt=user_prompt,
        decision=decision,
    )
    report_path = settings.storage_root / row.report_path

    trust_commitment: str | None = None
    trust_chain_mode: str | None = None
    if body.anchor_trust_chain:
        anchored_at = datetime.now(timezone.utc).isoformat()
        msg = "|".join(
            [
                row.public_id,
                job.public_id,
                str(decision["summary"]),
                str(decision["recommended_action"]),
                anchored_at,
            ]
        )
        trust_commitment = hashlib.sha256(msg.encode("utf-8")).hexdigest()
        trust_chain_mode = "demo_local_commitment"
        payload["trust_chain"] = {
            "mode": trust_chain_mode,
            "commitment_sha256": trust_commitment,
            "anchored_at_utc": anchored_at,
            "report_public_id": row.public_id,
            "prediction_job_public_id": job.public_id,
            "note": "Demo API: SHA-256 over report id, job id, summary, action, and time. "
            "Swap for on-chain notary / enterprise log in production.",
        }
        report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    out = agent_service.agentic_report_out(db, row)
    if trust_commitment is not None:
        out = out.model_copy(
            update={"trust_commitment": trust_commitment, "trust_chain_mode": trust_chain_mode}
        )
    return out
