"""Agentic LLM decisions from prediction outputs + optional RAG."""

import json
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import AgenticReport
from app.schemas.prediction import AgenticDecideRequest, AgenticReportOut
from app.services import agent_service, kb_service, llm_service, prediction_service
from app.services.rag_templates_from_predictions import build_rag_templates_from_summary

router = APIRouter(prefix="/agent", tags=["agent"])


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


@router.get("/reports", response_model=list[AgenticReportOut])
def list_agent_reports(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[AgenticReportOut]:
    """List agentic action records (newest first)."""
    rows = agent_service.list_agentic_reports(db, limit=limit, offset=offset)
    return [agent_service.agentic_report_out(db, r) for r in rows]


@router.get("/reports/{public_id}", response_model=AgenticReportOut)
def get_agent_report(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> AgenticReportOut:
    """Single agentic report by public id."""
    row = agent_service.get_agentic_report(db, public_id)
    return agent_service.agentic_report_out(db, row)


@router.post("/decide", response_model=AgenticReportOut)
async def agent_decide(
    body: AgenticDecideRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgenticReportOut:
    job = prediction_service.get_prediction_job(db, body.prediction_job_public_id)
    summary = prediction_service.load_prediction_summary(settings, job)

    rag_context = _build_rag_context(db, settings, summary, body)
    feature_notes = _merge_feature_notes(body)

    decision = await llm_service.agent_decide(
        settings,
        summary,
        feature_notes,
        rag_context,
    )

    report_path = settings.storage_root / "reports" / f"agentic_{job.public_id}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prediction_job_public_id": job.public_id,
        "summary": decision["summary"],
        "recommended_action": decision["recommended_action"],
        "raw_llm_response": decision.get("raw_llm_response"),
        "rag_context_used": decision.get("rag_context_used"),
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = str(report_path.relative_to(settings.storage_root))

    row = AgenticReport(
        prediction_job_id=job.id,
        summary=decision["summary"],
        recommended_action=decision["recommended_action"],
        raw_llm_response=decision.get("raw_llm_response"),
        rag_context_used=decision.get("rag_context_used"),
        report_path=rel,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return agent_service.agentic_report_out(db, row)
