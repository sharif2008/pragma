"""Agentic report listing and lookup."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import AgenticReport, PredictionJob
from app.schemas.prediction import AgenticReportOut


def agentic_report_out(db: Session, row: AgenticReport) -> AgenticReportOut:
    pj = db.get(PredictionJob, row.prediction_job_id)
    return AgenticReportOut.model_validate(row).model_copy(
        update={"prediction_job_public_id": pj.public_id if pj else None}
    )


def list_agentic_reports(db: Session, *, limit: int = 100, offset: int = 0) -> list[AgenticReport]:
    q = (
        select(AgenticReport)
        .order_by(AgenticReport.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def get_agentic_report(db: Session, public_id: str) -> AgenticReport:
    row = db.scalar(select(AgenticReport).where(AgenticReport.public_id == public_id))
    if not row:
        from fastapi import HTTPException

        raise HTTPException(404, "Agentic report not found")
    return row
