"""Agentic report listing and lookup."""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import AgenticJob, AgenticReport, JobStatus, PredictionJob
from app.schemas.prediction import AgenticJobOut, AgenticReportOut
from app.services import prediction_service
from app.utils.file_utils import remove_path

logger = logging.getLogger(__name__)


def agentic_job_out(db: Session, row: AgenticJob) -> AgenticJobOut:
    pj = db.get(PredictionJob, row.prediction_job_id)
    if not pj:
        return AgenticJobOut(
            public_id=row.public_id,
            prediction_job_public_id="",
            results_row_index=row.results_row_index,
            label=row.label,
            prediction_status=JobStatus.pending,
            rows_total=None,
            rows_flagged=None,
            results_model_kind=None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
    mk = prediction_service.results_model_kind_from_job(pj)
    return AgenticJobOut(
        public_id=row.public_id,
        prediction_job_public_id=pj.public_id,
        results_row_index=row.results_row_index,
        label=row.label,
        prediction_status=pj.status,
        rows_total=pj.rows_total,
        rows_flagged=pj.rows_flagged,
        results_model_kind=mk,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def create_agentic_job(
    db: Session,
    *,
    prediction_job_public_id: str,
    results_row_index: int | None,
    label: str | None,
) -> AgenticJob:
    pj = prediction_service.get_prediction_job(db, prediction_job_public_id)
    row = AgenticJob(
        prediction_job_id=pj.id,
        results_row_index=results_row_index,
        label=(label.strip() if isinstance(label, str) and label.strip() else None),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_agentic_jobs(db: Session, *, limit: int = 100, offset: int = 0) -> list[AgenticJob]:
    q = (
        select(AgenticJob)
        .order_by(AgenticJob.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def resolve_agentic_job_for_decide(
    db: Session,
    *,
    agentic_job_public_id: str | None,
    prediction_job_id: int,
    results_row_index: int | None,
) -> AgenticJob | None:
    if not agentic_job_public_id or not str(agentic_job_public_id).strip():
        return None
    from fastapi import HTTPException

    aj = db.scalar(select(AgenticJob).where(AgenticJob.public_id == str(agentic_job_public_id).strip()))
    if not aj:
        raise HTTPException(404, "Agentic job not found")
    if aj.prediction_job_id != prediction_job_id:
        raise HTTPException(400, "agentic_job_public_id does not match prediction_job_public_id")
    if aj.results_row_index != results_row_index:
        raise HTTPException(400, "agentic_job_public_id does not match results_row_index")
    return aj


def agentic_report_out(db: Session, row: AgenticReport) -> AgenticReportOut:
    pj = db.get(PredictionJob, row.prediction_job_id)
    aj = db.get(AgenticJob, row.agentic_job_id) if row.agentic_job_id else None
    return AgenticReportOut.model_validate(row).model_copy(
        update={
            "prediction_job_public_id": pj.public_id if pj else None,
            "agentic_job_public_id": aj.public_id if aj else None,
        }
    )


def list_agentic_reports(
    db: Session, *, limit: int = 100, offset: int = 0, agentic_job_id: int | None = None
) -> list[AgenticReport]:
    q = select(AgenticReport).order_by(AgenticReport.created_at.desc())
    if agentic_job_id is not None:
        q = q.where(AgenticReport.agentic_job_id == agentic_job_id)
    q = q.offset(offset).limit(limit)
    return list(db.scalars(q).all())


def get_agentic_report(db: Session, public_id: str) -> AgenticReport:
    row = db.scalar(select(AgenticReport).where(AgenticReport.public_id == public_id))
    if not row:
        from fastapi import HTTPException

        raise HTTPException(404, "Agentic report not found")
    return row


def delete_agentic_report(db: Session, settings: Settings, public_id: str) -> None:
    row = db.scalar(select(AgenticReport).where(AgenticReport.public_id == public_id))
    if not row:
        from fastapi import HTTPException

        raise HTTPException(404, "Agentic report not found")
    if row.report_path:
        out_abs = (settings.storage_root / row.report_path).resolve()
        root = settings.storage_root.resolve()
        try:
            out_abs.relative_to(root)
        except ValueError:
            logger.warning("Skipping removal of agent report file outside storage_root: %s", row.report_path)
        else:
            remove_path(out_abs)
    db.delete(row)
    db.commit()
