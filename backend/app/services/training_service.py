"""Training jobs lifecycle and background execution."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.models.domain import FileKind, JobStatus, ManagedFile, ModelVersion, TrainingJob
from app.services import file_service
from app.services.ml_training import train_from_csv

logger = logging.getLogger(__name__)


def create_training_job(
    db: Session,
    settings: Settings,
    dataset_file_public_id: str,
    target_column: str,
    algorithm: str,
    test_size: float,
    random_state: int,
    xgboost_params: dict[str, Any] | None,
    vfl_agent_definitions_path: str | None = None,
) -> TrainingJob:
    mf = file_service.get_by_public_id(db, dataset_file_public_id)
    if mf.file_kind != FileKind.training_dataset:
        raise ValueError("Dataset file must be uploaded with kind training_dataset")

    config = {
        "test_size": test_size,
        "random_state": random_state,
        "xgboost_params": xgboost_params,
        "vfl_agent_definitions_path": vfl_agent_definitions_path,
    }
    job = TrainingJob(
        status=JobStatus.pending,
        dataset_file_id=mf.id,
        target_column=target_column,
        algorithm=algorithm,
        config_json=config,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_training_job(db: Session, public_id: str) -> TrainingJob:
    job = db.scalar(select(TrainingJob).where(TrainingJob.public_id == public_id))
    if not job:
        from fastapi import HTTPException

        raise HTTPException(404, "Training job not found")
    return job


def list_training_jobs(db: Session, *, limit: int = 100, offset: int = 0) -> list[TrainingJob]:
    q = (
        select(TrainingJob)
        .order_by(TrainingJob.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def job_to_out(db: Session, job: TrainingJob):
    from app.schemas.training import TrainingJobOut

    mf = db.get(ManagedFile, job.dataset_file_id)
    mv_public: str | None = None
    if job.model_version_id:
        mv = db.get(ModelVersion, job.model_version_id)
        if mv:
            mv_public = mv.public_id
    base = TrainingJobOut.model_validate(job)
    return base.model_copy(
        update={
            "dataset_file_public_id": mf.public_id if mf else None,
            "dataset_original_name": mf.original_name if mf else None,
            "model_version_public_id": mv_public,
        }
    )


def rebuild_training_job(db: Session, settings: Settings, source_job_public_id: str) -> TrainingJob:
    src = get_training_job(db, source_job_public_id)
    mf = db.get(ManagedFile, src.dataset_file_id)
    if not mf:
        raise ValueError("Source job dataset file missing")
    cfg = src.config_json or {}
    return create_training_job(
        db,
        settings,
        mf.public_id,
        src.target_column,
        src.algorithm,
        float(cfg.get("test_size", 0.2)),
        int(cfg.get("random_state", 42)),
        cfg.get("xgboost_params"),
        vfl_agent_definitions_path=cfg.get("vfl_agent_definitions_path"),
    )


def list_models(db: Session) -> list[ModelVersion]:
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc())).all())


def _next_model_version_number(db: Session) -> int:
    m = db.scalar(select(func.max(ModelVersion.version_number)))
    return int(m or 0) + 1


def run_training_job_sync(job_db_id: int) -> None:
    """Executed inside BackgroundTasks (own DB session)."""
    settings = get_settings()
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_db_id)
        if not job:
            logger.error("Training job id=%s missing", job_db_id)
            return
        job.status = JobStatus.running
        db.commit()

        mf = db.get(ManagedFile, job.dataset_file_id)
        if not mf:
            job.status = JobStatus.failed
            job.error_message = "Dataset file row missing"
            db.commit()
            return

        csv_path = file_service.resolved_path(settings, mf)
        cfg = job.config_json or {}
        test_size = float(cfg.get("test_size", 0.2))
        random_state = int(cfg.get("random_state", 42))
        xgb_params = cfg.get("xgboost_params")
        vfl_defs = cfg.get("vfl_agent_definitions_path")
        repo_root = settings.storage_root.parent.parent

        artifact_name = f"model_{job.public_id}.joblib"
        artifact_abs = settings.storage_root / "models" / artifact_name

        try:
            result = train_from_csv(
                csv_path,
                job.target_column,
                job.algorithm,
                test_size=test_size,
                random_state=random_state,
                xgb_params=xgb_params,
                artifact_path=artifact_abs,
                vfl_agent_definitions_path=vfl_defs,
                repo_root=repo_root,
            )
        except Exception as e:
            logger.exception("Training failed job=%s", job.public_id)
            job.status = JobStatus.failed
            job.error_message = str(e)[:8000]
            db.commit()
            return

        rel_art = str(artifact_abs.relative_to(settings.storage_root))
        vnum = _next_model_version_number(db)
        mv = ModelVersion(
            version_number=vnum,
            training_job_id=job.id,
            algorithm=job.algorithm,
            artifact_path=rel_art,
            metrics_json=result["metrics"],
            feature_columns_json=result["feature_columns"],
            label_classes_json=result["label_classes"],
        )
        db.add(mv)
        db.flush()

        job.metrics_json = result["metrics"]
        job.model_version_id = mv.id
        job.status = JobStatus.completed
        job.error_message = None
        db.commit()
        logger.info("Training completed job=%s model_version=%s", job.public_id, mv.public_id)
    finally:
        db.close()
