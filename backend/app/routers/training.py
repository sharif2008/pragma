"""Model training jobs and model registry."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import JobStatus
from app.schemas.training import (
    ModelVersionOut,
    TrainingJobOut,
    TrainingRebuildRequest,
    TrainingStartRequest,
    TrainingStartResponse,
)
from app.services import training_service

router = APIRouter(prefix="/training", tags=["training"])


@router.get("", response_model=list[TrainingJobOut])
def list_training_jobs(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[TrainingJobOut]:
    """All training jobs (newest first), with dataset file trace for VFL workflows."""
    jobs = training_service.list_training_jobs(db, limit=limit, offset=offset)
    return [training_service.job_to_out(db, j) for j in jobs]


@router.post("/start", response_model=TrainingStartResponse)
def start_training(
    body: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrainingStartResponse:
    try:
        job = training_service.create_training_job(
            db,
            settings,
            body.dataset_file_public_id,
            body.target_column,
            body.algorithm,
            body.test_size,
            body.random_state,
            body.xgboost_params,
            vfl_agent_definitions_path=body.vfl_agent_definitions_path,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    background_tasks.add_task(training_service.run_training_job_sync, job.id)
    return TrainingStartResponse(job_public_id=job.public_id, status=JobStatus.pending)


@router.post("/rebuild", response_model=TrainingStartResponse)
def rebuild_training(
    body: TrainingRebuildRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> TrainingStartResponse:
    """Start a new job with the same dataset CSV and hyperparameters as an existing job (retrain / rebuild model)."""
    try:
        job = training_service.rebuild_training_job(db, settings, body.from_job_public_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    background_tasks.add_task(training_service.run_training_job_sync, job.id)
    return TrainingStartResponse(job_public_id=job.public_id, status=JobStatus.pending)


@router.get("/{public_id}", response_model=TrainingJobOut)
def get_training(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> TrainingJobOut:
    job = training_service.get_training_job(db, public_id)
    return training_service.job_to_out(db, job)


models_router = APIRouter(prefix="/models", tags=["models"])


@models_router.get("", response_model=list[ModelVersionOut])
def list_models(db: Annotated[Session, Depends(get_db)]) -> list[ModelVersionOut]:
    return training_service.list_models(db)
