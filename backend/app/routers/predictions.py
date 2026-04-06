"""Batch prediction jobs."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import FileKind
from app.schemas.files import FileUploadResponse, ManagedFileOut
from app.schemas.prediction import (
    PendingPredictionPurgeOut,
    PredictionJobOut,
    PredictionJobSummaryOut,
    PredictionStartRequest,
)
from app.services import file_service, prediction_service

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.post("/upload-input", response_model=FileUploadResponse)
async def upload_prediction_csv(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
) -> FileUploadResponse:
    row = await file_service.upload_file(db, settings, file, FileKind.prediction_input, replace_public_id=None)
    return FileUploadResponse(
        public_id=row.public_id,
        original_name=row.original_name,
        version=row.version,
        file_kind=row.file_kind,
    )


@router.post("/start", response_model=PredictionJobOut)
def start_prediction(
    body: PredictionStartRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[Session, Depends(get_db)],
) -> PredictionJobOut:
    cfg = {
        "anomaly_probability_threshold": body.anomaly_probability_threshold,
        "attack_label_values": body.attack_label_values or [],
        "compute_shap": body.compute_shap,
    }
    try:
        job = prediction_service.create_prediction_job(
            db,
            body.model_version_public_id,
            body.input_file_public_id,
            cfg,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e)) from e
    background_tasks.add_task(prediction_service.run_prediction_job_sync, job.id)
    mk = prediction_service.results_model_kind_from_job(job)
    return PredictionJobOut.model_validate(job).model_copy(update={"results_model_kind": mk})


@router.get("", response_model=list[PredictionJobSummaryOut])
def list_predictions(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> list[PredictionJobSummaryOut]:
    """List inference jobs (newest first) for tracking (no per-row ``results_json``)."""
    jobs = prediction_service.list_prediction_jobs(db, limit=limit, offset=offset)
    out: list[PredictionJobSummaryOut] = []
    for j in jobs:
        mk = prediction_service.results_model_kind_from_job(j)
        out.append(PredictionJobSummaryOut.model_validate(j).model_copy(update={"results_model_kind": mk}))
    return out


@router.get("/inputs", response_model=list[ManagedFileOut])
def list_prediction_inputs(db: Annotated[Session, Depends(get_db)]) -> list[ManagedFileOut]:
    """CSV files uploaded for batch inference (newest first). Declared before /{public_id} so 'inputs' is not parsed as a job id."""
    return file_service.list_files(db, file_kind=FileKind.prediction_input)


@router.delete("/pending", response_model=PendingPredictionPurgeOut)
def delete_all_pending_predictions(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> PendingPredictionPurgeOut:
    """Remove all jobs still in ``pending`` (e.g. stuck before the worker runs). Must be registered before ``DELETE /{public_id}``."""
    n = prediction_service.delete_all_pending_prediction_jobs(db, settings)
    return PendingPredictionPurgeOut(deleted=n)


@router.get("/{public_id}", response_model=PredictionJobOut)
def get_prediction(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    include_results: bool = Query(
        default=False,
        description="If true, include ``results_json`` (per-row predictions + SHAP). Can be large.",
    ),
) -> PredictionJobOut:
    job = prediction_service.get_prediction_job(db, public_id)
    mk = prediction_service.results_model_kind_from_job(job)
    out = PredictionJobOut.model_validate(job).model_copy(update={"results_model_kind": mk})
    if not include_results:
        return out.model_copy(update={"results_json": None})
    return out


@router.delete("/{public_id}", status_code=204)
def delete_prediction(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    """Delete a prediction job; removes stored output file if present. Pending (queued) jobs may be removed; blocked while running."""
    prediction_service.delete_prediction_job(db, settings, public_id)
