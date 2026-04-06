"""Training dataset uploads (CSV) with versioning."""

import json
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import FileKind
from app.schemas.files import DatasetPreviewOut, FileUploadResponse, ManagedFileOut
from app.services import file_service

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_dataset(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    replace_public_id: str | None = Form(default=None),
) -> FileUploadResponse:
    row = await file_service.upload_file(
        db, settings, file, FileKind.training_dataset, replace_public_id=replace_public_id
    )
    return FileUploadResponse(
        public_id=row.public_id,
        original_name=row.original_name,
        version=row.version,
        file_kind=row.file_kind,
    )


@router.get("", response_model=list[ManagedFileOut])
def list_datasets(db: Annotated[Session, Depends(get_db)]) -> list[ManagedFileOut]:
    return file_service.list_files(db, file_kind=FileKind.training_dataset)


@router.get("/{public_id}/preview", response_model=DatasetPreviewOut)
def preview_dataset(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    limit: int = Query(default=25, ge=1, le=200, description="Max rows to return"),
) -> DatasetPreviewOut:
    row = file_service.get_by_public_id(db, public_id)
    if row.file_kind != FileKind.training_dataset:
        raise HTTPException(400, "Not a training dataset file")
    path = file_service.resolved_path(settings, row)
    if not path.is_file():
        raise HTTPException(404, "Dataset file missing on disk")
    try:
        df = pd.read_csv(path, nrows=limit)
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}") from e
    columns = list(df.columns)
    rows = json.loads(df.to_json(orient="records")) if len(df) else []
    return DatasetPreviewOut(
        columns=columns,
        rows=rows,
        row_count=len(rows),
        preview_limit=limit,
    )


@router.delete("/{public_id}", status_code=204)
def delete_dataset(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    row = file_service.get_by_public_id(db, public_id)
    if row.file_kind != FileKind.training_dataset:
        from fastapi import HTTPException

        raise HTTPException(400, "Not a training dataset file")
    file_service.delete_file(db, settings, public_id)
