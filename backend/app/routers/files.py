"""General file upload and listing."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import FileKind
from app.schemas.files import FileUploadResponse, ManagedFileOut
from app.services import file_service

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
    replace_public_id: str | None = Form(default=None),
) -> FileUploadResponse:
    row = await file_service.upload_file(
        db, settings, file, FileKind.upload, replace_public_id=replace_public_id
    )
    return FileUploadResponse(
        public_id=row.public_id,
        original_name=row.original_name,
        version=row.version,
        file_kind=row.file_kind,
    )


@router.get("", response_model=list[ManagedFileOut])
def list_files(
    db: Annotated[Session, Depends(get_db)],
) -> list[ManagedFileOut]:
    return file_service.list_files(db, file_kind=None)


@router.delete("/{public_id}", status_code=204)
def delete_file(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    file_service.delete_file(db, settings, public_id)
