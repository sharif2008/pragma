"""Managed file CRUD and versioning."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import FileKind, ManagedFile
from app.utils.file_utils import remove_path, save_upload

logger = logging.getLogger(__name__)


ALLOWED_GENERAL = {
    "text/csv",
    "application/json",
    "application/pdf",
    "text/plain",
}
ALLOWED_TRAINING = {"text/csv"}
ALLOWED_KB = {"application/pdf", "text/plain", "text/markdown", "application/json"}


async def upload_file(
    db: Session,
    settings: Settings,
    upload: UploadFile,
    file_kind: FileKind,
    replace_public_id: str | None,
) -> ManagedFile:
    mime = upload.content_type or "application/octet-stream"
    if file_kind == FileKind.training_dataset and mime not in ALLOWED_TRAINING and not (
        upload.filename or ""
    ).lower().endswith(".csv"):
        raise HTTPException(400, "Training datasets must be CSV")
    if file_kind == FileKind.knowledge_doc:
        if mime not in ALLOWED_KB and not (upload.filename or "").lower().endswith((".pdf", ".txt", ".md", ".json")):
            raise HTTPException(400, "Knowledge documents: PDF, TXT, MD, or JSON")
    if file_kind == FileKind.upload and mime not in ALLOWED_GENERAL and not (upload.filename or "").lower().endswith(
        (".csv", ".pdf", ".json", ".txt")
    ):
        raise HTTPException(400, "Allowed: CSV, PDF, JSON, TXT for general upload")

    subdir = {
        FileKind.upload: settings.storage_root / "uploads",
        FileKind.training_dataset: settings.storage_root / "training_datasets",
        FileKind.prediction_input: settings.storage_root / "uploads",
        FileKind.knowledge_doc: settings.storage_root / "knowledge",
    }[file_kind]

    version = 1
    parent_id: int | None = None
    if replace_public_id:
        old = db.scalar(select(ManagedFile).where(ManagedFile.public_id == replace_public_id))
        if not old:
            raise HTTPException(404, "File to replace not found")
        version = old.version + 1
        parent_id = old.id

    dest, size = await save_upload(subdir, upload)
    rel = str(dest.relative_to(settings.storage_root))

    row = ManagedFile(
        original_name=upload.filename or "unnamed",
        storage_path=rel,
        mime_type=mime,
        file_kind=file_kind,
        version=version,
        parent_file_id=parent_id,
        size_bytes=size,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    logger.info("Stored file %s kind=%s v=%s", row.public_id, file_kind, version)
    return row


def list_files(db: Session, file_kind: FileKind | None = None) -> list[ManagedFile]:
    q = select(ManagedFile).order_by(ManagedFile.created_at.desc())
    if file_kind is not None:
        q = q.where(ManagedFile.file_kind == file_kind)
    return list(db.scalars(q).all())


def get_by_public_id(db: Session, public_id: str) -> ManagedFile:
    row = db.scalar(select(ManagedFile).where(ManagedFile.public_id == public_id))
    if not row:
        raise HTTPException(404, "File not found")
    return row


def delete_file(db: Session, settings: Settings, public_id: str) -> None:
    row = get_by_public_id(db, public_id)
    abs_path = settings.storage_root / row.storage_path
    remove_path(abs_path)
    db.delete(row)
    db.commit()
    logger.info("Deleted file %s", public_id)


def resolved_path(settings: Settings, row: ManagedFile) -> Path:
    return settings.storage_root / row.storage_path
