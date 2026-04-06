"""File and dataset schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.domain import FileKind
from app.schemas.common import ORMModel


class ManagedFileOut(ORMModel):
    id: int
    public_id: str
    original_name: str
    storage_path: str
    mime_type: str | None
    file_kind: FileKind
    version: int
    parent_file_id: int | None
    size_bytes: int | None
    created_at: datetime


class FileUploadResponse(ORMModel):
    public_id: str
    original_name: str
    version: int
    file_kind: FileKind
    message: str = Field(default="uploaded")


class DatasetPreviewOut(BaseModel):
    """First rows of a training CSV for UI preview."""

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int = Field(description="Number of rows returned (may be less than file length)")
    preview_limit: int
