"""Shared Pydantic types."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    detail: str


class Paginated(BaseModel):
    total: int
    items: list[Any]


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class TimestampMixin(BaseModel):
    created_at: datetime | None = None
