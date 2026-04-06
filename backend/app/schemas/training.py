"""Training and model schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.domain import JobStatus
from app.schemas.common import ORMModel


AlgorithmName = Literal["vfl"]


class TrainingStartRequest(BaseModel):
    dataset_file_public_id: str = Field(..., description="Managed file UUID (training CSV)")
    target_column: str
    algorithm: AlgorithmName = Field(
        default="vfl",
        description="Vertical federated learning (3-party concat-embeddings) is the only supported trainer.",
    )
    test_size: float = Field(default=0.2, ge=0.05, le=0.5)
    random_state: int = 42
    vfl_agent_definitions_path: str | None = Field(
        default=None,
        description=(
            "Optional path to agentic_features.json (e.g. storage/agentic_features.json — RAN/Edge/Core "
            "logged_features) for vertical column split. If omitted, IDS-style heuristics assign columns to 3 parties."
        ),
    )


class TrainingJobOut(ORMModel):
    id: int
    public_id: str
    status: JobStatus
    dataset_file_id: int
    dataset_file_public_id: str | None = Field(
        default=None, description="Managed file UUID for the training CSV (VFL dataset trace)."
    )
    dataset_original_name: str | None = Field(default=None, description="Original uploaded filename.")
    target_column: str
    algorithm: str
    config_json: dict | None
    metrics_json: dict | None
    model_version_id: int | None
    model_version_public_id: str | None = Field(default=None, description="Registered model UUID when completed.")
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class ModelVersionOut(ORMModel):
    id: int
    public_id: str
    version_number: int
    training_job_id: int | None
    algorithm: str
    artifact_path: str
    metrics_json: dict | None
    feature_columns_json: list | None
    label_classes_json: list | None
    created_at: datetime


class TrainingStartResponse(BaseModel):
    job_public_id: str
    status: JobStatus
    message: str = "Training scheduled"


class TrainingRebuildRequest(BaseModel):
    """Queue a new training run using the same dataset and hyperparameters as a previous job."""

    from_job_public_id: str = Field(..., description="Existing training job public_id to copy settings from")
