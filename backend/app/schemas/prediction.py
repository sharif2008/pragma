"""Prediction and agentic schemas."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.domain import JobStatus
from app.schemas.common import ORMModel
from app.schemas.kb import KBQueryHit


class PredictionStartRequest(BaseModel):
    model_version_public_id: str
    input_file_public_id: str
    anomaly_probability_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="If set, rows with max class probability below this are flagged as uncertain/anomaly-like.",
    )
    attack_label_values: list[str] | None = Field(
        default=None,
        description="Label values treated as 'attack' for counting (case-sensitive).",
    )


class PredictionJobOut(ORMModel):
    id: int
    public_id: str
    model_version_id: int
    input_file_id: int
    status: JobStatus
    output_path: str | None
    rows_total: int | None
    rows_flagged: int | None
    config_json: dict | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class AgenticDecideRequest(BaseModel):
    prediction_job_public_id: str
    use_rag: bool = True
    feature_notes: str | None = Field(
        default=None, description="Optional analyst notes or SHAP summary text."
    )
    extra_context: dict[str, Any] | None = None
    kb_citations: list[KBQueryHit] | None = Field(
        default=None,
        description="KB chunks from multi-query+MMR (Knowledge base tab). If set with use_rag, used as RAG context.",
    )
    agent_action_preset: Literal["standard", "containment_focus", "fp_review"] | None = Field(
        default="standard",
        description="Fixed analyst emphasis appended to feature_notes for the LLM.",
    )


class AgenticReportOut(ORMModel):
    id: int
    public_id: str
    prediction_job_id: int
    prediction_job_public_id: str | None = Field(
        default=None, description="UUID of the prediction job this report used."
    )
    summary: str
    recommended_action: str
    raw_llm_response: str | None
    rag_context_used: str | None
    report_path: str | None
    created_at: datetime
