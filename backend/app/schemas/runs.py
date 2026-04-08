"""Run tracing schemas for simulated customer messages."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


AttachmentType = Literal["none", "image", "pdf", "audio", "text", "unknown"]
RunStatus = Literal["running", "completed", "failed", "partial", "needs_input"]


class SimulatedAttachment(BaseModel):
    filename: str | None = None
    content_type: str | None = None
    url: str | None = None


class SimulateControls(BaseModel):
    latency_ms: int = Field(default=0, ge=0, le=60000)
    force_error_step: str | None = None


class SimulateCustomerMessageRequest(BaseModel):
    customer_id: str | None = None
    channel: str | None = None
    message: str = Field(min_length=1, max_length=20000)
    attachments: list[SimulatedAttachment] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    simulate: SimulateControls = Field(default_factory=SimulateControls)
    idempotency_key: str | None = Field(
        default=None,
        description="Optional. If provided, repeated requests will return the same run_id.",
        max_length=128,
    )


class SimulateCustomerMessageResponse(BaseModel):
    run_id: str
    trace_id: str
    status: RunStatus
    status_url: str
    events_url: str


class RunPredictionOut(BaseModel):
    attachment_type: AttachmentType
    shape_constraints: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    predicted_label: str | None = None
    max_class_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    flagged_attack_or_anomaly: bool | None = None


class RunRagOut(BaseModel):
    stored: bool = False
    kb_public_ids: list[str] = Field(default_factory=list)
    managed_file_public_ids: list[str] = Field(default_factory=list)


class RunErrorOut(BaseModel):
    step_name: str | None = None
    message: str | None = None


class RunSummaryOut(BaseModel):
    run_id: str
    trace_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    customer_id: str | None = None
    channel: str | None = None
    message_preview: str | None = None
    duration_ms: int | None = None
    last_step: str | None = None
    predictions: RunPredictionOut | None = None
    predictions_payload: dict[str, Any] | None = Field(
        default=None,
        description="Full agent_runs.predictions_json (job id, batch counts, model_kind, etc.).",
    )
    rag: RunRagOut | None = None
    actions: list[dict[str, Any]] | None = None
    error: RunErrorOut | None = None


class RunEventOut(BaseModel):
    event_id: str
    run_id: str
    trace_id: str
    timestamp: datetime
    step_name: str
    level: Literal["info", "warn", "error"]
    message: str
    payload: dict[str, Any] | None = None
    duration_ms: int | None = None


class RunListItemOut(BaseModel):
    run_id: str
    trace_id: str
    status: RunStatus
    created_at: datetime
    updated_at: datetime
    customer_id: str | None = None
    channel: str | None = None
    message_preview: str | None = None
    predicted_attachment_type: str | None = None
    predicted_label: str | None = None
    flagged_attack_or_anomaly: bool | None = None
    duration_ms: int | None = None
    last_step: str | None = None
