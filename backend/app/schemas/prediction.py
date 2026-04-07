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
    compute_shap: bool = Field(
        default=True,
        description="If true, compute explanations: TreeExplainer SHAP for sklearn tree pipelines (when `shap` is installed), "
        "or gradient×input attribution per feature for VFL torch models (same `per_feature` shape for RAG templates). "
        "Skipped for large batches (see server row limit).",
    )


class PredictionJobSummaryOut(ORMModel):
    """Prediction job without per-row JSON (for list endpoints)."""

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
    results_model_kind: str | None = Field(
        default=None,
        description="From results_json.model_kind after a successful run (e.g. sklearn_pipeline, vfl_torch).",
    )


class PredictionJobOut(PredictionJobSummaryOut):
    """Full job including optional per-row predictions + SHAP stored as JSON."""

    results_json: dict[str, Any] | list[Any] | None = None


class PendingPredictionPurgeOut(BaseModel):
    """Result of bulk-deleting jobs stuck in ``pending`` (queued but not yet picked up)."""

    deleted: int = Field(ge=0, description="Number of prediction job rows removed.")


class AgenticPromptPreviewOut(BaseModel):
    """Filled orchestration user prompt (same string the LLM receives on POST /agent/decide)."""

    prompt: str


class AgenticJobCreate(BaseModel):
    """Register a prediction job (+ optional row) as an agentic job for the UI list."""

    prediction_job_public_id: str
    results_row_index: int | None = Field(
        default=None,
        ge=0,
        description="Same as AgenticDecideRequest: SHAP / analyst row in results_json.rows.",
    )
    label: str | None = Field(default=None, description="Optional short note shown in job lists.")


class AgenticJobOut(BaseModel):
    """Agentic job row with joined prediction-batch status for dropdowns."""

    public_id: str
    prediction_job_public_id: str
    results_row_index: int | None
    label: str | None
    prediction_status: JobStatus
    rows_total: int | None
    rows_flagged: int | None
    results_model_kind: str | None = Field(
        default=None,
        description="From prediction results_json.model_kind after run.",
    )
    created_at: datetime
    updated_at: datetime


class AgenticDecideRequest(BaseModel):
    prediction_job_public_id: str
    use_rag: bool = True
    results_row_index: int | None = Field(
        default=None,
        ge=0,
        description="When set, use this results_json.rows entry (row_index field or list position). "
        "Otherwise use first flagged row, else first row — which often looks like BENIGN on sorted batches.",
    )
    agentic_job_public_id: str | None = Field(
        default=None,
        description="When set, must reference an agentic_jobs row whose prediction job and results_row_index match this request.",
    )
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
    anchor_trust_chain: bool = Field(
        default=False,
        description="If true, write a demo trust commitment (SHA-256) into the saved report JSON for attestation workflows.",
    )


class AgenticReportOut(ORMModel):
    id: int
    public_id: str
    prediction_job_id: int
    prediction_job_public_id: str | None = Field(
        default=None, description="UUID of the prediction job this report used."
    )
    results_row_index: int | None = Field(
        default=None,
        description="Row passed to POST /agent/decide (matches analyst RAG prep); null for legacy rows.",
    )
    agentic_job_public_id: str | None = Field(
        default=None,
        description="agentic_jobs.public_id when the run was tied to a registered handoff row.",
    )
    summary: str
    recommended_action: str
    raw_llm_response: str | None
    rag_context_used: str | None
    report_path: str | None
    created_at: datetime
    trust_commitment: str | None = Field(
        default=None,
        description="Present on POST /agent/decide when anchor_trust_chain was true (SHA-256 hex).",
    )
    trust_chain_mode: str | None = Field(
        default=None,
        description="e.g. demo_local_commitment when anchor_trust_chain was used.",
    )
    report_artifact: dict[str, Any] | None = Field(
        default=None,
        description="Full on-disk report JSON when present (sample_data, user_prompt, structured_plan, trust_chain, …). "
        "Only set on GET /agent/reports/{public_id}, not on list.",
    )
