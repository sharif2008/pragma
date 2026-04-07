"""Database tables: managed_files, training_jobs, model_versions, prediction_jobs, knowledge_base_files, agentic_jobs, agentic_reports."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy import JSON
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class FileKind(str, enum.Enum):
    upload = "upload"
    training_dataset = "training_dataset"
    prediction_input = "prediction_input"
    knowledge_doc = "knowledge_doc"


class JobStatus(str, enum.Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class AgentRunStatus(str, enum.Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    partial = "partial"
    needs_input = "needs_input"


class ManagedFile(Base):
    __tablename__ = "managed_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    original_name: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_kind: Mapped[FileKind] = mapped_column(
        SAEnum(FileKind, values_callable=lambda x: [e.value for e in x]), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    parent_file_id: Mapped[int | None] = mapped_column(
        ForeignKey("managed_files.id", ondelete="SET NULL"), nullable=True
    )
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    parent = relationship("ManagedFile", remote_side=[id], backref="child_versions")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, values_callable=lambda x: [e.value for e in x]),
        default=JobStatus.pending,
        nullable=False,
    )
    dataset_file_id: Mapped[int] = mapped_column(
        ForeignKey("managed_files.id", ondelete="RESTRICT"), nullable=False
    )
    target_column: Mapped[str] = mapped_column(String(256), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # Set after ModelVersion row is created (breaks circular create with training_job_id on model).
    model_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    dataset_file = relationship("ManagedFile", foreign_keys=[dataset_file_id])
    model_version = relationship("ModelVersion", foreign_keys=[model_version_id])


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    training_job_id: Mapped[int | None] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="SET NULL"), nullable=True
    )
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    metrics_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    feature_columns_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    label_classes_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    training_job = relationship("TrainingJob", foreign_keys=[training_job_id])


class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id", ondelete="RESTRICT"), nullable=False
    )
    input_file_id: Mapped[int] = mapped_column(
        ForeignKey("managed_files.id", ondelete="RESTRICT"), nullable=False
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, values_callable=lambda x: [e.value for e in x]),
        default=JobStatus.pending,
        nullable=False,
    )
    output_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    rows_total: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rows_flagged: Mapped[int | None] = mapped_column(Integer, nullable=True)
    config_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    #: Per-row predictions, class probabilities, and optional SHAP (TreeExplainer for sklearn trees; VFL stores a placeholder).
    results_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    model_version = relationship("ModelVersion")
    input_file = relationship("ManagedFile")


class AgenticJob(Base):
    """Registered agentic session: ties a prediction batch + optional results row to the Agentic actions UI (persisted)."""

    __tablename__ = "agentic_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    prediction_job_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    results_row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    label: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    prediction_job = relationship("PredictionJob")


class KnowledgeBaseFile(Base):
    __tablename__ = "knowledge_base_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    managed_file_id: Mapped[int] = mapped_column(
        ForeignKey("managed_files.id", ondelete="CASCADE"), nullable=False
    )
    vector_index_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    managed_file = relationship("ManagedFile")


class AgenticReport(Base):
    __tablename__ = "agentic_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    prediction_job_id: Mapped[int] = mapped_column(
        ForeignKey("prediction_jobs.id", ondelete="CASCADE"), nullable=False
    )
    #: Same semantics as POST /agent/decide ``results_row_index``; null = batch-default row selection.
    results_row_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Optional link to ``agentic_jobs`` when the analyst ran the agent from a registered handoff row.
    agentic_job_id: Mapped[int | None] = mapped_column(
        ForeignKey("agentic_jobs.id", ondelete="SET NULL"), nullable=True, index=True
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_llm_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    rag_context_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    prediction_job = relationship("PredictionJob")
    agentic_job = relationship("AgenticJob")


class AgentRun(Base):
    """End-to-end traceable run for simulated customer messages."""

    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    trace_id: Mapped[str] = mapped_column(CHAR(36), index=True, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), unique=True, nullable=True)

    status: Mapped[AgentRunStatus] = mapped_column(
        SAEnum(AgentRunStatus, values_callable=lambda x: [e.value for e in x]),
        default=AgentRunStatus.running,
        index=True,
        nullable=False,
    )

    customer_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    channel: Mapped[str | None] = mapped_column(String(64), nullable=True)
    message_preview: Mapped[str | None] = mapped_column(String(512), nullable=True)

    predicted_attachment_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    predicted_shape_constraints: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    predictions_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    rag_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    final_actions: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)

    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class AgentRunEvent(Base):
    __tablename__ = "agent_run_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    run_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("agent_runs.run_id", ondelete="CASCADE"), index=True)
    trace_id: Mapped[str] = mapped_column(CHAR(36), index=True, nullable=False)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    step_name: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    message: Mapped[str] = mapped_column(String(1024), nullable=False, default="")
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RawCustomerLog(Base):
    __tablename__ = "raw_customer_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    log_id: Mapped[str] = mapped_column(
        CHAR(36), unique=True, default=lambda: str(uuid.uuid4()), nullable=False
    )
    run_id: Mapped[str] = mapped_column(CHAR(36), ForeignKey("agent_runs.run_id", ondelete="CASCADE"), index=True)
    trace_id: Mapped[str] = mapped_column(CHAR(36), index=True, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    raw_payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    normalized_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
