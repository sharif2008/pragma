"""Persistence helpers for agent_runs + events + raw logs."""

from __future__ import annotations

import logging
import re
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Select, desc, select
from sqlalchemy.orm import Session

from app.models.domain import AgentRun, AgentRunEvent, AgentRunStatus, RawCustomerLog

logger = logging.getLogger(__name__)


def new_trace_id() -> str:
    return str(uuid.uuid4())


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _preview(text: str | None, n: int = 220) -> str | None:
    if not text:
        return None
    s = " ".join(str(text).split())
    return s[:n]


def get_run_by_id(db: Session, run_id: str) -> AgentRun | None:
    return db.scalar(select(AgentRun).where(AgentRun.run_id == run_id))


def get_run_by_idempotency_key(db: Session, key: str) -> AgentRun | None:
    k = (key or "").strip()
    if not k:
        return None
    return db.scalar(select(AgentRun).where(AgentRun.idempotency_key == k))


def create_run(
    db: Session,
    *,
    trace_id: str,
    idempotency_key: str | None,
    customer_id: str | None,
    channel: str | None,
    message: str,
) -> AgentRun:
    if idempotency_key:
        existing = get_run_by_idempotency_key(db, idempotency_key)
        if existing:
            return existing

    row = AgentRun(
        trace_id=trace_id,
        idempotency_key=(idempotency_key.strip() if isinstance(idempotency_key, str) and idempotency_key.strip() else None),
        status=AgentRunStatus.running,
        customer_id=(customer_id.strip() if isinstance(customer_id, str) and customer_id.strip() else None),
        channel=(channel.strip() if isinstance(channel, str) and channel.strip() else None),
        message_preview=_preview(message, 220),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def create_raw_log(
    db: Session,
    *,
    run_id: str,
    trace_id: str,
    raw_payload: dict[str, Any],
    normalized_payload: dict[str, Any] | None,
) -> RawCustomerLog:
    row = RawCustomerLog(
        run_id=run_id,
        trace_id=trace_id,
        raw_payload=raw_payload,
        normalized_payload=normalized_payload,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def emit_event(
    db: Session,
    *,
    run_id: str,
    trace_id: str,
    step_name: str,
    level: str = "info",
    message: str = "",
    payload: dict[str, Any] | None = None,
    duration_ms: int | None = None,
) -> AgentRunEvent:
    row = AgentRunEvent(
        run_id=run_id,
        trace_id=trace_id,
        step_name=str(step_name),
        level=str(level),
        message=str(message or "")[:1024],
        payload=payload,
        duration_ms=duration_ms,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_run(
    db: Session,
    run_id: str,
    *,
    status: AgentRunStatus | None = None,
    predictions_json: dict[str, Any] | None = None,
    predicted_attachment_type: str | None = None,
    predicted_shape_constraints: dict[str, Any] | None = None,
    rag_json: dict[str, Any] | None = None,
    final_actions: list[dict[str, Any]] | dict[str, Any] | None = None,
    error_summary: str | None = None,
    duration_ms: int | None = None,
    completed_at: datetime | None = None,
) -> AgentRun:
    row = get_run_by_id(db, run_id)
    if not row:
        raise ValueError(f"run_id not found: {run_id}")

    if status is not None:
        row.status = status
    if predictions_json is not None:
        row.predictions_json = predictions_json
    if predicted_attachment_type is not None:
        row.predicted_attachment_type = predicted_attachment_type
    if predicted_shape_constraints is not None:
        row.predicted_shape_constraints = predicted_shape_constraints
    if rag_json is not None:
        row.rag_json = rag_json
    if final_actions is not None:
        row.final_actions = final_actions
    if error_summary is not None:
        row.error_summary = error_summary
    if duration_ms is not None:
        row.duration_ms = duration_ms
    if completed_at is not None:
        row.completed_at = completed_at

    db.commit()
    db.refresh(row)
    return row


def list_runs(
    db: Session,
    *,
    status: str | None,
    limit: int,
    offset: int,
) -> list[AgentRun]:
    q: Select[tuple[AgentRun]] = select(AgentRun).order_by(desc(AgentRun.created_at))
    if status:
        q = q.where(AgentRun.status == status)
    q = q.offset(offset).limit(limit)
    return list(db.scalars(q).all())


def list_events(db: Session, run_id: str, *, limit: int = 5000) -> list[AgentRunEvent]:
    q = (
        select(AgentRunEvent)
        .where(AgentRunEvent.run_id == run_id)
        .order_by(AgentRunEvent.timestamp.asc(), AgentRunEvent.id.asc())
        .limit(limit)
    )
    return list(db.scalars(q).all())


def list_events_all(
    db: Session,
    *,
    limit: int = 200,
    offset: int = 0,
) -> list[AgentRunEvent]:
    q = (
        select(AgentRunEvent)
        .order_by(AgentRunEvent.timestamp.desc(), AgentRunEvent.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def latest_step_name(db: Session, run_id: str) -> str | None:
    row = db.scalar(
        select(AgentRunEvent)
        .where(AgentRunEvent.run_id == run_id)
        .order_by(AgentRunEvent.timestamp.desc(), AgentRunEvent.id.desc())
        .limit(1)
    )
    return row.step_name if row else None


_SECRET_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)openai[_-]?api[_-]?key\s*=\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"][^'\"]+['\"]"),
    re.compile(r"(?i)authorization:\s*bearer\s+[a-z0-9\-\._]+"),
]


def sanitize_error(e: BaseException, *, max_chars: int = 6000) -> str:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    out = tb
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    if len(out) > max_chars:
        out = out[:max_chars] + "\n[truncated]"
    return out


class StepTimer:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

