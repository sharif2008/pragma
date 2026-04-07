"""Run tracing endpoints (polling + optional SSE)."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.models.domain import AgentRunStatus
from app.schemas.runs import RunEventOut, RunListItemOut, RunSummaryOut
from app.services import run_service

router = APIRouter(prefix="/api/v1", tags=["runs"])


def _status_to_str(s: object) -> str:
    return str(getattr(s, "value", s))


def _norm_run_id(run_id: str) -> str:
    return (run_id or "").strip()


# --- Static paths must be registered before /runs/{run_id} or "events" is captured as run_id. ---


@router.get("/runs", response_model=list[RunListItemOut])
def list_runs(
    db: Annotated[Session, Depends(get_db)],
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[RunListItemOut]:
    rows = run_service.list_runs(db, status=status, limit=limit, offset=offset)
    out: list[RunListItemOut] = []
    for r in rows:
        pred_label = None
        flagged = None
        if isinstance(r.predictions_json, dict):
            pred_label = r.predictions_json.get("predicted_label")
            flagged = r.predictions_json.get("flagged_attack_or_anomaly")
        out.append(
            RunListItemOut(
                run_id=r.run_id,
                trace_id=r.trace_id,
                status=_status_to_str(r.status),  # type: ignore[arg-type]
                created_at=r.created_at,
                updated_at=r.updated_at,
                customer_id=r.customer_id,
                channel=r.channel,
                message_preview=r.message_preview,
                predicted_attachment_type=r.predicted_attachment_type,
                predicted_label=pred_label,
                flagged_attack_or_anomaly=flagged,
                duration_ms=r.duration_ms,
                last_step=run_service.latest_step_name(db, r.run_id),
            )
        )
    return out


@router.get("/runs/events", response_model=list[RunEventOut])
def list_run_events(
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> list[RunEventOut]:
    """List run events across all runs (newest first)."""
    events = run_service.list_events_all(db, limit=limit, offset=offset)
    return [
        RunEventOut(
            event_id=e.event_id,
            run_id=e.run_id,
            trace_id=e.trace_id,
            timestamp=e.timestamp,
            step_name=e.step_name,
            level=("error" if e.level == "error" else "warn" if e.level == "warn" else "info"),
            message=e.message,
            payload=e.payload,
            duration_ms=e.duration_ms,
        )
        for e in events
    ]


@router.get("/runs/{run_id}", response_model=RunSummaryOut)
def get_run_summary(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
) -> RunSummaryOut:
    rid = _norm_run_id(run_id)
    row = run_service.get_run_by_id(db, rid)
    if not row:
        raise HTTPException(404, "run not found")
    last = run_service.latest_step_name(db, rid)
    preds = None
    if isinstance(row.predictions_json, dict):
        preds = {
            "attachment_type": row.predictions_json.get("attachment_type") or row.predicted_attachment_type or "unknown",
            "shape_constraints": row.predictions_json.get("shape_constraints") or row.predicted_shape_constraints or {},
            "confidence": float(row.predictions_json.get("confidence") or 0.0),
            "predicted_label": row.predictions_json.get("predicted_label"),
            "max_class_probability": row.predictions_json.get("max_class_probability"),
            "flagged_attack_or_anomaly": row.predictions_json.get("flagged_attack_or_anomaly"),
        }
    rag = None
    if isinstance(row.rag_json, dict):
        rag = {
            "stored": bool(row.rag_json.get("stored")),
            "kb_public_ids": list(row.rag_json.get("kb_public_ids") or []),
            "managed_file_public_ids": list(row.rag_json.get("managed_file_public_ids") or []),
        }
    err = None
    if row.error_summary:
        err = {"step_name": last, "message": row.error_summary[:1000]}

    return RunSummaryOut(
        run_id=row.run_id,
        trace_id=row.trace_id,
        status=_status_to_str(row.status),  # type: ignore[arg-type]
        created_at=row.created_at,
        updated_at=row.updated_at,
        completed_at=row.completed_at,
        customer_id=row.customer_id,
        channel=row.channel,
        message_preview=row.message_preview,
        duration_ms=row.duration_ms,
        last_step=last,
        predictions=preds,  # type: ignore[arg-type]
        rag=rag,  # type: ignore[arg-type]
        actions=row.final_actions if isinstance(row.final_actions, list) else None,
        error=err,  # type: ignore[arg-type]
    )


@router.get("/runs/{run_id}/events", response_model=list[RunEventOut])
def get_run_events(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    limit: int = Query(default=5000, ge=1, le=20000),
) -> list[RunEventOut]:
    rid = _norm_run_id(run_id)
    row = run_service.get_run_by_id(db, rid)
    if not row:
        raise HTTPException(404, "run not found")
    events = run_service.list_events(db, rid, limit=limit)
    return [
        RunEventOut(
            event_id=e.event_id,
            run_id=e.run_id,
            trace_id=e.trace_id,
            timestamp=e.timestamp,
            step_name=e.step_name,
            level=("error" if e.level == "error" else "warn" if e.level == "warn" else "info"),
            message=e.message,
            payload=e.payload,
            duration_ms=e.duration_ms,
        )
        for e in events
    ]


@router.get("/runs/{run_id}/stream")
async def stream_run_events_sse(
    run_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamingResponse:
    """
    Server-Sent Events stream for dashboards.

    Emits: "event" objects as `data: {...}\n\n` until terminal status or timeout.
    """
    rid = _norm_run_id(run_id)
    row = run_service.get_run_by_id(db, rid)
    if not row:
        raise HTTPException(404, "run not found")

    async def gen():
        last_seen: set[str] = set()
        started = asyncio.get_event_loop().time()
        timeout_s = 60.0
        poll_s = 0.6
        while True:
            # Re-open a short-lived DB session via dependency session is tricky in generator;
            # use the same session object (works for MySQL in practice for reads) and refresh.
            events = run_service.list_events(db, rid, limit=20000)
            for e in events:
                if e.event_id in last_seen:
                    continue
                last_seen.add(e.event_id)
                payload = {
                    "event_id": e.event_id,
                    "run_id": e.run_id,
                    "trace_id": e.trace_id,
                    "timestamp": e.timestamp.isoformat(),
                    "step_name": e.step_name,
                    "level": e.level,
                    "message": e.message,
                    "payload": e.payload,
                    "duration_ms": e.duration_ms,
                }
                yield f"data: {json_dumps(payload)}\n\n"

            db.refresh(row)
            if row.status in (
                AgentRunStatus.completed,
                AgentRunStatus.failed,
                AgentRunStatus.partial,
                AgentRunStatus.needs_input,
            ):
                yield "event: done\ndata: {}\n\n"
                return

            if asyncio.get_event_loop().time() - started > timeout_s:
                yield "event: timeout\ndata: {}\n\n"
                return
            await asyncio.sleep(poll_s)

    def json_dumps(obj):
        import json

        return json.dumps(obj, ensure_ascii=False)

    return StreamingResponse(gen(), media_type="text/event-stream")
