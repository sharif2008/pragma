"""Adapter to push a simulated customer message through existing RAG + LLM pipeline pieces.

This is intentionally small: we reuse kb_service + llm_service and only add run/event persistence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlparse

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.services import kb_service, llm_service
from app.services.run_service import (
    StepTimer,
    emit_event,
    now_utc,
    sanitize_error,
    update_run,
)
from app.models.domain import AgentRunStatus

logger = logging.getLogger(__name__)


_ALLOWED_ATTACHMENT_MIME_PREFIXES = ("image/", "audio/")
_ALLOWED_ATTACHMENT_MIME_EXACT = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/json",
}


def _remote_attachment_url(url: str) -> str | None:
    """
    Only http(s) with a host are treated as remote URLs. Anything else (relative paths,
    bare strings from network-log payloads, file:, etc.) is ignored — this path never
    fetches URLs; attachments without a remote URL are handled like text/file-metadata only.
    """
    u = urlparse(url.strip())
    if u.scheme not in ("http", "https"):
        return None
    if not u.netloc:
        return None
    return url.strip()


def _predict_attachment(attachments: list[dict[str, Any]]) -> dict[str, Any]:
    # Heuristic predictor (wired as a “step” so it can be swapped later).
    if not attachments:
        return {
            "attachment_type": "none",
            "shape_constraints": {"text_only": True, "max_message_chars": 20000},
            "confidence": 0.99,
        }
    a0 = attachments[0] if isinstance(attachments[0], dict) else {}
    ct = str(a0.get("content_type") or "").lower().strip()
    fn = str(a0.get("filename") or "").lower().strip()

    if ct == "application/pdf" or fn.endswith(".pdf"):
        return {
            "attachment_type": "pdf",
            "shape_constraints": {
                "allowed_mime_types": ["application/pdf"],
                "max_pages": 50,
                "max_size_mb": 25,
            },
            "confidence": 0.9,
        }
    if ct.startswith("image/") or fn.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return {
            "attachment_type": "image",
            "shape_constraints": {
                "allowed_mime_types": ["image/png", "image/jpeg", "image/webp"],
                "max_width_px": 4096,
                "max_height_px": 4096,
                "max_size_mb": 15,
            },
            "confidence": 0.85,
        }
    if ct.startswith("audio/") or fn.endswith((".mp3", ".wav", ".m4a")):
        return {
            "attachment_type": "audio",
            "shape_constraints": {
                "allowed_mime_types": ["audio/mpeg", "audio/wav", "audio/mp4"],
                "max_duration_s": 600,
                "max_size_mb": 25,
            },
            "confidence": 0.8,
        }
    if ct.startswith("text/") or fn.endswith((".txt", ".md", ".json")):
        return {
            "attachment_type": "text",
            "shape_constraints": {
                "allowed_mime_types": ["text/plain", "text/markdown", "application/json"],
                "max_size_mb": 5,
            },
            "confidence": 0.75,
        }
    return {
        "attachment_type": "unknown",
        "shape_constraints": {"allowed_mime_types": ["application/pdf", "image/*", "audio/*", "text/*"]},
        "confidence": 0.35,
    }


def _normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
    # Minimal normalization (stable key shapes, trimmed strings).
    out = dict(raw)
    if "message" in out:
        out["message"] = str(out["message"] or "").strip()
    if "customer_id" in out and out["customer_id"] is not None:
        out["customer_id"] = str(out["customer_id"]).strip()
    if "channel" in out and out["channel"] is not None:
        out["channel"] = str(out["channel"]).strip()
    return out


async def run_simulated_customer_message(
    *,
    settings: Settings,
    run_id: str,
    trace_id: str,
    normalized_payload: dict[str, Any],
) -> None:
    t_run = StepTimer()
    db: Session = SessionLocal()
    try:
        message = str(normalized_payload.get("message") or "")
        attachments = normalized_payload.get("attachments") or []
        if not isinstance(attachments, list):
            attachments = []

        # Validate attachments early (optional remote URL + content type allowlist).
        for a in attachments:
            if not isinstance(a, dict):
                continue
            url = a.get("url")
            if isinstance(url, str) and url.strip():
                remote = _remote_attachment_url(url)
                a["url"] = remote
            ct = str(a.get("content_type") or "").lower().strip()
            if ct and not (
                ct in _ALLOWED_ATTACHMENT_MIME_EXACT or ct.startswith(_ALLOWED_ATTACHMENT_MIME_PREFIXES)
            ):
                raise ValueError(f"attachment content_type not allowed: {ct}")

        # Step: ingestion
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="ingestion", level="info", message="started")
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="ingestion",
            level="info",
            message="normalized payload ready",
            payload={"keys": sorted(list(normalized_payload.keys()))[:60]},
        )

        # Optional simulated latency (debug/demo)
        sim = normalized_payload.get("simulate") or {}
        if isinstance(sim, dict):
            lat = int(sim.get("latency_ms") or 0)
            if lat > 0:
                await asyncio.sleep(min(lat, 60000) / 1000.0)

        # Step: prediction
        t_pred = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="prediction", level="info", message="started")
        pred = _predict_attachment([a for a in attachments if isinstance(a, dict)])
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="prediction",
            level="info",
            message="completed",
            payload=pred,
            duration_ms=t_pred.ms(),
        )
        update_run(
            db,
            run_id,
            predicted_attachment_type=str(pred.get("attachment_type") or ""),
            predicted_shape_constraints=pred.get("shape_constraints") if isinstance(pred.get("shape_constraints"), dict) else {},
            predictions_json=pred,
        )

        # Forced failure hook for testing retries/dashboard states
        force_step = None
        if isinstance(sim, dict):
            force_step = sim.get("force_error_step")
        if isinstance(force_step, str) and force_step.strip() == "prediction":
            raise RuntimeError("Forced error at step=prediction")

        # Step: rag_write (store message as a KB document using existing kb_service)
        t_rag = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="rag_write", level="info", message="started")
        content = message.strip().encode("utf-8", errors="replace")
        up = UploadFile(filename=f"customer_message_{run_id}.txt", file=None)  # type: ignore[arg-type]
        # UploadFile requires a file-like; set after construction for compatibility across Starlette versions.
        import io

        up.file = io.BytesIO(content)  # type: ignore[assignment]
        up.content_type = "text/plain"  # type: ignore[assignment]
        kb = await kb_service.ingest_kb_document(db, settings, up)
        rag_info = {
            "stored": True,
            "kb_public_ids": [kb.public_id],
            "managed_file_public_ids": [kb.managed_file.public_id] if kb.managed_file else [],
        }
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="rag_write",
            level="info",
            message="completed",
            payload=rag_info,
            duration_ms=t_rag.ms(),
        )
        update_run(db, run_id, rag_json=rag_info)

        if isinstance(force_step, str) and force_step.strip() == "rag_write":
            raise RuntimeError("Forced error at step=rag_write")

        # Step: retrieval (optional, but recorded for dashboards)
        t_ret = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="retrieval", level="info", message="started")
        hits = kb_service.query_kb(db, settings, message[:2000], top_k=min(max(settings.rag_top_k, 3), 8), kb_public_ids=None)
        citations = [{"text": c.get("text", ""), "source": c.get("source"), "score": s, "kb_public_id": kb_pid} for s, c, kb_pid in hits]
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="retrieval",
            level="info",
            message="completed",
            payload={"hits": len(citations)},
            duration_ms=t_ret.ms(),
        )

        if isinstance(force_step, str) and force_step.strip() == "retrieval":
            raise RuntimeError("Forced error at step=retrieval")

        # Step: action_selection (LLM; uses existing llm_service OpenAI wiring if configured)
        t_act = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="action_selection", level="info", message="started")
        action_query = (
            "You are an automation orchestrator. Given a customer message and optional attachment prediction, "
            "return a JSON array of actions. Each action must be an object with keys: type, params.\n\n"
            f"CUSTOMER_MESSAGE:\n{message[:8000]}\n\n"
            f"ATTACHMENT_PREDICTION_JSON:\n{json.dumps(pred, ensure_ascii=False)}\n\n"
            "Return JSON only."
        )
        answer = await llm_service.rag_answer(settings, action_query, citations)
        actions: list[dict[str, Any]] = []
        try:
            start = answer.find("[")
            end = answer.rfind("]") + 1
            if start >= 0 and end > start:
                parsed = json.loads(answer[start:end])
                if isinstance(parsed, list):
                    actions = [x for x in parsed if isinstance(x, dict)]
        except Exception:
            actions = []
        if not actions:
            # Safe fallback: minimal deterministic action
            actions = [{"type": "respond_to_customer", "params": {"channel": normalized_payload.get("channel") or "unknown"}}]

        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="action_selection",
            level="info",
            message="completed",
            payload={"actions": actions[:20]},
            duration_ms=t_act.ms(),
        )
        update_run(
            db,
            run_id,
            status=AgentRunStatus.completed,
            final_actions=actions,
            duration_ms=t_run.ms(),
            completed_at=now_utc(),
        )
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="run",
            level="info",
            message="completed",
            payload={"duration_ms": t_run.ms()},
        )
    except Exception as e:
        err = sanitize_error(e)
        logger.exception("Simulated customer message run failed run_id=%s trace_id=%s", run_id, trace_id)
        try:
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="error",
                level="error",
                message=str(e)[:1024],
                payload={"error": err},
            )
            update_run(
                db,
                run_id,
                status=AgentRunStatus.failed,
                error_summary=err,
                duration_ms=t_run.ms(),
                completed_at=now_utc(),
            )
        except Exception:
            # Last resort: don't raise out of the runner.
            pass
    finally:
        db.close()

