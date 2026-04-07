"""Simulate a single network event row through the existing ML prediction pipeline."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import uuid
from typing import Any

from fastapi import UploadFile
from starlette.datastructures import Headers
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.session import SessionLocal
from app.models.domain import AgentRunStatus, ManagedFile, ModelVersion
from app.models.domain import FileKind, JobStatus, PredictionJob
from app.services import file_service, prediction_service
from app.services.agentic_llm_prompt import (
    build_sample_from_prediction_job,
    load_attack_agentic_config,
)
from app.services import llm_service
from app.services import kb_service
from app.services.rag_templates_row_context import build_row_agent_templates
from app.services.run_service import StepTimer, emit_event, now_utc, sanitize_error, update_run

logger = logging.getLogger(__name__)


def _upload_file_from_bytes(*, filename: str, content_type: str, data: bytes) -> UploadFile:
    # Starlette's UploadFile.content_type is derived from headers and is read-only in some versions.
    return UploadFile(
        filename=filename,
        file=io.BytesIO(data),
        headers=Headers({"content-type": content_type}),
    )


def _to_csv_bytes(columns: list[str], values: list[Any]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    w.writerow(values)
    return buf.getvalue().encode("utf-8", errors="replace")


def _latest_model_version_public_id(db: Session) -> str:
    mv = db.scalar(select(ModelVersion).order_by(ModelVersion.created_at.desc()).limit(1))
    if not mv:
        raise ValueError("No model_version found. Train a model first via POST /training/start.")
    return mv.public_id


def _model_feature_columns(db: Session, model_version_public_id: str) -> list[str]:
    mv = db.scalar(select(ModelVersion).where(ModelVersion.public_id == model_version_public_id))
    if not mv:
        raise ValueError("Model version not found")
    cols = mv.feature_columns_json
    if not isinstance(cols, list) or not cols:
        raise ValueError("Model feature_columns_json missing on this model")
    return [str(c) for c in cols if str(c).strip()]


def _align_to_model_columns(model_cols: list[str], incoming_cols: list[str], incoming_vals: list[Any]) -> list[Any]:
    idx = {c: i for i, c in enumerate(incoming_cols)}
    out: list[Any] = []
    for mc in model_cols:
        if mc in idx:
            out.append(incoming_vals[idx[mc]])
        else:
            out.append("")
    return out


async def run_simulated_network_traffic(
    *,
    settings: Settings,
    run_id: str,
    trace_id: str,
    rows: list[list[Any]],
    incoming_columns: list[str] | None,
    model_version_public_id: str | None,
    simulate: dict[str, Any] | None,
) -> None:
    """
    Multi-row ingestion. Uses model feature columns as canonical schema.
    For each flagged (or non-BENIGN) row, triggers SHAP-aware RAG query + agentic actions.
    """
    t_run = StepTimer()
    db: Session = SessionLocal()
    try:
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="ingestion", level="info", message="started")
        if simulate and int(simulate.get("latency_ms") or 0) > 0:
            await asyncio.sleep(min(int(simulate["latency_ms"]), 60000) / 1000.0)

        # Default behavior: use the latest trained model unless explicitly set.
        mv_public = model_version_public_id or _latest_model_version_public_id(db)
        model_cols = _model_feature_columns(db, mv_public)

        # Align incoming rows to model columns if a header was provided.
        aligned_rows: list[list[Any]] = []
        if incoming_columns:
            for r in rows:
                aligned_rows.append(_align_to_model_columns(model_cols, incoming_columns, r))
        else:
            aligned_rows = rows

        # Step: build + upload CSV
        t_csv = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="csv_build", level="info", message="started")
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(model_cols)
        for r in aligned_rows:
            w.writerow(r)
        data = buf.getvalue().encode("utf-8", errors="replace")

        up = _upload_file_from_bytes(filename=f"sim_traffic_{run_id}.csv", content_type="text/csv", data=data)
        mf = await file_service.upload_file(db, settings, up, FileKind.prediction_input, replace_public_id=None)
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="csv_build",
            level="info",
            message="completed",
            payload={"input_file_public_id": mf.public_id, "rows": len(aligned_rows), "bytes": len(data)},
            duration_ms=t_csv.ms(),
        )

        force_step = (simulate or {}).get("force_error_step") if isinstance(simulate, dict) else None
        if isinstance(force_step, str) and force_step.strip() == "csv_build":
            raise RuntimeError("Forced error at step=csv_build")

        # Step: prediction job (enable SHAP so RAG row templates can use it when available)
        t_pred = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="prediction", level="info", message="started")
        job = prediction_service.create_prediction_job(db, mv_public, mf.public_id, config={"compute_shap": True})
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="prediction",
            level="info",
            message="job_created",
            payload={"prediction_job_public_id": job.public_id, "model_version_public_id": mv_public},
        )
        prediction_service.run_prediction_job_sync(job.id)
        # run_prediction_job_sync uses its own SessionLocal; expunge so we reload this row from DB.
        jid = job.id
        db.expunge(job)
        job2 = db.get(PredictionJob, jid)
        if not job2 or job2.status != JobStatus.completed or not isinstance(job2.results_json, dict):
            raise RuntimeError(f"prediction failed status={(job2.status.value if job2 else 'missing')}")

        rj = job2.results_json
        rows_json = rj.get("rows") if isinstance(rj, dict) else None
        rows_list: list[dict[str, Any]] = [x for x in (rows_json or []) if isinstance(x, dict)]

        # Store run-level summary in predictions_json for dashboard list
        pred_summary = {
            "prediction_job_public_id": job2.public_id,
            "model_kind": rj.get("model_kind"),
            "rows_total": len(rows_list),
            "rows_flagged": sum(1 for x in rows_list if x.get("flagged_attack_or_anomaly")),
        }
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="prediction",
            level="info",
            message="completed",
            payload=pred_summary,
            duration_ms=t_pred.ms(),
        )
        update_run(db, run_id, predictions_json=pred_summary)

        if isinstance(force_step, str) and force_step.strip() == "prediction":
            raise RuntimeError("Forced error at step=prediction")

        # Step: rag_write — store run-linked traffic summary into KB (so later RAG queries can cite it)
        t_ragw = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="rag_write", level="info", message="started")
        try:
            # Compact JSON payload (avoid massive raw feature dumps in KB; keep it useful for retrieval).
            compact_rows: list[dict[str, Any]] = []
            for r in rows_list[: min(len(rows_list), 200)]:
                compact_rows.append(
                    {
                        "row_index": r.get("row_index"),
                        "predicted_label": r.get("predicted_label"),
                        "max_class_probability": r.get("max_class_probability"),
                        "flagged_attack_or_anomaly": r.get("flagged_attack_or_anomaly"),
                        "shap": r.get("shap") if isinstance(r.get("shap"), dict) else None,
                    }
                )
            kb_doc = {
                "kind": "network_traffic_run",
                "run_id": run_id,
                "trace_id": trace_id,
                "model_version_public_id": mv_public,
                "prediction_job_public_id": job2.public_id,
                "summary": pred_summary,
                "rows": compact_rows,
            }
            payload_bytes = (json.dumps(kb_doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8", errors="replace")
            up = _upload_file_from_bytes(
                filename=f"traffic_run_{run_id}.json",
                content_type="application/json",
                data=payload_bytes,
            )
            kb = await kb_service.ingest_kb_document(db, settings, up)
            rag_info = {
                "stored": True,
                "kb_public_ids": [kb.public_id],
                "managed_file_public_ids": [kb.managed_file.public_id] if kb.managed_file else [],
            }
            update_run(db, run_id, rag_json=rag_info)
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="rag_write",
                level="info",
                message="completed",
                payload=rag_info,
                duration_ms=t_ragw.ms(),
            )
        except Exception as e:
            # RAG write is useful but not required to complete the run.
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="rag_write",
                level="warn",
                message="failed (continuing)",
                payload={"error": str(e)[:800]},
                duration_ms=t_ragw.ms(),
            )

        # For each relevant row, run retrieval + agentic decide
        summary = prediction_service.load_prediction_summary(settings, job2)
        attack_actions_data, agentic_features_data = load_attack_agentic_config(verbose=False)

        actions_out: list[dict[str, Any]] = []
        for row in rows_list:
            row_idx = row.get("row_index")
            pred_label = str(row.get("predicted_label") or "").strip().upper()
            flagged = bool(row.get("flagged_attack_or_anomaly"))
            if not pred_label:
                continue
            if pred_label == "BENIGN" and not flagged:
                continue

            t_row = StepTimer()
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="row_process",
                level="info",
                message=f"started row_index={row_idx} label={pred_label}",
                payload={"row_index": row_idx, "predicted_label": pred_label, "flagged": flagged},
            )

            # RAG queries built from SHAP per row (existing template logic)
            base = f"{summary.get('rows_flagged')} flagged / {summary.get('rows_total')} total"
            extra_templates, row_ctx = build_row_agent_templates(
                job_public_id=job2.public_id,
                row=row,
                base_summary_line=str(base),
            )
            retrieval_queries = []
            if extra_templates and isinstance(extra_templates[0], dict):
                rq = extra_templates[0].get("retrieval_queries")
                if isinstance(rq, list):
                    retrieval_queries = [str(x) for x in rq if str(x).strip()]
            if not retrieval_queries:
                retrieval_queries = [f"SOC runbooks and response guidance for label={pred_label}"]

            t_ret = StepTimer()
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="retrieval",
                level="info",
                message=f"started row_index={row_idx}",
                payload={"row_index": row_idx, "queries": len(retrieval_queries)},
            )
            hits, meta = kb_service.query_kb_multi_mmr(
                db,
                settings,
                retrieval_queries[:6],
                final_k=min(max(settings.rag_top_k, 6), 10),
                per_query_k=12,
                mmr_lambda=0.55,
                kb_public_ids=None,
            )
            rag_context = "\n\n".join(f"- (sim={h['score']:.3f}) {h['text'][:800]}" for h in (hits or [])) if hits else None
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="retrieval",
                level="info",
                message=f"completed row_index={row_idx}",
                payload={"row_index": row_idx, "hits": len(hits or []), "meta": meta},
                duration_ms=t_ret.ms(),
            )

            t_act = StepTimer()
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="action_selection",
                level="info",
                message=f"started row_index={row_idx}",
                payload={"row_index": row_idx, "use_rag": True},
            )
            # Build agent sample_data from the same prediction row
            try:
                ri = int(row_idx) if row_idx is not None else 0
            except (TypeError, ValueError):
                ri = 0
            sample_data = build_sample_from_prediction_job(job2, summary, results_row_index=ri)
            decision = await llm_service.agent_decide(
                settings,
                sample_data=sample_data,
                feature_notes=None,
                rag_context=rag_context,
                attack_actions_data=attack_actions_data,
                agentic_features_data=agentic_features_data,
                use_rag=True,
            )
            row_action = {
                "row_index": ri,
                "predicted_label": pred_label,
                "flagged_attack_or_anomaly": flagged,
                "row_context": row_ctx,
                "agentic_summary": decision.get("summary"),
                "recommended_action": decision.get("recommended_action"),
            }
            actions_out.append(row_action)
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="action_selection",
                level="info",
                message=f"completed row_index={row_idx}",
                payload={"row_index": ri, "recommended_action": decision.get("recommended_action")},
                duration_ms=t_act.ms(),
            )
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="row_process",
                level="info",
                message=f"completed row_index={row_idx}",
                payload={"row_index": ri, "duration_ms": t_row.ms()},
                duration_ms=t_row.ms(),
            )

        update_run(db, run_id, final_actions=actions_out)

        update_run(
            db,
            run_id,
            status=AgentRunStatus.completed,
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
            payload={"duration_ms": t_run.ms(), "rows_processed": len(actions_out)},
        )
    except Exception as e:
        err = sanitize_error(e)
        logger.exception("Network traffic simulation failed run_id=%s trace_id=%s", run_id, trace_id)
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
            pass
    finally:
        db.close()


async def run_simulated_network_event(
    *,
    settings: Settings,
    run_id: str,
    trace_id: str,
    columns: list[str],
    values: list[Any],
    model_version_public_id: str | None,
    simulate: dict[str, Any] | None,
) -> None:
    t_run = StepTimer()
    db: Session = SessionLocal()
    try:
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="ingestion", level="info", message="started")
        if simulate and int(simulate.get("latency_ms") or 0) > 0:
            await asyncio.sleep(min(int(simulate["latency_ms"]), 60000) / 1000.0)

        mv_public = model_version_public_id or _latest_model_version_public_id(db)

        # Step: build + upload a 1-row CSV as a managed prediction_input file.
        t_csv = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="csv_build", level="info", message="started")
        data = _to_csv_bytes(columns, values)
        up = _upload_file_from_bytes(filename=f"sim_event_{run_id}.csv", content_type="text/csv", data=data)
        mf = await file_service.upload_file(db, settings, up, FileKind.prediction_input, replace_public_id=None)
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="csv_build",
            level="info",
            message="completed",
            payload={"input_file_public_id": mf.public_id, "bytes": len(data)},
            duration_ms=t_csv.ms(),
        )

        force_step = (simulate or {}).get("force_error_step") if isinstance(simulate, dict) else None
        if isinstance(force_step, str) and force_step.strip() == "csv_build":
            raise RuntimeError("Forced error at step=csv_build")

        # Step: prediction job
        t_pred = StepTimer()
        emit_event(db, run_id=run_id, trace_id=trace_id, step_name="prediction", level="info", message="started")
        job = prediction_service.create_prediction_job(db, mv_public, mf.public_id, config={"compute_shap": False})
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="prediction",
            level="info",
            message="job_created",
            payload={"prediction_job_public_id": job.public_id, "model_version_public_id": mv_public},
        )
        prediction_service.run_prediction_job_sync(job.id)
        jid = job.id
        db.expunge(job)
        job2 = db.get(PredictionJob, jid)
        if not job2 or job2.status != JobStatus.completed or not isinstance(job2.results_json, dict):
            raise RuntimeError(f"prediction failed status={(job2.status.value if job2 else 'missing')}")

        rows = job2.results_json.get("rows")
        r0 = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
        pred_payload = {
            "predicted_label": r0.get("predicted_label"),
            "max_class_probability": r0.get("max_class_probability"),
            "flagged_attack_or_anomaly": r0.get("flagged_attack_or_anomaly"),
            "prediction_job_public_id": job2.public_id,
            "model_kind": job2.results_json.get("model_kind"),
        }
        emit_event(
            db,
            run_id=run_id,
            trace_id=trace_id,
            step_name="prediction",
            level="info",
            message="completed",
            payload=pred_payload,
            duration_ms=t_pred.ms(),
        )
        update_run(db, run_id, predictions_json=pred_payload)

        if isinstance(force_step, str) and force_step.strip() == "prediction":
            raise RuntimeError("Forced error at step=prediction")

        # Step: action_selection (agentic orchestration for non-BENIGN predictions)
        predicted_label = str(pred_payload.get("predicted_label") or "").strip().upper()
        if predicted_label and predicted_label != "BENIGN":
            t_act = StepTimer()
            emit_event(db, run_id=run_id, trace_id=trace_id, step_name="action_selection", level="info", message="started")
            summary = prediction_service.load_prediction_summary(settings, job2)
            sample_data = build_sample_from_prediction_job(job2, summary, results_row_index=0)
            attack_actions_data, agentic_features_data = load_attack_agentic_config(verbose=False)
            prompt = build_agentic_decide_user_prompt(
                sample_data,
                rag_context=None,
                attack_actions_data=attack_actions_data,
                agentic_features_data=agentic_features_data,
                include_knowledge_base=False,
                feature_notes=None,
            )
            # Use the existing LLM client wrapper (OpenAI if configured; mock otherwise).
            decision = await llm_service.agent_decide(
                settings,
                sample_data=sample_data,
                feature_notes=None,
                rag_context=None,
                attack_actions_data=attack_actions_data,
                agentic_features_data=agentic_features_data,
                use_rag=False,
            )

            actions_payload: dict[str, Any] = {"agentic_summary": decision.get("summary"), "recommended_action": decision.get("recommended_action")}
            raw = decision.get("raw_llm_response")
            if isinstance(raw, str) and raw.strip():
                actions_payload["raw_llm_response"] = raw[:20000]

            update_run(db, run_id, final_actions=actions_payload)
            emit_event(
                db,
                run_id=run_id,
                trace_id=trace_id,
                step_name="action_selection",
                level="info",
                message="completed",
                payload={"recommended_action": decision.get("recommended_action")},
                duration_ms=t_act.ms(),
            )

        # Step: completed
        update_run(
            db,
            run_id,
            status=AgentRunStatus.completed,
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
        logger.exception("Network event simulation failed run_id=%s trace_id=%s", run_id, trace_id)
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
            pass
    finally:
        db.close()

