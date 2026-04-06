"""Batch prediction from CSV using a persisted model bundle."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import JobStatus, ManagedFile, ModelVersion, PredictionJob
from app.utils.file_utils import remove_path
from app.services import file_service
from app.services.ml_training import load_model_bundle
from app.services.prediction_shap import compute_sklearn_tree_shap_per_row

logger = logging.getLogger(__name__)

# TreeExplainer cost grows with rows; skip SHAP above this (per-row predictions still stored).
MAX_ROWS_FOR_SHAP = 800


def get_model_version_by_public_id(db: Session, public_id: str) -> ModelVersion:
    mv = db.scalar(select(ModelVersion).where(ModelVersion.public_id == public_id))
    if not mv:
        from fastapi import HTTPException

        raise HTTPException(404, "Model version not found")
    return mv


def create_prediction_job(
    db: Session,
    model_version_public_id: str,
    input_file_public_id: str,
    config: dict | None,
) -> PredictionJob:
    mv = get_model_version_by_public_id(db, model_version_public_id)
    inp = file_service.get_by_public_id(db, input_file_public_id)
    if inp.file_kind.value != "prediction_input" and inp.file_kind.value != "upload":
        pass  # allow upload kind for flexibility
    job = PredictionJob(
        model_version_id=mv.id,
        input_file_id=inp.id,
        status=JobStatus.pending,
        config_json=config or {},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_prediction_job(db: Session, public_id: str) -> PredictionJob:
    job = db.scalar(select(PredictionJob).where(PredictionJob.public_id == public_id))
    if not job:
        from fastapi import HTTPException

        raise HTTPException(404, "Prediction job not found")
    return job


def results_model_kind_from_job(job: PredictionJob) -> str | None:
    """Expose ``results_json.model_kind`` without loading full results in the API response."""
    rj = job.results_json
    if isinstance(rj, dict):
        v = rj.get("model_kind")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def list_prediction_jobs(db: Session, *, limit: int = 100, offset: int = 0) -> list[PredictionJob]:
    q = (
        select(PredictionJob)
        .order_by(PredictionJob.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def _remove_prediction_output_file(settings: Settings, job: PredictionJob) -> None:
    if not job.output_path:
        return
    out_abs = (settings.storage_root / job.output_path).resolve()
    root = settings.storage_root.resolve()
    try:
        out_abs.relative_to(root)
    except ValueError:
        logger.warning("Skipping removal of output outside storage_root: %s", job.output_path)
        return
    remove_path(out_abs)


def delete_prediction_job(db: Session, settings: Settings, public_id: str) -> None:
    """Remove job row (cascades agentic_reports). Deletes on-disk CSV output when present.

    Pending jobs may be removed (same as bulk pending purge). Running jobs cannot be deleted.
    """
    job = get_prediction_job(db, public_id)
    if job.status == JobStatus.running:
        from fastapi import HTTPException

        raise HTTPException(
            409,
            "Cannot delete a prediction job while it is running.",
        )
    _remove_prediction_output_file(settings, job)
    db.delete(job)
    db.commit()


def delete_all_pending_prediction_jobs(db: Session, settings: Settings) -> int:
    """Delete every job in ``pending`` status (output file removed if present). Does not touch running/completed/failed."""
    jobs = list(db.scalars(select(PredictionJob).where(PredictionJob.status == JobStatus.pending)).all())
    for job in jobs:
        _remove_prediction_output_file(settings, job)
        db.delete(job)
    db.commit()
    return len(jobs)


def run_prediction_job_sync(job_db_id: int) -> None:
    from app.core.config import get_settings
    from app.db.session import SessionLocal

    settings = get_settings()
    db = SessionLocal()
    try:
        job = db.get(PredictionJob, job_db_id)
        if not job:
            return
        job.status = JobStatus.running
        db.commit()

        mv = db.get(ModelVersion, job.model_version_id)
        inp = db.get(ManagedFile, job.input_file_id)
        if not mv or not inp:
            job.status = JobStatus.failed
            job.error_message = "Missing model or input file"
            db.commit()
            return

        artifact = settings.storage_root / mv.artifact_path
        if not artifact.is_file():
            job.status = JobStatus.failed
            job.error_message = "Model artifact missing on disk"
            db.commit()
            return

        try:
            bundle = load_model_bundle(artifact)
            target_col = bundle["target_column"]
            feature_columns: list[str] = bundle["feature_columns"]

            csv_path = file_service.resolved_path(settings, inp)
            df = pd.read_csv(csv_path)
            for c in feature_columns:
                if c not in df.columns:
                    df[c] = np.nan
            X = df[feature_columns]

            cfg = job.config_json or {}
            threshold = cfg.get("anomaly_probability_threshold")
            attack_vals = cfg.get("attack_label_values") or []
            want_shap = bool(cfg.get("compute_shap", True))

            proba_matrix: np.ndarray | None = None
            proba_class_names: list[str] = []

            if bundle.get("kind") == "vfl_torch":
                from app.services.ml_vfl import predict_vfl_batch, vfl_gradient_x_input_attribution_rows

                pred_idx, max_p_arr, probs_full = predict_vfl_batch(bundle, X, return_probs=True)
                classes: list = bundle["label_classes"]
                labels = np.array([classes[int(i)] for i in pred_idx], dtype=object)
                max_p = max_p_arr if max_p_arr is not None else np.ones(len(X))
                proba_matrix = probs_full
                proba_class_names = [str(c) for c in classes]
                model_kind = "vfl_torch"
                n_rows = len(X)
                shap_rows: list[dict[str, Any] | None] | None = None
                if not want_shap:
                    shap_meta = {"status": "skipped", "detail": "compute_shap=false in job config"}
                elif n_rows > MAX_ROWS_FOR_SHAP:
                    shap_meta = {
                        "status": "skipped",
                        "detail": (
                            f"n_rows={n_rows} exceeds API VFL attribution limit {MAX_ROWS_FOR_SHAP}; "
                            "use a smaller CSV or the VFL SHAP notebook for full analysis."
                        ),
                    }
                else:
                    try:
                        shap_rows = vfl_gradient_x_input_attribution_rows(bundle, X, pred_idx)
                        n_ok = sum(1 for r in shap_rows if r is not None and r.get("per_feature"))
                        shap_meta = {
                            "status": "computed" if n_ok == n_rows else "partial",
                            "detail": (
                                f"gradient×input attribution for predicted-class logit ({n_ok}/{n_rows} rows); "
                                "not TreeExplainer SHAP but compatible with per-agent RAG templates."
                            ),
                        }
                    except Exception as e:
                        logger.warning("VFL gradient attribution failed: %s", e)
                        shap_rows = None
                        shap_meta = {
                            "status": "unavailable",
                            "detail": str(e)[:500],
                        }
            else:
                pipe = bundle["pipeline"]
                le_y = bundle.get("target_encoder")
                pred = pipe.predict(X)
                raw_proba = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        raw_proba = pipe.predict_proba(X)
                    except Exception:
                        raw_proba = None

                if le_y is not None:
                    labels = le_y.inverse_transform(np.asarray(pred).astype(int))
                else:
                    labels = pred

                if raw_proba is not None:
                    proba_matrix = np.asarray(raw_proba, dtype=float)
                    max_p = proba_matrix.max(axis=1)
                    if le_y is not None:
                        proba_class_names = [str(x) for x in le_y.classes_.tolist()]
                    else:
                        proba_class_names = [str(j) for j in range(proba_matrix.shape[1])]
                else:
                    max_p = np.ones(len(X))

                model_kind = "sklearn_pipeline"
                shap_meta = {"status": "skipped", "detail": None}
                shap_rows = None
                n_rows = len(X)
                if not want_shap:
                    shap_meta = {"status": "skipped", "detail": "compute_shap=false in job config"}
                elif n_rows > MAX_ROWS_FOR_SHAP:
                    shap_meta = {
                        "status": "skipped",
                        "detail": f"n_rows={n_rows} exceeds API SHAP limit {MAX_ROWS_FOR_SHAP}",
                    }
                else:
                    shap_rows = compute_sklearn_tree_shap_per_row(pipe, X)
                    if shap_rows is None:
                        shap_meta = {
                            "status": "unavailable",
                            "detail": "TreeExplainer not run (install shap, or classifier is not a supported tree model).",
                        }
                    else:
                        n_ok = sum(1 for r in shap_rows if r is not None)
                        shap_meta = {
                            "status": "computed" if n_ok == n_rows else "partial",
                            "detail": f"SHAP values for {n_ok}/{n_rows} rows",
                        }

            out_df = df.copy()
            out_df["predicted_label"] = labels
            out_df["max_class_probability"] = max_p

            flags = np.zeros(len(out_df), dtype=bool)
            if threshold is not None:
                flags |= max_p < float(threshold)
            if attack_vals:
                flags |= out_df["predicted_label"].astype(str).isin([str(x) for x in attack_vals])

            out_df["flagged_attack_or_anomaly"] = flags

            n = len(out_df)
            rows_json: list[dict] = []
            for i in range(n):
                prob_row = None
                if proba_matrix is not None:
                    pr = proba_matrix[i]
                    names = (
                        proba_class_names
                        if len(proba_class_names) == len(pr)
                        else [str(j) for j in range(len(pr))]
                    )
                    prob_row = {names[j]: float(pr[j]) for j in range(len(pr))}

                if bundle.get("kind") == "vfl_torch":
                    if shap_rows is not None and i < len(shap_rows) and shap_rows[i] is not None:
                        shap_cell = shap_rows[i]
                    else:
                        shap_cell = {
                            "status": shap_meta.get("status", "unavailable"),
                            "model_kind": "vfl_torch",
                            "note": shap_meta.get("detail"),
                        }
                elif shap_rows is not None and i < len(shap_rows) and shap_rows[i] is not None:
                    shap_cell = shap_rows[i]
                elif shap_rows is not None:
                    shap_cell = {"status": "unavailable", "note": "no values for this row"}
                else:
                    shap_cell = {
                        "status": shap_meta.get("status", "skipped"),
                        "note": shap_meta.get("detail"),
                    }

                rows_json.append(
                    {
                        "row_index": i,
                        "predicted_label": str(labels[i]),
                        "max_class_probability": float(max_p[i]),
                        "flagged_attack_or_anomaly": bool(flags[i]),
                        "class_probabilities": prob_row,
                        "shap": shap_cell,
                    }
                )

            results_payload = {
                "schema_version": 1,
                "model_kind": model_kind,
                "target_column": target_col,
                "feature_columns": list(feature_columns),
                "shap": shap_meta,
                "rows": rows_json,
            }

            out_name = f"pred_{job.public_id}.csv"
            out_abs = settings.storage_root / "predictions" / out_name
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_abs, index=False)

            job.output_path = str(out_abs.relative_to(settings.storage_root))
            job.rows_total = int(len(out_df))
            job.rows_flagged = int(flags.sum())
            job.results_json = results_payload
            job.status = JobStatus.completed
            job.error_message = None
            db.commit()
            logger.info("Prediction job=%s rows=%s flagged=%s", job.public_id, job.rows_total, job.rows_flagged)
        except Exception as e:
            logger.exception("Prediction failed job=%s", job.public_id)
            job.status = JobStatus.failed
            job.error_message = str(e)[:8000]
            db.commit()
    finally:
        db.close()


def load_prediction_summary(settings: Settings, job: PredictionJob) -> dict:
    summary = {
        "prediction_job_public_id": job.public_id,
        "status": job.status.value,
        "rows_total": job.rows_total,
        "rows_flagged": job.rows_flagged,
        "output_relative_path": job.output_path,
    }
    rj = job.results_json
    if isinstance(rj, dict):
        sh = rj.get("shap") if isinstance(rj.get("shap"), dict) else {}
        summary["results_schema_version"] = rj.get("schema_version")
        summary["results_model_kind"] = rj.get("model_kind")
        summary["results_shap_status"] = sh.get("status")
        summary["results_shap_detail"] = sh.get("detail")
        rows = rj.get("rows")
        if isinstance(rows, list) and rows:
            r0 = rows[0] if isinstance(rows[0], dict) else {}
            sh0 = r0.get("shap") if isinstance(r0.get("shap"), dict) else {}
            summary["sample_row_0"] = {
                "predicted_label": r0.get("predicted_label"),
                "max_class_probability": r0.get("max_class_probability"),
                "flagged_attack_or_anomaly": r0.get("flagged_attack_or_anomaly"),
                "shap_method": sh0.get("method") or sh0.get("status"),
            }
    if job.output_path:
        p = settings.storage_root / job.output_path
        if p.is_file():
            df = pd.read_csv(p)
            summary["preview_columns"] = list(df.columns)
            summary["head_json"] = json.loads(df.head(5).to_json(orient="records"))
    return summary
