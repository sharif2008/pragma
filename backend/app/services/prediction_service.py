"""Batch prediction from CSV using a persisted model bundle."""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import JobStatus, ManagedFile, ModelVersion, PredictionJob
from app.services import file_service
from app.services.ml_training import load_model_bundle

logger = logging.getLogger(__name__)


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


def list_prediction_jobs(db: Session, *, limit: int = 100, offset: int = 0) -> list[PredictionJob]:
    q = (
        select(PredictionJob)
        .order_by(PredictionJob.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


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

            if bundle.get("kind") == "vfl_torch":
                from app.services.ml_vfl import predict_vfl_batch

                pred_idx, max_p_arr = predict_vfl_batch(bundle, X)
                classes: list = bundle["label_classes"]
                labels = np.array([classes[int(i)] for i in pred_idx], dtype=object)
                max_p = max_p_arr if max_p_arr is not None else np.ones(len(X))
            else:
                pipe = bundle["pipeline"]
                le_y = bundle.get("target_encoder")
                pred = pipe.predict(X)
                proba = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X)
                    except Exception:
                        proba = None

                if le_y is not None:
                    labels = le_y.inverse_transform(np.asarray(pred).astype(int))
                else:
                    labels = pred

                if proba is not None:
                    max_p = proba.max(axis=1)
                else:
                    max_p = np.ones(len(X))

            out_df = df.copy()
            out_df["predicted_label"] = labels
            out_df["max_class_probability"] = max_p

            cfg = job.config_json or {}
            threshold = cfg.get("anomaly_probability_threshold")
            attack_vals = cfg.get("attack_label_values") or []

            flags = np.zeros(len(out_df), dtype=bool)
            if threshold is not None:
                flags |= max_p < float(threshold)
            if attack_vals:
                flags |= out_df["predicted_label"].astype(str).isin([str(x) for x in attack_vals])

            out_df["flagged_attack_or_anomaly"] = flags

            out_name = f"pred_{job.public_id}.csv"
            out_abs = settings.storage_root / "predictions" / out_name
            out_abs.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_abs, index=False)

            job.output_path = str(out_abs.relative_to(settings.storage_root))
            job.rows_total = int(len(out_df))
            job.rows_flagged = int(flags.sum())
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
    if job.output_path:
        p = settings.storage_root / job.output_path
        if p.is_file():
            df = pd.read_csv(p)
            summary["preview_columns"] = list(df.columns)
            summary["head_json"] = json.loads(df.head(5).to_json(orient="records"))
    return summary
