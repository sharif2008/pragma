#!/usr/bin/env python3
"""
CSV → agent → on-chain trust-anchor benchmark (PRAGMA-style timings).

- Read rows from a CSV file
- Score rows with a **joblib** bundle (``--model-bundle``, or by default the **newest** ``*.joblib`` under ``STORAGE_ROOT/models``) for ``predicted_label``, ``max_class_probability``, and optional TreeSHAP / VFL attribution — or pass ``--no-model`` for CSV-only fallback
- When a bundle is used, the CSV **label** column (``--label-col``, default ``label``) is treated as **ground truth** if present (same name as training ``target_column``); it is not used as the model prediction
- After inference: derive top-5 **SHAP φ** contributions for the response (``shap.per_feature`` when present, else a column proxy — see ``shap_contribution_for_response`` in ``sample_data``)
- Auto-build **one** template RAG query from label + confidence + those conditions (no LLM-based query refinement — template only)
- Retrieve KB context via lightweight TF-IDF (meta.json only; no SentenceTransformers) unless ``--no-rag``. By default only chunks whose ingest ``source`` is a **.pdf** are scored (use ``--rag-all-sources`` for .md/.txt/.json too)
- Call the agent once with the same ``sample_data`` shape as the API (prediction_row + SHAP-style top-5 for the LLM)
- Anchor a hash-only commitment on-chain via AgenticTrustRegistry (Hardhat / permissioned-style Ethereum)
- Validate by reading commitment back from chain and recomputing hash from the canonical payload
- Track per-step timings and emit a per-row report table

Default output: ``storage/reports/csv_trust_anchor_<dataset>_<timestamp>.csv`` plus a matching
``*_benchmark_report.txt`` (mean timing, % of E2E, variability, trust outcomes).

This script is intentionally self-contained and does not require the API server to be running.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `python scripts\\trust_anchor_benchmark.py` from `backend/` without PYTHONPATH.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_backend_root = str(_BACKEND_DIR)
if _backend_root not in sys.path:
    sys.path.insert(0, _backend_root)

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
try:
    from web3 import Web3
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: web3.py.\n"
        "Run this script in the backend virtualenv, e.g.:\n"
        "  cd backend\n"
        "  .venv\\Scripts\\activate   (Windows)\n"
        "  pip install -r requirements.txt\n"
        "Then re-run the script.\n"
        f"Import error: {e}"
    )

from app.core.config import Settings, get_settings
from app.services import llm_service, trust_chain_service
from app.services.agentic_llm_prompt import (
    LLM_ORCHESTRATION_TOP_SHAP_FEATURES,
    load_attack_agentic_config,
    row_to_shap_explanation,
)
from app.services.ml_training import load_model_bundle
from app.services.prediction_shap import (
    RESULTS_JSON_TOP_SHAP_FEATURES,
    compute_sklearn_tree_shap_per_row,
    limit_shap_per_feature_by_abs,
)

# Align with prediction_service batch SHAP limits.
MAX_ROWS_FOR_MODEL_SHAP = 800


def _default_model_bundle_path(settings: Settings) -> Path | None:
    """
    Newest ``*.joblib`` under ``STORAGE_ROOT/models`` (by mtime), or None if none exist.
    Used when ``--model-bundle`` is omitted so the script can run real inference by default.
    """
    root = settings.storage_root / "models"
    if not root.is_dir():
        return None
    candidates = [p for p in root.rglob("*.joblib") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

TOP_CONDITIONS_FOR_LLM = 5

# ``shap_source`` values for ``shap_contribution_for_response`` (SHAP φ vs CSV proxy).
SHAP_SOURCE_MODEL = "shap_per_feature"
SHAP_SOURCE_PROXY = "synthetic_column_proxy"

_rag_pdf_fallback_logged = False
_rag_junk_only_logged = False


def _chunk_source_is_pdf(ch: dict[str, Any]) -> bool:
    """KB chunks from ``ingest_kb_document`` store the upload filename in ``source``."""
    src = str(ch.get("source") or "").strip().lower()
    return src.endswith(".pdf")


def _chunk_display_source(ch: dict[str, Any]) -> str:
    s = str(ch.get("source") or "").strip()
    return s if s else "(unknown source)"


def _is_junk_rag_chunk(text: str) -> bool:
    """
    Drop serialized prediction / SHAP JSON that was mistakenly indexed as KB — it TF-IDF-matches flow queries.
    """
    s = text.strip()
    if len(s) < 24:
        return False
    low = s.lower()
    markers = (
        "per_feature",
        "attribution_class_index",
        "expected_value",
        '"shap"',
        "'shap'",
        "gradient×input",
        "dominant_contribution_pct",
    )
    hit = sum(1 for m in markers if m in low)
    if hit >= 2:
        return True
    if "per_feature" in low and ("{" in s or '"' in s):
        return (s.count("{") + s.count("}")) >= 3
    return False


def _normalize_rag_excerpt(text: str, max_len: int = 800) -> str:
    """Single-line-ish excerpt for prompts (keeps words, trims noise)."""
    one = " ".join(text.split())
    return one[:max_len] + ("…" if len(one) > max_len else "")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _perf_ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _short_hex(h: str | None, n: int = 12) -> str:
    if not h:
        return "(none)"
    s = h.strip().lower()
    if s.startswith("0x"):
        s = s[2:]
    return (s[:n] + "…") if len(s) > n else s


def _print_step(name: str, ms: float, detail: str = "") -> None:
    suffix = f"  | {detail}" if detail else ""
    print(f"    {name:<28} {ms:8.2f} ms{suffix}")



def _parse_structured_plan(raw_llm_response: str | None) -> Any | None:
    if not isinstance(raw_llm_response, str) or not raw_llm_response.strip():
        return None
    m = re.search(r"\{[\s\S]*\}", raw_llm_response)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _infer_default(
    row: dict[str, Any],
    *,
    label_col: str | None,
    confidence_col: str | None,
) -> tuple[str, float]:
    """
    Lightweight inference when no ``--model-bundle`` is used (benchmark-only).

    - If ``label_col`` exists, its value is used as a stand-in **predicted** label (not real inference).
    - If ``confidence_col`` exists, use it; otherwise set a conservative default.
    When you need real scores and SHAP, pass ``--model-bundle``; the CSV label column is then ground truth only.
    """
    lab = None
    if label_col:
        v = row.get(label_col)
        if v is not None and str(v).strip():
            lab = str(v).strip()
    predicted = (lab or "UNKNOWN").upper()

    conf = None
    if confidence_col:
        cv = row.get(confidence_col)
        conf = _safe_float(cv, default=None) if cv is not None else None
    if conf is None:
        conf = 0.75 if predicted not in ("BENIGN", "NORMAL", "0") else 0.55
    conf = max(0.0, min(1.0, float(conf)))
    return predicted, conf


def _ground_truth_from_csv(
    rec: dict[str, Any],
    bundle: dict[str, Any] | None,
    label_col: str | None,
) -> str | None:
    """True label from CSV if the column exists (prefer bundle ``target_column``, else ``label_col``)."""
    keys: list[str] = []
    if bundle and isinstance(bundle.get("target_column"), str):
        keys.append(str(bundle["target_column"]))
    if label_col:
        keys.append(str(label_col))
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        if key not in rec:
            continue
        v = rec.get(key)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _batch_model_inference(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    compute_shap: bool,
) -> list[dict[str, Any]]:
    """
    One forward pass + optional XAI: same contract as the API prediction job (sklearn tree SHAP or VFL gradient×input).
    """
    feature_columns: list[str] = list(bundle["feature_columns"])
    dfc = df.copy()
    for c in feature_columns:
        if c not in dfc.columns:
            dfc[c] = np.nan
    X = dfc[feature_columns]
    n_rows = len(X)
    out: list[dict[str, Any]] = []

    if bundle.get("kind") == "vfl_torch":
        from app.services import ml_vfl

        pred_idx, max_p_arr, probs_full = ml_vfl.predict_vfl_batch(bundle, X, return_probs=True)
        classes: list = bundle["label_classes"]
        labels = np.array([classes[int(i)] for i in pred_idx], dtype=object)
        max_p = max_p_arr if max_p_arr is not None else np.ones(n_rows, dtype=float)
        proba_matrix = probs_full
        proba_class_names = [str(c) for c in classes]
        shap_meta: dict[str, Any] = {"status": "skipped", "detail": None}
        shap_rows: list[dict[str, Any] | None] | None = None
        if not compute_shap:
            shap_meta = {"status": "skipped", "detail": "benchmark --no-model-shap"}
        elif n_rows > MAX_ROWS_FOR_MODEL_SHAP:
            shap_meta = {"status": "skipped", "detail": f"n_rows={n_rows} exceeds limit {MAX_ROWS_FOR_MODEL_SHAP}"}
        else:
            try:
                shap_rows = ml_vfl.vfl_gradient_x_input_attribution_rows(bundle, X, pred_idx)
                n_ok = sum(1 for r in shap_rows if r is not None and isinstance(r, dict) and r.get("per_feature"))
                shap_meta = {
                    "status": "computed" if n_ok == n_rows else "partial",
                    "detail": f"VFL gradient×input attribution {n_ok}/{n_rows} rows",
                }
            except Exception as e:
                shap_rows = None
                shap_meta = {"status": "unavailable", "detail": str(e)[:500]}

        for i in range(n_rows):
            prob_row = None
            if proba_matrix is not None:
                pr = proba_matrix[i]
                prob_row = {proba_class_names[j]: float(pr[j]) for j in range(len(pr))}
            if shap_rows is not None and i < len(shap_rows) and shap_rows[i] is not None:
                shap_cell = shap_rows[i]
            else:
                shap_cell = {
                    "status": shap_meta.get("status", "unavailable"),
                    "model_kind": "vfl_torch",
                    "note": shap_meta.get("detail"),
                }
            if isinstance(shap_cell, dict) and isinstance(shap_cell.get("per_feature"), dict):
                shap_cell = limit_shap_per_feature_by_abs(shap_cell, RESULTS_JSON_TOP_SHAP_FEATURES)
            out.append(
                {
                    "predicted_label": str(labels[i]),
                    "max_class_probability": float(max_p[i]),
                    "shap": shap_cell,
                    "class_probabilities": prob_row,
                }
            )
        return out

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
        proba_matrix = None
        max_p = np.ones(n_rows, dtype=float)
        proba_class_names = []

    shap_meta = {"status": "skipped", "detail": None}
    shap_rows = None
    if not compute_shap:
        shap_meta = {"status": "skipped", "detail": "benchmark --no-model-shap"}
    elif n_rows > MAX_ROWS_FOR_MODEL_SHAP:
        shap_meta = {"status": "skipped", "detail": f"n_rows={n_rows} exceeds limit {MAX_ROWS_FOR_MODEL_SHAP}"}
    else:
        shap_rows = compute_sklearn_tree_shap_per_row(pipe, X)
        if shap_rows is None:
            shap_meta = {"status": "unavailable", "detail": "TreeExplainer not run (install shap or unsupported clf)"}
        else:
            n_ok = sum(1 for r in shap_rows if r is not None)
            shap_meta = {"status": "computed" if n_ok == n_rows else "partial", "detail": f"SHAP {n_ok}/{n_rows} rows"}

    for i in range(n_rows):
        prob_row = None
        if proba_matrix is not None:
            pr = proba_matrix[i]
            names = (
                proba_class_names
                if len(proba_class_names) == len(pr)
                else [str(j) for j in range(len(pr))]
            )
            prob_row = {names[j]: float(pr[j]) for j in range(len(pr))}
        if shap_rows is not None and i < len(shap_rows) and shap_rows[i] is not None:
            shap_cell = shap_rows[i]
        elif shap_rows is not None:
            shap_cell = {"status": "unavailable", "note": "no values for this row"}
        else:
            shap_cell = {"status": shap_meta.get("status", "skipped"), "note": shap_meta.get("detail")}
        if isinstance(shap_cell, dict) and isinstance(shap_cell.get("per_feature"), dict):
            shap_cell = limit_shap_per_feature_by_abs(shap_cell, RESULTS_JSON_TOP_SHAP_FEATURES)
        out.append(
            {
                "predicted_label": str(labels[i]),
                "max_class_probability": float(max_p[i]),
                "shap": shap_cell,
                "class_probabilities": prob_row,
            }
        )
    return out


def _ignore_cols(label_col: str | None, confidence_col: str | None) -> set[str]:
    out: set[str] = set()
    if label_col:
        out.add(str(label_col))
    if confidence_col:
        out.add(str(confidence_col))
    return out


def _top_conditions_from_row(
    rec: dict[str, Any],
    *,
    label_col: str | None,
    confidence_col: str | None,
    k: int = TOP_CONDITIONS_FOR_LLM,
) -> tuple[list[tuple[str, float]], str]:
    """
    Top-k **SHAP φ** (or proxy) contributions for the orchestration response.

    Prefer ``shap.per_feature`` from the row (true model attributions); otherwise rank raw CSV
    columns by magnitude — those values are **not** SHAP φ (see ``SHAP_SOURCE_PROXY``).
    """
    sh = rec.get("shap")
    if isinstance(sh, dict):
        pf = sh.get("per_feature")
        if isinstance(pf, dict) and pf:
            limited = limit_shap_per_feature_by_abs(sh, k)
            inner = limited.get("per_feature") if isinstance(limited.get("per_feature"), dict) else {}
            scored: list[tuple[str, float]] = []
            for name, val in inner.items():
                try:
                    scored.append((str(name), float(val)))
                except (TypeError, ValueError):
                    continue
            scored.sort(key=lambda x: abs(x[1]), reverse=True)
            return scored[:k], SHAP_SOURCE_MODEL

    ignore = _ignore_cols(label_col, confidence_col)
    scored = []
    for key, val in rec.items():
        if key in ignore or key == "shap":
            continue
        if isinstance(val, bool):
            continue
        if val is None:
            continue
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            continue
        try:
            fv = float(val)
            if not math.isnan(fv) and not math.isinf(fv):
                scored.append((str(key), fv))
                continue
        except (TypeError, ValueError):
            pass
        s = str(val).strip()
        if s:
            scored.append((str(key), float(len(s))))
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    return scored[:k], SHAP_SOURCE_PROXY


def _shap_contribution_for_response(
    top: list[tuple[str, float]],
    *,
    shap_source: str,
) -> dict[str, Any]:
    """
    Normalized SHAP-style contribution **shape** for the LLM: φ, |φ|, and share of total |φ|.

    Grounds the JSON plan in explainability; distinguish real φ from column proxies via ``shap_source``.
    """
    total_abs = sum(abs(v) for _, v in top) or 1.0
    feats: list[dict[str, Any]] = []
    for name, phi in top:
        a = abs(phi)
        feats.append(
            {
                "feature": name,
                "phi": phi,
                "abs_phi": a,
                "share_of_total_abs_phi": round(a / total_abs, 6),
            }
        )
    return {
        "shap_source": shap_source,
        "interpretation": (
            "Model SHAP values (φ). Tie actions and reasoning to these shares — they shape the response evidence."
            if shap_source == SHAP_SOURCE_MODEL
            else (
                "Column-magnitude proxy only (not φ). Use for ordering hints until batch predictions attach "
                "real SHAP per_feature to each row."
            )
        ),
        "top_features_by_abs_phi": feats,
    }


def _build_single_template_rag_query(
    predicted_label: str,
    confidence: float,
    conditions: list[tuple[str, float]],
    *,
    shap_source: str,
) -> str:
    """Single template retrieval string for TF-IDF (benchmark does not LLM-refine the query)."""
    label_phi = "SHAP φ (top features)" if shap_source == SHAP_SOURCE_MODEL else "top column signals (SHAP proxy)"
    bits = ", ".join(f"{name} ({val:+.4g})" for name, val in conditions[:TOP_CONDITIONS_FOR_LLM]) or "(no feature signals)"
    return (
        "SOC triage and mitigation runbooks for a scored network flow: "
        f"predicted_label={predicted_label}, confidence={confidence:.2f}. "
        f"{label_phi}: {bits}. "
        "Include containment, monitoring, validation, and escalation decision points. "
        "5G/6G operator context: triangulate RAN, edge MEC, and 5GC core using these indicators."
    )


def _prediction_row_for_prompt_local(row: dict[str, Any]) -> dict[str, Any]:
    pr = dict(row)
    shp = pr.get("shap")
    if isinstance(shp, dict):
        pr["shap"] = limit_shap_per_feature_by_abs(shp, LLM_ORCHESTRATION_TOP_SHAP_FEATURES)
    return pr


def _csv_row_to_sample_data(
    row_index: int,
    predicted_label: str,
    confidence: float,
    rec: dict[str, Any],
    top_conditions: list[tuple[str, float]],
    *,
    label_col: str | None,
    confidence_col: str | None,
    shap_source: str,
    ground_truth_label: str | None = None,
    class_probabilities: dict[str, float] | None = None,
) -> dict[str, Any]:
    sh_cell: dict[str, Any]
    if shap_source == SHAP_SOURCE_MODEL:
        base = rec.get("shap")
        if isinstance(base, dict) and isinstance(base.get("per_feature"), dict) and base.get("per_feature"):
            sh_cell = dict(base)
            sh_cell["per_feature"] = dict(base["per_feature"])
            sh_cell = limit_shap_per_feature_by_abs(sh_cell, TOP_CONDITIONS_FOR_LLM)
        else:
            sh_cell = {
                "method": "benchmark_top5",
                "status": "shap_missing_fallback",
                "per_feature": {str(k): float(v) for k, v in top_conditions},
            }
            sh_cell = limit_shap_per_feature_by_abs(sh_cell, TOP_CONDITIONS_FOR_LLM)
    else:
        sh_cell = {
            "method": "column_magnitude_proxy",
            "status": "not_model_shap",
            "per_feature": {str(k): float(v) for k, v in top_conditions},
        }
        sh_cell = limit_shap_per_feature_by_abs(sh_cell, TOP_CONDITIONS_FOR_LLM)

    prediction_row: dict[str, Any] = {
        "row_index": row_index,
        "predicted_label": predicted_label,
        "max_class_probability": confidence,
        "shap": sh_cell,
    }

    shap_for_response = _shap_contribution_for_response(top_conditions, shap_source=shap_source)

    # Slim JSON for the LLM user message (see app.services.agentic_llm_prompt orchestration prompt).
    orchestration_llm_payload: dict[str, Any] = {
        "sample_id": row_index,
        "predicted_label": predicted_label,
        "confidence": float(confidence),
        "max_class_probability": float(confidence),
        "top_5_feature_attributions": [
            {
                "feature": x.get("feature"),
                "phi": x.get("phi"),
                "abs_phi": x.get("abs_phi"),
                "share_of_total_abs_phi": x.get("share_of_total_abs_phi"),
            }
            for x in (shap_for_response.get("top_features_by_abs_phi") or [])[:TOP_CONDITIONS_FOR_LLM]
        ],
        "attribution_source": shap_for_response.get("shap_source"),
    }
    if ground_truth_label is not None:
        orchestration_llm_payload["ground_truth_label"] = ground_truth_label
    if class_probabilities is not None:
        orchestration_llm_payload["class_probabilities"] = class_probabilities

    out: dict[str, Any] = {
        "sample_id": row_index,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "shap_explanation": row_to_shap_explanation(prediction_row),
        "shap_contribution_for_response": shap_for_response,
        "prediction_row": _prediction_row_for_prompt_local(prediction_row),
        "row_selection": "csv_benchmark_top5",
        "orchestration_llm_payload": orchestration_llm_payload,
    }
    if ground_truth_label is not None:
        out["ground_truth_label"] = ground_truth_label
    if class_probabilities is not None:
        out["class_probabilities"] = class_probabilities
    return out


def _feature_notes_from_conditions(top: list[tuple[str, float]], *, shap_source: str) -> str:
    if not top:
        return "SHAP contribution for response: (none derived from this row)."
    body = "; ".join(f"{name} φ={val:.4g}" for name, val in top[:TOP_CONDITIONS_FOR_LLM])
    kind = "model SHAP φ" if shap_source == SHAP_SOURCE_MODEL else "proxy weights (not φ)"
    return (
        f"Top {min(len(top), TOP_CONDITIONS_FOR_LLM)} features by |φ| ({kind}) — use shares in "
        f"shap_contribution_for_response to shape reasoning: {body}"
    )


def _maybe_query_kb(
    settings,
    *,
    query: str,
    prefer_pdf_sources: bool = True,
) -> tuple[str | None, int]:
    """
    TF-IDF cosine similarity over KB chunk texts (reads meta.json under vector_db only).

    Cleans retrieval: drops artifact-like JSON/SHAP blobs; by default scores only chunks ingested from
    ``*.pdf`` (see ``source`` on each chunk). Falls back once to all non-junk sources if no PDF chunks exist.

    Avoids SentenceTransformers / FAISS embedding search so the benchmark does not load Bert/MiniLM
    or print transformers weight-loading noise.
    """
    global _rag_pdf_fallback_logged, _rag_junk_only_logged

    try:
        from app.db.session import SessionLocal
        from app.models.domain import KnowledgeBaseFile
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sqlalchemy import select

        db = SessionLocal()
    except Exception:
        return None, 0

    try:
        rows = list(db.scalars(select(KnowledgeBaseFile)).all())
        paired_raw: list[tuple[dict[str, Any], str]] = []
        for kb in rows:
            meta_path = settings.storage_root / kb.vector_index_dir / "meta.json"
            if not meta_path.is_file():
                continue
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            for ch in raw.get("chunks", []):
                if isinstance(ch, dict) and str(ch.get("text", "")).strip():
                    paired_raw.append((ch, kb.public_id))

        paired_clean = [
            p for p in paired_raw if not _is_junk_rag_chunk(str(p[0].get("text", "")))
        ]
        if not paired_clean and paired_raw and not _rag_junk_only_logged:
            print(
                "    (rag) all KB chunks look like JSON/SHAP artifacts — none usable; ingest policy PDFs via upload_kb_files.py",
                flush=True,
            )
            _rag_junk_only_logged = True
            return None, 0

        paired = paired_clean
        if prefer_pdf_sources:
            pdf_pairs = [p for p in paired if _chunk_source_is_pdf(p[0])]
            if pdf_pairs:
                paired = pdf_pairs
            elif paired and not _rag_pdf_fallback_logged:
                print(
                    "    (rag) no PDF-sourced chunks — using other cleaned sources (.md/.txt/.json)",
                    flush=True,
                )
                _rag_pdf_fallback_logged = True

        if not paired:
            return None, 0

        texts = [str(p[0].get("text", "")) for p in paired]
        max_df = 0.95 if len(texts) > 1 else 1.0
        vectorizer = TfidfVectorizer(
            max_features=12_000,
            stop_words="english",
            min_df=1,
            max_df=max_df,
        )
        vectorizer.fit(texts)
        qv = vectorizer.transform([query])
        X = vectorizer.transform(texts)
        sims = cosine_similarity(qv, X).flatten()
        top_k = min(int(settings.rag_top_k), len(paired))
        order = np.argsort(-sims)[:top_k]
        lines: list[str] = []
        for idx in order:
            score = float(sims[idx])
            ch = paired[idx][0]
            src = _chunk_display_source(ch)
            txt = _normalize_rag_excerpt(str(ch.get("text", "")))
            lines.append(f"- ({score:.3f}) [{src}] {txt}")
        return "\n\n".join(lines), len(lines)
    except Exception:
        return None, 0
    finally:
        try:
            db.close()
        except Exception:
            pass


def _preflight_chain(settings) -> tuple[Web3, str]:
    if not settings.trust_chain_enabled:
        raise RuntimeError("TRUST_CHAIN_ENABLED is false (enable trust chain to anchor on-chain).")
    if not settings.trust_chain_rpc_url:
        raise RuntimeError("TRUST_CHAIN_RPC_URL missing.")
    if not settings.trust_chain_contract_address:
        raise RuntimeError("TRUST_CHAIN_CONTRACT_ADDRESS missing.")

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("Could not connect to TRUST_CHAIN_RPC_URL.")

    addr = Web3.to_checksum_address(settings.trust_chain_contract_address)
    code = w3.eth.get_code(addr)
    if not code or code == b"\x00":
        raise RuntimeError("No contract code at TRUST_CHAIN_CONTRACT_ADDRESS (did you deploy AgenticTrustRegistry?).")

    return w3, addr


async def _agent_once(
    settings,
    *,
    sample_data: dict[str, Any],
    feature_notes: str | None,
    rag_context: str | None,
    attack_actions_data: dict[str, Any] | None,
    agentic_features_data: dict[str, Any] | None,
    use_rag: bool,
) -> dict[str, str | None]:
    return await llm_service.agent_decide(
        settings,
        sample_data,
        feature_notes,
        rag_context,
        attack_actions_data,
        agentic_features_data,
        use_rag=use_rag,
    )


@dataclass
class RowTiming:
    row_index: int
    predicted_label: str
    confidence: float
    kb_hits: int
    infer_ms: float
    infer_shap_amortized_ms: float
    rag_ms: float
    commitment_ms: float
    agentic_action_ms: float
    blockchain_store_ms: float
    validation_ms: float
    pipeline_infer_rag_llm_ms: float
    trust_anchor_ms: float
    end_to_end_ms: float
    anchor_tx_hash: str
    chain_integrity_valid: bool
    payload_integrity_valid: bool
    executed: bool


def _pct(part: float, whole: float) -> float:
    return (100.0 * part / whole) if whole and whole > 0 else 0.0


def _series_min_max_std(s: pd.Series) -> tuple[float, float, float]:
    if s is None or len(s) == 0:
        return 0.0, 0.0, 0.0
    return float(s.min()), float(s.max()), float(s.std(ddof=0) if len(s) > 1 else 0.0)


# Human-readable column order for the timing CSV (extras appended at end).
_BENCHMARK_CSV_COLUMN_ORDER = [
    "row_index",
    "predicted_label",
    "confidence",
    "kb_hits",
    "infer_shap_amortized_ms",
    "infer_ms",
    "rag_ms",
    "agentic_action_ms",
    "pipeline_infer_rag_llm_ms",
    "commitment_ms",
    "blockchain_store_ms",
    "validation_ms",
    "trust_anchor_ms",
    "end_to_end_ms",
    "executed",
    "chain_integrity_valid",
    "payload_integrity_valid",
    "anchor_tx_hash",
]


def _reorder_benchmark_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _BENCHMARK_CSV_COLUMN_ORDER if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest] if cols else df


def build_benchmark_report_text(
    *,
    generated_at: datetime,
    csv_input: Path,
    wall_total_ms: float,
    out_df: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    """Readable report for engineers / SOC benchmarking (also printed to stdout)."""
    lines: list[str] = []
    w = 72

    def hdr(title: str) -> None:
        lines.append("")
        lines.append(title)
        lines.append("-" * min(len(title), w))

    lines.append("ChainAgentVFL — Trust anchor benchmark report")
    lines.append(f"Generated (UTC): {generated_at.isoformat()}")
    lines.append("=" * w)

    hdr("Run configuration")
    lines.append(f"  Input CSV ........................ {csv_input.name}")
    lines.append(f"  Full path ........................ {csv_input}")
    lines.append(f"  Rows processed ................... {meta.get('rows_processed', len(out_df))}")
    lines.append(f"  Model bundle ..................... {meta.get('model_bundle_display') or '(none — CSV fallback)'}")
    if meta.get("model_algorithm"):
        lines.append(f"  Bundle algorithm / kind .......... {meta.get('model_algorithm')}")
    if meta.get("target_column"):
        lines.append(f"  Training target column ........... {meta.get('target_column')}")
    mb = meta.get("model_batch_total_ms")
    if mb is not None and float(mb) > 0:
        lines.append(f"  Model batch wall time ............ {float(mb):.1f} ms (predict + SHAP, whole dataframe)")
    lines.append(f"  RAG before LLM ................... {'yes' if meta.get('use_rag') else 'no'}")
    lines.append(f"  RAG PDF-first chunks ............. {'no (--rag-all-sources)' if meta.get('rag_all_sources') else 'yes (default)'}")
    lines.append("  RAG query generation .............. template only (no LLM query refine in this benchmark)")
    lines.append(f"  Model SHAP / VFL attribution ..... {'skipped (--no-model-shap)' if meta.get('no_model_shap') else 'enabled (when bundle)'}")

    if len(out_df) == 0:
        lines.append("")
        lines.append("No rows in output — nothing to aggregate.")
        return "\n".join(lines)

    def _mean(col: str) -> float:
        return float(out_df[col].mean()) if col in out_df.columns else 0.0

    m_e2e = _mean("end_to_end_ms")
    m_pipe = _mean("pipeline_infer_rag_llm_ms")
    m_trust = _mean("trust_anchor_ms")
    m_infer_amort = _mean("infer_shap_amortized_ms")
    m_rag = _mean("rag_ms")
    m_agent = _mean("agentic_action_ms")
    m_commit = _mean("commitment_ms")
    m_anchor = _mean("blockchain_store_ms")
    m_val = _mean("validation_ms")

    hdr("Mean timing per row (ms)")
    lines.append(f"  Benchmark wall clock (full run) .... {wall_total_ms:.1f}")
    lines.append("")
    lines.append("  Pipeline group — inference* + RAG + orchestration LLM")
    lines.append(
        f"    inference* + SHAP (amortized) .. {m_infer_amort:8.1f}   ({ _pct(m_infer_amort, m_e2e):5.1f}% of row E2E)"
    )
    lines.append(f"    RAG (retrieve + TF-IDF path) ... {m_rag:8.1f}   ({ _pct(m_rag, m_e2e):5.1f}% of row E2E)")
    lines.append(f"    Agent / LLM orchestration ...... {m_agent:8.1f}   ({ _pct(m_agent, m_e2e):5.1f}% of row E2E)")
    lines.append(
        f"    ── pipeline subtotal .......... {m_pipe:8.1f}   ({ _pct(m_pipe, m_e2e):5.1f}% of row E2E)"
    )
    lines.append("")
    lines.append("  Trust anchor group — commitment hash + chain + verify")
    lines.append(f"    Parse + SHA-256 payload ........ {m_commit:8.1f}   ({ _pct(m_commit, m_e2e):5.1f}% of row E2E)")
    lines.append(f"    Anchor transaction ............. {m_anchor:8.1f}   ({ _pct(m_anchor, m_e2e):5.1f}% of row E2E)")
    lines.append(f"    Validate (read + re-hash) ...... {m_val:8.1f}   ({ _pct(m_val, m_e2e):5.1f}% of row E2E)")
    lines.append(
        f"    ── trust anchor subtotal ....... {m_trust:8.1f}   ({ _pct(m_trust, m_e2e):5.1f}% of row E2E)"
    )
    lines.append("")
    lines.append(f"  End-to-end (mean per row) ........ {m_e2e:.1f}")
    ov = m_e2e - m_pipe - m_trust
    lines.append(f"  Unaccounted vs components ........ {max(0.0, ov):.1f}  (scheduler / GC / prints)")
    lines.append("")
    lines.append(f"  In-loop infer timer (lookup) ..... {_mean('infer_ms'):.1f}  (often ~0 if batch model)")

    pmin, pmax, pstd = _series_min_max_std(out_df["pipeline_infer_rag_llm_ms"])
    emin, emax, estd = _series_min_max_std(out_df["end_to_end_ms"])
    hdr("Variability across rows (ms)")
    lines.append(f"  pipeline_infer_rag_llm_ms  min {pmin:.1f}   max {pmax:.1f}   stdev {pstd:.1f}")
    lines.append(f"  end_to_end_ms              min {emin:.1f}   max {emax:.1f}   stdev {estd:.1f}")

    if "executed" in out_df.columns:
        hdr("Trust outcomes")
        n = len(out_df)
        ok_exec = int(out_df["executed"].astype(bool).sum())
        ok_chain = int(out_df["chain_integrity_valid"].astype(bool).sum()) if "chain_integrity_valid" in out_df.columns else 0
        ok_pay = int(out_df["payload_integrity_valid"].astype(bool).sum()) if "payload_integrity_valid" in out_df.columns else 0
        lines.append(f"  executed (chain + payload OK) .... {ok_exec} / {n}")
        lines.append(f"  chain_integrity_valid ............ {ok_chain} / {n}")
        lines.append(f"  payload_integrity_valid .......... {ok_pay} / {n}")

    hdr("Output files")
    lines.append(f"  Row timings CSV .................. {meta.get('csv_out_path', '(see console)')}")
    lines.append(f"  This report ...................... {meta.get('report_out_path', '')}")

    lines.append("")
    lines.append("* inference row share = model_batch_total_ms / n_rows when a joblib bundle is used.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    # Quieter HuggingFace/transformers if anything in the stack touches it during the run.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    ap = argparse.ArgumentParser(
        description="CSV rows → agentic action → on-chain trust anchor benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from the backend/ directory):
  python scripts\\upload_kb_files.py ..\\docs\\some_guide.pdf
  python scripts\\trust_anchor_benchmark.py --csv ..\\sample.csv
  python scripts\\trust_anchor_benchmark.py --csv C:\\data\\flows.csv --max-rows 100
  python scripts\\trust_anchor_benchmark.py --csv ..\\sample.csv --out reports\\my_run.csv
  python scripts\\trust_anchor_benchmark.py --csv ..\\sample.csv --no-rag
  python scripts\\trust_anchor_benchmark.py --csv data.csv --model-bundle storage\\models\\my_run\\model.joblib
  python scripts\\trust_anchor_benchmark.py --csv data.csv --no-model

Requires TRUST_CHAIN_ENABLED, TRUST_CHAIN_PRIVATE_KEY, and a local JSON-RPC node.
By default a new AgenticTrustRegistry is deployed each run (no TRUST_CHAIN_CONTRACT_ADDRESS needed).
Use --no-deploy-contract to reuse TRUST_CHAIN_CONTRACT_ADDRESS from the environment.
""",
    )
    ap.add_argument(
        "--csv",
        required=True,
        help="Path to the input CSV (required). Example: ..\\sample.csv",
    )
    ap.add_argument("--max-rows", type=int, default=25, help="Max rows to process (default 25).")
    ap.add_argument(
        "--model-bundle",
        default=None,
        metavar="PATH",
        help="Joblib artifact from training (sklearn pipeline or VFL). If omitted, uses the newest *.joblib under "
        "STORAGE_ROOT/models (set --no-model to skip and use CSV fallback only).",
    )
    ap.add_argument(
        "--no-model",
        action="store_true",
        help="Do not load any joblib bundle (disables default pick under STORAGE_ROOT/models); use CSV fallback inference.",
    )
    ap.add_argument(
        "--no-model-shap",
        action="store_true",
        help="With --model-bundle: skip TreeExplainer / VFL attribution (faster).",
    )
    ap.add_argument(
        "--label-col",
        default="label",
        help="CSV column for the true/target label. With --model-bundle this is ground truth only (not the prediction). "
        "Without --model-bundle: still used as the weak benchmark fallback \"prediction\".",
    )
    ap.add_argument(
        "--confidence-col",
        default=None,
        help="Optional confidence column name used for fallback inference.",
    )
    ap.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip KB retrieval before the LLM (default: single template query → TF-IDF over chunks).",
    )
    ap.add_argument(
        "--rag-all-sources",
        action="store_true",
        help="Score all ingested document types after junk-filtering (default: prefer chunks whose source ends with .pdf).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path for the timing report (default: backend/storage/reports/…).",
    )
    ap.add_argument(
        "--no-deploy-contract",
        action="store_true",
        help="Skip deploying a new registry; use TRUST_CHAIN_CONTRACT_ADDRESS from the environment.",
    )
    args = ap.parse_args()

    settings = get_settings()
    attack_actions_data, agentic_features_data = load_attack_agentic_config(verbose=False)

    if not settings.trust_chain_enabled:
        raise RuntimeError(
            "TRUST_CHAIN_ENABLED is false. Set TRUST_CHAIN_ENABLED=true for on-chain anchoring."
        )

    if not args.no_deploy_contract:
        t_dep0 = time.perf_counter()
        new_addr = trust_chain_service.deploy_fresh_registry(settings)
        os.environ["TRUST_CHAIN_CONTRACT_ADDRESS"] = new_addr
        get_settings.cache_clear()
        settings = get_settings()
        dep_ms = _perf_ms(t_dep0, time.perf_counter())
        print(f"[setup] deploy contract    {dep_ms:8.2f} ms  address={new_addr}")
    elif not settings.trust_chain_contract_address:
        raise RuntimeError(
            "TRUST_CHAIN_CONTRACT_ADDRESS is missing. Remove --no-deploy-contract for auto-deploy, "
            "or set the variable after deploying (e.g. npm run deploy:local in hardhat-blockchain)."
        )

    t_setup0 = time.perf_counter()
    _w3, contract_addr = _preflight_chain(settings)
    preflight_ms = _perf_ms(t_setup0, time.perf_counter())
    print(
        f"[setup] chain preflight     {preflight_ms:8.2f} ms  "
        f"rpc={settings.trust_chain_rpc_url}  contract={contract_addr}  chain_id={settings.trust_chain_chain_id}"
    )

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))

    t_csv0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows))
    read_csv_ms = _perf_ms(t_csv0, time.perf_counter())
    use_rag = not bool(args.no_rag)
    print(
        f"[setup] read csv            {read_csv_ms:8.2f} ms  "
        f"rows={len(df)}  file={csv_path.name}  rag_before_llm={use_rag}"
    )

    bundle: dict[str, Any] | None = None
    model_extras: list[dict[str, Any]] | None = None
    chosen_bundle: str | None = None
    model_batch_total_ms = 0.0
    if args.model_bundle:
        chosen_bundle = args.model_bundle
    elif not args.no_model:
        auto = _default_model_bundle_path(settings)
        if auto is not None:
            chosen_bundle = str(auto)

    if chosen_bundle:
        bp = Path(chosen_bundle).expanduser().resolve()
        if not bp.is_file():
            raise FileNotFoundError(f"Model bundle not found: {bp}")
        if not args.model_bundle:
            print(f"[setup] default model bundle     -> {bp}")
        bundle = load_model_bundle(bp)
        t_mb0 = time.perf_counter()
        model_extras = _batch_model_inference(
            df, bundle, compute_shap=not bool(args.no_model_shap)
        )
        mb_ms = _perf_ms(t_mb0, time.perf_counter())
        model_batch_total_ms = mb_ms
        algo = bundle.get("algorithm") or bundle.get("kind") or "unknown"
        tgt = bundle.get("target_column") or "(see bundle)"
        print(
            f"[setup] model batch          {mb_ms:8.2f} ms  file={bp.name}  "
            f"algorithm={algo}  target_column={tgt}  rows_scored={len(model_extras)}  "
            f"compute_shap={not bool(args.no_model_shap)}"
        )
    elif not args.no_model:
        print("[setup] no *.joblib under STORAGE_ROOT/models — using CSV fallback inference")
    print()

    rows: list[RowTiming] = []
    started = time.perf_counter()

    lc = args.label_col or None
    cc = args.confidence_col or None

    for i, rec in enumerate(df.to_dict(orient="records")):
        t_row0 = time.perf_counter()
        print(f"--- row {i} ---")

        # 1) Model bundle predict+proba+SHAP, or CSV fallback "prediction"
        t_inf0 = time.perf_counter()
        if model_extras is not None:
            me = model_extras[i]
            predicted_label = str(me["predicted_label"])
            conf = float(me["max_class_probability"])
            rec_for_xai = {**rec, "shap": me.get("shap")}
            gt_label = _ground_truth_from_csv(rec, bundle, lc)
            infer_detail = f"pred={predicted_label}  conf={conf:.2f}"
            if gt_label is not None:
                infer_detail += f"  csv_true_label={gt_label}"
            infer_tag = "inference (model bundle)"
        else:
            predicted_label, conf = _infer_default(rec, label_col=lc, confidence_col=cc)
            rec_for_xai = rec
            gt_label = _ground_truth_from_csv(rec, None, lc)
            infer_detail = f"label={predicted_label}  conf={conf:.2f}"
            if gt_label is not None:
                infer_detail += f"  csv_label_field={gt_label}"
            infer_tag = "inference (CSV fallback)"
        t_inf1 = time.perf_counter()
        infer_ms = _perf_ms(t_inf0, t_inf1)
        _print_step(infer_tag, infer_ms, infer_detail)

        # 2) Top conditions → one template RAG query → TF-IDF retrieval
        top_conditions, shap_src = _top_conditions_from_row(rec_for_xai, label_col=lc, confidence_col=cc)
        feature_notes = _feature_notes_from_conditions(top_conditions, shap_source=shap_src)
        draft_query = _build_single_template_rag_query(
            predicted_label, conf, top_conditions, shap_source=shap_src
        )

        rag_context: str | None = None
        kb_hits = 0
        rag_ms = 0.0
        if use_rag:
            t_rag0 = time.perf_counter()
            q_final = draft_query.strip()
            rag_context, kb_hits = _maybe_query_kb(
                settings,
                query=q_final,
                prefer_pdf_sources=not bool(args.rag_all_sources),
            )
            t_rag1 = time.perf_counter()
            rag_ms = _perf_ms(t_rag0, t_rag1)
            _print_step(
                "rag / kb retrieval",
                rag_ms,
                f"hits={kb_hits}  template_query_only  q_preview={q_final[:96]!r}",
            )
        else:
            _print_step("rag / kb retrieval", 0.0, "skipped (--no-rag)")

        # 3) Agentic action (single LLM/mock call — same sample_data shape as API orchestration)
        t_agent0 = time.perf_counter()
        probs = model_extras[i].get("class_probabilities") if model_extras else None
        sample_data = _csv_row_to_sample_data(
            i,
            predicted_label,
            conf,
            rec_for_xai,
            top_conditions,
            label_col=lc,
            confidence_col=cc,
            shap_source=shap_src,
            ground_truth_label=gt_label,
            class_probabilities=probs if isinstance(probs, dict) else None,
        )
        decision = asyncio.run(
            _agent_once(
                settings,
                sample_data=sample_data,
                feature_notes=feature_notes,
                rag_context=rag_context,
                attack_actions_data=attack_actions_data,
                agentic_features_data=agentic_features_data,
                use_rag=use_rag,
            )
        )
        t_agent1 = time.perf_counter()
        agent_ms = _perf_ms(t_agent0, t_agent1)
        _print_step("agent_decide (LLM/mock)", agent_ms)

        # 4) Parse plan + compute commitment payload (hash) — then anchor on-chain
        t_payload0 = time.perf_counter()
        structured_plan = _parse_structured_plan(decision.get("raw_llm_response"))
        created_at = _now_utc()
        commitment_sha256, canonical_payload = trust_chain_service.compute_trust_commitment_sha256(
            payload_version=settings.trust_chain_payload_version,
            agentic_report_public_id=f"csv_row_{i}",
            prediction_job_public_id=str(csv_path.name),
            results_row_index=i,
            created_at=created_at,
            raw_llm_response=decision.get("raw_llm_response"),
            rag_context_used=(decision.get("rag_context_used") or rag_context),
            structured_plan=structured_plan,
        )
        t_payload1 = time.perf_counter()
        commitment_ms = _perf_ms(t_payload0, t_payload1)
        _print_step("parse + commitment (sha256)", commitment_ms, f"sha256={_short_hex(commitment_sha256, 16)}")

        t_chain0 = time.perf_counter()
        tx_hash, contract_addr2, agent_key_sha, report_key_sha = trust_chain_service.anchor_report_commitment_on_chain(
            settings=settings,
            agentic_job_public_id="csv_benchmark",
            agentic_report_public_id=f"csv_row_{i}",
            commitment_sha256_hex=commitment_sha256,
        )
        t_chain1 = time.perf_counter()
        chain_ms = _perf_ms(t_chain0, t_chain1)
        _print_step("anchor on-chain (tx)", chain_ms, f"tx={_short_hex(tx_hash, 14)}")

        # 5) Validate: read from chain + recompute from canonical payload
        t_val0 = time.perf_counter()
        rpc_ok, on_chain_hex, err = trust_chain_service.read_commitment_from_chain(
            settings,
            contract_address=contract_addr2,
            agent_key_sha256_hex=agent_key_sha,
            report_key_sha256_hex=report_key_sha,
        )
        chain_valid = bool(rpc_ok and on_chain_hex and on_chain_hex.lower() == commitment_sha256.lower())
        if not rpc_ok and err:
            chain_valid = False

        # Recompute payload hash deterministically (same canonicalization rules)
        canonical_rejson = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        recomputed_sha256 = hashlib.sha256(canonical_rejson.encode("utf-8")).hexdigest()
        payload_valid = recomputed_sha256.lower() == commitment_sha256.lower()
        t_val1 = time.perf_counter()
        val_ms = _perf_ms(t_val0, t_val1)
        _print_step(
            "validate (read + rehash)",
            val_ms,
            f"chain_ok={chain_valid}  payload_ok={payload_valid}" + (f"  rpc_err={err}" if err else ""),
        )

        executed = bool(chain_valid and payload_valid)

        t_row1 = time.perf_counter()
        row_total_ms = _perf_ms(t_row0, t_row1)

        n_df = len(df)
        infer_shap_amortized_ms = (
            (model_batch_total_ms / n_df) if model_extras is not None and n_df > 0 else infer_ms
        )
        pipeline_infer_rag_llm_ms = infer_shap_amortized_ms + rag_ms + agent_ms
        trust_anchor_ms = commitment_ms + chain_ms + val_ms

        rows.append(
            RowTiming(
                row_index=i,
                predicted_label=predicted_label,
                confidence=float(conf),
                kb_hits=int(kb_hits),
                infer_ms=infer_ms,
                infer_shap_amortized_ms=infer_shap_amortized_ms,
                rag_ms=rag_ms,
                commitment_ms=commitment_ms,
                agentic_action_ms=agent_ms,
                blockchain_store_ms=chain_ms,
                validation_ms=val_ms,
                pipeline_infer_rag_llm_ms=pipeline_infer_rag_llm_ms,
                trust_anchor_ms=trust_anchor_ms,
                end_to_end_ms=row_total_ms,
                anchor_tx_hash=tx_hash,
                chain_integrity_valid=bool(chain_valid),
                payload_integrity_valid=bool(payload_valid),
                executed=executed,
            )
        )

        _print_step(
            "pipeline group (infer*+rag+agent)",
            pipeline_infer_rag_llm_ms,
            "infer*=batch infer+SHAP / n_rows when model bundle",
        )
        _print_step("trust anchor (hash+tx+verify)", trust_anchor_ms, "")

        other_ms = max(0.0, row_total_ms - pipeline_infer_rag_llm_ms - trust_anchor_ms)
        if other_ms > 0.05:
            _print_step("row overhead (unaccounted)", other_ms, "parsing / gc / etc.")
        _print_step("row total (wall)", row_total_ms, f"executed={executed}")
        print()

    total_ms = _perf_ms(started, time.perf_counter())

    out_rows = [asdict(r) for r in rows]
    out_df = _reorder_benchmark_df(pd.DataFrame(out_rows))

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = settings.storage_root / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_utc().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"csv_trust_anchor_{csv_path.stem}_{stamp}.csv"

    report_path = out_path.with_name(f"{out_path.stem}_benchmark_report.txt")
    model_bundle_display: str | None = None
    if chosen_bundle:
        model_bundle_display = str(Path(chosen_bundle).expanduser().resolve())

    report_meta = {
        "rows_processed": len(df),
        "model_bundle_display": model_bundle_display,
        "model_algorithm": (bundle.get("algorithm") or bundle.get("kind")) if bundle else None,
        "target_column": bundle.get("target_column") if bundle else None,
        "model_batch_total_ms": model_batch_total_ms,
        "use_rag": use_rag,
        "rag_all_sources": bool(args.rag_all_sources),
        "no_model_shap": bool(args.no_model_shap),
        "csv_out_path": str(out_path),
        "report_out_path": str(report_path),
    }

    report_text = build_benchmark_report_text(
        generated_at=_now_utc(),
        csv_input=csv_path,
        wall_total_ms=total_ms,
        out_df=out_df,
        meta=report_meta,
    )
    print("\n" + report_text)
    report_path.write_text(report_text, encoding="utf-8")

    out_df.to_csv(out_path, index=False)
    print(f"Wrote row timings CSV: {out_path}")
    print(f"Wrote benchmark report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

