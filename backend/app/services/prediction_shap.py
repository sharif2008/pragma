"""SHAP explanations for sklearn tree pipelines used in batch prediction (optional dependency)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum features kept in ``shap.per_feature`` when persisting prediction results (by |SHAP|).
RESULTS_JSON_TOP_SHAP_FEATURES = 10


def limit_shap_per_feature_by_abs(shap_cell: dict[str, Any], top_n: int) -> dict[str, Any]:
    """
    Copy a row-level ``shap`` dict, reducing ``per_feature`` to the top ``top_n`` keys by absolute value.
    Preserves method/status/model metadata and other scalar keys.
    """
    if not isinstance(shap_cell, dict):
        return {}
    if top_n < 1:
        out = {k: v for k, v in shap_cell.items() if k != "per_feature"}
        return out
    out: dict[str, Any] = {k: v for k, v in shap_cell.items() if k != "per_feature"}
    pf = shap_cell.get("per_feature")
    if not isinstance(pf, dict):
        if "per_feature" in shap_cell:
            out["per_feature"] = pf
        return out
    scored: list[tuple[str, float]] = []
    for name, val in pf.items():
        try:
            fv = float(val)
        except (TypeError, ValueError):
            continue
        scored.append((str(name), fv))
    scored.sort(key=lambda x: abs(x[1]), reverse=True)
    out["per_feature"] = {k: v for k, v in scored[:top_n]}
    return out


def compute_sklearn_tree_shap_per_row(pipe: Any, X: pd.DataFrame) -> list[dict[str, Any] | None] | None:
    """
    Return one SHAP payload per row for ``Pipeline([('prep', ...), ('clf', tree)])``.
    Uses TreeExplainer on the tree classifier with **preprocessed** features.
    Returns None if ``shap`` is not installed or the model is not supported.
    """
    try:
        import shap  # type: ignore[import-untyped]
    except ImportError:
        logger.info("shap not installed; skipping TreeExplainer (pip install shap)")
        return None

    clf = pipe.named_steps.get("clf")
    prep = pipe.named_steps.get("prep")
    if clf is None or prep is None:
        return None

    try:
        X_t = prep.transform(X)
    except Exception as e:
        logger.warning("SHAP: preprocess transform failed: %s", e)
        return None

    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_t)
    except Exception as e:
        logger.warning("SHAP: TreeExplainer failed: %s", e)
        return None

    try:
        feature_names = [str(x) for x in prep.get_feature_names_out()]
    except Exception:
        feature_names = [f"f{i}" for i in range(np.asarray(X_t).shape[1])]

    n = len(X)
    pred_proba: np.ndarray | None = None
    try:
        pred_proba = np.asarray(pipe.predict_proba(X))
    except Exception:
        pass

    expected = getattr(explainer, "expected_value", None)
    out: list[dict[str, Any] | None] = []

    def _pack_row(vec: np.ndarray, ev: Any) -> dict[str, Any]:
        arr = np.asarray(vec, dtype=float).ravel()
        m = min(len(feature_names), len(arr))
        per_feature = {feature_names[j]: float(arr[j]) for j in range(m)}
        ev_out: float | list[float] | None
        if ev is None:
            ev_out = None
        elif isinstance(ev, (list, np.ndarray)):
            ev_out = np.asarray(ev, dtype=float).ravel().tolist()
        else:
            ev_out = float(ev)
        return {"method": "treeexplainer", "per_feature": per_feature, "expected_value": ev_out}

    if isinstance(sv, list):
        # Multiclass: list of (n_samples, n_features) arrays
        if pred_proba is None or pred_proba.ndim != 2:
            for i in range(n):
                out.append(_pack_row(sv[0][i], expected[0] if isinstance(expected, list) else expected))
            return out
        pred_idx = np.argmax(pred_proba, axis=1)
        ev_list = expected if isinstance(expected, (list, np.ndarray)) else None
        for i in range(n):
            c = int(pred_idx[i])
            if c < len(sv):
                vec = sv[c][i]
                ev_c = ev_list[c] if ev_list is not None and c < len(ev_list) else None
                out.append(_pack_row(vec, ev_c))
            else:
                out.append(_pack_row(sv[0][i], None))
        return out

    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 1:
        sv_arr = sv_arr.reshape(1, -1)
    for i in range(n):
        out.append(_pack_row(sv_arr[i], expected))
    return out
