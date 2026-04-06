"""Augment RAG templates using a single prediction row + SHAP split by VFL-style agent buckets."""

from __future__ import annotations

import math
from typing import Any

from app.notebook_runtime.vfl_utils import FIXED_AGENT_NAMES, categorize_feature_by_evidence

# Appended to row-level retrieval strings for 5G/6G RAN · Edge · Core triangulation in the knowledge base.
_MOBILE_NETWORK_QUERY_TAIL = (
    " Frame as 5G NR / 5G-Advanced mobile network: RAN (gNB, O-RAN RIC), Edge MEC UPF offload, 5GC Core AMF/SMF/UPF "
    "and optional network slicing."
)


def _agent_bucket_for_shap_key(feat_key: str) -> int:
    raw = feat_key.split("__")[-1] if "__" in feat_key else feat_key
    cat = categorize_feature_by_evidence(str(raw))
    if cat == "evidence_volume_rate":
        return 0
    if cat == "evidence_packet_size":
        return 1
    return 2


def top_shap_features_by_agent(per_feature: dict[str, float], top_n: int = 3) -> dict[str, list[dict[str, Any]]]:
    """Group SHAP values into three agent buckets; keep top ``top_n`` by absolute value per bucket."""
    buckets: dict[int, list[tuple[str, float]]] = {0: [], 1: [], 2: []}
    for k, v in per_feature.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        buckets[_agent_bucket_for_shap_key(str(k))].append((str(k), fv))

    out: dict[str, list[dict[str, Any]]] = {}
    for i, name in enumerate(FIXED_AGENT_NAMES):
        ranked = sorted(buckets[i], key=lambda x: abs(x[1]), reverse=True)[:top_n]
        out[name] = [{"feature": fn, "shap": val} for fn, val in ranked]
    return out


def build_row_agent_templates(
    *,
    job_public_id: str,
    row: dict[str, Any],
    base_summary_line: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Returns extra template dicts (same shape as build_rag_templates_from_summary) and a ``row_context`` blob.
    """
    pred_label = str(row.get("predicted_label") or "unknown")
    flagged = bool(row.get("flagged_attack_or_anomaly"))
    max_p = row.get("max_class_probability")
    shap_obj = row.get("shap") if isinstance(row.get("shap"), dict) else {}
    per_feature = shap_obj.get("per_feature") if isinstance(shap_obj.get("per_feature"), dict) else {}

    agent_feats = top_shap_features_by_agent({str(k): float(v) for k, v in per_feature.items()}, top_n=3)

    lines = []
    queries: list[str] = []
    for aname, feats in agent_feats.items():
        if not feats:
            continue
        bit = ", ".join(f"{f['feature']} (SHAP {f['shap']:+.4f})" for f in feats)
        lines.append(f"{aname}: {bit}")
        queries.append(
            f"Network intrusion investigation for predicted_label={pred_label} with emphasis on {aname} evidence: {bit}. "
            f"Related SOC procedures and feature interpretation.{_MOBILE_NETWORK_QUERY_TAIL}"
        )

    if not queries:
        queries = [
            f"Security analyst guidance for flow classified as {pred_label} (job {job_public_id[:8]}…); "
            f"batch context: {base_summary_line}{_MOBILE_NETWORK_QUERY_TAIL}"
        ]

    summary_query = (
        f"Executive summary retrieval for SOC: single scored row predicted as {pred_label}, "
        f"max_probability={max_p}, flagged={flagged}. "
        f"Top influential features by party: {' | '.join(lines) if lines else 'SHAP not available for this row.'}. "
        f"5G/6G operator context: triage across RAN vs MEC edge vs 5GC core using this evidence."
    )

    row_context: dict[str, Any] = {
        "row_index": row.get("row_index"),
        "predicted_label": pred_label,
        "flagged_attack_or_anomaly": flagged,
        "max_class_probability": max_p,
        "agent_top_shap": agent_feats,
        "shap_method": shap_obj.get("method") or shap_obj.get("status"),
    }

    extra_templates: list[dict[str, Any]] = [
        {
            "id": "row_agent_shap_queries",
            "label": "Per-agent top-3 SHAP → RAG queries",
            "description": "One retrieval string per VFL-style agent bucket using the strongest SHAP features.",
            "retrieval_queries": queries[:6],
            "llm_prompt": (
                f"Prediction job {job_public_id}: row-level decision with label {pred_label}. "
                f"Using retrieved policy and runbook excerpts, explain how the top SHAP drivers per agent party "
                f"(map to RAN / Edge / Core where plausible) should guide analyst triage and what to verify next."
            ),
        },
        {
            "id": "row_summary_rag",
            "label": "Summary-style RAG query (template)",
            "description": "Single fused query summarizing label, flag, and cross-agent SHAP highlights.",
            "retrieval_queries": [summary_query],
            "llm_prompt": (
                f"Row summary: predicted {pred_label}, flagged={flagged}. "
                "Synthesize analyst-facing guidance from the KB using only retrieved text; "
                "when relevant, relate recommendations to 5G/6G RAN, MEC edge, or 5GC core responsibilities."
            ),
        },
    ]
    return extra_templates, row_context
