"""RAG retrieval templates: row-level SHAP queries and batch-summary query sets."""

from __future__ import annotations

import math
from typing import Any

from scripts.vfl import FIXED_AGENT_NAMES, categorize_feature_by_evidence
from scripts.network_domains import (
    ACCESS_ISP,
    DOMAIN_INFRA_BLURB,
    ENDPOINT_EDR,
    PERIMETER_IDS,
)

# Appended to row-level retrieval strings for enterprise Access / Perimeter / Endpoint triage.
_ENTERPRISE_NETWORK_QUERY_TAIL = (
    f" Frame as enterprise network intrusion response across {DOMAIN_INFRA_BLURB}"
)

_AGENT_NAME_TO_DOMAIN: dict[str, str] = {
    FIXED_AGENT_NAMES[0]: ACCESS_ISP,
    FIXED_AGENT_NAMES[1]: PERIMETER_IDS,
    FIXED_AGENT_NAMES[2]: ENDPOINT_EDR,
}


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


# Single-query RAG template filled with prediction + SHAP domain contributions (agent decide / pipeline).
STATIC_RAG_RETRIEVAL_TEMPLATE = (
    "Enterprise cybersecurity incident response and KB retrieval for network flow classification. "
    "predicted_label={predicted_label}; confidence={confidence}; row_flagged={flagged}. "
    "Batch context: {rows_flagged} flagged of {rows_total} scored rows. "
    "SHAP contributions — Access/ISP: {shap_access}; Perimeter/IDS: {shap_perimeter}; "
    "Endpoint/EDR: {shap_endpoint}. "
    "Find SOC runbooks, NIST CIS MITRE ATT&CK controls, IDS perimeter and EDR containment guidance."
)


def _shap_domain_summary(agent_feats: dict[str, list[dict[str, Any]]], agent_name: str) -> str:
    feats = agent_feats.get(agent_name) or []
    if not feats:
        return "none"
    parts = []
    for f in feats[:3]:
        fn = str(f.get("feature") or "").strip()
        if not fn:
            continue
        try:
            sv = float(f.get("shap") or 0.0)
            parts.append(f"{fn} (SHAP {sv:+.4f})")
        except (TypeError, ValueError):
            parts.append(fn)
    return ", ".join(parts) if parts else "none"


def build_templated_rag_retrieval_query(
    summary: dict[str, Any],
    row: dict[str, Any] | None = None,
) -> str:
    """
    One static template string with batch prediction stats and optional row label + SHAP contributions.
    Used for agent-decide RAG retrieval (single FAISS query, not multi-query fusion).
    """
    total = int(summary.get("rows_total") or 0)
    flagged_batch = int(summary.get("rows_flagged") or 0)

    if row and isinstance(row, dict):
        pred_label = str(row.get("predicted_label") or "unknown").strip() or "unknown"
        flagged = bool(row.get("flagged_attack_or_anomaly"))
        max_p = row.get("max_class_probability")
        try:
            confidence = f"{float(max_p):.4f}" if max_p is not None else "n/a"
        except (TypeError, ValueError):
            confidence = "n/a"
        shap_obj = row.get("shap") if isinstance(row.get("shap"), dict) else {}
        per_feature = shap_obj.get("per_feature") if isinstance(shap_obj.get("per_feature"), dict) else {}
        agent_feats = top_shap_features_by_agent(
            {str(k): float(v) for k, v in per_feature.items()},
            top_n=3,
        )
    else:
        head = summary.get("head_json") or []
        labels: list[str] = []
        for r in head[:8]:
            if isinstance(r, dict) and r.get("predicted_label") is not None:
                labels.append(str(r["predicted_label"]))
        pred_label = ", ".join(sorted(set(labels))[:5]) if labels else "batch_aggregate"
        flagged = flagged_batch > 0
        confidence = "n/a"
        agent_feats = {name: [] for name in FIXED_AGENT_NAMES}

    q = STATIC_RAG_RETRIEVAL_TEMPLATE.format(
        predicted_label=pred_label,
        confidence=confidence,
        flagged=flagged,
        rows_flagged=flagged_batch,
        rows_total=total,
        shap_access=_shap_domain_summary(agent_feats, FIXED_AGENT_NAMES[0]),
        shap_perimeter=_shap_domain_summary(agent_feats, FIXED_AGENT_NAMES[1]),
        shap_endpoint=_shap_domain_summary(agent_feats, FIXED_AGENT_NAMES[2]),
    )
    return q.strip()[:2000]


def build_row_retrieval_queries(
    *,
    row: dict[str, Any],
    job_public_id: str = "",
    max_queries: int = 6,
) -> list[str]:
    """
    Compact row-aware FAISS queries: predicted label + top SHAP + domain.
    Prefer short topical strings over long batch-stat prose.
    """
    pred_label = str(row.get("predicted_label") or "unknown").strip() or "unknown"
    flagged = bool(row.get("flagged_attack_or_anomaly"))
    shap_obj = row.get("shap") if isinstance(row.get("shap"), dict) else {}
    per_feature = shap_obj.get("per_feature") if isinstance(shap_obj.get("per_feature"), dict) else {}
    agent_feats = top_shap_features_by_agent(
        {str(k): float(v) for k, v in (per_feature or {}).items()},
        top_n=3,
    )

    queries: list[str] = []
    # Label-focused playbook / policy match.
    queries.append(
        f"{pred_label} network intrusion response containment monitoring SOC runbook"
    )
    if flagged:
        queries.append(
            f"{pred_label} attack mitigation rate limiting ACL blocking IDS EDR playbook"
        )
    else:
        queries.append(
            f"{pred_label} false positive review baseline verification network flow policy"
        )

    for aname, feats in agent_feats.items():
        if not feats:
            continue
        domain = _AGENT_NAME_TO_DOMAIN.get(aname, aname)
        feat_names = ", ".join(str(f.get("feature") or "") for f in feats[:3] if f.get("feature"))
        if not feat_names:
            continue
        queries.append(
            f"{pred_label} {domain} evidence features {feat_names} security controls investigation"
        )

    # Dedupe while preserving order; keep short.
    out: list[str] = []
    seen: set[str] = set()
    for q in queries:
        key = q.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(q.strip()[:500])
        if len(out) >= max_queries:
            break

    if not out:
        jid = (job_public_id or "")[:8]
        out = [f"security analyst guidance for flow classified as {pred_label} job {jid}"]
    return out


def resolve_prediction_row(
    rows: list[Any] | None,
    results_row_index: int | None,
) -> dict[str, Any] | None:
    """Match ``results_row_index`` to a results_json row (by field or list index)."""
    if not isinstance(rows, list) or not rows or results_row_index is None:
        return None
    try:
        ri = int(results_row_index)
    except (TypeError, ValueError):
        return None
    for r in rows:
        if isinstance(r, dict) and r.get("row_index") == ri:
            return r
    if 0 <= ri < len(rows) and isinstance(rows[ri], dict):
        return rows[ri]
    return None


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
        domain = _AGENT_NAME_TO_DOMAIN.get(aname, aname)
        bit = ", ".join(f"{f['feature']} (SHAP {f['shap']:+.4f})" for f in feats)
        lines.append(f"{domain}: {bit}")
        queries.append(
            f"Network intrusion investigation for predicted_label={pred_label} with emphasis on "
            f"{domain} evidence: {bit}. "
            f"Related SOC procedures and feature interpretation.{_ENTERPRISE_NETWORK_QUERY_TAIL}"
        )

    if not queries:
        queries = [
            f"Security analyst guidance for flow classified as {pred_label} (job {job_public_id[:8]}…); "
            f"batch context: {base_summary_line}{_ENTERPRISE_NETWORK_QUERY_TAIL}"
        ]

    summary_query = (
        f"Executive summary retrieval for SOC: single scored row predicted as {pred_label}, "
        f"max_probability={max_p}, flagged={flagged}. "
        f"Top influential features by domain: {' | '.join(lines) if lines else 'SHAP not available for this row.'}. "
        f"Enterprise context: triage across Access / ISP vs Perimeter / IDS vs Endpoint / EDR using this evidence."
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
            "label": "Per-domain top-3 SHAP → RAG queries",
            "description": (
                "One retrieval string per VFL domain agent (Access/ISP, Perimeter/IDS, Endpoint/EDR) "
                "using the strongest SHAP features."
            ),
            "retrieval_queries": queries[:6],
            "llm_prompt": (
                f"Prediction job {job_public_id}: row-level decision with label {pred_label}. "
                f"Using retrieved policy and runbook excerpts, explain how the top SHAP drivers per domain "
                f"(Access / ISP, Perimeter / IDS, Endpoint / EDR) "
                f"should guide analyst triage and what to verify next."
            ),
        },
        {
            "id": "row_summary_rag",
            "label": "Summary-style RAG query (template)",
            "description": "Single fused query summarizing label, flag, and cross-domain SHAP highlights.",
            "retrieval_queries": [summary_query],
            "llm_prompt": (
                f"Row summary: predicted {pred_label}, flagged={flagged}. "
                "Synthesize analyst-facing guidance from the KB using only retrieved text; "
                "when relevant, relate recommendations to Access / ISP, Perimeter / IDS, or Endpoint / EDR responsibilities."
            ),
        },
    ]
    return extra_templates, row_context

def build_rag_templates_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Each template supplies several retrieval queries (multi-query fusion + MMR on the backend)
    and one LLM synthesis prompt aligned with the batch.
    """
    total = int(summary.get("rows_total") or 0)
    flagged = int(summary.get("rows_flagged") or 0)
    ratio = (flagged / total) if total else 0.0
    job_id = summary.get("prediction_job_public_id") or "unknown"

    head = summary.get("head_json") or []
    labels: list[str] = []
    for row in head[:8]:
        if isinstance(row, dict) and row.get("predicted_label") is not None:
            labels.append(str(row["predicted_label"]))
    uniq_labels = sorted(set(labels))
    label_hint = ", ".join(uniq_labels[:5]) if uniq_labels else "unknown labels"

    ctx = f"{flagged} flagged rows out of {total} total ({ratio:.0%} flagged fraction). Sample predicted labels: {label_hint}."

    templates: list[dict[str, Any]] = [
        {
            "id": "soc_overview",
            "label": "SOC incident context",
            "description": "Broad retrieval for security operations context matching this batch profile.",
            "retrieval_queries": [
                f"security operations center procedures for network traffic batch scoring with {ctx}",
                "enterprise SOC runbook for reviewing IDS or ML prediction outputs and anomaly flags",
                "incident triage checklist when automated classifiers flag a subset of flows",
            ],
            "llm_prompt": (
                f"Batch prediction job {job_id}: {ctx} "
                "Summarize what analysts should verify first and which policies apply, using only the retrieved excerpts."
            ),
        },
        {
            "id": "policy_anomaly",
            "label": "Policy & anomaly handling",
            "description": "Documents on anomaly thresholds, escalation, and acceptable use.",
            "retrieval_queries": [
                f"organizational policy for anomaly detection alerts when {ratio:.0%} of rows are flagged",
                "network security policy escalation paths for probabilistic classifier outputs",
                "threshold-based response guidelines for max_class_probability style scores",
            ],
            "llm_prompt": (
                f"Given this batch ({ctx}), what policy sections govern threshold tuning, escalation, and false-positive review?"
            ),
        },
        {
            "id": "remediation",
            "label": "Remediation & containment",
            "description": "Containment, blocking, and monitoring aligned with flagged traffic.",
            "retrieval_queries": [
                "remediation steps for suspected malicious or anomalous network sessions after batch scoring",
                "containment and monitoring playbook when prediction outputs suggest attack classes",
                "safe rollback and verification after applying blocks based on classifier decisions",
            ],
            "llm_prompt": (
                f"For predictions where labels include: {label_hint}. "
                "Recommend concrete containment and monitoring steps grounded in the retrieved runbooks."
            ),
        },
        {
            "id": "label_focus",
            "label": "Label-focused investigation",
            "description": "Tighter queries around observed predicted_label values.",
            "retrieval_queries": [
                f"investigation guidance for traffic classified as: {label_hint}",
                "feature review and PCAP collection when predicted_label distribution shifts in batch scoring",
                "differentiating benign versus attack classes in flow-based ML outputs",
            ],
            "llm_prompt": (
                f"The latest batch shows these predicted labels in samples: {label_hint}. "
                "What investigative questions and evidence should analysts prioritize?"
            ),
        },
        {
            "id": "enterprise_access_perimeter_endpoint",
            "label": "Enterprise network (Access · Perimeter · Endpoint)",
            "description": (
                "PRAGMA-style domain retrieval: Access / ISP volume-rate signals, Perimeter / IDS inspection, "
                "and Endpoint / EDR host-proximate forensics—aligned with batch scoring context."
            ),
            "retrieval_queries": [
                (
                    f"Access / ISP edge security: rate limiting, scrubbing, ACL, and flood controls when batch scoring shows {ctx}"
                ),
                (
                    "Perimeter / IDS security operations: WAF, port hardening, scan tarpitting, DNS/HTTP exposure review "
                    "for north-south traffic anomalies"
                ),
                (
                    "Endpoint / EDR response: account lockout, MFA enforcement, service isolation, and host-proximate "
                    "forensics when classifier outputs suggest endpoint-impacting abuse"
                ),
                (
                    f"Cross-domain incident response for Access / ISP vs Perimeter / IDS vs Endpoint / EDR when "
                    f"multi-party (VFL-style) feature splits disagree; labels include {label_hint}"
                ),
                (
                    "Network-flow feature interpretation for SOC: packet counts, PIAT, TCP flags, unique ports, "
                    "and bidirectional vs src2dst evidence in enterprise intrusion detection"
                ),
            ],
            "llm_prompt": (
                f"Batch job {job_id}: {ctx} "
                "You are briefing SOC analysts in an enterprise network context (not 5G telecom). Using only retrieved passages, "
                "map findings to Access / ISP vs Perimeter / IDS vs Endpoint / EDR responsibilities, "
                "and list concrete verification steps (SIEM fields, IDS alerts, EDR host logs)."
            ),
        },
    ]
    return templates
