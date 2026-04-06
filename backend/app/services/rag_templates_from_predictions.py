"""Build multi-query RAG templates from the latest batch prediction summary (no LLM)."""

from __future__ import annotations

from typing import Any


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
    ]
    return templates
