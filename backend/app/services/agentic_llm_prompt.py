"""Build the agentic orchestration LLM prompt (RAG part 2 / notebook-aligned) for API agent/decide."""

from __future__ import annotations

import json
import math
from typing import Any

from app.notebook_runtime.rag_utils import (
    agentic_tiers_dict,
    get_dominant_party_info,
    load_attack_and_agentic,
    tier_allowed_actions,
)
from app.notebook_runtime.vfl_utils import canonical_attack_type, FIXED_AGENT_NAMES
from app.notebook_runtime.storage_paths import AGENTIC_FEATURES_JSON, ATTACK_OPTIONS_JSON
from app.services.network_domains import (
    ACCESS_ISP,
    DOMAIN_INFRA_BLURB,
    DOMAIN_LABELS,
    ENDPOINT_EDR,
    PERIMETER_IDS,
    domain_label,
    format_tier_for_prompt,
    normalize_domain,
    pragma_domain_from_tier_data,
)

# Keys still used in agentic_features.json and SHAP feature_contributions maps.
_STORAGE_BUCKETS: tuple[str, ...] = ("RAN", "Edge", "Core")
_STORAGE_TO_DOMAIN: dict[str, str] = {
    "RAN": ACCESS_ISP,
    "Edge": PERIMETER_IDS,
    "Core": ENDPOINT_EDR,
}
_DOMAIN_TO_STORAGE: dict[str, str] = {v: k for k, v in _STORAGE_TO_DOMAIN.items()}
from app.services.prediction_shap import limit_shap_per_feature_by_abs
from app.services.rag_templates_row_context import top_shap_features_by_agent

_LLM_RAG_SECTIONS_IN_PROMPT = 10
# Top |attribution| features included in orchestration LLM ``sample_data.prediction_row.shap.per_feature``.
LLM_ORCHESTRATION_TOP_SHAP_FEATURES = 5

# Canonical user prompt for POST /agent/decide (and notebook parity). Use .format(...) with:
# sample_id, predicted_label, confidence (float 0–1 for {confidence:.1%}),
# network_tier_info, sample_data (JSON text of full sample_data dict),
# attack_actions_context, agentic_context, rag_context.
AGENTIC_ORCHESTRATION_LLM_USER_PROMPT_TEMPLATE = """You are a cybersecurity decision-making agent specialized in attack response orchestration for an enterprise network.

        Your role is NOT to invent mitigations, but to SELECT and ASSIGN actions from a predefined
        action set using explainability signals and agentic features.

        Network domains (use these exact names everywhere — in JSON and in reasoning):
        - Access / ISP — subscriber/ISP-edge volume and rate telemetry.
        - Perimeter / IDS — north-south inspection, ports, WAF/scan signals.
        - Endpoint / EDR — host-proximate bidirectional and reverse-path forensics.

        The JSON field ``network_tier`` must be exactly one of:
        "Access / ISP" | "Perimeter / IDS" | "Endpoint / EDR"

        =====================
        INPUT CONTEXT
        =====================

        Prediction summary:
        - sample_id: {sample_id}
        - predicted_label: {predicted_label}
        - confidence: {confidence:.1%}

        Explainability & agentic evidence:
        - Domain contributions and dominance:
        {network_tier_info}

        - Feature-level evidence (JSON below). Contains prediction outcome and **top feature attributions only**
          (φ / contribution shares).
        {sample_data}

        Allowed actions (STRICT CONSTRAINT):
        {attack_actions_context}
        • Only actions listed above are allowed.
        • Do NOT invent, rename, generalize, or merge actions.

        Agentic decision signals:
        {agentic_context}

        Retrieved knowledge (RAG – optional support, may be empty):
        {rag_context}

        =====================
        TASK
        =====================

        Using ONLY the information above, generate an agent-ready action plan that:

        1) Interprets how {predicted_label} manifests across Access / ISP, Perimeter / IDS, and Endpoint / EDR.
        2) Identifies which domain MUST trigger mitigation first.
        3) Selects actions ONLY from the provided Allowed actions list.
        4) Assigns each action to the MOST appropriate domain via ``network_tier``.
        5) Provides explicit, evidence-grounded reasoning for EACH action (use the domain names above).
        6) Adapts aggressiveness and execution priority based on confidence ({confidence:.1%}).
        7) If no allowed action is suitable, return empty action lists.

        =====================
        OUTPUT FORMAT (STRICT)
        =====================

        Return a VALID JSON object ONLY:

        {{
          "threat_level": "Critical|High|Medium|Low",

          "all_actions": [
            "action_name_1",
            "action_name_2",
            "action_name_3"
          ],

          "primary_actions": [
            {{
              "action": "EXACT action name from Allowed actions to be taken ",
              "network_tier": "Access / ISP|Perimeter / IDS|Endpoint / EDR",
              "party_evidence_type": "type of evidence this domain observed",
              "reasoning": "clear explanation linking evidence to this action and domain"
            }}
          ],

          "supporting_actions": [
            {{
              "action": "EXACT action name from Allowed actions to be taken",
              "network_tier": "Access / ISP|Perimeter / IDS|Endpoint / EDR",
              "party_evidence_type": "type of evidence this domain observed",
              "reasoning": "why this action supports the primary"
            }}
          ],
          "overall_reasoning": "Concise summary of Access/ISP vs Perimeter/IDS vs Endpoint/EDR prioritization",
          "execution_priority": "Immediate|High|Standard|Low",
          "knowledge_sources_used": [
            "allowed_actions_context",
            "attack_actions_context"
          ]
        }}

        =====================
        HARD RULES (DO NOT VIOLATE)
        =====================

        - Do NOT output text outside the JSON.
        - Do NOT generate actions not listed in Allowed actions.
        - The "all_actions" list MUST be the union of primary_actions and supporting_actions.
        - Do NOT alter action names.
        - ``network_tier`` MUST be exactly "Access / ISP", "Perimeter / IDS", or "Endpoint / EDR".
        - Every action MUST include explicit reasoning tied to evidence or agentic rules.
        - Prefer the dominant domain for primary actions unless contradicted by evidence.
        - If RAG context is empty, rely ONLY on explainability and agentic context.
        - Describe the environment as an enterprise Access / Perimeter / Endpoint network only.
        """

def _relabel_tiers_for_llm_json(obj: Any) -> Any:
    """Rewrite storage/SHAP bucket keys in prompt JSON to enterprise domain labels."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            key = str(k)
            if key in ("dominant_agent", "dominant_tier") and isinstance(v, str):
                out[key] = domain_label(v)
                continue
            nk = _STORAGE_TO_DOMAIN.get(key) or normalize_domain(key) or key
            out[str(nk)] = _relabel_tiers_for_llm_json(v)
        return out
    if isinstance(obj, list):
        return [_relabel_tiers_for_llm_json(x) for x in obj]
    return obj


# VFL agent index → storage bucket (agentic_features / SHAP), then mapped to domain for prompts.
_TIER_BY_AGENT_INDEX = _STORAGE_BUCKETS


def _top_features_for_tier(
    sample: dict[str, Any],
    tier: str,
    *,
    top_n: int = 3,
    score_key: str = "pct_contribution",
) -> list[str]:
    shap_expl = sample.get("shap_explanation", {}) or {}
    feat_contribs = shap_expl.get("feature_contributions", {}) or {}
    feats = feat_contribs.get(tier, {}) or {}
    if not feats:
        alt = _STORAGE_TO_DOMAIN.get(tier) or _DOMAIN_TO_STORAGE.get(tier) or normalize_domain(tier)
        if alt:
            feats = feat_contribs.get(alt, {}) or {}

    scored: list[tuple[str, float]] = []
    if isinstance(feats, dict):
        for feat_name, meta in feats.items():
            if not isinstance(meta, dict):
                continue
            score = float(meta.get(score_key, 0.0) or 0.0)
            if score > 0.0:
                scored.append((feat_name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_n]]


def extract_sample_summary(sample: dict[str, Any]) -> dict[str, Any]:
    raw = sample.get("predicted_label") or sample.get("true_label") or "UNKNOWN"
    label = canonical_attack_type(str(raw).strip()) if str(raw).strip() else "UNKNOWN"
    confidence = float(sample.get("confidence", 0.0) or 0.0)

    shap_expl = sample.get("shap_explanation", {}) or {}
    dominant_agent = (shap_expl.get("dominant_agent") or "").strip()
    dominant_tier = normalize_domain(dominant_agent) or "Unknown"
    dominant_pct = float(shap_expl.get("dominant_contribution_pct", 0.0) or 0.0) * 100.0

    top_features = {
        _STORAGE_TO_DOMAIN[bucket]: _top_features_for_tier(sample, bucket, top_n=3)
        for bucket in _STORAGE_BUCKETS
    }

    return {
        "label": label,
        "confidence": confidence,
        "dominant_tier": dominant_tier,
        "dominant_pct": dominant_pct,
        "top_features": top_features,
    }


def _safe_per_feature(row: dict[str, Any]) -> dict[str, float]:
    shap_obj = row.get("shap") if isinstance(row.get("shap"), dict) else {}
    pf = shap_obj.get("per_feature")
    if not isinstance(pf, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in pf.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out[str(k)] = fv
    return out


def row_to_shap_explanation(row: dict[str, Any]) -> dict[str, Any]:
    per_feature = _safe_per_feature(row)
    if not per_feature:
        return {}
    agent_feats = top_shap_features_by_agent(per_feature, top_n=3)
    feature_contributions: dict[str, dict[str, dict[str, float]]] = {
        d: {} for d in DOMAIN_LABELS
    }
    tier_sums: list[tuple[str, float]] = []
    for i, name in enumerate(FIXED_AGENT_NAMES):
        bucket = _TIER_BY_AGENT_INDEX[i] if i < len(_TIER_BY_AGENT_INDEX) else "Core"
        domain = _STORAGE_TO_DOMAIN.get(bucket, ENDPOINT_EDR)
        feats = agent_feats.get(name, [])
        s = sum(abs(float(f.get("shap", 0) or 0)) for f in feats)
        tier_sums.append((domain, s))
        for f in feats:
            fn = str(f.get("feature", ""))
            if fn:
                feature_contributions[domain][fn] = {"pct_contribution": abs(float(f.get("shap", 0) or 0))}
    total_abs = sum(abs(v) for v in per_feature.values()) or 1.0
    dominant_tier = max(tier_sums, key=lambda x: x[1])[0] if tier_sums else ACCESS_ISP
    max_sum = max((x[1] for x in tier_sums), default=0.0)
    dominant_pct = min(1.0, max_sum / total_abs) if total_abs else 0.0
    return {
        "dominant_agent": dominant_tier,
        "dominant_contribution_pct": dominant_pct,
        "feature_contributions": feature_contributions,
    }


def _shap_cell_for_prompt(shap_cell: Any, *, top_per_feature: int = LLM_ORCHESTRATION_TOP_SHAP_FEATURES) -> dict[str, Any]:
    """Copy SHAP cell for the LLM: all scalar keys, ``per_feature`` reduced to top-N by |value|."""
    if not isinstance(shap_cell, dict):
        return {}
    return limit_shap_per_feature_by_abs(shap_cell, top_per_feature)


def _prediction_row_for_prompt(
    row: dict[str, Any],
    *,
    top_per_feature: int = LLM_ORCHESTRATION_TOP_SHAP_FEATURES,
) -> dict[str, Any]:
    pr = dict(row)
    shp = pr.get("shap")
    if isinstance(shp, dict):
        pr["shap"] = _shap_cell_for_prompt(shp, top_per_feature=top_per_feature)
    return pr


def _select_prediction_row(
    rows: list[Any],
    *,
    results_row_index: int | None,
) -> tuple[dict[str, Any] | None, str]:
    """
    Pick one row from ``results_json.rows``.

    If ``results_row_index`` is set, match ``row["row_index"]`` or use that list index.
    Otherwise: first ``flagged_attack_or_anomaly``, else first row (often BENIGN in index order).
    """
    if not isinstance(rows, list) or not rows:
        return None, "no_rows"
    if results_row_index is not None:
        try:
            ri = int(results_row_index)
        except (TypeError, ValueError):
            ri = -1
        if ri >= 0:
            for r in rows:
                if isinstance(r, dict) and r.get("row_index") == ri:
                    return r, "explicit_row_index"
            if ri < len(rows) and isinstance(rows[ri], dict):
                return rows[ri], "positional_row_index"
        # Fall through to default if explicit index invalid
    flagged = [r for r in rows if isinstance(r, dict) and r.get("flagged_attack_or_anomaly")]
    if flagged:
        return flagged[0], "first_flagged"
    first = rows[0]
    if isinstance(first, dict):
        return first, "first_row_fallback"
    return None, "invalid_row_type"


def build_sample_from_prediction_job(
    job: Any,
    summary: dict[str, Any],
    *,
    results_row_index: int | None = None,
) -> dict[str, Any]:
    """
    Map one prediction row to notebook-style ``sample_data`` for the orchestration prompt.

    Pass ``results_row_index`` from RAG prep so the agent sees the same row the analyst chose
    (label + SHAP), not only the first row of the batch.
    """
    rj = getattr(job, "results_json", None)
    rows = rj.get("rows") if isinstance(rj, dict) else None
    row_list: list[Any] = rows if isinstance(rows, list) else []
    row, how = _select_prediction_row(row_list, results_row_index=results_row_index)
    if row is not None:
        try:
            row_idx = int(row.get("row_index", row_list.index(row) if row in row_list else 0))
        except (TypeError, ValueError):
            row_idx = 0
        conf = row.get("max_class_probability")
        try:
            confidence = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        shap_expl = row_to_shap_explanation(row)
        prediction_row = _prediction_row_for_prompt(row)
        return {
            "sample_id": row_idx,
            "predicted_label": canonical_attack_type(str(row.get("predicted_label") or "UNKNOWN")),
            "confidence": confidence,
            "shap_explanation": shap_expl,
            "prediction_row": prediction_row,
            "row_selection": how,
            "results_row_index_requested": results_row_index,
        }
    s0 = summary.get("sample_row_0") if isinstance(summary.get("sample_row_0"), dict) else {}
    try:
        confidence = float(s0.get("max_class_probability") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "sample_id": 0,
        "predicted_label": canonical_attack_type(str(s0.get("predicted_label") or "UNKNOWN")),
        "confidence": confidence,
        "shap_explanation": {},
        "prediction_row": None,
        "row_selection": "summary_sample_row_0",
        "results_row_index_requested": results_row_index,
    }


def rag_context_string_to_rag_results(rag_context: str | None, *, max_sections: int = _LLM_RAG_SECTIONS_IN_PROMPT) -> list[dict[str, str]]:
    if not rag_context or not str(rag_context).strip():
        return []
    text = str(rag_context).strip()
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    if not chunks:
        return [{"title": "Retrieved knowledge", "text": text[:12000]}]
    out: list[dict[str, str]] = []
    for i, ch in enumerate(chunks[:max_sections]):
        out.append({"title": f"Excerpt {i + 1}", "text": ch[:8000]})
    return out


def create_agentic_orchestration_prompt(
    sample_data: dict[str, Any],
    rag_results: list[dict[str, Any]],
    attack_actions_data: dict[str, Any] | None,
    agentic_features_data: dict[str, Any] | None,
    *,
    include_knowledge_base: bool = True,
    extra_agentic_notes: str | None = None,
) -> str:
    """Same contract as notebook ``create_prompt`` (orchestration JSON plan)."""
    # Keep Prediction summary lines in sync with ``prediction_row`` (same row as the JSON block below).
    pr = sample_data.get("prediction_row") if isinstance(sample_data.get("prediction_row"), dict) else None
    if pr is not None:
        try:
            sample_id = int(pr.get("row_index", sample_data.get("sample_id", 0)))
        except (TypeError, ValueError):
            sample_id = int(sample_data.get("sample_id", 0) or 0)
        predicted_label = canonical_attack_type(str(pr.get("predicted_label") or sample_data.get("predicted_label") or "UNKNOWN"))
        try:
            confidence = float(pr.get("max_class_probability", sample_data.get("confidence", 0.0)) or 0.0)
        except (TypeError, ValueError):
            confidence = float(sample_data.get("confidence", 0.0) or 0.0)
    else:
        try:
            sample_id = int(sample_data.get("sample_id", 0) or 0)
        except (TypeError, ValueError):
            sample_id = 0
        predicted_label = canonical_attack_type(str(sample_data.get("predicted_label") or "UNKNOWN"))
        confidence = float(sample_data.get("confidence", 0.0) or 0.0)
    dominant_party, dominant_tier, dominant_pct = get_dominant_party_info(sample_data)

    s = extract_sample_summary(sample_data)
    condensed_evidence = {
        "predicted_label": str(s.get("label") or predicted_label),
        "confidence": float(s.get("confidence", confidence) or 0.0),
        "dominant_tier": s.get("dominant_tier", dominant_tier),
        "dominant_contribution_pct": float(s.get("dominant_pct", dominant_pct) or 0.0),
        "top_features_by_tier": s.get("top_features", {}),
    }

    attack_actions_context = ""
    if attack_actions_data and "attacks" in attack_actions_data:
        attack_type = predicted_label
        attacks = attack_actions_data.get("attacks") or {}
        if isinstance(attacks, dict) and attack_type in attacks:
            recommended_actions = attacks[attack_type]
            if isinstance(recommended_actions, list):
                attack_actions_context = (
                    f"\n\nAttack-Specific Recommended Actions (from attack_options.json):\n"
                    f"For {predicted_label} attack, recommended actions: {', '.join(str(x) for x in recommended_actions)}\n"
                )
            else:
                attack_actions_context = f"\n\nAttack-Specific Recommended Actions:\n{recommended_actions}\n"
        else:
            attack_actions_context = (
                f"\n\nAttack-Specific Actions: No specific recommendations for {predicted_label}.\n"
            )
        primary_domains = (attack_actions_data.get("primary_domains") or {}) if isinstance(attack_actions_data, dict) else {}
        if isinstance(primary_domains, dict) and attack_type in primary_domains:
            domains = primary_domains[attack_type]
            if isinstance(domains, list) and domains:
                attack_actions_context += (
                    f"Preferred response domains for this label: {', '.join(str(d) for d in domains)}.\n"
                )
        evidence_cues = (attack_actions_data.get("evidence_cues") or {}) if isinstance(attack_actions_data, dict) else {}
        if isinstance(evidence_cues, dict) and attack_type in evidence_cues:
            cues = evidence_cues[attack_type]
            if isinstance(cues, list) and cues:
                attack_actions_context += (
                    f"Typical network-flow evidence cues: {', '.join(str(c) for c in cues)}.\n"
                )

    agentic_context = ""
    tiers = agentic_tiers_dict(agentic_features_data)
    if tiers:
        agentic_context = (
            "\n\nAgentic Features and Actions by Network Domain (from agentic_features.json):\n"
            f"{DOMAIN_INFRA_BLURB}\n"
            "Use the condensed evidence (top features by domain) to GATE which domain gets priority actions.\n"
        )
        tf = condensed_evidence.get("top_features_by_tier", {}) or {}
        for bucket in _STORAGE_BUCKETS:
            if bucket not in tiers:
                continue
            tier_data = tiers[bucket]
            if not isinstance(tier_data, dict):
                continue
            actions = tier_allowed_actions(tier_data)
            domain = pragma_domain_from_tier_data(bucket, tier_data)
            tier_feats: list[Any] = []
            if isinstance(tf, dict):
                raw_feats = tf.get(domain) or tf.get(bucket) or []
                tier_feats = list(raw_feats) if isinstance(raw_feats, list) else []
            agentic_context += f"\n{domain}:\n"
            desc = tier_data.get("description")
            if isinstance(desc, str) and desc.strip():
                agentic_context += f"  - Role: {desc.strip()}\n"
            agentic_context += f"  - Top evidence features (top 3): {', '.join(str(x) for x in tier_feats) if tier_feats else 'none'}\n"
            agentic_context += f"  - Allowed domain actions: {', '.join(actions)}\n"

    if extra_agentic_notes and str(extra_agentic_notes).strip():
        agentic_context += (
            "\n\nAnalyst emphasis (action preset only — RAG prep narrative is not injected here):\n"
            f"{str(extra_agentic_notes).strip()}\n"
        )

    if include_knowledge_base:
        top_results = rag_results[:_LLM_RAG_SECTIONS_IN_PROMPT]
        if top_results:
            rag_context = "\n\nKnowledge Base Context (from RAG search):\n"
            for idx, result in enumerate(top_results, 1):
                title = result.get("title", "Unknown")
                body = result.get("text") or result.get("chunk_text") or ""
                rag_context += f"\n[{idx}] {title}\n"
                rag_context += f"{body}\n"
        else:
            rag_context = "\n\nKnowledge Base: No relevant documents found from RAG search."
    else:
        rag_context = "\n\nKnowledge base was not included in this request.\n"

    network_tier_info = ""
    if dominant_tier or dominant_party:
        dom = format_tier_for_prompt(str(dominant_tier or dominant_party))
        network_tier_info = (
            f"\n- Dominant network domain: {dom} "
            f"(contribution: {dominant_pct:.1f}%)"
        )

    slim = sample_data.get("orchestration_llm_payload")
    if isinstance(slim, dict) and slim:
        sample_data_json = json.dumps(
            _relabel_tiers_for_llm_json(slim), indent=2, ensure_ascii=False, default=str
        )
    else:
        sample_data_json = json.dumps(
            _relabel_tiers_for_llm_json(sample_data), indent=2, ensure_ascii=False, default=str
        )

    return AGENTIC_ORCHESTRATION_LLM_USER_PROMPT_TEMPLATE.format(
        sample_id=sample_id,
        predicted_label=predicted_label,
        confidence=confidence,
        network_tier_info=network_tier_info,
        sample_data=sample_data_json,
        attack_actions_context=attack_actions_context,
        agentic_context=agentic_context,
        rag_context=rag_context,
    )


def build_agentic_decide_user_prompt(
    sample_data: dict[str, Any],
    rag_context: str | None,
    attack_actions_data: dict[str, Any] | None,
    agentic_features_data: dict[str, Any] | None,
    *,
    include_knowledge_base: bool,
    feature_notes: str | None,
) -> str:
    """Full user message for orchestration JSON (same string POST /agent/decide sends to the LLM)."""
    rag_results = rag_context_string_to_rag_results(rag_context) if include_knowledge_base else []
    return create_agentic_orchestration_prompt(
        sample_data,
        rag_results,
        attack_actions_data,
        agentic_features_data,
        include_knowledge_base=include_knowledge_base,
        extra_agentic_notes=feature_notes,
    )


def load_attack_agentic_config(*, verbose: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    return load_attack_and_agentic(ATTACK_OPTIONS_JSON, AGENTIC_FEATURES_JSON, verbose=verbose)


def summarize_plan_for_db(parsed: dict[str, Any], raw_fallback: str) -> tuple[str, str]:
    """Derive (summary text, recommended_action) for AgenticReport columns."""
    overall = str(parsed.get("overall_reasoning") or "").strip()
    threat = str(parsed.get("threat_level") or "")
    pri = str(parsed.get("execution_priority") or "")
    summary_parts = [p for p in (threat, pri, overall) if p]
    summary = " · ".join(summary_parts) if summary_parts else raw_fallback[:2000]

    rec = ""
    prim = parsed.get("primary_actions")
    if isinstance(prim, list) and prim:
        first = prim[0] if isinstance(prim[0], dict) else None
        if isinstance(first, dict):
            rec = str(first.get("action") or "").strip()
    if not rec:
        aa = parsed.get("all_actions")
        if isinstance(aa, list) and aa:
            rec = str(aa[0]).strip()
    if not rec:
        rec = str(parsed.get("execution_priority") or "monitor")[:512]
    return summary[:8000], rec[:512]
