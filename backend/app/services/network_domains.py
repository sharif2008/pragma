"""PRAGMA enterprise network domains (Access / Perimeter / Endpoint).

Canonical ``network_tier`` / prompt labels — no telecom RAN / Edge / Core naming.
"""

from __future__ import annotations

from typing import Any

ACCESS_ISP = "Access / ISP"
PERIMETER_IDS = "Perimeter / IDS"
ENDPOINT_EDR = "Endpoint / EDR"

# Canonical domain labels for LLM JSON ``network_tier`` and UI.
DOMAIN_LABELS: tuple[str, ...] = (ACCESS_ISP, PERIMETER_IDS, ENDPOINT_EDR)

DOMAIN_TO_AGENT: dict[str, str] = {
    ACCESS_ISP: "D1",
    PERIMETER_IDS: "D2",
    ENDPOINT_EDR: "D3",
}

AGENT_TO_DOMAIN: dict[str, str] = {v: k for k, v in DOMAIN_TO_AGENT.items()}

# Short blurb for prompts / RAG framing.
DOMAIN_INFRA_BLURB = (
    "Enterprise network domains: Access / ISP (subscriber edge volume/rate), "
    "Perimeter / IDS (north-south inspection, ports, WAF/scan signals), "
    "Endpoint / EDR (host-proximate bidirectional and reverse-path forensics)."
)


def normalize_domain(raw: str) -> str | None:
    """Map any known alias to a canonical domain label. None if unknown."""
    s = str(raw or "").strip()
    if not s:
        return None
    if s in DOMAIN_LABELS:
        return s
    if s in AGENT_TO_DOMAIN:
        return AGENT_TO_DOMAIN[s]
    compact = s.lower().replace(" ", "")
    aliases = {
        "access/isp": ACCESS_ISP,
        "accessisp": ACCESS_ISP,
        "d1": ACCESS_ISP,
        "perimeter/ids": PERIMETER_IDS,
        "perimeterids": PERIMETER_IDS,
        "d2": PERIMETER_IDS,
        "endpoint/edr": ENDPOINT_EDR,
        "endpointedr": ENDPOINT_EDR,
        "d3": ENDPOINT_EDR,
        # Legacy storage / SHAP bucket names → domains (input only).
        "ran": ACCESS_ISP,
        "edge": PERIMETER_IDS,
        "core": ENDPOINT_EDR,
    }
    return aliases.get(compact)


def domain_label(raw: str) -> str:
    """Return canonical domain label, or the original string if not recognized."""
    return normalize_domain(raw) or str(raw or "").strip()


def format_tier_for_prompt(raw: str) -> str:
    """Human-readable domain for prompts."""
    return domain_label(raw)


def domain_label_for_display(raw: str) -> str:
    """Always prefer a canonical domain label when mappable."""
    return domain_label(raw)


def rewrite_plan_network_tiers(plan: Any) -> Any:
    """Force structured-plan ``network_tier`` values to enterprise domain labels."""
    if not isinstance(plan, dict):
        return plan
    out = dict(plan)
    for key in ("primary_actions", "supporting_actions"):
        block = out.get(key)
        if not isinstance(block, list):
            continue
        rewritten: list[Any] = []
        for item in block:
            if not isinstance(item, dict):
                rewritten.append(item)
                continue
            row = dict(item)
            raw = str(row.get("network_tier") or "").strip()
            label = normalize_domain(raw) or domain_label(raw)
            if label:
                row["network_tier"] = label
            rewritten.append(row)
        out[key] = rewritten
    return out


def pragma_domain_from_tier_data(_tier_key: str, tier_data: dict[str, Any] | None) -> str:
    if isinstance(tier_data, dict):
        pd = tier_data.get("pragma_domain")
        if isinstance(pd, str) and pd.strip():
            return pd.strip()
    return domain_label(_tier_key)


# Back-compat aliases used by older imports (same behavior as domain helpers).
canonicalize_network_tier = normalize_domain
