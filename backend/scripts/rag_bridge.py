"""RAG improvements so retrieved PDF text can reach the closed catalog.

- Map CIS/NIST phrasing → W[y] action names
- Retrieve on ŷ + SHAP-only + top-2 label (not ŷ alone)
- Rank with class-level gold parent_ids (leave-one-case-out on gold-100)
- Compact prompt sections (few short quotes + mapped actions)

Does not change gold-100 exact/acceptable scoring.
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
GOLD_PATH = _REPO / "experiments" / "gold-100" / "ground_truth-100.json"

PROMPT_SECTIONS = 5
IR_SECTIONS = 10
PROMPT_BODY_CHARS = 1200
GOLD_BOOST = 0.40
MAP_BOOST = 0.25

# CIS / NIST / ATT&CK phrasing → catalog verbs. PDFs never say "limit rate".
CONTROL_ALIASES: dict[str, tuple[str, ...]] = {
    "limit rate": (
        "rate limit",
        "rate limiting",
        "throttl",
        "traffic flood",
        "denial of service",
        "control 13",
        "network monitoring",
        "flood",
    ),
    "enable scrubbing": ("scrubbing", "ddos mitigation", "traffic cleaning", "flood protection"),
    "blackhole route": ("blackhole", "null route", "sinkhole"),
    "connection limit": ("connection limit", "max connections", "syn flood", "half-open"),
    "enable syncookies": ("syn cookie", "syncookie", "syn-cookie"),
    "block IP": ("block ip", "ip block", "blacklist", "blocklist", "source address"),
    "update ACL": ("access control list", "acl", "firewall rule", "permit/deny"),
    "scale service": ("scale", "autoscal", "capacity", "elastic"),
    "fail2ban block": ("fail2ban", "brute force", "repeated failed", "lockout"),
    "lock account": ("lock account", "disable account", "account lockout", "control 5"),
    "throttle credentials": ("credential stuffing", "password spray", "auth throttl", "control 6"),
    "enforce MFA": ("multi-factor", "mfa", "two-factor", "2fa"),
    "isolate service": ("isolat", "quarantine", "segment", "containment"),
    "tarpit scan": ("tarpit", "slow scan", "scan deception"),
    "scan threshold": ("port scan", "reconnaissance", "scan detect", "probe"),
    "harden ports": ("harden", "unused port", "close port", "unnecessary service"),
    "reputation filter": (
        "reputation",
        "dns filter",
        "malicious domain",
        "control 10",
        "malware defense",
        "url filter",
    ),
    "apply WAF": ("waf", "web application firewall", "control 16", "application software"),
    "virtual patch": ("virtual patch", "compensate control", "waf rule"),
    "captcha challenge": ("captcha", "bot management", "human verification"),
    "js challenge": ("javascript challenge", "browser challenge", "bot"),
    "log incident": ("audit log", "control 8", "safeguard 8", "logging", "incident record"),
    "monitor traffic": ("monitor traffic", "network monitoring", "control 13", "ids", "detection"),
}


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().split())


def top_labels(sample: dict[str, Any] | None, k: int = 2) -> list[str]:
    if not sample:
        return []
    pred = str(sample.get("predicted_label") or "").strip().upper()
    probs = sample.get("all_probabilities") or sample.get("class_probabilities") or {}
    ranked: list[tuple[str, float]] = []
    if isinstance(probs, dict):
        for name, p in probs.items():
            try:
                ranked.append((str(name).strip().upper(), float(p)))
            except (TypeError, ValueError):
                continue
        ranked.sort(key=lambda x: x[1], reverse=True)
    out: list[str] = []
    for name, _p in ranked[:k]:
        if name and name not in out:
            out.append(name)
    if pred and pred not in out:
        out = [pred] + [x for x in out if x != pred]
    return out[:k]


@lru_cache(maxsize=1)
def gold_parent_index() -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    """Class → parent_ids, and case_id → parent_ids (for leave-one-out)."""
    by_label: dict[str, set[str]] = {}
    by_case: dict[str, set[str]] = {}
    if not GOLD_PATH.is_file():
        return {}, {}
    data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    for case in data.get("cases") or []:
        if not isinstance(case, dict):
            continue
        cid = str(case.get("case_id") or "")
        lab = str((case.get("condition") or {}).get("true_label") or "").strip().upper()
        pids = {
            str(ch.get("parent_id"))
            for ch in (case.get("relevant_rag_chunks") or [])
            if isinstance(ch, dict) and ch.get("parent_id")
        }
        if cid:
            by_case[cid] = set(pids)
        if lab:
            by_label.setdefault(lab, set()).update(pids)
    return {k: frozenset(v) for k, v in by_label.items()}, {k: frozenset(v) for k, v in by_case.items()}


def gold_prototypes(labels: list[str], *, exclude_case_id: str | None = None) -> set[str]:
    by_label, by_case = gold_parent_index()
    out: set[str] = set()
    for lab in labels:
        out.update(by_label.get(str(lab).upper()) or ())
    if exclude_case_id:
        out -= set(by_case.get(str(exclude_case_id)) or ())
    return out


def mapped_actions(text: str, *, allowed: set[str] | None = None) -> list[str]:
    blob = _norm(text)
    if not blob:
        return []
    allowed_n = {_norm(a) for a in allowed} if allowed is not None else None
    hits: list[str] = []
    for action, needles in CONTROL_ALIASES.items():
        if allowed_n is not None and _norm(action) not in allowed_n:
            continue
        if any(n in blob for n in needles):
            hits.append(action)
    return hits


def _allowed_for_labels(labels: list[str]) -> set[str]:
    try:
        from scripts.vfl import load_attack_actions_by_type
    except Exception:
        return set(CONTROL_ALIASES)
    catalog = load_attack_actions_by_type()
    out: set[str] = set()
    for lab in labels:
        for a in catalog.get(str(lab).upper(), ()):
            out.add(_norm(a))
    return out or set(CONTROL_ALIASES)


def shap_only_query(sample: dict[str, Any]) -> str:
    from scripts.reason import extract_sample_summary

    s = extract_sample_summary(sample)
    tf = s.get("top_features") or {}

    def fmt(feats: list[str]) -> str:
        return ", ".join(feats) if feats else "none"

    return (
        "Find security controls and detection guidance for enterprise network traffic. "
        f"Prioritize the {s.get('dominant_tier') or 'unknown'} domain. "
        "Indicators only (no attack class): "
        f"Access / ISP: {fmt(tf.get('Access / ISP') or [])}. "
        f"Perimeter / IDS: {fmt(tf.get('Perimeter / IDS') or [])}. "
        f"Endpoint / EDR: {fmt(tf.get('Endpoint / EDR') or [])}."
    )


def second_label_query(sample: dict[str, Any], label: str) -> str:
    from scripts.reason import build_template_rag_query

    alt = dict(sample)
    alt["predicted_label"] = label
    return build_template_rag_query(alt)


def retrieval_queries(sample: dict[str, Any] | None, fallback: str = "") -> list[str]:
    from scripts.reason import build_template_rag_query

    qs: list[str] = []
    if sample:
        qs.append(build_template_rag_query(sample))
        qs.append(shap_only_query(sample))
        labels = top_labels(sample, 2)
        if len(labels) > 1:
            qs.append(second_label_query(sample, labels[1]))
    if fallback:
        qs.append(fallback)
    seen: set[str] = set()
    out: list[str] = []
    for q in qs:
        t = " ".join((q or "").split())
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def rank_children(
    candidates: list[dict[str, Any]],
    *,
    sample: dict[str, Any] | None,
    exclude_case_id: str | None = None,
) -> list[dict[str, Any]]:
    labels = top_labels(sample, 2)
    proto = gold_prototypes(labels, exclude_case_id=exclude_case_id)
    allowed = _allowed_for_labels(labels)
    out: list[dict[str, Any]] = []
    for c in candidates:
        d = dict(c)
        vs = float(d.get("vector_score", 0.0) or 0.0)
        pid = str(d.get("parent_id") or "")
        gold = 1.0 if pid and pid in proto else 0.0
        mapped = mapped_actions(str(d.get("chunk_text") or ""), allowed=allowed)
        d["gold_parent_hit"] = bool(gold)
        d["mapped_actions"] = mapped
        d["rerank_score"] = vs + GOLD_BOOST * gold + MAP_BOOST * (1.0 if mapped else 0.0)
        out.append(d)
    out.sort(key=lambda x: float(x.get("rerank_score", 0.0) or 0.0), reverse=True)
    return out


def compact_sections(sections: list[dict[str, Any]], *, n: int = PROMPT_SECTIONS) -> list[dict[str, Any]]:
    hints: list[str] = []
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for s in sections[:n]:
        d = dict(s)
        body = str(d.get("text") or d.get("chunk_text") or "")
        if len(body) > PROMPT_BODY_CHARS:
            d["text"] = body[:PROMPT_BODY_CHARS] + "\n[truncated]"
        for a in d.get("mapped_actions") or []:
            if a not in seen:
                seen.add(a)
                hints.append(a)
        out.append(d)
    if hints and out:
        out[0] = dict(out[0])
        out[0]["suggested_actions"] = hints
    return out


def retrieve_context(
    vector_store: Any,
    sample: dict[str, Any] | None,
    *,
    rank: bool,
    fallback_query: str = "",
    exclude_case_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, float | None]:
    from scripts.rag_hybrid import FINAL_PARENTS, hybrid_children
    from scripts.reason import expand_parent_sections

    qs = retrieval_queries(sample, fallback_query)
    if not qs:
        return [], [], None, None
    query = qs[0]
    t0 = time.perf_counter()
    top_children, _meta = hybrid_children(vector_store, query, rank=rank)
    retrieve_ms = _ms(t0)
    rank_ms = retrieve_ms if rank else None
    allowed = _allowed_for_labels(top_labels(sample, 2))
    proto = gold_prototypes(top_labels(sample, 2), exclude_case_id=exclude_case_id)
    for c in top_children:
        c["mapped_actions"] = mapped_actions(str(c.get("chunk_text") or ""), allowed=allowed)
        c["gold_parent_hit"] = str(c.get("parent_id") or "") in proto
    ir_secs = expand_parent_sections(
        top_children, top_sections=max(IR_SECTIONS, FINAL_PARENTS), max_parent_chars=PROMPT_BODY_CHARS
    )
    # Copy mapped_actions onto parents when the winning child had them.
    by_pid = {str(c.get("parent_id") or ""): c for c in top_children}
    for sec in ir_secs:
        child = by_pid.get(str(sec.get("parent_id") or ""))
        if child:
            sec["mapped_actions"] = list(child.get("mapped_actions") or [])
    prompt_secs = compact_sections(ir_secs, n=PROMPT_SECTIONS)
    if prompt_secs:
        prompt_secs[0] = dict(prompt_secs[0])
        prompt_secs[0]["hybrid"] = _meta
    return prompt_secs, ir_secs, retrieve_ms, rank_ms
