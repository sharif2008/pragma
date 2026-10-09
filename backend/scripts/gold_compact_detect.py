"""Compact 90-row detect product + per-class parent candidates for gold draft."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent

LIVE = _REPO / "experiments" / "gold-100"
FLOWS = _REPO / "experiments" / "data" / "gold-100" / "flows.csv"
PARENTS = _REPO / "experiments" / "rag-index" / "vector_store" / "rag_parents.json"
DRAFTS = LIVE / "drafts"
CATALOG = DRAFTS / "_parent_catalog_slim.json"
OUT_DETECT = DRAFTS / "_detect_compact.json"
OUT_CANDS = DRAFTS / "_parent_candidates_by_class.json"
N_GOLD = 90

KEYS = {
    "BENIGN": ["audit log", "monitor", "logging", "incident", "continuous monitoring", "control 8", "control 13"],
    "BOT": ["botnet", "malware", "command and control", "control 10", "web", "captcha"],
    "DDOS": ["denial of service", "ddos", "flood", "rate", "availability", "network defense", "control 13"],
    "DOS": ["denial of service", "dos", "flood", "rate", "syncookie", "availability"],
    "FTPPATATOR": ["brute", "credential", "account", "ftp", "authentication", "control 5", "control 6"],
    "OTHERS": ["incident", "monitor", "unknown", "anomaly", "log"],
    "PORTSCAN": ["scan", "reconnaissance", "port", "network monitoring", "control 13", "exposure"],
    "SSHPATATOR": ["brute", "credential", "ssh", "account", "authentication", "control 5", "control 6"],
    "WEBATTACK": ["web", "application", "waf", "injection", "control 16", "browser"],
}


def top_feats(feat_map: dict, n: int = 3) -> list[dict]:
    items = []
    for name, rec in (feat_map or {}).items():
        items.append((float(rec.get("abs_shap_value") or 0), name, rec.get("pct_contribution")))
    items.sort(reverse=True)
    return [{"name": name, "abs_shap": a, "pct": p} for a, name, p in items[:n]]


def resolve_pred_json() -> Path:
    files = sorted(LIVE.glob("predictions_detailed_*.json"))
    if files:
        return files[-1]
    raise SystemExit(f"No predictions_detailed_*.json in {LIVE} (gold-100 does not read detect-predict/)")


def load_or_build_catalog() -> dict:
    if CATALOG.is_file():
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    raw = json.loads(PARENTS.read_text(encoding="utf-8"))["parents"]
    slim = []
    for pid, rec in raw.items():
        text = rec.get("text") or ""
        slim.append(
            {
                "parent_id": rec.get("parent_id") or pid,
                "source_file": rec.get("source_file"),
                "section_heading": rec.get("section_heading") or "",
                "preview": text[:800],
                "text_hash": "sha256:" + __import__("hashlib").sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    catalog = {"n": len(slim), "parents": slim}
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    CATALOG.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog


def main() -> int:
    pred_path = resolve_pred_json()
    with FLOWS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if len(preds) != N_GOLD or len(rows) != N_GOLD:
        raise SystemExit(f"expected {N_GOLD} rows, got flows={len(rows)} preds={len(preds)}")

    compact = []
    for i, (flow, pred) in enumerate(zip(rows, preds), start=1):
        shap = pred.get("shap_explanation") or {}
        fc = shap.get("feature_contributions") or {}
        compact.append(
            {
                "case_id": f"G-{i:03d}",
                "gold_row": int(flow.get("gold_row") or i),
                "sample_id": pred.get("sample_id"),
                "split_index": int(float(flow["split_index"])),
                "flow_id": f"split_{flow['split_index']}",
                "true_label": str(flow.get("label_simplified") or pred.get("true_label") or "").upper(),
                "predicted_label": pred.get("predicted_label"),
                "confidence": pred.get("confidence"),
                "is_correct": pred.get("is_correct"),
                "dominant_domain": shap.get("dominant_agent"),
                "domain_shares": shap.get("party_contributions_pct"),
                "top_features": {
                    dom: top_feats(fc.get(dom))
                    for dom in ("Access / ISP", "Perimeter / IDS", "Endpoint / EDR")
                },
            }
        )

    catalog = load_or_build_catalog()
    cands: dict[str, list] = {}
    for cls, kws in KEYS.items():
        scored = []
        for rec in catalog["parents"]:
            blob = ((rec.get("section_heading") or "") + " " + (rec.get("preview") or "")).lower()
            hit = sum(1 for k in kws if k.lower() in blob)
            if hit:
                scored.append((hit, rec))
        scored.sort(key=lambda x: -x[0])
        cands[cls] = [r for _, r in scored[:12]]

    OUT_DETECT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DETECT.write_text(
        json.dumps({"n": len(compact), "detect_json": str(pred_path), "cases": compact}, indent=2),
        encoding="utf-8",
    )
    OUT_CANDS.write_text(json.dumps(cands, indent=2), encoding="utf-8")
    print("compact", len(compact), "correct", sum(1 for c in compact if c["is_correct"]))
    print("true", dict(Counter(c["true_label"] for c in compact)))
    print("pred", dict(Counter(c["predicted_label"] for c in compact)))
    print("candidates", {k: len(v) for k, v in cands.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
