"""SPEC quality gates for the single ground_truth-100.json freeze."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.vfl import load_attack_actions_by_type

LIVE = _REPO / "experiments" / "gold-100"
OUT_NAME = "ground_truth-100.json"
PARENTS = _REPO / "experiments" / "rag-index" / "vector_store" / "rag_parents.json"
FIXTURE = _REPO / "experiments" / "data" / "gold-100" / "manifest.json"
PDFS = {
    "NIST-SP-800-53-Rev5-Security-Privacy-Controls.pdf",
    "NIST-SP-800-207-Zero-Trust-Architecture.pdf",
    "ISO-IEC-27001-ISMS-Requirements.pdf",
    "CIS-Controls-v8.1.pdf",
    "MITRE-ATTCK-Design-and-Philosophy.pdf",
    "MITRE-ATTCK-Building-Better-Defenses.pdf",
    "CISO-Guide-Top-Cybersecurity-Frameworks.pdf",
    "Open-RAN-Security-Report.pdf",
    "NIST-SP-800-53-vs-ISO-IEC-27001-Comparison.pdf",
    "Review-NIST-ISO27001-HIPAA-MITRE-ATTCK.pdf",
}
TIERS = {"Access / ISP", "Perimeter / IDS", "Endpoint / EDR"}
NINE = {
    "BENIGN",
    "BOT",
    "DDOS",
    "DOS",
    "FTPPATATOR",
    "OTHERS",
    "PORTSCAN",
    "SSHPATATOR",
    "WEBATTACK",
}
FORBIDDEN = {"BLOCK_IP", "RATE_LIMIT_IP", "ALERT_SOC", "HUMAN_APPROVAL", "NO_ACTION", "ISOLATE_HOST"}
CLASS_NEEDLES = {
    "BENIGN": ("audit log", "logging", "safeguard 8.1"),
    "BOT": ("malware", "botnet", "control 10"),
    "DDOS": ("denial of service", "ddos", "network monitoring"),
    "DOS": ("denial of service", "dos", "network monitoring"),
    "FTPPATATOR": ("account", "authentication", "brute force", "access control"),
    "OTHERS": ("incident", "audit log", "monitor"),
    "PORTSCAN": ("reconnaissance", "network monitoring", "control 13"),
    "SSHPATATOR": ("account", "authentication", "brute force", "access control"),
    "WEBATTACK": ("application software", "control 16", "web"),
}


def main() -> int:
    live_path = LIVE / OUT_NAME
    if not live_path.is_file():
        print(f"FAIL missing {live_path}")
        return 1
    live = json.loads(live_path.read_text(encoding="utf-8"))
    cases = live.get("cases") or []
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parents = json.loads(PARENTS.read_text(encoding="utf-8"))["parents"]
    by_type = load_attack_actions_by_type()
    allowed = {a for acts in by_type.values() for a in acts}
    errs: list[str] = []

    ids = [c["case_id"] for c in cases]
    six = [c["split_index"] for c in cases]
    hist = Counter(str((c.get("condition") or {}).get("true_label") or "") for c in cases)
    if len(cases) != 90:
        errs.append(f"n={len(cases)} expected 90")
    if len(set(ids)) != len(cases):
        errs.append("duplicate case_id")
    if len(set(six)) != len(cases):
        errs.append("duplicate split_index")
    extra = set(six) - set(fixture["selected_split_index"])
    if extra:
        errs.append(f"split_index not in fixture sample: {len(extra)}")
    for lab in sorted(NINE):
        if hist.get(lab, 0) != 10:
            errs.append(f"{lab} count {hist.get(lab, 0)} expected 10")

    for c in cases:
        tl = (c.get("condition") or {}).get("true_label")
        if tl not in NINE:
            errs.append(f"{c.get('case_id')} bad label {tl}")
            continue
        acts = c["actions"]
        w = set(by_type[tl])
        if acts["primary_action"] not in w:
            errs.append(f"{c['case_id']} primary not in W")
        if tl == "BENIGN" and acts["primary_action"] not in {"log incident", "monitor traffic"}:
            errs.append(f"{c['case_id']} BENIGN primary")
        if acts["primary_action"] in acts["unsafe_actions"]:
            errs.append(f"{c['case_id']} primary in unsafe")
        if acts["primary_action"] not in acts["acceptable_actions"]:
            errs.append(f"{c['case_id']} primary not acceptable")
        if c["condition"]["primary_network_tier"] not in TIERS:
            errs.append(f"{c['case_id']} bad tier")
        for a in list(acts["acceptable_actions"]) + list(acts["unsafe_actions"]) + [acts["primary_action"]]:
            if a in FORBIDDEN or a not in allowed:
                errs.append(f"{c['case_id']} bad action {a!r}")
        rat = c.get("rationale") or ""
        chunks = c.get("relevant_rag_chunks") or []
        if not (c.get("rationale_summary") or "").strip():
            errs.append(f"{c['case_id']} missing rationale_summary")
        csum = c.get("chunks_summary") or ""
        if chunks and not csum.strip():
            errs.append(f"{c['case_id']} missing chunks_summary")
        if chunks and len(csum.split()) < 40:
            errs.append(f"{c['case_id']} chunks_summary too short ({len(csum.split())} words)")
        if re.search(r"\.pdf\b|\bp_[0-9a-f]{8,}\b", csum, re.I):
            errs.append(f"{c['case_id']} chunks_summary cites a document")
        if chunks:
            blob = " ".join((ch.get("text") or "") for ch in chunks).lower()
            tokens = {w for w in re.findall(r"[a-z]{5,}", csum.lower())}
            overlap = sum(1 for w in tokens if w in blob)
            if overlap < 5:
                errs.append(f"{c['case_id']} chunks_summary not grounded in chunk text")
        if chunks and not (1 <= len(chunks) <= 20):
            errs.append(f"{c['case_id']} chunk count {len(chunks)} (want 1–20)")
        for ch in chunks:
            pid = ch["parent_id"]
            if pid not in parents:
                errs.append(f"{c['case_id']} missing parent {pid}")
                continue
            rec = parents[pid]
            if ch["source_file"] not in PDFS or rec.get("source_file") != ch["source_file"]:
                errs.append(f"{c['case_id']} bad source {ch['source_file']}")
            text = rec.get("text") or ""
            expect = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            stored = ch.get("text")
            if not stored:
                errs.append(f"{c['case_id']} chunk {pid} missing full text")
            elif stored != text:
                errs.append(f"{c['case_id']} stored text != parent {pid}")
            if ch.get("text_hash") != expect:
                errs.append(f"{c['case_id']} hash mismatch {pid}")
            needles = CLASS_NEEDLES.get(tl) or ()
            if needles and not any(n in text.lower() for n in needles):
                errs.append(f"{c['case_id']} parent {pid} text does not support {tl}")
        if chunks:
            if chunks[0]["parent_id"] not in rat and chunks[0]["source_file"] not in rat:
                errs.append(f"{c['case_id']} rationale missing PDF cite")
            ptext = parents[chunks[0]["parent_id"]].get("text") or ""
            words = [w for w in re.sub(r"\s+", " ", ptext).split() if w]
            found_quote = False
            for i in range(0, max(0, len(words) - 7)):
                span = " ".join(words[i : i + 8])
                if span and span in rat:
                    found_quote = True
                    break
            if not found_quote:
                errs.append(f"{c['case_id']} rationale missing PDF quote")

    gen = live.get("generation") or {}
    if gen.get("openai_api_used") is not False:
        errs.append("openai_api_used not false")

    extra = [
        p.name
        for p in LIVE.iterdir()
        if p.is_file()
        and p.name not in {OUT_NAME, "README.md", ".gitkeep"}
        and not p.name.startswith(("predictions_", "decision_summary_"))
    ]
    if extra:
        errs.append(f"extra files in experiments/gold-100: {extra}")

    if errs:
        print("FAIL", len(errs))
        for e in errs[:30]:
            print(" -", e)
        return 1
    print("OK 90 cases in experiments/gold-100/ground_truth-100.json; parents+actions valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
