"""Freeze 90 gold cases: catalog actions chosen from PDF parent text.

Does not call OpenAI. Eval (pragma-rag-eval100) uses the LLM API against this gold.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from datetime import date
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.env import named_live_dir
from scripts.vfl import load_attack_actions_by_type, not_allowed_actions_for_type

LIVE = named_live_dir("gold-100", mkdir=True)
FLOWS = _REPO / "experiments" / "data" / "gold-100" / "flows.csv"
OUT_NAME = "ground_truth-100.json"
OPTIONS = _BACKEND / "storage" / "attack_options.json"
KEEP_LIVE = {".gitkeep", "README.md", OUT_NAME}
KEEP_LIVE_PREFIX = ("predictions_", "decision_summary_")
JUNK_GROUND = {
    "gold_security_cases.json",
    "gold_security_cases.csv",
    "generation.json",
    "split_manifest.json",
    "SHA256.txt",
}
FIXTURE_MANIFEST = _REPO / "experiments" / "data" / "gold-100" / "manifest.json"
PARENTS = _REPO / "experiments" / "rag-index" / "vector_store" / "rag_parents.json"
GROUND = LIVE
TIERS = {"Access / ISP", "Perimeter / IDS", "Endpoint / EDR"}
DATE = date.today().isoformat()
# Needles must appear in parent *body text*. Primary is the W[class] action that control supports.
CLASS_PDF: dict[str, dict] = {
    "BENIGN": {
        "needles": ("control 8: audit log", "safeguard 8.1"),
        "primary": "log incident",
        "accept_extra": ("monitor traffic",),
    },
    "BOT": {
        "needles": ("control 10: malware", "malware defenses"),
        "primary": "reputation filter",
        "accept_extra": ("apply WAF", "monitor traffic"),
    },
    "DDOS": {
        "needles": ("denial of service", "control 13: network monitoring"),
        "primary": "limit rate",
        "accept_extra": ("enable scrubbing", "block IP"),
    },
    "DOS": {
        "needles": ("denial of service", "control 13: network monitoring"),
        "primary": "limit rate",
        "accept_extra": ("enable scrubbing", "monitor traffic"),
    },
    "SSHPATATOR": {
        "needles": ("control 6: access control", "control 5: account", "brute force"),
        "primary": "enforce MFA",
        "accept_extra": ("lock account", "fail2ban block"),
    },
    "FTPPATATOR": {
        "needles": ("control 5: account", "control 6: access control", "brute force"),
        "primary": "lock account",
        "accept_extra": ("enforce MFA", "fail2ban block"),
    },
    "PORTSCAN": {
        "needles": ("control 13: network monitoring", "reconnaissance"),
        "primary": "block IP",
        "accept_extra": ("harden ports", "log incident"),
    },
    "WEBATTACK": {
        "needles": ("control 16: application", "application software security"),
        "primary": "apply WAF",
        "accept_extra": ("virtual patch", "monitor traffic"),
    },
    "OTHERS": {
        "needles": ("control 17: incident", "incident response"),
        "primary": "log incident",
        "accept_extra": ("monitor traffic", "update ACL"),
    },
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _tier(true_label: str, dominant: str, primary_domains: dict[str, list[str]]) -> str:
    prefs = list(primary_domains.get(true_label) or [])
    if dominant in TIERS and dominant in prefs:
        return dominant
    if prefs and prefs[0] in TIERS:
        return prefs[0]
    if dominant in TIERS:
        return dominant
    return "Perimeter / IDS"


def _primary(true_label: str, w: tuple[str, ...]) -> str:
    want = str((CLASS_PDF.get(true_label) or {}).get("primary") or "")
    if want in w:
        return want
    if true_label == "BENIGN" and "log incident" in w:
        return "log incident"
    return w[0]


def _acceptable(true_label: str, w: tuple[str, ...], primary: str) -> list[str]:
    extra = list((CLASS_PDF.get(true_label) or {}).get("accept_extra") or ())
    out = [primary]
    for a in extra + list(w):
        if a in w and a not in out:
            out.append(a)
        if len(out) >= 3:
            break
    return out


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


_HAS_SAFEGUARD = re.compile(r"Safeguard\s+\d+\.\d+\s*:", re.I)


def _is_junk_parent(text: str) -> bool:
    """Front matter / TOC — not usable as RAG evidence."""
    raw = text or ""
    low = raw.lower()
    head = low[:1000]
    if ("acknowledgments" in head or "creative commons" in head) and not _HAS_SAFEGUARD.search(raw):
        return True
    if "contents" in head and raw.count("....") >= 4:
        return True
    return False


def _pick_parents(parents: dict, true_label: str, k: int = 20) -> list[dict]:
    needles = [str(n).lower() for n in (CLASS_PDF.get(true_label) or {}).get("needles") or ()]
    scored: list[tuple[int, int, str, dict]] = []
    for pid, rec in parents.items():
        text = rec.get("text") or ""
        low = text.lower()
        hit = sum(1 for n in needles if n in low)
        if hit == 0 or _is_junk_parent(text):
            continue
        page = int(rec.get("page_start") or 0)
        if page < 10 and "safeguard" not in low:
            continue
        bonus = 2 if "safeguard" in low else 0
        scored.append((hit + bonus, len(text), rec.get("parent_id") or pid, rec))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    out: list[dict] = []
    seen: set[str] = set()
    for _s, _n, pid, rec in scored:
        pref = _normalize_ws(rec.get("text") or "")[:160]
        if pref in seen:
            continue
        seen.add(pref)
        out.append(rec)
        if len(out) >= k:
            break
    return out


def _quote(text: str, needles: tuple[str, ...] | list[str], max_words: int = 22) -> str:
    compact = _normalize_ws(text)
    low = compact.lower()
    pos = -1
    for n in needles:
        i = low.find(str(n).lower())
        if i >= 0:
            pos = i
            break
    if pos < 0:
        pos = 0
    start = compact.rfind(". ", 0, pos)
    start = 0 if start < 0 else start + 2
    words = compact[start:].split()
    return " ".join(words[:max_words])


def _clean_task_outputs() -> None:
    """Drop gold scratch. Keep detect/SHAP files. Overwrite ground_truth-100.json only."""
    drafts = LIVE / "drafts"
    if drafts.is_dir():
        shutil.rmtree(drafts)
    for p in list(LIVE.iterdir()):
        if p.name.startswith(".") and p.name != ".gitkeep":
            continue
        if p.name in KEEP_LIVE or p.name.startswith(KEEP_LIVE_PREFIX):
            continue
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)
    for name in JUNK_GROUND:
        leftover = LIVE / name
        if leftover.is_file():
            leftover.unlink()


def _top_feats(feat_map: dict, n: int = 3) -> list[dict]:
    items = []
    for name, rec in (feat_map or {}).items():
        items.append((float(rec.get("abs_shap_value") or 0), name, rec.get("pct_contribution")))
    items.sort(reverse=True)
    return [{"name": name, "abs_shap": a, "pct": p} for a, name, p in items[:n]]


def _load_detect_rows() -> tuple[list[dict], Path]:
    pred_files = sorted(LIVE.glob("predictions_detailed_*.json"))
    if not pred_files:
        raise SystemExit(f"No predictions_detailed_*.json in {LIVE}")
    pred_path = pred_files[-1]
    with FLOWS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if len(rows) != 90 or len(preds) != 90:
        raise SystemExit(f"expected 90 rows, got flows={len(rows)} preds={len(preds)}")
    compact = []
    for i, (flow, pred) in enumerate(zip(rows, preds), start=1):
        shap = pred.get("shap_explanation") or {}
        fc = shap.get("feature_contributions") or {}
        compact.append(
            {
                "case_id": f"G-{i:03d}",
                "split_index": int(float(flow["split_index"])),
                "flow_id": f"split_{flow['split_index']}",
                "true_label": str(flow.get("label_simplified") or pred.get("true_label") or "").upper(),
                "predicted_label": pred.get("predicted_label"),
                "confidence": pred.get("confidence"),
                "dominant_domain": shap.get("dominant_agent"),
                "domain_shares": shap.get("party_contributions_pct"),
                "top_features": {
                    dom: _top_feats(fc.get(dom))
                    for dom in ("Access / ISP", "Perimeter / IDS", "Endpoint / EDR")
                },
            }
        )
    return compact, pred_path


def _shap_list(case: dict) -> list[dict]:
    out = []
    for domain, feats in (case.get("top_features") or {}).items():
        for f in feats or []:
            out.append(
                {
                    "domain": domain,
                    "name": f.get("name"),
                    "abs_shap": f.get("abs_shap"),
                    "pct": f.get("pct"),
                }
            )
    return out


def _rationale(
    case: dict,
    primary: str,
    tier: str,
    chunks: list[dict],
    quotes: list[tuple[dict, str]],
) -> tuple[str, str, list[str]]:
    tl = case["true_label"]
    pred = case["predicted_label"]
    conf = float(case.get("confidence") or 0)
    dom = case.get("dominant_domain")
    feats = []
    for domain, rows in (case.get("top_features") or {}).items():
        if rows:
            feats.append(f"{domain}: {rows[0]['name']}")
    feat_s = "; ".join(feats) if feats else "no top features"
    if quotes:
        parts = []
        for cite, quote in quotes:
            parts.append(f"From {cite['source_file']} ({cite['parent_id']}): \"{quote}\"")
        rag = " ".join(parts)
        rag += f" These PDF sections support catalog action '{primary}' at {tier}."
        empty = False
    else:
        rag = "The closed ten-PDF index has no parent whose text supports this class; relevant_rag_chunks is empty."
        empty = True
    full = (
        f"True label {tl} (predicted {pred}, confidence {conf:.1%}). "
        f"SHAP-dominant domain is {dom}. Top evidence: {feat_s}. {rag} "
        f"Gold uses {len(chunks)} supporting PDF parent(s) (top 20 by body-text match)."
    )
    if empty:
        summary = f"{tl}: do '{primary}' at {tier}."
    else:
        summary = (
            f"{tl}: do '{primary}' at {tier}. "
            f"Detect labeled it {pred} ({conf:.0%}); SHAP led by {dom} ({feat_s})."
        )
    points = [
        f"Class {tl} restricts actions to W[{tl}] in attack_options.json.",
        f"PDF-supported primary action is '{primary}' at {tier} (SHAP domain {dom}).",
        f"Evidence cues / top SHAP: {feat_s}.",
    ]
    if empty:
        points.append("No matching parent_id in rag_parents.json for this case.")
    else:
        points.append(
            f"Top parent {quotes[0][0]['parent_id']} from {quotes[0][0]['source_file']}: {quotes[0][1]}"
        )
        if len(chunks) > 1:
            points.append(f"Kept {len(chunks)} PDF parents (full text) for this rationale.")
    return full, summary, points[:5]


_DOC_NOISE = re.compile(
    r"(---\s*Page\s+\d+\s*---)|"
    r"(CIS Controls? v?[\d.]+)|"
    r"(NIST[\w.\- ]{0,40})|"
    r"(ISO-IEC[^\s,]*)|"
    r"\b[\w\-]+\.pdf\b|"
    r"\bp_[0-9a-f]+\b",
    re.IGNORECASE,
)

ACTION_BLURB = {
    "BENIGN": "Log the incident and monitor traffic. Do not block or rate-limit.",
    "BOT": "Filter bot reputation and challenge automated clients. Limit abusive rate.",
    "DDOS": "Limit rate and enable scrubbing so the service stays available.",
    "DOS": "Limit rate and tighten connection controls to stop the flood.",
    "SSHPATATOR": "Lock the account, enforce MFA, and throttle credential guesses.",
    "FTPPATATOR": "Lock the account, enforce MFA, and stop the brute-force login.",
    "PORTSCAN": "Block the scanner IP, harden exposed ports, and log the scan.",
    "WEBATTACK": "Apply WAF or a virtual patch; isolate the app if needed.",
    "OTHERS": "Log the incident and monitor; escalate only if the signal grows.",
}


def _clean_quote(text: str) -> str:
    cleaned = _DOC_NOISE.sub(" ", text or "")
    return _normalize_ws(cleaned)


_LABEL_PHRASE = {
    "BENIGN": "benign traffic",
    "BOT": "bot activity",
    "DDOS": "distributed denial of service",
    "DOS": "denial of service",
    "SSHPATATOR": "SSH credential stuffing",
    "FTPPATATOR": "FTP brute-force login",
    "PORTSCAN": "port scanning and reconnaissance",
    "WEBATTACK": "web application attack",
    "OTHERS": "uncommon or uncategorized incident",
}
_CONTROL_TITLE = re.compile(
    r"Control\s+(\d+)\s*:\s*([A-Z][A-Za-z0-9 /&-]{6,55})",
    re.I,
)
_SAFEGUARD_SPAN = re.compile(
    r"Safeguard\s+(\d+\.\d+)\s*:\s*(.+?)(?=Safeguard\s+\d+\.\d+\s*:|Control\s+\d+\s*:|CONTROL\s+\d+\b|Why is this Control|Procedures and tools|$)",
    re.I | re.S,
)
_OVERVIEW = re.compile(
    r"(?:Overview|Why is this Control critical\?)\s+(.{60,420}?)(?:\s+(?:Safeguard|Procedures|CONTROL)\b|$)",
    re.I | re.S,
)
_IMPERATIVE = re.compile(
    r"\b(Deploy|Use|Enable|Configure|Enforce|Require|Establish|Perform|Conduct|"
    r"Prevent|Delete|Disable|Apply|Collect|Maintain|Limit|Block|Filter|Lock|"
    r"Review|Monitor|Implement|Authenticate|Segment|Isolate|Ensure|Restrict|"
    r"Centralize|Designate|Assign|Determine|Remediate|Validate)\b",
    re.I,
)
_META_RES = (
    re.compile(r"---\s*Page\s+\d+\s*---", re.I),
    re.compile(r"Asset Type:\s*[^\n|]+", re.I),
    re.compile(r"Security Function:\s*[^\n|]+", re.I),
    re.compile(r"\|\s*I\s*G[123]\b", re.I),
    re.compile(r"\bIG[123]\b", re.I),
    re.compile(r"CIS Controls? v?[\d.]+", re.I),
    re.compile(r"\s*\|\s*"),
)
_SCORE_EXTRA = {
    "BENIGN": ("audit log", "logging", "anomaly"),
    "BOT": ("anti-malware", "malware", "dns filter", "url filter", "malicious"),
    "DDOS": ("intrusion", "traffic filter", "denial of service", "network monitoring"),
    "DOS": ("intrusion", "traffic filter", "denial of service", "network monitoring"),
    "SSHPATATOR": ("mfa", "account", "authentication", "access", "brute"),
    "FTPPATATOR": ("mfa", "account", "authentication", "access", "brute"),
    "PORTSCAN": ("intrusion", "reconnaissance", "network monitoring", "traffic filter", "penetration"),
    "WEBATTACK": ("application", "secure coding", "waf", "web", "penetration"),
    "OTHERS": ("incident", "response", "report", "communicate", "roles and responsibilities", "handling"),
}
_CLASS_CONTROLS = {
    "BENIGN": {8},
    "BOT": {9, 10},
    "DDOS": {12, 13},
    "DOS": {12, 13},
    "SSHPATATOR": {5, 6},
    "FTPPATATOR": {5, 6},
    "PORTSCAN": {13, 18},
    "WEBATTACK": {16},
    "OTHERS": {17},
}
_OCR_FIX = (
    (re.compile(r"\badmi\s*tor\b", re.I), "administrator"),
    (re.compile(r"\badmi\s*istrative\b", re.I), "administrative"),
    (re.compile(r"\badmi\b", re.I), "admin"),
    (re.compile(r"\band/\s*or\b", re.I), "and/or"),
)
_BODY_VERB = re.compile(
    r"\b(Deploy|Use|Enable|Configure|Enforce|Require|Establish|Perform|"
    r"Conduct|Prevent|Delete|Disable|Apply|Collect|Maintain|Limit|Block|"
    r"Filter|Lock|Review|Monitor|Implement|Ensure|Restrict|Centralize|"
    r"Designate|Assign|Determine|Remediate|Validate)\s+[a-z]",
)


def _strip_cis_meta(text: str) -> str:
    out = text or ""
    for pat in _META_RES:
        out = pat.sub(" ", out)
    return _normalize_ws(_DOC_NOISE.sub(" ", out))


def _first_sentences(text: str, n: int = 2, max_words: int = 40) -> str:
    parts = re.split(r"(?<=[.!?])\s+", _normalize_ws(text))
    keep: list[str] = []
    words = 0
    for part in parts:
        if len(part.split()) < 6:
            continue
        keep.append(part.rstrip(" ."))
        words += len(part.split())
        if len(keep) >= n or words >= max_words:
            break
    return ". ".join(keep)


def _fix_ocr(text: str) -> str:
    out = text or ""
    for pat, repl in _OCR_FIX:
        out = pat.sub(repl, out)
    return out


def _strip_title_echo(text: str) -> str:
    """Drop 'Collect DNS Query Audit Logs Collect DNS query audit logs' heading echo."""
    words = (text or "").split()
    n = len(words)
    for length in range(min(12, n // 2), 2, -1):
        head = [w.lower().rstrip(".,:;") for w in words[:length]]
        rest = [w.lower().rstrip(".,:;") for w in words[length : length + length]]
        if head and head == rest:
            return " ".join(words[length:])
    echo = re.match(
        r"^((?:[A-Z][\w'’/-]+)(?:\s+[A-Z][\w'’/-]+){1,8})\s+([A-Z][a-z].+)$",
        text or "",
    )
    if echo and echo.group(1).split()[0].lower() == echo.group(2).split()[0].lower().rstrip(","):
        return echo.group(2)
    return text


def _safeguard_description(raw: str) -> str:
    """Prefer the imperative body, not the Title-Case heading."""
    raw = _fix_ocr(_strip_cis_meta(raw))
    bodies = list(_BODY_VERB.finditer(raw))
    if bodies:
        rest = raw[bodies[-1].start() :]
    else:
        match = _IMPERATIVE.search(raw)
        rest = raw[match.start() :] if match else raw
    return _first_sentences(_strip_title_echo(rest), n=1, max_words=36)


def _control_themes(texts: list[str], true_label: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    skip = ("index", "contents", "acknowledgment", "license", "version", "appendix")
    allow = _CLASS_CONTROLS.get(true_label)
    for text in texts:
        for match in _CONTROL_TITLE.finditer(text):
            num = int(match.group(1))
            if allow and num not in allow:
                continue
            title = _normalize_ws(match.group(2)).rstrip(" .")
            key = title.lower()
            if key in seen or len(title) < 8 or any(s in key for s in skip):
                continue
            seen.add(key)
            out.append(title)
            if len(out) >= 3:
                return out
    return out


def _sent_score(sent: str, needles: list[str], extras: tuple[str, ...]) -> int:
    low = sent.lower()
    return sum(2 for n in needles if n in low) + sum(1 for e in extras if e in low)


def _chunk_measures(
    texts: list[str],
    needles: tuple[str, ...] | list[str],
    true_label: str,
) -> list[str]:
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    junk = ("creative commons", "acknowledgments", "all rights reserved", "licensed under", "public release")
    needle_l = [str(n).lower() for n in needles]
    extras = _SCORE_EXTRA.get(true_label) or ()

    def add(sent: str) -> None:
        sent = _strip_title_echo(_fix_ocr(_normalize_ws(_DOC_NOISE.sub(" ", sent or "")).strip(" .")))
        sent = re.sub(r"\s*\|\s*", " ", sent)
        sent = re.sub(r"\badmin\s+,", "admin,", sent)
        sent = _normalize_ws(sent)
        if not sent or len(sent.split()) < 7:
            return
        low = sent.lower()
        if any(tok in low for tok in junk):
            return
        if "this safeguard" in low or low.startswith("review and update") or low.startswith("review annually"):
            return
        if re.search(r"\bSafeguard\s+\d", sent, re.I) or re.match(r"Control\s+\d", sent, re.I):
            return
        if sent.rstrip(".").endswith(",") or sent.endswith(",."):
            return
        if re.search(r"\.pdf\b|\bp_[0-9a-f]{8,}", sent, re.I):
            return
        if re.search(r"\b[A-Z][a-z]{2,12}\s+\|\s+", sent):
            return
        key = low[:72]
        if key in seen:
            return
        seen.add(key)
        if not sent.endswith("."):
            sent += "."
        scored.append((_sent_score(sent, needle_l, extras), sent))

    for text in texts:
        cleaned = _strip_cis_meta(text)
        for match in _SAFEGUARD_SPAN.finditer(cleaned):
            desc = _safeguard_description(match.group(2))
            if desc:
                add(desc)
        for match in _OVERVIEW.finditer(text):
            add(_first_sentences(_strip_cis_meta(match.group(1)), n=2, max_words=42))
        if len(scored) >= 24:
            break
    if sum(1 for n, _s in scored if n > 0) < 5:
        for text in texts:
            cleaned = _strip_cis_meta(text)
            for sent in re.split(r"(?<=[.!?])\s+", cleaned):
                low = sent.lower()
                if any(n in low for n in needle_l) or any(e in low for e in extras):
                    add(_first_sentences(sent, n=1, max_words=36))
            if sum(1 for n, _s in scored if n > 0) >= 8:
                break
    scored.sort(key=lambda x: -x[0])
    positive = [s for n, s in scored if n > 0]
    if len(positive) >= 2:
        return positive[:8]
    return [s for _n, s in scored[:8]]


def _join_themes(themes: list[str]) -> str:
    if not themes:
        return "matching defensive controls"
    if len(themes) == 1:
        return themes[0]
    if len(themes) == 2:
        return f"{themes[0]} and {themes[1]}"
    return f"{', '.join(themes[:-1])}, and {themes[-1]}"


def _chunks_summary(
    true_label: str,
    primary: str,
    tier: str,
    chunks: list[dict],
    needles: tuple[str, ...] | list[str],
) -> str:
    """Semantic digest of retrieved parent texts. No document names. BERTScore reference."""
    phrase = _LABEL_PHRASE.get(true_label, true_label.lower())
    if not chunks:
        return (
            f"No retrieved guidance matched this {phrase} detection. "
            f"Do '{primary}' at {tier}."
        )
    texts = [c.get("text") or "" for c in chunks if not _is_junk_parent(c.get("text") or "")]
    if not texts:
        texts = [c.get("text") or "" for c in chunks]
    themes = _control_themes(texts, true_label)
    measures = _chunk_measures(texts, needles, true_label)
    lead = (
        f"For this {phrase} detection, retrieved guidance covers {_join_themes(themes)}."
    )
    body = " ".join(measures[:8])
    closer = f"Taken together, these support '{primary}' at {tier}."
    core = _normalize_ws(f"{lead} {body}")
    words = core.split()
    if len(words) > 190:
        core = " ".join(words[:190]).rstrip(".,") + "."
    return _normalize_ws(f"{core} {closer}")


def main() -> int:
    _clean_task_outputs()
    compact_rows, pred_path = _load_detect_rows()
    options = json.loads(OPTIONS.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    parents = json.loads(PARENTS.read_text(encoding="utf-8"))["parents"]
    by_type = load_attack_actions_by_type()
    primary_domains = options.get("primary_domains") or {}

    cases = []
    for row in compact_rows:
        tl = str(row["true_label"]).upper()
        w = by_type[tl]
        primary = _primary(tl, w)
        accept = _acceptable(tl, w, primary)
        unsafe = list(not_allowed_actions_for_type(tl)[:3])
        unsafe = [u for u in unsafe if u not in accept]
        tier = _tier(tl, str(row.get("dominant_domain") or ""), primary_domains)
        picked = _pick_parents(parents, tl, k=20)
        chunks = []
        quotes: list[tuple[dict, str]] = []
        for rec in picked:
            text = rec.get("text") or ""
            chunk = {
                "parent_id": rec.get("parent_id"),
                "source_file": rec.get("source_file"),
                "section_heading": rec.get("section_heading") or "",
                "page_start": rec.get("page_start"),
                "page_end": rec.get("page_end"),
                "text": text,
                "text_hash": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            chunks.append(chunk)
            if len(quotes) < 3:
                quotes.append((chunk, _quote(text, CLASS_PDF[tl]["needles"])))
        rationale, summary, points = _rationale(row, primary, tier, chunks, quotes)
        chunk_sum = _chunks_summary(tl, primary, tier, chunks, CLASS_PDF[tl]["needles"])
        cases.append(
            {
                "case_id": row["case_id"],
                "flow_id": row["flow_id"],
                "split_index": int(row["split_index"]),
                "detection": {
                    "true_label": tl,
                    "predicted_label": row["predicted_label"],
                    "confidence": float(row.get("confidence") or 0),
                    "dominant_domain": row.get("dominant_domain"),
                    "domain_shares": row.get("domain_shares") or {},
                    "shap_features": _shap_list(row),
                },
                "condition": {
                    "true_label": tl,
                    "primary_network_tier": tier,
                },
                "rationale": rationale,
                "rationale_summary": summary,
                "chunks_summary": chunk_sum,
                "atomic_reasoning_points": points,
                "actions": {
                    "primary_action": primary,
                    "acceptable_actions": accept,
                    "unsafe_actions": unsafe,
                },
                "relevant_rag_chunks": chunks,
            }
        )

    hist = dict(sorted(Counter(c["condition"]["true_label"] for c in cases).items()))
    payload = {
        "n": len(cases),
        "generation": {
            "model_name": "pdf-grounded catalog freeze",
            "model_source": "cursor-parent",
            "openai_api_used": False,
            "eval_model_reserved": "gpt-6-luna",
            "gold_grounding": "pdf_parent_quote",
            "date": DATE,
        },
        "split": {
            "dataset_paths": fixture.get("dataset_paths"),
            "dataset_sha256": fixture.get("dataset_sha256"),
            "seed": fixture.get("seed"),
            "split": fixture.get("split"),
            "selected_split_index": [c["split_index"] for c in cases],
            "class_counts_gold": hist,
            "detect_json": str(pred_path),
            "flows_csv": fixture.get("flows_csv"),
            "flows_sha256": fixture.get("flows_sha256"),
        },
        "cases": cases,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    digest = _sha256_bytes(text.encode("utf-8"))

    live_out = LIVE / OUT_NAME
    live_out.write_text(text, encoding="utf-8")
    (LIVE / "README.md").write_text(
        "\n".join(
            [
                "# Gold-100 results",
                "",
                f"Main output: `experiments/gold-100/{OUT_NAME}`. Eval reads this file.",
                "Detect/SHAP files (`predictions_*`) stay here. Do not add extra gold JSON.",
                f"N={len(cases)} hist={hist} empty_rag={sum(1 for c in cases if not c['relevant_rag_chunks'])}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("live", live_out)
    print("n", len(cases), "hist", hist, "sha256", digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
