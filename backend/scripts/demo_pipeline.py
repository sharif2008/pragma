#!/usr/bin/env python3
"""Commented Detect -> Reason -> Apply walkthrough (no blockchain).

Demo-only math (tiny encoder, bag-of-words RAG, rule planner) lives here.
Catalogs, labels, chunking, domains, and Apply gates are imported from
`scripts/`::

    cd backend
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --tamper
    python scripts/demo_pipeline.py --rows 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.env import ATTACK_OPTIONS_JSON
from scripts.vfl import canonical_attack_type, load_pipeline_catalogs
from scripts.rag_chunking import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, RAG_TOP_K, chunk_text
from scripts.apply_gates import action_in_whitelist, decide_apply_gates
from scripts.network_domains import (
    ACCESS_ISP,
    DOMAIN_LABELS,
    DOMAIN_TO_AGENT,
    ENDPOINT_EDR,
    PERIMETER_IDS,
)

DOMAINS = DOMAIN_LABELS
K9 = (
    "BENIGN",
    "DDOS",
    "DOS",
    "SSHPATATOR",
    "FTPPATATOR",
    "PORTSCAN",
    "WEBATTACK",
    "BOT",
    "OTHERS",
)

PROD_INPUT_DIMS = (23, 32, 42)
PROD_HIDDEN = 128
PROD_EMBED = 64
PROD_FUSION = 192
DEMO_PER_DOMAIN = 4
DEMO_HIDDEN = 8
DEMO_EMBED = 4

# Compact feature names (subset of the 97 CIC-style columns).
D1_FEATURES = (
    "bidirectional_packets",
    "bidirectional_bytes",
    "udps.syn_packet_count",
    "bidirectional_duration_ms",
)
D2_FEATURES = (
    "udps.srcdst_unique_ports_count",
    "udps.srcdst_http_ports_count",
    "udps.srcdst_vul_ports_count",
    "src2dst_syn_packets",
)
D3_FEATURES = (
    "dst2src_rst_packets",
    "bidirectional_mean_piat_ms",
    "src2dst_psh_packets",
    "bidirectional_ack_packets",
)
ALL_FEATURES = D1_FEATURES + D2_FEATURES + D3_FEATURES

# Policy snippets stand in for NIST / CIS / ATT&CK PDFs used by RAG Part 1.
POLICY_CORPUS: tuple[tuple[str, str], ...] = (
    (
        "nist-ddos",
        "Rate-limit and scrub volumetric floods at the subscriber Access / ISP edge. "
        "Blackhole or ACL a confirmed source after connection-limit and SYN-cookie controls. "
        "DDOS playbook: limit rate, enable scrubbing, blackhole route, block IP.",
    ),
    (
        "cis-web",
        "Perimeter / IDS must apply WAF rules and virtual patching for WEBATTACK. "
        "Isolate the service and update ACL when HTTP port abuse and PSH bursts are present.",
    ),
    (
        "cis-scan",
        "Port scans show high unique and vulnerable port counts at Perimeter / IDS. "
        "Tarpit the scanner, raise scan threshold, harden ports, then block IP.",
    ),
    (
        "nist-brute",
        "SSH and FTP patator brute force is host-proximate: Endpoint / EDR should "
        "fail2ban block, lock account, throttle credentials, and enforce MFA.",
    ),
    (
        "cis-bot",
        "BOT traffic often uses DNS/HTTP beacons. Challenge with captcha or JS at "
        "the perimeter, then reputation-filter and rate-limit the source.",
    ),
    (
        "nist-benign",
        "BENIGN flows are logged and monitored only. Do not block or isolate without "
        "a matching attack class on the whitelist.",
    ),
)


# =============================================================================
# Small linear algebra (stdlib). Replaces numpy / PyTorch for this walkthrough.
# =============================================================================
def _zeros(n: int) -> list[float]:
    return [0.0] * n


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


def _relu(v: list[float]) -> list[float]:
    return [x if x > 0.0 else 0.0 for x in v]


def _softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _l2(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1e-9


class Linear:
    """y = W x + b. W is [out][in]. Production: nn.Linear."""

    def __init__(self, n_in: int, n_out: int, rng: random.Random) -> None:
        scale = math.sqrt(2.0 / max(n_in, 1))
        self.w = [[rng.gauss(0.0, scale) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out

    def forward(self, x: list[float]) -> list[float]:
        return [_dot(row, x) + bi for row, bi in zip(self.w, self.b)]

    def grad_x_input(self, x: list[float]) -> list[float]:
        """Sum_j |W_ji * x_i| per input -- analogue of gradient x input."""
        out = _zeros(len(x))
        for row in self.w:
            for i, (wi, xi) in enumerate(zip(row, x)):
                out[i] += abs(wi * xi)
        return out


class LocalEncoder:
    """input -> hidden -> embed with ReLU. Production: LocalEncoder  input->128->64."""

    def __init__(self, n_in: int, hidden: int, embed: int, rng: random.Random) -> None:
        self.fc1 = Linear(n_in, hidden, rng)
        self.fc2 = Linear(hidden, embed, rng)

    def forward(self, x: list[float]) -> list[float]:
        h = _relu(self.fc1.forward(x))
        return _relu(self.fc2.forward(h))


# =============================================================================
# Module: synthetic dataset (no CSV / pandas required)
# =============================================================================
def make_row(label: str, rng: random.Random) -> dict[str, float]:
    """One CIC-style numeric row. True-class cues are raised so Detection has signal."""
    row = {name: rng.uniform(0.05, 0.25) for name in ALL_FEATURES}
    if label == "DDOS":
        row["bidirectional_packets"] = rng.uniform(0.85, 1.0)
        row["bidirectional_bytes"] = rng.uniform(0.8, 1.0)
        row["udps.syn_packet_count"] = rng.uniform(0.7, 1.0)
    elif label == "WEBATTACK":
        row["udps.srcdst_http_ports_count"] = rng.uniform(0.8, 1.0)
        row["src2dst_psh_packets"] = rng.uniform(0.7, 1.0)
    elif label == "PORTSCAN":
        row["udps.srcdst_unique_ports_count"] = rng.uniform(0.85, 1.0)
        row["udps.srcdst_vul_ports_count"] = rng.uniform(0.7, 1.0)
        row["src2dst_syn_packets"] = rng.uniform(0.6, 0.95)
    elif label in ("SSHPATATOR", "FTPPATATOR"):
        row["dst2src_rst_packets"] = rng.uniform(0.75, 1.0)
        row["bidirectional_mean_piat_ms"] = rng.uniform(0.05, 0.2)
        row["bidirectional_ack_packets"] = rng.uniform(0.6, 0.95)
    elif label == "BOT":
        row["udps.srcdst_unique_ports_count"] = rng.uniform(0.5, 0.8)
        row["bidirectional_mean_piat_ms"] = rng.uniform(0.4, 0.7)
    elif label == "BENIGN":
        row["bidirectional_duration_ms"] = rng.uniform(0.3, 0.6)
    return row


def split_domains(row: dict[str, float]) -> dict[str, list[float]]:
    """Vertical split: each party sees only its columns (VFL)."""
    return {
        ACCESS_ISP: [float(row[c]) for c in D1_FEATURES],
        PERIMETER_IDS: [float(row[c]) for c in D2_FEATURES],
        ENDPOINT_EDR: [float(row[c]) for c in D3_FEATURES],
    }


# =============================================================================
# Module: Detection (VFL encode + meta + (y_hat, p_star))
# =============================================================================
@dataclass
class DetectionResult:
    y_hat: str
    p_star: float
    probs: dict[str, float]
    embeddings: dict[str, list[float]]
    fusion: list[float]
    party_shap: dict[str, float]
    feature_attr: dict[str, float]


class DetectModule:
    """Three LocalEncoders + fused nearest-centroid head (stdlib stand-in for AgentMetaModel)."""

    def __init__(self, rng: random.Random) -> None:
        self.encoders = {
            ACCESS_ISP: LocalEncoder(DEMO_PER_DOMAIN, DEMO_HIDDEN, DEMO_EMBED, rng),
            PERIMETER_IDS: LocalEncoder(DEMO_PER_DOMAIN, DEMO_HIDDEN, DEMO_EMBED, rng),
            ENDPOINT_EDR: LocalEncoder(DEMO_PER_DOMAIN, DEMO_HIDDEN, DEMO_EMBED, rng),
        }
        self.centroids: dict[str, list[float]] = {}
        self.feat_layers = {
            ACCESS_ISP: self.encoders[ACCESS_ISP].fc1,
            PERIMETER_IDS: self.encoders[PERIMETER_IDS].fc1,
            ENDPOINT_EDR: self.encoders[ENDPOINT_EDR].fc1,
        }

    def encode(self, parts: dict[str, list[float]]) -> tuple[dict[str, list[float]], list[float]]:
        embeds = {d: self.encoders[d].forward(parts[d]) for d in DOMAINS}
        fusion: list[float] = []
        for d in DOMAINS:
            fusion.extend(embeds[d])
        return embeds, fusion

    def fit(self, labeled_rows: list[tuple[dict[str, float], str]]) -> None:
        """Train step: class mean of fused embeddings (demo stand-in for VFL SGD)."""
        buckets: dict[str, list[list[float]]] = {k: [] for k in K9}
        for row, label in labeled_rows:
            _, z = self.encode(split_domains(row))
            buckets[label].append(z)
        dim = DEMO_EMBED * 3
        for k in K9:
            vecs = buckets[k]
            if not vecs:
                self.centroids[k] = _zeros(dim)
                continue
            acc = _zeros(dim)
            for v in vecs:
                acc = _add(acc, v)
            self.centroids[k] = _scale(acc, 1.0 / len(vecs))

    def _cue_logits(self, row: dict[str, float]) -> list[float]:
        """Cheap evidence-cue head so the toy set is separable without PyTorch SGD."""
        cues = {
            "BENIGN": row["bidirectional_duration_ms"] - row["bidirectional_packets"],
            "DDOS": row["bidirectional_packets"] + row["bidirectional_bytes"] + row["udps.syn_packet_count"],
            "DOS": row["udps.syn_packet_count"] + row["src2dst_syn_packets"],
            "SSHPATATOR": row["dst2src_rst_packets"] + row["bidirectional_ack_packets"],
            "FTPPATATOR": row["dst2src_rst_packets"] + row["bidirectional_ack_packets"] * 0.8,
            "PORTSCAN": row["udps.srcdst_unique_ports_count"] + row["udps.srcdst_vul_ports_count"],
            "WEBATTACK": row["udps.srcdst_http_ports_count"] + row["src2dst_psh_packets"],
            "BOT": row["udps.srcdst_unique_ports_count"] * 0.5 + row["bidirectional_mean_piat_ms"],
            "OTHERS": 0.15,
        }
        return [cues[k] for k in K9]

    def predict(self, row: dict[str, float]) -> DetectionResult:
        parts = split_domains(row)
        embeds, fusion = self.encode(parts)
        # Mix fused-embedding distance with cue scores, then sharpen (T<1).
        dist = [-_l2(_add(fusion, _scale(self.centroids[k], -1.0))) for k in K9]
        cues = self._cue_logits(row)
        logits = [d + 2.5 * c for d, c in zip(dist, cues)]
        t = 0.35
        probs = _softmax([x / t for x in logits])
        idx = max(range(len(K9)), key=lambda i: probs[i])
        y_hat = K9[idx]
        p_star = probs[idx]
        # Party SHAP stand-in: L2 of each 64-d (here 4-d) embedding, normalized.
        mags = {d: _l2(embeds[d]) for d in DOMAINS}
        mag_sum = sum(mags.values()) or 1.0
        party_shap = {d: mags[d] / mag_sum for d in DOMAINS}
        feature_attr: dict[str, float] = {}
        for domain, cols in (
            (ACCESS_ISP, D1_FEATURES),
            (PERIMETER_IDS, D2_FEATURES),
            (ENDPOINT_EDR, D3_FEATURES),
        ):
            gx = self.feat_layers[domain].grad_x_input(parts[domain])
            for name, val in zip(cols, gx):
                feature_attr[name] = val
        return DetectionResult(
            y_hat=y_hat,
            p_star=p_star,
            probs={k: p for k, p in zip(K9, probs)},
            embeddings=embeds,
            fusion=fusion,
            party_shap=party_shap,
            feature_attr=feature_attr,
        )


# =============================================================================
# Module: Reasoning (chunk + retrieve + structured plan)
# =============================================================================
def _tokens(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        counts[tok] = counts.get(tok, 0.0) + 1.0
    return counts


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) | set(b)
    num = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    da = math.sqrt(sum(v * v for v in a.values())) or 1e-9
    db = math.sqrt(sum(v * v for v in b.values())) or 1e-9
    return num / (da * db)


@dataclass
class RetrievedChunk:
    doc_id: str
    text: str
    score: float


class RagIndex:
    """In-memory FAISS stand-in: bag-of-words cosine, top-k=5, then overlap re-rank."""

    def __init__(self, docs: tuple[tuple[str, str], ...]) -> None:
        self.chunks: list[tuple[str, str]] = []
        for doc_id, body in docs:
            for i, ch in enumerate(chunk_text(body, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)):
                self.chunks.append((f"{doc_id}#{i}", ch))
        self.vecs = [_tokens(t) for _, t in self.chunks]

    def search(self, query: str, k: int = RAG_TOP_K) -> list[RetrievedChunk]:
        q = _tokens(query)
        scored = [
            RetrievedChunk(doc_id=cid, text=txt, score=_cosine(q, vec))
            for (cid, txt), vec in zip(self.chunks, self.vecs)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[: max(k * 2, k)]
        # Cross-encoder stand-in: boost chunks that mention the predicted label.
        qset = set(q)
        for r in top:
            overlap = len(qset & set(_tokens(r.text)))
            r.score = 0.7 * r.score + 0.3 * (overlap / (1.0 + len(qset)))
        top.sort(key=lambda r: r.score, reverse=True)
        return top[:k]


@dataclass
class ActionItem:
    action: str
    network_tier: str
    role: str  # primary | supporting


@dataclass
class MitigationPlan:
    threat_level: str
    attack_type: str
    all_actions: list[str]
    items: list[ActionItem]
    rag_ids: list[str]
    digest: str = ""


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def plan_digest(plan: MitigationPlan) -> str:
    """Local SHA-256 of canonical plan P. Production also anchors only this digest."""
    payload = {
        "attack_type": plan.attack_type,
        "threat_level": plan.threat_level,
        "items": [{"action": i.action, "network_tier": i.network_tier, "role": i.role} for i in plan.items],
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def build_plan(
    det: DetectionResult,
    rag: list[RetrievedChunk],
    whitelist: dict[str, list[str]],
    caps: dict[str, list[str]],
    primary: dict[str, list[str]],
) -> MitigationPlan:
    """Whitelist planner (GPT-4o-mini stand-in). Never invents actions outside W[a]."""
    attack = det.y_hat
    allowed = [a.lower() for a in whitelist.get(attack, [])]
    preferred_domains = primary.get(attack, list(DOMAINS))
    # Dominant domain from party SHAP.
    dominant = max(det.party_shap, key=det.party_shap.get)
    if dominant not in preferred_domains:
        dominant = preferred_domains[0]

    picked: list[ActionItem] = []
    used: set[str] = set()

    def take(domain: str, role: str) -> None:
        domain_caps = {c.lower() for c in caps.get(domain, [])}
        for name in whitelist.get(attack, []):
            key = name.lower()
            if key in used or key not in domain_caps or key not in set(allowed):
                continue
            used.add(key)
            picked.append(ActionItem(action=name, network_tier=domain, role=role))
            return

    take(dominant, "primary")
    for d in preferred_domains:
        if len(picked) >= 3:
            break
        if d == dominant:
            continue
        take(d, "supporting")
    if not picked:
        # BENIGN / empty intersection -> log only if whitelisted.
        take(ACCESS_ISP, "primary")

    if det.p_star >= 0.75:
        level = "Critical" if attack in ("DDOS", "WEBATTACK") else "High"
    elif det.p_star >= 0.55:
        level = "Medium"
    else:
        level = "Low"

    plan = MitigationPlan(
        threat_level=level,
        attack_type=attack,
        all_actions=[i.action for i in picked],
        items=picked,
        rag_ids=[c.doc_id for c in rag],
    )
    plan.digest = plan_digest(plan)
    return plan


# =============================================================================
# Module: Apply (software gates only -- no chain)
# =============================================================================
@dataclass
class GateResult:
    action: str
    network_tier: str
    result: str
    failure_reason: str | None = None
    whitelisted: bool | None = None


@dataclass
class ApplyReport:
    status: str
    human_hold: bool
    digest_ok: bool
    items: list[GateResult] = field(default_factory=list)


def apply_plan(
    plan: MitigationPlan,
    whitelist: dict[str, list[str]],
    *,
    tamper: bool = False,
    rng: random.Random | None = None,
    confidence_hold: float = 0.0,
    p_star: float = 1.0,
) -> ApplyReport:
    """Event-listener + execution gate. Order matches agent_service: whitelist then plan-bind.

    Production then calls applyAction on-chain. Here we stop after the two software
    gates plus a local digest check (hash of P, not an RPC).
    """
    rng = rng or random.Random(0)
    if p_star < confidence_hold:
        return ApplyReport(status="held_for_human", human_hold=True, digest_ok=True, items=[])

    expected = plan_digest(plan)
    digest_ok = expected == plan.digest

    # Working copy of units (may be rewritten by --tamper).
    units = [{"action": i.action, "planned_action": i.action, "network_tier": i.network_tier} for i in plan.items]
    if tamper and units:
        pool = [a for acts in whitelist.values() for a in acts]
        off = [a for a in pool if a.lower() not in {x.lower() for x in whitelist.get(plan.attack_type, [])}]
        i0 = 0
        if off:
            units[i0]["action"] = rng.choice(off)  # off-whitelist -> Gate 1 fail
        if len(units) > 1:
            same = [a for a in whitelist.get(plan.attack_type, []) if a.lower() != units[1]["planned_action"].lower()]
            if same:
                units[1]["action"] = rng.choice(same)  # on-list but not in P -> Gate 2 fail

    items: list[GateResult] = []
    for u in units:
        action = u["action"]
        planned = u["planned_action"]
        gr = GateResult(action=action, network_tier=u["network_tier"], result="skipped")
        allowed = action_in_whitelist(plan.attack_type, action, whitelist)
        decision = decide_apply_gates(
            attack_type=plan.attack_type,
            action=action,
            planned_action=planned,
            whitelist_allowed=allowed,
            integrity_valid=digest_ok,
        )
        gr.whitelisted = decision.whitelisted
        gr.result = decision.result
        gr.failure_reason = decision.failure_reason
        items.append(gr)

    if any(x.result != "success" for x in items):
        status = "failed"
    elif items:
        status = "applied"
    else:
        status = "empty_plan"
    return ApplyReport(status=status, human_hold=False, digest_ok=digest_ok, items=items)


# =============================================================================
# Module: Evaluate (accuracy / macro-F1 on the toy set)
# =============================================================================
def confusion(y_true: list[str], y_pred: list[str]) -> dict[str, dict[str, int]]:
    labels = sorted(set(y_true) | set(y_pred))
    table = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(y_true, y_pred):
        table[t][p] += 1
    return table


def f1_scores(y_true: list[str], y_pred: list[str]) -> tuple[float, float, dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    per: dict[str, float] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        per[lab] = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)
    macro = sum(per.values()) / max(len(per), 1)
    return acc, macro, per


# =============================================================================
# CLI / step-by-step runner
# =============================================================================
def _banner(step: str, title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {step}  {title}")
    print("=" * 78)


def _fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def run(seed: int, n_rows: int, tamper: bool, hold_below: float) -> int:
    rng = random.Random(seed)
    whitelist, caps, primary = load_pipeline_catalogs()
    keys = frozenset(whitelist)

    _banner("0/7", "Catalogs (shared vfl + network_domains)")
    print(f"  attack classes K={len(whitelist)}: {', '.join(sorted(whitelist))}")
    print(f"  domains: {', '.join(f'{DOMAIN_TO_AGENT[d]}={d}' for d in DOMAINS)}")
    print(f"  catalog source: {ATTACK_OPTIONS_JSON}")
    print(f"  production VFL dims {PROD_INPUT_DIMS} -> {PROD_HIDDEN} -> {PROD_EMBED} (fusion {PROD_FUSION})")
    print(f"  this demo encoder: {DEMO_PER_DOMAIN}x3 -> {DEMO_HIDDEN} -> {DEMO_EMBED} (fusion {DEMO_EMBED * 3})")

    _banner("1/7", "Detection train (synthetic rows, nearest-centroid on fused embeds)")
    train_labels = list(K9) * 3
    rng.shuffle(train_labels)
    train_set = [(make_row(lab, rng), lab) for lab in train_labels]
    detect = DetectModule(rng)
    detect.fit(train_set)
    print(f"  fitted centroids for {len(detect.centroids)} classes on {len(train_set)} rows")

    _banner("2/7", "RAG index (chunk 512 / overlap 64, bag-of-words, top-k=5)")
    index = RagIndex(POLICY_CORPUS)
    print(f"  chunks indexed: {len(index.chunks)}  (production: FAISS + MiniLM over RAG_docs/)")

    labels_demo = ["DDOS", "WEBATTACK", "PORTSCAN", "SSHPATATOR", "BENIGN", "BOT"]
    labels_demo = labels_demo[: max(1, n_rows)]
    y_true: list[str] = []
    y_pred: list[str] = []

    for n, raw_label in enumerate(labels_demo, start=1):
        truth = canonical_attack_type(raw_label, attack_keys=keys)
        row = make_row(truth, rng)

        _banner(f"3/7  row {n}/{len(labels_demo)}", f"Detection predict  truth={truth}")
        det = detect.predict(row)
        y_true.append(truth)
        y_pred.append(det.y_hat)
        print(f"  (y_hat, p_star) = ({det.y_hat}, {det.p_star:.3f})")
        print("  party SHAP shares:")
        for d in DOMAINS:
            print(f"    {DOMAIN_TO_AGENT[d]:<2} {d:<18} {_fmt_pct(det.party_shap[d])}")
        top_feat = sorted(det.feature_attr.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print("  top |W x| features:")
        for name, val in top_feat:
            print(f"    {name:<36} {val:.4f}")

        _banner(f"4/7  row {n}", "Reasoning  retrieve + whitelist plan")
        query = (
            f"{det.y_hat} mitigation {' '.join(DOMAINS)} "
            + " ".join(name for name, _ in top_feat[:3])
        )
        hits = index.search(query, k=RAG_TOP_K)
        for h in hits:
            print(f"    {h.score:.3f}  {h.doc_id:<16}  {h.text[:88]}...")
        plan = build_plan(det, hits, whitelist, caps, primary)
        print(f"  threat_level={plan.threat_level}  digest={plan.digest[:16]}...")
        for it in plan.items:
            print(f"    [{it.role:<10}] {it.action:<22} @ {it.network_tier}")

        _banner(f"5/7  row {n}", "Apply  Gate1 whitelist then Gate2 plan-binding (no chain)")
        report = apply_plan(
            plan,
            whitelist,
            tamper=tamper,
            rng=rng,
            p_star=det.p_star,
            confidence_hold=hold_below,
        )
        print(f"  status={report.status}  human_hold={report.human_hold}  digest_ok={report.digest_ok}")
        for g in report.items:
            extra = f"  ({g.failure_reason})" if g.failure_reason else ""
            print(f"    {g.result:<8} wl={g.whitelisted!s:<5} {g.action} @ {g.network_tier}{extra}")

    _banner("6/7", "Evaluate  accuracy / macro-F1 on this demo batch")
    acc, macro, per = f1_scores(y_true, y_pred)
    print(f"  y_true={y_true}")
    print(f"  y_pred={y_pred}")
    print(f"  accuracy={acc:.3f}  macro-F1={macro:.3f}")
    for lab, f1 in per.items():
        print(f"    F1[{lab}]={f1:.3f}")

    _banner("7/7", "Not in this file (live stack)")
    print("  Blockchain Commit/Apply : hardhat-blockchain + trust_chain_service.py")
    print("  Live HTTP E2E           : python run/attack_monitor.py")
    print("  Stage launcher          : python scripts/pipeline.py --list")
    print()
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Stdlib Detect-Reason-Apply walkthrough (no blockchain).")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rows", type=int, default=6, help="How many demo flows to run (max 6 classes).")
    p.add_argument(
        "--tamper",
        action="store_true",
        help="Rewrite 1-2 planned units per row (off-whitelist + plan-drift), same idea as attack_monitor --tamper.",
    )
    p.add_argument(
        "--hold-below",
        type=float,
        default=0.0,
        help="If p_star is below this, skip Apply and mark held_for_human (default 0 = never hold).",
    )
    args = p.parse_args()
    raise SystemExit(run(seed=args.seed, n_rows=args.rows, tamper=args.tamper, hold_below=args.hold_below))


if __name__ == "__main__":
    main()
