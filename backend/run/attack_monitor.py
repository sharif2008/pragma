#!/usr/bin/env python3
"""
Batch console demo: read a sample CSV, run prediction + RAG + agentic planning row-by-row,
record each row's action set, then apply detection actions (bulk by default).

Apply uses the backend API, which:
  1. Verifies trust-anchor / on-chain commitment for the report
  2. Checks each action against the smart-contract whitelist (attack_type + action label)
  3. Checks the unit against the committed plan (plan-binding)
  4. Calls applyAction on-chain only when both gates pass
  5. Persists results in agentic execution reports (actions_chain_json)

Usage (from backend/, API + trained model + Hardhat chain recommended):
  python run/attack_monitor.py
  python run/attack_monitor.py --input-file run/data/sample_1000.csv --max-rows 10
  python run/attack_monitor.py --no-apply
  python run/attack_monitor.py --apply-mode per-action
  python run/attack_monitor.py --tamper          # mix off-whitelist + plan-drift rewrites
  python run/attack_monitor.py --no-tamper       # default: apply planned actions as-is

Each run writes one folder under run/output/ with report.json, report.txt, report.html,
and ledger.json (enforcement + chain store/verify latency in the reports).

See run/README.md for full instructions.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

# REVIEW: This script is run/attack_monitor.py; data and output stay under run/.
RUN_DIR = Path(__file__).resolve().parent
BACKEND_DIR = RUN_DIR.parent

try:
    from app.notebook_runtime.vfl_utils import (
        canonical_attack_type,
        load_attack_option_keys,
        pick_allowed_alternate_action,
        pick_disallowed_action,
    )
except ImportError:
    sys.path.insert(0, str(BACKEND_DIR))
    from app.notebook_runtime.vfl_utils import (
        canonical_attack_type,
        load_attack_option_keys,
        pick_allowed_alternate_action,
        pick_disallowed_action,
    )

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "partial", "needs_input"})

PROJECT_DIR = BACKEND_DIR.parent
DEFAULT_DATA_DIR = RUN_DIR / "data"
DEFAULT_OUTPUT_DIR = RUN_DIR / "output"
DEFAULT_INPUT_FILE = DEFAULT_DATA_DIR / "sample.csv"


@dataclass
class PlannedAction:
    index: int
    action: str
    network_tier: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RowPipelineRecord:
    file_row: int
    run_id: str | None = None
    trace_id: str | None = None
    pipeline_status: str | None = None
    predicted_label_raw: str | None = None
    predicted_label: str | None = None
    attack_type: str | None = None
    flagged: bool | None = None
    recommended_action: str | None = None
    agentic_job_public_id: str | None = None
    report_public_id: str | None = None
    planned_actions: list[PlannedAction] = field(default_factory=list)
    action_overrides: dict[int, str] = field(default_factory=dict)
    # REVIEW: Per-index off_whitelist | plan_drift plus chain store/verify ms for reports.
    tamper_kinds: dict[int, str] = field(default_factory=dict)
    pipeline_error: str | None = None
    apply_results: list[dict[str, Any]] = field(default_factory=list)
    execution_report_id: int | None = None
    execution_status: str | None = None
    anchor_ms: float | None = None
    verify_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["planned_actions"] = [a.to_dict() if isinstance(a, PlannedAction) else a for a in self.planned_actions]
        if self.action_overrides:
            out["action_overrides"] = {str(k): v for k, v in self.action_overrides.items()}
        if self.tamper_kinds:
            out["tamper_kinds"] = {str(k): v for k, v in self.tamper_kinds.items()}
        return out


@dataclass
class BatchRunSummary:
    total_rows: int = 0
    pipeline_failures: int = 0
    total_actions: int = 0
    actions_applied: int = 0
    whitelist_failures: int = 0
    integrity_failures: int = 0
    other_action_failures: int = 0
    rows_without_apply: int = 0
    labels_mapped_to_others: int = 0
    actions_modified_before_apply: int = 0
    tamper_rejected_on_chain: int = 0
    tamper_accepted_on_chain: int = 0

    @property
    def failed_actions(self) -> int:
        return self.whitelist_failures + self.integrity_failures + self.other_action_failures

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_report_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_actions": self.total_actions,
            "failed_actions": self.failed_actions,
            "whitelist_fail_actions": self.whitelist_failures,
            "actions_applied": self.actions_applied,
            "pipeline_failures": self.pipeline_failures,
            "integrity_fail_actions": self.integrity_failures,
            "other_fail_actions": self.other_action_failures,
            "rows_without_apply": self.rows_without_apply,
            "labels_mapped_to_others": self.labels_mapped_to_others,
            "actions_modified_before_apply": self.actions_modified_before_apply,
            "tamper_rejected_on_chain": self.tamper_rejected_on_chain,
            "tamper_accepted_on_chain": self.tamper_accepted_on_chain,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:8]}"


def create_run_output_dir(run_id: str, *, parent: Path | None = None) -> Path:
    base = parent or DEFAULT_OUTPUT_DIR
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_row_attack_label(
    record: RowPipelineRecord,
    raw_label: str | None,
    *,
    attack_keys: frozenset[str],
) -> None:
    """Map raw prediction to one attack_options.json key (whitelist / chain parity)."""
    if not raw_label or not str(raw_label).strip():
        record.predicted_label_raw = None
        record.predicted_label = None
        record.attack_type = None
        return
    record.predicted_label_raw = str(raw_label).strip()
    attack_type = canonical_attack_type(record.predicted_label_raw, attack_keys=attack_keys)
    record.attack_type = attack_type
    record.predicted_label = attack_type


def ping_api_health(client: httpx.Client) -> None:
    response = client.get("/health")
    response.raise_for_status()
    payload = response.json()
    print(f"API health: {payload.get('status', payload)}")


def load_csv_feature_rows(
    path: Path,
    *,
    start_row: int,
    max_rows: int | None,
) -> list[tuple[int, str]]:
    """
    Return (file_row_index, values_csv) for each data row.
    Strips a trailing `label` column when the header ends with label.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows_out: list[tuple[int, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        drop_label = bool(header and str(header[-1]).strip().lower() == "label")

        for file_row, raw in enumerate(reader):
            if file_row < start_row:
                continue
            if max_rows is not None and len(rows_out) >= max_rows:
                break
            if not raw or all(not str(c).strip() for c in raw):
                continue
            values = raw[:-1] if drop_label and len(raw) > 1 else raw
            rows_out.append((file_row, ",".join(str(v).strip() for v in values)))
    return rows_out


def post_network_row_simple(
    client: httpx.Client,
    *,
    values_csv: str,
    idempotency_key: str,
    simulate: dict[str, Any],
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/simulate/network-row/simple",
        json={"values_csv": values_csv, "metadata": {}, "simulate": simulate},
        headers={"Idempotency-Key": idempotency_key},
    )
    response.raise_for_status()
    return response.json()


def fetch_run_status(client: httpx.Client, run_id: str) -> dict[str, Any]:
    response = client.get(f"/api/v1/runs/{run_id}")
    response.raise_for_status()
    return response.json()


def fetch_agent_report(client: httpx.Client, public_id: str) -> dict[str, Any]:
    response = client.get(f"/agent/reports/{public_id}")
    response.raise_for_status()
    return response.json()


def list_reports_for_job(client: httpx.Client, agentic_job_public_id: str) -> list[dict[str, Any]]:
    response = client.get(
        "/agent/reports",
        params={"agentic_job_public_id": agentic_job_public_id, "limit": 5},
    )
    response.raise_for_status()
    data = response.json()
    return data if isinstance(data, list) else []


def _actions_payload_from_run_or_response(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize actions/final_actions from simulate or GET /runs responses."""
    # REVIEW: Keep return inside the list branch so a missing `actions` key
    # falls through to `final_actions` instead of UnboundLocalError (`first`).
    for key in ("actions", "final_actions"):
        raw = payload.get(key)
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list) and raw:
            first = raw[0]
            return first if isinstance(first, dict) else None
    return None


def _predictions_payload_from_run(run: dict[str, Any]) -> dict[str, Any]:
    preds = run.get("predictions")
    if isinstance(preds, dict):
        return preds
    payload = run.get("predictions_payload")
    return payload if isinstance(payload, dict) else {}


def extract_pipeline_action_entry(response: dict[str, Any], run: dict[str, Any]) -> dict[str, Any] | None:
    entry = _actions_payload_from_run_or_response(response)
    if entry is None:
        entry = _actions_payload_from_run_or_response(run)
    if entry is not None:
        return entry

    preds = _predictions_payload_from_run(run)
    if preds:
        return {
            "predicted_label": preds.get("predicted_label"),
            "flagged_attack_or_anomaly": preds.get("flagged_attack_or_anomaly"),
            "max_class_probability": preds.get("max_class_probability"),
        }
    return None


def resolve_report_public_id(
    client: httpx.Client,
    pipeline_entry: dict[str, Any] | None,
    *,
    run: dict[str, Any] | None = None,
) -> str | None:
    if not pipeline_entry:
        pipeline_entry = _actions_payload_from_run_or_response(run or {})
    if not pipeline_entry:
        return None
    job_id = str(pipeline_entry.get("agentic_job_public_id") or "").strip()
    if not job_id:
        return None
    reports = list_reports_for_job(client, job_id)
    if reports:
        public_id = reports[0].get("public_id")
        return str(public_id) if public_id else None
    return None


def parse_structured_plan_from_report(report: dict[str, Any]) -> dict[str, Any] | None:
    art = report.get("report_artifact")
    if isinstance(art, dict):
        sp = art.get("structured_plan")
        if isinstance(sp, dict):
            return sp
    raw = report.get("raw_llm_response")
    if isinstance(raw, str):
        import re

        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    return None


def planned_actions_from_report(report: dict[str, Any]) -> list[PlannedAction]:
    plan = parse_structured_plan_from_report(report)
    out: list[PlannedAction] = []
    if plan:
        idx = 0
        for key in ("primary_actions", "supporting_actions"):
            block = plan.get(key)
            if not isinstance(block, list):
                continue
            for item in block:
                if not isinstance(item, dict):
                    continue
                out.append(
                    PlannedAction(
                        index=idx,
                        action=str(item.get("action") or "—"),
                        network_tier=str(item.get("network_tier") or ""),
                    )
                )
                idx += 1
    if out:
        return out
    fallback = str(report.get("recommended_action") or "").strip()
    if fallback:
        return [PlannedAction(index=0, action=fallback, network_tier="")]
    return []


def auto_tamper_overrides(
    planned: list[PlannedAction],
    attack_type: str | None,
    *,
    file_row: int = 0,
    tamper_seed: int = 42,
) -> tuple[dict[int, str], dict[int, str]]:
    """Rewrite 1–2 planned units: mix off-whitelist vs whitelist-legal plan drift."""
    # REVIEW: Dual-gate tamper — threat A fails Gate 1, threat B fails Gate 2.
    if not planned:
        return {}, {}
    rng = random.Random((int(tamper_seed) << 16) ^ int(file_row))
    tamper_count = min(len(planned), 2, 1 + (file_row % 2))
    used: set[str] = set()
    overrides: dict[int, str] = {}
    kinds: dict[int, str] = {}
    for act in planned[:tamper_count]:
        kind = "off_whitelist" if rng.random() < 0.5 else "plan_drift"
        if kind == "off_whitelist":
            alt = pick_disallowed_action(attack_type, exclude=frozenset(used), rng=rng)
            if not alt:
                alt = pick_allowed_alternate_action(
                    act.action, attack_type, exclude=frozenset(used), rng=rng
                )
                kind = "plan_drift" if alt else kind
        else:
            alt = pick_allowed_alternate_action(
                act.action, attack_type, exclude=frozenset(used), rng=rng
            )
            if not alt:
                alt = pick_disallowed_action(attack_type, exclude=frozenset(used), rng=rng)
                kind = "off_whitelist" if alt else kind
        if not alt:
            continue
        if alt.strip().lower() == act.action.strip().lower():
            continue
        overrides[act.index] = alt
        kinds[act.index] = kind
        used.add(alt)
    return overrides, kinds


def chain_item_was_tampered(item: dict[str, Any]) -> bool:
    return bool(item.get("action_modified_before_apply")) or bool(item.get("planned_action"))


def chain_item_tamper_rejected(item: dict[str, Any]) -> bool:
    if not chain_item_was_tampered(item):
        return False
    return str(item.get("result") or "") != "success"


def chain_item_tamper_accepted(item: dict[str, Any]) -> bool:
    if not chain_item_was_tampered(item):
        return False
    return str(item.get("result") or "") == "success"


def apply_all_detection_actions(
    client: httpx.Client,
    report_public_id: str,
    *,
    action_overrides: dict[int, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if action_overrides:
        payload["action_overrides"] = {str(k): v for k, v in action_overrides.items()}
    response = client.post(f"/agent/reports/{report_public_id}/apply", json=payload)
    response.raise_for_status()
    return response.json()


def apply_one_detection_action(
    client: httpx.Client,
    report_public_id: str,
    action_index: int,
    *,
    action_override: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"action_index": action_index}
    if action_override:
        body["action_override"] = action_override
    response = client.post(
        f"/agent/reports/{report_public_id}/apply-action",
        json=body,
    )
    response.raise_for_status()
    return response.json()


def chain_items_from_exec(exec_report: dict[str, Any]) -> list[dict[str, Any]]:
    chain = exec_report.get("actions_chain_json")
    if not isinstance(chain, dict):
        return []
    items = chain.get("items")
    if not isinstance(items, list):
        return []
    return [x for x in items if isinstance(x, dict)]


def _verify_ms_from_exec(exec_report: dict[str, Any]) -> float | None:
    chain = exec_report.get("actions_chain_json")
    if isinstance(chain, dict) and chain.get("verify_ms") is not None:
        try:
            return float(chain["verify_ms"])
        except (TypeError, ValueError):
            return None
    raw = exec_report.get("verify_ms")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def summarize_chain_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": item.get("index"),
        "action": item.get("action"),
        "network_tier": item.get("network_tier"),
        "attack_type": item.get("attack_type"),
        "whitelisted": item.get("whitelisted"),
        "whitelist_error": item.get("whitelist_error"),
        "result": item.get("result"),
        "failure_reason": item.get("failure_reason"),
        "apply_tx_hash": item.get("apply_tx_hash"),
        "apply_error": item.get("apply_error"),
        "planned_action": item.get("planned_action"),
        "action_modified_before_apply": item.get("action_modified_before_apply"),
    }


def chain_items_from_record(record: RowPipelineRecord) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in record.apply_results:
        if block.get("error"):
            continue
        elif block.get("mode") == "bulk":
            exec_report = block.get("exec")
            if isinstance(exec_report, dict):
                for item in chain_items_from_exec(exec_report):
                    items.append(summarize_chain_item(item))
            else:
                for item in block.get("items") or []:
                    if isinstance(item, dict):
                        items.append(item)
        elif block.get("mode") == "per-action":
            item = block.get("item")
            if isinstance(item, dict):
                items.append(item)
    return items


WHITELIST_FAILURE_REASONS = frozenset(
    {"action_not_whitelisted", "whitelist_unavailable", "missing_attack_type"}
)
INTEGRITY_FAILURE_REASONS = frozenset({"integrity_validation_error", "action_plan_mismatch"})


def classify_action_item(item: dict[str, Any]) -> str:
    reason = str(item.get("failure_reason") or "").strip()
    if reason in WHITELIST_FAILURE_REASONS:
        return "whitelist"
    if reason in INTEGRITY_FAILURE_REASONS:
        return "integrity"
    if item.get("whitelisted") is False:
        return "whitelist"
    if item.get("result") == "success":
        return "success"
    if reason:
        return "other"
    if item.get("result") == "failed":
        return "other"
    return "unknown"


def accumulate_tamper_metrics(
    record: RowPipelineRecord,
    chain_items: list[dict[str, Any]],
    summary: BatchRunSummary,
) -> None:
    """Count tamper from client overrides + chain results (plan mismatch must not apply)."""
    # REVIEW: After gate reorder, a failed override has result=failed and Gate 1/2 reason.
    overrides = record.action_overrides or {}
    if not overrides:
        for item in chain_items:
            if chain_item_was_tampered(item):
                summary.actions_modified_before_apply += 1
            if chain_item_tamper_rejected(item):
                summary.tamper_rejected_on_chain += 1
            if chain_item_tamper_accepted(item):
                summary.tamper_accepted_on_chain += 1
        return

    for idx, override in overrides.items():
        summary.actions_modified_before_apply += 1
        planned = next((a.action for a in record.planned_actions if a.index == idx), "")
        item = next((x for x in chain_items if x.get("index") == idx), None)
        if not item:
            summary.tamper_rejected_on_chain += 1
            continue

        result = str(item.get("result") or "")
        applied = str(item.get("action") or "").strip().lower()
        planned_norm = str(planned or "").strip().lower()
        override_norm = str(override or "").strip().lower()

        if result != "success":
            summary.tamper_rejected_on_chain += 1
            continue

        if applied == override_norm and override_norm != planned_norm:
            summary.tamper_accepted_on_chain += 1
            continue

        if applied == planned_norm and override_norm != planned_norm:
            # Override sent but anchored plan was applied — treat as blocked.
            summary.tamper_rejected_on_chain += 1
            summary.integrity_failures += 1
            summary.actions_applied = max(0, summary.actions_applied - 1)
            continue

        summary.tamper_rejected_on_chain += 1


def compute_batch_summary(
    ledger: list[RowPipelineRecord],
    *,
    apply_enabled: bool,
    attack_keys: frozenset[str],
) -> BatchRunSummary:
    summary = BatchRunSummary(total_rows=len(ledger))
    for record in ledger:
        if record.pipeline_error or record.pipeline_status != "completed":
            summary.pipeline_failures += 1

        if record.attack_type == "OTHERS" and record.predicted_label_raw:
            simplified = canonical_attack_type(record.predicted_label_raw, attack_keys=attack_keys)
            raw_key = simplified.upper()
            if raw_key == "OTHERS" and record.predicted_label_raw.upper() not in attack_keys:
                summary.labels_mapped_to_others += 1

        planned = len(record.planned_actions)
        chain_items = chain_items_from_record(record)
        action_count = len(chain_items) if chain_items else planned
        summary.total_actions += action_count

        if apply_enabled and planned and not chain_items and not record.pipeline_error:
            summary.rows_without_apply += 1

        for item in chain_items:
            bucket = classify_action_item(item)
            if bucket == "success":
                summary.actions_applied += 1
            elif bucket == "whitelist":
                summary.whitelist_failures += 1
            elif bucket == "integrity":
                summary.integrity_failures += 1
            elif bucket == "other":
                summary.other_action_failures += 1

        accumulate_tamper_metrics(record, chain_items, summary)

        for block in record.apply_results:
            if block.get("error"):
                summary.other_action_failures += 1

    return summary


def format_report_text(
    *,
    run_id: str,
    input_path: Path,
    summary: BatchRunSummary,
    apply_enabled: bool,
    ledger: list[RowPipelineRecord] | None = None,
) -> str:
    lines = [
        "Run report",
        "==========",
        f"Run id:                  {run_id}",
        f"Input file:              {input_path}",
        f"Total rows:              {summary.total_rows}",
        f"Total actions:           {summary.total_actions}",
    ]
    if apply_enabled:
        lines.extend(
            [
                f"Failed actions:          {summary.failed_actions}",
                f"Whitelist fail actions:  {summary.whitelist_failures}",
                f"Actions applied OK:      {summary.actions_applied}",
                f"Integrity fail actions:  {summary.integrity_failures}",
                f"Other fail actions:      {summary.other_action_failures}",
                f"Actions modified:        {summary.actions_modified_before_apply}",
                f"Tamper rejected (chain): {summary.tamper_rejected_on_chain}",
                f"Tamper accepted (chain): {summary.tamper_accepted_on_chain}",
            ]
        )
    lines.append(f"Pipeline row failures:   {summary.pipeline_failures}")
    if summary.labels_mapped_to_others:
        lines.append(f"Mapped to OTHERS:        {summary.labels_mapped_to_others}  (raw label not in attack_options.json)")
    if ledger:
        lines.extend(["", "By attack class (proposed / changed / WL reject / plan reject / applied / block% of changed)"])
        class_rows = attack_class_report_rows(ledger)
        for row in class_rows:
            lines.append(
                f"  {row['attack']:<12} n={row['n']:<4} prop={row['proposed']:<4} "
                f"chg={row['changed']:<4} wl={row['wl_reject']:<4} plan={row['plan_reject']:<4} "
                f"ok={row['applied']:<4} block%={row['block_pct_changed']:.1f}"
            )
        total = _class_totals_row(class_rows)
        lines.append(
            f"  {'TOTAL':<12} n={total['n']:<4} prop={total['proposed']:<4} "
            f"chg={total['changed']:<4} wl={total['wl_reject']:<4} plan={total['plan_reject']:<4} "
            f"ok={total['applied']:<4} block%={total['block_pct_changed']:.1f}"
        )
        lines.append(
            "  Caption: not-allowed rewrites blocked at whitelist; whitelist-legal drift blocked at plan list."
        )
        latency = build_latency_report(ledger, run_id=run_id)
        store = latency.get("anchor_store") or {}
        verify = latency.get("chain_verify") or {}
        lines.extend(
            [
                "",
                "Chain latency (anchor store + getCommitment verify; not LLM/RAG)",
                f"  Anchor store:  n={store.get('count', 0)}  total_ms={store.get('total_ms', 0)}  mean_ms={store.get('mean_ms', 0)}",
                f"  Chain verify:  n={verify.get('count', 0)}  total_ms={verify.get('total_ms', 0)}  mean_ms={verify.get('mean_ms', 0)}",
                "  Per row: file_row, attack, anchor_ms, verify_ms",
            ]
        )
        for row in latency.get("rows") or []:
            lines.append(
                f"    {row.get('file_row')}  {row.get('attack_type') or '—'}  "
                f"{row.get('anchor_ms')}  {row.get('verify_ms')}"
            )
    return "\n".join(lines) + "\n"


def build_latency_report(ledger: list[RowPipelineRecord], *, run_id: str) -> dict[str, Any]:
    """Per-row and run-level chain store/verify timings (ms), embedded in report.json."""
    # REVIEW: Chain RPC only; folded into report.json/txt/html — no latency.txt.
    rows: list[dict[str, Any]] = []
    anchors: list[float] = []
    verifies: list[float] = []
    for rec in ledger:
        row = {
            "file_row": rec.file_row,
            "attack_type": rec.attack_type,
            "anchor_ms": rec.anchor_ms,
            "verify_ms": rec.verify_ms,
        }
        rows.append(row)
        if rec.anchor_ms is not None:
            anchors.append(float(rec.anchor_ms))
        if rec.verify_ms is not None:
            verifies.append(float(rec.verify_ms))

    def _stats(vals: list[float]) -> dict[str, float | int]:
        if not vals:
            return {"count": 0, "total_ms": 0.0, "mean_ms": 0.0}
        total = sum(vals)
        return {"count": len(vals), "total_ms": round(total, 3), "mean_ms": round(total / len(vals), 3)}

    return {
        "run_id": run_id,
        "generated_at": utc_now_iso(),
        "anchor_store": _stats(anchors),
        "chain_verify": _stats(verifies),
        "rows": rows,
    }


def print_batch_report(
    *,
    run_id: str,
    run_dir: Path,
    input_path: Path,
    summary: BatchRunSummary,
    apply_enabled: bool,
    ledger: list[RowPipelineRecord] | None = None,
) -> None:
    print("\n========== Batch report ==========")
    print(f"Run folder:     {run_dir}")
    print(f"Run id:         {run_id}")
    print(f"Input file:     {input_path}")
    print(f"Total rows:     {summary.total_rows}")
    print(f"Pipeline fail:  {summary.pipeline_failures}")
    print(f"Total actions:  {summary.total_actions}")
    if apply_enabled:
        print(f"Failed actions: {summary.failed_actions}")
        print(f"Whitelist fail: {summary.whitelist_failures}")
        print(f"Applied OK:     {summary.actions_applied}")
        print(f"Integrity fail: {summary.integrity_failures}  (modified / unauthorized plan)")
        print(f"Other fail:     {summary.other_action_failures}")
        if summary.actions_modified_before_apply:
            print(f"Modified:       {summary.actions_modified_before_apply}")
        if summary.tamper_rejected_on_chain:
            print(f"Tamper reject:  {summary.tamper_rejected_on_chain}")
        if summary.tamper_accepted_on_chain:
            print(f"Tamper accept:  {summary.tamper_accepted_on_chain}")
        if summary.rows_without_apply:
            print(f"Rows no apply:  {summary.rows_without_apply}")
    if summary.labels_mapped_to_others:
        print(f"Mapped OTHERS:  {summary.labels_mapped_to_others}  (raw label not in attack_options.json)")
    if ledger:
        print("By class:       attack  n  proposed  changed  WL  plan  applied  block%changed")
        class_rows = attack_class_report_rows(ledger)
        for row in class_rows:
            print(
                f"  {row['attack']:<12} {row['n']:<4} {row['proposed']:<4} {row['changed']:<4} "
                f"{row['wl_reject']:<4} {row['plan_reject']:<4} {row['applied']:<4} {row['block_pct_changed']:.1f}"
            )
        total = _class_totals_row(class_rows)
        print(
            f"  {'TOTAL':<12} {total['n']:<4} {total['proposed']:<4} {total['changed']:<4} "
            f"{total['wl_reject']:<4} {total['plan_reject']:<4} {total['applied']:<4} {total['block_pct_changed']:.1f}"
        )
        latency = build_latency_report(ledger, run_id=run_id)
        store = latency.get("anchor_store") or {}
        verify = latency.get("chain_verify") or {}
        print(
            f"Anchor store:   n={store.get('count', 0)}  total_ms={store.get('total_ms', 0)}  "
            f"mean_ms={store.get('mean_ms', 0)}"
        )
        print(
            f"Chain verify:   n={verify.get('count', 0)}  total_ms={verify.get('total_ms', 0)}  "
            f"mean_ms={verify.get('mean_ms', 0)}"
        )
    print("==================================")


def print_planned_actions(record: RowPipelineRecord) -> None:
    if not record.planned_actions:
        print("    planned actions: (none parsed from report)")
        return
    print(f"    planned actions: {len(record.planned_actions)}")
    for act in record.planned_actions:
        tier = f" [{act.network_tier}]" if act.network_tier else ""
        print(f"      [{act.index}] {act.action}{tier}")


def print_apply_result(exec_report: dict[str, Any], *, action_index: int | None = None) -> None:
    label = f"action_index={action_index}" if action_index is not None else "bulk apply"
    print(
        f"      apply ({label}): status={exec_report.get('status')} "
        f"integrity={exec_report.get('integrity_overall')} attack={exec_report.get('attack_type')}"
    )
    items = chain_items_from_exec(exec_report)
    if not items:
        if exec_report.get("error_reason"):
            print(f"        error: {exec_report.get('error_reason')} — {exec_report.get('error_detail')}")
        return
    for item in items:
        wl = item.get("whitelisted")
        wl_txt = "yes" if wl is True else "no" if wl is False else "—"
        print(
            f"        [{item.get('index')}] {item.get('action')} "
            f"tier={item.get('network_tier') or '—'} whitelisted={wl_txt} result={item.get('result')}"
        )
        if item.get("planned_action"):
            print(f"          planned={item.get('planned_action')} submitted={item.get('action')}")
        if item.get("apply_tx_hash"):
            print(f"          tx={item.get('apply_tx_hash')}")
        reason = item.get("failure_reason") or item.get("whitelist_error") or item.get("apply_error")
        if reason:
            print(f"          detail: {reason}")


def run_pipeline_for_file_row(
    client: httpx.Client,
    *,
    file_row: int,
    values_csv: str,
    idempotency_key: str,
    simulate: dict[str, Any],
    attack_keys: frozenset[str],
) -> RowPipelineRecord:
    record = RowPipelineRecord(file_row=file_row)
    print(f"\n=== Pipeline file_row={file_row} ===")
    try:
        created = post_network_row_simple(
            client,
            values_csv=values_csv,
            idempotency_key=idempotency_key,
            simulate=simulate,
        )
        run_id = str(created.get("run_id") or "")
        record.run_id = run_id
        record.trace_id = str(created.get("trace_id") or "") or None
        run = fetch_run_status(client, run_id) if run_id else {}
        record.pipeline_status = str(run.get("status") or created.get("status") or "")

        entry = extract_pipeline_action_entry(created, run)
        raw_label: str | None = None
        if entry:
            raw_label = str(entry.get("predicted_label") or "") or None
            record.flagged = bool(entry.get("flagged_attack_or_anomaly"))
            record.recommended_action = str(entry.get("recommended_action") or "") or None
            record.agentic_job_public_id = str(entry.get("agentic_job_public_id") or "") or None

        if not raw_label:
            preds = _predictions_payload_from_run(run)
            label = preds.get("predicted_label")
            if label:
                raw_label = str(label)
            if record.flagged is None and preds.get("flagged_attack_or_anomaly") is not None:
                record.flagged = bool(preds.get("flagged_attack_or_anomaly"))

        normalize_row_attack_label(record, raw_label, attack_keys=attack_keys)

        report_id = resolve_report_public_id(client, entry, run=run)
        record.report_public_id = report_id
        if report_id:
            report = fetch_agent_report(client, report_id)
            record.planned_actions = planned_actions_from_report(report)
            ta = report.get("trust_anchor") if isinstance(report, dict) else None
            if isinstance(ta, dict) and ta.get("anchor_ms") is not None:
                try:
                    record.anchor_ms = float(ta["anchor_ms"])
                except (TypeError, ValueError):
                    record.anchor_ms = None

        print(
            f"  run_id={record.run_id} status={record.pipeline_status} "
            f"label={record.predicted_label or '—'} "
            f"(raw={record.predicted_label_raw or '—'}) report={record.report_public_id or '—'}"
        )
        if record.attack_type == "OTHERS" and record.predicted_label_raw:
            print(f"    attack_type mapped to OTHERS (not in attack_options.json): raw={record.predicted_label_raw!r}")
        print_planned_actions(record)
    except Exception as exc:
        record.pipeline_error = str(exc)
        record.pipeline_status = "failed"
        print(f"  pipeline failed: {exc}")
    return record


def apply_actions_for_record(
    client: httpx.Client,
    record: RowPipelineRecord,
    *,
    apply_mode: str,
    pause_s: float,
    tamper: bool,
    tamper_seed: int = 42,
) -> None:
    if not record.report_public_id:
        print("  (skip apply: no agentic report for this row)")
        return
    if not record.planned_actions:
        print("  (skip apply: no planned actions on report)")
        return

    overrides, kinds = (
        auto_tamper_overrides(
            record.planned_actions,
            record.attack_type,
            file_row=record.file_row,
            tamper_seed=tamper_seed,
        )
        if tamper
        else ({}, {})
    )
    record.action_overrides = overrides
    record.tamper_kinds = kinds

    print(f"\n--- Apply file_row={record.file_row} report={record.report_public_id[:8]}… ---")
    if tamper:
        if overrides:
            print(
                f"  tamper on — substituting {len(overrides)} action(s) "
                f"(off-whitelist and/or plan-drift; attack_type={record.attack_type or '—'}):"
            )
            for idx, label in sorted(overrides.items()):
                orig = next((a.action for a in record.planned_actions if a.index == idx), "?")
                kind = kinds.get(idx, "?")
                print(f"    [{idx}] {orig!r} -> {label!r} ({kind})")
        else:
            print("  tamper on — no substitute action found for this plan")

    if apply_mode == "bulk":
        try:
            exec_report = apply_all_detection_actions(
                client,
                record.report_public_id,
                action_overrides=overrides or None,
            )
            record.execution_report_id = exec_report.get("id")
            record.execution_status = str(exec_report.get("status") or "")
            record.verify_ms = _verify_ms_from_exec(exec_report)
            items = [summarize_chain_item(x) for x in chain_items_from_exec(exec_report)]
            record.apply_results.append({"mode": "bulk", "items": items, "exec": exec_report})
            print_apply_result(exec_report)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            record.apply_results.append({"mode": "bulk", "error": detail})
            print(f"  bulk apply HTTP error: {detail}")
        except Exception as exc:
            record.apply_results.append({"mode": "bulk", "error": str(exc)})
            print(f"  bulk apply failed: {exc}")
        return

    for act in record.planned_actions:
        idx = act.index
        override = overrides.get(idx)
        submitted = override or act.action
        print(f"  applying action [{idx}]: {submitted}" + (f" (planned {act.action})" if override else ""))
        try:
            exec_report = apply_one_detection_action(
                client,
                record.report_public_id,
                idx,
                action_override=override,
            )
            record.execution_report_id = exec_report.get("id")
            record.execution_status = str(exec_report.get("status") or "")
            if record.verify_ms is None:
                record.verify_ms = _verify_ms_from_exec(exec_report)
            items = chain_items_from_exec(exec_report)
            picked = next((x for x in items if x.get("index") == idx), items[-1] if items else {})
            summary = summarize_chain_item(picked) if picked else {}
            record.apply_results.append({"mode": "per-action", "action_index": idx, "item": summary, "exec_status": record.execution_status})
            print_apply_result(exec_report, action_index=idx)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            record.apply_results.append({"mode": "per-action", "action_index": idx, "error": detail})
            print(f"    HTTP error: {detail}")
        except Exception as exc:
            record.apply_results.append({"mode": "per-action", "action_index": idx, "error": str(exc)})
            print(f"    failed: {exc}")
        if pause_s > 0:
            time.sleep(pause_s)


@dataclass
class _AttackBucket:
    rows: int = 0
    proposed: int = 0
    changed: int = 0
    wl_reject: int = 0
    plan_reject: int = 0
    applied: int = 0

    @property
    def blocked_changed(self) -> int:
        return self.wl_reject + self.plan_reject


def _report_pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.0"
    return f"{100.0 * count / total:.1f}"


def _row_gate_counts(record: RowPipelineRecord) -> tuple[int, int, int, int, int]:
    """Return (proposed, changed, wl_reject, plan_reject, applied) without double-counting."""
    # REVIEW: Changed units count once as WL or plan reject; unmodified successes are Applied.
    planned = record.planned_actions
    n = len(planned)
    overrides = {int(k): v for k, v in (record.action_overrides or {}).items()}
    items = chain_items_from_record(record)
    changed = wl_reject = plan_reject = applied = 0

    if not record.apply_results:
        return n, len(overrides), 0, 0, n

    for act in planned:
        idx = int(act.index)
        item = next((x for x in items if x.get("index") == idx), None)
        is_changed = idx in overrides
        reason = str((item or {}).get("failure_reason") or "")
        result = str((item or {}).get("result") or "")
        if is_changed:
            changed += 1
            if reason in WHITELIST_FAILURE_REASONS or (item and item.get("whitelisted") is False):
                wl_reject += 1
            elif reason == "action_plan_mismatch" or (
                item and item.get("action_modified_before_apply") and result != "success"
            ):
                plan_reject += 1
            elif result != "success":
                plan_reject += 1
            elif result == "success":
                applied += 1
        elif item and result == "success":
            applied += 1
        elif not item:
            applied += 1

    return n, changed, wl_reject, plan_reject, applied


def _attack_buckets(ledger: list[RowPipelineRecord]) -> dict[str, _AttackBucket]:
    by_attack: dict[str, _AttackBucket] = {}
    for record in ledger:
        atk = str(record.attack_type or "UNKNOWN").upper()
        bucket = by_attack.setdefault(atk, _AttackBucket())
        proposed, changed, wl_reject, plan_reject, applied = _row_gate_counts(record)
        bucket.rows += 1
        bucket.proposed += proposed
        bucket.changed += changed
        bucket.wl_reject += wl_reject
        bucket.plan_reject += plan_reject
        bucket.applied += applied
    return dict(sorted(by_attack.items()))


def attack_class_report_rows(ledger: list[RowPipelineRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atk, bucket in _attack_buckets(ledger).items():
        blocked = bucket.blocked_changed
        rows.append(
            {
                "attack": atk,
                "n": bucket.rows,
                "proposed": bucket.proposed,
                "changed": bucket.changed,
                "wl_reject": bucket.wl_reject,
                "plan_reject": bucket.plan_reject,
                "applied": bucket.applied,
                "block_pct_changed": float(_report_pct(blocked, bucket.changed)) if bucket.changed else 100.0,
            }
        )
    return rows


def _class_totals_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-class gate counts into one TOTAL row."""
    n = proposed = changed = wl_reject = plan_reject = applied = 0
    for row in rows:
        n += int(row.get("n") or 0)
        proposed += int(row.get("proposed") or 0)
        changed += int(row.get("changed") or 0)
        wl_reject += int(row.get("wl_reject") or 0)
        plan_reject += int(row.get("plan_reject") or 0)
        applied += int(row.get("applied") or 0)
    blocked = wl_reject + plan_reject
    return {
        "attack": "TOTAL",
        "n": n,
        "proposed": proposed,
        "changed": changed,
        "wl_reject": wl_reject,
        "plan_reject": plan_reject,
        "applied": applied,
        "block_pct_changed": float(_report_pct(blocked, changed)) if changed else 100.0,
    }


def render_attack_wise_report_html(
    *,
    run_id: str,
    input_file: str,
    apply_mode: str,
    tamper: bool,
    generated_at: str,
    summary: BatchRunSummary,
    ledger: list[RowPipelineRecord],
    latency: dict[str, Any] | None = None,
) -> str:
    esc = html.escape
    by_attack = _attack_buckets(ledger)
    class_rows = attack_class_report_rows(ledger)
    changed_total = sum(b.changed for b in by_attack.values())
    wl_total = sum(b.wl_reject for b in by_attack.values())
    plan_total = sum(b.plan_reject for b in by_attack.values())
    applied_total = sum(b.applied for b in by_attack.values())
    proposed_total = sum(b.proposed for b in by_attack.values())
    blocked_changed_total = wl_total + plan_total

    attack_rows_html: list[str] = []
    for row in class_rows:
        attack_rows_html.append(
            f"""<tr>
  <td>{esc(str(row['attack']))}</td>
  <td class="num">{row['n']}</td>
  <td class="num">{row['proposed']}</td>
  <td class="num">{row['changed']}</td>
  <td class="num">{row['wl_reject']}</td>
  <td class="num">{row['plan_reject']}</td>
  <td class="num">{row['applied']}</td>
  <td class="num">{row['block_pct_changed']:.1f}</td>
</tr>"""
        )

    tamper_note = (
        "Tamper enabled — 1–2 units rewritten per row (mix of off-whitelist and plan-drift). "
        "Block.% of changed is (WL reject + plan reject) / changed (expected 100)."
        if tamper
        else "Tamper disabled — actions applied as planned."
    )

    latency = latency or build_latency_report(ledger, run_id=run_id)
    store = latency.get("anchor_store") or {}
    verify = latency.get("chain_verify") or {}
    latency_rows_html: list[str] = []
    for row in latency.get("rows") or []:
        latency_rows_html.append(
            f"""<tr>
  <td class="num">{esc(str(row.get('file_row')))}</td>
  <td>{esc(str(row.get('attack_type') or '—'))}</td>
  <td class="num">{esc(str(row.get('anchor_ms')))}</td>
  <td class="num">{esc(str(row.get('verify_ms')))}</td>
</tr>"""
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PRAGMA batch report — {esc(run_id)}</title>
  <style>
    :root {{
      --bg: #f8f9fb;
      --card: #fff;
      --text: #1a1a2e;
      --muted: #5c6370;
      --border: #dfe3ea;
      --accent: #0d6e4f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem 1.5rem;
      line-height: 1.5;
    }}
    .wrap {{ max-width: 1100px; margin: 0 auto; }}
    h1 {{ font-size: 1.5rem; margin: 0 0 0.25rem; color: var(--accent); }}
    .sub {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 1.25rem 1.5rem;
      margin-bottom: 1.5rem;
    }}
    h2 {{ font-size: 1.1rem; margin: 0 0 1rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.875rem;
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 0.5rem 0.65rem;
      text-align: left;
    }}
    th {{ background: #eef2f7; font-weight: 600; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    tr.total {{ font-weight: 600; background: #f0f7f4; }}
    .meta {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 0.5rem 1rem; font-size: 0.875rem; }}
    .meta dt {{ color: var(--muted); margin: 0; }}
    .meta dd {{ margin: 0 0 0.5rem; }}
    footer {{ font-size: 0.8rem; color: var(--muted); margin-top: 2rem; }}
    .note {{ font-size: 0.85rem; color: var(--muted); margin-top: 0.75rem; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PRAGMA batch evaluation — research report</h1>
    <p class="sub">Enforcement by attack class — whitelist vs plan-list rejects</p>

    <div class="card">
      <h2>Run metadata</h2>
      <dl class="meta">
        <div><dt>Run ID</dt><dd>{esc(run_id)}</dd></div>
        <div><dt>Input file</dt><dd>{esc(input_file)}</dd></div>
        <div><dt>Apply mode</dt><dd>{esc(apply_mode or "—")}</dd></div>
        <div><dt>Tamper</dt><dd>{"on" if tamper else "off"}</dd></div>
        <div><dt>Generated</dt><dd>{esc(generated_at)}</dd></div>
      </dl>
      <p class="note">{esc(tamper_note)} Per-row detail is in <code>ledger.json</code>.</p>
    </div>

    <div class="card">
      <h2>By attack class</h2>
      <table>
        <thead>
          <tr>
            <th>Attack</th>
            <th>n</th>
            <th>Proposed</th>
            <th>Changed</th>
            <th>WL reject</th>
            <th>Plan reject</th>
            <th>Applied</th>
            <th>Block.% of changed</th>
          </tr>
        </thead>
        <tbody>
          {"".join(attack_rows_html)}
          <tr class="total">
            <td>Total</td>
            <td class="num">{summary.total_rows}</td>
            <td class="num">{proposed_total}</td>
            <td class="num">{changed_total}</td>
            <td class="num">{wl_total}</td>
            <td class="num">{plan_total}</td>
            <td class="num">{applied_total}</td>
            <td class="num">{_report_pct(blocked_changed_total, changed_total)}</td>
          </tr>
        </tbody>
      </table>
      <p class="note">WL reject = Gate 1 (action_not_whitelisted). Plan reject = Gate 2 (action_plan_mismatch). Block.% of changed = (WL + plan) / changed; expected 100% when every rewrite is refused. Caption: all not-allowed rewrites blocked at whitelist; all whitelist-legal drift blocked at plan list.</p>
    </div>

    <div class="card">
      <h2>Chain latency</h2>
      <p class="note">Anchor store = <code>anchor()</code> RPC ms. Verify = <code>getCommitment</code> RPC ms. LLM/RAG times are not included.</p>
      <dl class="meta">
        <div><dt>Anchor store n</dt><dd>{store.get('count', 0)}</dd></div>
        <div><dt>Anchor total ms</dt><dd>{store.get('total_ms', 0)}</dd></div>
        <div><dt>Anchor mean ms</dt><dd>{store.get('mean_ms', 0)}</dd></div>
        <div><dt>Verify n</dt><dd>{verify.get('count', 0)}</dd></div>
        <div><dt>Verify total ms</dt><dd>{verify.get('total_ms', 0)}</dd></div>
        <div><dt>Verify mean ms</dt><dd>{verify.get('mean_ms', 0)}</dd></div>
      </dl>
      <table>
        <thead>
          <tr>
            <th>file_row</th>
            <th>Attack</th>
            <th>anchor_ms</th>
            <th>verify_ms</th>
          </tr>
        </thead>
        <tbody>
          {"".join(latency_rows_html) if latency_rows_html else "<tr><td colspan='4'>No chain timings for this run.</td></tr>"}
        </tbody>
      </table>
    </div>

    <footer>Generated by backend/run/attack_monitor.py — ChainAgentVFL / PRAGMA</footer>
  </div>
</body>
</html>
"""


def write_attack_wise_report_html(
    run_dir: Path,
    ledger: list[RowPipelineRecord],
    meta: dict[str, Any],
    summary: BatchRunSummary,
    *,
    input_path: Path,
    latency: dict[str, Any] | None = None,
) -> Path:
    generated_at = utc_now_iso()
    html_text = render_attack_wise_report_html(
        run_id=str(meta.get("run_id") or run_dir.name),
        input_file=str(input_path),
        apply_mode=str(meta.get("apply_mode") or ""),
        tamper=bool(meta.get("tamper")),
        generated_at=generated_at,
        summary=summary,
        ledger=ledger,
        latency=latency,
    )
    out = run_dir / "report.html"
    out.write_text(html_text, encoding="utf-8")
    return out


def save_run_outputs(
    run_dir: Path,
    ledger: list[RowPipelineRecord],
    meta: dict[str, Any],
    summary: BatchRunSummary,
    *,
    input_path: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    class_rows = attack_class_report_rows(ledger)
    class_report = class_rows + [_class_totals_row(class_rows)]
    latency_payload = build_latency_report(ledger, run_id=str(meta.get("run_id") or run_dir.name))
    report_payload = {
        "generated_at": utc_now_iso(),
        "run_id": meta.get("run_id"),
        "input_file": str(input_path),
        "apply": meta.get("apply"),
        "apply_mode": meta.get("apply_mode"),
        "tamper": meta.get("tamper"),
        "tamper_seed": meta.get("tamper_seed"),
        "by_attack_class": class_report,
        "chain_latency": {
            "anchor_store": latency_payload.get("anchor_store"),
            "chain_verify": latency_payload.get("chain_verify"),
            "rows": latency_payload.get("rows"),
        },
        **summary.to_report_dict(),
    }
    report_json_path = run_dir / "report.json"
    report_json_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_txt_path = run_dir / "report.txt"
    report_txt_path.write_text(
        format_report_text(
            run_id=str(meta.get("run_id") or ""),
            input_path=input_path,
            summary=summary,
            apply_enabled=bool(meta.get("apply")),
            ledger=ledger,
        ),
        encoding="utf-8",
    )

    ledger_path = run_dir / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "generated_at": utc_now_iso(),
                "meta": meta,
                "report": summary.to_report_dict(),
                "by_attack_class": class_report,
                "chain_latency": {
                    "anchor_store": latency_payload.get("anchor_store"),
                    "chain_verify": latency_payload.get("chain_verify"),
                    "rows": latency_payload.get("rows"),
                },
                "rows": [r.to_dict() for r in ledger],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nOutputs written to: {run_dir}")
    print(f"  report.json  — summary totals + by-class gates + chain latency")
    print(f"  report.txt   — summary (human-readable)")
    print(f"  ledger.json  — full batch detail ({len(ledger)} row(s))")
    try:
        write_attack_wise_report_html(
            run_dir, ledger, meta, summary, input_path=input_path, latency=latency_payload
        )
        print(f"  report.html     — attack-wise dual-gate report + chain latency")
    except Exception as exc:
        print(f"  (report.html skipped: {exc})")


def run_attack_monitor(args: argparse.Namespace) -> int:
    input_path = Path(args.input_file).expanduser().resolve()
    run_id = new_run_id()
    simulate = {"latency_ms": max(0, args.latency_ms)}
    if args.force_error_step.strip():
        simulate["force_error_step"] = args.force_error_step.strip()

    output_parent = Path(args.output_dir).expanduser().resolve() if args.output_dir else DEFAULT_OUTPUT_DIR
    run_dir: Path | None = None
    if not args.no_output_json:
        run_dir = create_run_output_dir(run_id, parent=output_parent)

    feature_rows = load_csv_feature_rows(
        input_path,
        start_row=args.start_row,
        max_rows=args.max_rows,
    )
    if not feature_rows:
        print(f"No data rows loaded from {input_path}", file=sys.stderr)
        return 1

    print(f"Run id: {run_id}")
    if run_dir:
        print(f"Output folder: {run_dir}")
    print(f"Input file: {input_path} ({len(feature_rows)} row(s) to process)")

    attack_keys = load_attack_option_keys()
    print(f"Attack types (attack_options.json): {', '.join(sorted(attack_keys))}")
    print(f"Tamper before apply: {'on' if args.tamper else 'off'}")

    ledger: list[RowPipelineRecord] = []
    timeout = httpx.Timeout(args.timeout_s)

    with httpx.Client(base_url=args.base.rstrip("/"), timeout=timeout) as client:
        ping_api_health(client)

        for file_row, values_csv in feature_rows:
            idem = f"{run_id}-row-{file_row}"
            record = run_pipeline_for_file_row(
                client,
                file_row=file_row,
                values_csv=values_csv,
                idempotency_key=idem,
                simulate=simulate,
                attack_keys=attack_keys,
            )
            ledger.append(record)
            if args.pause_between_rows_s > 0:
                time.sleep(args.pause_between_rows_s)

        if args.apply:
            print("\n========== Apply phase (whitelist + on-chain via API) ==========")
            for record in ledger:
                if record.pipeline_error or record.pipeline_status not in TERMINAL_RUN_STATUSES:
                    print(f"\n(skip apply file_row={record.file_row}: pipeline not completed)")
                    continue
                apply_actions_for_record(
                    client,
                    record,
                    apply_mode=args.apply_mode,
                    pause_s=args.pause_between_actions_s,
                    tamper=args.tamper,
                    tamper_seed=int(getattr(args, "tamper_seed", 42)),
                )

    summary = compute_batch_summary(ledger, apply_enabled=args.apply, attack_keys=attack_keys)
    meta = {
        "run_id": run_id,
        "input_file": str(input_path),
        "apply": args.apply,
        "apply_mode": args.apply_mode,
        "tamper": args.tamper,
        "tamper_seed": getattr(args, "tamper_seed", 42),
        "rows_processed": len(ledger),
        "output_dir": str(run_dir) if run_dir else None,
        "attack_option_keys": sorted(attack_keys),
    }
    if run_dir:
        save_run_outputs(run_dir, ledger, meta, summary, input_path=input_path)

    print_batch_report(
        run_id=run_id,
        run_dir=run_dir or output_parent,
        input_path=input_path,
        summary=summary,
        apply_enabled=args.apply,
        ledger=ledger,
    )

    has_failures = summary.pipeline_failures > 0
    if args.apply:
        has_failures = has_failures or (
            summary.whitelist_failures
            + summary.integrity_failures
            + summary.other_action_failures
            + summary.rows_without_apply
        ) > 0
    return 1 if has_failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run network-traffic pipeline row-by-row from a CSV file, then apply actions with on-chain whitelist checks.",
    )
    parser.add_argument(
        "--input-file",
        default=str(DEFAULT_INPUT_FILE),
        help=f"CSV with VFL feature columns (optional label column) (default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL (default: http://127.0.0.1:8000)")
    parser.add_argument(
        "--output-dir",
        default="",
        help=f"Parent folder for run output (default: {DEFAULT_OUTPUT_DIR}/run_{{timestamp}}_{{id}}/)",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Skip first N data rows after header (default: 0)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process at most this many rows (default: all rows)",
    )
    parser.add_argument(
        "--latency-ms",
        type=int,
        default=0,
        help="Simulated ingestion delay per row in ms (default: 0)",
    )
    parser.add_argument(
        "--force-error-step",
        default="",
        help="Inject pipeline failure at step name (default: none)",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=900.0,
        help="HTTP timeout per request in seconds (default: 900)",
    )
    parser.add_argument(
        "--pause-between-rows-s",
        type=float,
        default=0.0,
        help="Sleep between pipeline rows in seconds (default: 0)",
    )
    parser.add_argument(
        "--apply",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After all rows, apply detection actions via API (default: enabled)",
    )
    parser.add_argument(
        "--apply-mode",
        choices=("per-action", "bulk"),
        default="bulk",
        help="bulk=single apply call for all actions per row (default); per-action=POST apply-action per index",
    )
    parser.add_argument(
        "--pause-between-actions-s",
        type=float,
        default=0.25,
        help="Sleep between per-action applies in seconds (default: 0.25; only for --apply-mode per-action)",
    )
    parser.add_argument(
        "--tamper",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            # REVIEW: Mix of off-whitelist (Gate 1) and whitelist-legal plan-drift (Gate 2).
            "Before apply, rewrite 1-2 planned units per row: mix off-whitelist "
            "actions (Gate 1) and whitelist-legal plan drift (Gate 2). Default: off"
        ),
    )
    parser.add_argument(
        "--tamper-seed",
        type=int,
        default=42,
        help="RNG seed mixed with file_row for --tamper rewrites (default: 42)",
    )
    parser.add_argument(
        "--no-output-json",
        action="store_true",
        help="Skip writing output files (default: write run folder under output/)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        code = run_attack_monitor(args)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800] if exc.response is not None else str(exc)
        print(f"HTTP error {exc.response.status_code if exc.response else '?'}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Batch demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
