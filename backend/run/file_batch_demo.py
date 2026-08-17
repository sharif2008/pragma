#!/usr/bin/env python3
"""
Batch console demo: read a sample CSV, run prediction + RAG + agentic planning row-by-row,
record each row's action set, then apply detection actions (bulk by default).

Apply uses the backend API, which:
  1. Verifies trust-anchor / on-chain commitment for the report
  2. Checks each action against the smart-contract whitelist (attack_type + action label)
  3. Calls applyAction on-chain only when whitelisted and integrity is valid
  4. Persists results in agentic execution reports (actions_chain_json)

Usage (from backend/, API + trained model + Hardhat chain recommended):
  python run/file_batch_demo.py
  python run/file_batch_demo.py --input-file ../data/sample_1000.csv --max-rows 10
  python run/file_batch_demo.py --no-apply
  python run/file_batch_demo.py --apply-mode per-action

Each run writes one folder under run/output/ with report.json, report.txt, and ledger.json (all rows inside ledger).

See run/README.md for full instructions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

RUN_DIR = Path(__file__).resolve().parent
BACKEND_DIR = RUN_DIR.parent

try:
    from app.notebook_runtime.vfl_utils import canonical_attack_type, load_attack_option_keys
except ImportError:
    sys.path.insert(0, str(BACKEND_DIR))
    from app.notebook_runtime.vfl_utils import canonical_attack_type, load_attack_option_keys

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
    pipeline_error: str | None = None
    apply_results: list[dict[str, Any]] = field(default_factory=list)
    execution_report_id: int | None = None
    execution_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["planned_actions"] = [a.to_dict() if isinstance(a, PlannedAction) else a for a in self.planned_actions]
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


def apply_all_detection_actions(client: httpx.Client, report_public_id: str) -> dict[str, Any]:
    response = client.post(f"/agent/reports/{report_public_id}/apply")
    response.raise_for_status()
    return response.json()


def apply_one_detection_action(
    client: httpx.Client,
    report_public_id: str,
    action_index: int,
) -> dict[str, Any]:
    response = client.post(
        f"/agent/reports/{report_public_id}/apply-action",
        json={"action_index": action_index},
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
    }


def chain_items_from_record(record: RowPipelineRecord) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for block in record.apply_results:
        if block.get("error"):
            continue
        if block.get("mode") == "bulk":
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
INTEGRITY_FAILURE_REASONS = frozenset({"integrity_validation_error"})


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
            ]
        )
    lines.append(f"Pipeline row failures:   {summary.pipeline_failures}")
    if summary.labels_mapped_to_others:
        lines.append(f"Mapped to OTHERS:        {summary.labels_mapped_to_others}  (raw label not in attack_options.json)")
    return "\n".join(lines) + "\n"


def print_batch_report(
    *,
    run_id: str,
    run_dir: Path,
    input_path: Path,
    summary: BatchRunSummary,
    apply_enabled: bool,
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
        if summary.rows_without_apply:
            print(f"Rows no apply:  {summary.rows_without_apply}")
    if summary.labels_mapped_to_others:
        print(f"Mapped OTHERS:  {summary.labels_mapped_to_others}  (raw label not in attack_options.json)")
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
) -> None:
    if not record.report_public_id:
        print("  (skip apply: no agentic report for this row)")
        return
    if not record.planned_actions:
        print("  (skip apply: no planned actions on report)")
        return

    print(f"\n--- Apply file_row={record.file_row} report={record.report_public_id[:8]}… ---")

    if apply_mode == "bulk":
        try:
            exec_report = apply_all_detection_actions(client, record.report_public_id)
            record.execution_report_id = exec_report.get("id")
            record.execution_status = str(exec_report.get("status") or "")
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
        print(f"  applying action [{idx}]: {act.action}")
        try:
            exec_report = apply_one_detection_action(client, record.report_public_id, idx)
            record.execution_report_id = exec_report.get("id")
            record.execution_status = str(exec_report.get("status") or "")
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


def save_run_outputs(
    run_dir: Path,
    ledger: list[RowPipelineRecord],
    meta: dict[str, Any],
    summary: BatchRunSummary,
    *,
    input_path: Path,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    report_payload = {
        "generated_at": utc_now_iso(),
        "run_id": meta.get("run_id"),
        "input_file": str(input_path),
        "apply": meta.get("apply"),
        "apply_mode": meta.get("apply_mode"),
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
                "rows": [r.to_dict() for r in ledger],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nOutputs written to: {run_dir}")
    print(f"  report.json  — summary totals")
    print(f"  report.txt   — summary (human-readable)")
    print(f"  ledger.json  — full batch detail ({len(ledger)} row(s))")


def run_file_batch_demo(args: argparse.Namespace) -> int:
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
                )

    summary = compute_batch_summary(ledger, apply_enabled=args.apply, attack_keys=attack_keys)
    meta = {
        "run_id": run_id,
        "input_file": str(input_path),
        "apply": args.apply,
        "apply_mode": args.apply_mode,
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
        "--no-output-json",
        action="store_true",
        help="Skip writing output files (default: write run folder under output/)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        code = run_file_batch_demo(args)
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
