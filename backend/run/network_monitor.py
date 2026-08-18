#!/usr/bin/env python3
"""
Console demo: ingest a VFL network-traffic row, run prediction + RAG + agentic planning,
and optionally apply detection actions on-chain.

Requires the API server to be running and at least one trained model in the database.

Usage (from backend/):
  python run/network_monitor.py
  python run/network_monitor.py --mode simple --apply
  python run/attack_monitor.py

See run/README.md for full instructions.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

import httpx

# REVIEW: This script is run/network_monitor.py (renamed from network_monitorpy).

# Full VFL feature row (includes trailing ground-truth label for traffic/header modes).
SAMPLE_ROW_WITH_LABEL = (
    "75,6,425,75,4,305,0,2,120,60,70.83333333333333,17.440374613713626,100,60,76.25,19.73786547054502,100,"
    "60,60,0,60,0,15,20.54263858417414,38,0,25,21.65640782770772,38,0,0,0,0,0,0,0,0,6,2,0,2,0,0,0,0,4,2,0,1,0,0,"
    "0,0,2,0,0,1,20,0,0,10,31,281,844,6,16,0,0,107,269,6,16,1,0,0,10,31,860,909.3548387096774,0,0,0,0,10,361,852,"
    "121,284,BENIGN"
)

SAMPLE_COLUMNS_WITH_LABEL = (
    "bidirectional_duration_ms,bidirectional_packets,bidirectional_bytes,src2dst_duration_ms,src2dst_packets,"
    "src2dst_bytes,dst2src_duration_ms,dst2src_packets,dst2src_bytes,bidirectional_min_ps,bidirectional_mean_ps,"
    "bidirectional_stddev_ps,bidirectional_max_ps,src2dst_min_ps,src2dst_mean_ps,src2dst_stddev_ps,src2dst_max_ps,"
    "dst2src_min_ps,dst2src_mean_ps,dst2src_stddev_ps,dst2src_max_ps,bidirectional_min_piat_ms,"
    "bidirectional_mean_piat_ms,bidirectional_stddev_piat_ms,bidirectional_max_piat_ms,src2dst_min_piat_ms,"
    "src2dst_mean_piat_ms,src2dst_stddev_piat_ms,src2dst_max_piat_ms,dst2src_min_piat_ms,dst2src_mean_piat_ms,"
    "dst2src_stddev_piat_ms,dst2src_max_piat_ms,bidirectional_syn_packets,bidirectional_cwr_packets,"
    "bidirectional_ece_packets,bidirectional_urg_packets,bidirectional_ack_packets,bidirectional_psh_packets,"
    "bidirectional_rst_packets,bidirectional_fin_packets,src2dst_syn_packets,src2dst_cwr_packets,src2dst_ece_packets,"
    "src2dst_urg_packets,src2dst_ack_packets,src2dst_psh_packets,src2dst_rst_packets,src2dst_fin_packets,"
    "dst2src_syn_packets,dst2src_cwr_packets,dst2src_ece_packets,dst2src_urg_packets,dst2src_ack_packets,"
    "dst2src_psh_packets,dst2src_rst_packets,dst2src_fin_packets,udps.srcdst_packet_size_variation,"
    "udps.srcdst_udp_packet_count,udps.udp_packet_count,udps.srcdst_tcp_packet_count,udps.tcp_packet_count,"
    "udps.srcdst_ack_packet_count,udps.ack_packet_count,udps.srcdst_fin_packet_count,udps.fin_packet_count,"
    "udps.srcdst_rst_packet_count,udps.rst_packet_count,udps.srcdst_psh_packet_count,udps.psh_packet_count,"
    "udps.srcdst_syn_packet_count,udps.syn_packet_count,udps.srcdst_unique_ports_count,udps.srcdst_icmp_packet_count,"
    "udps.icmp_packet_count,udps.srcdst_http_ports_count,udps.http_ports_count,udps.srcdst_bidirectional_duration_avg,"
    "udps.bidirectional_duration_avg,udps.srcdst_dns_port_count,udps.dns_port_count,udps.srcdst_dns_port_src_count,"
    "udps.dns_port_src_count,udps.srcdst_vul_ports_count,udps.src2dst_packet_count,udps.bidirectional_packet_count,"
    "udps.srcdst_src2dst_packet_count,udps.srcdst_bidirectional_packet_count,label"
)

TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "partial", "needs_input"})


def fixed_feature_values_csv() -> str:
    """Values for POST /simulate/network-row* (no header, no label column)."""
    if SAMPLE_ROW_WITH_LABEL.rsplit(",", 1)[-1].upper() == "BENIGN":
        return SAMPLE_ROW_WITH_LABEL.rsplit(",", 1)[0]
    return SAMPLE_ROW_WITH_LABEL


def build_idempotency_key(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return f"demo-network-{uuid.uuid4().hex[:12]}"


def build_simulate_payload(force_error_step: str, latency_ms: int) -> dict[str, Any]:
    out: dict[str, Any] = {"latency_ms": max(0, latency_ms)}
    step = (force_error_step or "").strip()
    if step:
        out["force_error_step"] = step
    return out


def ping_api_health(client: httpx.Client) -> None:
    response = client.get("/health")
    response.raise_for_status()
    payload = response.json()
    print(f"API health: {payload.get('status', payload)}")


def post_network_row_simple(
    client: httpx.Client,
    *,
    values_csv: str,
    idempotency_key: str,
    simulate: dict[str, Any],
) -> dict[str, Any]:
    """Run full pipeline synchronously; response includes final agentic actions."""
    response = client.post(
        "/api/v1/simulate/network-row/simple",
        json={"values_csv": values_csv, "metadata": {}, "simulate": simulate},
        headers={"Idempotency-Key": idempotency_key},
    )
    response.raise_for_status()
    return response.json()


def post_network_row_async(
    client: httpx.Client,
    *,
    values_csv: str,
    idempotency_key: str,
    simulate: dict[str, Any],
) -> dict[str, Any]:
    """Enqueue single-row pipeline (202); poll run status separately."""
    response = client.post(
        "/api/v1/simulate/network-row",
        json={"values_csv": values_csv, "metadata": {}, "simulate": simulate},
        headers={"Idempotency-Key": idempotency_key},
    )
    response.raise_for_status()
    return response.json()


def post_network_traffic(
    client: httpx.Client,
    *,
    columns_csv: str,
    rows_csv: list[str],
    idempotency_key: str,
    simulate: dict[str, Any],
    model_version_public_id: str | None,
) -> dict[str, Any]:
    """Enqueue multi-row traffic ingest (202); poll run status separately."""
    response = client.post(
        "/api/v1/simulate/network-traffic",
        json={
            "model_version_public_id": model_version_public_id,
            "columns_csv": columns_csv,
            "rows_csv": rows_csv,
            "metadata": {},
            "simulate": simulate,
        },
        headers={"Idempotency-Key": idempotency_key},
    )
    response.raise_for_status()
    return response.json()


def fetch_run_status(client: httpx.Client, run_id: str) -> dict[str, Any]:
    response = client.get(f"/api/v1/runs/{run_id}")
    response.raise_for_status()
    return response.json()


def fetch_run_events(client: httpx.Client, run_id: str) -> list[dict[str, Any]]:
    response = client.get(f"/api/v1/runs/{run_id}/events")
    response.raise_for_status()
    events = response.json()
    return events if isinstance(events, list) else []


def poll_run_until_finished(
    client: httpx.Client,
    run_id: str,
    *,
    poll_interval_s: float,
    max_attempts: int,
) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for attempt in range(1, max_attempts + 1):
        last = fetch_run_status(client, run_id)
        status = str(last.get("status") or "")
        print(
            f"  poll {attempt}: status={status} "
            f"last_step={last.get('last_step')} duration_ms={last.get('duration_ms')}"
        )
        if status in TERMINAL_RUN_STATUSES:
            return last
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Run {run_id} did not reach a terminal status within {max_attempts} polls")


def print_run_created(response: dict[str, Any]) -> str:
    run_id = str(response["run_id"])
    print(
        f"run_id={run_id} trace_id={response.get('trace_id')} "
        f"status={response.get('status')}"
    )
    if response.get("status_url"):
        print(f"  status_url: {response['status_url']}")
    if response.get("events_url"):
        print(f"  events_url: {response['events_url']}")
    return run_id


def print_run_outcome(run: dict[str, Any]) -> None:
    print("\n--- Run outcome ---")
    print(f"status: {run.get('status')}")
    if run.get("error_summary"):
        print(f"error: {run.get('error_summary')}")

    predictions = run.get("predictions_json")
    if isinstance(predictions, dict) and predictions:
        print("prediction summary:")
        for key in (
            "predicted_label",
            "flagged_attack_or_anomaly",
            "max_class_probability",
            "rows_total",
            "rows_flagged",
            "prediction_job_public_id",
        ):
            if key in predictions:
                print(f"  {key}: {predictions[key]}")

    actions = run.get("final_actions")
    if isinstance(actions, list) and actions:
        print(f"agentic row actions: {len(actions)}")
        for item in actions:
            if not isinstance(item, dict):
                continue
            print(
                f"  row={item.get('row_index')} label={item.get('predicted_label')} "
                f"flagged={item.get('flagged_attack_or_anomaly')}"
            )
            if item.get("recommended_action"):
                print(f"    recommended: {item.get('recommended_action')}")
            if item.get("agentic_job_public_id"):
                print(f"    agentic_job: {item.get('agentic_job_public_id')}")
    elif isinstance(actions, dict):
        print(f"final_actions: {actions}")


def print_recent_run_events(events: list[dict[str, Any]], *, tail: int) -> None:
    if not events:
        print("\n(no run events)")
        return
    print(f"\n--- Last {min(tail, len(events))} run events ---")
    for event in events[-tail:]:
        print(
            f"{event.get('timestamp')} [{event.get('level')}] "
            f"{event.get('step_name')}: {event.get('message')}"
        )


def resolve_agentic_report_public_id(
    client: httpx.Client,
    final_actions: list[dict[str, Any]] | None,
) -> str | None:
    if not final_actions:
        return None
    for item in final_actions:
        if not isinstance(item, dict):
            continue
        job_id = str(item.get("agentic_job_public_id") or "").strip()
        if not job_id:
            continue
        response = client.get("/agent/reports", params={"agentic_job_public_id": job_id, "limit": 5})
        response.raise_for_status()
        reports = response.json()
        if isinstance(reports, list) and reports:
            public_id = reports[0].get("public_id")
            if public_id:
                return str(public_id)
    return None


def apply_agentic_report_on_chain(client: httpx.Client, report_public_id: str) -> dict[str, Any]:
    response = client.post(f"/agent/reports/{report_public_id}/apply")
    response.raise_for_status()
    return response.json()


def print_execution_chain_results(exec_report: dict[str, Any]) -> None:
    print("\n--- On-chain apply results ---")
    print(f"execution_id: {exec_report.get('id')}")
    print(f"status: {exec_report.get('status')}")
    print(f"attack_type: {exec_report.get('attack_type')}")
    print(f"integrity: {exec_report.get('integrity_overall')}")

    chain = exec_report.get("actions_chain_json")
    items: list[Any] = []
    if isinstance(chain, dict):
        raw = chain.get("items")
        if isinstance(raw, list):
            items = raw

    if not items:
        print("(no actions_chain_json items — report may predate chain apply metadata)")
        if exec_report.get("error_reason"):
            print(f"error_reason: {exec_report.get('error_reason')}")
        return

    print(f"chain actions: {len(items)}")
    for item in items:
        if not isinstance(item, dict):
            continue
        action = item.get("action", "—")
        tier = item.get("network_tier") or "—"
        result = item.get("result") or "—"
        whitelisted = item.get("whitelisted")
        wl = "yes" if whitelisted is True else "no" if whitelisted is False else "—"
        print(f"  [{item.get('index')}] {action}")
        print(f"       tier={tier} whitelisted={wl} result={result}")
        if item.get("apply_tx_hash"):
            print(f"       tx={item.get('apply_tx_hash')}")
        reason = item.get("failure_reason") or item.get("whitelist_error") or item.get("apply_error")
        if reason:
            print(f"       detail: {reason}")


def run_console_demo(args: argparse.Namespace) -> int:
    idempotency_key = build_idempotency_key(args.idempotency_key)
    simulate = build_simulate_payload(args.force_error_step, args.latency_ms)
    values_csv = args.values_csv.strip() if args.values_csv else fixed_feature_values_csv()

    timeout = httpx.Timeout(args.timeout_s)
    with httpx.Client(base_url=args.base.rstrip("/"), timeout=timeout) as client:
        ping_api_health(client)
        print(f"\nmode={args.mode} idempotency_key={idempotency_key}")

        if args.mode == "simple":
            created = post_network_row_simple(
                client,
                values_csv=values_csv,
                idempotency_key=idempotency_key,
                simulate=simulate,
            )
            run_id = print_run_created(created)
            run_status = fetch_run_status(client, run_id)
            if isinstance(created.get("actions"), list):
                run_status = {**run_status, "final_actions": created.get("actions")}
        elif args.mode == "async-row":
            created = post_network_row_async(
                client,
                values_csv=values_csv,
                idempotency_key=idempotency_key,
                simulate=simulate,
            )
            run_id = print_run_created(created)
            run_status = poll_run_until_finished(
                client,
                run_id,
                poll_interval_s=args.poll_interval_s,
                max_attempts=args.max_polls,
            )
        elif args.mode == "traffic":
            created = post_network_traffic(
                client,
                columns_csv=SAMPLE_COLUMNS_WITH_LABEL,
                rows_csv=[SAMPLE_ROW_WITH_LABEL],
                idempotency_key=idempotency_key,
                simulate=simulate,
                model_version_public_id=args.model_version_public_id,
            )
            run_id = print_run_created(created)
            run_status = poll_run_until_finished(
                client,
                run_id,
                poll_interval_s=args.poll_interval_s,
                max_attempts=args.max_polls,
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        print_run_outcome(run_status)

        if not args.skip_events:
            events = fetch_run_events(client, run_id)
            print_recent_run_events(events, tail=args.event_tail)

        if args.apply:
            final_actions = run_status.get("final_actions")
            actions_list = final_actions if isinstance(final_actions, list) else None
            report_id = resolve_agentic_report_public_id(client, actions_list)
            if not report_id:
                print("\n(skip apply: no agentic report found for this run)")
                return 0 if run_status.get("status") == "completed" else 1
            print(f"\nApplying agentic report {report_id[:8]}… on-chain")
            exec_report = apply_agentic_report_on_chain(client, report_id)
            print_execution_chain_results(exec_report)

    status = str(run_status.get("status") or "")
    return 0 if status == "completed" else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate a VFL network-traffic row through prediction, RAG, and agentic planning.",
    )
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--mode",
        choices=("simple", "async-row", "traffic"),
        default="simple",
        help="simple=sync network-row/simple; async-row=202 + poll; traffic=multi-row ingest",
    )
    parser.add_argument("--idempotency-key", default="", help="Reuse to return the same run (if still cached)")
    parser.add_argument(
        "--values-csv",
        default="",
        help="Override feature values (no header/label) for simple/async-row modes",
    )
    parser.add_argument("--model-version-public-id", default=None, help="Traffic mode only")
    parser.add_argument("--force-error-step", default="", help="Inject failure at pipeline step name")
    parser.add_argument("--latency-ms", type=int, default=0, help="Artificial ingestion delay")
    parser.add_argument("--apply", action="store_true", help="POST /agent/reports/{id}/apply after pipeline")
    parser.add_argument("--poll-interval-s", type=float, default=0.75, help="Async modes poll interval")
    parser.add_argument("--max-polls", type=int, default=120, help="Async modes max poll attempts")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="HTTP client timeout (simple mode can be slow)")
    parser.add_argument("--skip-events", action="store_true", help="Do not print run event tail")
    parser.add_argument("--event-tail", type=int, default=12, help="How many recent events to print")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        code = run_console_demo(args)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:800] if exc.response is not None else str(exc)
        print(f"HTTP error {exc.response.status_code if exc.response else '?'}: {detail}", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:
        print(f"Demo failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    raise SystemExit(code)


if __name__ == "__main__":
    main()
