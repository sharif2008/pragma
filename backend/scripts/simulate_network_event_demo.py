#!/usr/bin/env python3
"""
Demo: simulate a single network-event row and poll run status/events.

Usage (server running):
  cd backend
  python scripts/simulate_network_event_demo.py --base http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import httpx


SAMPLE_COLUMNS = """bidirectional_duration_ms,bidirectional_packets,bidirectional_bytes,src2dst_duration_ms,src2dst_packets,src2dst_bytes,dst2src_duration_ms,dst2src_packets,dst2src_bytes,bidirectional_min_ps,bidirectional_mean_ps,bidirectional_stddev_ps,bidirectional_max_ps,src2dst_min_ps,src2dst_mean_ps,src2dst_stddev_ps,src2dst_max_ps,dst2src_min_ps,dst2src_mean_ps,dst2src_stddev_ps,dst2src_max_ps,bidirectional_min_piat_ms,bidirectional_mean_piat_ms,bidirectional_stddev_piat_ms,bidirectional_max_piat_ms,src2dst_min_piat_ms,src2dst_mean_piat_ms,src2dst_stddev_piat_ms,src2dst_max_piat_ms,dst2src_min_piat_ms,dst2src_mean_piat_ms,dst2src_stddev_piat_ms,dst2src_max_piat_ms,bidirectional_syn_packets,bidirectional_cwr_packets,bidirectional_ece_packets,bidirectional_urg_packets,bidirectional_ack_packets,bidirectional_psh_packets,bidirectional_rst_packets,bidirectional_fin_packets,src2dst_syn_packets,src2dst_cwr_packets,src2dst_ece_packets,src2dst_urg_packets,src2dst_ack_packets,src2dst_psh_packets,src2dst_rst_packets,src2dst_fin_packets,dst2src_syn_packets,dst2src_cwr_packets,dst2src_ece_packets,dst2src_urg_packets,dst2src_ack_packets,dst2src_psh_packets,dst2src_rst_packets,dst2src_fin_packets,udps.srcdst_packet_size_variation,udps.srcdst_udp_packet_count,udps.udp_packet_count,udps.srcdst_tcp_packet_count,udps.tcp_packet_count,udps.srcdst_ack_packet_count,udps.ack_packet_count,udps.srcdst_fin_packet_count,udps.fin_packet_count,udps.srcdst_rst_packet_count,udps.rst_packet_count,udps.srcdst_psh_packet_count,udps.psh_packet_count,udps.srcdst_syn_packet_count,udps.syn_packet_count,udps.srcdst_unique_ports_count,udps.srcdst_icmp_packet_count,udps.icmp_packet_count,udps.srcdst_http_ports_count,udps.http_ports_count,udps.srcdst_bidirectional_duration_avg,udps.bidirectional_duration_avg,udps.srcdst_dns_port_count,udps.dns_port_count,udps.srcdst_dns_port_src_count,udps.dns_port_src_count,udps.srcdst_vul_ports_count,udps.src2dst_packet_count,udps.bidirectional_packet_count,udps.srcdst_src2dst_packet_count,udps.srcdst_bidirectional_packet_count,label"""

SAMPLE_ROW = """75,6,425,75,4,305,0,2,120,60,70.83333333333333,17.440374613713626,100,60,76.25,19.73786547054502,100,60,60,0,60,0,15,20.54263858417414,38,0,25,21.65640782770772,38,0,0,0,0,0,0,0,0,6,2,0,2,0,0,0,0,4,2,0,1,0,0,0,0,2,0,0,1,20,0,0,10,31,281,844,6,16,0,0,107,269,6,16,1,0,0,10,31,860,909.3548387096774,0,0,0,0,10,361,852,121,284,BENIGN"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--idempotency-key", default="demo_network_event_001")
    ap.add_argument("--force-error-step", default="")
    args = ap.parse_args()

    body: dict[str, Any] = {
        "model_version_public_id": None,
        "columns_csv": SAMPLE_COLUMNS,
        "rows_csv": [SAMPLE_ROW],
        "simulate": {"latency_ms": 0, "force_error_step": (args.force_error_step or None)},
        "idempotency_key": args.idempotency_key,
    }

    with httpx.Client(base_url=args.base, timeout=30.0) as c:
        r = c.post("/api/v1/simulate/network-traffic", json=body)
        r.raise_for_status()
        data = r.json()
        run_id = data["run_id"]
        print("run_id:", run_id, "trace_id:", data["trace_id"], "status:", data["status"])

        # Poll status
        for _ in range(60):
            s = c.get(f"/api/v1/runs/{run_id}")
            s.raise_for_status()
            sj = s.json()
            print("status:", sj["status"], "last_step:", sj.get("last_step"), "duration_ms:", sj.get("duration_ms"))
            if sj["status"] in ("completed", "failed", "partial", "needs_input"):
                break
            time.sleep(0.75)

        ev = c.get(f"/api/v1/runs/{run_id}/events")
        ev.raise_for_status()
        events = ev.json()
        print("events:", len(events))
        for e in events[-10:]:
            print(e["timestamp"], e["step_name"], e["level"], e["message"])


if __name__ == "__main__":
    main()

