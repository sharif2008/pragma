"""Simulation APIs: network traffic logs (VFL)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services import run_service
from app.services.run_service import create_raw_log, create_run, emit_event, new_trace_id
from app.services.simulate_network_pipeline import run_simulated_network_event, run_simulated_network_traffic

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["simulate"])

VFL_FIXED_COLUMNS_CSV = (
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
    "udps.srcdst_src2dst_packet_count,udps.srcdst_bidirectional_packet_count"
)
VFL_FIXED_COLUMNS = [c.strip() for c in VFL_FIXED_COLUMNS_CSV.split(",") if c.strip()]


def _split_csv_values(s: str) -> list[str]:
    return [v.strip() for v in str(s).split(",")]


def _validate_fixed_row(values: list[str]) -> None:
    if len(values) != len(VFL_FIXED_COLUMNS):
        raise HTTPException(
            400,
            f"values_csv count ({len(values)}) != fixed_columns count ({len(VFL_FIXED_COLUMNS)})",
        )


class SimulateNetworkEventRequest(BaseModel):
    model_version_public_id: str | None = None
    columns_csv: str = Field(min_length=1)
    row_csv: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    simulate: dict[str, Any] = Field(default_factory=dict)


class SimulateNetworkEventResponse(BaseModel):
    run_id: str
    trace_id: str
    status: str
    status_url: str
    events_url: str


class SimulateNetworkTrafficRequest(BaseModel):
    """Multiple log rows. Header is optional; if omitted, row_csv values must match model feature columns length."""

    model_version_public_id: str | None = None
    columns_csv: str | None = None
    rows_csv: list[str] = Field(min_length=1, description="One CSV row string per log sample (no header).")
    metadata: dict[str, Any] = Field(default_factory=dict)
    simulate: dict[str, Any] = Field(default_factory=dict)


class SimulateNetworkTrafficResponse(BaseModel):
    run_id: str
    trace_id: str
    status: str
    status_url: str
    events_url: str


class SimulateNetworkRowRequest(BaseModel):
    """
    Simplified VFL request: server uses a fixed feature header and latest model.
    Provide exactly one CSV row (comma-separated values, no header, no label).
    """

    values_csv: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    simulate: dict[str, Any] = Field(default_factory=dict)


class SimulateNetworkRowResponse(BaseModel):
    run_id: str
    trace_id: str
    status: str
    status_url: str
    events_url: str


class SimulateNetworkRowSimpleResponse(BaseModel):
    run_id: str
    trace_id: str
    status: str
    status_url: str
    events_url: str
    actions: list[dict[str, Any]] | dict[str, Any] | None = None


@router.post("/simulate/network-event", response_model=SimulateNetworkEventResponse, status_code=202)
async def simulate_network_event(
    body: SimulateNetworkEventRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    idempotency_key_hdr: Annotated[
        str | None,
        Header(alias="Idempotency-Key", include_in_schema=False),
    ] = None,
) -> SimulateNetworkEventResponse:
    trace_id = new_trace_id()
    idem = idempotency_key_hdr or f"auto-{uuid.uuid4().hex}"

    if idem:
        existing = run_service.get_run_by_idempotency_key(db, idem)
        if existing:
            return SimulateNetworkEventResponse(
                run_id=existing.run_id,
                trace_id=existing.trace_id,
                status=str(getattr(existing.status, "value", existing.status)),
                status_url=f"/api/v1/runs/{existing.run_id}",
                events_url=f"/api/v1/runs/{existing.run_id}/events",
            )

    cols = [c.strip() for c in body.columns_csv.split(",") if c.strip()]
    vals = [v.strip() for v in body.row_csv.split(",")]
    if len(cols) != len(vals):
        raise HTTPException(400, f"columns_csv count ({len(cols)}) != row_csv count ({len(vals)})")

    if cols and cols[-1].lower() == "label":
        cols_model = cols[:-1]
        vals_model = vals[:-1]
    else:
        cols_model = cols
        vals_model = vals

    normalized = {
        "kind": "network_event",
        "columns": cols_model,
        "values": vals_model,
        "model_version_public_id": body.model_version_public_id,
        "metadata": body.metadata,
        "simulate": body.simulate,
    }

    run = create_run(
        db,
        trace_id=trace_id,
        idempotency_key=idem,
        customer_id=None,
        channel="network",
        message=f"network_event columns={len(cols_model)}",
    )
    create_raw_log(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        raw_payload=body.model_dump(),
        normalized_payload=normalized,
    )
    emit_event(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        step_name="run",
        level="info",
        message="created",
        payload={"idempotency_key": idem, "kind": "network_event"},
    )

    asyncio.create_task(
        run_simulated_network_event(
            settings=settings,
            run_id=run.run_id,
            trace_id=run.trace_id,
            columns=cols_model,
            values=vals_model,
            model_version_public_id=body.model_version_public_id,
            simulate=body.simulate,
        )
    )

    return SimulateNetworkEventResponse(
        run_id=run.run_id,
        trace_id=run.trace_id,
        status="running",
        status_url=f"/api/v1/runs/{run.run_id}",
        events_url=f"/api/v1/runs/{run.run_id}/events",
    )


@router.post("/simulate/network-traffic", response_model=SimulateNetworkTrafficResponse, status_code=202)
async def simulate_network_traffic(
    body: SimulateNetworkTrafficRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    idempotency_key_hdr: Annotated[
        str | None,
        Header(alias="Idempotency-Key", include_in_schema=False),
    ] = None,
) -> SimulateNetworkTrafficResponse:
    trace_id = new_trace_id()
    idem = idempotency_key_hdr or f"auto-{uuid.uuid4().hex}"

    if idem:
        existing = run_service.get_run_by_idempotency_key(db, idem)
        if existing:
            return SimulateNetworkTrafficResponse(
                run_id=existing.run_id,
                trace_id=existing.trace_id,
                status=str(getattr(existing.status, "value", existing.status)),
                status_url=f"/api/v1/runs/{existing.run_id}",
                events_url=f"/api/v1/runs/{existing.run_id}/events",
            )

    incoming_cols: list[str] | None = None
    if body.columns_csv and str(body.columns_csv).strip():
        incoming_cols = [c.strip() for c in str(body.columns_csv).split(",") if c.strip()]
        if incoming_cols and incoming_cols[-1].lower() == "label":
            incoming_cols = incoming_cols[:-1]

    parsed_rows: list[list[Any]] = []
    for line in body.rows_csv:
        vals = [v.strip() for v in str(line).split(",")]
        parsed_rows.append(vals)

    normalized = {
        "kind": "network_traffic",
        "columns": incoming_cols,
        "rows": parsed_rows,
        "model_version_public_id": body.model_version_public_id,
        "metadata": body.metadata,
        "simulate": body.simulate,
    }

    run = create_run(
        db,
        trace_id=trace_id,
        idempotency_key=idem,
        customer_id=None,
        channel="network",
        message=f"network_traffic rows={len(parsed_rows)}",
    )
    create_raw_log(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        raw_payload=body.model_dump(),
        normalized_payload=normalized,
    )
    emit_event(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        step_name="run",
        level="info",
        message="created",
        payload={"idempotency_key": idem, "kind": "network_traffic", "rows": len(parsed_rows)},
    )

    asyncio.create_task(
        run_simulated_network_traffic(
            settings=settings,
            run_id=run.run_id,
            trace_id=run.trace_id,
            rows=parsed_rows,
            incoming_columns=incoming_cols,
            model_version_public_id=body.model_version_public_id,
            simulate=body.simulate,
        )
    )

    return SimulateNetworkTrafficResponse(
        run_id=run.run_id,
        trace_id=run.trace_id,
        status="running",
        status_url=f"/api/v1/runs/{run.run_id}",
        events_url=f"/api/v1/runs/{run.run_id}/events",
    )


@router.post("/simulate/network-row", response_model=SimulateNetworkRowResponse, status_code=202)
async def simulate_network_row(
    body: SimulateNetworkRowRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    idempotency_key_hdr: Annotated[
        str | None,
        Header(alias="Idempotency-Key", include_in_schema=False),
    ] = None,
) -> SimulateNetworkRowResponse:
    """
    Simplified async endpoint:
    - fixed server-side header (VFL_FIXED_COLUMNS)
    - latest model (model_version_public_id=None)
    - caller sends only a comma-separated values row
    """
    trace_id = new_trace_id()
    idem = idempotency_key_hdr or f"auto-{uuid.uuid4().hex}"

    if idem:
        existing = run_service.get_run_by_idempotency_key(db, idem)
        if existing:
            return SimulateNetworkRowResponse(
                run_id=existing.run_id,
                trace_id=existing.trace_id,
                status=str(getattr(existing.status, "value", existing.status)),
                status_url=f"/api/v1/runs/{existing.run_id}",
                events_url=f"/api/v1/runs/{existing.run_id}/events",
            )

    values = _split_csv_values(body.values_csv)
    _validate_fixed_row(values)

    normalized = {
        "kind": "network_row",
        "columns": VFL_FIXED_COLUMNS,
        "values": values,
        "model_version_public_id": None,
        "metadata": body.metadata,
        "simulate": body.simulate,
    }

    run = create_run(
        db,
        trace_id=trace_id,
        idempotency_key=idem,
        customer_id=None,
        channel="network",
        message=f"network_row columns={len(VFL_FIXED_COLUMNS)}",
    )
    create_raw_log(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        raw_payload=body.model_dump(),
        normalized_payload=normalized,
    )
    emit_event(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        step_name="run",
        level="info",
        message="created",
        payload={"idempotency_key": idem, "kind": "network_row"},
    )

    asyncio.create_task(
        run_simulated_network_traffic(
            settings=settings,
            run_id=run.run_id,
            trace_id=run.trace_id,
            rows=[values],
            incoming_columns=VFL_FIXED_COLUMNS,
            model_version_public_id=None,
            simulate=body.simulate,
        )
    )

    return SimulateNetworkRowResponse(
        run_id=run.run_id,
        trace_id=run.trace_id,
        status="running",
        status_url=f"/api/v1/runs/{run.run_id}",
        events_url=f"/api/v1/runs/{run.run_id}/events",
    )


@router.post("/simulate/network-row/simple", response_model=SimulateNetworkRowSimpleResponse)
async def simulate_network_row_simple(
    body: SimulateNetworkRowRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    idempotency_key_hdr: Annotated[
        str | None,
        Header(alias="Idempotency-Key", include_in_schema=False),
    ] = None,
) -> SimulateNetworkRowSimpleResponse:
    """
    Simplified sync endpoint (returns final actions):
    - fixed server-side header (VFL_FIXED_COLUMNS)
    - latest model (model_version_public_id=None)
    - runs prediction + RAG + agentic actions pipeline and returns actions list/dict
    """
    trace_id = new_trace_id()
    idem = idempotency_key_hdr or f"auto-{uuid.uuid4().hex}"

    if idem:
        existing = run_service.get_run_by_idempotency_key(db, idem)
        if existing and str(getattr(existing.status, "value", existing.status)) in (
            "completed",
            "failed",
            "partial",
            "needs_input",
        ):
            return SimulateNetworkRowSimpleResponse(
                run_id=existing.run_id,
                trace_id=existing.trace_id,
                status=str(getattr(existing.status, "value", existing.status)),
                status_url=f"/api/v1/runs/{existing.run_id}",
                events_url=f"/api/v1/runs/{existing.run_id}/events",
                actions=existing.final_actions,
            )

    values = _split_csv_values(body.values_csv)
    _validate_fixed_row(values)

    normalized = {
        "kind": "network_row",
        "columns": VFL_FIXED_COLUMNS,
        "values": values,
        "model_version_public_id": None,
        "metadata": body.metadata,
        "simulate": body.simulate,
    }

    run = create_run(
        db,
        trace_id=trace_id,
        idempotency_key=idem,
        customer_id=None,
        channel="network",
        message=f"network_row_simple columns={len(VFL_FIXED_COLUMNS)}",
    )
    create_raw_log(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        raw_payload=body.model_dump(),
        normalized_payload=normalized,
    )
    emit_event(
        db,
        run_id=run.run_id,
        trace_id=run.trace_id,
        step_name="run",
        level="info",
        message="created",
        payload={"idempotency_key": idem, "kind": "network_row_simple"},
    )

    await run_simulated_network_traffic(
        settings=settings,
        run_id=run.run_id,
        trace_id=run.trace_id,
        rows=[values],
        incoming_columns=VFL_FIXED_COLUMNS,
        model_version_public_id=None,
        simulate=body.simulate,
    )

    # Pipeline uses its own DB session; expire so we read committed status/actions for this response.
    db.expire_all()
    row2 = run_service.get_run_by_id(db, run.run_id)
    status2 = str(getattr(row2.status, "value", row2.status)) if row2 else "unknown"
    return SimulateNetworkRowSimpleResponse(
        run_id=run.run_id,
        trace_id=run.trace_id,
        status=status2,
        status_url=f"/api/v1/runs/{run.run_id}",
        events_url=f"/api/v1/runs/{run.run_id}/events",
        actions=(row2.final_actions if row2 else None),
    )

