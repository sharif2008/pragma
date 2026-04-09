"""Agentic report listing and lookup."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models.domain import (
    AgenticJob,
    AgenticReport,
    AgenticReportTrustAnchor,
    JobStatus,
    PredictionJob,
)
from app.schemas.prediction import (
    AgenticJobOut,
    AgenticReportOut,
    TrustAnchorListItemOut,
    TrustAnchorVerifyOut,
)
from app.services import prediction_service
from app.services import trust_chain_service
from app.utils.file_utils import remove_path

logger = logging.getLogger(__name__)


def agentic_job_out(db: Session, row: AgenticJob) -> AgenticJobOut:
    pj = db.get(PredictionJob, row.prediction_job_id)
    if not pj:
        return AgenticJobOut(
            public_id=row.public_id,
            prediction_job_public_id="",
            results_row_index=row.results_row_index,
            label=row.label,
            prediction_status=JobStatus.pending,
            rows_total=None,
            rows_flagged=None,
            results_model_kind=None,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
    mk = prediction_service.results_model_kind_from_job(pj)
    return AgenticJobOut(
        public_id=row.public_id,
        prediction_job_public_id=pj.public_id,
        results_row_index=row.results_row_index,
        label=row.label,
        prediction_status=pj.status,
        rows_total=pj.rows_total,
        rows_flagged=pj.rows_flagged,
        results_model_kind=mk,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def persist_agentic_report_from_decision(
    db: Session,
    settings: Settings,
    *,
    job: PredictionJob,
    results_row_index: int | None,
    agentic_job_id: int | None,
    agentic_job_public_id: str | None,
    sample_data: dict[str, Any],
    user_prompt: str,
    decision: dict[str, Any],
) -> tuple[AgenticReport, dict[str, Any]]:
    """
    Write on-disk report JSON and insert ``agentic_reports`` (same shape as POST /agent/decide).

    Returns the ORM row and the payload dict (caller may append e.g. trust_chain and rewrite the file).
    """
    report_abs = settings.storage_root / "reports" / f"agentic_{job.public_id}_{uuid.uuid4().hex[:12]}.json"
    report_abs.parent.mkdir(parents=True, exist_ok=True)
    summary = str(decision.get("summary") or "").strip() or "—"
    recommended_action = str(decision.get("recommended_action") or "").strip() or "—"
    payload: dict[str, Any] = {
        "prediction_job_public_id": job.public_id,
        "results_row_index": results_row_index,
        "agentic_job_public_id": (agentic_job_public_id.strip() if isinstance(agentic_job_public_id, str) else None),
        "sample_data": sample_data,
        "user_prompt": user_prompt,
        "summary": summary,
        "recommended_action": recommended_action,
        "raw_llm_response": decision.get("raw_llm_response"),
        "rag_context_used": decision.get("rag_context_used"),
    }
    raw_llm = decision.get("raw_llm_response")
    if isinstance(raw_llm, str) and raw_llm.strip():
        m = re.search(r"\{[\s\S]*\}", raw_llm)
        if m:
            try:
                payload["structured_plan"] = json.loads(m.group())
            except json.JSONDecodeError:
                pass
    report_abs.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rel = str(report_abs.relative_to(settings.storage_root))

    row = AgenticReport(
        prediction_job_id=job.id,
        results_row_index=results_row_index,
        agentic_job_id=agentic_job_id,
        summary=summary,
        recommended_action=recommended_action,
        raw_llm_response=decision.get("raw_llm_response"),
        rag_context_used=decision.get("rag_context_used"),
        report_path=rel,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Optional: anchor a hash-only commitment on the local chain (non-fatal).
    if settings.trust_chain_enabled:
        try:
            structured_plan = payload.get("structured_plan")
            commitment, _canonical_payload = trust_chain_service.compute_trust_commitment_sha256(
                payload_version=settings.trust_chain_payload_version,
                agentic_report_public_id=row.public_id,
                prediction_job_public_id=job.public_id,
                results_row_index=results_row_index,
                created_at=row.created_at,
                raw_llm_response=row.raw_llm_response,
                rag_context_used=row.rag_context_used,
                structured_plan=structured_plan,
            )
            tx_hash, contract_addr, agent_key_sha, report_key_sha = (
                trust_chain_service.anchor_report_commitment_on_chain(
                    settings=settings,
                    agentic_job_public_id=agentic_job_public_id,
                    agentic_report_public_id=row.public_id,
                    commitment_sha256_hex=commitment,
                )
            )
            anchor = AgenticReportTrustAnchor(
                agentic_report_id=row.id,
                chain_id=settings.trust_chain_chain_id,
                contract_address=contract_addr,
                tx_hash=tx_hash,
                payload_version=settings.trust_chain_payload_version,
                commitment_sha256=commitment,
                agent_key_sha256=agent_key_sha,
                report_key_sha256=report_key_sha,
                error=None,
            )
            db.add(anchor)
            db.commit()
        except Exception as e:
            # Persist error for later inspection; do not fail the report creation.
            try:
                anchor = AgenticReportTrustAnchor(
                    agentic_report_id=row.id,
                    chain_id=settings.trust_chain_chain_id,
                    contract_address=str(settings.trust_chain_contract_address or ""),
                    tx_hash="",
                    payload_version=settings.trust_chain_payload_version,
                    commitment_sha256="",
                    agent_key_sha256="",
                    report_key_sha256="",
                    error=str(e)[:2000],
                )
                db.add(anchor)
                db.commit()
            except Exception:
                pass
    return row, payload


def create_agentic_job(
    db: Session,
    *,
    prediction_job_public_id: str,
    results_row_index: int | None,
    label: str | None,
) -> AgenticJob:
    pj = prediction_service.get_prediction_job(db, prediction_job_public_id)
    row = AgenticJob(
        prediction_job_id=pj.id,
        results_row_index=results_row_index,
        label=(label.strip() if isinstance(label, str) and label.strip() else None),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def list_agentic_jobs(db: Session, *, limit: int = 100, offset: int = 0) -> list[AgenticJob]:
    q = (
        select(AgenticJob)
        .order_by(AgenticJob.updated_at.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(db.scalars(q).all())


def resolve_agentic_job_for_decide(
    db: Session,
    *,
    agentic_job_public_id: str | None,
    prediction_job_id: int,
    results_row_index: int | None,
) -> AgenticJob | None:
    if not agentic_job_public_id or not str(agentic_job_public_id).strip():
        return None
    from fastapi import HTTPException

    aj = db.scalar(select(AgenticJob).where(AgenticJob.public_id == str(agentic_job_public_id).strip()))
    if not aj:
        raise HTTPException(404, "Agentic job not found")
    if aj.prediction_job_id != prediction_job_id:
        raise HTTPException(400, "agentic_job_public_id does not match prediction_job_public_id")
    if aj.results_row_index != results_row_index:
        raise HTTPException(400, "agentic_job_public_id does not match results_row_index")
    return aj


def agentic_report_out(db: Session, row: AgenticReport) -> AgenticReportOut:
    pj = db.get(PredictionJob, row.prediction_job_id)
    aj = db.get(AgenticJob, row.agentic_job_id) if row.agentic_job_id else None
    anchor = db.scalar(
        select(AgenticReportTrustAnchor).where(AgenticReportTrustAnchor.agentic_report_id == row.id)
    )
    anchor_out = None
    if anchor:
        anchor_out = {
            "chain_id": anchor.chain_id,
            "contract_address": anchor.contract_address,
            "tx_hash": anchor.tx_hash,
            "payload_version": anchor.payload_version,
            "commitment_sha256": anchor.commitment_sha256,
            "agent_key_sha256": anchor.agent_key_sha256,
            "report_key_sha256": anchor.report_key_sha256,
            "anchored_at": anchor.anchored_at.isoformat() if anchor.anchored_at else None,
            "error": anchor.error,
        }
    return AgenticReportOut.model_validate(row).model_copy(
        update={
            "prediction_job_public_id": pj.public_id if pj else None,
            "agentic_job_public_id": aj.public_id if aj else None,
            "trust_anchor": anchor_out,
        }
    )


def list_agentic_reports(
    db: Session, *, limit: int = 100, offset: int = 0, agentic_job_id: int | None = None
) -> list[AgenticReport]:
    q = select(AgenticReport).order_by(AgenticReport.created_at.desc())
    if agentic_job_id is not None:
        q = q.where(AgenticReport.agentic_job_id == agentic_job_id)
    q = q.offset(offset).limit(limit)
    return list(db.scalars(q).all())


def get_agentic_report(db: Session, public_id: str) -> AgenticReport:
    row = db.scalar(select(AgenticReport).where(AgenticReport.public_id == public_id))
    if not row:
        from fastapi import HTTPException

        raise HTTPException(404, "Agentic report not found")
    return row


def delete_agentic_report(db: Session, settings: Settings, public_id: str) -> None:
    row = db.scalar(select(AgenticReport).where(AgenticReport.public_id == public_id))
    if not row:
        from fastapi import HTTPException

        raise HTTPException(404, "Agentic report not found")
    if row.report_path:
        out_abs = (settings.storage_root / row.report_path).resolve()
        root = settings.storage_root.resolve()
        try:
            out_abs.relative_to(root)
        except ValueError:
            logger.warning("Skipping removal of agent report file outside storage_root: %s", row.report_path)
        else:
            remove_path(out_abs)
    db.delete(row)
    db.commit()


def _structured_plan_from_saved_payload(data: dict[str, Any]) -> Any:
    sp = data.get("structured_plan")
    if sp is not None:
        return sp
    raw_llm = data.get("raw_llm_response")
    if isinstance(raw_llm, str) and raw_llm.strip():
        m = re.search(r"\{[\s\S]*\}", raw_llm)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def list_trust_anchor_rows(
    db: Session, *, limit: int = 100, offset: int = 0
) -> list[TrustAnchorListItemOut]:
    q = (
        select(AgenticReportTrustAnchor, AgenticReport, PredictionJob, AgenticJob)
        .join(AgenticReport, AgenticReportTrustAnchor.agentic_report_id == AgenticReport.id)
        .join(PredictionJob, AgenticReport.prediction_job_id == PredictionJob.id)
        .outerjoin(AgenticJob, AgenticReport.agentic_job_id == AgenticJob.id)
        .order_by(AgenticReportTrustAnchor.anchored_at.desc())
        .offset(offset)
        .limit(limit)
    )
    out: list[TrustAnchorListItemOut] = []
    for anchor, report, pj, aj in db.execute(q).all():
        summ = (report.summary or "").strip()
        preview = summ if len(summ) <= 160 else summ[:157] + "…"
        out.append(
            TrustAnchorListItemOut(
                id=anchor.id,
                agentic_report_public_id=report.public_id,
                prediction_job_public_id=pj.public_id,
                agentic_job_public_id=(aj.public_id if aj else None),
                summary_preview=preview,
                recommended_action=(report.recommended_action or "")[:512],
                tx_hash=anchor.tx_hash or "",
                chain_id=anchor.chain_id,
                contract_address=anchor.contract_address or "",
                commitment_sha256=anchor.commitment_sha256 or "",
                payload_version=anchor.payload_version,
                anchored_at=anchor.anchored_at,
                anchor_error=anchor.error,
            )
        )
    return out


def get_trust_anchor_bundle(
    db: Session, anchor_id: int
) -> tuple[AgenticReportTrustAnchor, AgenticReport, PredictionJob, AgenticJob | None] | None:
    q = (
        select(AgenticReportTrustAnchor, AgenticReport, PredictionJob, AgenticJob)
        .join(AgenticReport, AgenticReportTrustAnchor.agentic_report_id == AgenticReport.id)
        .join(PredictionJob, AgenticReport.prediction_job_id == PredictionJob.id)
        .outerjoin(AgenticJob, AgenticReport.agentic_job_id == AgenticJob.id)
        .where(AgenticReportTrustAnchor.id == anchor_id)
        .limit(1)
    )
    row = db.execute(q).first()
    if not row:
        return None
    return row[0], row[1], row[2], row[3]


def verify_trust_anchor_row(db: Session, settings: Settings, anchor_id: int) -> TrustAnchorVerifyOut | None:
    bundle = get_trust_anchor_bundle(db, anchor_id)
    if not bundle:
        return None
    anchor, report, pj, aj = bundle
    summ = (report.summary or "").strip()
    preview = summ if len(summ) <= 160 else summ[:157] + "…"
    aj_pub = aj.public_id if aj else None
    db_commit = (anchor.commitment_sha256 or "").strip().lower()
    tx_ok = bool(anchor.tx_hash and anchor.tx_hash.strip() and not anchor.error)
    anchor_failed = bool(anchor.error) or not anchor.tx_hash.strip() or len(db_commit) != 64

    chain_valid: bool | None = None
    chain_detail: str | None = None
    on_chain_hex: str | None = None

    payload_valid: bool | None = None
    payload_detail: str | None = None
    recomputed: str | None = None

    if not anchor_failed and anchor.contract_address.strip():
        rpc_ok, oc_hex, err = trust_chain_service.read_commitment_from_chain(
            settings,
            contract_address=anchor.contract_address,
            agent_key_sha256_hex=anchor.agent_key_sha256,
            report_key_sha256_hex=anchor.report_key_sha256,
        )
        if not rpc_ok:
            chain_valid = None
            chain_detail = err or "RPC error"
        elif oc_hex is None:
            chain_valid = None
            chain_detail = err or "no commitment returned"
        else:
            on_chain_hex = oc_hex
            chain_valid = oc_hex == db_commit
            chain_detail = None if chain_valid else "on-chain commitment does not match database record"
            if oc_hex == "0" * 64:
                chain_valid = False
                chain_detail = "on-chain commitment is empty (wrong keys or never anchored)"
    elif not anchor_failed:
        chain_detail = "missing contract_address"

    if report.report_path:
        path = settings.storage_root / report.report_path
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    payload_valid = None
                    payload_detail = "report file is not a JSON object"
                else:
                    structured = _structured_plan_from_saved_payload(data)
                    recomputed, _pl = trust_chain_service.compute_trust_commitment_sha256(
                        payload_version=anchor.payload_version,
                        agentic_report_public_id=report.public_id,
                        prediction_job_public_id=pj.public_id,
                        results_row_index=report.results_row_index,
                        created_at=report.created_at,
                        raw_llm_response=report.raw_llm_response,
                        rag_context_used=report.rag_context_used,
                        structured_plan=structured,
                    )
                    recomputed = recomputed.lower()
                    if len(db_commit) == 64:
                        payload_valid = recomputed == db_commit
                        payload_detail = (
                            None
                            if payload_valid
                            else "recomputed hash from report file does not match anchored commitment"
                        )
                    else:
                        payload_valid = None
                        payload_detail = "database has no commitment to compare"
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                payload_valid = None
                payload_detail = f"could not read or parse report file: {e}"[:500]
        else:
            payload_valid = None
            payload_detail = "report file missing on disk"
    else:
        payload_valid = None
        payload_detail = "no report_path on agentic report"

    overall_lit: Literal["valid", "invalid", "unknown", "anchor_failed"]
    if anchor_failed:
        overall_lit = "anchor_failed"
    elif chain_valid is False or payload_valid is False:
        overall_lit = "invalid"
    elif chain_valid is True and payload_valid is True:
        overall_lit = "valid"
    else:
        overall_lit = "unknown"

    rpc_connected = False
    if settings.trust_chain_rpc_url:
        try:
            from web3 import Web3

            w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
            rpc_connected = bool(w3.is_connected())
        except Exception:
            rpc_connected = False

    return TrustAnchorVerifyOut(
        anchor_id=anchor.id,
        agentic_report_public_id=report.public_id,
        prediction_job_public_id=pj.public_id,
        agentic_job_public_id=aj_pub,
        summary_preview=preview,
        recommended_action=(report.recommended_action or "")[:512],
        tx_hash=anchor.tx_hash or "",
        chain_id=anchor.chain_id,
        contract_address=anchor.contract_address or "",
        rpc_url=settings.trust_chain_rpc_url,
        rpc_connected=rpc_connected,
        db_commitment_sha256=db_commit or anchor.commitment_sha256,
        on_chain_commitment_hex=on_chain_hex,
        chain_integrity_valid=chain_valid,
        chain_integrity_detail=chain_detail,
        recomputed_commitment_sha256=recomputed,
        payload_integrity_valid=payload_valid,
        payload_integrity_detail=payload_detail,
        overall_integrity=overall_lit,
    )
