#!/usr/bin/env python3
"""
PRAGMA CSV benchmark runner:

- Read rows from a CSV file
- Run a "default model" inference (either via an explicit model bundle, or a lightweight fallback)
- Optionally retrieve a default RAG context from the KB (if DB + vector store are configured)
- Call the agent once to produce an action plan (LLM or deterministic mock when OPENAI_API_KEY is unset)
- Anchor a hash-only commitment on-chain via AgenticTrustRegistry (Hardhat / permissioned-style Ethereum)
- Validate by reading commitment back from chain and recomputing hash from the canonical payload
- Track per-step timings and emit a per-row report table

This script is intentionally self-contained and does not require the API server to be running.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
try:
    from web3 import Web3
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: web3.py.\n"
        "Run this script in the backend virtualenv, e.g.:\n"
        "  cd backend\n"
        "  .venv\\Scripts\\activate   (Windows)\n"
        "  pip install -r requirements.txt\n"
        "Then re-run the script.\n"
        f"Import error: {e}"
    )

from app.core.config import get_settings
from app.services import kb_service, llm_service, trust_chain_service


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _perf_ms(t0: float, t1: float) -> float:
    return (t1 - t0) * 1000.0


def _parse_structured_plan(raw_llm_response: str | None) -> Any | None:
    if not isinstance(raw_llm_response, str) or not raw_llm_response.strip():
        return None
    m = re.search(r"\{[\s\S]*\}", raw_llm_response)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _infer_default(
    row: dict[str, Any],
    *,
    label_col: str | None,
    confidence_col: str | None,
) -> tuple[str, float]:
    """
    Lightweight inference fallback for benchmark runs when a real model bundle isn't provided.

    - If label_col exists, treat it as predicted_label (uppercased string)
    - If confidence_col exists, use it; otherwise set a conservative default
    """
    lab = None
    if label_col:
        v = row.get(label_col)
        if v is not None and str(v).strip():
            lab = str(v).strip()
    predicted = (lab or "UNKNOWN").upper()

    conf = None
    if confidence_col:
        cv = row.get(confidence_col)
        conf = _safe_float(cv, default=None) if cv is not None else None
    if conf is None:
        conf = 0.75 if predicted not in ("BENIGN", "NORMAL", "0") else 0.55
    conf = max(0.0, min(1.0, float(conf)))
    return predicted, conf


def _build_rag_query(predicted_label: str, confidence: float) -> str:
    return (
        "Enterprise intrusion detection mitigation guidance and triage steps for "
        f"predicted_label={predicted_label} confidence={confidence:.2f}. "
        "Include containment, monitoring, and escalation decision points."
    )


def _maybe_query_kb(settings, *, query: str) -> tuple[str | None, int]:
    """
    Attempt a KB query. If DB/vector store isn't configured, return (None, 0).
    """
    try:
        from app.db.session import SessionLocal

        db = SessionLocal()
    except Exception:
        return None, 0

    try:
        hits = kb_service.query_kb(db, settings, query, settings.rag_top_k, None)
        if not hits:
            return None, 0
        lines: list[str] = []
        for score, chunk, _meta in hits:
            txt = str(chunk.get("text", ""))[:800]
            lines.append(f"- ({score:.3f}) {txt}")
        return "\n\n".join(lines), len(hits)
    except Exception:
        return None, 0
    finally:
        try:
            db.close()
        except Exception:
            pass


def _preflight_chain(settings) -> tuple[Web3, str]:
    if not settings.trust_chain_enabled:
        raise RuntimeError("TRUST_CHAIN_ENABLED is false (enable trust chain to anchor on-chain).")
    if not settings.trust_chain_rpc_url:
        raise RuntimeError("TRUST_CHAIN_RPC_URL missing.")
    if not settings.trust_chain_contract_address:
        raise RuntimeError("TRUST_CHAIN_CONTRACT_ADDRESS missing.")

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("Could not connect to TRUST_CHAIN_RPC_URL.")

    addr = Web3.to_checksum_address(settings.trust_chain_contract_address)
    code = w3.eth.get_code(addr)
    if not code or code == b"\x00":
        raise RuntimeError("No contract code at TRUST_CHAIN_CONTRACT_ADDRESS (did you deploy AgenticTrustRegistry?).")

    return w3, addr


async def _agent_once(
    settings,
    *,
    sample_data: dict[str, Any],
    rag_context: str | None,
    use_rag: bool,
) -> dict[str, str | None]:
    # These are optional inputs; llm_service can work without them for mock runs.
    return await llm_service.agent_decide(
        settings,
        sample_data,
        feature_notes=None,
        rag_context=rag_context,
        attack_actions_data=None,
        agentic_features_data=None,
        use_rag=use_rag,
    )


@dataclass
class RowTiming:
    row_index: int
    predicted_label: str
    confidence: float
    kb_hits: int
    agentic_action_ms: float
    blockchain_store_ms: float
    validation_ms: float
    end_to_end_ms: float
    anchor_tx_hash: str
    chain_integrity_valid: bool
    payload_integrity_valid: bool
    executed: bool


def main() -> int:
    ap = argparse.ArgumentParser(description="PRAGMA CSV → agentic action → blockchain anchor benchmark.")
    ap.add_argument("--csv", required=True, help="Path to CSV dataset file.")
    ap.add_argument("--max-rows", type=int, default=25, help="Max rows to process (default 25).")
    ap.add_argument("--label-col", default="label", help="Label column name used for fallback inference (default 'label').")
    ap.add_argument(
        "--confidence-col",
        default=None,
        help="Optional confidence column name used for fallback inference.",
    )
    ap.add_argument(
        "--use-rag",
        action="store_true",
        help="Enable KB retrieval (requires DB + vector store configured).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path for the timing report (default backend/storage/reports/...).",
    )
    args = ap.parse_args()

    settings = get_settings()

    # Preflight chain connectivity + contract deployment
    _w3, contract_addr = _preflight_chain(settings)
    print(f"[ok] chain rpc={settings.trust_chain_rpc_url} contract={contract_addr} chain_id={settings.trust_chain_chain_id}")

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)
    if args.max_rows and args.max_rows > 0:
        df = df.head(int(args.max_rows))

    rows: list[RowTiming] = []
    started = time.perf_counter()

    for i, rec in enumerate(df.to_dict(orient="records")):
        t_row0 = time.perf_counter()

        # 1) "Default model" inference (fallback)
        predicted_label, conf = _infer_default(
            rec,
            label_col=(args.label_col or None),
            confidence_col=(args.confidence_col or None),
        )

        # 2) RAG retrieval (optional)
        rag_context: str | None = None
        kb_hits = 0
        if bool(args.use_rag):
            q = _build_rag_query(predicted_label, conf)
            rag_context, kb_hits = _maybe_query_kb(settings, query=q)

        # 3) Agentic action (single call)
        t_agent0 = time.perf_counter()
        sample_data: dict[str, Any] = {
            "predicted_label": predicted_label,
            "confidence": conf,
            "row": rec,  # bind input evidence into prompt context
        }
        decision = asyncio.run(
            _agent_once(settings, sample_data=sample_data, rag_context=rag_context, use_rag=bool(args.use_rag))
        )
        t_agent1 = time.perf_counter()

        structured_plan = _parse_structured_plan(decision.get("raw_llm_response"))

        # 4) Compute commitment payload + anchor on-chain
        created_at = _now_utc()
        commitment_sha256, canonical_payload = trust_chain_service.compute_trust_commitment_sha256(
            payload_version=settings.trust_chain_payload_version,
            agentic_report_public_id=f"csv_row_{i}",
            prediction_job_public_id=str(csv_path.name),
            results_row_index=i,
            created_at=created_at,
            raw_llm_response=decision.get("raw_llm_response"),
            rag_context_used=(decision.get("rag_context_used") or rag_context),
            structured_plan=structured_plan,
        )

        t_chain0 = time.perf_counter()
        tx_hash, contract_addr2, agent_key_sha, report_key_sha = trust_chain_service.anchor_report_commitment_on_chain(
            settings=settings,
            agentic_job_public_id="csv_benchmark",
            agentic_report_public_id=f"csv_row_{i}",
            commitment_sha256_hex=commitment_sha256,
        )
        t_chain1 = time.perf_counter()

        # 5) Validate: read from chain + recompute from canonical payload
        t_val0 = time.perf_counter()
        rpc_ok, on_chain_hex, err = trust_chain_service.read_commitment_from_chain(
            settings,
            contract_address=contract_addr2,
            agent_key_sha256_hex=agent_key_sha,
            report_key_sha256_hex=report_key_sha,
        )
        chain_valid = bool(rpc_ok and on_chain_hex and on_chain_hex.lower() == commitment_sha256.lower())
        if not rpc_ok and err:
            chain_valid = False

        # Recompute payload hash deterministically (same canonicalization rules)
        canonical_rejson = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        recomputed_sha256 = hashlib.sha256(canonical_rejson.encode("utf-8")).hexdigest()
        payload_valid = recomputed_sha256.lower() == commitment_sha256.lower()
        t_val1 = time.perf_counter()

        executed = bool(chain_valid and payload_valid)

        t_row1 = time.perf_counter()

        rows.append(
            RowTiming(
                row_index=i,
                predicted_label=predicted_label,
                confidence=float(conf),
                kb_hits=int(kb_hits),
                agentic_action_ms=_perf_ms(t_agent0, t_agent1),
                blockchain_store_ms=_perf_ms(t_chain0, t_chain1),
                validation_ms=_perf_ms(t_val0, t_val1),
                end_to_end_ms=_perf_ms(t_row0, t_row1),
                anchor_tx_hash=tx_hash,
                chain_integrity_valid=bool(chain_valid),
                payload_integrity_valid=bool(payload_valid),
                executed=executed,
            )
        )

        print(
            f"[row {i}] label={predicted_label} conf={conf:.2f} "
            f"agent_ms={rows[-1].agentic_action_ms:.1f} "
            f"anchor_ms={rows[-1].blockchain_store_ms:.1f} "
            f"val_ms={rows[-1].validation_ms:.1f} "
            f"ok={executed}"
        )

    total_ms = _perf_ms(started, time.perf_counter())

    out_rows = [asdict(r) for r in rows]
    out_df = pd.DataFrame(out_rows)

    # Summary
    def _mean(col: str) -> float:
        return float(out_df[col].mean()) if len(out_df) else 0.0

    print("\n=== Summary (ms) ===")
    print(f"rows={len(out_df)} total_ms={total_ms:.1f}")
    if len(out_df):
        print(f"agentic_action_mean_ms={_mean('agentic_action_ms'):.1f}")
        print(f"blockchain_store_mean_ms={_mean('blockchain_store_ms'):.1f}")
        print(f"validation_mean_ms={_mean('validation_ms'):.1f}")
        print(f"end_to_end_mean_ms={_mean('end_to_end_ms'):.1f}")

    # Output path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = settings.storage_root / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _now_utc().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"pragma_benchmark_{csv_path.stem}_{stamp}.csv"

    out_df.to_csv(out_path, index=False)
    print(f"\nWrote timing report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

