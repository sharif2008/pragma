"""Trust anchoring: hash-only commitments written to a local JSON-RPC chain."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from web3 import Web3

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _bytes32_hex_from_hex(hex64: str) -> str:
    h = (hex64 or "").strip().lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ValueError("expected 32-byte hex (64 chars)")
    return "0x" + h


def compute_trust_commitment_sha256(
    *,
    payload_version: str,
    agentic_report_public_id: str,
    prediction_job_public_id: str,
    results_row_index: int | None,
    created_at: datetime,
    raw_llm_response: str | None,
    rag_context_used: str | None,
    structured_plan: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Deterministic commitment hash over a canonical JSON payload (hash-only on-chain).

    Returns (commitment_sha256_hex, canonical_payload_dict).
    """
    payload: dict[str, Any] = {
        "payload_version": payload_version,
        "agentic_report_public_id": agentic_report_public_id,
        "prediction_job_public_id": prediction_job_public_id,
        "results_row_index": results_row_index,
        "created_at_utc": created_at.astimezone(timezone.utc).isoformat(),
        "raw_llm_response": raw_llm_response,
        "rag_context_used": rag_context_used,
        "structured_plan": structured_plan,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_hex(canonical), payload


# Minimal ABI for AgenticTrustRegistry
_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "agentKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "reportKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "commitment", "type": "bytes32"},
        ],
        "name": "anchor",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "agentKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "reportKey", "type": "bytes32"},
        ],
        "name": "getCommitment",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _agent_key_sha256(agentic_job_public_id: str | None) -> str:
    return _sha256_hex((agentic_job_public_id or "unlinked").strip() or "unlinked")


def _report_key_sha256(agentic_report_public_id: str) -> str:
    return _sha256_hex(agentic_report_public_id.strip())


def anchor_report_commitment_on_chain(
    *,
    settings: Settings,
    agentic_job_public_id: str | None,
    agentic_report_public_id: str,
    commitment_sha256_hex: str,
) -> tuple[str, str, str, str]:
    """
    Submit an on-chain anchor tx.

    Returns: (tx_hash, contract_address, agent_key_sha256, report_key_sha256)
    """
    if not settings.trust_chain_enabled:
        raise RuntimeError("trust chain disabled")
    if not settings.trust_chain_private_key:
        raise RuntimeError("TRUST_CHAIN_PRIVATE_KEY missing")
    if not settings.trust_chain_contract_address:
        raise RuntimeError("TRUST_CHAIN_CONTRACT_ADDRESS missing")

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("could not connect to TRUST_CHAIN_RPC_URL")

    acct = w3.eth.account.from_key(settings.trust_chain_private_key)
    contract_addr = Web3.to_checksum_address(settings.trust_chain_contract_address)
    contract = w3.eth.contract(address=contract_addr, abi=_REGISTRY_ABI)

    agent_key_hex = _agent_key_sha256(agentic_job_public_id)
    report_key_hex = _report_key_sha256(agentic_report_public_id)
    agent_key_b32 = _bytes32_hex_from_hex(agent_key_hex)
    report_key_b32 = _bytes32_hex_from_hex(report_key_hex)
    commitment_b32 = _bytes32_hex_from_hex(commitment_sha256_hex)

    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract.functions.anchor(agent_key_b32, report_key_b32, commitment_b32).build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "chainId": int(settings.trust_chain_chain_id),
        }
    )
    # Hardhat supports EIP-1559; Web3 will estimate if provided. Keep it simple for local.
    tx.setdefault("gas", 200_000)
    tx.setdefault("maxFeePerGas", w3.to_wei(2, "gwei"))
    tx.setdefault("maxPriorityFeePerGas", w3.to_wei(1, "gwei"))

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    hex_hash = tx_hash.hex()

    logger.info(
        "trust_anchor submitted tx=%s report=%s agent=%s",
        hex_hash,
        agentic_report_public_id,
        (agentic_job_public_id or "unlinked"),
    )
    return hex_hash, contract_addr, agent_key_hex, report_key_hex

