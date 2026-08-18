"""Trust anchoring: hash-only commitments written to a local JSON-RPC chain."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from web3 import Web3

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _elapsed_ms(started: float) -> float:
    # REVIEW: Shared wall-clock helper for store (anchor) and getCommitment verify ms.
    return round((time.perf_counter() - started) * 1000.0, 3)


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
    {
        "inputs": [{"internalType": "string", "name": "attackType", "type": "string"}],
        "name": "attackKey",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "pure",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "string", "name": "action", "type": "string"}],
        "name": "actionKey",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "pure",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "attackKey_", "type": "bytes32"},
            {"internalType": "bytes32", "name": "actionKey_", "type": "bytes32"},
        ],
        "name": "isActionWhitelisted",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "attackKey_", "type": "bytes32"},
            {"internalType": "bytes32", "name": "actionKey_", "type": "bytes32"},
            {"internalType": "bytes32", "name": "agentKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "reportKey", "type": "bytes32"},
        ],
        "name": "applyAction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "agentKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "reportKey", "type": "bytes32"},
            {"internalType": "bytes32", "name": "actionKey_", "type": "bytes32"},
        ],
        "name": "isActionApplied",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _registry_contract(settings: Settings):
    if not settings.trust_chain_contract_address:
        raise RuntimeError("TRUST_CHAIN_CONTRACT_ADDRESS missing")
    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("could not connect to TRUST_CHAIN_RPC_URL")
    addr = Web3.to_checksum_address(settings.trust_chain_contract_address.strip())
    return w3, w3.eth.contract(address=addr, abi=_REGISTRY_ABI)


def _keccak256_utf8_key_hex64(label: str) -> str:
    """keccak256(bytes(utf8)) — matches AgenticTrustRegistry.attackKey/actionKey."""
    return Web3.keccak(text=(label or "").strip()).hex().removeprefix("0x").lower()


def is_action_whitelisted_on_chain(
    settings: Settings,
    *,
    attack_type: str,
    action: str,
) -> tuple[bool | None, str | None]:
    """
    Check registry whitelist for attack_type + action (attack_options.json labels).
    Returns (allowed, error). allowed is None when the RPC call failed.
    """
    if not settings.trust_chain_rpc_url or not settings.trust_chain_contract_address:
        return None, "trust chain not configured"
    try:
        _w3, contract = _registry_contract(settings)
        attack_b32 = _bytes32_hex_from_hex(_keccak256_utf8_key_hex64(attack_type))
        action_b32 = _bytes32_hex_from_hex(_keccak256_utf8_key_hex64(action))
        allowed = contract.functions.isActionWhitelisted(attack_b32, action_b32).call()
        return bool(allowed), None
    except Exception as e:
        return None, str(e)[:500]


def apply_action_on_chain(
    settings: Settings,
    *,
    attack_type: str,
    action: str,
    agentic_job_public_id: str | None,
    agentic_report_public_id: str,
) -> tuple[str | None, str | None]:
    """
    Submit applyAction() after anchor + whitelist check.
    Returns (tx_hash, error_message).
    """
    if not settings.trust_chain_enabled:
        return None, "trust chain disabled"
    if not settings.trust_chain_private_key:
        return None, "TRUST_CHAIN_PRIVATE_KEY missing"
    try:
        w3, contract = _registry_contract(settings)
        acct = w3.eth.account.from_key(settings.trust_chain_private_key)
        attack_b32 = _bytes32_hex_from_hex(_keccak256_utf8_key_hex64(attack_type))
        action_b32 = _bytes32_hex_from_hex(_keccak256_utf8_key_hex64(action))
        agent_b32 = _bytes32_hex_from_hex(_agent_key_sha256(agentic_job_public_id))
        report_b32 = _bytes32_hex_from_hex(_report_key_sha256(agentic_report_public_id))

        nonce = w3.eth.get_transaction_count(acct.address)
        tx = contract.functions.applyAction(attack_b32, action_b32, agent_b32, report_b32).build_transaction(
            {
                "from": acct.address,
                "nonce": nonce,
                "chainId": int(settings.trust_chain_chain_id),
            }
        )
        tx.setdefault("gas", 250_000)
        tx.setdefault("maxFeePerGas", w3.to_wei(2, "gwei"))
        tx.setdefault("maxPriorityFeePerGas", w3.to_wei(1, "gwei"))
        signed = acct.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex(), None
    except Exception as e:
        return None, str(e)[:500]


def _agent_key_sha256(agentic_job_public_id: str | None) -> str:
    return _sha256_hex((agentic_job_public_id or "unlinked").strip() or "unlinked")


def _report_key_sha256(agentic_report_public_id: str) -> str:
    return _sha256_hex(agentic_report_public_id.strip())


def _repo_root() -> Path:
    """Repository root (parent of ``backend``)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _hardhat_blockchain_dir() -> Path:
    return _repo_root() / "hardhat-blockchain"


def _deploy_registry_from_artifact(settings: Settings, artifact_path: Path) -> str:
    """Deploy using compiled Hardhat artifact (bytecode + ABI)."""
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    abi = data.get("abi")
    bytecode = data.get("bytecode") or ""
    if not abi or not bytecode or bytecode == "0x":
        raise RuntimeError(
            "Artifact missing bytecode. Run: cd hardhat-blockchain && npx hardhat compile"
        )

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("could not connect to TRUST_CHAIN_RPC_URL")
    acct = w3.eth.account.from_key(settings.trust_chain_private_key)

    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = factory.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "chainId": int(settings.trust_chain_chain_id),
        }
    )
    tx.setdefault("gas", 2_000_000)
    tx.setdefault("maxFeePerGas", w3.to_wei(100, "gwei"))
    tx.setdefault("maxPriorityFeePerGas", w3.to_wei(10, "gwei"))

    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    addr = getattr(receipt, "contractAddress", None) or getattr(receipt, "contract_address", None)
    if addr is None and isinstance(receipt, dict):
        addr = receipt.get("contractAddress") or receipt.get("contract_address")
    if not addr:
        raise RuntimeError("deploy receipt missing contractAddress")
    return Web3.to_checksum_address(addr)


def _deploy_registry_via_npm(settings: Settings) -> str:
    """Run ``npm run deploy:local`` in ``hardhat-blockchain`` and parse the deployed address."""
    hh = _hardhat_blockchain_dir()
    pkg = hh / "package.json"
    if not pkg.is_file():
        raise RuntimeError(f"hardhat-blockchain not found or missing package.json: {hh}")

    proc = subprocess.run(
        ["npm", "run", "deploy:local"],
        cwd=str(hh),
        capture_output=True,
        text=True,
        timeout=300,
        encoding="utf-8",
        errors="replace",
    )
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"npm deploy failed (exit {proc.returncode}):\n{out[-6000:]}")

    m = re.search(r"AgenticTrustRegistry deployed to:\s*(0x[a-fA-F0-9]{40})", out)
    if not m:
        raise RuntimeError(f"Could not parse deployed contract address from npm output:\n{out[-4000:]}")
    return Web3.to_checksum_address(m.group(1))


def deploy_fresh_registry(settings: Settings) -> str:
    """
    Deploy a new AgenticTrustRegistry so each run starts with an empty commitment mapping.

    Uses the compiled artifact + JSON-RPC when available; otherwise runs ``npm run deploy:local``.
    Requires ``TRUST_CHAIN_PRIVATE_KEY`` and a reachable ``TRUST_CHAIN_RPC_URL``.
    """
    if not settings.trust_chain_private_key:
        raise RuntimeError("TRUST_CHAIN_PRIVATE_KEY missing")

    art = (
        _hardhat_blockchain_dir()
        / "artifacts"
        / "contracts"
        / "AgenticTrustRegistry.sol"
        / "AgenticTrustRegistry.json"
    )
    if art.is_file():
        try:
            addr = _deploy_registry_from_artifact(settings, art)
            logger.info("deployed AgenticTrustRegistry via artifact at %s", addr)
            return addr
        except Exception as e:
            logger.warning("artifact deploy failed (%s); trying npm run deploy:local", e)

    addr = _deploy_registry_via_npm(settings)
    logger.info("deployed AgenticTrustRegistry via npm at %s", addr)
    return addr


def anchor_report_commitment_on_chain(
    *,
    settings: Settings,
    agentic_job_public_id: str | None,
    agentic_report_public_id: str,
    commitment_sha256_hex: str,
) -> tuple[str, str, str, str, float]:
    """
    Submit an on-chain anchor tx.

    Returns: (tx_hash, contract_address, agent_key_sha256, report_key_sha256, anchor_ms)
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
    started = time.perf_counter()
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
    # REVIEW: Store (anchor) latency is RPC wall-clock only, not hash canonicalization.
    anchor_ms = _elapsed_ms(started)

    logger.info(
        "trust_anchor submitted tx=%s report=%s agent=%s",
        hex_hash,
        agentic_report_public_id,
        (agentic_job_public_id or "unlinked"),
    )
    return hex_hash, contract_addr, agent_key_hex, report_key_hex, anchor_ms


def read_commitment_from_chain(
    settings: Settings,
    *,
    contract_address: str,
    agent_key_sha256_hex: str,
    report_key_sha256_hex: str,
) -> tuple[bool, str | None, str | None, float]:
    """
    Call getCommitment on the registry.
    Returns (rpc_ok, commitment_hex_lowercase_64chars, error_message, verify_ms).
    """
    if not settings.trust_chain_rpc_url:
        return False, None, "TRUST_CHAIN_RPC_URL missing", 0.0
    started = time.perf_counter()
    try:
        w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
        if not w3.is_connected():
            return False, None, "could not connect to TRUST_CHAIN_RPC_URL", _elapsed_ms(started)
        addr = Web3.to_checksum_address(contract_address.strip())
        contract = w3.eth.contract(address=addr, abi=_REGISTRY_ABI)
        agent_b32 = _bytes32_hex_from_hex(agent_key_sha256_hex)
        report_b32 = _bytes32_hex_from_hex(report_key_sha256_hex)
        raw = contract.functions.getCommitment(agent_b32, report_b32).call()
        if isinstance(raw, (bytes, bytearray)):
            h = raw.hex()
        else:
            h = str(raw).removeprefix("0x")
        h = h.lower()
        if len(h) != 64:
            return True, None, f"unexpected bytes32 length from RPC: {len(h)}", _elapsed_ms(started)
        return True, h, None, _elapsed_ms(started)
    except Exception as e:
        return False, None, str(e)[:500], _elapsed_ms(started)

