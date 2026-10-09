"""Trust chain: plain-text mitigation plans and apply receipts on a local JSON-RPC chain.

On-chain per plan (``AgenticTrustRegistry.storePlan`` / ``revisePlan``): job id, prediction id,
row index, attack type, threat level, primary/supporting ``{action, tier}`` units,
``sha256(overall_reasoning)``, and the revision link (parent plan id, origin, superseded).
The reasoning text stays off-chain.

Signing keys: the planner agent (``TRUST_CHAIN_PRIVATE_KEY``) stores plans and agent re-plans,
the reviewer key commits human corrections, and each tier executor key applies actions on its tier.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from web3 import Web3

from app.core.config import Settings
from scripts.network_domains import DOMAIN_LABELS, normalize_domain

logger = logging.getLogger(__name__)

MAX_ACTIONS_PER_LIST = 10
THREAT_LEVELS: tuple[str, ...] = ("Critical", "High", "Medium", "Low")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def reasoning_hash_sha256(overall_reasoning: str | None) -> str:
    """sha256 of the plan's ``overall_reasoning`` text (stripped)."""
    return _sha256_hex(str(overall_reasoning or "").strip())


# ------------------------------------------------------------------ plan input


@dataclass(frozen=True)
class PlanInput:
    job_id: str
    prediction_id: str
    row_index: int
    attack_type: str
    threat_level: str
    primary_actions: tuple[tuple[str, str], ...]
    supporting_actions: tuple[tuple[str, str], ...]
    reasoning_hash: str  # 64 hex chars

    def as_contract_tuple(self) -> tuple[Any, ...]:
        return (
            self.job_id,
            self.prediction_id,
            int(self.row_index),
            self.attack_type,
            self.threat_level,
            [list(u) for u in self.primary_actions],
            [list(u) for u in self.supporting_actions],
            _bytes32_hex_from_hex(self.reasoning_hash),
        )


def _normalize_threat_level(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    for level in THREAT_LEVELS:
        if s == level.lower():
            return level
    raise ValueError(f"threat_level must be one of {THREAT_LEVELS}, got {raw!r}")


def _units_from_block(block: Any, kind: str) -> tuple[tuple[str, str], ...]:
    if block is None:
        return ()
    if not isinstance(block, list):
        raise ValueError(f"{kind} must be a list")
    units: list[tuple[str, str]] = []
    for item in block:
        if not isinstance(item, dict):
            raise ValueError(f"{kind} entries must be objects")
        action = str(item.get("action") or "").strip()
        if not action:
            raise ValueError(f"{kind} entry has an empty action")
        raw_tier = str(item.get("network_tier") or "").strip()
        tier = normalize_domain(raw_tier)
        if tier not in DOMAIN_LABELS:
            raise ValueError(f"{kind} action {action!r} has unknown network_tier {raw_tier!r}")
        units.append((action, tier))
    if len(units) > MAX_ACTIONS_PER_LIST:
        raise ValueError(f"{kind} has {len(units)} actions (max {MAX_ACTIONS_PER_LIST})")
    return tuple(units)


def build_plan_input(
    *,
    structured_plan: Any,
    job_id: str | None,
    prediction_id: str,
    row_index: int | None,
    attack_type: str | None,
) -> PlanInput:
    """Map a structured mitigation plan onto the ``storePlan`` input. Raises ValueError if invalid."""
    if not isinstance(structured_plan, dict):
        raise ValueError("structured_plan missing or not an object")
    if not attack_type or not str(attack_type).strip():
        raise ValueError("attack_type missing")
    primary = _units_from_block(structured_plan.get("primary_actions"), "primary_actions")
    supporting = _units_from_block(structured_plan.get("supporting_actions"), "supporting_actions")
    seen: set[tuple[str, str]] = set()
    for unit in primary + supporting:
        if unit in seen:
            raise ValueError(f"duplicate action {unit[0]!r} on {unit[1]!r}")
        seen.add(unit)
    return PlanInput(
        job_id=(str(job_id).strip() if job_id and str(job_id).strip() else "unlinked"),
        prediction_id=str(prediction_id).strip(),
        row_index=int(row_index) if row_index is not None else -1,
        attack_type=str(attack_type).strip(),
        threat_level=_normalize_threat_level(structured_plan.get("threat_level")),
        primary_actions=primary,
        supporting_actions=supporting,
        reasoning_hash=reasoning_hash_sha256(structured_plan.get("overall_reasoning")),
    )


def plan_input_matches_chain(expected: PlanInput, on_chain: dict[str, Any]) -> list[str]:
    """Field names where the on-chain plan differs from ``expected`` (empty list = identical)."""
    diffs: list[str] = []
    pairs = (
        ("job_id", expected.job_id),
        ("prediction_id", expected.prediction_id),
        ("row_index", expected.row_index),
        ("attack_type", expected.attack_type),
        ("threat_level", expected.threat_level),
        ("primary_actions", list(expected.primary_actions)),
        ("supporting_actions", list(expected.supporting_actions)),
        ("reasoning_hash", expected.reasoning_hash.lower()),
    )
    for name, want in pairs:
        if on_chain.get(name) != want:
            diffs.append(name)
    return diffs


# ----------------------------------------------------------------------- ABI

_ACTION_UNIT = [
    {"internalType": "string", "name": "action", "type": "string"},
    {"internalType": "string", "name": "tier", "type": "string"},
]

_PLAN_INPUT_COMPONENTS = [
    {"internalType": "string", "name": "jobId", "type": "string"},
    {"internalType": "string", "name": "predictionId", "type": "string"},
    {"internalType": "int256", "name": "rowIndex", "type": "int256"},
    {"internalType": "string", "name": "attackType", "type": "string"},
    {"internalType": "string", "name": "threatLevel", "type": "string"},
    {
        "internalType": "struct AgenticTrustRegistry.ActionUnit[]",
        "name": "primaryActions",
        "type": "tuple[]",
        "components": _ACTION_UNIT,
    },
    {
        "internalType": "struct AgenticTrustRegistry.ActionUnit[]",
        "name": "supportingActions",
        "type": "tuple[]",
        "components": _ACTION_UNIT,
    },
    {"internalType": "bytes32", "name": "reasoningHash", "type": "bytes32"},
]

_PLAN_COMPONENTS = _PLAN_INPUT_COMPONENTS + [
    {"internalType": "address", "name": "storedBy", "type": "address"},
    {"internalType": "uint256", "name": "storedAt", "type": "uint256"},
    {"internalType": "bool", "name": "exists", "type": "bool"},
    {"internalType": "string", "name": "parentPlanId", "type": "string"},
    {"internalType": "string", "name": "origin", "type": "string"},
    {"internalType": "bool", "name": "superseded", "type": "bool"},
]

_STR = {"internalType": "string", "type": "string"}
_PLAN_INPUT_ARG = {
    "internalType": "struct AgenticTrustRegistry.PlanInput",
    "name": "p",
    "type": "tuple",
    "components": _PLAN_INPUT_COMPONENTS,
}

_REGISTRY_ABI = [
    {
        "inputs": [{**_STR, "name": "planId"}, _PLAN_INPUT_ARG],
        "name": "storePlan",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "planId"}, {**_STR, "name": "parentPlanId"}, _PLAN_INPUT_ARG],
        "name": "revisePlan",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": ""}],
        "name": "tierExecutor",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "", "type": "address"}],
        "name": "reviewers",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "planId"}],
        "name": "getPlan",
        "outputs": [
            {
                "internalType": "struct AgenticTrustRegistry.Plan",
                "name": "",
                "type": "tuple",
                "components": _PLAN_COMPONENTS,
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": ""}, {**_STR, "name": ""}],
        "name": "whitelist",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "attackType"}],
        "name": "getAllowedActions",
        "outputs": [{"internalType": "string[]", "name": "", "type": "string[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "planId"}, {**_STR, "name": "action"}, {**_STR, "name": "tier"}],
        "name": "isInPlan",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "planId"}, {**_STR, "name": "action"}, {**_STR, "name": "tier"}],
        "name": "markApplied",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{**_STR, "name": "planId"}, {**_STR, "name": "action"}, {**_STR, "name": "tier"}],
        "name": "isApplied",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
]


def _bytes32_hex_from_hex(hex64: str) -> str:
    h = (hex64 or "").strip().lower()
    if h.startswith("0x"):
        h = h[2:]
    if len(h) != 64:
        raise ValueError("expected 32-byte hex (64 chars)")
    return "0x" + h


def _registry_contract(settings: Settings, contract_address: str | None = None):
    addr_raw = contract_address or settings.trust_chain_contract_address
    if not addr_raw:
        raise RuntimeError("TRUST_CHAIN_CONTRACT_ADDRESS missing")
    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        raise RuntimeError("could not connect to TRUST_CHAIN_RPC_URL")
    addr = Web3.to_checksum_address(addr_raw.strip())
    return w3, w3.eth.contract(address=addr, abi=_REGISTRY_ABI)


_REASON_STRING_RE = re.compile(r"reverted with reason string '([^']+)'")
_REVERT_RE = re.compile(r"execution reverted:?\s*(?!Error\b)([A-Za-z_][A-Za-z0-9_=]*)")


def revert_reason(err: Any) -> str:
    """Short contract revert reason (e.g. ``action_not_whitelisted``), else the trimmed error text."""
    msg = str(err)
    named = _REASON_STRING_RE.search(msg)
    if named:
        return named.group(1)
    m = _REVERT_RE.search(msg)
    return m.group(1) if m else msg[:500]


def address_for_key(private_key: str | None) -> str | None:
    if not private_key:
        return None
    return Web3().eth.account.from_key(private_key).address


def _send_and_wait(
    settings: Settings, w3: Web3, fn: Any, *, gas: int, private_key: str | None = None
) -> str:
    """Sign, send, and wait for a mined receipt. Raises RuntimeError on revert."""
    acct = w3.eth.account.from_key(private_key or settings.trust_chain_private_key)
    tx = fn.build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "chainId": int(settings.trust_chain_chain_id),
        }
    )
    tx.setdefault("gas", gas)
    tx.setdefault("maxFeePerGas", w3.to_wei(2, "gwei"))
    tx.setdefault("maxPriorityFeePerGas", w3.to_wei(1, "gwei"))
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    if int(receipt.status) != 1:
        raise RuntimeError(f"transaction reverted: {tx_hash.hex()}")
    return tx_hash.hex()


# ---------------------------------------------------------------- plan calls


def _plan_write_fn(contract: Any, plan_id: str, plan: PlanInput, parent_plan_id: str | None) -> Any:
    if parent_plan_id:
        return contract.functions.revisePlan(plan_id, parent_plan_id, plan.as_contract_tuple())
    return contract.functions.storePlan(plan_id, plan.as_contract_tuple())


def preflight_plan(
    settings: Settings,
    *,
    plan: PlanInput,
    parent_plan_id: str | None = None,
    private_key: str | None = None,
) -> str | None:
    """Dry-run ``storePlan`` / ``revisePlan`` (eth_call). Returns the revert reason, or None if it would pass."""
    if not settings.trust_chain_enabled:
        return None
    key = private_key or settings.trust_chain_private_key
    if not key:
        return "TRUST_CHAIN_PRIVATE_KEY missing"
    try:
        w3, contract = _registry_contract(settings)
        fn = _plan_write_fn(contract, f"preflight_{time.time_ns()}", plan, parent_plan_id)
        fn.call({"from": w3.eth.account.from_key(key).address})
        return None
    except Exception as e:
        return revert_reason(e)


def store_plan_on_chain(
    settings: Settings,
    *,
    plan_id: str,
    plan: PlanInput,
    parent_plan_id: str | None = None,
    private_key: str | None = None,
) -> tuple[str, str, float]:
    """
    Submit ``storePlan`` (or ``revisePlan`` when ``parent_plan_id`` is set).

    ``private_key`` defaults to the planner agent key; pass the reviewer key for human corrections.
    Returns (tx_hash, contract_address, store_ms).
    """
    if not settings.trust_chain_enabled:
        raise RuntimeError("trust chain disabled")
    key = private_key or settings.trust_chain_private_key
    if not key:
        raise RuntimeError("TRUST_CHAIN_PRIVATE_KEY missing")
    w3, contract = _registry_contract(settings)
    fn = _plan_write_fn(contract, plan_id, plan, parent_plan_id)
    try:
        fn.call({"from": w3.eth.account.from_key(key).address})
    except Exception as e:
        raise RuntimeError(revert_reason(e)) from e
    started = time.perf_counter()
    tx_hash = _send_and_wait(settings, w3, fn, gas=2_500_000, private_key=key)
    store_ms = _elapsed_ms(started)
    logger.info(
        "trust_chain %s tx=%s plan=%s parent=%s job=%s",
        "revisePlan" if parent_plan_id else "storePlan",
        tx_hash,
        plan_id,
        parent_plan_id,
        plan.job_id,
    )
    return tx_hash, contract.address, store_ms


def get_allowed_actions_on_chain(settings: Settings, attack_type: str) -> list[str]:
    """Readable whitelist for one attack type ([] when unavailable)."""
    try:
        _w3, contract = _registry_contract(settings)
        return [str(a) for a in contract.functions.getAllowedActions(attack_type.strip()).call()]
    except Exception:
        return []


def _plan_from_tuple(raw: Any) -> dict[str, Any]:
    (
        job_id,
        prediction_id,
        row_index,
        attack_type,
        threat_level,
        primary,
        supporting,
        reasoning_hash,
        stored_by,
        stored_at,
        exists,
        parent_plan_id,
        origin,
        superseded,
    ) = raw
    rh = reasoning_hash.hex() if isinstance(reasoning_hash, (bytes, bytearray)) else str(reasoning_hash)
    return {
        "job_id": job_id,
        "prediction_id": prediction_id,
        "row_index": int(row_index),
        "attack_type": attack_type,
        "threat_level": threat_level,
        "primary_actions": [(str(a), str(t)) for a, t in primary],
        "supporting_actions": [(str(a), str(t)) for a, t in supporting],
        "reasoning_hash": rh.removeprefix("0x").lower(),
        "stored_by": stored_by,
        "stored_at": int(stored_at),
        "exists": bool(exists),
        "parent_plan_id": str(parent_plan_id),
        "origin": str(origin),
        "superseded": bool(superseded),
    }


def read_plan_from_chain(
    settings: Settings,
    *,
    contract_address: str,
    plan_id: str,
) -> tuple[bool, dict[str, Any] | None, str | None, float]:
    """Call ``getPlan``. Returns (rpc_ok, plan_dict, error, verify_ms); plan is None when not stored."""
    if not settings.trust_chain_rpc_url:
        return False, None, "TRUST_CHAIN_RPC_URL missing", 0.0
    started = time.perf_counter()
    try:
        _w3, contract = _registry_contract(settings, contract_address)
    except Exception as e:
        return False, None, str(e)[:500], _elapsed_ms(started)
    try:
        raw = contract.functions.getPlan(plan_id).call()
    except Exception as e:
        msg = str(e)
        if "not_stored" in msg:
            return True, None, "plan not stored on chain", _elapsed_ms(started)
        return False, None, msg[:500], _elapsed_ms(started)
    return True, _plan_from_tuple(raw), None, _elapsed_ms(started)


# --------------------------------------------------------------- apply calls


def is_action_whitelisted_on_chain(
    settings: Settings,
    *,
    attack_type: str,
    action: str,
) -> tuple[bool | None, str | None]:
    """Gate 1: ``whitelist[attack_type][action]``. allowed is None when the RPC call failed."""
    if not settings.trust_chain_rpc_url or not settings.trust_chain_contract_address:
        return None, "trust chain not configured"
    try:
        _w3, contract = _registry_contract(settings)
        allowed = contract.functions.whitelist(attack_type.strip(), action.strip()).call()
        return bool(allowed), None
    except Exception as e:
        return None, str(e)[:500]


def is_action_in_plan_on_chain(
    settings: Settings,
    *,
    plan_id: str,
    action: str,
    tier: str,
) -> tuple[bool | None, str | None]:
    """Gate 2: the ``{action, tier}`` unit is in the stored plan. None when the RPC call failed."""
    if not settings.trust_chain_rpc_url or not settings.trust_chain_contract_address:
        return None, "trust chain not configured"
    try:
        _w3, contract = _registry_contract(settings)
        ok = contract.functions.isInPlan(plan_id, action.strip(), tier.strip()).call()
        return bool(ok), None
    except Exception as e:
        return None, str(e)[:500]


def apply_action_on_chain(
    settings: Settings,
    *,
    plan_id: str,
    action: str,
    tier: str,
    private_key: str | None,
) -> tuple[str | None, str | None]:
    """
    Submit ``markApplied`` signed by the tier executor's key.

    Returns (tx_hash, revert_reason_or_error). The contract accepts only the key registered
    for ``tier`` via ``setTierExecutor``.
    """
    if not settings.trust_chain_enabled:
        return None, "trust chain disabled"
    if not private_key:
        return None, f"executor key missing for {tier}"
    try:
        w3, contract = _registry_contract(settings)
        fn = contract.functions.markApplied(plan_id, action.strip(), tier.strip())
        fn.call({"from": w3.eth.account.from_key(private_key).address})
        return _send_and_wait(settings, w3, fn, gas=250_000, private_key=private_key), None
    except Exception as e:
        return None, revert_reason(e)


def tier_executor_on_chain(settings: Settings, tier: str) -> str | None:
    """Address registered for ``tier`` (None when unset or unreachable)."""
    try:
        _w3, contract = _registry_contract(settings)
        addr = contract.functions.tierExecutor(tier).call()
        return None if int(addr, 16) == 0 else str(addr)
    except Exception:
        return None


# -------------------------------------------------------------------- deploy


def _repo_root() -> Path:
    """Repository root (parent of ``backend``)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def _hardhat_blockchain_dir() -> Path:
    return _repo_root() / "hardhat-blockchain"


def _attack_options_path() -> Path:
    return _hardhat_blockchain_dir() / "contracts" / "attack_options.json"


def _seed_whitelist(settings: Settings, w3: Web3, contract: Any) -> int:
    """Seed ``addAllowedActions`` from contracts/attack_options.json. Returns action slots written."""
    data = json.loads(_attack_options_path().read_text(encoding="utf-8"))
    attacks = data.get("attacks") or {}
    total = 0
    for attack_type, actions in attacks.items():
        if not isinstance(actions, list) or not actions:
            continue
        labels = [str(a) for a in actions]
        _send_and_wait(
            settings, w3, contract.functions.addAllowedActions(str(attack_type), labels), gas=3_000_000
        )
        total += len(labels)
    return total


def _deploy_registry_from_artifact(settings: Settings, artifact_path: Path) -> str:
    """Deploy using the compiled Hardhat artifact, then seed the whitelist."""
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
    tx = factory.constructor().build_transaction(
        {
            "from": acct.address,
            "nonce": w3.eth.get_transaction_count(acct.address),
            "chainId": int(settings.trust_chain_chain_id),
        }
    )
    tx.setdefault("gas", 5_000_000)
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
    addr = Web3.to_checksum_address(addr)

    contract = w3.eth.contract(address=addr, abi=abi)
    slots = _seed_whitelist(settings, w3, contract)
    logger.info("seeded %d whitelist slot(s) on %s", slots, addr)
    _configure_roles(settings, w3, contract)
    return addr


def executor_keys_by_tier(settings: Settings) -> dict[str, str | None]:
    """Configured executor private key per canonical tier label."""
    from scripts.network_domains import ACCESS_ISP, ENDPOINT_EDR, PERIMETER_IDS

    return {
        ACCESS_ISP: settings.executor_access_private_key,
        PERIMETER_IDS: settings.executor_perimeter_private_key,
        ENDPOINT_EDR: settings.executor_endpoint_private_key,
    }


def _configure_roles(settings: Settings, w3: Web3, contract: Any) -> None:
    """Grant the reviewer role and register one executor address per tier (keys from settings)."""
    reviewer = address_for_key(settings.trust_chain_reviewer_private_key)
    if reviewer:
        _send_and_wait(settings, w3, contract.functions.setReviewer(reviewer, True), gas=100_000)
        logger.info("reviewer role -> %s", reviewer)
    for tier, key in executor_keys_by_tier(settings).items():
        addr = address_for_key(key)
        if addr:
            _send_and_wait(settings, w3, contract.functions.setTierExecutor(tier, addr), gas=120_000)
            logger.info("tier executor %s -> %s", tier, addr)


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
    Deploy a new, seeded AgenticTrustRegistry (whitelist + reviewer + tier executors)
    so each run starts with no stored plans.

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
