"""Smoke test for local Hardhat trust-chain config.

Usage (from backend/):
  python scripts/test_trust_chain.py
  python scripts/test_trust_chain.py --store-test
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

from web3 import Web3

# Allow running from repo root or backend/ by ensuring backend/ is on sys.path.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402
from app.services import trust_chain_service  # noqa: E402

SAMPLE_PLAN = {
    "threat_level": "High",
    "primary_actions": [
        {"action": "tarpit scan", "network_tier": "Perimeter / IDS"},
        {"action": "harden ports", "network_tier": "Perimeter / IDS"},
    ],
    "supporting_actions": [{"action": "update ACL", "network_tier": "Access / ISP"}],
    "overall_reasoning": "Perimeter share dominates; the perimeter acts first and Access tightens the ACL.",
}


def _strip_opt(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    return t or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--store-test",
        action="store_true",
        help="Store a sample PORTSCAN plan, read it back, and mark one action applied.",
    )
    args = ap.parse_args()

    settings = get_settings()
    contract = _strip_opt(settings.trust_chain_contract_address)
    private_key = _strip_opt(settings.trust_chain_private_key)

    print("TRUST_CHAIN_ENABLED:", settings.trust_chain_enabled)
    print("TRUST_CHAIN_RPC_URL:", settings.trust_chain_rpc_url)
    print("TRUST_CHAIN_CHAIN_ID:", settings.trust_chain_chain_id)
    print("TRUST_CHAIN_CONTRACT_ADDRESS:", contract or "(not set)")
    print("TRUST_CHAIN_PRIVATE_KEY:", "(set)" if private_key else "(not set)")

    if not settings.trust_chain_enabled:
        print("ERROR: TRUST_CHAIN_ENABLED is false.")
        return 2
    if not contract:
        print("ERROR: TRUST_CHAIN_CONTRACT_ADDRESS is missing.")
        return 2
    if args.store_test and not private_key:
        print("ERROR: TRUST_CHAIN_PRIVATE_KEY is missing (required for --store-test).")
        return 2

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        print("ERROR: Could not connect to RPC.")
        return 3

    chain_id = int(w3.eth.chain_id)
    print("RPC chain_id:", chain_id)
    if chain_id != int(settings.trust_chain_chain_id):
        print("WARN: RPC chain_id != TRUST_CHAIN_CHAIN_ID")

    code = w3.eth.get_code(Web3.to_checksum_address(contract))
    print("Contract code bytes:", len(code))
    if not code or code == b"\x00":
        print("ERROR: No contract code at TRUST_CHAIN_CONTRACT_ADDRESS (did you deploy?)")
        return 4

    allowed, err = trust_chain_service.is_action_whitelisted_on_chain(
        settings, attack_type="PORTSCAN", action="tarpit scan"
    )
    print("Read test whitelist(PORTSCAN, tarpit scan) ->", allowed, err or "")
    if allowed is not True:
        print("ERROR: whitelist not seeded (run npm run seed:whitelist in hardhat-blockchain/)")
        return 4

    if not args.store_test:
        print("OK: Connected + contract readable. (Use --store-test to send transactions.)")
        return 0

    plan_id = f"smoke_{uuid.uuid4().hex[:12]}"
    plan = trust_chain_service.build_plan_input(
        structured_plan=SAMPLE_PLAN,
        job_id="smoke_job",
        prediction_id="smoke_prediction",
        row_index=0,
        attack_type="PORTSCAN",
    )
    tx_hash, addr, store_ms = trust_chain_service.store_plan_on_chain(settings, plan_id=plan_id, plan=plan)
    print(f"storePlan tx_hash: {tx_hash} ({store_ms} ms)")

    rpc_ok, on_chain, err, verify_ms = trust_chain_service.read_plan_from_chain(
        settings, contract_address=addr, plan_id=plan_id
    )
    print(f"getPlan ({verify_ms} ms) ->", on_chain if rpc_ok else err)
    diffs = trust_chain_service.plan_input_matches_chain(plan, on_chain or {})
    if diffs:
        print("ERROR: on-chain plan differs in:", ", ".join(diffs))
        return 5

    in_plan, _ = trust_chain_service.is_action_in_plan_on_chain(
        settings, plan_id=plan_id, action="limit rate", tier="Perimeter / IDS"
    )
    print("isInPlan(limit rate) ->", in_plan)

    apply_tx, apply_err = trust_chain_service.apply_action_on_chain(
        settings, plan_id=plan_id, action="tarpit scan", tier="Perimeter / IDS"
    )
    print("markApplied(tarpit scan) ->", apply_tx or apply_err)
    if not apply_tx:
        return 6

    repeat_tx, repeat_err = trust_chain_service.apply_action_on_chain(
        settings, plan_id=plan_id, action="tarpit scan", tier="Perimeter / IDS"
    )
    print("markApplied again ->", repeat_tx or repeat_err)
    if repeat_tx:
        print("ERROR: repeat apply was accepted")
        return 7

    print("OK: storePlan, getPlan, isInPlan, and markApplied behave as expected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
