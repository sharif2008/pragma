"""Smoke test for local Hardhat trust anchoring config.

Usage (from backend/):
  python scripts/test_trust_chain.py
  python scripts/test_trust_chain.py --anchor-test
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from web3 import Web3

# Allow running from repo root or backend/ by ensuring backend/ is on sys.path.
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import get_settings  # noqa: E402


REGISTRY_ABI = [
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


def _b32() -> str:
    return "0x" + secrets.token_hex(32)


def _bytes32_hex(v: bytes | str) -> str:
    """Normalize web3 bytes32 (bytes or 0x hex str) to lowercase 0x-prefixed hex."""
    if isinstance(v, str):
        return v.lower()
    return Web3.to_hex(v).lower()


def _strip_opt(s: str | None) -> str | None:
    if s is None:
        return None
    t = s.strip()
    return t or None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--anchor-test",
        action="store_true",
        help="Send a test anchor() transaction and read back getCommitment().",
    )
    args = ap.parse_args()

    settings = get_settings()
    contract = _strip_opt(settings.trust_chain_contract_address)
    private_key = _strip_opt(settings.trust_chain_private_key)

    print("TRUST_CHAIN_ENABLED:", settings.trust_chain_enabled)
    print("TRUST_CHAIN_RPC_URL:", settings.trust_chain_rpc_url)
    print("TRUST_CHAIN_CHAIN_ID:", settings.trust_chain_chain_id)
    print("TRUST_CHAIN_CONTRACT_ADDRESS:", contract or "(not set)")
    print("TRUST_CHAIN_PRIVATE_KEY:", private_key or "(not set)")
    if private_key:
        print(
            "  (local Hardhat only — do not use real keys; avoid sharing this output)",
            file=sys.stderr,
        )

    if not settings.trust_chain_enabled:
        print("ERROR: TRUST_CHAIN_ENABLED is false.")
        return 2
    if not contract:
        print("ERROR: TRUST_CHAIN_CONTRACT_ADDRESS is missing.")
        return 2
    if args.anchor_test and not private_key:
        print("ERROR: TRUST_CHAIN_PRIVATE_KEY is missing (required for --anchor-test).")
        return 2

    w3 = Web3(Web3.HTTPProvider(settings.trust_chain_rpc_url))
    if not w3.is_connected():
        print("ERROR: Could not connect to RPC.")
        return 3

    chain_id = int(w3.eth.chain_id)
    print("RPC chain_id:", chain_id)
    if chain_id != int(settings.trust_chain_chain_id):
        print("WARN: RPC chain_id != TRUST_CHAIN_CHAIN_ID")

    addr = Web3.to_checksum_address(contract)
    code = w3.eth.get_code(addr)
    print("Contract code bytes:", len(code))
    if not code or code == b"\x00":
        print("ERROR: No contract code at TRUST_CHAIN_CONTRACT_ADDRESS (did you deploy?)")
        return 4

    c = w3.eth.contract(address=addr, abi=REGISTRY_ABI)

    # Always do a read-only call first.
    agent_key = _b32()
    report_key = _b32()
    got = c.functions.getCommitment(agent_key, report_key).call()
    print("Read test getCommitment() ->", got)

    signer = None
    if private_key:
        signer = w3.eth.account.from_key(private_key)
        bal_wei = w3.eth.get_balance(signer.address)
        print("Signer address:", signer.address)
        print("Signer balance:", w3.from_wei(bal_wei, "ether"), "ETH (", bal_wei, "wei)")

    if not args.anchor_test:
        print("OK: Connected + contract readable. (Use --anchor-test to send a tx.)")
        return 0

    assert signer is not None
    acct = signer

    commitment = _b32()
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = c.functions.anchor(agent_key, report_key, commitment).build_transaction(
        {
            "from": acct.address,
            "nonce": nonce,
            "chainId": int(settings.trust_chain_chain_id),
            "gas": 200_000,
            "maxFeePerGas": w3.to_wei(2, "gwei"),
            "maxPriorityFeePerGas": w3.to_wei(1, "gwei"),
        }
    )
    signed = acct.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction).hex()
    print("anchor tx_hash:", tx_hash)

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    print("tx status:", receipt.status, "block:", receipt.blockNumber)

    got2 = c.functions.getCommitment(agent_key, report_key).call()
    print("After anchor getCommitment() ->", got2)
    if _bytes32_hex(got2) != commitment.lower():
        print("ERROR: commitment mismatch")
        return 5

    print("OK: anchor() wrote commitment successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

