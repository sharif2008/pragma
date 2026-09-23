"""Software Apply gates shared by ``agent_service`` and the offline demo.

Gate order: whitelist, then plan-binding, then digest/integrity.
On-chain ``applyAction`` is the caller's job after ``result == "success"``.
"""

from __future__ import annotations

from dataclasses import dataclass

REASON_MISSING_ATTACK = "missing_attack_type"
REASON_NOT_WHITELISTED = "action_not_whitelisted"
REASON_WHITELIST_UNAVAILABLE = "whitelist_unavailable"
REASON_PLAN_MISMATCH = "action_plan_mismatch"
REASON_INTEGRITY = "integrity_validation_error"


@dataclass(frozen=True)
class ApplyGateDecision:
    result: str  # success | failed | skipped
    failure_reason: str | None
    whitelisted: bool | None


def action_in_whitelist(attack_type: str, action: str, whitelist: dict[str, list[str] | tuple[str, ...]]) -> bool:
    """Local Gate 1: action label is in W[attack] (case-insensitive)."""
    allow = {str(a).strip().lower() for a in whitelist.get(str(attack_type).upper(), ()) if str(a).strip()}
    return str(action).strip().lower() in allow


def decide_apply_gates(
    *,
    attack_type: str | None,
    action: str,
    planned_action: str | None,
    whitelist_allowed: bool | None,
    integrity_valid: bool,
) -> ApplyGateDecision:
    """Evaluate whitelist then plan-binding then integrity. Does not talk to the chain."""
    if not attack_type:
        return ApplyGateDecision("failed", REASON_MISSING_ATTACK, None)
    if whitelist_allowed is not True:
        if whitelist_allowed is False:
            return ApplyGateDecision("failed", REASON_NOT_WHITELISTED, False)
        return ApplyGateDecision("skipped", REASON_WHITELIST_UNAVAILABLE, whitelist_allowed)
    if planned_action is not None and str(planned_action) != str(action):
        return ApplyGateDecision("failed", REASON_PLAN_MISMATCH, True)
    if not integrity_valid:
        return ApplyGateDecision("failed", REASON_INTEGRITY, True)
    return ApplyGateDecision("success", None, True)
