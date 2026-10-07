"""Per-tier execution agent: map a gated action to one domain agent and POST it.

D1 Access / ISP, D2 Perimeter / IDS, D3 Endpoint / EDR each have their own HTTP
endpoint (``EXEC_AGENT_*_URL``). If a URL is unset the call is recorded as a stub
with the same JSON body a live agent would receive.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

from app.core.config import Settings
from scripts.network_domains import ACCESS_ISP, ENDPOINT_EDR, PERIMETER_IDS, normalize_domain

logger = logging.getLogger(__name__)

# Canonical domain -> (agent id, settings field, typical live target)
_TIER_AGENT: dict[str, tuple[str, str, str]] = {
    ACCESS_ISP: ("D1", "exec_agent_access_url", "ISP edge / ACL controller"),
    PERIMETER_IDS: ("D2", "exec_agent_perimeter_url", "IDS / WAF / firewall API"),
    ENDPOINT_EDR: ("D3", "exec_agent_endpoint_url", "EDR / host isolation API"),
}


def domain_agent_for_tier(tier: str) -> str | None:
    domain = normalize_domain(tier)
    if not domain:
        return None
    info = _TIER_AGENT.get(domain)
    return info[0] if info else None


def exec_url_for_tier(settings: Settings, tier: str) -> str | None:
    domain = normalize_domain(tier)
    if not domain:
        return None
    info = _TIER_AGENT.get(domain)
    if not info:
        return None
    raw = getattr(settings, info[1], None)
    url = str(raw or "").strip()
    return url or None


def build_exec_payload(
    *,
    action: str,
    tier: str,
    attack_type: str | None,
    plan_id: str,
    report_id: str,
    job_id: str | None,
) -> dict[str, Any]:
    domain = normalize_domain(tier) or str(tier or "").strip()
    agent = domain_agent_for_tier(tier)
    info = _TIER_AGENT.get(domain) if domain in _TIER_AGENT else None
    return {
        "agent": agent,
        "network_tier": domain,
        "action": str(action).strip(),
        "attack_type": attack_type,
        "plan_id": plan_id,
        "report_id": report_id,
        "job_id": job_id,
        "target": info[2] if info else None,
    }


def dispatch_gated_action(
    settings: Settings,
    *,
    action: str,
    tier: str,
    attack_type: str | None,
    plan_id: str,
    report_id: str,
    job_id: str | None,
) -> dict[str, Any]:
    """POST the gated unit to the domain agent. Never raises; errors go on the result dict."""
    payload = build_exec_payload(
        action=action,
        tier=tier,
        attack_type=attack_type,
        plan_id=plan_id,
        report_id=report_id,
        job_id=job_id,
    )
    if payload.get("agent") is None:
        return {
            "mode": "failed",
            "http_status": None,
            "endpoint": None,
            "payload": payload,
            "error": "unknown_network_tier",
        }
    url = exec_url_for_tier(settings, tier)
    if not url:
        logger.info("exec stub %s %s @ %s", payload["agent"], action, payload["network_tier"])
        return {
            "mode": "stub",
            "http_status": None,
            "endpoint": None,
            "payload": payload,
            "note": "No EXEC_AGENT_*_URL for this tier; receipt only (same JSON a live agent would POST).",
        }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    timeout = float(getattr(settings, "exec_agent_timeout_s", 5) or 5)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
            raw = resp.read(4096).decode("utf-8", errors="replace")
        return {
            "mode": "http",
            "http_status": status,
            "endpoint": url,
            "payload": payload,
            "response_preview": raw[:500],
        }
    except urllib.error.HTTPError as e:
        return {
            "mode": "http",
            "http_status": int(e.code),
            "endpoint": url,
            "payload": payload,
            "error": str(e.reason)[:300],
        }
    except Exception as e:
        return {
            "mode": "http",
            "http_status": None,
            "endpoint": url,
            "payload": payload,
            "error": str(e)[:300],
        }
