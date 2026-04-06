"""LLM reasoning: OpenAI when configured, otherwise deterministic mock."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import Settings

logger = logging.getLogger(__name__)


def _mock_decision(prediction_summary: dict[str, Any], feature_notes: str | None, rag_context: str | None) -> dict[str, str]:
    flagged = prediction_summary.get("rows_flagged") or 0
    total = prediction_summary.get("rows_total") or 0
    ratio = flagged / total if total else 0.0
    if ratio > 0.2:
        action = "block_ip"
        summary = (
            f"Mock agent: high share of flagged rows ({flagged}/{total}). "
            "Recommend blocking suspicious sources and alerting the security team."
        )
    elif flagged > 0:
        action = "alert_admin"
        summary = (
            f"Mock agent: moderate signals ({flagged}/{total} flagged). "
            "Recommend alerting an administrator and collecting PCAPs for review."
        )
    else:
        action = "monitor"
        summary = "Mock agent: no strong attack signals in batch; continue routine monitoring."

    if feature_notes:
        summary += f" Notes incorporated: {feature_notes[:500]}"
    if rag_context:
        summary += " Policy context from knowledge base was supplied (mock does not quote it)."
    return {"summary": summary, "recommended_action": action, "raw": json.dumps({"mode": "mock"})}


def _parse_json_loose(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"summary": text[:2000], "recommended_action": "alert_admin", "raw": text}


async def agent_decide(
    settings: Settings,
    prediction_summary: dict[str, Any],
    feature_notes: str | None,
    rag_context: str | None,
) -> dict[str, str]:
    if not settings.openai_api_key:
        m = _mock_decision(prediction_summary, feature_notes, rag_context)
        return {
            "summary": m["summary"],
            "recommended_action": m["recommended_action"],
            "raw_llm_response": m["raw"],
            "rag_context_used": (rag_context[:12000] if rag_context else None),
        }

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai package missing; using mock LLM")
        m = _mock_decision(prediction_summary, feature_notes, rag_context)
        return {
            "summary": m["summary"],
            "recommended_action": m["recommended_action"],
            "raw_llm_response": m["raw"],
            "rag_context_used": rag_context[:12000] if rag_context else None,
        }

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    system = (
        "You are a security operations agent. Given batch prediction statistics and optional policy context, "
        "respond with a JSON object only, keys: summary (string), recommended_action (one of: "
        "block_ip, alert_admin, throttle, monitor, escalate_soc)."
    )
    user_payload = {
        "prediction_summary": prediction_summary,
        "feature_notes": feature_notes,
        "knowledge_excerpts": rag_context,
    }
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, default=str)[:24000]},
        ],
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    parsed = _parse_json_loose(content)
    action = str(parsed.get("recommended_action", "alert_admin"))
    summary = str(parsed.get("summary", content[:2000]))
    return {
        "summary": summary,
        "recommended_action": action[:512],
        "raw_llm_response": content[:16000],
        "rag_context_used": rag_context[:12000] if rag_context else None,
    }


async def rag_answer(
    settings: Settings,
    query: str,
    citations: list[dict[str, Any]],
) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c.get('text','')[:1500]}" for i, c in enumerate(citations))
    if not settings.openai_api_key:
        return (
            "Mock LLM answer (set OPENAI_API_KEY for full reasoning): "
            + query[:200]
            + " | Based on retrieved snippets, review internal runbooks and align with SOC policy."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": "Answer using the provided context. Cite snippet numbers when relevant.",
            },
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{ctx}"[:24000]},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""
