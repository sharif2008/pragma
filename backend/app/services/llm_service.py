"""LLM reasoning: OpenAI when configured, otherwise deterministic mock."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.config import Settings
from app.services.agentic_llm_prompt import (
    build_agentic_decide_user_prompt,
    summarize_plan_for_db,
)

logger = logging.getLogger(__name__)


def _log_final_llm_messages(call: str, messages: list[dict[str, Any]]) -> None:
    """Echo the exact chat.completions messages payload (server console + logs)."""
    lines = [f"========== LLM API final messages: {call} =========="]
    for m in messages:
        role = str(m.get("role", ""))
        content = str(m.get("content", ""))
        lines.append(f"--- {role} ({len(content)} chars) ---")
        lines.append(content)
    lines.append(f"========== end {call} ==========")
    block = "\n".join(lines)
    print(block, flush=True)
    logger.info("%s", block)


def _mock_decision_orchestration(
    sample_data: dict[str, Any],
    feature_notes: str | None,
    rag_context: str | None,
    *,
    use_rag: bool,
) -> dict[str, str | None]:
    label = str(sample_data.get("predicted_label") or "UNKNOWN")
    conf = float(sample_data.get("confidence") or 0.0)
    plan = {
        "mode": "mock",
        "threat_level": "Medium" if conf > 0.7 else "Low",
        "all_actions": ["monitor"],
        "primary_actions": [
            {
                "action": "monitor",
                "network_tier": "Core",
                "party_evidence_type": "mock",
                "reasoning": "OpenAI API key not set; deterministic stub.",
            }
        ],
        "supporting_actions": [],
        "overall_reasoning": (
            f"Mock orchestration for label={label}, confidence={conf:.1%}. "
            f"Batch/RAG notes length={len(feature_notes or '')}, rag_chars={len(rag_context or '') if use_rag else 0}."
        ),
        "execution_priority": "Standard",
        "knowledge_sources_used": ["mock"],
    }
    raw = json.dumps(plan, indent=2, ensure_ascii=False)
    summary, rec = summarize_plan_for_db(plan, raw)
    return {
        "summary": summary,
        "recommended_action": rec,
        "raw_llm_response": raw,
        "rag_context_used": (rag_context[:12000] if use_rag and rag_context else None),
    }


def _parse_orchestration_json(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        out = json.loads(m.group())
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


async def agent_decide(
    settings: Settings,
    sample_data: dict[str, Any],
    feature_notes: str | None,
    rag_context: str | None,
    attack_actions_data: dict[str, Any] | None,
    agentic_features_data: dict[str, Any] | None,
    *,
    use_rag: bool = True,
) -> dict[str, str | None]:
    if not settings.openai_api_key:
        return _mock_decision_orchestration(sample_data, feature_notes, rag_context, use_rag=use_rag)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai package missing; using mock LLM")
        return _mock_decision_orchestration(sample_data, feature_notes, rag_context, use_rag=use_rag)

    prompt = build_agentic_decide_user_prompt(
        sample_data,
        rag_context,
        attack_actions_data,
        agentic_features_data,
        include_knowledge_base=use_rag,
        feature_notes=feature_notes,
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    system = "You are a cybersecurity expert. Return only valid JSON matching the user schema. No markdown or prose outside JSON."
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt[:100000]},
    ]
    _log_final_llm_messages(
        f"agent_decide model={settings.openai_model} (user truncated to 100000 chars if longer)",
        messages,
    )
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content or ""
    parsed = _parse_orchestration_json(content)
    if not parsed:
        parsed = {
            "overall_reasoning": content[:4000] if content else "Empty model response",
            "execution_priority": "Standard",
            "all_actions": [],
            "primary_actions": [],
            "supporting_actions": [],
        }
    summary, rec = summarize_plan_for_db(parsed, content)
    rag_used = rag_context[:12000] if use_rag and rag_context else None
    return {
        "summary": summary,
        "recommended_action": rec,
        "raw_llm_response": content[:65000] if len(content) > 65000 else content,
        "rag_context_used": rag_used,
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
    user_content = f"Question: {query}\n\nContext:\n{ctx}"[:24000]
    messages = [
        {
            "role": "system",
            "content": "Answer using the provided context. Cite snippet numbers when relevant.",
        },
        {"role": "user", "content": user_content},
    ]
    _log_final_llm_messages(f"rag_answer model={settings.openai_model}", messages)
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.3,
    )
    return resp.choices[0].message.content or ""


async def refine_shap_rag_retrieval_query(
    settings: Settings,
    *,
    draft_queries_text: str,
    analyst_synthesis_prompt: str | None,
) -> tuple[str, bool]:
    """
    Turn a human-readable SHAP/template draft into one dense string for vector search.

    Returns (retrieval_query, used_llm). Without an API key, returns the trimmed draft and used_llm=False.
    """
    draft = (draft_queries_text or "").strip()
    if not draft:
        return "", False

    if not settings.openai_api_key:
        return (" ".join(draft.split())[:4000], False)

    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai package missing; skipping SHAP retrieval LLM refine")
        return (" ".join(draft.split())[:4000], False)

    task = (analyst_synthesis_prompt or "").strip() or (
        "Using retrieved policy and runbook excerpts, explain how the top SHAP drivers per agent party "
        "should guide analyst triage and what to verify next."
    )
    user = (
        "You help SOC analysts retrieve knowledge base chunks via embedding search.\n\n"
        "HUMAN-READABLE DRAFT (built from templates + row/SHAP data — factual grounding only; "
        "do not invent features or labels):\n---\n"
        f"{draft[:8000]}\n"
        "---\n\n"
        "Downstream, the analyst will ask the synthesis LLM to:\n---\n"
        f"{task[:2000]}\n"
        "---\n\n"
        "Write ONE dense retrieval query (2–4 sentences) optimized for vector search. "
        "Keep exact feature names, SHAP tokens, and predicted_label strings from the draft. "
        "Return JSON only: {\"retrieval_query\": \"...\"}."
    )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    messages = [
        {"role": "system", "content": "Return only valid JSON with key retrieval_query. No markdown."},
        {"role": "user", "content": user[:12000]},
    ]
    _log_final_llm_messages(f"refine_shap_rag_retrieval_query model={settings.openai_model}", messages)
    resp = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.0,
        max_tokens=400,
    )
    content = resp.choices[0].message.content or ""
    parsed = _parse_orchestration_json(content)
    q = parsed.get("retrieval_query") if isinstance(parsed.get("retrieval_query"), str) else None
    out = (q or draft).strip()
    out = " ".join(out.split())
    if not out:
        out = " ".join(draft.split())[:4000]
    return (out[:4000], True)
