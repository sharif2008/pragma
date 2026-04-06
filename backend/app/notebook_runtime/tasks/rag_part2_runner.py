"""Standalone RAG Part 2 (merged from RAG_part2_agent_actions.ipynb)."""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.notebook_runtime.env import load_project_dotenv
from openai import OpenAI

load_project_dotenv()

from app.notebook_runtime.rag_utils import (
    agentic_tiers_dict,
    balance_vector_hits_by_source_file,
    get_dominant_party_info,
    load_attack_and_agentic,
    load_parent_store,
    load_predictions,
    load_vector_store,
    save_action_plan,
    save_comparison_file,
    tier_allowed_actions,
)

VECTOR_STORE_DIR = Path("RAG_docs/vector_store")
PREDICTIONS_DIR = Path("RAG_docs/predictions")
RESULTS_DIR = Path("RAG_docs/action_plans")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

vector_store = load_vector_store(VECTOR_STORE_DIR)
predictions_data = load_predictions(PREDICTIONS_DIR, verbose=True)
attack_actions_data, agentic_features_data = load_attack_and_agentic(verbose=True)

n_attack = (
    len(attack_actions_data.get("attacks", {}))
    if isinstance(attack_actions_data, dict)
    else 0
)
_tiers = agentic_tiers_dict(agentic_features_data)
has_agentic = bool(_tiers and any(t in _tiers for t in ("RAN", "Edge", "Core")))
print("-" * 60)
print(
    f"Predictions: {len(predictions_data)} | "
    f"Vector store: {'loaded' if vector_store is not None else 'missing'}"
)
print(f"Attack types in attack_options: {n_attack}")
print(f"Agentic features config: {'yes' if has_agentic else 'no'}")
print("-" * 60)

# RAG retrieval + LLM query/prompts/pipeline (search lives here next to query construction)

_RAG_PARENTS = load_parent_store(VECTOR_STORE_DIR)


_PER_QUERY_RETRIEVE_K = 20

# Defaults for multi-stage ranking pipeline
_DEFAULT_FINAL_SECTIONS = 5
_DEFAULT_MMR_K = 60
_DEFAULT_RERANK_K = 20
# Max KB sections embedded in create_prompt (must match slice in create_prompt)
_LLM_RAG_SECTIONS_IN_PROMPT = 5


def _chunk_key(d: Dict[str, Any]) -> Tuple[Any, ...]:
    """Stable identity key for a child chunk."""
    pid = d.get("parent_id")
    cix = d.get("child_index")
    if pid is not None and cix is not None:
        return ("pc", str(pid), int(cix))
    return (
        "legacy",
        str(d.get("source_file") or "").strip(),
        str(d.get("title") or "").strip(),
        str(d.get("chunk_text") or "").strip()[:120],
    )


def _as_score_from_dist(dist: Any) -> float:
    d = float(dist) if dist is not None else 0.0
    return 1.0 / (1.0 + max(d, 0.0))


def retrieve_child_chunks_for_query(
    vector_store: Any,
    query: str,
    *,
    top_k: int = _PER_QUERY_RETRIEVE_K,
    oversample_factor: int = 6,
    balance_by_source_file: bool = True,
) -> List[Dict[str, Any]]:
    """Retrieve child chunks only (NO parent expansion)."""
    if not vector_store or not query:
        return []

    fetch_k = min(300, max(int(top_k), int(top_k) * int(oversample_factor)))
    vector_results = vector_store.similarity_search_with_score(query, k=fetch_k)

    if balance_by_source_file:
        vector_results = balance_vector_hits_by_source_file(vector_results, int(top_k))
    else:
        vector_results = vector_results[: int(top_k)]

    out: List[Dict[str, Any]] = []
    for doc, dist in vector_results:
        meta = doc.metadata or {}
        title = meta.get("retrieval_title") or meta.get("title", "Unknown")
        src = meta.get("source_file", "")
        pid = meta.get("parent_id")
        cix = meta.get("child_index")
        chunk_text = doc.page_content or meta.get("text", "") or ""

        out.append(
            {
                "title": title,
                "source_file": src,
                "parent_id": pid,
                "child_index": cix,
                "chunk_text": chunk_text,
                "vector_score": _as_score_from_dist(dist),
            }
        )

    return out


def merge_and_dedupe_child_chunks(
    results_by_query: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge child-chunk sets; dedupe by (parent_id, child_index) when available."""
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    for results in results_by_query:
        for d in results:
            k = _chunk_key(d)
            prev = best.get(k)
            if not prev or float(d.get("vector_score", 0.0) or 0.0) > float(prev.get("vector_score", 0.0) or 0.0):
                best[k] = d

    merged = list(best.values())
    merged.sort(key=lambda x: float(x.get("vector_score", 0.0) or 0.0), reverse=True)
    return merged


def _cosine(a: List[float], b: List[float]) -> float:
    import math

    if not a or not b or len(a) != len(b):
        return 0.0
    da = sum(x * x for x in a)
    db = sum(x * x for x in b)
    if da <= 0.0 or db <= 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    return float(dot) / float(math.sqrt(da) * math.sqrt(db))


def _embed_query(vector_store: Any, query: str) -> List[float]:
    # LangChain FAISS typically exposes `embedding_function`.
    ef = getattr(vector_store, "embedding_function", None)
    if ef is None:
        raise ValueError("vector_store.embedding_function missing; cannot run MMR")
    if hasattr(ef, "embed_query"):
        return list(ef.embed_query(query))
    # Fallback for older wrappers
    if callable(ef):
        return list(ef(query))
    raise ValueError("Unsupported embedding function")


def _embed_texts(vector_store: Any, texts: List[str]) -> List[List[float]]:
    ef = getattr(vector_store, "embedding_function", None)
    if ef is None:
        raise ValueError("vector_store.embedding_function missing; cannot run MMR")
    if hasattr(ef, "embed_documents"):
        return [list(v) for v in ef.embed_documents(texts)]
    # Slow fallback: embed one-by-one via embed_query
    if hasattr(ef, "embed_query"):
        return [list(ef.embed_query(t)) for t in texts]
    raise ValueError("Unsupported embedding function")


def mmr_select(
    vector_store: Any,
    query: str,
    candidates: List[Dict[str, Any]],
    *,
    k: int = _DEFAULT_MMR_K,
    lambda_mult: float = 0.5,
) -> List[Dict[str, Any]]:
    """MMR over candidate chunk embeddings."""
    if not candidates:
        return []

    k = max(1, min(int(k), len(candidates)))

    texts = [str(c.get("chunk_text") or "") for c in candidates]
    qv = _embed_query(vector_store, query)
    dvs = _embed_texts(vector_store, texts)

    # Precompute query similarities
    sim_q = [_cosine(qv, dv) for dv in dvs]

    selected: List[int] = []
    remaining = set(range(len(candidates)))

    # Start from best by query similarity
    first = max(remaining, key=lambda i: sim_q[i])
    selected.append(first)
    remaining.remove(first)

    while remaining and len(selected) < k:
        def mmr_score(i: int) -> float:
            # max similarity to already-selected docs
            max_sim = max(_cosine(dvs[i], dvs[j]) for j in selected) if selected else 0.0
            return float(lambda_mult) * float(sim_q[i]) - (1.0 - float(lambda_mult)) * float(max_sim)

        nxt = max(remaining, key=mmr_score)
        selected.append(nxt)
        remaining.remove(nxt)

    return [candidates[i] for i in selected]


_CROSSENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_CROSS_ENCODER = None
_RERANKER_CONFIRMED = False


def ensure_cross_encoder_loaded(
    *, crossencoder_model: str = _CROSSENCODER_MODEL_NAME
) -> Any:
    """Load CrossEncoder once; raises if sentence-transformers is missing."""
    global _CROSS_ENCODER, _RERANKER_CONFIRMED

    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as e:
            raise ImportError(
                "CrossEncoder reranker is REQUIRED. Install: pip install sentence-transformers"
            ) from e
        _CROSS_ENCODER = CrossEncoder(crossencoder_model)

    if not _RERANKER_CONFIRMED:
        print(f"Reranker loaded OK: CrossEncoder='{crossencoder_model}'")
        _RERANKER_CONFIRMED = True

    return _CROSS_ENCODER


def crossencoder_rerank(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[float]:
    """Cross-encoder relevance scores for query–passage pairs."""
    ce = ensure_cross_encoder_loaded()
    pairs = [(query, str(c.get("chunk_text") or "")) for c in candidates]
    scores = ce.predict(pairs)
    return [float(s) for s in scores]


def rerank_with_cross_encoder(
    query: str,
    candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach cross-encoder scores as rerank_score and sort descending."""
    ce_scores = crossencoder_rerank(query, candidates)
    out: List[Dict[str, Any]] = []
    for c, s_ce in zip(candidates, ce_scores):
        d = dict(c)
        d["crossencoder_score"] = float(s_ce)
        d["rerank_score"] = float(s_ce)
        out.append(d)

    out.sort(key=lambda x: float(x.get("rerank_score", 0.0) or 0.0), reverse=True)
    return out


def expand_parent_sections(
    ranked_children: List[Dict[str, Any]],
    *,
    top_sections: int = _DEFAULT_FINAL_SECTIONS,
    max_parent_chars: int = 12000,
) -> List[Dict[str, Any]]:
    """Expand top-ranked child chunks into parent contextual sections.

    Removes duplicates by keeping the most relevant section when the same
    (parent_id) or (source_file,title) repeats.
    """
    if not ranked_children:
        return []

    # Build candidate sections first (may include duplicates)
    candidates: List[Dict[str, Any]] = []

    for c in ranked_children:
        score = float(c.get("rerank_score", c.get("vector_score", 0.0) or 0.0))
        pid = c.get("parent_id")
        extra: Dict[str, Any] = {}
        if c.get("vector_score") is not None:
            extra["vector_similarity"] = float(c.get("vector_score", 0.0) or 0.0)
        if c.get("crossencoder_score") is not None:
            extra["crossencoder_score"] = float(c.get("crossencoder_score", 0.0) or 0.0)

        if pid is None:
            candidates.append(
                {
                    "title": c.get("title", "Unknown"),
                    "source_file": c.get("source_file", ""),
                    "text": str(c.get("chunk_text") or ""),
                    "score": score,
                    **extra,
                }
            )
            continue

        pid_s = str(pid)
        parent = _RAG_PARENTS.get(pid_s) or _RAG_PARENTS.get(pid) or {}
        ptext = str(parent.get("text") or "")
        if max_parent_chars and len(ptext) > int(max_parent_chars):
            ptext = ptext[: int(max_parent_chars)] + "\n\n[parent truncated]"

        title = parent.get("retrieval_title") or c.get("title") or "Unknown"
        candidates.append(
            {
                "title": title,
                "source_file": c.get("source_file", ""),
                "text": f"{title}\n\n{ptext}" if ptext else str(c.get("chunk_text") or ""),
                "score": score,
                "parent_id": pid_s,
                **extra,
            }
        )

    # Dedupe: prefer parent_id when present, otherwise (source_file,title)
    best: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for s in candidates:
        if s.get("parent_id"):
            k: Tuple[Any, ...] = ("parent", str(s.get("parent_id")))
        else:
            k = ("doc", str(s.get("source_file") or "").strip(), str(s.get("title") or "").strip())

        prev = best.get(k)
        if not prev or float(s.get("score", 0.0) or 0.0) > float(prev.get("score", 0.0) or 0.0):
            best[k] = s

    sections = list(best.values())
    sections.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    return sections[: int(top_sections)]


def retrieve_rag_context_multi(
    vector_store: Any,
    queries: List[str],
    *,
    top_k: int = _DEFAULT_FINAL_SECTIONS,
    oversample_factor: int = 6,
    balance_by_source_file: bool = True,
    mmr_k: int = _DEFAULT_MMR_K,
    rerank_k: int = _DEFAULT_RERANK_K,
    lambda_mult: float = 0.5,
    max_parent_chars: int = 12000,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """Full retrieval pipeline (functional):

    Merge + dedupe
      ↓
    MMR
      ↓
    CrossEncoder reranker
      ↓
    Top ranked child chunks
      ↓
    Parent document expansion
      ↓
    Top 3–5 full contextual sections

    Returns: (final_sections, per_query_child_chunks)
    """
    qs = [" ".join((q or "").split()) for q in (queries or [])]
    qs = [q for q in qs if q]
    if not qs:
        return ([], [])

    per_query: List[List[Dict[str, Any]]] = []
    for i, q in enumerate(qs):
        chunks = retrieve_child_chunks_for_query(
            vector_store,
            q,
            top_k=int(_PER_QUERY_RETRIEVE_K),
            oversample_factor=oversample_factor,
            balance_by_source_file=balance_by_source_file,
        )
        per_query.append(chunks)
        print(
            f"[RAG pipeline] Step 1 — retrieve query {i + 1}/{len(qs)}: "
            f"{len(chunks)} child chunks (per-query cap {_PER_QUERY_RETRIEVE_K})"
        )
    n_before_dedupe = sum(len(x) for x in per_query)
    print(
        f"[RAG pipeline] Step 1 sum: {n_before_dedupe} child-chunk rows before merge/dedupe"
    )

    merged = merge_and_dedupe_child_chunks(per_query)
    print(
        f"[RAG pipeline] Step 2 — merge + dedupe: {len(merged)} unique child chunks"
    )

    # Use the first query as the anchor for MMR/reranking.
    anchor_query = qs[0]

    mmr_pool = mmr_select(
        vector_store,
        anchor_query,
        merged,
        k=int(mmr_k),
        lambda_mult=float(lambda_mult),
    )
    print(
        f"[RAG pipeline] Step 3 — MMR: {len(mmr_pool)} child chunks "
        f"(k_mmr={mmr_k}, input {len(merged)})"
    )

    reranked = rerank_with_cross_encoder(anchor_query, mmr_pool)
    print(
        f"[RAG pipeline] Step 4 — CrossEncoder rerank: {len(reranked)} scored chunks"
    )

    top_children = reranked[: int(rerank_k)]
    print(
        f"[RAG pipeline] Step 5 — top by rerank: {len(top_children)} child chunks "
        f"(cap rerank_k={rerank_k})"
    )

    final_sections = expand_parent_sections(
        top_children,
        top_sections=int(top_k),
        max_parent_chars=int(max_parent_chars),
    )
    print(
        f"[RAG pipeline] Step 6 — parent expand + dedupe: {len(final_sections)} "
        f"contextual sections (cap top_sections={top_k})"
    )

    # Sort final sections by score
    final_sections.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    out = final_sections[: int(top_k)]
    print(
        f"[RAG pipeline] Step 7 — return to caller: {len(out)} sections (cap top_k={top_k}); "
        f"create_prompt embeds up to {_LLM_RAG_SECTIONS_IN_PROMPT} in the LLM message"
    )
    return (out, per_query)


def summarize_rag_docs(
    docs: List[Dict[str, Any]],
    *,
    max_docs: int = 5,
    snippet_chars: int = 220,
) -> List[Dict[str, Any]]:
    """Compact summary objects for printing/debug.

    Supports both shapes:
    - final sections: {title, text, score, source_file}
    - child chunks:   {title, chunk_text, vector_score, source_file}
    """
    out: List[Dict[str, Any]] = []
    for d in (docs or [])[: int(max_docs)]:
        txt = str(d.get("text") or d.get("chunk_text") or "")
        snippet = txt[: int(snippet_chars)]
        if len(txt) > int(snippet_chars):
            snippet += "..."

        score = d.get("score")
        if score is None:
            score = d.get("rerank_score")
        if score is None:
            score = d.get("vector_score")

        out.append(
            {
                "title": d.get("title", "Unknown"),
                "source_file": d.get("source_file", ""),
                "score": float(score or 0.0),
                "snippet": " ".join(snippet.split()),
            }
        )
    return out


def _top_features_for_tier(
    sample: Dict[str, Any],
    tier: str,
    *,
    top_n: int = 3,
    score_key: str = "pct_contribution",
) -> List[str]:
    """Deterministic top-N feature names for a tier (RAN/Edge/Core)."""
    shap_expl = sample.get("shap_explanation", {}) or {}
    feat_contribs = shap_expl.get("feature_contributions", {}) or {}
    feats = feat_contribs.get(tier, {}) or {}

    scored: List[Tuple[str, float]] = []
    if isinstance(feats, dict):
        for feat_name, meta in feats.items():
            if not isinstance(meta, dict):
                continue
            score = float(meta.get(score_key, 0.0) or 0.0)
            if score > 0.0:
                scored.append((feat_name, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored[:top_n]]


def extract_sample_summary(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize prediction fields used by query builders (single source of truth)."""
    label = (sample.get("predicted_label") or sample.get("true_label") or "UNKNOWN").strip().upper()
    confidence = float(sample.get("confidence", 0.0) or 0.0)

    shap_expl = sample.get("shap_explanation", {}) or {}
    dominant_agent = (shap_expl.get("dominant_agent") or "").strip()
    dominant_tier = dominant_agent if dominant_agent in ("RAN", "Edge", "Core") else "Unknown"
    dominant_pct = float(shap_expl.get("dominant_contribution_pct", 0.0) or 0.0) * 100.0

    top_features = {
        "RAN": _top_features_for_tier(sample, "RAN", top_n=3),
        "Edge": _top_features_for_tier(sample, "Edge", top_n=3),
        "Core": _top_features_for_tier(sample, "Core", top_n=3),
    }

    return {
        "label": label,
        "confidence": confidence,
        "dominant_tier": dominant_tier,
        "dominant_pct": dominant_pct,
        "top_features": top_features,
    }


_TEMPLATE_QUERY = (
    "Find mitigation actions, detection steps, and response playbooks for a {label} attack in a telecom network. "
    "Prioritize the {dominant_tier} tier (dominant contribution ~{dominant_pct:.0f}%). "
    "Use these top indicators as keywords: "
    "RAN: {ran_feats}. Edge: {edge_feats}. Core: {core_feats}. "
    "Focus on controls appropriate for confidence {confidence:.0%}."
)


def build_template_rag_query(sample: Dict[str, Any]) -> str:
    """Deterministic template query used for vector retrieval (NO LLM).

    Query options supported in this notebook:
    1) Template-based (deterministic)            -> `build_template_rag_query(sample)`
    2) "BERT-based" style rephrase (optional)     -> `rephrase_template_query_deterministic(template_query)`
       (Note: implemented as deterministic rules; no model downloads.)
    3) LLM-based query variants (optional)       -> `build_llm_rag_queries(sample)`
       (Returns two variants derived from the template + top features in prediction JSON.)

    For RAG retrieval, we intentionally use **only option (1)** for determinism.
    """
    s = extract_sample_summary(sample)

    def fmt(feats: List[str]) -> str:
        return ", ".join(feats) if feats else "no strong features"

    return _TEMPLATE_QUERY.format(
        label=s["label"],
        dominant_tier=s["dominant_tier"],
        dominant_pct=s["dominant_pct"],
        confidence=s["confidence"],
        ran_feats=fmt(s["top_features"]["RAN"]),
        edge_feats=fmt(s["top_features"]["Edge"]),
        core_feats=fmt(s["top_features"]["Core"]),
    )


def build_llm_rag_queries(sample: Dict[str, Any]) -> Tuple[str, str]:
    """Optional: return (concise_query, expanded_query) using an LLM.

    Not used for retrieval by default.
    """
    base = build_template_rag_query(sample)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (base, base)

    client = OpenAI(api_key=api_key)
    prompt = (
        "Rewrite the following retrieval query into two variants for vector search.\n\n"
        "Variant A: concise (1 sentence, keep feature tokens).\n"
        "Variant B: expanded (2-3 sentences, add synonyms, keep feature tokens).\n\n"
        "Return JSON only: {\"concise\": \"...\", \"expanded\": \"...\"}.\n\n"
        f"QUERY:\n{base}"
    )

    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=250,
    )

    txt = (resp.choices[0].message.content or "").strip()
    try:
        start = txt.find("{")
        end = txt.rfind("}") + 1
        obj = json.loads(txt[start:end]) if start >= 0 and end > start else {}
        concise = str(obj.get("concise") or base).strip()
        expanded = str(obj.get("expanded") or base).strip()
        return (concise, expanded)
    except Exception:
        return (base, base)


def rephrase_template_query_deterministic(template_query: str) -> str:
    """Deterministic query rephrase (no models / no downloads).

    Not used for retrieval by default.
    """
    q = (template_query or "").strip()
    if not q:
        return q

    # Simple, stable synonym swaps + light restructuring.
    swaps = [
        ("Find mitigation actions, detection steps, and response playbooks for a ", "Retrieve mitigation and detection guidance for "),
        ("telecom network", "telecommunications network"),
        ("Prioritize the ", "Focus on the "),
        ("Use these top indicators as keywords:", "Key indicators:"),
        ("Focus on controls appropriate for confidence", "Tailor controls to confidence"),
    ]

    for a, b in swaps:
        q = q.replace(a, b)

    # Ensure a compact single-line output for embedding/search.
    q = " ".join(q.split())
    return q


def create_prompt(
    sample_data: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    attack_actions_data: Optional[Dict[str, Any]] = None,
    agentic_features_data: Optional[Dict[str, Any]] = None,
    *,
    include_knowledge_base: bool = True,
) -> str:
    sample_id = sample_data.get("sample_id", 0)
    predicted_label = sample_data.get("predicted_label", "UNKNOWN")
    confidence = sample_data.get("confidence", 0.0)
    dominant_party, dominant_tier, dominant_pct = get_dominant_party_info(sample_data)

    # IMPORTANT: Do NOT pass the full prediction/SHAP payload to the LLM.
    # Only pass a compact evidence summary: top-3 features per tier + dominant tier.
    s = extract_sample_summary(sample_data)
    condensed_evidence = {
        "predicted_label": str(s.get("label") or predicted_label),
        "confidence": float(s.get("confidence", confidence) or 0.0),
        "dominant_tier": s.get("dominant_tier", dominant_tier),
        "dominant_contribution_pct": float(s.get("dominant_pct", dominant_pct) or 0.0),
        "top_features_by_tier": s.get("top_features", {}),
    }

    attack_actions_context = ""
    if attack_actions_data and "attacks" in attack_actions_data:
        attack_type = predicted_label.upper()
        if attack_type in attack_actions_data["attacks"]:
            recommended_actions = attack_actions_data["attacks"][attack_type]
            attack_actions_context = (
                f"\n\nAttack-Specific Recommended Actions (from attack_options.json):\n"
                f"For {predicted_label} attack, recommended actions: {', '.join(recommended_actions)}\n"
            )
        else:
            attack_actions_context = (
                f"\n\nAttack-Specific Actions: No specific recommendations for {predicted_label}.\n"
            )

    agentic_context = ""
    tiers = agentic_tiers_dict(agentic_features_data)
    if tiers:
        agentic_context = (
            "\n\nAgentic Features and Actions by Network Tier (from agentic_features.json):\n"
            "Use the condensed evidence (top features by tier) to GATE which tier gets priority actions.\n"
        )
        tf = condensed_evidence.get("top_features_by_tier", {}) or {}
        for tier in ["RAN", "Edge", "Core"]:
            if tier in tiers:
                tier_data = tiers[tier]
                if not isinstance(tier_data, dict):
                    continue
                actions = tier_allowed_actions(tier_data)
                tier_feats = tf.get(tier, []) if isinstance(tf, dict) else []
                agentic_context += f"\n{tier} Network Tier:\n"
                agentic_context += f"  - Top evidence features (top 3): {', '.join(tier_feats) if tier_feats else 'none'}\n"
                agentic_context += f"  - Allowed tier actions: {', '.join(actions)}\n"

    # Knowledge-base block omitted when include_knowledge_base is False (no-RAG ablation).
    kb_block = ""
    rag_hard_rule = ""
    if include_knowledge_base:
        top_5_results = rag_results[:_LLM_RAG_SECTIONS_IN_PROMPT]
        rag_context = ""
        if top_5_results:
            rag_context = "\n\nKnowledge Base Context (from RAG search):\n"
            for idx, result in enumerate(top_5_results, 1):
                rag_context += f"\n[{idx}] {result['title']}\n"
                rag_context += f"{result['text']}\n"
        else:
            rag_context = "\n\nKnowledge Base: No relevant documents found from RAG search."
        kb_block = (
            "\n        Retrieved knowledge (RAG – optional support, may be empty):\n"
            f"        {rag_context}\n"
        )
        rag_hard_rule = (
            "        - If RAG context is empty, rely ONLY on explainability and agentic context.\n"
        )

    network_tier_info = ""
    if dominant_tier:
        network_tier_info = f"\n- Dominant network tier: {dominant_tier} (contribution: {dominant_pct:.1f}%)"
        if dominant_party:
            network_tier_info += f"\n- Dominant party: {dominant_party}"

    return f"""You are a cybersecurity decision-making agent specialized in attack response orchestration.
        Your role is NOT to invent mitigations, but to SELECT and ASSIGN actions from a predefined
        action set using explainability signals and agentic features.

        =====================
        INPUT CONTEXT
        =====================

        Prediction summary:
        - sample_id: {sample_id}
        - predicted_label: {predicted_label}
        - confidence: {confidence:.1%}

        Explainability & agentic evidence:
        - Party-level contributions and dominance:
        {network_tier_info}

        - Condensed feature evidence (top 3 per tier):
        {json.dumps(condensed_evidence, indent=2)}

        Allowed actions (STRICT CONSTRAINT):
        {attack_actions_context}
        • Only actions listed above are allowed.
        • Do NOT invent, rename, generalize, or merge actions.

        Agentic decision signals:
        {agentic_context}
{kb_block}
        =====================
        TASK
        =====================

        Using ONLY the information above, generate an agent-ready action plan that:

        1) Interprets how {predicted_label} manifests across network tiers (RAN, Edge, Core).
        2) Identifies which evidence party MUST trigger mitigation first.
        3) Selects actions ONLY from the provided Allowed actions list.
        4) Assigns each action to the MOST appropriate executing party and network tier.
        5) Provides explicit, evidence-grounded reasoning for EACH action.
        6) Adapts aggressiveness and execution priority based on confidence ({confidence:.1%}).
        7) If no allowed action is suitable, return empty action lists.

        =====================
        OUTPUT FORMAT (STRICT)
        =====================

        Return a VALID JSON object ONLY:

        {{
          "threat_level": "Critical|High|Medium|Low",

          "all_actions": [
            "action_name_1",
            "action_name_2",
            "action_name_3"
          ],

          "primary_actions": [
            {{
              "action": "EXACT action name from Allowed actions to be taken ",
              "network_tier": "RAN|Edge|Core",
              "party_evidence_type": "type of evidence this party observed",
              "reasoning": "clear explanation linking evidence + agentic signals to this action"
            }}
          ],

          "supporting_actions": [
            {{
              "action": "EXACT action name from Allowed actions to be taken",
              "network_tier": "RAN|Edge|Core",
              "party_evidence_type": "type of evidence this party observed",
              "reasoning": "why this action supports or complements the primary action"
            }}
          ],
          "overall_reasoning": "Concise summary explaining party prioritization, tier ordering, and action selection logic",
          "execution_priority": "Immediate|High|Standard|Low",
          "knowledge_sources_used": [
            "allowed_actions_context",
            "attack_actions_context"
          ]
        }}

        =====================
        HARD RULES (DO NOT VIOLATE)
        =====================

        - Do NOT output text outside the JSON.
        - Do NOT generate actions not listed in Allowed actions.
        - The "all_actions" list MUST be the union of primary_actions and supporting_actions.
        - Do NOT alter action or party names.
        - Every action MUST include explicit reasoning tied to evidence or agentic rules.
        - Prefer dominant party and tier for primary actions unless contradicted by evidence.
{rag_hard_rule}        """


def create_prompt_without_RAG(
    sample_data: Dict[str, Any],
    attack_actions_data: Optional[Dict[str, Any]] = None,
    agentic_features_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Same as `create_prompt` but no KB/RAG section (paired with-RAG vs no-RAG saves)."""
    return create_prompt(
        sample_data,
        [],
        attack_actions_data=attack_actions_data,
        agentic_features_data=agentic_features_data,
        include_knowledge_base=False,
    )


def call_llm_api(prompt: str) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        return None

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "You are a cybersecurity expert. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    response_text = response.choices[0].message.content.strip()
    try:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response_text[start:end])
    except Exception as e:
        print(f"Warning: Could not parse JSON: {e}")

    return {
        "threat_level": "Unknown",
        "primary_actions": ["Unable to parse response"],
        "supporting_actions": [],
        "reasoning": response_text[:200],
        "execution_priority": "Standard",
        "knowledge_sources_used": [],
    }


def _normalize_strategy_to_keys(strategy: Any) -> List[str]:
    """Normalize a strategy value into a list of strategy keys."""
    if isinstance(strategy, list):
        return [str(x).strip().lower() for x in strategy if str(x).strip()]
    s = str(strategy or "").strip().lower()
    return [s] if s else ["template"]


def _vprint(msg: str = "") -> None:
    print(msg)


def _print_doc_summary(title: str, docs: List[Dict[str, Any]]) -> None:
    _vprint(title)
    if not docs:
        _vprint("  (no documents retrieved)")
        return
    for j, d in enumerate(summarize_rag_docs(docs, max_docs=5), 1):
        _vprint(
            f"  [{j}] {d['title']} | {d['source_file']} | sim={d['score']:.2%}\n"
            f"      {d['snippet']}"
        )


def run_agent_pipeline_for_sample(
    sample: Dict[str, Any],
    vector_store: Any,
    attack_actions_data: Optional[Dict[str, Any]],
    agentic_features_data: Optional[Dict[str, Any]],
    results_action_dir: Path,
    *,
    top_k_retrieve: int = 10,
    query_strategy: Any = "template",
) -> None:
    """Run the end-to-end action plan pipeline for one sample.

    query_strategy can be:
      - a STRING: "template" | "rephrase" | "llm_concise" | "llm_expanded"
      - a LIST of strings: ["template", "rephrase", ...] -> multi-query + merged retrieval

    If list length is 1, it behaves like single-query but uses the same unified path.
    """

    _vprint("=" * 80)
    _vprint(
        f"Sample {sample.get('sample_id')} | "
        f"label={sample.get('predicted_label', 'UNKNOWN')} | "
        f"conf={sample.get('confidence', 0):.1%}"
    )

    # -----------------------------
    # Generate query variants
    # -----------------------------
    q_template = build_template_rag_query(sample)
    q_rephrased = rephrase_template_query_deterministic(q_template)
    q_llm_concise, q_llm_expanded = build_llm_rag_queries(sample)

    generated_queries: Dict[str, str] = {
        "template": q_template,
        "rephrase": q_rephrased,
        "llm_concise": q_llm_concise,
        "llm_expanded": q_llm_expanded,
    }

    # -----------------------------
    # Resolve query_strategy -> retrieval_queries (1+)
    # -----------------------------
    strategy_keys = _normalize_strategy_to_keys(query_strategy)
    if strategy_keys == ["multi"]:
        strategy_keys = ["template", "rephrase"]

    strategy_keys = [k for k in strategy_keys if k in generated_queries]
    if not strategy_keys:
        strategy_keys = ["template"]

    retrieval_queries = [generated_queries[k] for k in strategy_keys]
    retrieval_queries = [q for q in retrieval_queries if q] or [generated_queries["template"]]

    _vprint(f"Query strategy: {strategy_keys} | queries={len(retrieval_queries)}")

    _vprint("\nGenerated queries:")
    for k in ("template", "rephrase", "llm_concise", "llm_expanded"):
        _vprint(f"\n[{k}]\n{generated_queries.get(k, '')}")

    # -----------------------------
    # Retrieve (20 per query) + merge
    # -----------------------------
    rag_results, per_query_results = retrieve_rag_context_multi(
        vector_store,
        retrieval_queries,
        top_k=top_k_retrieve,
    )

    _vprint("\nRAG retrieval results:")
    for i, q in enumerate(retrieval_queries):
        _vprint(f"\nQuery {i+1}/{len(retrieval_queries)}:\n{q}")
        _print_doc_summary(
            f"Retrieved {len(per_query_results[i])} docs (showing up to 5):",
            per_query_results[i],
        )

    _vprint("")
    _print_doc_summary(
        f"Merged docs used in prompt: {len(rag_results)} total (showing up to 5):",
        rag_results,
    )
    _vprint("")

    # Persist which query(ies) were used for retrieval.
    if len(retrieval_queries) == 1:
        query_for_save = retrieval_queries[0]
    else:
        parts = [f"[{i+1}] {q}" for i, q in enumerate(retrieval_queries)]
        query_for_save = "MULTI_QUERY\n\n" + "\n\n---\n\n".join(parts)

    pair_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    _llm_rag_sections = min(_LLM_RAG_SECTIONS_IN_PROMPT, len(rag_results))
    print(
        f"[RAG→LLM] Sections returned by pipeline: {len(rag_results)}; "
        f"injected into with-RAG prompt: {_llm_rag_sections} (create_prompt uses first {_LLM_RAG_SECTIONS_IN_PROMPT})"
    )
    print("[RAG→LLM] No-RAG prompt: 0 KB sections (same structured actions/agentic context)")

    prompt_with = create_prompt(
        sample, rag_results, attack_actions_data, agentic_features_data
    )
    prompt_without = create_prompt_without_RAG(
        sample, attack_actions_data, agentic_features_data
    )

    print("Calling LLM API (with RAG context)...")
    llm_with_rag = call_llm_api(prompt_with)
    print("Calling LLM API (without RAG context)...")
    llm_no_rag = call_llm_api(prompt_without)

    if not llm_with_rag and not llm_no_rag:
        print("Failed to get LLM responses")
        print("=" * 80)
        return

    def _print_resp(label: str, resp: Optional[Dict[str, Any]]) -> None:
        print(f"\n--- LLM response ({label}) ---")
        if not resp:
            print("  (no response)")
            return
        print(f"  Threat level: {resp.get('threat_level')}")
        print(f"  Execution priority: {resp.get('execution_priority')}")
        print(f"  Primary actions: {resp.get('primary_actions', [])}")
        reasoning = (resp.get("overall_reasoning") or "")[:500]
        if len(resp.get("overall_reasoning") or "") > 500:
            reasoning += "..."
        print(f"  Overall reasoning: {reasoning}")

    _print_resp("with RAG", llm_with_rag)
    _print_resp("without RAG", llm_no_rag)

    path_with: Optional[Path] = None
    path_without: Optional[Path] = None
    try:
        if llm_with_rag:
            path_with = save_action_plan(
                sample,
                query_for_save,
                rag_results,
                llm_with_rag,
                results_action_dir,
                variant="with_rag",
                timestamp=pair_ts,
                prompt_uses_rag=True,
            )
            print(f"Saved action plan (with RAG) to {path_with.name}")
        if llm_no_rag:
            path_without = save_action_plan(
                sample,
                query_for_save,
                rag_results,
                llm_no_rag,
                results_action_dir,
                variant="no_rag",
                timestamp=pair_ts,
                prompt_uses_rag=False,
            )
            print(f"Saved action plan (no RAG) to {path_without.name}")
    except Exception as e:
        print(f"Error saving action plan(s): {e}")

    try:
        if llm_with_rag or llm_no_rag:
            cmp_path = save_comparison_file(
                sample,
                query_for_save,
                rag_results,
                llm_with_rag,
                llm_no_rag,
                results_action_dir,
                path_with,
                path_without,
            )
            print(f"Saved comparison to {cmp_path.name}")
    except Exception as e:
        print(f"Error saving comparison file: {e}")

    print("=" * 80)


# Only run action planning when the model predicts an attack (skip predicted BENIGN).
def _is_predicted_attack(sample: Dict[str, Any]) -> bool:
    pl = str(sample.get("predicted_label") or "").strip().upper()
    return bool(pl) and pl != "BENIGN"


# -----------------------------
# Retrieval configuration
# -----------------------------
# You can pass either:
# - a STRING: "template" | "rephrase" | "llm_concise" | "llm_expanded"
# - a LIST of strings: ["template", "rephrase", ...]  -> multi-query + merged retrieval
QUERY_STRATEGY = "template"  # or: ["template", "rephrase"]

# Per-query retrieval is fixed to 20 docs/query (see `_PER_QUERY_RETRIEVE_K`).


samples_for_actions = [s for s in predictions_data if _is_predicted_attack(s)]
n_skip = len(predictions_data) - len(samples_for_actions)
print(
    f"Action pipeline: {len(samples_for_actions)} sample(s) with predicted attack; "
    f"skipping {n_skip} predicted benign."
)

for sample in samples_for_actions:
    # Top 5 contextual sections passed to the LLM by default.
    run_agent_pipeline_for_sample(
        sample,
        vector_store,
        attack_actions_data,
        agentic_features_data,
        RESULTS_DIR,
        top_k_retrieve=5,
        query_strategy=QUERY_STRATEGY,
    )

print("Done. Action plans in:", RESULTS_DIR)