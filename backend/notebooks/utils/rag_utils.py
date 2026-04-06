"""RAG pipeline helpers: predictions/config loaders, FAISS save/load, SHAP party/tier + JSON export.

Used by RAG_part1_build_vector_store.ipynb and RAG_part2_agent_actions.ipynb.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# LangChain still pulls pydantic.v1; that emits a noisy UserWarning on Python 3.14+.
if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

import numpy as np
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# ---------------------------------------------------------------------------
# Predictions + storage JSON configs (backend/storage)
# ---------------------------------------------------------------------------

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
_STORAGE_DIR = _BACKEND_ROOT / "storage"
ATTACK_OPTIONS_JSON = _STORAGE_DIR / "attack_options.json"
AGENTIC_FEATURES_JSON = _STORAGE_DIR / "agentic_features.json"


def agentic_tiers_dict(agentic_features_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Return the per-tier map (RAN / Edge / Core) from loaded agentic_features JSON.

    Supports schema with top-level ``agents`` or a legacy flat dict of tiers.
    """
    if not isinstance(agentic_features_data, dict):
        return None
    agents = agentic_features_data.get("agents")
    if isinstance(agents, dict) and agents:
        return agents
    legacy = {k: agentic_features_data[k] for k in ("RAN", "Edge", "Core") if k in agentic_features_data}
    return legacy or None


def tier_allowed_actions(tier_data: Dict[str, Any]) -> List[str]:
    """Action strings for prompt / UI (``action_capabilities`` in current schema)."""
    raw = tier_data.get("action_capabilities") or tier_data.get("actions") or []
    return list(raw) if isinstance(raw, list) else []


def load_predictions(predictions_dir: Path, verbose: bool = True) -> List[Dict[str, Any]]:
    """Load all *.json prediction files; each sample gets _source_file."""
    prediction_files = list(predictions_dir.glob("*.json"))
    if not prediction_files:
        raise FileNotFoundError(f"No JSON files found in {predictions_dir}")
    if verbose:
        print(f"Found {len(prediction_files)} prediction file(s) in {predictions_dir}")

    predictions_data: List[Dict[str, Any]] = []
    for prediction_file in prediction_files:
        if verbose:
            print(f"  Loading: {prediction_file.name}")
        with open(prediction_file, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        if isinstance(file_data, dict):
            file_data = [file_data]
        elif not isinstance(file_data, list):
            if verbose:
                print(f"    Skipping unexpected type in {prediction_file.name}")
            continue
        for sample in file_data:
            if isinstance(sample, dict):
                sample["_source_file"] = prediction_file.name
        predictions_data.extend(file_data)
        if verbose:
            print(f"    Loaded {len(file_data)} prediction(s) from {prediction_file.name}")
    if verbose:
        print(f"Total: {len(predictions_data)} prediction(s)")
    return predictions_data


def load_attack_and_agentic(
    attack_path: Path | None = None,
    agentic_path: Path | None = None,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load attack_options.json and agentic_features.json from ``backend/storage`` by default."""
    attack_path = attack_path if attack_path is not None else ATTACK_OPTIONS_JSON
    agentic_path = agentic_path if agentic_path is not None else AGENTIC_FEATURES_JSON
    attack_actions_data: Optional[Dict[str, Any]] = None
    if attack_path.exists():
        with open(attack_path, "r", encoding="utf-8") as f:
            attack_actions_data = json.load(f)
        if verbose:
            print(f"Loaded attack actions from {attack_path.name}")
    elif verbose:
        print(f"Attack actions file not found: {attack_path}")

    agentic_features_data: Optional[Dict[str, Any]] = None
    if agentic_path.exists():
        with open(agentic_path, "r", encoding="utf-8") as f:
            agentic_features_data = json.load(f)
        if verbose:
            print(f"Loaded agentic features from {agentic_path.name}")
    elif verbose:
        print(f"Agentic features file not found: {agentic_path}")

    return attack_actions_data, agentic_features_data


# ---------------------------------------------------------------------------
# Vector store (FAISS persist / load; build stays in Part 1 notebook)
# ---------------------------------------------------------------------------

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
# Defaults align with RAG_part1_build_vector_store.ipynb (smaller chunks for retrieval precision)
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
MANIFEST_NAME = "rag_manifest.json"


def _faiss_build_batch_size() -> int:
    raw = os.getenv("RAG_FAISS_BUILD_BATCH")
    if raw is None or not str(raw).strip():
        return 64
    try:
        return max(1, int(str(raw).strip()))
    except ValueError:
        return 64


def faiss_from_documents_batched(
    documents: List[Document],
    embedding: Embeddings,
    batch_size: Optional[int] = None,
) -> FAISS:
    """
    Build a FAISS store without embedding all ``page_content`` strings in one call
    (``FAISS.from_documents`` does that and can OOM on low-RAM machines).

    ``batch_size`` defaults to env ``RAG_FAISS_BUILD_BATCH`` or 64.
    """
    if not documents:
        raise ValueError("documents must be non-empty")
    bs = batch_size if batch_size is not None else _faiss_build_batch_size()
    first = documents[:bs]
    store = FAISS.from_documents(first, embedding)
    for i in range(bs, len(documents), bs):
        chunk = documents[i : i + bs]
        store.add_documents(chunk)
    return store


def save_vector_store(
    vector_store: FAISS,
    vector_store_dir: Path,
    knowledge_base: List[Dict[str, Any]],
    embed_model: str = DEFAULT_EMBED_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    n_chunks: Optional[int] = None,
    extra_manifest: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist FAISS index and a small JSON manifest next to it."""
    vector_store_dir = Path(vector_store_dir)
    vector_store_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(vector_store_dir))
    manifest: Dict[str, Any] = {
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "n_source_docs": len(knowledge_base),
        "n_chunks": n_chunks,
    }
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path = vector_store_dir / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def read_manifest(vector_store_dir: Path) -> Dict[str, Any]:
    p = Path(vector_store_dir) / MANIFEST_NAME
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _env_truthy(name: str) -> bool:
    v = os.environ.get(name)
    return v is not None and str(v).strip().lower() in ("1", "true", "yes")


def _normalize_sentence_transformers_model_id(model: str) -> str:
    """Use HF repo ids for short names (manifest uses ``all-MiniLM-L6-v2``); keep local dirs as-is."""
    m = (model or "").strip() or DEFAULT_EMBED_MODEL
    if os.path.isdir(m):
        return m
    p = Path(m)
    try:
        if p.is_dir():
            return os.fspath(p.resolve())
    except OSError:
        pass
    if "/" not in m:
        return f"sentence-transformers/{m}"
    return m


def _sentence_transformer_embeddings_for_faiss_load(model: str) -> SentenceTransformerEmbeddings:
    """
    Load the same embedding model used when the FAISS index was built.

    Tries ``local_files_only=True`` first so an already-cached model does not hit the Hub
    (avoids DNS/offline failures and flaky httpx retry shutdown). If that fails and Hub
    offline env vars are set, raises a clear error. Otherwise retries with downloads enabled.

    Env:
    - ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``: only use local cache (no fallback download).
    - ``RAG_EMBED_ONLINE_FIRST``: skip local-only attempt (always allow Hub).
    """
    model_id = _normalize_sentence_transformers_model_id(model)
    hub_offline = _env_truthy("HF_HUB_OFFLINE") or _env_truthy("TRANSFORMERS_OFFLINE")
    online_first = _env_truthy("RAG_EMBED_ONLINE_FIRST")

    def _with_local_only() -> SentenceTransformerEmbeddings:
        return SentenceTransformerEmbeddings(
            model_name=model_id,
            model_kwargs={"local_files_only": True},
        )

    def _with_hub() -> SentenceTransformerEmbeddings:
        return SentenceTransformerEmbeddings(model_name=model_id)

    if online_first:
        return _with_hub()
    if hub_offline:
        return _with_local_only()
    try:
        return _with_local_only()
    except Exception as e_local:
        try:
            return _with_hub()
        except Exception as e_hub:
            raise RuntimeError(
                f"Failed to load embedding model {model_id!r}: "
                f"cache/local load failed ({e_local!r}); "
                f"hub download failed ({e_hub!r}). "
                "Use a working network once to populate the Hugging Face cache, or copy "
                "the cache from another machine. For air-gapped runs with a filled cache, "
                "set HF_HUB_OFFLINE=1."
            ) from e_hub


def load_parent_store(vector_store_dir: Path) -> Dict[str, Any]:
    """Load rag_parents.json written by Part 1 (parent-child indexing). Empty dict if missing."""
    from utils.rag_index_build import load_parent_store as _load

    return _load(vector_store_dir)


def _doc_dedupe_key(doc: Any) -> Tuple[Any, ...]:
    """Stable key for a retrieved chunk (parent+child or legacy title)."""
    m = getattr(doc, "metadata", None) or {}
    pid, cid = m.get("parent_id"), m.get("child_index")
    if pid is not None and cid is not None:
        return ("pc", pid, cid)
    title = m.get("retrieval_title") or m.get("title", "")
    return ("legacy", title, m.get("source_file", ""))


def balance_vector_hits_by_source_file(
    hits: List[Tuple[Any, float]],
    top_k: int,
    *,
    max_per_source: Optional[int] = None,
) -> List[Tuple[Any, float]]:
    """
    Cap how many hits may come from one ``metadata['source_file']`` so retrieval is not
    dominated by a single KB file (e.g. a large JSON vs. sparse PDFs). Still respects
    global similarity order within those caps; remaining slots are filled from the
    sorted list without a per-source cap.

    ``max_per_source`` defaults to ``ceil(top_k / n_distinct_sources)`` among hits.
    """
    if not hits or top_k <= 0:
        return []

    best: Dict[Tuple[Any, ...], Tuple[Any, float]] = {}
    for doc, dist in hits:
        k = _doc_dedupe_key(doc)
        d = float(dist)
        if k not in best or d < float(best[k][1]):
            best[k] = (doc, d)

    unique_sorted = sorted(best.values(), key=lambda x: float(x[1]))
    sources = list(
        {
            (getattr(d, "metadata", None) or {}).get("source_file", "?")
            for d, _ in unique_sorted
        }
    )
    n_sources = len(sources) or 1
    if max_per_source is None:
        max_per_source = max(1, (top_k + n_sources - 1) // n_sources)

    picked: List[Tuple[Any, float]] = []
    seen: Set[Tuple[Any, ...]] = set()
    counts: Dict[str, int] = {}

    for doc, dist in unique_sorted:
        if len(picked) >= top_k:
            break
        dk = _doc_dedupe_key(doc)
        if dk in seen:
            continue
        src = (getattr(doc, "metadata", None) or {}).get("source_file", "?")
        if counts.get(src, 0) >= max_per_source:
            continue
        picked.append((doc, dist))
        seen.add(dk)
        counts[src] = counts.get(src, 0) + 1

    for doc, dist in unique_sorted:
        if len(picked) >= top_k:
            break
        dk = _doc_dedupe_key(doc)
        if dk in seen:
            continue
        picked.append((doc, dist))
        seen.add(dk)

    return picked[:top_k]


def load_vector_store(
    vector_store_dir: Path,
    embed_model: Optional[str] = None,
) -> FAISS:
    """Load FAISS from disk; uses manifest embed_model if embed_model is None."""
    vector_store_dir = Path(vector_store_dir)
    if not vector_store_dir.is_dir() or not any(vector_store_dir.iterdir()):
        raise FileNotFoundError(
            f"Vector store directory missing or empty: {vector_store_dir}. "
            "Run RAG_part1_build_vector_store.ipynb first."
        )
    manifest = read_manifest(vector_store_dir)
    model = embed_model or manifest.get("embed_model") or DEFAULT_EMBED_MODEL
    embeddings = _sentence_transformer_embeddings_for_faiss_load(model)
    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------------
# SHAP party/tier + action-plan JSON export (Part 2)
# ---------------------------------------------------------------------------


def get_party_to_tier_mapping(sample: Dict[str, Any]) -> Dict[str, str]:
    shap_expl = sample.get("shap_explanation", {}) or {}
    party_contributions = shap_expl.get("party_contributions", {}) or {}
    party_to_tier: Dict[str, str] = {}
    for party_name in party_contributions.keys():
        party_lower = party_name.lower()
        if party_lower.startswith("ran"):
            party_to_tier[party_name] = "RAN"
        elif party_lower.startswith("edge"):
            party_to_tier[party_name] = "Edge"
        elif party_lower.startswith("core"):
            party_to_tier[party_name] = "Core"
    return party_to_tier


def get_dominant_party_info(sample: Dict[str, Any]) -> Tuple[str, str, float]:
    shap_expl = sample.get("shap_explanation", {}) or {}
    dominant_agent = shap_expl.get("dominant_agent", "")
    dominant_pct = shap_expl.get("dominant_contribution_pct", 0.0) or shap_expl.get(
        "party_contributions_pct", {}
    ).get(dominant_agent, 0.0)
    if isinstance(dominant_pct, dict):
        dominant_pct = dominant_pct.get(dominant_agent, 0.0)
    party_to_tier = get_party_to_tier_mapping(sample)
    network_tier = party_to_tier.get(dominant_agent, "") if dominant_agent else ""
    return (dominant_agent, network_tier, float(dominant_pct) * 100.0)


def _rag_result_score_fields(r: Dict[str, Any]) -> Dict[str, Any]:
    """Split ranking (CrossEncoder logit, can be negative) vs vector similarity (typically [0, 1])."""
    rank = float(r.get("score", 0.0) or 0.0)
    vs = r.get("vector_similarity")
    out: Dict[str, Any] = {"ranking_score": rank}
    if vs is not None:
        fv = float(vs)
        out["vector_similarity"] = fv
        out["similarity_score"] = fv
    else:
        out["similarity_score"] = rank
    ce = r.get("crossencoder_score")
    if ce is not None:
        out["crossencoder_score"] = float(ce)
    return out


def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    return obj


def save_comparison_file(
    sample: Dict[str, Any],
    query: str,
    rag_results: List[Dict[str, Any]],
    llm_response_with_rag: Optional[Dict[str, Any]],
    llm_response_without_rag: Optional[Dict[str, Any]],
    results_action_dir: Path,
    with_rag_file: Optional[Path],
    without_rag_file: Optional[Path],
) -> Path:
    comparison = {
        "sample_id": sample.get("sample_id"),
        "prediction": {
            "predicted_label": sample.get("predicted_label"),
            "confidence": float(sample.get("confidence", 0.0)),
        },
        "comparison_metadata": {
            "compared_at": datetime.now().isoformat(),
            "with_rag_file": with_rag_file.name if with_rag_file else None,
            "without_rag_file": without_rag_file.name if without_rag_file else None,
            "rag_query": query,
            "rag_documents_found": len(rag_results),
        },
        "rag_context": {
            "documents_used": len(rag_results),
            "top_documents": [
                {
                    **_rag_result_score_fields(r),
                    "title": r["title"],
                    "text_preview": (
                        r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"]
                    ),
                }
                for r in rag_results[:5]
            ],
        },
        "action_plans": {
            "with_rag": {
                "threat_level": (
                    llm_response_with_rag.get("threat_level")
                    if llm_response_with_rag
                    else None
                ),
                "execution_priority": (
                    llm_response_with_rag.get("execution_priority")
                    if llm_response_with_rag
                    else None
                ),
                "primary_actions": (
                    llm_response_with_rag.get("primary_actions", [])
                    if llm_response_with_rag
                    else []
                ),
                "supporting_actions": (
                    llm_response_with_rag.get("supporting_actions", [])
                    if llm_response_with_rag
                    else []
                ),
                "overall_reasoning": (
                    (
                        llm_response_with_rag.get("overall_reasoning")
                        or llm_response_with_rag.get("reasoning", "")
                    )
                    if llm_response_with_rag
                    else ""
                ),
                "knowledge_sources_used": (
                    llm_response_with_rag.get("knowledge_sources_used", [])
                    if llm_response_with_rag
                    else []
                ),
            },
            "without_rag": {
                "threat_level": (
                    llm_response_without_rag.get("threat_level")
                    if llm_response_without_rag
                    else None
                ),
                "execution_priority": (
                    llm_response_without_rag.get("execution_priority")
                    if llm_response_without_rag
                    else None
                ),
                "primary_actions": (
                    llm_response_without_rag.get("primary_actions", [])
                    if llm_response_without_rag
                    else []
                ),
                "supporting_actions": (
                    llm_response_without_rag.get("supporting_actions", [])
                    if llm_response_without_rag
                    else []
                ),
                "overall_reasoning": (
                    (
                        llm_response_without_rag.get("overall_reasoning")
                        or llm_response_without_rag.get("reasoning", "")
                    )
                    if llm_response_without_rag
                    else ""
                ),
                "knowledge_sources_used": (
                    llm_response_without_rag.get("knowledge_sources_used", [])
                    if llm_response_without_rag
                    else []
                ),
            },
        },
        "differences": {
            "threat_level_different": (
                (llm_response_with_rag.get("threat_level") if llm_response_with_rag else None)
                != (
                    llm_response_without_rag.get("threat_level")
                    if llm_response_without_rag
                    else None
                )
            ),
            "priority_different": (
                (
                    llm_response_with_rag.get("execution_priority")
                    if llm_response_with_rag
                    else None
                )
                != (
                    llm_response_without_rag.get("execution_priority")
                    if llm_response_without_rag
                    else None
                )
            ),
            "primary_actions_different": (
                (
                    llm_response_with_rag.get("primary_actions", [])
                    if llm_response_with_rag
                    else []
                )
                != (
                    llm_response_without_rag.get("primary_actions", [])
                    if llm_response_without_rag
                    else []
                )
            ),
            "num_primary_actions_with_rag": (
                len(llm_response_with_rag.get("primary_actions", []))
                if llm_response_with_rag
                else 0
            ),
            "num_primary_actions_without_rag": (
                len(llm_response_without_rag.get("primary_actions", []))
                if llm_response_without_rag
                else 0
            ),
            "overall_reasoning_different": (
                (
                    (
                        llm_response_with_rag.get("overall_reasoning", "")
                        or llm_response_with_rag.get("reasoning", "")
                    )
                    if llm_response_with_rag
                    else ""
                )
                != (
                    (
                        llm_response_without_rag.get("overall_reasoning", "")
                        or llm_response_without_rag.get("reasoning", "")
                    )
                    if llm_response_without_rag
                    else ""
                )
            ),
        },
    }
    comparison = convert_to_json_serializable(comparison)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = (
        results_action_dir / f"comparison_sample_{sample.get('sample_id')}_{timestamp}.json"
    )
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    return comparison_file


def save_action_plan(
    sample: Dict[str, Any],
    query: str,
    rag_results: List[Dict[str, Any]],
    llm_response: Optional[Dict[str, Any]],
    results_action_dir: Path,
    *,
    variant: Optional[str] = None,
    timestamp: Optional[str] = None,
    prompt_uses_rag: Optional[bool] = None,
) -> Path:
    result = {
        "sample_id": sample.get("sample_id"),
        "prediction": {
            "predicted_label": sample.get("predicted_label"),
            "confidence": float(sample.get("confidence", 0.0)),
        },
        "rag_info": {
            "documents_used": len(rag_results),
            "search_method": "vector_similarity",
            "search_queries": [query] if query else [],
            "top_results": [
                {
                    **_rag_result_score_fields(r),
                    "title": r["title"],
                    "text": r["text"],
                }
                for r in rag_results[:5]
            ],
        },
        "llm_action_plan": llm_response,
        "processed_at": datetime.now().isoformat(),
    }
    if prompt_uses_rag is not None:
        result["prompt_uses_rag"] = prompt_uses_rag
    result = convert_to_json_serializable(result)
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    sid = sample.get("sample_id")
    if variant:
        output_file = results_action_dir / f"action_plan_sample_{sid}_{variant}_{ts}.json"
    else:
        output_file = results_action_dir / f"action_plan_sample_{sid}_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    if not output_file.exists():
        raise FileNotFoundError(f"File was not created: {output_file}")
    return output_file
