"""RAG retrieval scoring (BERTScore / ROUGE) for pragma-rag-eval-x / gold-core.

Gold-core writes ``experiments/rag/rag_eval100/``. Off-gold ``--offgold-n`` writes
``experiments/rag/hybrid_eval{n}/``. Reads gold JSON as the text baseline only.
Does not edit gold or write into gold-100 / reason / detect-predict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from openai import OpenAI

from scripts.env import LIVE_RAG_EVAL, experiment_dir, load_project_dotenv, named_live_dir
from scripts.llm_prompt import create_agentic_orchestration_prompt
from scripts.rag_io import load_attack_and_agentic, load_parent_store, load_predictions, load_vector_store
from scripts import reason as reason_mod
from scripts.reason import (
    VECTOR_STORE_DIR,
    _DEFAULT_MMR_K,
    _DEFAULT_RERANK_K,
    _LLM_RAG_SECTIONS_IN_PROMPT,
    _PER_QUERY_RETRIEVE_K,
    build_template_rag_query,
    expand_parent_sections,
    merge_and_dedupe_child_chunks,
    mmr_select,
    rank_by_vector_score,
    retrieve_child_chunks_for_query,
)
from scripts.vfl import load_attack_actions_by_type

load_project_dotenv()

GOLD = _REPO / "experiments" / "gold-100" / "ground_truth-100.json"
FIXTURE_FLOWS = _REPO / "experiments" / "data" / "gold-100" / "flows.csv"
FIXTURE_MANIFEST = _REPO / "experiments" / "data" / "gold-100" / "manifest.json"
GOLD_LIVE = _REPO / "experiments" / "gold-100"
DOCS_REPORT = _REPO / "docs" / "rag_experiment_report.md"
NINE = (
    "BENIGN",
    "BOT",
    "DDOS",
    "DOS",
    "FTPPATATOR",
    "OTHERS",
    "PORTSCAN",
    "SSHPATATOR",
    "WEBATTACK",
)
SYSTEMS = (
    {"id": "LLM_only", "rag": False, "rank": False, "shap": False, "report": "No RAG"},
    {"id": "RAG_No_Ranking", "rag": True, "rank": False, "shap": False, "report": "RAG only"},
    {"id": "RAG_RANKING", "rag": True, "rank": True, "shap": True, "report": "Ranked RAG + SHAP"},
)
SYSTEM_IDS = [s["id"] for s in SYSTEMS]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _norm_action(s: str) -> str:
    return " ".join((s or "").lower().strip().split())


def _out_dir() -> Path:
    return named_live_dir("rag", LIVE_RAG_EVAL, mkdir=True)


def verify_inputs() -> dict[str, Any]:
    if not GOLD.is_file():
        raise SystemExit(f"Gold missing: {GOLD}. Run pragma-gold-100 first.")
    if not FIXTURE_FLOWS.is_file():
        raise SystemExit(f"Fixture flows missing: {FIXTURE_FLOWS}")
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    fixture = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    flows_hash = _sha256_file(FIXTURE_FLOWS)
    expect_hash = str(fixture.get("flows_sha256") or "")
    if expect_hash and flows_hash != expect_hash:
        raise SystemExit(f"flows.csv SHA-256 mismatch: {flows_hash} != {expect_hash}")
    gold_hash = _sha256_file(GOLD)
    cases = gold.get("cases") or []
    if len(cases) != 90:
        raise SystemExit(f"gold n={len(cases)} expected 90")
    hist = Counter(str((c.get("condition") or {}).get("true_label") or "") for c in cases)
    for lab in NINE:
        if hist.get(lab, 0) != 10:
            raise SystemExit(f"{lab} count {hist.get(lab)} expected 10")
    gold_idx = {int(c["split_index"]) for c in cases}
    fixture_idx = set(int(x) for x in fixture.get("selected_split_index") or [])
    if gold_idx != fixture_idx:
        raise SystemExit("gold split_index set != data/gold-100 manifest")
    for reason_name in ("rag_reason_500", "e2e-detect-chain"):
        reason_man = _REPO / "experiments" / "data" / reason_name / "manifest.json"
        if not reason_man.is_file():
            continue
        other = set(json.loads(reason_man.read_text(encoding="utf-8")).get("selected_split_index") or [])
        overlap = gold_idx & {int(x) for x in other}
        if overlap:
            raise SystemExit(f"gold split_index overlaps {reason_name}: {len(overlap)}")
    pred_files = sorted(GOLD_LIVE.glob("predictions_detailed_*.json"))
    if not pred_files:
        raise SystemExit(f"No predictions_detailed_*.json in {GOLD_LIVE}")
    return {
        "gold": gold,
        "gold_hash": gold_hash,
        "flows_hash": flows_hash,
        "pred_path": pred_files[-1],
        "fixture": fixture,
    }


def load_paired_cases() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta = verify_inputs()
    gold_cases = meta["gold"]["cases"]
    with FIXTURE_FLOWS.open(encoding="utf-8", newline="") as f:
        flows = list(csv.DictReader(f))
    preds = load_predictions(GOLD_LIVE, verbose=False)
    if len(flows) != 90 or len(preds) != 90 or len(gold_cases) != 90:
        raise SystemExit(f"expected 90, got flows={len(flows)} preds={len(preds)} gold={len(gold_cases)}")
    by_split: dict[int, dict[str, Any]] = {}
    for flow, pred in zip(flows, preds):
        six = int(float(flow["split_index"]))
        row = dict(pred)
        row["split_index"] = six
        row["sample_id"] = six
        by_split[six] = row
    paired = []
    for case in gold_cases:
        six = int(case["split_index"])
        pred = by_split.get(six)
        if pred is None:
            raise SystemExit(f"no detect row for split_index={six}")
        paired.append({"case": case, "sample": pred})
    return paired, meta


def flatten_out_dir(out: Path) -> None:
    """Keep jsonl + scored files at the task root. Drop leftover subfolders."""
    raw = out / "raw"
    if raw.is_dir():
        for src in raw.glob("*.jsonl"):
            dest = out / src.name
            if not dest.is_file() or src.stat().st_mtime >= dest.stat().st_mtime:
                dest.write_bytes(src.read_bytes())
        shutil.rmtree(raw)
    for name in ("action", "examples", "rationale", "retrieval", "summaries"):
        extra = out / name
        if extra.is_dir():
            shutil.rmtree(extra)


def _jsonl_path(out: Path, system_id: str) -> Path:
    """Traces live at the task root (no raw/ subfolder)."""
    flatten_out_dir(out)
    return out / f"{system_id}.jsonl"


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = rec.get("case_id")
        if cid:
            done.add(str(cid))
    return done


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _llm_call(prompt: str) -> tuple[dict[str, Any] | None, str, dict[str, Any], bool]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            try:
                resp = client.chat.completions.create(model=model, messages=messages, temperature=0)
            except Exception as exc:
                err = str(exc).lower()
                if "temperature" in err and "unsupported" in err:
                    resp = client.chat.completions.create(model=model, messages=messages)
                else:
                    raise
            raw = (resp.choices[0].message.content or "").strip()
            usage: dict[str, Any] = {}
            if resp.usage is not None:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }
            parsed: dict[str, Any] | None = None
            format_error = True
            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(raw[start:end])
                    format_error = not isinstance(parsed, dict) or "parse_error" in parsed
            except Exception as exc:
                parsed = {"parse_error": str(exc)}
                format_error = True
            return parsed, raw, usage, format_error
        except Exception as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise SystemExit(f"LLM call failed after retries: {last_err}")


def _retrieve(
    vector_store: Any,
    query: str,
    *,
    rank: bool,
    sample: dict[str, Any] | None = None,
    exclude_case_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float | None, float | None]:
    from scripts.rag_bridge import retrieve_context

    return retrieve_context(
        vector_store,
        sample,
        rank=rank,
        fallback_query=query,
        exclude_case_id=exclude_case_id,
    )


def _first_action(plan: dict[str, Any] | None) -> tuple[str, str]:
    if not plan:
        return "", ""
    items = plan.get("primary_actions") or []
    if not isinstance(items, list) or not items:
        return "", ""
    first = items[0]
    if isinstance(first, dict):
        return _norm_action(str(first.get("action") or "")), str(first.get("network_tier") or "")
    return _norm_action(str(first)), ""


def _action_items(items: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            act = _norm_action(str(it.get("action") or ""))
            if act:
                out.append(
                    {
                        "action": act,
                        "network_tier": str(it.get("network_tier") or ""),
                        "party_evidence_type": it.get("party_evidence_type") or "",
                        "reasoning": it.get("reasoning") or "",
                    }
                )
        else:
            act = _norm_action(str(it or ""))
            if act:
                out.append({"action": act, "network_tier": "", "party_evidence_type": "", "reasoning": ""})
    return out


def _plan_action_lists(plan: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if not isinstance(plan, dict):
        return [], [], []
    prim = _action_items(plan.get("primary_actions"))
    supp = _action_items(plan.get("supporting_actions"))
    all_a = [_norm_action(str(a)) for a in (plan.get("all_actions") or []) if str(a).strip()]
    if not all_a:
        seen: set[str] = set()
        all_a = []
        for it in prim + supp:
            a = it["action"]
            if a not in seen:
                seen.add(a)
                all_a.append(a)
    return prim, supp, all_a


def _llm_rationale(plan: dict[str, Any] | None) -> str:
    if not plan:
        return ""
    parts = [str(plan.get("overall_reasoning") or "").strip()]
    for key in ("primary_actions", "supporting_actions"):
        for it in plan.get(key) or []:
            if isinstance(it, dict) and it.get("reasoning"):
                parts.append(str(it.get("reasoning")))
    return " ".join(p for p in parts if p)


def run_systems(
    paired: list[dict[str, Any]],
    systems: list[str],
    limit: int | None,
    *,
    out: Path | None = None,
) -> Path:
    out = out or _out_dir()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (backend/.env)")
    attack_actions, agentic_features = load_attack_and_agentic(verbose=False)
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    reason_mod.vector_store = vector_store
    reason_mod._RAG_PARENTS = load_parent_store(VECTOR_STORE_DIR)
    reason_mod.predictions_data = [{"_eval100": True}]
    reason_mod.attack_actions_data = attack_actions
    reason_mod.agentic_features_data = agentic_features

    rows = paired[: int(limit)] if limit else paired
    want = {s["id"]: s for s in SYSTEMS if s["id"] in systems}
    for sys_id, spec in want.items():
        path = _jsonl_path(out, sys_id)
        done = _done_ids(path)
        print(f"\n=== {sys_id} already={len(done)} remain={len(rows) - sum(1 for r in rows if r['case']['case_id'] in done)} ===")
        for i, item in enumerate(rows, 1):
            case = item["case"]
            sample = item["sample"]
            cid = str(case["case_id"])
            if cid in done:
                continue
            print(f"  [{i}/{len(rows)}] {cid} {case['condition']['true_label']} …", flush=True)
            t_all = time.perf_counter()
            query = build_template_rag_query(sample)
            prompt_secs: list[dict[str, Any]] = []
            ir_secs: list[dict[str, Any]] = []
            retrieve_ms = None
            rank_ms = None
            if spec["rag"]:
                prompt_secs, ir_secs, retrieve_ms, rank_ms = _retrieve(
                    vector_store,
                    query,
                    rank=bool(spec["rank"]),
                    sample=sample,
                    exclude_case_id=cid,
                )
            prompt = create_agentic_orchestration_prompt(
                sample,
                prompt_secs if spec["rag"] else [],
                attack_actions,
                agentic_features,
                include_knowledge_base=bool(spec["rag"]),
                include_conditions=bool(spec["shap"]),
            )
            t0 = time.perf_counter()
            plan, raw, usage, format_error = _llm_call(prompt)
            llm_ms = _ms(t0)
            rec = {
                "case_id": cid,
                "split_index": int(case["split_index"]),
                "system": sys_id,
                "true_label": case["condition"]["true_label"],
                "predicted_label": sample.get("predicted_label"),
                "confidence": sample.get("confidence"),
                "query": query if spec["rag"] else "",
                "prompt_rag": bool(spec["rag"]),
                "prompt_shap": bool(spec["shap"]),
                "retrieved_parent_ids": [str(s.get("parent_id") or "") for s in ir_secs if s.get("parent_id")],
                "prompt_parent_ids": [str(s.get("parent_id") or "") for s in prompt_secs if s.get("parent_id")],
                "retrieved_sources": [str(s.get("source_file") or "") for s in prompt_secs],
                "plan": plan,
                "overall_reasoning": (plan or {}).get("overall_reasoning") if isinstance(plan, dict) else "",
                "format_error": format_error,
                "latency": {
                    "retrieve_ms": retrieve_ms,
                    "rank_ms": rank_ms,
                    "llm_ms": llm_ms,
                    "total_ms": _ms(t_all),
                },
                "usage": usage,
                "raw_chars": len(raw),
                "hybrid": (prompt_secs[0].get("hybrid") if prompt_secs else None),
            }
            _append_jsonl(path, rec)
            done.add(cid)
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def _recall(gold: set[str], ranked: list[str], k: int) -> float | None:
    if not gold:
        return None
    return len(set(ranked[:k]) & gold) / len(gold)


def _mrr(gold: set[str], ranked: list[str]) -> float | None:
    if not gold:
        return None
    for i, pid in enumerate(ranked, 1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def _ndcg(gold: set[str], ranked: list[str], k: int) -> float | None:
    if not gold:
        return None
    rels = [1.0 if p in gold else 0.0 for p in ranked[:k]]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal = [1.0] * min(len(gold), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg else 0.0


def _mean(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return None
    return sum(xs) / len(xs)


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if len(p.split()) >= 5]


def _atomic_coverage(points: list[str], hyp: str) -> float | None:
    if not points:
        return None
    hyp_tok = _content_tokens(hyp)
    if not hyp_tok:
        return 0.0
    hits = 0
    for p in points:
        toks = _content_tokens(p)
        if toks and len(toks & hyp_tok) >= min(3, len(toks)):
            hits += 1
    return hits / len(points)


def _unsupported_rate(hyp: str, ref: str, extra: str = "") -> float | None:
    sents = _sentences(hyp)
    if not sents:
        return None
    blob = _content_tokens(ref + " " + extra)
    if not blob:
        return 1.0
    bad = 0
    for s in sents:
        toks = _content_tokens(s)
        if len(toks & blob) < 2:
            bad += 1
    return bad / len(sents)


def _groundedness(hyp: str, retrieved_text: str) -> float | None:
    sents = _sentences(hyp)
    if not sents:
        return None
    blob = _content_tokens(retrieved_text)
    if not blob:
        return 0.0
    ok = 0
    for s in sents:
        toks = _content_tokens(s)
        if len(toks & blob) >= 2:
            ok += 1
    return ok / len(sents)


# BERTScore/RoBERTa sees ~512 tokens; keep the start of each concat (gold parents are ordered).
_SCORE_CHARS = 4000


def _gold_chunk_concat(case: dict[str, Any]) -> str:
    """Baseline: concatenate gold retrieval parent bodies (not chunks_summary)."""
    parts = []
    for ch in case.get("relevant_rag_chunks") or []:
        t = str(ch.get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip() or " "


def _prompt_parent_concat(rec: dict[str, Any], parents: dict[str, Any]) -> str:
    parts = []
    for pid in rec.get("prompt_parent_ids") or []:
        t = str((parents.get(str(pid)) or {}).get("text") or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def _system_score_text(rationale: str, prompt_parents: str, rag: bool) -> str:
    """No-RAG = rationale only. RAG = retrieved parents first, then rationale."""
    rat = (rationale or "").strip()
    ctx = (prompt_parents or "").strip()
    if rag and ctx:
        return f"{ctx}\n\n{rat}".strip()
    return rat or " "


def _clip_score_text(s: str) -> str:
    t = (s or "").strip() or " "
    return t[:_SCORE_CHARS]


@lru_cache(maxsize=1)
def _rouge_scorer():
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def _rouge_scores(ref: str, hyp: str) -> tuple[float, float]:
    s = _rouge_scorer().score(ref or " ", hyp or " ")
    return float(s["rouge1"].fmeasure), float(s["rougeL"].fmeasure)


_BERT_SCORER = None


def _bert_scorer():
    """Load roberta-large once. Calling bert_score() per batch reloads the model (~1 min each)."""
    global _BERT_SCORER
    if _BERT_SCORER is None:
        from bert_score import BERTScorer

        print("BERTScore: loading roberta-large once (CPU)")
        _BERT_SCORER = BERTScorer(lang="en", device="cpu", batch_size=16)
    return _BERT_SCORER


def _bertscore_f1(refs: list[str], hyps: list[str], batch_size: int = 32) -> list[float]:
    """One model load; chunk only to avoid Windows 0xC0000005 on huge tensors."""
    import gc

    scorer = _bert_scorer()
    safe_refs = [r if (r or "").strip() else " " for r in refs]
    safe_hyps = [h if (h or "").strip() else " " for h in hyps]
    out: list[float] = []
    for i in range(0, len(safe_refs), batch_size):
        chunk_r = safe_refs[i : i + batch_size]
        chunk_h = safe_hyps[i : i + batch_size]
        print(f"  BERTScore {i + 1}–{i + len(chunk_r)} / {len(safe_refs)}")
        _p, _r, f1 = scorer.score(chunk_h, chunk_r)
        out.extend(float(x) for x in f1.tolist())
        del _p, _r, f1
        gc.collect()
    return out


def score_run(paired: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    out = _out_dir()
    gold_by_id = {c["case"]["case_id"]: c["case"] for c in paired}
    catalog = load_attack_actions_by_type()
    allowed_all = {_norm_action(a) for acts in catalog.values() for a in acts}
    parents = load_parent_store(VECTOR_STORE_DIR)

    case_rows: list[dict[str, Any]] = []
    by_sys: dict[str, list[dict[str, Any]]] = {}
    for spec in SYSTEMS:
        recs = _read_jsonl(_jsonl_path(out, spec["id"]))
        by_sys[spec["id"]] = recs
        for rec in recs:
            case = gold_by_id.get(rec["case_id"])
            if not case:
                continue
            plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
            format_error = bool(rec.get("format_error"))
            act, tier = _first_action(plan)
            gold_act = _norm_action(case["actions"]["primary_action"])
            accept = {_norm_action(a) for a in case["actions"].get("acceptable_actions") or []}
            unsafe = {_norm_action(a) for a in case["actions"].get("unsafe_actions") or []}
            hyp = _llm_rationale(plan)
            gold_ref = _gold_chunk_concat(case)
            gold_ids = {str(ch.get("parent_id")) for ch in (case.get("relevant_rag_chunks") or []) if ch.get("parent_id")}
            ranked = [p for p in (rec.get("retrieved_parent_ids") or []) if p]
            prompt_txt = _prompt_parent_concat(rec, parents) if spec["rag"] else ""
            system_text = _system_score_text(hyp, prompt_txt, spec["rag"])
            retrieved_text = prompt_txt
            skip_ir = (not spec["rag"]) or (not gold_ids)
            case_rows.append(
                {
                    "case_id": rec["case_id"],
                    "split_index": rec.get("split_index"),
                    "system": spec["id"],
                    "true_label": case["condition"]["true_label"],
                    "predicted_label": rec.get("predicted_label"),
                    "format_error": int(format_error),
                    "exact_action": int(bool(act) and act == gold_act and not format_error),
                    "acceptable_action": int(bool(act) and act in accept and not format_error),
                    "unsafe_action": int(bool(act) and act in unsafe and not format_error),
                    "catalog_violation": int(bool(act) and act not in allowed_all),
                    "tier_match": int(tier == case["condition"]["primary_network_tier"] and not format_error),
                    "pred_action": act,
                    "gold_action": gold_act,
                    "hyp": hyp,
                    "gold_ref": gold_ref,
                    "system_text": system_text,
                    "chunks_summary": case.get("chunks_summary") or "",
                    "rationale_summary": case.get("rationale_summary") or "",
                    "atomic_coverage": _atomic_coverage(case.get("atomic_reasoning_points") or [], hyp),
                    "unsupported_claim": _unsupported_rate(hyp, gold_ref, retrieved_text),
                    "groundedness": None if not spec["rag"] else _groundedness(hyp, retrieved_text or gold_ref),
                    "recall@5": None if skip_ir else _recall(gold_ids, ranked, 5),
                    "recall@10": None if skip_ir else _recall(gold_ids, ranked, 10),
                    "mrr": None if skip_ir else _mrr(gold_ids, ranked),
                    "ndcg@5": None if skip_ir else _ndcg(gold_ids, ranked, 5),
                    "ndcg@10": None if skip_ir else _ndcg(gold_ids, ranked, 10),
                    "retrieve_ms": (rec.get("latency") or {}).get("retrieve_ms"),
                    "rank_ms": (rec.get("latency") or {}).get("rank_ms"),
                    "llm_ms": (rec.get("latency") or {}).get("llm_ms"),
                }
            )

    retrieve_pairs: list[tuple[dict[str, Any], str]] = []
    for spec in SYSTEMS:
        if not spec["rag"]:
            continue
        for rec in by_sys.get(spec["id"]) or []:
            retrieve_pairs.append((rec, _prompt_parent_concat(rec, parents)))

    if case_rows:
        refs = [_clip_score_text(r["gold_ref"]) for r in case_rows]
        hyps = [_clip_score_text(r["system_text"]) for r in case_rows]
        print(f"BERTScore on {len(case_rows)} pairs vs gold chunk concat …")
        f1s = _bertscore_f1(refs, hyps)
        for row, f1 in zip(case_rows, f1s):
            row["bertscore_f1"] = f1
            r1, rl = _rouge_scores(_clip_score_text(row["gold_ref"]), _clip_score_text(row["system_text"]))
            row["rouge_1"] = r1
            row["rouge_l"] = rl
    else:
        print("No jsonl rows to score")

    retrieve_ix: dict[tuple[str, str], dict[str, float]] = {}
    if retrieve_pairs:
        recs_rt, hyps_rt = zip(*retrieve_pairs)
        refs_rt = [_clip_score_text(_gold_chunk_concat(gold_by_id[r["case_id"]])) for r in recs_rt]
        print(f"BERTScore on {len(hyps_rt)} retrieve-only vs gold chunk concat …")
        f1s_rt = _bertscore_f1(refs_rt, [_clip_score_text(h) for h in hyps_rt])
        for rec, hyp, f1 in zip(recs_rt, hyps_rt, f1s_rt):
            r1, rl = _rouge_scores(
                _clip_score_text(_gold_chunk_concat(gold_by_id[rec["case_id"]])),
                _clip_score_text(hyp),
            )
            retrieve_ix[(rec["system"], rec["case_id"])] = {
                "bertscore_f1": f1,
                "rouge_1": r1,
                "rouge_l": rl,
            }
    for row in case_rows:
        extra = retrieve_ix.get((row["system"], row["case_id"]))
        if extra:
            row["retrieve_bertscore_f1"] = extra["bertscore_f1"]
            row["retrieve_rouge_1"] = extra["rouge_1"]
            row["retrieve_rouge_l"] = extra["rouge_l"]

    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in case_rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)

    cases_out: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for item in paired:
        gold = item["case"]
        cid = str(gold["case_id"])
        rows = {r["system"]: r for r in by_case.get(cid, [])}
        entry: dict[str, Any] = {
            "case_id": cid,
            "split_index": gold.get("split_index"),
            "true_label": gold["condition"]["true_label"],
            "gold": {
                "primary_action": gold["actions"]["primary_action"],
                "acceptable_actions": gold["actions"].get("acceptable_actions") or [],
                "tier": gold["condition"]["primary_network_tier"],
                "chunks_summary": gold.get("chunks_summary") or "",
                "chunk_concat": _gold_chunk_concat(gold),
            },
        }
        csv_row: dict[str, Any] = {
            "case_id": cid,
            "true_label": entry["true_label"],
            "gold_action": entry["gold"]["primary_action"],
            "gold_tier": entry["gold"]["tier"],
        }
        for spec in SYSTEMS:
            sid = spec["id"]
            r = rows.get(sid)
            if r:
                rec = next((x for x in (by_sys.get(sid) or []) if str(x.get("case_id")) == cid), None)
                plan = rec.get("plan") if rec and isinstance(rec.get("plan"), dict) else None
                _act, pred_tier = _first_action(plan)
                block = {
                    "plan": plan,
                    "action": r.get("pred_action") or "",
                    "tier": pred_tier,
                    "overall_reasoning": (plan or {}).get("overall_reasoning") or "",
                    "exact": bool(r.get("exact_action")),
                    "acceptable": bool(r.get("acceptable_action")),
                    "unsafe": bool(r.get("unsafe_action")),
                    "format_error": bool(r.get("format_error")),
                    "bertscore_f1": r.get("bertscore_f1"),
                    "rouge_1": r.get("rouge_1"),
                    "rouge_l": r.get("rouge_l"),
                }
                if spec["rag"]:
                    block["recall@5"] = r.get("recall@5")
                    block["mrr"] = r.get("mrr")
                    block["retrieve_bertscore_f1"] = r.get("retrieve_bertscore_f1")
                    block["retrieve_rouge_1"] = r.get("retrieve_rouge_1")
                    block["retrieve_rouge_l"] = r.get("retrieve_rouge_l")
            else:
                block = None
            entry[sid] = block
            prefix = {"LLM_only": "A", "RAG_No_Ranking": "B", "RAG_RANKING": "C"}[sid]
            csv_row[f"{prefix}_action"] = (block or {}).get("action") if block else ""
            csv_row[f"{prefix}_exact"] = (block or {}).get("exact") if block else ""
            csv_row[f"{prefix}_acceptable"] = (block or {}).get("acceptable") if block else ""
            csv_row[f"{prefix}_bertscore"] = (block or {}).get("bertscore_f1") if block else ""
            csv_row[f"{prefix}_rouge1"] = (block or {}).get("rouge_1") if block else ""
            csv_row[f"{prefix}_rougeL"] = (block or {}).get("rouge_l") if block else ""
            if spec["rag"]:
                csv_row[f"{prefix}_recall@5"] = (block or {}).get("recall@5") if block else ""
                csv_row[f"{prefix}_mrr"] = (block or {}).get("mrr") if block else ""
        cases_out.append(entry)
        csv_rows.append(csv_row)

    def agg(sys_id: str) -> dict[str, Any]:
        rows = [r for r in case_rows if r["system"] == sys_id]
        n = len(rows)

        def rate(key: str) -> float | None:
            if not rows:
                return None
            return sum(int(r[key]) for r in rows) / n

        return {
            "system": sys_id,
            "n": n,
            "exact_action": rate("exact_action"),
            "acceptable_action": rate("acceptable_action"),
            "unsafe_action": rate("unsafe_action"),
            "catalog_violation": rate("catalog_violation"),
            "format_error": rate("format_error"),
            "bertscore_f1": _mean([r.get("bertscore_f1") for r in rows]),
            "rouge_1": _mean([r.get("rouge_1") for r in rows]),
            "rouge_l": _mean([r.get("rouge_l") for r in rows]),
            "retrieve_bertscore_f1": _mean([r.get("retrieve_bertscore_f1") for r in rows]),
            "retrieve_rouge_1": _mean([r.get("retrieve_rouge_1") for r in rows]),
            "retrieve_rouge_l": _mean([r.get("retrieve_rouge_l") for r in rows]),
            "recall@5": _mean([r.get("recall@5") for r in rows]),
            "mrr": _mean([r.get("mrr") for r in rows]),
        }

    means = [agg(s["id"]) for s in SYSTEMS]
    means_map = {m["system"]: m for m in means}
    means_by_class = {}
    for lab in NINE:
        means_by_class[lab] = {}
        for spec in SYSTEMS:
            sub = [r for r in case_rows if r["system"] == spec["id"] and r["true_label"] == lab]
            n = len(sub)
            means_by_class[lab][spec["id"]] = {
                "n": n,
                "bertscore_f1": _mean([r.get("bertscore_f1") for r in sub]),
                "rouge_1": _mean([r.get("rouge_1") for r in sub]),
                "rouge_l": _mean([r.get("rouge_l") for r in sub]),
                "exact_action": (sum(int(r["exact_action"]) for r in sub) / n) if n else None,
            }

    def _sub(a: str, b: str, key: str) -> float | None:
        va = (means_map.get(a) or {}).get(key)
        vb = (means_map.get(b) or {}).get(key)
        if va is None or vb is None:
            return None
        return float(va) - float(vb)

    deltas = {
        "rag_vs_norag": {
            "note": "B−A and C−A: RAG vs no RAG (system text vs gold relevant_rag_chunks concat)",
            "B_minus_A": {
                "bertscore_f1": _sub("RAG_No_Ranking", "LLM_only", "bertscore_f1"),
                "rouge_1": _sub("RAG_No_Ranking", "LLM_only", "rouge_1"),
                "rouge_l": _sub("RAG_No_Ranking", "LLM_only", "rouge_l"),
                "exact_action": _sub("RAG_No_Ranking", "LLM_only", "exact_action"),
            },
            "C_minus_A": {
                "bertscore_f1": _sub("RAG_RANKING", "LLM_only", "bertscore_f1"),
                "rouge_1": _sub("RAG_RANKING", "LLM_only", "rouge_1"),
                "rouge_l": _sub("RAG_RANKING", "LLM_only", "rouge_l"),
                "exact_action": _sub("RAG_RANKING", "LLM_only", "exact_action"),
            },
        },
        "rank_vs_norank": {
            "note": "C−B: ranking vs no ranking",
            "C_minus_B": {
                "bertscore_f1": _sub("RAG_RANKING", "RAG_No_Ranking", "bertscore_f1"),
                "rouge_1": _sub("RAG_RANKING", "RAG_No_Ranking", "rouge_1"),
                "rouge_l": _sub("RAG_RANKING", "RAG_No_Ranking", "rouge_l"),
                "retrieve_bertscore_f1": _sub("RAG_RANKING", "RAG_No_Ranking", "retrieve_bertscore_f1"),
                "recall@5": _sub("RAG_RANKING", "RAG_No_Ranking", "recall@5"),
                "mrr": _sub("RAG_RANKING", "RAG_No_Ranking", "mrr"),
                "exact_action": _sub("RAG_RANKING", "RAG_No_Ranking", "exact_action"),
            },
        },
    }
    payload = {
        "n": len(cases_out),
        "systems": SYSTEM_IDS,
        "means": means_map,
        "means_by_class": means_by_class,
        "deltas": deltas,
        "cases": cases_out,
    }
    (out / "mitigation_plans.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    csv_fields = list(csv_rows[0].keys()) if csv_rows else ["case_id"]
    with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        w.writerows(csv_rows)

    flatten_out_dir(out)
    for leftover in ("comparison.json", "README.md"):
        p = out / leftover
        if p.is_file():
            p.unlink()

    manifest = {
        "task": "pragma-rag-eval100",
        "date": date.today().isoformat(),
        "n": 90,
        "systems": SYSTEM_IDS,
        "gold": str(GOLD),
        "gold_sha256": meta["gold_hash"],
        "flows": str(FIXTURE_FLOWS),
        "flows_sha256": meta["flows_hash"],
        "detect_json": str(meta["pred_path"]),
        "index": str(VECTOR_STORE_DIR),
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "temperature": 0,
        "bertscore_reference": "relevant_rag_chunks[].text concat",
        "jsonl_counts": {s: len(by_sys.get(s) or []) for s in SYSTEM_IDS},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_charts(out, means_map, deltas)
    report = _render_report(means, manifest, deltas)
    (out / "report.md").write_text(report, encoding="utf-8")
    DOCS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOCS_REPORT.write_text(report, encoding="utf-8")
    print("scored →", out / "mitigation_plans.json")
    return out


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _delta_line(label: str, d: dict[str, Any] | None) -> str:
    d = d or {}
    parts = []
    for k in ("bertscore_f1", "rouge_1", "rouge_l", "exact_action", "recall@5", "mrr", "retrieve_bertscore_f1"):
        if k in d and d[k] is not None:
            parts.append(f"{k} {float(d[k]):+.3f}")
    return f"- {label}: " + (", ".join(parts) if parts else "not enough systems scored")


def _write_charts(out: Path, means_map: dict[str, Any], deltas: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    write_ac_text_chart(out, means_map, deltas)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    rag = deltas.get("rag_vs_norag") or {}
    ba = rag.get("B_minus_A") or {}
    ca = rag.get("C_minus_A") or {}
    keys = ["bertscore_f1", "rouge_1", "rouge_l", "exact_action"]
    axes[0].bar([k.replace("_", "\n") for k in keys], [float(ba.get(k) or 0) for k in keys], color="#3b82f6")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("RAG (no rank) − no RAG")
    axes[0].set_ylabel("Δ mean")
    cb = (deltas.get("rank_vs_norank") or {}).get("C_minus_B") or {}
    keys2 = ["bertscore_f1", "rouge_l", "retrieve_bertscore_f1", "recall@5", "mrr"]
    axes[1].bar([k.replace("_", "\n") for k in keys2], [float(cb.get(k) or 0) for k in keys2], color="#10b981")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Ranking − no ranking")
    fig.suptitle("Positive = better than the weaker step (vs gold)")
    fig.tight_layout()
    dest = out / "rag_vs_norag.png"
    try:
        fig.savefig(dest, dpi=140)
    except OSError:
        fig.savefig(out / "rag_vs_norag_deltas.png", dpi=140)
    plt.close(fig)


def _render_report(
    comparison: list[dict[str, Any]],
    manifest: dict[str, Any],
    deltas: dict[str, Any] | None = None,
) -> str:
    by = {c["system"]: c for c in comparison}

    def cell(sys_id: str, key: str) -> str:
        row = by.get(sys_id) or {}
        if not row.get("n"):
            return "NOT RUN"
        return _fmt(row.get(key))

    lines = [
        "# RAG vs LLM-only (reserved gold cores)",
        "",
        f"**Run:** `experiments/rag/rag_eval100/`",
        f"**Gold:** `experiments/gold-100/ground_truth-100.json` SHA-256 `{manifest.get('gold_sha256')}`",
        f"**Eval model:** `{manifest.get('model')}` (temperature {manifest.get('temperature')})",
        f"**Index:** `{manifest.get('index')}`  retrieve = FAISS+BM25 → RRF → MMR (C) → 5 parents",
        f"**Flows:** `experiments/data/gold-100/flows.csv` SHA-256 `{manifest.get('flows_sha256')}`",
        "",
        "## 1. What was compared",
        "",
        "| ID | System | Retrieval |",
        "|----|--------|-----------|",
        "| A | No RAG | no SHAP, no PDFs |",
        "| B | RAG only | hybrid RRF, no SHAP |",
        "| C | Ranked RAG + SHAP | hybrid RRF + MMR λ=0.5 + SHAP |",
        "",
        "Same $W[\\hat{y}]$, **90 reserved cases (10 per class)**. Retrieve path unchanged; only prompt flags differ.",
        "",
        "## 2. Gold",
        "",
        "N=90 held-out VFL test rows (seed 42). Histogram 10 × 9 classes from `ground_truth-100.json`.",
        "BERTScore / ROUGE reference = concatenation of `relevant_rag_chunks[].text`. IR labels = those `parent_id`s.",
        "",
        "## 3. Results",
        "",
        "Means below; one row per gold flow in `mitigation_plans.json` / `comparison.csv`.",
        "",
        "| Metric | No RAG | RAG only | Ranked RAG + SHAP |",
        "|--------|-------:|-------------:|-----------:|",
        f"| Exact action | {cell('LLM_only', 'exact_action')} | {cell('RAG_No_Ranking', 'exact_action')} | {cell('RAG_RANKING', 'exact_action')} |",
        f"| Acceptable action | {cell('LLM_only', 'acceptable_action')} | {cell('RAG_No_Ranking', 'acceptable_action')} | {cell('RAG_RANKING', 'acceptable_action')} |",
        f"| BERTScore F1 (vs gold chunk concat) | {cell('LLM_only', 'bertscore_f1')} | {cell('RAG_No_Ranking', 'bertscore_f1')} | {cell('RAG_RANKING', 'bertscore_f1')} |",
        f"| ROUGE-1 | {cell('LLM_only', 'rouge_1')} | {cell('RAG_No_Ranking', 'rouge_1')} | {cell('RAG_RANKING', 'rouge_1')} |",
        f"| ROUGE-L | {cell('LLM_only', 'rouge_l')} | {cell('RAG_No_Ranking', 'rouge_l')} | {cell('RAG_RANKING', 'rouge_l')} |",
        f"| BERTScore F1 (retrieve-only, no LLM) | — | {cell('RAG_No_Ranking', 'retrieve_bertscore_f1')} | {cell('RAG_RANKING', 'retrieve_bertscore_f1')} |",
        f"| ROUGE-L (retrieve-only) | — | {cell('RAG_No_Ranking', 'retrieve_rouge_l')} | {cell('RAG_RANKING', 'retrieve_rouge_l')} |",
        f"| Recall@5 | — | {cell('RAG_No_Ranking', 'recall@5')} | {cell('RAG_RANKING', 'recall@5')} |",
        f"| MRR | — | {cell('RAG_No_Ranking', 'mrr')} | {cell('RAG_RANKING', 'mrr')} |",
        "",
        "## 4. Why RAG context vs no RAG",
        "",
        "Gold reference = concat of gold `relevant_rag_chunks[].text`. No-RAG hyp = rationale only; RAG hyp = retrieved parents + rationale.",
        "Headline chart `bertscore_rouge.png`: No RAG vs Ranked RAG + SHAP (BERTScore F1 / ROUGE-1 / ROUGE-L, C−A). Extra deltas: `rag_vs_norag.png`.",
        "",
        _delta_line("RAG no-rank − no RAG (B−A)", ((deltas or {}).get("rag_vs_norag") or {}).get("B_minus_A")),
        _delta_line("RAG+rank − no RAG (C−A)", ((deltas or {}).get("rag_vs_norag") or {}).get("C_minus_A")),
        "",
        "A positive BERTScore / ROUGE delta means RAG system text is closer to the gold retrieved parents than no-RAG rationale.",
        "",
        "## 5. Why ranking vs no ranking",
        "",
        _delta_line("Rank − no rank (C−B)", ((deltas or {}).get("rank_vs_norank") or {}).get("C_minus_B")),
        "",
        "A positive retrieve-only BERTScore means the context itself (no LLM) is closer to gold after MMR + vector-score rank.",
        "If a delta is negative, ranking or RAG did not help on that metric — report the number.",
        "",
        "## 6. Not this report",
        "",
        "The 1000-row `pragma-e2e-detect-chain` set (later blockchain / e2e) is a different fixture.",
        "No McNemar / Wilcoxon. No Commit / Apply.",
        "",
    ]
    return "\n".join(lines)


def _stratified_quotas(n: int) -> dict[str, int]:
    """Even split across 9 labels; remainder goes to NINE in order (BENIGN, BOT, …)."""
    base = int(n) // len(NINE)
    rem = int(n) - base * len(NINE)
    quotas = {lab: base for lab in NINE}
    for i in range(rem):
        quotas[NINE[i]] += 1
    return quotas


def offgold_out_dir(n: int) -> Path:
    name = "hybrid_smoke10" if int(n) == 10 else f"hybrid_eval{int(n)}"
    path = experiment_dir("rag") / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_offgold_pairs(n: int = 10) -> list[dict[str, Any]]:
    """n flows from rag_reason_500 — disjoint from gold-100, stratified by true_label."""
    from scripts.rag_reason import _pair_rows

    paired, man = _pair_rows()
    gold = {int(x) for x in (man.get("excluded_gold_split_index") or [])}
    by: dict[str, list[dict[str, Any]]] = {c: [] for c in NINE}
    for item in paired:
        six = int(item["sample"]["split_index"])
        if six in gold:
            continue
        lab = str(item["sample"].get("true_label") or item["flow"].get("label_simplified") or "").upper()
        by.setdefault(lab, []).append(item)
    quotas = _stratified_quotas(n)
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for lab in NINE:
        pool = by.get(lab) or []
        take = min(int(quotas.get(lab) or 0), len(pool))
        for item in pool[:take]:
            used.add(int(item["sample"]["split_index"]))
            picked.append(item)
    if len(picked) < n:
        for lab in NINE:
            for item in by.get(lab) or []:
                six = int(item["sample"]["split_index"])
                if six in used:
                    continue
                used.add(six)
                picked.append(item)
                if len(picked) >= n:
                    break
            if len(picked) >= n:
                break
    rows: list[dict[str, Any]] = []
    for item in picked[:n]:
        s = item["sample"]
        six = int(s["split_index"])
        true = str(s.get("true_label") or item["flow"].get("label_simplified") or "").upper()
        rows.append(
            {
                "case": {
                    "case_id": f"OG-{six}",
                    "split_index": six,
                    "condition": {"true_label": true, "primary_network_tier": ""},
                    "actions": {},
                    "relevant_rag_chunks": [],
                },
                "sample": s,
            }
        )
    overlap = {int(r["case"]["split_index"]) for r in rows} & gold
    if overlap:
        raise SystemExit(f"off-gold set overlaps gold: {sorted(overlap)}")
    if len(rows) != n:
        raise SystemExit(f"off-gold n={len(rows)} expected {n}")
    return rows


def load_offgold_pairs_from_jsonl(out: Path) -> list[dict[str, Any]]:
    """Rebuild the 200-flow list from traces when detect JSON is not on disk."""
    recs = _read_jsonl(_jsonl_path(out, "LLM_only")) or _read_jsonl(_jsonl_path(out, "RAG_RANKING"))
    if not recs:
        raise SystemExit(f"no off-gold jsonl in {out}")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rec in recs:
        cid = str(rec.get("case_id") or "")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        true = str(rec.get("true_label") or "").upper()
        rows.append(
            {
                "case": {
                    "case_id": cid,
                    "split_index": int(rec.get("split_index") or 0),
                    "condition": {"true_label": true, "primary_network_tier": ""},
                    "actions": {},
                    "relevant_rag_chunks": [],
                },
                "sample": {
                    "split_index": int(rec.get("split_index") or 0),
                    "true_label": true,
                    "predicted_label": rec.get("predicted_label"),
                    "confidence": rec.get("confidence"),
                },
            }
        )
    return rows


def _offgold_system_block(rec: dict[str, Any] | None, w_true: set[str]) -> dict[str, Any] | None:
    if not rec:
        return None
    plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
    prim, supp, all_a = _plan_action_lists(plan)
    first, tier = _first_action(plan)
    return {
        "plan": plan,
        "all_actions": all_a,
        "primary_actions": prim,
        "supporting_actions": supp,
        "action": first,
        "tier": tier,
        "overall_reasoning": (plan or {}).get("overall_reasoning") or rec.get("overall_reasoning") or "",
        "in_w_true": bool(first and first in w_true),
        "all_in_w_true": bool(all_a) and all(a in w_true for a in all_a),
        "format_error": bool(rec.get("format_error")),
        "prompt_parent_ids": [str(x) for x in (rec.get("prompt_parent_ids") or []) if x],
        "retrieved_parent_ids": [str(x) for x in (rec.get("retrieved_parent_ids") or []) if x],
        "hybrid": rec.get("hybrid"),
    }


def class_chunk_refs() -> dict[str, str]:
    """One gold parent-concat per attack type (first gold-100 case of that label)."""
    if not GOLD.is_file():
        return {}
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    refs: dict[str, str] = {}
    for case in data.get("cases") or []:
        if not isinstance(case, dict):
            continue
        lab = str((case.get("condition") or {}).get("true_label") or "").upper()
        if not lab or lab in refs:
            continue
        refs[lab] = _gold_chunk_concat(case)
    return refs


def score_offgold_text(out: Path, paired: list[dict[str, Any]]) -> dict[str, Any]:
    """BERTScore + ROUGE vs class gold chunk concat. Headline contrast: No RAG vs Ranked RAG+SHAP."""
    refs = class_chunk_refs()
    parents = load_parent_store(VECTOR_STORE_DIR)
    by_sys: dict[str, dict[str, dict[str, Any]]] = {}
    for sid in SYSTEM_IDS:
        by_sys[sid] = {str(r.get("case_id")): r for r in _read_jsonl(_jsonl_path(out, sid))}

    rows: list[dict[str, Any]] = []
    for spec in SYSTEMS:
        for item in paired:
            cid = str(item["case"]["case_id"])
            true = str(item["case"]["condition"]["true_label"]).upper()
            rec = by_sys.get(spec["id"], {}).get(cid)
            if not rec:
                continue
            plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
            hyp = _llm_rationale(plan)
            prompt_txt = _prompt_parent_concat(rec, parents) if spec["rag"] else ""
            system_text = _system_score_text(hyp, prompt_txt, spec["rag"])
            gold_ref = refs.get(true) or " "
            rows.append(
                {
                    "case_id": cid,
                    "true_label": true,
                    "system": spec["id"],
                    "rag": spec["rag"],
                    "gold_ref": gold_ref,
                    "system_text": system_text,
                    "prompt_txt": prompt_txt,
                }
            )

    for row in rows:
        r1, rl = _rouge_scores(_clip_score_text(row["gold_ref"]), _clip_score_text(row["system_text"]))
        row["rouge_1"] = r1
        row["rouge_l"] = rl
        if row["rag"] and row["prompt_txt"]:
            rr1, rrl = _rouge_scores(_clip_score_text(row["gold_ref"]), _clip_score_text(row["prompt_txt"]))
            row["retrieve_rouge_1"] = rr1
            row["retrieve_rouge_l"] = rrl
        else:
            row["retrieve_rouge_1"] = None
            row["retrieve_rouge_l"] = None

    if rows:
        f1s = _bertscore_f1(
            [_clip_score_text(r["gold_ref"]) for r in rows],
            [_clip_score_text(r["system_text"]) for r in rows],
        )
        for row, f1 in zip(rows, f1s):
            row["bertscore_f1"] = f1

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_key[(str(row["case_id"]), str(row["system"]))] = {
            "bertscore_f1": row.get("bertscore_f1"),
            "rouge_1": row.get("rouge_1"),
            "rouge_l": row.get("rouge_l"),
            "retrieve_bertscore_f1": row.get("retrieve_bertscore_f1"),
            "retrieve_rouge_1": row.get("retrieve_rouge_1"),
            "retrieve_rouge_l": row.get("retrieve_rouge_l"),
        }

    means: dict[str, dict[str, Any]] = {}
    for sid in SYSTEM_IDS:
        sub = [r for r in rows if r["system"] == sid]
        means[sid] = {
            "n": len(sub),
            "bertscore_f1": _mean([r.get("bertscore_f1") for r in sub]),
            "rouge_1": _mean([r.get("rouge_1") for r in sub]),
            "rouge_l": _mean([r.get("rouge_l") for r in sub]),
            "retrieve_bertscore_f1": _mean([r.get("retrieve_bertscore_f1") for r in sub]),
            "retrieve_rouge_1": _mean([r.get("retrieve_rouge_1") for r in sub]),
            "retrieve_rouge_l": _mean([r.get("retrieve_rouge_l") for r in sub]),
        }

    def _sub(a: str, b: str, key: str) -> float | None:
        va = (means.get(a) or {}).get(key)
        vb = (means.get(b) or {}).get(key)
        if va is None or vb is None:
            return None
        return float(va) - float(vb)

    deltas = {
        "C_minus_A": {
            "bertscore_f1": _sub("RAG_RANKING", "LLM_only", "bertscore_f1"),
            "rouge_1": _sub("RAG_RANKING", "LLM_only", "rouge_1"),
            "rouge_l": _sub("RAG_RANKING", "LLM_only", "rouge_l"),
        }
    }
    csv_rows: list[dict[str, Any]] = []
    for item in paired:
        cid = str(item["case"]["case_id"])
        row = {
            "case_id": cid,
            "true_label": item["case"]["condition"]["true_label"],
            "predicted_label": item["sample"].get("predicted_label"),
        }
        for sid, prefix in (("LLM_only", "A"), ("RAG_No_Ranking", "B"), ("RAG_RANKING", "C")):
            sc = by_key.get((cid, sid)) or {}
            row[f"{prefix}_bertscore"] = sc.get("bertscore_f1")
            row[f"{prefix}_rouge1"] = sc.get("rouge_1")
            row[f"{prefix}_rougeL"] = sc.get("rouge_l")
        csv_rows.append(row)
    if csv_rows:
        with (out / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    return {"by_key": by_key, "means": means, "deltas": deltas, "rows": rows, "refs": refs}


def write_ac_text_chart(out: Path, means: dict[str, Any], deltas: dict[str, Any]) -> Path:
    """Headline figure: BERTScore / ROUGE-1 / ROUGE-L for No RAG (A) vs RAG + SHAP (B)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    keys = ("bertscore_f1", "rouge_1", "rouge_l")
    names = ("BERTScore F1", "ROUGE-1", "ROUGE-L")
    a_vals = [float((means.get("LLM_only") or {}).get(k) or 0) for k in keys]
    c_vals = [float((means.get("RAG_RANKING") or {}).get(k) or 0) for k in keys]
    d = deltas.get("C_minus_A") or {}
    if not d and isinstance((deltas.get("rag_vs_norag") or {}).get("C_minus_A"), dict):
        d = deltas["rag_vs_norag"]["C_minus_A"]
    d_vals = [float(d.get(k) or (c_vals[i] - a_vals[i])) for i, k in enumerate(keys)]
    x = list(range(len(names)))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    b1 = ax.bar([i - w / 2 for i in x], a_vals, w, label="No RAG", color="#94a3b8")
    b2 = ax.bar([i + w / 2 for i in x], c_vals, w, label="RAG + SHAP", color="#2563eb")
    ax.set_xticks(x)
    ax.set_xticklabels(list(names))
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("No RAG vs RAG + SHAP")
    ax.legend(frameon=False)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for i, dv in enumerate(d_vals):
        ax.text(i, max(a_vals[i], c_vals[i]) + 0.09, f"B-A {dv:+.3f}", ha="center", va="bottom", fontsize=8, color="#1e3a8a")
    fig.tight_layout()
    path = out / "bertscore_rouge.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("chart ->", path)
    return path


def write_offgold_text_chart(out: Path, means: dict[str, Any], deltas: dict[str, Any]) -> Path:
    return write_ac_text_chart(out, means, deltas)


def write_offgold_3d_charts(out: Path) -> list[Path]:
    """3D headline + per-class bars from comparison.csv (A vs C columns)."""
    csv_path = out / "comparison.csv"
    if not csv_path.is_file():
        return []
    import csv as _csv
    from collections import defaultdict

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    short = {
        "BENIGN": "BENIGN",
        "BOT": "BOT",
        "DDOS": "DDoS",
        "DOS": "DoS",
        "FTPPATATOR": "FTP",
        "OTHERS": "OTHERS",
        "PORTSCAN": "PORT",
        "SSHPATATOR": "SSH",
        "WEBATTACK": "WEB",
    }
    keys = (
        ("A_bertscore", "C_bertscore", "BERTScore F1"),
        ("A_rouge1", "C_rouge1", "ROUGE-1"),
        ("A_rougeL", "C_rougeL", "ROUGE-L"),
    )
    no_rag, rag = "#94a3b8", "#2563eb"
    overall: dict[str, list[float]] = defaultdict(list)
    by_cls: dict[str, dict[str, list[float]]] = {c: defaultdict(list) for c in NINE}
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            lab = str(row.get("true_label") or "").upper()
            for ak, ck, _name in keys:
                try:
                    a, c = float(row[ak]), float(row[ck])
                except (KeyError, TypeError, ValueError):
                    continue
                overall[ak].append(a)
                overall[ck].append(c)
                if lab in by_cls:
                    by_cls[lab][ak].append(a)
                    by_cls[lab][ck].append(c)

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    def _pane(ax: Any) -> None:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor("#d1d5db")
        ax.grid(True, color="#e5e7eb", linewidth=0.5)
        ax.tick_params(labelsize=8)
        ax.set_zlabel("Score", fontsize=9, labelpad=6)

    a_vals = [_mean(overall[k[0]]) for k in keys]
    c_vals = [_mean(overall[k[1]]) for k in keys]
    names = [k[2] for k in keys]
    fig = plt.figure(figsize=(8.8, 6.6), facecolor="white")
    fig.text(0.50, 0.97, "No RAG vs RAG + SHAP", ha="center", va="top", fontsize=12)
    ax = fig.add_axes([0.04, 0.28, 0.90, 0.62], projection="3d")
    xs = np.arange(len(names), dtype=float)
    dx, dy = 0.50, 0.34
    for yi, (vals, color) in enumerate(((a_vals, no_rag), (c_vals, rag))):
        ax.bar3d(
            xs - dx / 2,
            np.full(len(xs), float(yi)) - dy / 2,
            np.zeros(len(xs)),
            dx,
            dy,
            vals,
            color=color,
            shade=True,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.96,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["No RAG", "RAG + SHAP"], fontsize=8)
    ax.set_zlim(0, 1.0)
    _pane(ax)
    ax.view_init(elev=21, azim=-56)
    try:
        ax.set_box_aspect((1.85, 0.72, 1.05))
    except Exception:
        pass
    ax.legend(
        handles=[Patch(facecolor=no_rag, label="No RAG"), Patch(facecolor=rag, label="RAG + SHAP")],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    tax = fig.add_axes([0.12, 0.04, 0.76, 0.20])
    tax.axis("off")
    cell = [[f"{a_vals[i]:.3f}", f"{c_vals[i]:.3f}", f"{c_vals[i] - a_vals[i]:+.3f}"] for i in range(3)]
    tbl = tax.table(
        cellText=cell,
        rowLabels=names,
        colLabels=["No RAG (A)", "RAG + SHAP (B)", "B − A"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.45)
    for (r, col), cell_obj in tbl.get_celld().items():
        cell_obj.set_edgecolor("#d1d5db")
        if r == 0:
            cell_obj.set_facecolor("#f1f5f9")
            cell_obj.set_text_props(weight="bold")
        elif col == 2:
            cell_obj.set_text_props(color="#1e3a8a", weight="bold")
    headline = out / "bertscore_rouge_3d.png"
    fig.savefig(headline, dpi=170, facecolor="white")
    plt.close(fig)
    print("chart ->", headline)

    fig = plt.figure(figsize=(15.8, 5.8), facecolor="white")
    zmaxes = (1.0, 0.50, 0.28)
    labels = [short[c] for c in NINE]
    xs = np.arange(len(NINE), dtype=float)
    dx, dy = 0.58, 0.32
    for i, ((ak, ck, title), zmax) in enumerate(zip(keys, zmaxes)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        av = [_mean(by_cls[lab][ak]) for lab in NINE]
        cv = [_mean(by_cls[lab][ck]) for lab in NINE]
        for yi, (vals, color) in enumerate(((av, no_rag), (cv, rag))):
            ax.bar3d(
                xs - dx / 2,
                np.full(len(xs), float(yi)) - dy / 2,
                np.zeros(len(xs)),
                dx,
                dy,
                vals,
                color=color,
                shade=True,
                edgecolor="white",
                linewidth=0.3,
                alpha=0.96,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=6.5)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["No RAG", "RAG+SHAP"], fontsize=7)
        ax.set_zlim(0, zmax)
        ax.set_title(title, fontsize=11, pad=2)
        _pane(ax)
        ax.view_init(elev=23, azim=-58)
        try:
            ax.set_box_aspect((1.9, 0.7, 1.05))
        except Exception:
            pass
    fig.legend(
        handles=[Patch(facecolor=no_rag, label="No RAG"), Patch(facecolor=rag, label="RAG + SHAP")],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle("Text overlap by attack type  ·  n = 200 off-gold", fontsize=12, y=1.04)
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.86, wspace=0.12)
    by_class = out / "bertscore_rouge_by_class_3d.png"
    fig.savefig(by_class, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("chart ->", by_class)
    return [headline, by_class]


def write_pipeline_png(out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams["font.family"] = "DejaVu Sans"
    boxes = [
        (0.04, 0.55, 0.14, 0.22, "Query"),
        (0.22, 0.72, 0.16, 0.18, "FAISS N=80"),
        (0.22, 0.38, 0.16, 0.18, "BM25 N=80"),
        (0.42, 0.55, 0.12, 0.22, "RRF"),
        (0.57, 0.55, 0.14, 0.22, "MMR λ=0.5"),
        (0.74, 0.55, 0.10, 0.22, "20 kids"),
        (0.87, 0.55, 0.11, 0.22, "5 parents"),
    ]
    fig, ax = plt.subplots(figsize=(10.4, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B RAG + SHAP retrieve")
    for x, y, w, h, label in boxes:
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", facecolor="#dbeafe", edgecolor="#1e3a8a", linewidth=1.2)
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    arrows = [
        ((0.18, 0.66), (0.22, 0.81)),
        ((0.18, 0.66), (0.22, 0.47)),
        ((0.38, 0.81), (0.42, 0.70)),
        ((0.38, 0.47), (0.42, 0.62)),
        ((0.54, 0.66), (0.57, 0.66)),
        ((0.71, 0.66), (0.74, 0.66)),
        ((0.84, 0.66), (0.87, 0.66)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": "#1e3a8a", "lw": 1.1})
    ax.text(0.50, 0.12, "No RAG skips retrieve. C only: same child ids → RRF → MMR → LLM + SHAP.", ha="center", fontsize=8)
    fig.tight_layout()
    path = out / "pipeline.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("pipeline ->", path)
    return path


def _offgold_systems_present(out: Path, paired: list[dict[str, Any]]) -> list[str]:
    present = []
    for sid in SYSTEM_IDS:
        recs = {str(r.get("case_id")): r for r in _read_jsonl(_jsonl_path(out, sid))}
        if any(str(item["case"]["case_id"]) in recs for item in paired):
            present.append(sid)
    return present or ["LLM_only", "RAG_RANKING"]


def _proposed_action_names(rec: dict[str, Any] | None) -> list[str]:
    if not rec:
        return []
    plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
    prim, supp, _all_a = _plan_action_lists(plan)
    seen: list[str] = []
    for it in prim + supp:
        a = it["action"]
        if a and a not in seen:
            seen.append(a)
    return seen


def _action_table_lines(
    paired: list[dict[str, Any]],
    by_sys: dict[str, dict[str, dict[str, Any]]],
    systems: list[str],
    catalog: dict[str, Any],
) -> list[str]:
    labels = {"LLM_only": "No RAG", "RAG_No_Ranking": "RAG", "RAG_RANKING": "RAG + SHAP"}
    heads = [labels.get(s, s) for s in systems]
    lines = [
        "## Proposed actions by attack type",
        "",
        "An action is proposed if it appears in `primary_actions` or `supporting_actions`. Cell = k/n (p%). Underlined = higher share (> 0).",
        "",
    ]
    by_lab: dict[str, list[dict[str, Any]]] = {c: [] for c in NINE}
    for item in paired:
        lab = str(item["case"]["condition"]["true_label"]).upper()
        by_lab.setdefault(lab, []).append(item)
    for lab in NINE:
        items = by_lab.get(lab) or []
        n = len(items)
        if not n:
            continue
        w = [_norm_action(a) for a in catalog.get(lab, ())]
        counts: dict[str, dict[str, int]] = {a: {sid: 0 for sid in systems} for a in w}
        extra: dict[str, dict[str, int]] = {}
        for item in items:
            cid = str(item["case"]["case_id"])
            for sid in systems:
                for act in _proposed_action_names(by_sys.get(sid, {}).get(cid)):
                    if act in counts:
                        counts[act][sid] += 1
                    else:
                        extra.setdefault(act, {s: 0 for s in systems})
                        extra[act][sid] += 1
        lines += [f"### {lab}  n={n}  W[y] = {', '.join(catalog.get(lab) or ()) or '—'}", ""]
        lines.append("| Action | " + " | ".join(heads) + " |")
        lines.append("|--------|" + "|".join([":------:" for _ in systems]) + "|")

        def row(name: str, per: dict[str, int], *, off: bool) -> None:
            rates = {sid: (per[sid] / n) for sid in systems}
            best = max(rates.values()) if rates else 0.0
            cells = []
            for sid in systems:
                k = per[sid]
                text = f"{k}/{n} ({100.0 * k / n:.1f}%)"
                if best > 0 and rates[sid] == best:
                    text = f"<u>{text}</u>"
                cells.append(text)
            mark = " *(off-catalog)*" if off else ""
            lines.append(f"| {name}{mark} | " + " | ".join(cells) + " |")

        for act in w:
            row(act, counts[act], off=False)
        for act in sorted(extra):
            row(act, extra[act], off=True)
        lines.append("")
    return lines


def write_offgold_plans(
    out: Path,
    paired: list[dict[str, Any]],
    *,
    text_scores: dict[str, Any] | None = None,
) -> Path:
    """Same case × A/B/C plan shape as rag_eval100/mitigation_plans.json (no gold scores)."""
    catalog = load_attack_actions_by_type()
    by_sys: dict[str, dict[str, dict[str, Any]]] = {}
    for sid in SYSTEM_IDS:
        by_sys[sid] = {str(r.get("case_id")): r for r in _read_jsonl(_jsonl_path(out, sid))}
    by_key = (text_scores or {}).get("by_key") or {}
    cases_out: list[dict[str, Any]] = []
    legal = {sid: 0 for sid in SYSTEM_IDS}
    legal_all = {sid: 0 for sid in SYSTEM_IDS}
    have = {sid: 0 for sid in SYSTEM_IDS}
    for item in paired:
        case = item["case"]
        sample = item["sample"]
        cid = str(case["case_id"])
        true = str(case["condition"]["true_label"]).upper()
        w_true = {_norm_action(a) for a in catalog.get(true, ())}
        entry: dict[str, Any] = {
            "case_id": cid,
            "split_index": int(case["split_index"]),
            "true_label": true,
            "predicted_label": sample.get("predicted_label"),
            "confidence": sample.get("confidence"),
            "gold": None,
            "w_true": sorted(w_true),
        }
        for sid in SYSTEM_IDS:
            block = _offgold_system_block(by_sys.get(sid, {}).get(cid), w_true)
            sc = by_key.get((cid, sid)) or {}
            if block:
                block.update(sc)
                have[sid] += 1
                if block["in_w_true"]:
                    legal[sid] += 1
                if block["all_in_w_true"]:
                    legal_all[sid] += 1
            entry[sid] = block
        cases_out.append(entry)
    n = len(cases_out)
    text_means = (text_scores or {}).get("means") or {}
    payload = {
        "n": n,
        "outside_gold": True,
        "systems": SYSTEM_IDS,
        "note": "Full A/B/C Mitigation Plans. BERTScore/ROUGE vs class gold chunk concat (not per-flow gold).",
        "bertscore_reference": "gold-100 relevant_rag_chunks[].text concat, one per true_label",
        "means": {
            sid: {
                "n": have[sid],
                "primary_in_w_true": (legal[sid] / have[sid]) if have[sid] else None,
                "all_actions_in_w_true": (legal_all[sid] / have[sid]) if have[sid] else None,
                **(text_means.get(sid) or {}),
            }
            for sid in SYSTEM_IDS
        },
        "deltas": (text_scores or {}).get("deltas") or {},
        "cases": cases_out,
    }
    path = out / "mitigation_plans.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("off-gold plans ->", path)
    return path


def write_offgold_report(out: Path, paired: list[dict[str, Any]]) -> Path:
    catalog = load_attack_actions_by_type()
    by_sys: dict[str, dict[str, dict[str, Any]]] = {}
    for sid in SYSTEM_IDS:
        by_sys[sid] = {str(r.get("case_id")): r for r in _read_jsonl(_jsonl_path(out, sid))}
    text_scores = score_offgold_text(out, paired)
    write_offgold_plans(out, paired, text_scores=text_scores)
    write_offgold_text_chart(out, text_scores.get("means") or {}, text_scores.get("deltas") or {})
    write_offgold_3d_charts(out)
    write_pipeline_png(out)
    present = _offgold_systems_present(out, paired)

    def fmt_actions(rec: dict[str, Any] | None, w_true: set[str]) -> str:
        if not rec:
            return "—"
        plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
        prim, supp, _all_a = _plan_action_lists(plan)
        seen: set[str] = set()
        parts: list[str] = []
        for it in prim:
            if it["action"] not in seen:
                seen.add(it["action"])
                parts.append(it["action"])
        extra: list[str] = []
        for it in supp:
            if it["action"] not in seen:
                seen.add(it["action"])
                extra.append(it["action"])
        text = ", ".join(parts) if parts else "—"
        if extra:
            text += " (+ " + ", ".join(extra) + ")"
        first = parts[0] if parts else ""
        mark = " ✓" if first and first in w_true else (" ✗" if first else "")
        return text + mark

    compare = [s for s in ("LLM_only", "RAG_RANKING") if s in present] or ["LLM_only", "RAG_RANKING"]
    n = len(paired)
    hist = Counter(str(p["case"]["condition"]["true_label"]).upper() for p in paired)
    n_ok = {sid: 0 for sid in compare}
    n_have = {sid: 0 for sid in compare}
    for item in paired:
        true = str(item["case"]["condition"]["true_label"]).upper()
        w_true = {_norm_action(a) for a in catalog.get(true, ())}
        cid = str(item["case"]["case_id"])
        for sid in compare:
            rec = by_sys.get(sid, {}).get(cid)
            plan = rec.get("plan") if rec and isinstance(rec.get("plan"), dict) else None
            first, _tier = _first_action(plan)
            if rec:
                n_have[sid] += 1
                if first and first in w_true:
                    n_ok[sid] += 1
    tm = text_scores.get("means") or {}
    td = (text_scores.get("deltas") or {}).get("C_minus_A") or {}
    lines = [
        f"# Off-gold hybrid eval (n={n}, not gold-100)",
        "",
        "FAISS N=80 + BM25 N=80 (same child ids) → RRF → MMR λ=0.5 → 20 children → 5 parents → LLM.",
        "Systems: **A No RAG (LLM)** vs **B RAG + SHAP**. Full plans in `mitigation_plans.json`. Pipeline: `pipeline.png`.",
        "",
        "Class mix: " + ", ".join(f"{lab} {hist.get(lab, 0)}" for lab in NINE),
        "",
        f"Primary ∈ W[true]: No RAG {n_ok.get('LLM_only', 0)}/{n_have.get('LLM_only') or n} · "
        f"RAG + SHAP {n_ok.get('RAG_RANKING', 0)}/{n_have.get('RAG_RANKING') or n}",
        "",
        "## BERTScore / ROUGE — No RAG vs RAG + SHAP",
        "",
        "Reference = gold-100 `relevant_rag_chunks` concat **per attack type**.",
        "No-RAG hyp = rationale only. Ranked RAG+SHAP hyp = 5 prompt parents + rationale.",
        "",
        "| Metric | No RAG (A) | RAG + SHAP (B) | B − A |",
        "|--------|-------:|------------------:|------:|",
        f"| BERTScore F1 | {_fmt((tm.get('LLM_only') or {}).get('bertscore_f1'))} | {_fmt((tm.get('RAG_RANKING') or {}).get('bertscore_f1'))} | {_fmt(td.get('bertscore_f1'))} |",
        f"| ROUGE-1 | {_fmt((tm.get('LLM_only') or {}).get('rouge_1'))} | {_fmt((tm.get('RAG_RANKING') or {}).get('rouge_1'))} | {_fmt(td.get('rouge_1'))} |",
        f"| ROUGE-L | {_fmt((tm.get('LLM_only') or {}).get('rouge_l'))} | {_fmt((tm.get('RAG_RANKING') or {}).get('rouge_l'))} | {_fmt(td.get('rouge_l'))} |",
        "",
        "Charts: `bertscore_rouge.png` (2D), `bertscore_rouge_3d.png` (3D + table), `bertscore_rouge_by_class_3d.png` (per attack type).",
        "",
    ]
    lines += _action_table_lines(paired, by_sys, compare, catalog)
    if n <= 12:
        lines += [
            "## Per-flow plans",
            "",
            "| case | true | pred | No RAG | RAG + SHAP |",
            "|------|------|------|--------|-------------------|",
        ]
        for item in paired:
            case = item["case"]
            cid = str(case["case_id"])
            true = str(case["condition"]["true_label"]).upper()
            pred = str(item["sample"].get("predicted_label") or "").upper()
            w_true = {_norm_action(a) for a in catalog.get(true, ())}
            cells = [fmt_actions(by_sys.get(sid, {}).get(cid), w_true) for sid in compare]
            lines.append(f"| {cid} | {true} | {pred} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "These flows are **not** gold-100 quality scores (no human gold actions).",
        "",
    ]
    report = out / "report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "task": "hybrid-offgold-ac",
                "n": n,
                "outside_gold": True,
                "class_counts": {lab: hist.get(lab, 0) for lab in NINE},
                "split_index": [int(p["case"]["split_index"]) for p in paired],
                "pipeline": "faiss80+bm25 80 → RRF → MMR(C) → 20 children → 5 parents",
                "systems": compare,
                "mitigation_plans": "mitigation_plans.json",
                "bertscore_reference": "gold-100 class chunk concat",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("off-gold report ->", report)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="RAG retrieval scoring (BERTScore / ROUGE). Gold-core → experiments/rag/rag_eval100/."
    )
    p.add_argument("--systems", default=",".join(SYSTEM_IDS), help="Comma-separated system ids")
    p.add_argument("--limit", type=int, default=0, help="Optional gold case cap (smoke). 0 = all 90.")
    p.add_argument("--score-only", action="store_true")
    p.add_argument(
        "--offgold-n",
        type=int,
        default=0,
        help="Run A/B/C on N flows from rag_reason_500 (outside gold-100). Does not write gold-100 traces.",
    )
    p.add_argument(
        "--offgold-report-only",
        action="store_true",
        help="Rebuild hybrid_smoke10 mitigation_plans.json + report.md from existing jsonl (no LLM).",
    )
    args = p.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    for s in systems:
        if s not in SYSTEM_IDS:
            raise SystemExit(f"unknown system {s}")

    if args.offgold_n or args.offgold_report_only:
        n = int(args.offgold_n) if args.offgold_n else 10
        out = offgold_out_dir(n)
        if args.offgold_report_only:
            jsonl_n = len(_read_jsonl(_jsonl_path(out, "LLM_only")) or _read_jsonl(_jsonl_path(out, "RAG_RANKING")))
            if jsonl_n:
                n = jsonl_n
                out = offgold_out_dir(n)
                paired = load_offgold_pairs_from_jsonl(out)
            else:
                paired = load_offgold_pairs(n)
        else:
            paired = load_offgold_pairs(n)
        print(f"off-gold n={len(paired)} -> {out}")
        print("class mix", dict(Counter(str(p["case"]["condition"]["true_label"]).upper() for p in paired)))
        if not args.offgold_report_only:
            run_systems(paired, systems, None, out=out)
        write_offgold_report(out, paired)
        return 0

    paired, meta = load_paired_cases()
    print("gold", GOLD, "sha256", meta["gold_hash"][:16], "n", len(paired))
    if not args.score_only:
        run_systems(paired, systems, args.limit or None)
    score_run(paired, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
