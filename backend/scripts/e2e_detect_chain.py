"""Happy-path e2e: detect → ranked RAG → one Mitigation Plan → unmodified chain.

Reads a frozen flows.csv. Default input: ``experiments/data/e2e-detect-chain/``.
Writes only ``experiments/e2e-detect-chain/``. Size comes from the CSV, not this filename.

Does not import ``agentic_attack_eval.py``. Does not inject or score BERTScore.

Usage (from backend/)::

    python scripts/e2e_detect_chain.py
    python scripts/e2e_detect_chain.py --input ../experiments/data/eval-10
    python scripts/e2e_detect_chain.py --predict-only
    python scripts/e2e_detect_chain.py --reason-only
    python scripts/e2e_detect_chain.py --chain-only
    python scripts/e2e_detect_chain.py --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from app.core.config import get_settings  # noqa: E402
from app.services.trust_chain_service import (  # noqa: E402
    apply_action_on_chain,
    build_plan_input,
    read_plan_from_chain,
    reasoning_hash_sha256,
    revert_reason,
    store_plan_on_chain,
)
from scripts.env import (  # noqa: E402
    FIXTURE_E2E_DETECT_CHAIN,
    experiment_dir,
    fixture_set_dir,
    load_project_dotenv,
)
from scripts.gold_sample_100 import NINE  # noqa: E402
from scripts.llm_prompt import create_agentic_orchestration_prompt  # noqa: E402
from scripts.rag_bridge import PROMPT_BODY_CHARS, PROMPT_SECTIONS, compact_sections  # noqa: E402
from scripts.rag_retrieval_scoring import _llm_call, _ms  # noqa: E402
from scripts.rag_hybrid import (  # noqa: E402
    BM25_N,
    DENSE_N,
    FINAL_CHILDREN,
    FINAL_PARENTS,
    MMR_LAMBDA,
    bm25_search,
    dense_search,
    rrf_fuse,
)
from scripts.reason import (  # noqa: E402
    VECTOR_STORE_DIR,
    build_template_rag_query,
    expand_parent_sections,
    mmr_select,
)
from scripts.rag_io import load_attack_and_agentic, load_parent_store, load_vector_store  # noqa: E402
from scripts import reason as reason_mod  # noqa: E402

load_project_dotenv()

SYSTEM = "RAG_RANKING"
CLOCKS = ("detect", "retrieve", "rank", "llm", "commit", "verify", "apply")
STEPS = CLOCKS + ("e2e",)
AGENT_REPORT = _REPO / ".agent" / "pragma-e2e-detect-chain" / "REPORT.md"


def _out() -> Path:
    return experiment_dir("e2e-detect-chain", mkdir=True)


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_input(raw: str) -> Path:
    text = (raw or "").strip()
    path = Path(text) if text else fixture_set_dir(FIXTURE_E2E_DETECT_CHAIN, mkdir=False)
    if not path.is_absolute():
        path = (_BACKEND / path).resolve() if not path.exists() else path.resolve()
        if not path.exists():
            path = (_REPO / text).resolve()
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    csv_path = path / "flows.csv"
    if csv_path.is_file():
        return csv_path
    raise SystemExit(f"Input flows.csv not found: {path}")


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _chunk_pred_file(chunk_dir: Path) -> Path | None:
    files = sorted(chunk_dir.glob("predictions_detailed_*.json"))
    return files[-1] if files else None


def run_detect(flows: Path, *, chunk_size: int = 100) -> Path:
    import pandas as pd

    live = _out()
    detect_root = live / "detect"
    detect_root.mkdir(parents=True, exist_ok=True)
    merged = detect_root / "predictions_detailed.json"
    df = pd.read_csv(flows)
    n = len(df)
    if merged.is_file():
        existing = json.loads(merged.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) == n:
            print(f"Detect already complete: {merged} ({n} rows)")
            return merged

    chunk_root = detect_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    parts: list[list[dict[str, Any]]] = []
    detect_times: list[dict[str, Any]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        name = f"chunk_{start:04d}_{end:04d}"
        chunk_dir = chunk_root / name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        done = _chunk_pred_file(chunk_dir)
        if done is not None:
            print(f"  reuse {done}")
            chunk_preds = json.loads(done.read_text(encoding="utf-8"))
            wall_ms = 0.0
        else:
            chunk_csv = chunk_dir / "flows.csv"
            df.iloc[start:end].to_csv(chunk_csv, index=False)
            os.environ["CHAINAGENT_SAMPLE_CSV"] = str(chunk_csv)
            os.environ["CHAINAGENT_PREDICT_OUT"] = str(chunk_dir)
            os.environ["PYTHONIOENCODING"] = "utf-8"
            _force_utf8_stdio()
            print(f"  detect {name} ({end - start} rows) ...", flush=True)
            t0 = time.perf_counter()
            runpy.run_path(str(_BACKEND / "scripts" / "detect_predict.py"), run_name="__main__")
            wall_ms = _ms(t0)
            done = _chunk_pred_file(chunk_dir)
            if done is None:
                raise SystemExit(f"detect produced no predictions_detailed_*.json in {chunk_dir}")
            chunk_preds = json.loads(done.read_text(encoding="utf-8"))
        if not isinstance(chunk_preds, list) or len(chunk_preds) != (end - start):
            raise SystemExit(f"{name}: expected {end - start} preds, got {len(chunk_preds)}")
        per = round(wall_ms / max(end - start, 1), 1) if wall_ms else None
        for i, rec in enumerate(chunk_preds):
            rec["sample_id"] = start + i
            rec["split_index"] = int(df.iloc[start + i]["split_index"])
            rec["reason_row"] = int(df.iloc[start + i]["reason_row"])
            rec["detect_ms"] = per
        detect_times.append({"chunk": name, "n": end - start, "wall_ms": wall_ms, "per_flow_ms": per})
        parts.append(chunk_preds)

    merged_rows = [row for part in parts for row in part]
    _dump(merged, merged_rows)
    _dump(detect_root / "detect_latency.json", detect_times)
    print(f"Merged detect -> {merged} ({len(merged_rows)} rows)")
    return merged


def _pair_rows(flows: Path) -> list[dict[str, Any]]:
    import csv

    pred_path = _out() / "detect" / "predictions_detailed.json"
    if not pred_path.is_file():
        raise SystemExit(f"Detect missing: {pred_path}. Run --predict-only first.")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    with flows.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(preds):
        raise SystemExit(f"flows={len(rows)} preds={len(preds)}")
    paired = []
    for flow, pred in zip(rows, preds):
        sample = dict(pred)
        six = int(float(flow["split_index"]))
        sample["split_index"] = six
        sample["true_label"] = str(flow.get("label_simplified") or sample.get("true_label") or "")
        paired.append({"flow": flow, "sample": sample})
    return paired


def _ranked_retrieve(vector_store: Any, query: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float, float]:
    """Same cascade as ``hybrid_children(..., rank=True)`` with retrieve vs rank clocks."""
    t0 = time.perf_counter()
    dense = dense_search(vector_store, query, top_k=DENSE_N)
    lexical = bm25_search(vector_store, query, top_k=BM25_N)
    fused = rrf_fuse([dense, lexical])
    retrieve_ms = _ms(t0)
    t1 = time.perf_counter()
    mmr_pool = fused[: max(int(FINAL_CHILDREN) * 2, 40)]
    picked = mmr_select(vector_store, query, mmr_pool, k=int(FINAL_CHILDREN), lambda_mult=MMR_LAMBDA)
    rank_ms = _ms(t1)
    for child in picked:
        if child.get("rerank_score") is None:
            child["rerank_score"] = float(child.get("rrf") or child.get("vector_score") or 0.0)
    ir_secs = expand_parent_sections(
        picked, top_sections=FINAL_PARENTS, max_parent_chars=PROMPT_BODY_CHARS
    )
    prompt_secs = compact_sections(ir_secs, n=PROMPT_SECTIONS)
    return prompt_secs, ir_secs, retrieve_ms, rank_ms


def run_reason(flows: Path) -> Path:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (backend/.env)")
    paired = _pair_rows(flows)
    live = _out()
    path = live / "runs.jsonl"
    done = {int(r["split_index"]) for r in _read_jsonl(path) if r.get("split_index") is not None}
    remain = [p for p in paired if int(p["sample"]["split_index"]) not in done]
    print(f"Reason {SYSTEM}: already={len(done)} remain={len(remain)} flows={len(paired)}")
    if not remain:
        _write_mitigation_plans(path)
        print(f"runs.jsonl -> {path} ({len(done)} rows)")
        return path

    attack_actions, agentic_features = load_attack_and_agentic(verbose=False)
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    reason_mod.vector_store = vector_store
    reason_mod._RAG_PARENTS = load_parent_store(VECTOR_STORE_DIR)
    reason_mod.predictions_data = [{"_e2e": True}]
    reason_mod.attack_actions_data = attack_actions
    reason_mod.agentic_features_data = agentic_features

    for i, item in enumerate(paired, 1):
        sample = item["sample"]
        six = int(sample["split_index"])
        if six in done:
            continue
        pred = str(sample.get("predicted_label") or "")
        true = str(sample.get("true_label") or "")
        print(f"  [{i}/{len(paired)}] split={six} true={true} pred={pred} ...", flush=True)
        query = build_template_rag_query(sample)
        prompt_secs, ir_secs, retrieve_ms, rank_ms = _ranked_retrieve(vector_store, query)
        prompt = create_agentic_orchestration_prompt(
            sample,
            prompt_secs,
            attack_actions,
            agentic_features,
            include_knowledge_base=True,
            include_conditions=True,
        )
        t0 = time.perf_counter()
        plan, raw, usage, format_error = _llm_call(prompt)
        llm_ms = _ms(t0)
        rec = {
            "split_index": six,
            "system": SYSTEM,
            "true_label": true,
            "predicted_label": pred,
            "confidence": sample.get("confidence"),
            "detect_ms": sample.get("detect_ms"),
            "query": query,
            "prompt_parent_ids": [str(s.get("parent_id") or "") for s in prompt_secs if s.get("parent_id")],
            "plan": plan,
            "format_error": format_error,
            "latency": {
                "detect_ms": sample.get("detect_ms"),
                "retrieve_ms": retrieve_ms,
                "rank_ms": rank_ms,
                "llm_ms": llm_ms,
            },
            "usage": usage,
            "raw_chars": len(raw),
        }
        _append_jsonl(path, rec)
        done.add(six)
    _write_mitigation_plans(path)
    print(f"runs.jsonl -> {path} ({len(done)} rows)")
    return path


def _write_mitigation_plans(runs_path: Path) -> Path:
    recs = _read_jsonl(runs_path)
    cases = []
    for rec in recs:
        six = int(rec["split_index"])
        pred = str(rec.get("predicted_label") or "")
        plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else {}
        cases.append(
            {
                "case_id": str(six),
                "split_index": six,
                "true_label": rec.get("true_label"),
                "predicted_label": pred,
                SYSTEM: {"predicted_label": pred, "plan": plan},
            }
        )
    out = _out() / "mitigation_plans.json"
    _dump(out, {"system": SYSTEM, "n": len(cases), "cases": cases})
    return out


def _executor_for(settings: Any, tier: str) -> str | None:
    return {
        "Access / ISP": settings.executor_access_private_key,
        "Perimeter / IDS": settings.executor_perimeter_private_key,
        "Endpoint / EDR": settings.executor_endpoint_private_key,
    }.get(tier)


def run_chain() -> Path:
    settings = get_settings()
    if not settings.trust_chain_enabled or not settings.trust_chain_contract_address:
        raise SystemExit("TRUST_CHAIN is not configured for the running Hardhat service")
    live = _out()
    runs = _read_jsonl(live / "runs.jsonl")
    if not runs:
        raise SystemExit("runs.jsonl missing. Run --reason-only first.")
    path = live / "honest.jsonl"
    done = {int(r["split_index"]) for r in _read_jsonl(path) if r.get("split_index") is not None}
    contract = settings.trust_chain_contract_address or ""
    for i, rec in enumerate(runs, 1):
        six = int(rec["split_index"])
        if six in done:
            continue
        pred = str(rec.get("predicted_label") or "").strip()
        plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
        plan_id = f"{six}-{SYSTEM}-honest"
        row: dict[str, Any] = {
            "split_index": six,
            "plan_id": plan_id,
            "system": SYSTEM,
            "true_label": rec.get("true_label"),
            "predicted_label": pred,
            "stored": False,
            "applied": 0,
            "units": 0,
            "unresolved": 0,
            "error": None,
            "latency": dict(rec.get("latency") or {}),
        }
        if not pred:
            row["error"] = "attack_type_unresolved"
            row["unresolved"] = 1
            _append_jsonl(path, row)
            done.add(six)
            print(f"  chain {i}/{len(runs)} {plan_id} unresolved", flush=True)
            continue
        try:
            pin = build_plan_input(
                structured_plan=plan,
                job_id=plan_id,
                prediction_id=plan_id,
                row_index=six,
                attack_type=pred,
            )
        except (TypeError, ValueError) as exc:
            row["error"] = str(exc)
            _append_jsonl(path, row)
            done.add(six)
            print(f"  chain {i}/{len(runs)} {plan_id} build_fail {exc}", flush=True)
            continue
        units = list(pin.primary_actions) + list(pin.supporting_actions)
        row["units"] = len(units)
        try:
            _tx, _addr, store_ms = store_plan_on_chain(settings, plan_id=plan_id, plan=pin)
            row["stored"] = True
            row["latency"]["commit_ms"] = store_ms
        except Exception as exc:
            row["error"] = revert_reason(exc)
            row["latency"]["commit_ms"] = 0.0
            _append_jsonl(path, row)
            done.add(six)
            print(f"  chain {i}/{len(runs)} {plan_id} store {row['error']}", flush=True)
            continue
        _rpc, on_chain, err, verify_ms = read_plan_from_chain(
            settings, contract_address=contract, plan_id=plan_id
        )
        row["latency"]["verify_ms"] = verify_ms
        want = reasoning_hash_sha256(str((plan or {}).get("overall_reasoning") or ""))
        got = ((on_chain or {}).get("reasoning_hash") or "").replace("0x", "")
        row["reasoning_hash_match"] = bool(on_chain) and got == want
        if err:
            row["verify_error"] = err
        apply_ms = 0.0
        for action, tier in units:
            t0 = time.perf_counter()
            _tx, apply_err = apply_action_on_chain(
                settings,
                plan_id=plan_id,
                action=action,
                tier=tier,
                private_key=_executor_for(settings, tier),
            )
            apply_ms += _ms(t0)
            if apply_err:
                row["apply_error"] = apply_err
            else:
                row["applied"] += 1
        row["latency"]["apply_ms"] = round(apply_ms, 1)
        _append_jsonl(path, row)
        done.add(six)
        print(f"  chain {i}/{len(runs)} {plan_id} stored={row['stored']} applied={row['applied']}/{row['units']}", flush=True)
    print(f"honest.jsonl -> {path} ({len(done)} rows)")
    return path


def _stat(vals: list[float]) -> dict[str, Any]:
    xs = [float(v) for v in vals if v is not None]
    if not xs:
        return {"sum_ms": None, "n": 0, "mean_ms": None, "min_ms": None, "max_ms": None}
    return {
        "sum_ms": round(sum(xs), 1),
        "n": len(xs),
        "mean_ms": round(sum(xs) / len(xs), 2),
        "min_ms": round(min(xs), 1),
        "max_ms": round(max(xs), 1),
    }


def _cell(block: dict[str, Any], key: str, field: str = "sum_ms") -> str:
    step = block.get(key) or {}
    val = step.get(field)
    if val is None:
        return "—"
    if field == "mean_ms":
        return f"{float(val):.2f}"
    return f"{float(val):.1f}"


def _latency_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[float]] = {k: [] for k in CLOCKS}
    e2e_vals: list[float] = []
    mapping_keys = {
        "detect": "detect_ms",
        "retrieve": "retrieve_ms",
        "rank": "rank_ms",
        "llm": "llm_ms",
        "commit": "commit_ms",
        "verify": "verify_ms",
        "apply": "apply_ms",
    }
    for row in rows:
        lat = dict(row.get("latency") or {})
        parts: list[float] = []
        for key, src in mapping_keys.items():
            val = lat.get(src)
            if val is None:
                continue
            buckets[key].append(float(val))
            parts.append(float(val))
        if len(parts) == len(CLOCKS):
            e2e_vals.append(sum(parts))
    out: dict[str, Any] = {"n": len(rows)}
    for key in CLOCKS:
        out[key] = _stat(buckets[key])
    out["e2e"] = _stat(e2e_vals)
    return out


def _join_rows() -> list[dict[str, Any]]:
    live = _out()
    runs = {int(r["split_index"]): r for r in _read_jsonl(live / "runs.jsonl")}
    honest = {int(r["split_index"]): r for r in _read_jsonl(live / "honest.jsonl")}
    rows = []
    for six, rec in sorted(runs.items()):
        hon = honest.get(six) or {}
        lat = dict(rec.get("latency") or {})
        lat.update((hon.get("latency") or {}))
        rows.append({**rec, **hon, "latency": lat, "split_index": six})
    return rows


def write_report(flows: Path) -> Path:
    live = _out()
    rows = _join_rows()
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        lab = str(row.get("true_label") or "").strip().upper()
        by_label[lab].append(row)
    matrix: dict[str, Any] = {lab: _latency_block(by_label.get(lab) or []) for lab in NINE}
    matrix["Overall"] = _latency_block(rows)
    _dump(live / "latency.json", matrix)

    def store_line(label: str, group: list[dict[str, Any]]) -> str:
        n = len(group)
        stored = sum(1 for r in group if r.get("stored"))
        applied = sum(int(r.get("applied") or 0) for r in group)
        fail = sum(1 for r in group if not r.get("stored"))
        unresolved = sum(int(r.get("unresolved") or 0) for r in group)
        return f"| {label} | {n} | {stored} | {applied} | {fail} | {unresolved} |"

    def _matrix_lines(field: str) -> list[str]:
        lines = [
            "| Attack | n | Detect | Retrieve | Rank | LLM | Commit | Verify | Apply | E2E |",
            "|--------|--:|-------:|---------:|-----:|----:|-------:|-------:|------:|----:|",
        ]
        for label in (*NINE, "Overall"):
            block = matrix[label]
            cells = " | ".join(_cell(block, s, field) for s in STEPS)
            lines.append(f"| {label} | {block['n']} | {cells} |")
        return lines

    lat_lines = _matrix_lines("sum_ms")
    mean_lines = _matrix_lines("mean_ms")
    min_lines = _matrix_lines("min_ms")
    max_lines = _matrix_lines("max_ms")

    e2e_lines = [
        "| Attack | n | mean | min | max |",
        "|--------|--:|-----:|----:|----:|",
    ]
    for label in (*NINE, "Overall"):
        block = matrix[label]
        e2e_lines.append(
            f"| {label} | {block['n']} | {_cell(block, 'e2e', 'mean_ms')} | {_cell(block, 'e2e', 'min_ms')} | {_cell(block, 'e2e', 'max_ms')} |"
        )

    step_title = {
        "detect": "Detect",
        "retrieve": "Retrieve",
        "rank": "Rank",
        "llm": "LLM",
        "commit": "Commit",
        "verify": "Verify",
        "apply": "Apply",
        "e2e": "E2E",
    }
    overall = matrix["Overall"]
    step_lines = [
        "| Step | n | mean | min | max | sum |",
        "|------|--:|-----:|----:|----:|----:|",
    ]
    for key in STEPS:
        step = overall.get(key) or {}
        step_lines.append(
            f"| {step_title[key]} | {step.get('n') or 0} | {_cell(overall, key, 'mean_ms')} | {_cell(overall, key, 'min_ms')} | {_cell(overall, key, 'max_ms')} | {_cell(overall, key)} |"
        )

    fails = [r for r in rows if not r.get("stored")]
    fail_lines = ["None."]
    if fails:
        fail_lines = [f"- `{r.get('split_index')}` {r.get('error')}" for r in fails]

    hist = {lab: len(by_label.get(lab) or []) for lab in NINE}
    body = "\n".join(
        [
            "# E2E detect → RAG → reason → chain — latency",
            "",
            f"Input: `{flows}`. N = **{len(rows)}**. System: `{SYSTEM}`. LLM calls = **{len(rows)}**.",
            "Happy path only. Plans stored as generated. No inject. No BERTScore.",
            "",
            f"Class histogram (true_label): `{json.dumps(hist)}`.",
            "",
            "## Latency (total ms)",
            "",
            "Cell = **total ms** over that row’s n flows. **E2E** is the row sum of Detect…Apply. **Overall** is the column sum of the nine attack types (n and every step).",
            "",
            *lat_lines,
            "",
            "| Step | Clock |",
            "|------|--------|",
            "| Detect | VFL + SHAP, amortized per flow |",
            "| Retrieve | FAISS + BM25 + RRF |",
            "| Rank | MMR |",
            "| LLM | Mitigation Plan |",
            "| Commit | `storePlan` |",
            "| Verify | `getPlan` / `isInPlan` |",
            "| Apply | honest `markApplied` (per plan) |",
            "| E2E | sum of the seven steps |",
            "",
            "## Latency (mean ms)",
            "",
            "Cell = **mean ms** per flow in that row. **Overall** is mean over all n flows.",
            "",
            *mean_lines,
            "",
            "## Latency (min ms)",
            "",
            "Cell = **min ms** among that row’s flows.",
            "",
            *min_lines,
            "",
            "## Latency (max ms)",
            "",
            "Cell = **max ms** among that row’s flows.",
            "",
            *max_lines,
            "",
            "## E2E per flow (ms)",
            "",
            "Per-flow E2E = Detect + Retrieve + Rank + LLM + Commit + Verify + Apply. **Overall** is all n flows (not a mean of means).",
            "",
            *e2e_lines,
            "",
            "### Overall by step (per-flow ms)",
            "",
            *step_lines,
            "",
            "## Honest store",
            "",
            "| Attack | n | Stored | Applied | Fail | unresolved |",
            "|--------|--:|-------:|--------:|-----:|-----------:|",
            *[store_line(lab, by_label.get(lab) or []) for lab in NINE],
            store_line("Overall", rows),
            "",
            "Overall n / Stored / Applied / Fail / unresolved are the sums of the attack-type rows.",
            "",
            "## Chain fails",
            "",
            *fail_lines,
            "",
        ]
    )
    report = live / "report.md"
    _write(report, body)
    if AGENT_REPORT.parent.is_dir():
        _write(AGENT_REPORT, body)
    manifest = {
        "input": str(flows),
        "n": len(rows),
        "system": SYSTEM,
        "llm_calls": len(rows),
        "honest_stored": sum(1 for r in rows if r.get("stored")),
        "unresolved": sum(int(r.get("unresolved") or 0) for r in rows),
        "output": str(live),
    }
    _dump(live / "manifest.json", manifest)
    _write(
        live / "README.md",
        "Happy-path e2e results for pragma-e2e-detect-chain. Input CSV is read-only; this folder holds detect / plans / chain / latency.\n",
    )
    print("wrote", report)
    print("wrote", AGENT_REPORT)
    return report


def main() -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="Happy-path e2e latency (detect → RAG_RANKING → chain)")
    parser.add_argument(
        "--input",
        default="",
        help="Fixture folder or flows.csv (default: experiments/data/e2e-detect-chain/flows.csv)",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Confirm the frozen e2e-detect-chain CSV exists (does not resample)",
    )
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--reason-only", action="store_true")
    parser.add_argument("--chain-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    only = [args.sample_only, args.predict_only, args.reason_only, args.chain_only, args.report_only]
    if sum(bool(x) for x in only) > 1:
        raise SystemExit(
            "Pick at most one of --sample-only / --predict-only / --reason-only / --chain-only / --report-only"
        )
    if args.sample_only:
        frozen = fixture_set_dir(FIXTURE_E2E_DETECT_CHAIN, mkdir=False) / "flows.csv"
        if not frozen.is_file():
            raise SystemExit(
                f"Frozen e2e fixture missing: {frozen}. Do not resample; restore experiments/data/e2e-detect-chain/flows.csv"
            )
        print(f"Fixture exists: {frozen}")
        return 0
    flows = resolve_input(args.input)
    print("input", flows)
    if args.report_only:
        write_report(flows)
        return 0
    if args.predict_only:
        run_detect(flows, chunk_size=args.chunk_size)
        return 0
    if args.reason_only:
        run_reason(flows)
        write_report(flows)
        return 0
    if args.chain_only:
        run_chain()
        write_report(flows)
        return 0
    run_detect(flows, chunk_size=args.chunk_size)
    run_reason(flows)
    run_chain()
    write_report(flows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
