"""9-class × 6-cell reasoning ablation (paper path + RAG/condition/ranking switches).

Writes ``experiments/reason/all_types_<ts>/``. Does not Commit/Apply.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from openai import OpenAI

from scripts.env import experiment_dir, load_project_dotenv, resolve_latest_predict_dir
from scripts.llm_prompt import create_agentic_orchestration_prompt
from scripts.rag_io import load_attack_and_agentic, load_predictions, load_vector_store
from scripts.reason import (
    VECTOR_STORE_DIR,
    _DEFAULT_MMR_K,
    _DEFAULT_RERANK_K,
    _LLM_RAG_SECTIONS_IN_PROMPT,
    _PER_QUERY_RETRIEVE_K,
    _ensure_runtime_loaded,
    build_template_rag_query,
    build_template_rag_query_nocond,
    expand_parent_sections,
    extract_sample_summary,
    merge_and_dedupe_child_chunks,
    mmr_select,
    rerank_with_cross_encoder,
    retrieve_child_chunks_for_query,
)

load_project_dotenv()

EMPTY_KB_SENTENCE = "No relevant documents found from RAG search."

CELLS: list[dict[str, Any]] = [
    {"id": "rag_cond_rank", "rag": True, "cond": True, "rank": True},
    {"id": "rag_cond_norank", "rag": True, "cond": True, "rank": False},
    {"id": "rag_nocond_rank", "rag": True, "cond": False, "rank": True},
    {"id": "rag_nocond_norank", "rag": True, "cond": False, "rank": False},
    {"id": "norag_cond", "rag": False, "cond": True, "rank": False},
    {"id": "norag_nocond", "rag": False, "cond": False, "rank": False},
]

CLASS_ORDER = (
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


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


def _dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _compact_prediction(sample: dict[str, Any]) -> dict[str, Any]:
    s = extract_sample_summary(sample)
    shap = sample.get("shap_explanation") or {}
    shares = shap.get("party_contributions_pct") or shap.get("party_contributions") or {}
    return {
        "sample_id": sample.get("sample_id"),
        "true_label": sample.get("true_label"),
        "predicted_label": sample.get("predicted_label"),
        "confidence": sample.get("confidence"),
        "is_correct": sample.get("is_correct"),
        "dominant_domain": s.get("dominant_tier"),
        "dominant_contribution_pct": s.get("dominant_pct"),
        "domain_shares": shares,
        "top_features_by_domain": s.get("top_features"),
        "source_file": sample.get("_source_file"),
    }


def _one_per_true_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_true: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("true_label") or row.get("predicted_label") or "").strip().upper()
        if not key or key in by_true:
            continue
        by_true[key] = row
    out: list[dict[str, Any]] = []
    for name in CLASS_ORDER:
        if name in by_true:
            out.append(by_true[name])
    for name, row in sorted(by_true.items()):
        if name not in CLASS_ORDER:
            out.append(row)
    return out


def _action_pairs(plan: dict[str, Any] | None, key: str) -> list[str]:
    if not plan:
        return []
    items = plan.get(key) or []
    out: list[str] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if isinstance(it, dict):
            act = str(it.get("action") or "").strip()
            tier = str(it.get("network_tier") or "").strip()
            out.append(f"{act} @ {tier}" if act else tier)
        else:
            out.append(str(it))
    return out


def _llm_call(prompt: str) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3
        )
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
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
    except Exception as exc:
        parsed = {"parse_error": str(exc)}
    return parsed, raw, usage


def _cell_readme(
    *,
    class_name: str,
    cell: dict[str, Any],
    pred: dict[str, Any],
    query: str,
    rag_titles: list[str],
    plan: dict[str, Any] | None,
    latency: dict[str, Any],
    rag_on: bool,
) -> str:
    lines = [
        f"# {class_name} / `{cell['id']}`",
        "",
        f"- RAG: {'on' if cell['rag'] else 'off'}  |  condition: {'on' if cell['cond'] else 'off'}  |  ranking: {'on' if cell['rank'] else 'off (n/a)' if not cell['rag'] else 'off'}",
        f"- predicted: **{pred.get('predicted_label')}**  conf={float(pred.get('confidence') or 0):.1%}  true={pred.get('true_label')}",
        f"- dominant: {pred.get('dominant_domain')} ({float(pred.get('dominant_contribution_pct') or 0):.1f}%)",
        f"- shares: {pred.get('domain_shares')}",
        "",
        "## Query",
        "",
        "```",
        query.strip() or "(none)",
        "```",
        "",
        "## RAG",
        "",
    ]
    if not rag_on:
        lines.append(EMPTY_KB_SENTENCE)
    elif rag_titles:
        for i, t in enumerate(rag_titles, 1):
            lines.append(f"{i}. {t}")
    else:
        lines.append(EMPTY_KB_SENTENCE)
    lines += ["", "## Plan", ""]
    if plan and "parse_error" not in (plan or {}):
        lines.append(f"- threat_level: {plan.get('threat_level')}")
        lines.append(f"- execution_priority: {plan.get('execution_priority')}")
        lines.append(f"- primary: {', '.join(_action_pairs(plan, 'primary_actions')) or '(none)'}")
        lines.append(f"- supporting: {', '.join(_action_pairs(plan, 'supporting_actions')) or '(none)'}")
        lines.append(f"- knowledge_sources_used: {plan.get('knowledge_sources_used')}")
    else:
        lines.append("- parse error or empty LLM response")
    lines += ["", "## Latency (ms)", ""]
    for step in ("predict", "query", "retrieve", "rank", "prompt", "llm"):
        val = latency.get(step)
        lines.append(f"- {step}: {val if val is not None else 'skipped'}")
    lines.append(f"- total_cell: {latency.get('total_cell')}")
    return "\n".join(lines) + "\n"


def _class_readme(class_name: str, pred: dict[str, Any], cells: list[dict[str, Any]]) -> str:
    lines = [
        f"# {class_name}",
        "",
        f"- predicted **{pred.get('predicted_label')}**  conf={float(pred.get('confidence') or 0):.1%}  true={pred.get('true_label')}  correct={pred.get('is_correct')}",
        f"- dominant **{pred.get('dominant_domain')}** ({float(pred.get('dominant_contribution_pct') or 0):.1f}%)",
        f"- shares: {pred.get('domain_shares')}",
        "",
        "| cell | RAG | cond | rank | threat | priority | primary | llm_ms |",
        "|------|-----|------|------|--------|----------|---------|--------|",
    ]
    for c in cells:
        plan = c.get("plan") or {}
        lines.append(
            "| `{id}` | {rag} | {cond} | {rank} | {threat} | {pri} | {prim} | {llm} |".format(
                id=c["id"],
                rag="on" if c["rag"] else "off",
                cond="on" if c["cond"] else "off",
                rank="on" if c["rank"] else "n/a",
                threat=plan.get("threat_level", ""),
                pri=plan.get("execution_priority", ""),
                prim="; ".join(_action_pairs(plan, "primary_actions"))[:80],
                llm=c.get("latency", {}).get("llm"),
            )
        )
    return "\n".join(lines) + "\n"


def _root_readme(
    out_dir: Path,
    predict_dir: Path,
    samples: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
    cell_ok: int,
) -> str:
    by_step: dict[str, list[float]] = {}
    for r in latency_rows:
        step = str(r.get("step") or "")
        ms = r.get("ms")
        if ms is None:
            continue
        by_step.setdefault(step, []).append(float(ms))
    lines = [
        "# All-types reasoning ablation",
        "",
        f"- output: `{out_dir}`",
        f"- predictions: `{predict_dir}`",
        f"- samples: {len(samples)}  cells/sample: {len(CELLS)}  LLM calls: {len(samples) * len(CELLS)}",
        f"- parsed plans: {cell_ok}/{len(samples) * len(CELLS)}",
        f"- index: `{VECTOR_STORE_DIR}`",
        f"- model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`  T=0.3",
        "",
        "## Latency totals / mean (ms)",
        "",
        "| step | n | sum | mean |",
        "|------|---|-----|------|",
    ]
    for step in ("predict", "query", "retrieve", "rank", "prompt", "llm"):
        vals = by_step.get(step) or []
        if not vals:
            lines.append(f"| {step} | 0 |  |  |")
            continue
        lines.append(f"| {step} | {len(vals)} | {sum(vals):.1f} | {sum(vals)/len(vals):.1f} |")
    lines += ["", "## Matrix", "", "| class | true | pred | conf | dominant |"]
    lines.append("|-------|------|------|------|----------|")
    for s in samples:
        p = _compact_prediction(s)
        cls = str(p.get("true_label") or p.get("predicted_label") or "").upper()
        lines.append(
            f"| {cls} | {p.get('true_label')} | {p.get('predicted_label')} | "
            f"{float(p.get('confidence') or 0):.1%} | {p.get('dominant_domain')} |"
        )
    return "\n".join(lines) + "\n"


def run_cell(
    sample: dict[str, Any],
    cell: dict[str, Any],
    vector_store: Any,
    attack_actions: dict[str, Any] | None,
    agentic_features: dict[str, Any] | None,
    predict_ms: float | None,
) -> dict[str, Any]:
    t_cell = time.perf_counter()
    latency: dict[str, Any] = {"predict": predict_ms}

    t0 = time.perf_counter()
    query = (
        build_template_rag_query(sample) if cell["cond"] else build_template_rag_query_nocond(sample)
    )
    latency["query"] = _ms(t0)

    rag_results: list[dict[str, Any]] = []
    if cell["rag"]:
        t0 = time.perf_counter()
        children = retrieve_child_chunks_for_query(
            vector_store, query, top_k=int(_PER_QUERY_RETRIEVE_K)
        )
        merged = merge_and_dedupe_child_chunks([children])
        latency["retrieve"] = _ms(t0)
        if cell["rank"]:
            t0 = time.perf_counter()
            pool = mmr_select(vector_store, query, merged, k=int(_DEFAULT_MMR_K), lambda_mult=0.5)
            ranked = rerank_with_cross_encoder(query, pool)
            top_children = ranked[: int(_DEFAULT_RERANK_K)]
            rag_results = expand_parent_sections(top_children, top_sections=5)
            latency["rank"] = _ms(t0)
        else:
            latency["rank"] = None
            top_children = merged[: int(_DEFAULT_RERANK_K)]
            for c in top_children:
                if c.get("rerank_score") is None:
                    c["rerank_score"] = float(c.get("vector_score", 0.0) or 0.0)
            rag_results = expand_parent_sections(top_children, top_sections=5)
        rag_results = rag_results[: min(5, _LLM_RAG_SECTIONS_IN_PROMPT)]
    else:
        latency["retrieve"] = None
        latency["rank"] = None

    t0 = time.perf_counter()
    prompt = create_agentic_orchestration_prompt(
        sample,
        rag_results if cell["rag"] else [],
        attack_actions,
        agentic_features,
        include_knowledge_base=True,
        include_conditions=bool(cell["cond"]),
    )
    latency["prompt"] = _ms(t0)

    t0 = time.perf_counter()
    plan, raw, usage = _llm_call(prompt)
    latency["llm"] = _ms(t0)
    latency["llm_usage"] = usage
    latency["total_cell"] = _ms(t_cell)

    titles = []
    for sec in rag_results:
        src = sec.get("source_file") or ""
        title = sec.get("title") or ""
        titles.append(f"{title} [{src}]" if src else str(title))

    return {
        "query": query,
        "rag_results": rag_results if cell["rag"] else [],
        "rag_titles": titles,
        "prompt": prompt,
        "plan": plan,
        "raw": raw,
        "latency": latency,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="9×6 reasoning ablation under experiments/reason/all_types_<ts>/")
    p.add_argument("--classes", default="", help="Comma-separated true labels (smoke). Default: all 9.")
    p.add_argument("--strip-prompt", action="store_true", help="Optional extra free-form cell (off by default).")
    args = p.parse_args(argv)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (backend/.env)")

    want = {x.strip().upper() for x in args.classes.split(",") if x.strip()}
    predict_dir = resolve_latest_predict_dir(require=True)
    rows = load_predictions(predict_dir, verbose=True)
    samples = _one_per_true_label(rows)
    if want:
        samples = [
            s
            for s in samples
            if str(s.get("true_label") or "").strip().upper() in want
        ]
    if not samples:
        raise SystemExit("No prediction rows to ablate")

    _ensure_runtime_loaded(verbose=True)
    vector_store = load_vector_store(VECTOR_STORE_DIR)
    attack_actions, agentic_features = load_attack_and_agentic(verbose=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = experiment_dir("reason") / f"all_types_{stamp}"
    by_attack = out_dir / "by_attack"
    by_attack.mkdir(parents=True, exist_ok=True)

    latency_rows: list[dict[str, Any]] = []
    parsed_ok = 0

    for sample in samples:
        pred = _compact_prediction(sample)
        class_name = str(pred.get("true_label") or pred.get("predicted_label") or "UNKNOWN").upper()
        class_dir = by_attack / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        _dump(class_dir / "prediction.json", pred)

        cell_summaries: list[dict[str, Any]] = []
        print(f"\n=== {class_name} pred={pred.get('predicted_label')} conf={float(pred.get('confidence') or 0):.1%} ===")
        for cell in CELLS:
            print(f"  cell {cell['id']} ...")
            result = run_cell(sample, cell, vector_store, attack_actions, agentic_features, predict_ms=None)
            cell_dir = class_dir / cell["id"]
            cell_dir.mkdir(parents=True, exist_ok=True)
            _write(cell_dir / "query.txt", result["query"] + "\n")
            _dump(cell_dir / "rag_sections.json", result["rag_results"])
            _write(cell_dir / "prompt.txt", result["prompt"])
            _write(cell_dir / "response_raw.txt", result["raw"] + "\n")
            plan = result["plan"]
            if plan and "parse_error" not in plan:
                _dump(cell_dir / "plan.json", plan)
                parsed_ok += 1
            else:
                _dump(cell_dir / "plan.json", plan or {"parse_error": "empty"})
            _dump(cell_dir / "latency.json", result["latency"])
            _write(
                cell_dir / "README.md",
                _cell_readme(
                    class_name=class_name,
                    cell=cell,
                    pred=pred,
                    query=result["query"],
                    rag_titles=result["rag_titles"],
                    plan=plan,
                    latency=result["latency"],
                    rag_on=bool(cell["rag"]),
                ),
            )
            cell_summaries.append({**cell, "plan": plan, "latency": result["latency"]})
            for step in ("predict", "query", "retrieve", "rank", "prompt", "llm"):
                latency_rows.append(
                    {
                        "sample": class_name,
                        "cell": cell["id"],
                        "step": step,
                        "ms": result["latency"].get(step),
                    }
                )

        _write(class_dir / "README.md", _class_readme(class_name, pred, cell_summaries))

    _write(out_dir / "README.md", _root_readme(out_dir, predict_dir, samples, latency_rows, parsed_ok))
    _dump(out_dir / "latency_summary.json", latency_rows)
    with (out_dir / "latency_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "cell", "step", "ms"])
        w.writeheader()
        w.writerows(latency_rows)

    if args.strip_prompt:
        print("--strip-prompt is reserved; default matrix does not include that cell.")

    print(f"\nDone. {parsed_ok} parsed plans → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
