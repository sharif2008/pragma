"""A/B/C reason scorecard. N is the row count of ``--input``.

Not the planner library (``reason.py``) and not the e2e chain run (``e2e_detect_chain.py``).

    python scripts/rag_reason.py --input ../experiments/data/eval-10
    python scripts/rag_reason.py --input ../experiments/data/eval-200 --reason-only
    python scripts/rag_reason.py --input ../experiments/data/rag_reason_500
    python scripts/rag_reason.py --input ../experiments/data/eval-10 --report-only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import runpy
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from sklearn.model_selection import train_test_split

from scripts.env import (
    FIXTURE_GOLD_100,
    FIXTURE_RAG_REASON_500,
    LIVE_RAG_REASON_500,
    fixture_set_dir,
    load_project_dotenv,
    named_live_dir,
    new_named_live_dir,
)
from scripts.gold_sample_100 import NINE, SEED, _sha256_file, load_training_frame
from scripts.llm_prompt import create_agentic_orchestration_prompt
from scripts.network_domains import DOMAIN_LABELS
from scripts.rag_retrieval_scoring import _llm_call, _norm_action, _retrieve
from scripts.rag_io import load_attack_and_agentic, load_parent_store, load_vector_store
from scripts.reason import VECTOR_STORE_DIR
from scripts import reason as reason_mod
from scripts.vfl import load_attack_actions_by_type, not_allowed_actions_for_type

load_project_dotenv()

DETECT_CHUNK = 100
_FLOWS: Path | None = None
_LIVE_NAME = LIVE_RAG_REASON_500
_N = 0
VALID_TIERS = {str(x) for x in DOMAIN_LABELS}
SYSTEMS = (
    {"id": "LLM_only", "rag": False, "rank": False, "label": "No RAG"},
    {"id": "RAG_No_Ranking", "rag": True, "rank": False, "label": "RAG"},
    {"id": "RAG_RANKING", "rag": True, "rank": True, "label": "Ranked RAG"},
)
SYSTEM_IDS = [s["id"] for s in SYSTEMS]
SYSTEM_LABEL = {s["id"]: s["label"] for s in SYSTEMS}
PHASES = (
    {"key": "action_correct", "title": "Action correctness", "higher": True},
    {"key": "policy_compliant", "title": "Policy compliance", "higher": True},
    {"key": "unsafe_action", "title": "Unsafe action rate", "higher": False},
)
# Same class needles as gold_freeze.CLASS_PDF — lexical evidence proxy only.
CLASS_NEEDLES: dict[str, tuple[str, ...]] = {
    "BENIGN": ("control 8: audit log", "safeguard 8.1"),
    "BOT": ("control 10: malware", "malware defenses"),
    "DDOS": ("denial of service", "control 13: network monitoring"),
    "DOS": ("denial of service", "control 13: network monitoring"),
    "SSHPATATOR": ("control 6: access control", "control 5: account", "brute force"),
    "FTPPATATOR": ("control 5: account", "control 6: access control", "brute force"),
    "PORTSCAN": ("control 13: network monitoring", "reconnaissance"),
    "WEBATTACK": ("control 16: application", "application software security"),
    "OTHERS": ("control 17: incident", "incident response"),
}


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _n_target() -> int:
    return int(_N)


def _fixture_dir(*, mkdir: bool = False) -> Path:
    if _FLOWS is not None:
        return _FLOWS.parent
    return fixture_set_dir(FIXTURE_RAG_REASON_500, mkdir=mkdir)


def _live_dir(*, reset: bool = False) -> Path:
    name = _LIVE_NAME or LIVE_RAG_REASON_500
    if reset:
        return new_named_live_dir("reason", name)
    return named_live_dir("reason", name, mkdir=True)


def resolve_input_csv(raw: str) -> Path:
    text = (raw or "").strip()
    path = Path(text) if text else fixture_set_dir(FIXTURE_RAG_REASON_500, mkdir=False)
    if not path.is_absolute():
        path = (_BACKEND / path).resolve() if not path.exists() else path.resolve()
        if not path.exists():
            path = (_REPO / text).resolve() if text else path
    if path.is_file() and path.suffix.lower() == ".csv":
        return path
    csv_path = path / "flows.csv"
    if csv_path.is_file():
        return csv_path
    raise SystemExit(f"Input flows.csv not found: {path}. Pass --input PATH (N = row count of that file).")


def configure(flows: Path) -> int:
    global _FLOWS, _LIVE_NAME, _N
    import pandas as pd

    _FLOWS = flows
    _LIVE_NAME = flows.parent.name if flows.name.lower() == "flows.csv" else flows.stem
    _N = int(len(pd.read_csv(flows)))
    return _N


def _flows_csv() -> Path:
    if _FLOWS is not None:
        return _FLOWS
    path = fixture_set_dir(FIXTURE_RAG_REASON_500, mkdir=False) / "flows.csv"
    if not path.is_file():
        raise SystemExit(f"Fixture missing: {path}. Pass --input.")
    return path


def _gold_indices() -> set[int]:
    man = fixture_set_dir(FIXTURE_GOLD_100, mkdir=False) / "manifest.json"
    if not man.is_file():
        raise SystemExit(f"Gold manifest missing: {man}")
    data = json.loads(man.read_text(encoding="utf-8"))
    return {int(x) for x in (data.get("selected_split_index") or [])}


def _waterfill_quotas(available: dict[str, int], n_target: int) -> dict[str, int]:
    quota = {c: 0 for c in NINE}
    remaining = int(n_target)
    while remaining > 0:
        growable = [c for c in NINE if quota[c] < available.get(c, 0)]
        if not growable:
            break
        for c in growable:
            if remaining == 0:
                break
            quota[c] += 1
            remaining -= 1
    return quota


def sample_fixture(n_target: int, out_dir: Path) -> Path:
    n_target = int(n_target)
    if n_target <= 0:
        raise SystemExit("--n must be > 0 to sample")
    gold_idx = _gold_indices()
    df, csvs = load_training_frame()
    stratify = df["label_numeric"]
    _trainval_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,
    )
    test_idx = list(test_idx)
    test = df.iloc[test_idx].copy()
    test["split_index"] = test_idx
    pool = test[~test["split_index"].isin(gold_idx)].copy()

    by_class: dict[str, list[int]] = {}
    for cls in NINE:
        pos = [int(i) for i, lab in enumerate(pool["label_simplified"]) if str(lab) == cls]
        by_class[cls] = pos
    available = {c: len(by_class[c]) for c in NINE}
    quota = _waterfill_quotas(available, n_target)
    if sum(quota.values()) < n_target:
        print(f"Shortfall: test-minus-gold can supply {sum(quota.values())} of {n_target}")

    seeds: list[int] = []
    for cls in NINE:
        k = quota[cls]
        pos = by_class[cls]
        if k <= 0:
            continue
        if k == len(pos):
            take = pos
        else:
            take, _rest = train_test_split(pos, train_size=k, random_state=SEED)
            take = list(take)
        seeds.extend(take)

    picked = pool.iloc[seeds].copy()
    picked = picked.sort_values("split_index").reset_index(drop=True)
    picked.insert(0, "reason_row", range(1, len(picked) + 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flows = out_dir / "flows.csv"
    picked.to_csv(flows, index=False)
    hist = Counter(str(x) for x in picked["label_simplified"])
    catalog = {k: list(v) for k, v in load_attack_actions_by_type().items()}
    manifest = {
        "dataset_paths": [str(p) for p in csvs],
        "dataset_sha256": {p.name: _sha256_file(p) for p in csvs},
        "seed": SEED,
        "n_target": n_target,
        "n_sampled": len(picked),
        "systems": list(SYSTEM_IDS),
        "llm_calls": len(picked) * len(SYSTEM_IDS),
        "excluded_gold_split_index": sorted(gold_idx),
        "selected_split_index": [int(x) for x in picked["split_index"]],
        "class_counts_test": dict(Counter(str(x) for x in test["label_simplified"])),
        "class_counts_available": available,
        "class_quotas": quota,
        "class_counts_sampled": dict(sorted(hist.items())),
        "aevra_catalog_size": {k: len(v) for k, v in catalog.items()},
        "aevra_catalog": catalog,
        "overlap_gold": 0,
        "flows_csv": str(flows),
        "flows_sha256": _sha256_file(flows),
    }
    overlap = set(manifest["selected_split_index"]) & gold_idx
    if overlap:
        raise SystemExit(f"sampled split_index overlaps gold: {sorted(overlap)[:8]}")
    _dump(out_dir / "manifest.json", manifest)
    n_llm = len(picked) * len(SYSTEM_IDS)
    catalog_lines = [
        f"# {out_dir.name}",
        "",
        f"{len(picked)} held-out test flows, **disjoint** from gold cores (`data/gold-100`).",
        "",
        f"A/B/C: **No RAG** vs **RAG** vs **Ranked RAG** — **three LLM calls per flow** ({n_llm} total).",
        "",
        "Headline: per-attack phase table (underline the better system) and action shares inside W[y].",
        f"These {len(picked)} rows are **not** gold-100.",
        "",
        "## Class counts (true label)",
        "",
        "| class | n | AEVARA catalog |",
        "|-------|--:|---------------:|",
    ]
    for cls in NINE:
        catalog_lines.append(
            f"| {cls} | {hist.get(cls, 0)} | {len(catalog.get(cls, []))} |"
        )
    catalog_lines += [
        "",
        "Do not resample. Do not include any gold `split_index`.",
        "Blockchain / e2e should reuse this fixture.",
        "",
    ]
    _write(out_dir / "README.md", "\n".join(catalog_lines))
    print(f"Wrote {flows} ({len(picked)} rows)")
    print("sampled histogram:", dict(sorted(hist.items())))
    print("AEVARA catalog sizes:", manifest["aevra_catalog_size"])
    return out_dir


def _chunk_pred_file(chunk_dir: Path) -> Path | None:
    files = sorted(chunk_dir.glob("predictions_detailed_*.json"))
    return files[-1] if files else None


def _merged_pred_path(live: Path) -> Path:
    detect = live / "detect"
    stable = detect / "predictions_detailed.json"
    if stable.is_file():
        return stable
    files = sorted(detect.glob("predictions_detailed_*.json"))
    if files:
        return files[-1]
    return stable


def _force_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def run_detect(*, chunk_size: int = DETECT_CHUNK) -> Path:
    import pandas as pd

    flows = _flows_csv()
    df = pd.read_csv(flows)
    n = len(df)
    live = _live_dir(reset=False)
    detect_root = live / "detect"
    detect_root.mkdir(parents=True, exist_ok=True)
    merged = _merged_pred_path(live)
    if merged.is_file():
        existing = json.loads(merged.read_text(encoding="utf-8"))
        if isinstance(existing, list) and len(existing) >= n:
            print(f"Detect already complete: {merged} ({len(existing)} rows)")
            return merged
    chunk_root = detect_root / "chunks"
    chunk_root.mkdir(parents=True, exist_ok=True)
    parts: list[list[dict[str, Any]]] = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        name = f"chunk_{start:04d}_{end:04d}"
        chunk_dir = chunk_root / name
        chunk_dir.mkdir(parents=True, exist_ok=True)
        done = _chunk_pred_file(chunk_dir)
        if done is not None:
            print(f"  reuse {done}")
            chunk_preds = json.loads(done.read_text(encoding="utf-8"))
        else:
            chunk_csv = chunk_dir / "flows.csv"
            df.iloc[start:end].to_csv(chunk_csv, index=False)
            os.environ["CHAINAGENT_SAMPLE_CSV"] = str(chunk_csv)
            os.environ["CHAINAGENT_PREDICT_OUT"] = str(chunk_dir)
            os.environ["PYTHONIOENCODING"] = "utf-8"
            _force_utf8_stdio()
            print(f"  detect {name} ({end - start} rows) ...", flush=True)
            runpy.run_path(str(_BACKEND / "scripts" / "detect_predict.py"), run_name="__main__")
            done = _chunk_pred_file(chunk_dir)
            if done is None:
                raise SystemExit(f"detect produced no predictions_detailed_*.json in {chunk_dir}")
            chunk_preds = json.loads(done.read_text(encoding="utf-8"))
        if not isinstance(chunk_preds, list) or len(chunk_preds) != (end - start):
            raise SystemExit(f"{name}: expected {end - start} preds, got {len(chunk_preds) if isinstance(chunk_preds, list) else type(chunk_preds)}")
        for i, rec in enumerate(chunk_preds):
            rec["sample_id"] = start + i
            rec["split_index"] = int(df.iloc[start + i]["split_index"])
            rec["reason_row"] = (
                int(df.iloc[start + i]["reason_row"])
                if "reason_row" in df.columns
                else start + i + 1
            )
        parts.append(chunk_preds)

    merged_rows = [row for part in parts for row in part]
    out = detect_root / "predictions_detailed.json"
    _dump(out, merged_rows)
    print(f"Merged detect → {out} ({len(merged_rows)} rows)")
    return out


def _pair_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flows_path = _flows_csv()
    man_path = flows_path.parent / "manifest.json"
    if not flows_path.is_file():
        raise SystemExit(f"Fixture missing: {flows_path}. Pass --input.")
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    with flows_path.open(encoding="utf-8", newline="") as f:
        flows = list(csv.DictReader(f))
    live = _live_dir(reset=False)
    pred_path = _merged_pred_path(live)
    if not pred_path.is_file():
        raise SystemExit(f"Detect missing: {pred_path}. Run --predict-only first.")
    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(preds, list):
        raise SystemExit(f"Expected a list in {pred_path}")
    if len(flows) != len(preds):
        raise SystemExit(f"flows={len(flows)} preds={len(preds)}")
    paired = []
    for flow, pred in zip(flows, preds):
        six = int(float(flow["split_index"]))
        row = dict(pred)
        row["split_index"] = six
        row["sample_id"] = six
        row["reason_row"] = int(float(flow.get("reason_row") or 0) or 0)
        paired.append({"flow": flow, "sample": row})
    gold = set(int(x) for x in (manifest.get("excluded_gold_split_index") or [])) or _gold_indices()
    overlap = {int(p["sample"]["split_index"]) for p in paired} & gold
    if overlap:
        raise SystemExit(f"paired rows overlap gold: {len(overlap)}")
    return paired, manifest


def _jsonl_path(live: Path) -> Path:
    path = live / "runs.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _done_keys(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("split_index") is None:
            continue
        sys_id = str(rec.get("system") or "").strip() or "RAG_RANKING"
        done.add((int(rec["split_index"]), sys_id))
    return done


def _append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _plan_actions(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for key in ("primary_actions", "supporting_actions"):
        for it in plan.get(key) or []:
            raw = it.get("action") if isinstance(it, dict) else it
            act = _norm_action(str(raw or ""))
            if act and act not in seen:
                seen.add(act)
                out.append(act)
    for raw in plan.get("all_actions") or []:
        act = _norm_action(str(raw if not isinstance(raw, dict) else raw.get("action") or ""))
        if act and act not in seen:
            seen.add(act)
            out.append(act)
    return out


def _plan_tiers(plan: dict[str, Any] | None) -> list[str]:
    if not isinstance(plan, dict):
        return []
    tiers: list[str] = []
    for key in ("primary_actions", "supporting_actions"):
        for it in plan.get(key) or []:
            if isinstance(it, dict) and it.get("network_tier"):
                tiers.append(str(it.get("network_tier") or "").strip())
    return tiers


def _primary_action(plan: dict[str, Any] | None, actions: list[str]) -> str:
    if isinstance(plan, dict):
        for it in plan.get("primary_actions") or []:
            raw = it.get("action") if isinstance(it, dict) else it
            act = _norm_action(str(raw or ""))
            if act:
                return act
    return actions[0] if actions else ""


def _parent_blob(parents: dict[str, Any], pids: list[Any]) -> str:
    parts: list[str] = []
    for pid in pids:
        rec = parents.get(str(pid)) or {}
        parts.append(str(rec.get("text") or ""))
    return " ".join(parts).lower()


def _evidence_support(true_label: str, actions: list[str], parent_text: str, empty_kb: bool) -> bool:
    if empty_kb or not parent_text.strip():
        return False
    for act in actions:
        norm = _norm_action(act)
        if norm and norm in parent_text:
            return True
    for needle in CLASS_NEEDLES.get(true_label, ()):
        if needle in parent_text:
            return True
    return False


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return sum(int(r.get(key) or 0) for r in rows) / len(rows)


def _cell(rate: float | None, k: int, n: int) -> str:
    if not n or rate is None:
        return "—"
    return f"{100.0 * rate:.1f}% ({k}/{n})"


def _underline(text: str) -> str:
    return f"<u>{text}</u>" if text != "—" else text


def _winner_ids(values: dict[str, float | None], *, higher: bool) -> set[str]:
    present = {k: float(v) for k, v in values.items() if v is not None}
    if not present:
        return set()
    best = max(present.values()) if higher else min(present.values())
    return {k for k, v in present.items() if v == best}


def _phase_cells(blocks: dict[str, dict[str, Any]], key: str, *, higher: bool) -> list[str]:
    rates: dict[str, float | None] = {}
    texts: dict[str, str] = {}
    for sid in SYSTEM_IDS:
        b = blocks.get(sid) or {"n": 0, "true_n": {}}
        n = int(b.get("n") or 0)
        k = int((b.get("true_n") or {}).get(key) or 0)
        rate = b.get(key)
        rates[sid] = None if not n else rate
        texts[sid] = _cell(rate, k, n)
    win = _winner_ids(rates, higher=higher)
    return [_underline(texts[sid]) if sid in win else texts[sid] for sid in SYSTEM_IDS]


def _score_row(rec: dict[str, Any], catalog: dict[str, tuple[str, ...]], parents: dict[str, Any]) -> dict[str, Any]:
    true = str(rec.get("true_label") or "").strip().upper()
    pred = str(rec.get("predicted_label") or "").strip().upper()
    format_error = bool(rec.get("format_error"))
    empty_kb = bool(rec.get("empty_kb"))
    plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
    acts = [_norm_action(a) for a in (rec.get("actions") or _plan_actions(plan)) if _norm_action(a)]
    primary = _primary_action(plan, acts)
    w_true = {_norm_action(a) for a in catalog.get(true, ())}
    w_pred = {_norm_action(a) for a in catalog.get(pred, ())}
    unsafe_set = {_norm_action(a) for a in not_allowed_actions_for_type(true)}
    tiers = _plan_tiers(plan)
    bad_tier = any(t not in VALID_TIERS for t in tiers)
    catalog_violation = any(a not in w_pred for a in acts)
    action_correct = (not format_error) and bool(primary) and primary in w_true
    policy_compliant = (not format_error) and bool(acts) and (not catalog_violation) and (not bad_tier)
    unsafe_action = any(a in unsafe_set for a in acts)
    parent_text = _parent_blob(parents, list(rec.get("prompt_parent_ids") or []))
    return {
        "split_index": rec.get("split_index"),
        "true_label": true,
        "predicted_label": pred,
        "detect_match": int(true == pred and bool(true)),
        "primary_action": primary,
        "actions": acts,
        "format_error": int(format_error),
        "empty_kb": int(empty_kb),
        "catalog_violation": int(catalog_violation),
        "invalid_tier": int(bad_tier),
        "action_correct": int(action_correct),
        "policy_compliant": int(policy_compliant),
        "unsafe_action": int(unsafe_action),
        "evidence_support": int(_evidence_support(true, acts, parent_text, empty_kb)),
        "retrieve_ms": (rec.get("latency") or {}).get("retrieve_ms"),
        "rank_ms": (rec.get("latency") or {}).get("rank_ms"),
        "llm_ms": (rec.get("latency") or {}).get("llm_ms"),
    }


def run_reason(
    *,
    limit: int | None = None,
    reset_live: bool = False,
    systems: list[str] | None = None,
) -> Path:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (backend/.env)")
    want = [s for s in (systems or list(SYSTEM_IDS)) if s in SYSTEM_IDS]
    if not want:
        raise SystemExit(f"No systems selected. Use one of: {', '.join(SYSTEM_IDS)}")
    specs = [s for s in SYSTEMS if s["id"] in want]
    paired, _manifest = _pair_rows()
    if limit:
        paired = paired[: int(limit)]
    live = _live_dir(reset=reset_live)
    out = _jsonl_path(live)
    done = _done_keys(out)
    remain = sum(
        1
        for spec in specs
        for p in paired
        if (int(p["sample"]["split_index"]), spec["id"]) not in done
    )
    print(
        f"Reason A/B/C: systems={want} already={len(done)} remain={remain} "
        f"(flows={len(paired)} × systems={len(specs)})"
    )

    attack_actions, agentic_features = load_attack_and_agentic(verbose=False)
    need_rag = any(s["rag"] for s in specs)
    vector_store = load_vector_store(VECTOR_STORE_DIR) if need_rag else None
    if need_rag:
        reason_mod.vector_store = vector_store
        reason_mod._RAG_PARENTS = load_parent_store(VECTOR_STORE_DIR)
    reason_mod.predictions_data = [{"_reason500": True}]
    reason_mod.attack_actions_data = attack_actions
    reason_mod.agentic_features_data = agentic_features

    n_flow = len(paired)
    for spec in specs:
        sid = spec["id"]
        for i, item in enumerate(paired, 1):
            sample = item["sample"]
            six = int(sample["split_index"])
            if (six, sid) in done:
                continue
            pred = str(sample.get("predicted_label") or "")
            true = str(sample.get("true_label") or item["flow"].get("label_simplified") or "")
            print(f"  [{sid} {i}/{n_flow}] split={six} true={true} pred={pred} …", flush=True)
            t_all = time.perf_counter()
            query = reason_mod.build_template_rag_query(sample)
            retrieve_ms = None
            rank_ms = None
            prompt_secs: list[dict[str, Any]] = []
            ir_secs: list[dict[str, Any]] = []
            if spec["rag"]:
                prompt_secs, ir_secs, retrieve_ms, rank_ms = _retrieve(
                    vector_store,
                    query,
                    rank=bool(spec["rank"]),
                    sample=sample,
                )
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
            llm_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            actions = _plan_actions(plan if isinstance(plan, dict) else None)
            rec = {
                "split_index": six,
                "reason_row": sample.get("reason_row"),
                "system": sid,
                "true_label": true,
                "predicted_label": pred,
                "confidence": sample.get("confidence"),
                "query": query,
                "retrieved_parent_ids": [str(s.get("parent_id") or "") for s in ir_secs if s.get("parent_id")],
                "prompt_parent_ids": [str(s.get("parent_id") or "") for s in prompt_secs if s.get("parent_id")],
                "retrieved_sources": [str(s.get("source_file") or "") for s in prompt_secs],
                "plan": plan,
                "actions": actions,
                "format_error": format_error,
                "empty_kb": bool(spec["rag"]) and not bool(prompt_secs),
                "latency": {
                    "retrieve_ms": retrieve_ms,
                    "rank_ms": rank_ms,
                    "llm_ms": llm_ms,
                    "total_ms": round((time.perf_counter() - t_all) * 1000.0, 1),
                },
                "usage": usage,
                "raw_chars": len(raw),
            }
            _append_jsonl(out, rec)
            done.add((six, sid))
    print(f"runs.jsonl → {out} ({len(done)} rows)")
    return live


def _coverage_block(recs: list[dict[str, Any]], catalog: dict[str, tuple[str, ...]], label_key: str) -> dict[str, Any]:
    by: dict[str, list[dict[str, Any]]] = {c: [] for c in NINE}
    for rec in recs:
        lab = str(rec.get(label_key) or "").strip().upper()
        if lab in by:
            by[lab].append(rec)
        else:
            by.setdefault(lab or "UNKNOWN", []).append(rec)

    out: dict[str, Any] = {}
    for cls in list(NINE) + [k for k in by if k not in NINE]:
        rows = by.get(cls) or []
        allowed = [_norm_action(a) for a in catalog.get(cls, ())]
        allowed_set = set(allowed)
        allowed_orig = list(catalog.get(cls, ()))
        counts: Counter[str] = Counter()
        extra: Counter[str] = Counter()
        n_parse = 0
        n_empty = 0
        n_violate = 0
        n_bad_tier = 0
        for rec in rows:
            if rec.get("format_error"):
                n_parse += 1
            if rec.get("empty_kb"):
                n_empty += 1
            plan = rec.get("plan") if isinstance(rec.get("plan"), dict) else None
            acts = [_norm_action(a) for a in (rec.get("actions") or _plan_actions(plan))]
            if any(a and a not in allowed_set for a in acts):
                n_violate += 1
            for a in acts:
                if not a:
                    continue
                if a in allowed_set:
                    counts[a] += 1
                else:
                    extra[a] += 1
            for tier in _plan_tiers(plan):
                if tier not in VALID_TIERS:
                    n_bad_tier += 1
                    break
        covered = [a for a in allowed_orig if _norm_action(a) in counts]
        missing = [a for a in allowed_orig if _norm_action(a) not in counts]
        n_cat = len(allowed_orig)
        out[cls] = {
            "n_plans": len(rows),
            "catalog": allowed_orig,
            "catalog_n": n_cat,
            "covered": covered,
            "covered_n": len(covered),
            "missing": missing,
            "missing_n": len(missing),
            "coverage": (len(covered) / n_cat) if n_cat else None,
            "action_counts": {a: counts[_norm_action(a)] for a in allowed_orig},
            "off_catalog": dict(extra),
            "format_errors": n_parse,
            "empty_kb": n_empty,
            "catalog_violations": n_violate,
            "invalid_tier_plans": n_bad_tier,
        }
    return out


def _class_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        for act in row.get("actions") or []:
            if act:
                counts[str(act)] += 1
    return {
        "n": len(rows),
        "action_correct": _rate(rows, "action_correct"),
        "policy_compliant": _rate(rows, "policy_compliant"),
        "unsafe_action": _rate(rows, "unsafe_action"),
        "evidence_support": _rate(rows, "evidence_support"),
        "detect_match": _rate(rows, "detect_match"),
        "format_error": _rate(rows, "format_error"),
        "empty_kb": _rate(rows, "empty_kb"),
        "catalog_violation": _rate(rows, "catalog_violation"),
        "true_n": {
            "detect_match": sum(int(r.get("detect_match") or 0) for r in rows),
            "action_correct": sum(int(r.get("action_correct") or 0) for r in rows),
            "policy_compliant": sum(int(r.get("policy_compliant") or 0) for r in rows),
            "unsafe_action": sum(int(r.get("unsafe_action") or 0) for r in rows),
            "evidence_support": sum(int(r.get("evidence_support") or 0) for r in rows),
        },
        "actions_used": dict(counts),
    }


def write_report(*, live: Path | None = None) -> Path:
    live = live or _live_dir(reset=False)
    recs = _read_jsonl(_jsonl_path(live))
    catalog = load_attack_actions_by_type()
    man_path = _flows_csv().parent / "manifest.json"
    fixture_man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.is_file() else {}
    parents = load_parent_store(VECTOR_STORE_DIR)

    scored = [_score_row(rec, catalog, parents) for rec in recs]
    for rec, row in zip(recs, scored):
        row["system"] = str(rec.get("system") or "RAG_RANKING")

    by_sys_rows: dict[str, list[dict[str, Any]]] = {sid: [] for sid in SYSTEM_IDS}
    for row in scored:
        sid = str(row.get("system") or "")
        by_sys_rows.setdefault(sid, []).append(row)

    by_true_sys: dict[str, dict[str, list[dict[str, Any]]]] = {
        cls: {sid: [] for sid in SYSTEM_IDS} for cls in NINE
    }
    for row in scored:
        lab = str(row.get("true_label") or "")
        sid = str(row.get("system") or "")
        if lab in by_true_sys and sid in by_true_sys[lab]:
            by_true_sys[lab][sid].append(row)

    n_by_sys = {sid: len(by_sys_rows.get(sid) or []) for sid in SYSTEM_IDS}
    n = len(scored)
    parse_n = sum(int(r.get("format_error") or 0) for r in scored)
    empty_n = sum(int(r.get("empty_kb") or 0) for r in scored)
    llm_ms = [float(r.get("llm_ms") or 0) for r in scored]
    tokens = sum(int((r.get("usage") or {}).get("total_tokens") or 0) for r in recs)

    overall_sys = {sid: _class_block(by_sys_rows.get(sid) or []) for sid in SYSTEM_IDS}
    by_true = {
        cls: {sid: _class_block(by_true_sys[cls].get(sid) or []) for sid in SYSTEM_IDS} for cls in NINE
    }

    def _primary_share(rows: list[dict[str, Any]], action: str) -> tuple[float | None, int, int]:
        n_rows = len(rows)
        if not n_rows:
            return None, 0, 0
        k = sum(1 for r in rows if _norm_action(str(r.get("primary_action") or "")) == action)
        return k / n_rows, k, n_rows

    action_tables: dict[str, dict[str, Any]] = {}
    for cls in NINE:
        allowed = [_norm_action(a) for a in catalog.get(cls, ()) if _norm_action(a)]
        extras: set[str] = set()
        for sid in SYSTEM_IDS:
            for row in by_true_sys[cls].get(sid) or []:
                p = _norm_action(str(row.get("primary_action") or ""))
                if p and p not in allowed:
                    extras.add(p)
        names = list(allowed) + sorted(extras)
        rows_out: list[dict[str, Any]] = []
        for act in names:
            shares = {}
            for sid in SYSTEM_IDS:
                rate, k, nn = _primary_share(by_true_sys[cls].get(sid) or [], act)
                shares[sid] = {"rate": rate, "k": k, "n": nn}
            rows_out.append({"action": act, "in_catalog": act in allowed, "systems": shares})
        action_tables[cls] = {"catalog": list(catalog.get(cls, ())), "rows": rows_out}

    winners: dict[str, dict[str, list[str]]] = {}
    for name, blocks in [("Overall", overall_sys)] + [(cls, by_true[cls]) for cls in NINE]:
        winners[name] = {}
        for ph in PHASES:
            rates = {sid: (blocks.get(sid) or {}).get(ph["key"]) for sid in SYSTEM_IDS}
            nmap = {sid: int((blocks.get(sid) or {}).get("n") or 0) for sid in SYSTEM_IDS}
            rates = {sid: (None if not nmap[sid] else rates[sid]) for sid in SYSTEM_IDS}
            winners[name][ph["key"]] = sorted(_winner_ids(rates, higher=ph["higher"]))

    summaries = live / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    quality_fields = [
        "system",
        "split_index",
        "true_label",
        "predicted_label",
        "detect_match",
        "primary_action",
        "action_correct",
        "policy_compliant",
        "unsafe_action",
        "format_error",
        "empty_kb",
        "catalog_violation",
        "invalid_tier",
        "n_actions",
        "actions",
        "retrieve_ms",
        "rank_ms",
        "llm_ms",
    ]
    with (summaries / "quality.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=quality_fields)
        w.writeheader()
        for row in scored:
            w.writerow(
                {
                    **{k: row.get(k) for k in quality_fields if k not in {"n_actions", "actions"}},
                    "n_actions": len(row.get("actions") or []),
                    "actions": " | ".join(row.get("actions") or []),
                }
            )

    mitigation = {
        "systems": list(SYSTEM_IDS),
        "labels": dict(SYSTEM_LABEL),
        "n": n,
        "n_target": _n_target(),
        "n_by_system": n_by_sys,
        "llm_calls_target": _n_target() * len(SYSTEM_IDS),
        "disjoint_from_gold": True,
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "metrics": {
            "action_correct": "↑ primary action in W[true]",
            "policy_compliant": "↑ parsed + all actions in W[pred] + valid tiers",
            "unsafe_action": "↓ any action illegal for true class",
        },
        "overall": overall_sys,
        "by_true_label": by_true,
        "actions_by_true_label": action_tables,
        "winners": winners,
    }
    _dump(live / "mitigation.json", mitigation)
    _dump(summaries / "mitigation.json", mitigation)

    mit_fields = ["attack", "phase", "direction"] + list(SYSTEM_IDS) + ["winner"]
    with (summaries / "mitigation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mit_fields)
        w.writeheader()
        for name, blocks in [("Overall", overall_sys)] + [(cls, by_true[cls]) for cls in NINE]:
            for ph in PHASES:
                row = {
                    "attack": name,
                    "phase": ph["title"],
                    "direction": "higher" if ph["higher"] else "lower",
                    "winner": ", ".join(SYSTEM_LABEL[s] for s in winners[name][ph["key"]]) or "—",
                }
                for sid in SYSTEM_IDS:
                    row[sid] = (blocks.get(sid) or {}).get(ph["key"])
                w.writerow(row)

    by_pred = _coverage_block(recs, catalog, "predicted_label")
    coverage_payload = {
        "systems": list(SYSTEM_IDS),
        "n_runs": n,
        "n_target": _n_target(),
        "n_by_system": n_by_sys,
        "llm_calls": n,
        "model": os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        "disjoint_from_gold": True,
        "excluded_gold_n": len(fixture_man.get("excluded_gold_split_index") or []),
        "quality": {
            "parse_error_rate": (parse_n / n) if n else None,
            "empty_kb_rate": (empty_n / n) if n else None,
            "mean_llm_ms": (sum(llm_ms) / len(llm_ms)) if llm_ms else None,
            "total_tokens": tokens,
        },
        "aevra_catalog": {k: list(v) for k, v in catalog.items()},
        "by_predicted_label": by_pred,
    }
    _dump(live / "coverage.json", coverage_payload)
    _dump(summaries / "coverage.json", coverage_payload)

    head = " | ".join(["Phase"] + [SYSTEM_LABEL[s] for s in SYSTEM_IDS])
    rule = "|".join(["------"] + [":------:" for _ in SYSTEM_IDS])
    model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    counts = ", ".join(f"{SYSTEM_LABEL[s]} {n_by_sys[s]}/{_n_target()}" for s in SYSTEM_IDS)

    def _phase_section(title: str, blocks: dict[str, dict[str, Any]], detect_rows: list[dict[str, Any]]) -> list[str]:
        n_flow = max((int(b.get("n") or 0) for b in blocks.values()), default=0)
        det = _class_block(detect_rows) if detect_rows else _class_block([])
        det_n = int(det["n"] or 0)
        det_cell = _cell(det["detect_match"], det["true_n"]["detect_match"], det_n)
        out = [
            f"### {title}",
            "",
            f"n = {n_flow} flows · detect match {det_cell} (same detect for all systems)",
            "",
            f"| {head} |",
            f"|{rule}|",
        ]
        for ph in PHASES:
            cells = _phase_cells(blocks, ph["key"], higher=ph["higher"])
            arrow = "↑" if ph["higher"] else "↓"
            out.append(f"| {ph['title']} {arrow} | " + " | ".join(cells) + " |")
        return out

    # Detect match is identical across systems; use the first system that has rows.
    def _detect_rows_for(cls: str | None) -> list[dict[str, Any]]:
        if cls is None:
            for sid in SYSTEM_IDS:
                if by_sys_rows.get(sid):
                    return by_sys_rows[sid]
            return []
        for sid in SYSTEM_IDS:
            rows = by_true_sys[cls].get(sid) or []
            if rows:
                return rows
        return []

    lines = [
        "# Mitigation — No RAG vs RAG vs Ranked RAG (disjoint from gold-100)",
        "",
        "Catalog-legal responses at scale. Policy PDFs do not name AEVRA actions or tiers, so this table scores the closed catalog, not PDF lexical overlap.",
        "",
        f"Model `{model}` · {counts} · parse errors {parse_n}/{n}",
        "",
        "Underlined cell = better system for that phase (ties underlined together).",
        "",
        "| Phase | Want | Meaning |",
        "|-------|------|---------|",
        "| Action correctness ↑ | high | primary action ∈ W[true] |",
        "| Policy compliance ↑ | high | parsed; all actions ∈ W[predicted]; valid project tier |",
        "| Unsafe action rate ↓ | low | any action illegal for the true class |",
        "",
        "## Phases",
        "",
    ]
    lines += _phase_section("Overall", overall_sys, _detect_rows_for(None))
    for cls in NINE:
        lines += [""] + _phase_section(cls, by_true[cls], _detect_rows_for(cls))

    lines += [
        "",
        "## Actions within each attack type",
        "",
        "Share of flows whose **primary** action is that catalog slot. Extra (off-catalog) primaries are listed last.",
        "",
    ]
    act_head = " | ".join(["Action"] + [SYSTEM_LABEL[s] for s in SYSTEM_IDS])
    act_rule = "|".join(["------"] + [":------:" for _ in SYSTEM_IDS])
    for cls in NINE:
        table = action_tables[cls]
        n_flow = max((int(by_true[cls][sid]["n"] or 0) for sid in SYSTEM_IDS), default=0)
        if not n_flow:
            continue
        lines += [f"### {cls} — W[y] = {', '.join(table['catalog']) or '—'}", "", f"| {act_head} |", f"|{act_rule}|"]
        for row in table["rows"]:
            cells = []
            rates = {sid: row["systems"][sid]["rate"] for sid in SYSTEM_IDS}
            win = _winner_ids(rates, higher=True)
            if not any((rates[sid] or 0) > 0 for sid in SYSTEM_IDS):
                win = set()
            for sid in SYSTEM_IDS:
                sh = row["systems"][sid]
                text = _cell(sh["rate"], sh["k"], sh["n"])
                cells.append(_underline(text) if sid in win else text)
            mark = "" if row["in_catalog"] else " *(off-catalog)*"
            lines.append(f"| {row['action']}{mark} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += [
        "## Catalog coverage (appendix)",
        "",
        "By predicted label (union across systems).",
        "",
        "| Attack | n | W[y] | Covered | Missing |",
        "|--------|--:|-----:|--------:|---------|",
    ]
    for cls in NINE:
        b = by_pred[cls]
        if not b["n_plans"]:
            continue
        missing = ", ".join(b["missing"]) if b["missing"] else "—"
        lines.append(
            f"| {cls} | {b['n_plans']} | {b['catalog_n']} | {b['covered_n']}/{b['catalog_n']} | {missing} |"
        )

    lines += [
        "",
        "Artifacts: `runs.jsonl`, `mitigation.json`, `summaries/mitigation.csv`, `coverage.json`.",
        "",
    ]
    report = live / "report.md"
    _write(report, "\n".join(lines) + "\n")
    _write(live / "README.md", "\n".join(lines[:14]) + f"\nFull report: `{report.name}`\n")
    _dump(
        live / "manifest.json",
        {
            "systems": list(SYSTEM_IDS),
            "n_runs": n,
            "n_by_system": n_by_sys,
            "n_target": _n_target(),
            "llm_calls": n,
            "llm_calls_target": _n_target() * len(SYSTEM_IDS),
            "fixture": str(_flows_csv()),
            "fixture_sha256": fixture_man.get("flows_sha256"),
            "runs_sha256": hashlib.sha256(_jsonl_path(live).read_bytes()).hexdigest() if _jsonl_path(live).is_file() else None,
            "mitigation_json": str(live / "mitigation.json"),
            "coverage_json": str(live / "coverage.json"),
            "report": str(report),
        },
    )
    print(f"mitigation.json + report.md → {live}")
    return report


def _bind_input(input_raw: str) -> Path:
    flows = resolve_input_csv(input_raw)
    n_rows = configure(flows)
    print(f"input {flows} n={n_rows} live={_live_dir(reset=False)}")
    return flows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="A/B/C reason scorecard. N = row count of --input (folder or flows.csv)."
    )
    p.add_argument(
        "--input",
        default="",
        help="Fixture folder or flows.csv (default: experiments/data/rag_reason_500). N = row count.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=0,
        help="Only with --sample-only: how many rows to write into --input (does not select a preset set)",
    )
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--predict-only", action="store_true")
    p.add_argument("--reason-only", action="store_true", help="Reason then report (skip sample/detect)")
    p.add_argument("--report-only", action="store_true", help="Rebuild report from existing runs.jsonl")
    p.add_argument("--reset-live", action="store_true", help="Archive the live folder for this input set")
    p.add_argument("--limit", type=int, default=0, help="Reason only the first N paired rows (smoke)")
    p.add_argument("--chunk-size", type=int, default=DETECT_CHUNK)
    p.add_argument(
        "--systems",
        default=",".join(SYSTEM_IDS),
        help="Comma-separated system ids: LLM_only,RAG_No_Ranking,RAG_RANKING",
    )
    args = p.parse_args(argv)
    _force_utf8_stdio()
    systems = [x.strip() for x in args.systems.split(",") if x.strip()]
    for s in systems:
        if s not in SYSTEM_IDS:
            raise SystemExit(f"unknown system {s}; use {', '.join(SYSTEM_IDS)}")

    only = [args.sample_only, args.predict_only, args.reason_only, args.report_only]
    if sum(bool(x) for x in only) > 1:
        raise SystemExit("Pick at most one of --sample-only / --predict-only / --reason-only / --report-only")

    if args.sample_only:
        want = int(args.n)
        if want <= 0:
            raise SystemExit("--sample-only requires --n (row count to write)")
        if not (args.input or "").strip():
            raise SystemExit("--sample-only requires --input (folder for flows.csv)")
        dest_dir = Path(args.input.strip())
        if not dest_dir.is_absolute():
            dest_dir = (_REPO / args.input.strip()).resolve() if not dest_dir.exists() else dest_dir.resolve()
        dest = dest_dir / "flows.csv" if dest_dir.suffix.lower() != ".csv" else dest_dir
        if dest.suffix.lower() == ".csv":
            dest_dir = dest.parent
            dest = dest_dir / "flows.csv"
        if dest.is_file():
            print(f"Fixture exists: {dest} (not resampled)")
            return 0
        sample_fixture(want, dest_dir)
        return 0

    if args.n:
        raise SystemExit("N comes from the input CSV. Use --input PATH. --n is only valid with --sample-only.")

    _bind_input(args.input)
    if args.predict_only:
        run_detect(chunk_size=args.chunk_size)
        return 0
    if args.reason_only:
        run_reason(limit=args.limit or None, reset_live=args.reset_live, systems=systems)
        write_report()
        return 0
    if args.report_only:
        write_report()
        return 0

    run_detect(chunk_size=args.chunk_size)
    run_reason(limit=args.limit or None, reset_live=args.reset_live, systems=systems)
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
