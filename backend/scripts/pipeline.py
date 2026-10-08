#!/usr/bin/env python3
"""Paper end-to-end launchers: Detect → Reason → Evaluate, plus live Commit/Apply.

Maps Fig. 1 layers to the runners in ``scripts/``. Console artifacts go to
``experiments/<stage>/`` at the repo root. Run from ``backend/``::

    python scripts/pipeline.py --list
    python scripts/pipeline.py detect-train
    python scripts/pipeline.py e2e -- --max-rows 1

Offline path (no API)::

    detect-train     scripts/detect_train.py
    detect-predict   scripts/detect_predict.py
    rag-index        scripts/rag_build.py
    reason           scripts/reason.py
    evaluate         scripts/evaluate.py

Live API path (Commit + Apply on Hardhat)::

    e2e              run/attack_monitor.py
"""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO = _BACKEND.parent

STAGES: dict[str, dict[str, str]] = {
    "detect-train": {
        "layer": "1 Detection",
        "target": "scripts/detect_train.py",
        "help": "Train three-party VFL encoders + meta-classifier",
    },
    "detect-predict": {
        "layer": "1 Detection",
        "target": "scripts/detect_predict.py",
        "help": "Predict attack class and KernelSHAP domain shares",
    },
    "rag-index": {
        "layer": "2 Reasoning",
        "target": "scripts/rag_build.py",
        "help": "Build FAISS policy index (NIST / CIS / ATT&CK corpus)",
    },
    "reason": {
        "layer": "2 Reasoning",
        "target": "scripts/reason.py",
        "help": "RAG + GPT-4o-mini structured mitigation plans",
    },
    "evaluate": {
        "layer": "1 Detection (metrics)",
        "target": "scripts/evaluate.py",
        "help": "Scoring tables (accuracy / F1 vs centralized NN)",
    },
    "e2e": {
        "layer": "1-4 Detect-Reason-Commit-Apply",
        "target": "run/attack_monitor.py",
        "help": "Online row-at-a-time HTTP pipeline (whitelist + plan-binding)",
    },
}


def _ensure_backend_on_path() -> None:
    root = str(_BACKEND)
    if root not in sys.path:
        sys.path.insert(0, root)


def _set_pipeline_scope(stage: str) -> None:
    os.environ["CHAINAGENT_SCOPE"] = "pipeline"
    os.environ["CHAINAGENT_TASK"] = stage


def _run_script(relative_under_backend: str, extra: list[str] | None = None) -> None:
    """Run a scripts/*.py file. Artifact roots are absolute (``experiments/<task>/``)."""
    _ensure_backend_on_path()
    os.chdir(_REPO)
    script = str(_BACKEND / relative_under_backend)
    sys.argv = [script, *(extra or [])]
    runpy.run_path(script, run_name="__main__")


def run_stage(stage: str, extra: list[str] | None = None) -> None:
    spec = STAGES[stage]
    target = spec["target"]
    _set_pipeline_scope(stage)
    if stage == "e2e":
        cmd = [sys.executable, str(_BACKEND / target), *(extra or [])]
        raise SystemExit(subprocess.call(cmd, cwd=str(_BACKEND)))
    _run_script(target, extra)


def _print_list() -> None:
    print("Paper pipeline stages (Fig. 1)\n")
    print(f"{'stage':<16} {'layer':<36} description")
    print("-" * 88)
    for name, spec in STAGES.items():
        print(f"{name:<16} {spec['layer']:<36} {spec['help']}")
    print("\nDirect scripts (same as stages above):")
    for name, spec in STAGES.items():
        if name == "e2e":
            continue
        print(f"  {spec['target']:<42}  {name}")
    print("\nRelated:")
    print("  scripts/demo_pipeline.py      stdlib Detect->Reason->Apply walkthrough (no chain)")
    print("  notebooks/00_pipeline.ipynb           train, SHAP, RAG, plans, scoring")
    print("  scripts/trust_anchor_benchmark.py  offline CSV->RAG->LLM->anchor timings (no API)")
    print("  scripts/test_trust_chain.py        Hardhat smoke (storePlan / getPlan / markApplied)")
    print("  run/attack_monitor.py              same as stage e2e")
    print("\nConsole artifacts: experiments/<stage>/  (API artifacts: backend/storage/)")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        extra = argv[i + 1 :]
        argv = argv[:i]

    p = argparse.ArgumentParser(
        description="Launch Detect / Reason / Evaluate / live E2E stages from the paper setup.",
    )
    p.add_argument(
        "stage",
        nargs="?",
        choices=[*STAGES.keys(), "list"],
        help="Pipeline stage (use list to print the map)",
    )
    p.add_argument("--list", action="store_true", help="Print stage map and exit")
    args = p.parse_args(argv)

    if args.list or args.stage in (None, "list"):
        _print_list()
        return

    run_stage(args.stage, extra=extra)


if __name__ == "__main__":
    main()
