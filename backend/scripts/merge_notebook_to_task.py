#!/usr/bin/env python3
"""
Merge selected code cells from a Jupyter notebook into a standalone .py under app/notebook_runtime/tasks/.
Run from backend/: python scripts/merge_notebook_to_task.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
NOTEBOOKS = BACKEND / "notebooks"
TASKS = BACKEND / "app" / "notebook_runtime" / "tasks"


def merge_cells(nb_path: Path, *, only_indices: list[int] | None = None) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        if only_indices is not None and i not in only_indices:
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        parts.append(src)
    return "\n\n".join(parts)


def transform_source(src: str, *, header: str) -> str:
    src = src.replace("from utils.", "from app.notebook_runtime.")
    src = src.replace("import utils.", "import app.notebook_runtime.")
    # Notebook IPython display
    src = re.sub(r"^get_ipython\(\)\.run_line_magic\([^)]+\)\s*\n?", "", src, flags=re.MULTILINE)
    return header + src


def write_task(name: str, src: str) -> Path:
    TASKS.mkdir(parents=True, exist_ok=True)
    out = TASKS / name
    out.write_text(src, encoding="utf-8")
    return out


def main() -> None:
    # RAG Part 2: notebook cells 1–4 are code (after markdown cell 0)
    p2 = NOTEBOOKS / "RAG_part2_agent_actions.ipynb"
    s2 = transform_source(
        merge_cells(p2, only_indices=[1, 2, 3, 4]),
        header='"""Standalone RAG Part 2 (merged from RAG_part2_agent_actions.ipynb)."""\n',
    )
    write_task("rag_part2_runner.py", s2)
    print("Wrote rag_part2_runner.py", len(s2.splitlines()), "lines")

    vfl = NOTEBOOKS / "VFL_SHAP_MultiClass.ipynb"
    sv = transform_source(
        merge_cells(vfl, only_indices=None),
        header='"""Standalone VFL SHAP multiclass (merged from VFL_SHAP_MultiClass.ipynb)."""\n',
    )
    write_task("vfl_shap_multiclass_runner.py", sv)
    print("Wrote vfl_shap_multiclass_runner.py", len(sv.splitlines()), "lines")

    pred = NOTEBOOKS / "VFL_SHAP_Prediction.ipynb"
    sp = transform_source(
        merge_cells(pred, only_indices=None),
        header='"""Standalone VFL SHAP prediction (merged from VFL_SHAP_Prediction.ipynb)."""\n',
    )
    write_task("vfl_shap_prediction_runner.py", sp)
    print("Wrote vfl_shap_prediction_runner.py", len(sp.splitlines()), "lines")

    score = NOTEBOOKS / "Scoring_Evaluation.ipynb"
    # Skip leading markdown-only; merge all code cells
    ss = transform_source(
        merge_cells(score, only_indices=None),
        header='"""Standalone scoring evaluation (merged from Scoring_Evaluation.ipynb)."""\n',
    )
    write_task("scoring_evaluation_runner.py", ss)
    print("Wrote scoring_evaluation_runner.py", len(ss.splitlines()), "lines")


if __name__ == "__main__":
    main()
