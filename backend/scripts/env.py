"""Project paths, ``backend/.env`` loading, and sys.path setup for notebooks/scripts."""

from __future__ import annotations

import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
STORAGE_DIR = BACKEND_ROOT / "storage"
BACKEND_STORAGE = STORAGE_DIR
ATTACK_OPTIONS_JSON = STORAGE_DIR / "attack_options.json"
AGENTIC_FEATURES_JSON = STORAGE_DIR / "agentic_features.json"
DATASETS_DIR = REPO_ROOT / "datasets"
SAMPLE_CSV = BACKEND_ROOT / "run" / "data" / "sample.csv"

EXPERIMENT_TASKS = (
    "detect-train",
    "detect-predict",
    "rag-index",
    "reason",
    "evaluate",
    "e2e",
)

# Legacy alias; console predict writes go to experiment_dir("detect-predict").
PAPER_OUTPUT_DIR = REPO_ROOT / "outputs"

RUN_DIR_PREFIX = "run_"
ARCHIVE_DIRNAME = "archive"
_RUN_STAMP_RE = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")
# Stay at the task root across runs (fixtures / indexes / placeholders).
_TASK_ROOT_KEEP = frozenset(
    {
        ".gitkeep",
        ARCHIVE_DIRNAME,
        "inputs",
        "knowledge",
        "vector_store",
        "action_plans",
    }
)
TRAIN_CHECKPOINT = "vfl_model_best.pth"
PREDICT_DETAIL_GLOB = "predictions_detailed_*.json"


def experiments_root() -> Path:
    raw = os.environ.get("EXPERIMENTS_ROOT", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (REPO_ROOT / "experiments").resolve()


def experiment_dir(task: str, *, mkdir: bool = True) -> Path:
    """Absolute ``experiments/<task>/`` folder (pipeline stage name)."""
    if task not in EXPERIMENT_TASKS:
        raise ValueError(f"Unknown experiment task {task!r}; expected one of {EXPERIMENT_TASKS}")
    path = experiments_root() / task
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def chainagent_scope() -> str:
    return (os.environ.get("CHAINAGENT_SCOPE") or "backend").strip().lower() or "backend"


def resolve_datasets_dir() -> Path:
    """First directory that contains ``*.csv`` (repo ``datasets/``, then cwd)."""
    for cand in (DATASETS_DIR, BACKEND_ROOT / "datasets", Path.cwd() / "datasets"):
        if cand.is_dir() and any(cand.glob("*.csv")):
            return cand
    return DATASETS_DIR


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_run_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith(RUN_DIR_PREFIX)


def _stamp_from_name(name: str) -> str | None:
    m = _RUN_STAMP_RE.search(name)
    return m.group(1) if m else None


def _infer_run_stamp(paths: list[Path]) -> str:
    stamps = [s for p in paths if (s := _stamp_from_name(p.name))]
    return max(stamps) if stamps else run_stamp()


def _run_has_marker(run: Path, marker: str) -> bool:
    if any(ch in marker for ch in "*?[]"):
        return any(run.glob(marker))
    return (run / marker).is_file()


def live_run_dirs(task_dir: Path) -> list[Path]:
    """``run_*`` folders at the task root (not under ``archive/``), newest name first."""
    if not task_dir.is_dir():
        return []
    return sorted(
        (p for p in task_dir.iterdir() if is_run_dir(p)),
        key=lambda p: p.name,
        reverse=True,
    )


def wrap_flat_run_files(task_dir: Path) -> Path | None:
    """Move loose artifacts at the task root into ``run_<stamp>/`` (once)."""
    if not task_dir.is_dir():
        return None
    loose: list[Path] = []
    for p in task_dir.iterdir():
        if p.name in _TASK_ROOT_KEEP or p.name.startswith("."):
            continue
        if is_run_dir(p):
            continue
        loose.append(p)
    if not loose:
        return None
    dest = task_dir / f"{RUN_DIR_PREFIX}{_infer_run_stamp(loose)}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in loose:
        target = dest / p.name
        if target.exists():
            continue
        shutil.move(str(p), str(target))
    return dest


def latest_run_dir(task: str, *, marker: str | None = None, mkdir: bool = False) -> Path | None:
    """Newest live ``run_*`` (optionally requiring ``marker``). Wraps leftover flat files first."""
    task_dir = experiment_dir(task, mkdir=mkdir)
    wrap_flat_run_files(task_dir)
    for run in live_run_dirs(task_dir):
        if marker is None or _run_has_marker(run, marker):
            return run
    return None


def new_run_dir(task: str) -> Path:
    """Create ``experiments/<task>/run_<timestamp>/`` for this invocation."""
    task_dir = experiment_dir(task, mkdir=True)
    wrap_flat_run_files(task_dir)
    path = task_dir / f"{RUN_DIR_PREFIX}{run_stamp()}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def archive_older_runs(task: str, *, keep: Path) -> list[Path]:
    """Move every other live ``run_*`` into ``experiments/<task>/archive/`` (complete folder)."""
    task_dir = experiment_dir(task, mkdir=True)
    archive = task_dir / ARCHIVE_DIRNAME
    keep_res = keep.resolve()
    moved: list[Path] = []
    for run in live_run_dirs(task_dir):
        if run.resolve() == keep_res:
            continue
        archive.mkdir(parents=True, exist_ok=True)
        dest = archive / run.name
        if dest.exists():
            dest = archive / f"{run.name}_{run_stamp()}"
        shutil.move(str(run), str(dest))
        moved.append(dest)
    return moved


def resolve_model_dir(*, require_checkpoint: bool = False) -> Path:
    """Latest detect-train run with ``vfl_model_best.pth``, then leftover ``backend/storage/models``."""
    train_root = experiment_dir("detect-train", mkdir=not require_checkpoint)
    wrap_flat_run_files(train_root)
    latest = latest_run_dir("detect-train", marker=TRAIN_CHECKPOINT, mkdir=False)
    candidates = tuple(
        p
        for p in (
            latest,
            train_root,
            STORAGE_DIR / "models",
            REPO_ROOT / "model",
            BACKEND_ROOT / "model",
            Path.cwd() / "model",
        )
        if p is not None
    )
    for cand in candidates:
        if (cand / TRAIN_CHECKPOINT).is_file():
            return cand
    if require_checkpoint:
        searched = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "VFL checkpoint vfl_model_best.pth not found. Looked in: "
            f"{searched}. Train first with: python scripts/pipeline.py detect-train"
        )
    train_root.mkdir(parents=True, exist_ok=True)
    return train_root


def resolve_latest_predict_dir(*, require: bool = False) -> Path:
    """Latest detect-predict ``run_*`` with detailed JSON. Falls back to the task root."""
    root = experiment_dir("detect-predict", mkdir=not require)
    wrap_flat_run_files(root)
    latest = latest_run_dir("detect-predict", marker=PREDICT_DETAIL_GLOB, mkdir=False)
    if latest is not None:
        return latest
    if any(root.glob(PREDICT_DETAIL_GLOB)):
        return root
    if require:
        raise FileNotFoundError(
            f"No {PREDICT_DETAIL_GLOB} under {root} (live run_* or task root). "
            "Run: python scripts/pipeline.py detect-predict"
        )
    return root


def resolve_sample_csv() -> Path:
    """Prediction CSV: one-row-per-class fixture, then ``backend/run/data/sample.csv``."""
    candidates = (
        experiments_root() / "detect-predict" / "inputs" / "all_attack_types.csv",
        BACKEND_ROOT / "run" / "data" / "sample_all_attack_types.csv",
        SAMPLE_CSV,
        REPO_ROOT / "inputs" / "sample.csv",
        Path.cwd() / "inputs" / "sample.csv",
        DATASETS_DIR / "sample.csv",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    searched = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Sample CSV not found. Looked in: {searched}")


def _dir_has_rag_sources(path: Path) -> bool:
    """True when a folder contains indexable ``*.pdf`` or ``*.json`` (empty mkdir does not count)."""
    if not path.is_dir():
        return False
    return any(path.glob("*.pdf")) or any(path.glob("*.json"))


def resolve_rag_knowledge_dir() -> Path:
    """CLI corpus: rag-index/knowledge, else experiments/knowledge, else storage/base_docs."""
    override = experiment_dir("rag-index") / "knowledge"
    override.mkdir(parents=True, exist_ok=True)
    staged = experiments_root() / "knowledge"
    base = STORAGE_DIR / "base_docs"
    for cand in (override, staged, base):
        if _dir_has_rag_sources(cand):
            return cand
    return override


def find_backend_root() -> Path:
    here = Path.cwd().resolve()
    candidates = [
        here,
        here.parent,
        here / "backend",
        here.parent / "backend",
        BACKEND_ROOT,
    ]
    for cand in candidates:
        if (cand / "app" / "main.py").is_file() and (cand / "scripts").is_dir():
            return cand
    raise FileNotFoundError("Could not find backend/ (expected app/main.py + scripts/)")


def ensure_backend_on_sys_path() -> Path:
    root = find_backend_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root


def load_project_dotenv() -> None:
    ensure_backend_on_sys_path()
    cwd = Path.cwd().resolve()
    for path in (
        cwd / "backend" / ".env",
        cwd / ".env",
        cwd.parent / ".env",
        cwd.parent / "backend" / ".env",
        BACKEND_ROOT / ".env",
    ):
        if path.is_file():
            load_dotenv(path)
            break
    hf = STORAGE_DIR / "hf_home"
    hf.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf))
