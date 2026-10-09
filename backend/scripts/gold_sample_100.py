"""Reconstruct VFL test split (seed 42) and reserve 10 held-out flows per trained class (N=90)."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.env import (
    FIXTURE_GOLD_100,
    fixture_set_dir,
    resolve_datasets_dir,
)
from scripts.vfl import simplify_label

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
SEED = 42
PER_CLASS = 10
N_GOLD = PER_CLASS * len(NINE)  # 90
MIN_ROWS = 200


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_frame() -> tuple[pd.DataFrame, list[Path]]:
    folder = resolve_datasets_dir()
    csvs = sorted(p for p in folder.glob("*.csv") if p.suffix == ".csv")
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {folder}")
    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop(columns=["Flow ID", "Src IP", "Dst IP", "Timestamp"], errors="ignore")
    df["label_simplified"] = df["label"].apply(simplify_label)
    counts = df["label_simplified"].value_counts()
    small = counts[counts < MIN_ROWS].index.tolist()
    if small:
        df.loc[df["label_simplified"].isin(small), "label_simplified"] = "OTHERS"
    unique = sorted(df["label_simplified"].unique())
    mapping = {lab: i for i, lab in enumerate(unique)}
    df["label_numeric"] = df["label_simplified"].map(mapping)
    return df, csvs


def main() -> int:
    df, csvs = load_training_frame()
    stratify = df["label_numeric"]
    trainval_idx, test_idx = train_test_split(
        range(len(df)),
        test_size=0.2,
        random_state=SEED,
        stratify=stratify,
    )
    test_idx = list(test_idx)
    test = df.iloc[test_idx].copy()
    test["split_index"] = test_idx
    missing = [c for c in NINE if c not in set(test["label_simplified"])]
    if missing:
        raise SystemExit(f"Test split missing classes: {missing}")

    # Exactly PER_CLASS rows per trained class from the test split only.
    seeds: list[int] = []
    for cls in NINE:
        pos = [i for i, lab in enumerate(test["label_simplified"]) if str(lab) == cls]
        if len(pos) < PER_CLASS:
            raise SystemExit(f"Test split has {len(pos)} {cls} rows; need {PER_CLASS}")
        take, _rest = train_test_split(pos, train_size=PER_CLASS, random_state=SEED)
        seeds.extend(list(take))
    gold = test.iloc[seeds].copy()
    gold = gold.sort_values("split_index").reset_index(drop=True)
    gold.insert(0, "gold_row", range(1, len(gold) + 1))

    out_dir = fixture_set_dir(FIXTURE_GOLD_100, mkdir=True)
    flows = out_dir / "flows.csv"
    gold.to_csv(flows, index=False)

    hist = Counter(str(x) for x in gold["label_simplified"])
    manifest = {
        "dataset_paths": [str(p) for p in csvs],
        "dataset_sha256": {p.name: _sha256_file(p) for p in csvs},
        "seed": SEED,
        "split": {
            "trainval": len(trainval_idx),
            "test": len(test_idx),
            "n_total": len(df),
        },
        "n_gold": len(gold),
        "selected_split_index": [int(x) for x in gold["split_index"]],
        "class_counts_test": dict(Counter(str(x) for x in test["label_simplified"])),
        "class_counts_gold": dict(sorted(hist.items())),
        "flows_csv": str(flows),
        "flows_sha256": _sha256_file(flows),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {flows} ({len(gold)} rows)")
    print("gold histogram:", dict(sorted(hist.items())))
    print("test size:", len(test_idx), "trainval:", len(trainval_idx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
