"""Run detect_predict into experiments/gold-100/ (does not touch detect-predict)."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from scripts.env import FIXTURE_GOLD_100, named_live_dir, resolve_fixture_csv

LIVE = named_live_dir("gold-100", mkdir=True)
SRC_CSV = resolve_fixture_csv(FIXTURE_GOLD_100)

os.environ["CHAINAGENT_SAMPLE_CSV"] = str(SRC_CSV)
os.environ["CHAINAGENT_PREDICT_OUT"] = str(LIVE)
print(f"gold-100 results: {LIVE}")
print(f"shared input CSV: {SRC_CSV}")
print("detect-predict folder will not be written or archived")
runpy.run_path(str(_BACKEND / "scripts" / "detect_predict.py"), run_name="__main__")
