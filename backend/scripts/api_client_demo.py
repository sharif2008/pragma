#!/usr/bin/env python3
"""
Minimal HTTP client examples (requires running API: uvicorn from backend/).

  cd backend
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument(
        "--train-csv",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "samples" / "sample_train.csv",
    )
    args = ap.parse_args()
    base = args.base.rstrip("/")

    with httpx.Client(timeout=120.0) as client:
        r = client.get(f"{base}/health")
        print("health:", r.json())

        files = {"file": (args.train_csv.name, args.train_csv.read_bytes(), "text/csv")}
        r = client.post(f"{base}/datasets/upload", files=files)
        r.raise_for_status()
        ds = r.json()
        print("dataset upload:", ds)
        dataset_id = ds["public_id"]

        r = client.post(
            f"{base}/training/start",
            json={
                "dataset_file_public_id": dataset_id,
                "target_column": "label",
                "algorithm": "random_forest",
                "test_size": 0.25,
            },
        )
        r.raise_for_status()
        job = r.json()
        print("training start:", job)
        job_id = job["job_public_id"]

        import time

        for _ in range(60):
            r = client.get(f"{base}/training/{job_id}")
            r.raise_for_status()
            st = r.json()
            if st["status"] in ("completed", "failed"):
                print("training done:", json.dumps(st, indent=2, default=str))
                break
            time.sleep(1)
        else:
            print("timeout waiting for training")
            return

        r = client.get(f"{base}/models")
        r.raise_for_status()
        models = r.json()
        print("models:", json.dumps(models, indent=2, default=str))
        if not models:
            return
        model_pid = models[0]["public_id"]

        files = {"file": (args.train_csv.name, args.train_csv.read_bytes(), "text/csv")}
        r = client.post(f"{base}/predictions/upload-input", files=files)
        r.raise_for_status()
        pred_file = r.json()
        print("prediction input:", pred_file)

        r = client.post(
            f"{base}/predictions/start",
            json={
                "model_version_public_id": model_pid,
                "input_file_public_id": pred_file["public_id"],
                "attack_label_values": ["attack"],
            },
        )
        r.raise_for_status()
        pj = r.json()
        print("prediction job:", pj)
        pj_id = pj["public_id"]

        for _ in range(60):
            r = client.get(f"{base}/predictions/{pj_id}")
            r.raise_for_status()
            st = r.json()
            if st["status"] in ("completed", "failed"):
                print("prediction done:", json.dumps(st, indent=2, default=str))
                break
            time.sleep(0.5)

        r = client.post(
            f"{base}/agent/decide",
            json={"prediction_job_public_id": pj_id, "use_rag": False},
        )
        r.raise_for_status()
        print("agent report:", json.dumps(r.json(), indent=2, default=str))


if __name__ == "__main__":
    main()
