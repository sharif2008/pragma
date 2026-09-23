#!/usr/bin/env python3
"""Generate a larger synthetic CSV for local training / API tests (writes under ``scripts/data/``)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=500)
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "synthetic_flows.csv",
    )
    args = p.parse_args()
    rng = np.random.default_rng(42)
    n = args.rows
    label = rng.choice(["normal", "attack"], size=n, p=[0.85, 0.15])
    duration = rng.integers(1, 400, size=n).astype(float)
    duration[label == "attack"] *= 0.3
    src_bytes = rng.lognormal(6, 1.5, size=n)
    src_bytes[label == "attack"] *= 4
    dst_bytes = rng.lognormal(5, 1.2, size=n)
    protocol = rng.choice(["tcp", "udp", "icmp"], size=n)
    df = pd.DataFrame(
        {
            "duration": duration.astype(int),
            "src_bytes": src_bytes.astype(int),
            "dst_bytes": dst_bytes.astype(int),
            "protocol": protocol,
            "label": label,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
