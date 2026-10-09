# Shared input: gold-100 flows

Owner task: **gold-100**. Other tasks (`pragma-rag-eval100`, `pragma-e2e-detect-chain`) may **read** this folder. They must not write results here.

**10 held-out test flows × 9 trained classes (N=90)**, seed 42.

`flows.csv` + `manifest.json` are locked. Do not resample without a new freeze. Do not reuse these `split_index` values in `fixtures/rag_reason_500`.

Gold **results** (detect + freeze) live in `experiments/gold-100/`, not here.
Eval labels: `data/ground_truth/`.
