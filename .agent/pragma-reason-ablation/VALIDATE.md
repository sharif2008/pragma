# Validate: pragma-reason-ablation

Checked 2026-10-08 after `python scripts/reason_ablation.py`.

## Run

`experiments/reason/all_types_20261008_035216/`

- 9 class folders (incl. BENIGN), 6 cells each, **54/54** parsed `plan.json`
- Predictions reused: `experiments/detect-predict/run_20261008_030634` (9 fixture rows; detect-predict not re-run)
- Index: `experiments/rag-index/vector_store/`
- Model from `backend/.env`: `gpt-6-luna` (paper asks T=0.3; this model rejects non-default temperature, so the runner retries without `temperature`)

## Paper cell (`DDOS/rag_cond_rank`)

- Query contains **DDOS**, confidence, **Access / ISP**, and per-domain feature keywords
- `rag_sections.json`: 5 parents from the ten-PDF corpus
- `plan.json` keys match the schema; actions ∈ $W[\mathrm{DDOS}]$; `network_tier` ∈ {Access / ISP, Perimeter / IDS, Endpoint / EDR}
- `norag_cond` README uses the empty-KB sentence; `rag_sections.json` is `[]`

## Matrix / latency

Root README lists the 9×6 matrix. `latency_summary.csv` has query / retrieve / rank / prompt / llm. Predict ms is empty (amortized prior SHAP run). Rank is recorded only for the 18 ranking-on cells.

## Code

- `backend/scripts/reason_ablation.py` — new runner
- `python scripts/pipeline.py reason -- --all-types-ablation` dispatches via `reason.py`
- `include_conditions` on the paper prompt; `use_ranking` / dense-only path in retrieve
- Query template uses paper domain names (fix: do not reuse `label` as the DOMAIN_LABELS loop variable)

## Verdict

**Pass.** Grouped 9×6 tree is on disk under `experiments/reason/all_types_20261008_035216/`. No chain / API KB writes.
