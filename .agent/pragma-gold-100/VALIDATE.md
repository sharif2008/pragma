# Validate: pragma-gold-100

Checked after single-file freeze with top-20 full PDF parents + rationale summary.

## Shared input

- [x] `experiments/data/gold-100/flows.csv` — 90 rows, 10 per class

## Results

- [x] Detect/SHAP kept under `experiments/gold-100/predictions_*`
- [x] Leftover gold JSON (`drafts/`, `gold_security_cases.*`, `generation.json`, …) removed
- [x] Main output: `ground_truth-100.json` (live == `data/ground_truth/`)

## Per case

- [x] `detection`, `condition`, `actions`
- [x] `rationale` (PDF-quoted), `rationale_summary`, `chunks_summary` (semantic digest of chunks; BERTScore ref)
- [x] `relevant_rag_chunks` ≤ 20 with **full `text`**
- [x] `gold_validate.py` OK

## Verdict

**Ready for `pragma-rag-eval100`.** Eval reads only `data/ground_truth/ground_truth-100.json`.
