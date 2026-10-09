# Plan: 10 flows × 9 classes, reserve, then freeze

## Why

The first freeze drew 100 **stratified** test rows and collapsed to 71 BENIGN / 1 BOT. Eval needs **10 reserved flows per class** (N=90). Scratch files (`_detect_compact.json`, parent dumps, drafts) must not live in `data/ground_truth/`.

## Steps

1. **Clean** — Delete leftover gold files (`drafts/`, `gold_security_cases.*`, `generation.json`, `split_manifest.json`, `SHA256.txt`). Keep `predictions_*` / `decision_summary_*`.
2. **Sample** — `gold_sample_100.py` → `experiments/fixtures/gold-100/` (shared input only).
3. **Detect** — `gold_predict_100.py` writes SHAP into `experiments/gold-100/` (keep these).
4. **Freeze** — One overwrite: `ground_truth-100.json`. Each case has detection, condition, `rationale`, `rationale_summary`, **`chunks_summary` (semantic digest of retrieved parents; BERTScore ref)**, actions, and up to 20 full PDF parent texts. Copy that file to `data/ground_truth/`. Keep `predictions_*`. Do not call the eval LLM API.

Do not start `pragma-rag-eval100` until `data/ground_truth/ground_truth-100.json` exists.

## Done when

- Fixture histogram is 10 × 9
- One `ground_truth-100.json` (90 PDF-grounded cases)
- Detect/SHAP files still in `experiments/gold-100/`
- `pragma-rag-eval100` reads only that JSON
