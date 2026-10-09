# e2e-detect-chain

Owner task: **pragma-e2e-detect-chain**. This folder is the **only experiment input** (1000 flow rows).

**N = 1000** test flows, seed 42, VFL 20% stratified split. Water-fill across the nine labels. Confirm counts in `manifest.json` after `--sample-only`.

One system: **`RAG_RANKING`**. **One LLM call per flow.** No inject. No BERTScore.

## Results (not here)

Generated detect / plans / chain / latency live in `experiments/e2e-detect-chain/`.
