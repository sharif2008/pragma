# Experiments

Console paper stages. Each case keeps **one live result**; older complete trees go under that task’s `archive/`. Specs live in `.agent/`, not here.

| Case | Live path | Script |
|------|-----------|--------|
| Train | `detect-train/run_*/` | `pipeline.py detect-train` |
| Predict | `detect-predict/run_*/` | `pipeline.py detect-predict` |
| RAG index | `rag-index/vector_store/` | `pipeline.py rag-index` |
| Reason ablation | `reason/reason_ablation_9/` | `reason_ablation.py` |
| Gold freeze | `gold-100/` (detect + freeze) | `gold_predict_100.py` / `pragma-gold-100` |
| RAG eval (gold cores) | `rag/rag_eval100/` (flat: plans + jsonl) | `pragma-rag-eval100` |
| Detect → RAG → reason → chain (1000) | `e2e-detect-chain/` | `reason_1000.py` / `pragma-e2e-detect-chain` |
| Agentic attack + authorization (A1–A25) | `agentic-attack/report.md` | `agentic_attack_eval.py` / `pragma-agentic-attack` |
| Shared inputs | `data/<owner-task>/` | read-only for other tasks |

API artifacts stay in `backend/storage/`.
