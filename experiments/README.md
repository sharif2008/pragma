# Experiments

Console paper stages. Each case keeps **one live result**; older complete trees go under that task’s `archive/`. Specs live in `.agent/`, not here.

| Case | Live path | Script |
|------|-----------|--------|
| Train | `detect-train/run_*/` | `pipeline.py detect-train` |
| Predict | `detect-predict/run_*/` | `pipeline.py detect-predict` |
| RAG index | `rag-index/vector_store/` | `pipeline.py rag-index` |
| Reason ablation | `reason/reason_ablation_9/` | `reason_ablation.py` |
| Gold freeze | `gold-100/ground_truth-100.json` | `gold_freeze.py` / `pragma-gold-100` |
| RAG eval-x (A LLM vs B RAG+SHAP) | `rag/eval-200/` (also `hybrid_eval200/`) | `rag_retrieval_scoring.py` / `pragma-rag-eval-x` |
| Detect → RAG → reason → chain (1000) | `e2e-detect-chain/` | `e2e_detect_chain.py` / `pragma-e2e-detect-chain` |
| Agentic attack + authorization (A1–A25) | `agentic-attack/report.md` | `agentic_attack_eval.py` / `pragma-agentic-attack` |
| Shared inputs | `data/<set>/` (gold-100, eval-10/100/200, e2e) | read-only for other tasks |

API artifacts stay in `backend/storage/`.
