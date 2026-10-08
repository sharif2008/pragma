# Validate: pragma-reason-ablation

Checked 2026-10-08 against `docs/Pragma_v2.tex` §Reason, `reason.py`, `llm_prompt.py`, and on-disk fixtures. **Not implemented yet** — this is the gap list the runner must close.

## Fixture is ready

`experiments/detect-predict/inputs/all_attack_types.csv` has 9 rows (one per trained class) plus `original_label`. `resolve_sample_csv()` prefers that file. `backend/run/data/sample_all_attack_types.csv` is a copy. Detect-predict has not been re-run on this 9-row set (last predict CSV is DDoS/BENIGN-only).

## Predict → reason handoff exists

`reason.py` loads `experiments/detect-predict/` JSON and `experiments/rag-index/vector_store/`. Template query, 7-step cascade (dense → merge → MMR $\lambda=0.5$ → MiniLM cross-encoder → top-20 → parent expand → budget 5), and `create_agentic_orchestration_prompt` already match the paper slots (role, three domains, $W[\hat{y}_i]$, JSON schema, $T=0.3$). Empty RAG path exists (`include_knowledge_base=False`).

## Gaps vs this spec

| Spec need | Today |
|-----------|--------|
| One grouped folder per class with README | Flat `experiments/reason/action_plans/` JSON dumps |
| Six-cell ablation (RAG × condition × ranking) | Only with-RAG vs no-RAG; ranking and SHAP always on |
| Condition-off query/prompt | `build_template_rag_query` always embeds dominant tier + top features |
| Ranking-off | `retrieve_rag_context_multi` always MMR + cross-encoder |
| BENIGN included | `_is_predicted_attack` skips BENIGN |
| Per-step latency | Not recorded (`trust_anchor_benchmark.py` times a different, TF-IDF path) |
| Domain names in query | Template still says RAN / Edge / Core |
| `pipeline.py reason -- …` | `extra` forwarded only for `e2e` |
| Paper empty-KB sentence in README | Prompt has a hard rule; not persisted as a first-class RAG slot artifact |

## Index / key prereqs (external)

- `experiments/rag-index/vector_store/` must exist (`rag-index` task). If missing, ablation cannot retrieve.
- `OPENAI_API_KEY` required; no LLM → no `plan.json`.
- Prompt already forbids invented policy when RAG is empty; ablation must **assert** `knowledge_sources_used` does not cite fake section tags on `norag_*` cells.

## Runtime

- 9 KernelSHAP predicts (once).
- 54 GPT-4o-mini calls (6 cells × 9 classes). Smoke: `--classes DDOS,PORTSCAN`.
- Cross-encoder load once per process (`sentence-transformers`).

## Verdict

**Fail until runner exists.** Building blocks (predict JSON, FAISS cascade, paper prompt, with/without RAG) are in place. Remaining work is `reason_ablation.py`: 9×6 grouped tree, SHAP/ranking switches, BENIGN, domain labels in the query, per-step ms, README generation. Do not treat `action_plans/` pair files as this task’s output.
