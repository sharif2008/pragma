# Shared experiment inputs

`experiments/fixtures/<owner-task>/` holds **input CSVs** named by the task that created them. Other tasks may **read** these files. They must **not** write results here or into another task’s `experiments/<task>/` folder.

Resolver: `scripts.env.fixtures_dir()` / `resolve_fixture_csv(name)`.

| Folder (owner) | N | Readers |
|----------------|--:|---------|
| `gold-100/` | 90 | gold-100, rag-eval100, rag-reason (exclude these `split_index`) |
| `reason_ablation_9/` | 9 | detect-predict default sample, reason ablation |
| `rag_reason_500/` | 500 | rag-reason (ranked), blockchain, e2e (disjoint from gold-100) |
| `rag_reason_1000/` | — | pointer: use `rag_reason_500/` |
| `rag_retrieval_test_50/` | 50 | retrieval smoke only |
| `blockchain_all_1000/` | — | pointer: use `rag_reason_500/` |
| `e2e_detect_to_blockchain_1000/` | — | pointer: use `rag_reason_500/` |

`rag_ground_100/` is a redirect README only; use `fixtures/gold-100/`.
