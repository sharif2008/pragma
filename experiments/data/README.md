# Shared experiment inputs

`experiments/data/<set>/` holds **input CSVs**. Other tasks may **read** these files. They must **not** write results here.

| Folder | N | Owner |
|--------|--:|-------|
| `gold-100/` | 90 | gold-100 (reserved cores; not an eval-x run set) |
| `eval-10/` | 10 | pragma-rag-eval-x smoke |
| `eval-100/` | 100 | pragma-rag-eval-x |
| `eval-200/` | 200 | pragma-rag-eval-x (default) |
| `rag_reason_500/` | 500 | pool those eval sets were sliced from |
| `e2e-detect-chain/` | 1000 | pragma-e2e-detect-chain |

`eval-10` ⊂ `eval-100` ⊂ `eval-200` ⊂ `rag_reason_500`. All disjoint from gold-100.

Results never belong in this tree. Eval writes `experiments/rag/eval-10/` (etc.). E2E writes `experiments/e2e-detect-chain/`.
