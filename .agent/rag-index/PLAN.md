# Plan: RAG index for post-detection reason

## Live result

`experiments/rag-index/vector_store/` — current FAISS + `rag_parents.json` + manifest. Corpus: ten PDFs via `resolve_rag_knowledge_dir()` (`experiments/knowledge/` today).

Do not keep a second index copy. Rebuild overwrites this folder.

## Re-run

```powershell
cd D:\Projects\ChainAgentVFL\backend
.\.venv\Scripts\Activate.ps1
python scripts/pipeline.py rag-index
```

## Sanity

- Manifest `n_source_docs` is 10
- Empty `experiments/rag-index/knowledge/` does not win the resolver
- `reason.py` can `load_vector_store` this folder

## Done when

One live vector store under `experiments/rag-index/vector_store/`. No API `/kb` writes.
