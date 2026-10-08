# Plan: build RAG index for post-detection reason

## Why

The ten named PDFs live in `experiments/knowledge/`, but the CLI resolver does not see that folder. `experiments/rag-index/` has no vector store yet, so `reason.py` cannot retrieve after detect-predict.

## Steps

1. **Resolver** — in `scripts/env.py` `resolve_rag_knowledge_dir()`, treat a dir as a corpus only if it has `*.pdf` or `*.json` (not merely `mkdir` + empty). Order: `experiments/rag-index/knowledge` → `experiments/knowledge` → `backend/storage/base_docs`.

2. **PDF load** — `rag_build.load_pdf_file` uses PyPDFLoader. `MITRE-ATTCK-Building-Better-Defenses.pdf` is AES-encrypted. If that file errors in the build log, add a pymupdf (or `pypdf[crypto]`) fallback so all ten sources index. Other files should already load.

3. **Build** (from `backend/`; minutes, MiniLM download possible):

   ```powershell
   cd D:\Projects\ChainAgentVFL\backend
   .\.venv\Scripts\Activate.ps1
   python scripts/pipeline.py rag-index
   ```

   Writes only under `experiments/rag-index/vector_store/`. Does not need `pipeline.py` extra args (those are currently ignored except for `e2e`). Direct equivalent: `python scripts/rag_build.py`.

4. **Confirm corpus in the log** — ten PDFs loaded; `n_source_docs` / parent count in the saved manifest; `rag_parents.json` present.

5. **Smoke load** — `reason.py` `load_vector_store(experiments/rag-index/vector_store)` must not raise FileNotFoundError. Full planner run waits until detect-predict outputs exist:

   ```powershell
   python scripts/pipeline.py detect-predict
   python scripts/pipeline.py reason
   ```

## Implementation order (code, then run)

1. `env.py`: PDF/JSON-aware knowledge-dir resolution including `experiments/knowledge`.
2. `rag_build.py`: encrypted-PDF fallback only if PyPDFLoader fails on the ATT&CK defenses file.
3. Run `pipeline.py rag-index`.
4. Smoke-check vector store load. Do not upload the same PDFs into the API KB.

## Done when

- Resolver selects `experiments/knowledge` (ten PDFs) when `rag-index/knowledge` is empty.
- `experiments/rag-index/vector_store/` has FAISS + `rag_parents.json` + manifest.
- Build log includes all ten sources (or a documented skip).
- `reason.py` can load that folder.
