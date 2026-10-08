# Validate: rag-index

Checked 2026-10-08 after `python scripts/pipeline.py rag-index`.

## Corpus

`experiments/knowledge/` has exactly 10 PDFs. Build log loaded all ten, including encrypted `MITRE-ATTCK-Building-Better-Defenses.pdf` (18 pages). Manifest `n_source_docs` is 10.

## Resolver

`resolve_rag_knowledge_dir()` now requires `*.pdf` or `*.json`. Empty `experiments/rag-index/knowledge/` does not win. Runtime resolved to `experiments/knowledge`.

## Index

`experiments/rag-index/vector_store/`:

| File | Role |
|------|------|
| `index.faiss` | FAISS child-chunk index (~16 MB) |
| `index.pkl` | LangChain docstore (~13 MB) |
| `rag_parents.json` | 2111 parent sections |
| `rag_manifest.json` | MiniLM, chunk 384/96, 10511 children, 10 sources |

`load_vector_store` + `load_parent_store` succeed. Similarity search returns NIST SP 800-53 chunks.

## Pipeline argv

`pipeline.py rag-index` used to pass `rag-index` into `rag_build.py` argparse (`unrecognized arguments`). `_run_script` now sets `sys.argv` to the target script (+ forwarded extras). Required for the spec command to work.

## Encrypted PDF

venv has `cryptography`; pypdf decrypts the ATT&CK defenses PDF. `load_pdf_file` falls back to `PdfReader` if PyPDFLoader fails. `_merge_pdf_pages` decrypts when `is_encrypted`. No pymupdf dependency added.

## Out of scope (unchanged)

FastAPI `/kb` / `storage/vector_db` is a different store. Full `reason.py` still needs detect-predict outputs.

## Verdict

**Pass.** Ten-PDF FAISS index is on disk under `experiments/rag-index/vector_store/` and can be loaded by reason.
