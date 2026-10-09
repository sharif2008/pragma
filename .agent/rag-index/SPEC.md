# Spec: build CLI RAG index from ten knowledge PDFs

Action: `rag-index`

Build the closed policy index once. Detection (VFL) does not read it. **reason** queries it after **detect-predict**.

## Corpus (max 10 sources)

PDF/JSON only. Spreadsheets are not indexed (`rag_build.py` globs `*.pdf` and `*.json`).

Staged originals: `experiments/knowledge/` (also mirrored in `backend/storage/base_docs/`):

| File | Document |
|------|----------|
| `NIST-SP-800-53-Rev5-Security-Privacy-Controls.pdf` | NIST SP 800-53 Rev. 5 |
| `NIST-SP-800-207-Zero-Trust-Architecture.pdf` | NIST SP 800-207 Zero Trust |
| `ISO-IEC-27001-ISMS-Requirements.pdf` | ISO/IEC 27001 ISMS |
| `CIS-Controls-v8.1.pdf` | CIS Controls v8.1 |
| `MITRE-ATTCK-Design-and-Philosophy.pdf` | ATT&CK design and philosophy |
| `MITRE-ATTCK-Building-Better-Defenses.pdf` | ATT&CK building better defenses |
| `CISO-Guide-Top-Cybersecurity-Frameworks.pdf` | CISO guide to frameworks |
| `Open-RAN-Security-Report.pdf` | Open RAN security report |
| `NIST-SP-800-53-vs-ISO-IEC-27001-Comparison.pdf` | NIST vs ISO comparison |
| `Review-NIST-ISO27001-HIPAA-MITRE-ATTCK.pdf` | Framework review (NIST / ISO / HIPAA / ATT&CK) |

Do not add an 11th source. Do not copy PDFs into `experiments/rag-index/knowledge/` unless that folder is an explicit override for a run.

## Knowledge-dir resolution

`resolve_rag_knowledge_dir()` must pick the first directory that actually contains `.pdf` or `.json`:

1. `experiments/rag-index/knowledge` (per-run override)
2. `experiments/knowledge` (staged ten sources)
3. `backend/storage/base_docs` (catalog fallback)

An empty `experiments/rag-index/knowledge/` created by `mkdir` must not win.

## Index write (this task only)

`rag_build.py` writes only under `experiments/rag-index/vector_store/` (the **current** index; rebuild overwrites in place):

| File | Role |
|------|------|
| LangChain FAISS (`index.faiss` / pickle) | Child-chunk nearest-neighbor search |
| `rag_parents.json` | Parent sections the LLM reads |
| manifest JSON | `embed_model`, `n_source_docs`, `n_chunks`, `n_parents`, chunk sizes |

Chunking: child size 384, overlap 96, embed `all-MiniLM-L6-v2`. MiniLM cache: `backend/storage/hf_home` (do not add a second cache under `experiments/`).

## Later consume (not this write)

| Reader | Reads |
|--------|--------|
| `reason.py` | `experiments/rag-index/vector_store/` + `experiments/detect-predict/` |
| `evaluate.py` | reason outputs as needed |

Query: templated retrieval from predicted class + SHAP domain shares → MMR / cross-encoder → up to 5 parent sections in the planner prompt.

## Out of scope

- Retrain / detect-predict (separate stages; index can be built before or after predict)
- FastAPI `/kb` upload and `backend/storage/vector_db` (web agent; different store)
- Changing JSON keys RAN/Edge/Core in `agentic_features.json`
