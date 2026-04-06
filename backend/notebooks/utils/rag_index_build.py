"""
Build FAISS index with PDF sections, semantic parent splits, child chunks,
retrieval titles (default: local Transformer summarization; optional extractive / OpenAI), and rag_parents.json.

Used by RAG_part1_build_vector_store.ipynb.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Before langchain imports: silence pydantic.v1 UserWarning on Python 3.14+.
if sys.version_info >= (3, 14):
    warnings.filterwarnings(
        "ignore",
        message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
        category=UserWarning,
    )

import numpy as np


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Paragraphs embedded per ``embed_documents`` call inside semantic parent splitting (lowers peak RAM).
SEMANTIC_SPLIT_EMBED_CHUNK = max(1, _int_env("RAG_SEMANTIC_EMBED_CHUNK", 48))
# Max chars per paragraph used **only** for similarity embeddings (full text kept for parents). <=0 = no cap.
_sem_para_raw = os.getenv("RAG_SEMANTIC_PARA_CHARS")
if _sem_para_raw is None or not str(_sem_para_raw).strip():
    SEMANTIC_SPLIT_PARA_EMBED_MAX: Optional[int] = 12288
else:
    try:
        _sp = int(str(_sem_para_raw).strip())
    except ValueError:
        _sp = 12288
    SEMANTIC_SPLIT_PARA_EMBED_MAX = None if _sp <= 0 else _sp


def _embed_documents_chunked(
    texts: List[str],
    embed_documents: Callable[[List[str]], List[List[float]]],
    chunk_size: int,
) -> np.ndarray:
    """Run ``embed_documents`` in slices to avoid huge single ``encode`` / allocation spikes."""
    if chunk_size < 1:
        chunk_size = 1
    parts: List[List[float]] = []
    for i in range(0, len(texts), chunk_size):
        parts.extend(embed_documents(texts[i : i + chunk_size]))
    return np.array(parts, dtype=np.float64)
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# PDF sections (outline when possible; else heading heuristics)
# ---------------------------------------------------------------------------

_HEADING_PATTERNS = [
    re.compile(
        r"^(?:\d+(?:\.\d+){0,4}\s+[^\n]{4,120})$",  # 1.2.3 Title
        re.MULTILINE,
    ),
    re.compile(
        r"^(?:(?:Chapter|Section|Appendix)\s+[IVX\d.]+[^\n]{0,100})$",
        re.MULTILINE | re.IGNORECASE,
    ),
    re.compile(
        r"^[A-Z][A-Z0-9 \-–]{10,80}$",  # ALL CAPS style headings
        re.MULTILINE,
    ),
]


def _merge_pdf_pages(pdf_path: Path) -> Tuple[str, List[int]]:
    """Full document text with page markers; char_to_page[i] = 1-based page for full[i]."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf required. pip install pypdf") from e

    reader = PdfReader(str(pdf_path))
    full_parts: List[str] = []
    char_to_page: List[int] = []
    for i, page in enumerate(reader.pages):
        pnum = i + 1
        if i > 0:
            sep = f"\n\n--- Page {pnum} ---\n\n"
            full_parts.append(sep)
            char_to_page.extend([pnum] * len(sep))
        t = page.extract_text() or ""
        full_parts.append(t)
        char_to_page.extend([pnum] * len(t))
    return "".join(full_parts), char_to_page


def _page_range_for_span(char_to_page: List[int], start: int, end: int) -> Tuple[int, int]:
    if not char_to_page or start >= len(char_to_page):
        return (1, 1)
    end = min(end, len(char_to_page))
    pages = char_to_page[start:end]
    if not pages:
        return (1, 1)
    return (min(pages), max(pages))


def _split_sections_by_headings(full_text: str, char_to_page: List[int]) -> List[Dict[str, Any]]:
    """Find heading line positions and slice sections."""
    boundaries = [0]
    for pat in _HEADING_PATTERNS:
        for m in pat.finditer(full_text):
            line_start = m.start()
            # Only treat as section start if at line beginning or after newline
            if line_start > 0 and full_text[line_start - 1] not in "\n\r":
                continue
            if line_start not in boundaries:
                boundaries.append(line_start)
    boundaries.append(len(full_text))
    boundaries = sorted(set(boundaries))

    sections: List[Dict[str, Any]] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        chunk = full_text[a:b].strip()
        if not chunk:
            continue
        first_line = chunk.split("\n", 1)[0].strip()[:200]
        ps, pe = _page_range_for_span(char_to_page, a, min(b, len(char_to_page)))
        sections.append(
            {
                "section_heading": first_line or f"section_{i}",
                "text": chunk,
                "char_start": a,
                "char_end": b,
                "page_start": ps,
                "page_end": pe,
            }
        )

    # Merge tiny sections into the next to avoid empty retrieval units
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and len(sec["text"]) < 100:
            prev = merged[-1]
            prev["text"] = (prev["text"] + "\n\n" + sec["text"]).strip()
            prev["page_end"] = max(prev["page_end"], sec["page_end"])
        else:
            merged.append(sec)
    sections = merged

    if not sections and full_text.strip():
        ps, pe = _page_range_for_span(char_to_page, 0, len(char_to_page))
        return [
            {
                "section_heading": "document_body",
                "text": full_text.strip(),
                "char_start": 0,
                "char_end": len(full_text),
                "page_start": ps,
                "page_end": pe,
            }
        ]
    return sections


def extract_pdf_sections(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return logical sections: section_heading, text, page_start, page_end."""
    full_text, char_to_page = _merge_pdf_pages(pdf_path)
    if not full_text.strip():
        return []
    return _split_sections_by_headings(full_text, char_to_page)


# ---------------------------------------------------------------------------
# Semantic parent units (paragraph embedding similarity)
# ---------------------------------------------------------------------------


def semantic_split_to_parents(
    text: str,
    embed_documents: Callable[[List[str]], List[List[float]]],
    percentile_break: float = 15.0,
    min_parent_chars: int = 400,
    max_parent_chars: int = 6000,
    *,
    embed_chunk_size: int = SEMANTIC_SPLIT_EMBED_CHUNK,
    para_embed_max_chars: Optional[int] = SEMANTIC_SPLIT_PARA_EMBED_MAX,
) -> List[str]:
    """
    Split text into parent chunks where adjacent-paragraph similarity is low
    (breakpoint at given percentile of pairwise cosine sims).
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        return []
    if len(paras) == 1:
        return [paras[0]]

    if para_embed_max_chars is not None and para_embed_max_chars > 0:
        paras_for_emb = [p[:para_embed_max_chars] for p in paras]
    else:
        paras_for_emb = paras
    embs = _embed_documents_chunked(paras_for_emb, embed_documents, embed_chunk_size)
    sims: List[float] = []
    for i in range(len(paras) - 1):
        sims.append(float(cosine_similarity([embs[i]], [embs[i + 1]])[0][0]))

    thresh = float(np.percentile(sims, percentile_break)) if sims else 0.0
    parents: List[str] = []
    buf: List[str] = [paras[0]]

    def flush() -> None:
        nonlocal buf
        if buf:
            parents.append("\n\n".join(buf))
            buf = []

    for i, s in enumerate(sims):
        nxt = paras[i + 1]
        cur_text = "\n\n".join(buf + [nxt])
        if s < thresh and len("\n\n".join(buf)) >= min_parent_chars:
            flush()
            buf = [nxt]
        else:
            buf.append(nxt)
        if len(cur_text) > max_parent_chars and len(buf) > 1:
            # oversized: flush all but last
            last = buf[-1]
            buf = buf[:-1]
            if buf:
                parents.append("\n\n".join(buf))
            buf = [last]

    flush()

    merged: List[str] = []
    for p in parents:
        if merged and len(p) < min_parent_chars // 2:
            merged[-1] = merged[-1] + "\n\n" + p
        else:
            merged.append(p)
    return merged if merged else [text.strip()]


# ---------------------------------------------------------------------------
# Retrieval titles: extractive (no LLM) or optional OpenAI
# ---------------------------------------------------------------------------

_TITLE_STOPWORDS = frozenset(
    """
    the and for are but not you all can her was one our out day get has him his how its
    may new now old see two who way use that this with from have been were said each which
    their will into more such than them these some what when your about also would there
    could other after first many must being those should where through during before between
    under while against http https www com pdf page any same into than then them they this
    those though thus too very were what when where whether which while whom whose why yet
    both each few most other some such than very just also only own same such both few more
    most other some such than very will shall onto off per via unto onto
    """.split()
)


def _tokenize_for_title(text: str) -> List[str]:
    """Words and short acronyms (e.g. TLS, DDoS) suitable for keyword ranking."""
    if not text:
        return []
    return re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}|[A-Z]{2,8}", text)


def generate_retrieval_title_extractive(
    excerpt: str,
    section_heading: str,
    source_file: str,
    excerpt_char_budget: int = 3500,
    max_keywords: int = 12,
) -> str:
    """
    Build a single dense line for embeddings / display **without an LLM**:
    source stem + cleaned heading + highest-signal keywords from the excerpt (by frequency,
    with heading terms boosted). No external APIs.

    Use ``title_mode="extractive"`` (see ``TITLE_MODE_EXTRACTIVE``) in Part 1 for this only.
    """
    from collections import Counter

    head_clean = re.sub(r"\s+", " ", (section_heading or "").strip())[:120]
    stem = Path(source_file).stem[:50] if source_file else "doc"
    body = (excerpt or "")[:excerpt_char_budget]
    tokens_body = _tokenize_for_title(body)
    tokens_head = _tokenize_for_title(section_heading or "")

    def norm(w: str) -> str:
        return w.lower()

    counts = Counter(norm(w) for w in tokens_body if norm(w) not in _TITLE_STOPWORDS)
    head_set = {norm(w) for w in tokens_head if norm(w) not in _TITLE_STOPWORDS}

    # Surface form from first occurrence (heading before body)
    display: Dict[str, str] = {}
    for w in tokens_head + tokens_body:
        k = norm(w)
        if k in _TITLE_STOPWORDS or len(k) < 3:
            continue
        if k not in display:
            display[k] = w

    def score_key(k: str) -> Tuple[int, int, int]:
        c = counts.get(k, 0)
        in_head = 1 if k in head_set else 0
        return (in_head, c, len(k))

    ranked = sorted(display.keys(), key=score_key, reverse=True)
    picked: List[str] = []
    for k in ranked:
        if len(picked) >= max_keywords:
            break
        picked.append(display[k])

    kw_part = " ".join(picked)
    if head_clean:
        line = f"{stem}: {head_clean} — {kw_part}".strip()
    else:
        line = f"{stem}: {kw_part}".strip()
    line = re.sub(r"\s*—\s*$", "", line)
    return (line[:300] if line else f"{stem}: {head_clean or 'section'}")[:300]


def generate_retrieval_title_openai(
    excerpt: str,
    section_heading: str,
    source_file: str,
    model: Optional[str] = None,
) -> str:
    """Single dense line via OpenAI; falls back to extractive if no key or on error."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        m = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        body = excerpt[:3500]
        user = (
            f"Source file: {source_file}\n"
            f"Structural heading: {section_heading}\n\n"
            f"Excerpt:\n{body}\n\n"
            "Output exactly ONE line (no quotes): a dense retrieval title, 12–20 words, "
            "with attack types, security controls, network tiers (RAN/Edge/Core), protocols, "
            "and mitigation verbs so a vector search matches user security queries."
        )
        r = client.chat.completions.create(
            model=m,
            messages=[
                {
                    "role": "system",
                    "content": "You write search-optimized titles for cybersecurity KB chunks. One line only.",
                },
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        line = (r.choices[0].message.content or "").strip().split("\n")[0].strip()
        line = line.strip("\"'")
        return (
            line[:300]
            if line
            else generate_retrieval_title_extractive(excerpt, section_heading, source_file)
        )
    except Exception:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)


# ---------------------------------------------------------------------------
# Transformer summarization (local; DistilBART-style seq2seq, no OpenAI)
# ---------------------------------------------------------------------------

DEFAULT_SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"
_SUMMARIZER_PIPE: Any = None
_SUMMARIZER_MODEL: Optional[str] = None


def _summarizer_device() -> int:
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def generate_retrieval_title_transformer_summarize(
    excerpt: str,
    section_heading: str,
    source_file: str,
    model_name: Optional[str] = None,
    max_input_chars: int = 2000,
    max_summary_tokens: int = 90,
    min_summary_tokens: int = 14,
) -> str:
    """
    One-line retrieval title from a **local** Hugging Face ``summarization`` pipeline
    (default: DistilBART CNN — Transformer encoder/decoder, no API calls).
    Falls back to :func:`generate_retrieval_title_extractive` if ``transformers`` is
    missing or inference fails.
    """
    global _SUMMARIZER_PIPE, _SUMMARIZER_MODEL
    stem = Path(source_file).stem[:50] if source_file else "doc"
    raw = f"{(section_heading or '').strip()}\n\n{(excerpt or '').strip()}".strip()
    if len(raw) < 80:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)
    raw = raw[:max_input_chars]

    try:
        from transformers import pipeline
    except ImportError:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)

    model = (
        model_name
        or os.getenv("RAG_SUMMARY_MODEL")
        or DEFAULT_SUMMARIZATION_MODEL
    )
    try:
        if _SUMMARIZER_PIPE is None or _SUMMARIZER_MODEL != model:
            _SUMMARIZER_PIPE = pipeline(
                "summarization",
                model=model,
                tokenizer=model,
                device=_summarizer_device(),
            )
            _SUMMARIZER_MODEL = model
        out = _SUMMARIZER_PIPE(
            raw,
            max_length=max_summary_tokens,
            min_length=min_summary_tokens,
            do_sample=False,
            truncation=True,
        )
        summary = (out[0].get("summary_text") or "").strip().replace("\n", " ")
    except Exception:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)

    if not summary:
        return generate_retrieval_title_extractive(excerpt, section_heading, source_file)
    return f"{stem}: {summary}"[:300]


# ---------------------------------------------------------------------------
# Build child Documents + parent store
# ---------------------------------------------------------------------------

CHILD_CHUNK_SIZE = 384
CHILD_CHUNK_OVERLAP = 96
CHILD_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Retrieval title strategies (see ``title_mode`` on ``parents_and_children_to_documents``)
TITLE_MODE_TRANSFORMER = "transformer"
TITLE_MODE_EXTRACTIVE = "extractive"
TITLE_MODE_LLM = "llm"


def _stable_parent_id(source_file: str, section_heading: str, parent_index: int, text_prefix: str) -> str:
    h = hashlib.sha256(
        f"{source_file}|{section_heading}|{parent_index}|{text_prefix[:200]}".encode()
    ).hexdigest()[:16]
    return f"p_{h}"


def parents_and_children_to_documents(
    *,
    source_file: str,
    doc_type: str,
    section_heading: str,
    page_start: int,
    page_end: int,
    parent_bodies: List[str],
    embeddings: SentenceTransformerEmbeddings,
    title_mode: str = TITLE_MODE_TRANSFORMER,
    summarization_model: Optional[str] = None,
    print_titles: bool = True,
    title_counter: Optional[List[int]] = None,
    max_llm_parents: int = 80,
    llm_title_counter: Optional[List[int]] = None,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    For each semantic parent: build a retrieval title (Transformer summary, extractive, or LLM),
    then split into child chunks. Returns (documents for FAISS, parent_records dict).
    """
    if llm_title_counter is None:
        llm_title_counter = [0]
    if title_counter is None:
        title_counter = [0]

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        length_function=len,
        separators=CHILD_SEPARATORS,
    )

    parent_store: Dict[str, Any] = {}
    all_docs: List[Document] = []

    for pi, parent_text in enumerate(parent_bodies):
        if not parent_text.strip():
            continue
        pid = _stable_parent_id(source_file, section_heading, pi, parent_text)
        if title_mode == TITLE_MODE_LLM:
            use_llm = llm_title_counter[0] < max_llm_parents
            if use_llm:
                llm_title_counter[0] += 1
                retrieval_title = generate_retrieval_title_openai(
                    parent_text, section_heading, source_file
                )
            else:
                retrieval_title = generate_retrieval_title_extractive(
                    parent_text, section_heading, source_file
                )
        elif title_mode == TITLE_MODE_EXTRACTIVE:
            retrieval_title = generate_retrieval_title_extractive(
                parent_text, section_heading, source_file
            )
        else:
            # Default: local Transformer summarization (no API)
            retrieval_title = generate_retrieval_title_transformer_summarize(
                parent_text,
                section_heading,
                source_file,
                model_name=summarization_model,
            )

        title_counter[0] += 1
        if print_titles:
            print(
                f"  [title {title_counter[0]}] {source_file!r} | {pid} | {retrieval_title}"
            )

        parent_store[pid] = {
            "parent_id": pid,
            "retrieval_title": retrieval_title,
            "section_heading": section_heading,
            "structural_title": f"{source_file} (pages {page_start}-{page_end})",
            "source_file": source_file,
            "doc_type": doc_type,
            "page_start": page_start,
            "page_end": page_end,
            "text": parent_text,
        }

        # Embed child text prefixed with retrieval title for better query overlap
        raw_chunks = child_splitter.split_text(parent_text)
        for ci, ch in enumerate(raw_chunks):
            page_content = f"{retrieval_title}\n\n{ch}"
            all_docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "retrieval_title": retrieval_title,
                        "title": retrieval_title,
                        "structural_title": parent_store[pid]["structural_title"],
                        "section_heading": section_heading[:500],
                        "parent_id": pid,
                        "child_index": ci,
                        "n_children": len(raw_chunks),
                        "source_file": source_file,
                        "doc_type": doc_type,
                        "page_start": page_start,
                        "page_end": page_end,
                        "text": ch,
                        "parent_text": parent_text,
                    },
                )
            )

    return all_docs, parent_store


def json_kb_item_to_sections(item: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
    """One synthetic section per JSON KB record."""
    title = (item.get("title") or "untitled")[:300]
    text = item.get("text") or ""
    return [
        {
            "section_heading": title,
            "text": text,
            "page_start": 1,
            "page_end": 1,
        }
    ]


def build_documents_from_knowledge_base(
    knowledge_base: List[Dict[str, Any]],
    embeddings: SentenceTransformerEmbeddings,
    *,
    title_mode: str = TITLE_MODE_TRANSFORMER,
    summarization_model: Optional[str] = None,
    print_titles: bool = True,
    max_llm_parents: int = 80,
    verbose: bool = True,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    knowledge_base entries must include title, text, source_file, doc_type, optional page.
    PDFs should be expanded to sections BEFORE calling this (see expand_pdf_kb_items_with_sections).

    ``title_mode``: ``transformer`` (default, local HF summarizer), ``extractive``, or ``llm``.
    """
    all_docs: List[Document] = []
    merged_parents: Dict[str, Any] = {}
    llm_ctr = [0]
    title_ctr = [0]

    if print_titles and verbose:
        print("Retrieval titles (one line per parent chunk stored in the index):")

    for idx, item in enumerate(knowledge_base):
        doc_type = item.get("doc_type", "json")
        source = item.get("source_file", "unknown")
        if item.get("_pdf_sections"):
            sections = item["_pdf_sections"]
        elif doc_type == "pdf" and item.get("_section_mode") and not item.get("_pdf_sections"):
            sections = [
                {
                    "section_heading": item.get("title", "")[:200],
                    "text": item.get("text", ""),
                    "page_start": item.get("page", 1),
                    "page_end": item.get("page", 1),
                }
            ]
        else:
            sections = json_kb_item_to_sections(item, source)

        for sec in sections:
            body = sec["text"]
            if not body.strip():
                continue
            parents = semantic_split_to_parents(body, embeddings.embed_documents)
            docs, pstore = parents_and_children_to_documents(
                source_file=source,
                doc_type=doc_type,
                section_heading=sec.get("section_heading", "")[:500],
                page_start=int(sec.get("page_start", 1)),
                page_end=int(sec.get("page_end", 1)),
                parent_bodies=parents,
                embeddings=embeddings,
                title_mode=title_mode,
                summarization_model=summarization_model,
                print_titles=print_titles,
                title_counter=title_ctr,
                max_llm_parents=max_llm_parents,
                llm_title_counter=llm_ctr,
            )
            all_docs.extend(docs)
            merged_parents.update(pstore)

    if verbose:
        print(
            f"Built {len(all_docs)} child chunks, {len(merged_parents)} parents "
            f"(title_mode={title_mode!r}; LLM API calls={llm_ctr[0]}, cap={max_llm_parents})"
        )
    return all_docs, merged_parents


def expand_pdf_kb_items_with_sections(
    kb: List[Dict[str, Any]], knowledge_dir: Path, verbose: bool = True
) -> List[Dict[str, Any]]:
    """Replace flat per-page PDF rows with one logical item per PDF containing section list."""
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    rest: List[Dict[str, Any]] = []
    for item in kb:
        if item.get("doc_type") != "pdf":
            rest.append(item)
            continue
        fn = item.get("source_file", "")
        by_file.setdefault(fn, []).append(item)

    out: List[Dict[str, Any]] = []
    out.extend(rest)
    knowledge_dir = Path(knowledge_dir)
    for fn, pages in sorted(by_file.items()):
        if not fn:
            continue
        path = knowledge_dir / fn
        if not path.exists():
            if verbose:
                print(f"  Warning: PDF not found for merge {fn}, keeping flat pages")
            out.extend(pages)
            continue
        try:
            sections = extract_pdf_sections(path)
        except Exception as e:
            if verbose:
                print(f"  Warning: section extract failed for {fn}: {e}; using per-page rows")
            out.extend(pages)
            continue
        if not sections:
            out.extend(pages)
            continue
        out.append(
            {
                "title": fn,
                "text": "",
                "source_file": fn,
                "doc_type": "pdf",
                "_pdf_sections": sections,
                "_section_mode": True,
            }
        )
        if verbose:
            print(f"  PDF sections: {fn} -> {len(sections)} section(s)")
    return out


def save_parent_store(vector_store_dir: Path, parent_records: Dict[str, Any]) -> Path:
    path = Path(vector_store_dir) / "rag_parents.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "parents": parent_records}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_parent_store(vector_store_dir: Path) -> Dict[str, Any]:
    p = Path(vector_store_dir) / "rag_parents.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("parents") or {}
