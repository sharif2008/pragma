"""Knowledge base schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class KnowledgeFileOut(ORMModel):
    id: int
    public_id: str
    managed_file_id: int
    vector_index_dir: str
    chunk_count: int
    embedding_model: str
    created_at: datetime


class KBUploadResponse(ORMModel):
    kb_public_id: str
    managed_file_public_id: str
    chunk_count: int
    message: str = "indexed"


class KBQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=50)
    kb_public_ids: list[str] | None = Field(
        default=None,
        description="Restrict search to these KB file UUIDs; if empty, search all indices.",
    )


class KBQueryHit(BaseModel):
    score: float
    text: str
    source: str | None = None
    kb_public_id: str | None = None
    rerank_score: float | None = Field(default=None, description="Fusion rerank score before MMR (when using multi-query).")
    mmr_margin: float | None = Field(default=None, description="MMR objective at selection time.")


class KBQueryResponse(BaseModel):
    hits: list[KBQueryHit]


class RAGLLMRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    kb_public_ids: list[str] | None = None
    precomputed_citations: list[KBQueryHit] | None = Field(
        default=None,
        description="If set, skip vector search and pass these snippets to the LLM (e.g. after /kb/query-multi).",
    )


class RAGLLMResponse(BaseModel):
    answer: str
    citations: list[KBQueryHit]


class KBMultiQueryRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1)
    final_k: int = Field(default=8, ge=1, le=30, description="Documents returned after fusion + MMR.")
    per_query_k: int = Field(default=12, ge=4, le=50, description="FAISS depth per query per KB index.")
    mmr_lambda: float = Field(default=0.55, ge=0.0, le=1.0, description="MMR tradeoff: relevance vs diversity.")
    kb_public_ids: list[str] | None = None

    @field_validator("queries")
    @classmethod
    def nonempty_queries(cls, v: list[str]) -> list[str]:
        out = [q.strip() for q in v if q and str(q).strip()]
        if not out:
            raise ValueError("At least one non-empty query is required")
        return out[:12]


class KBMultiQueryResponse(BaseModel):
    hits: list[KBQueryHit]
    meta: dict[str, Any] = Field(default_factory=dict)


class RAGTemplateItem(BaseModel):
    id: str
    label: str
    description: str
    retrieval_queries: list[str]
    llm_prompt: str


class KBRAGLatestPredictionResponse(BaseModel):
    prediction_job_public_id: str | None = None
    summary: dict[str, Any] | None = None
    templates: list[RAGTemplateItem] = Field(default_factory=list)
    message: str | None = None
