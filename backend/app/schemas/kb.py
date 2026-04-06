"""Knowledge base schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

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
    use_mmr: bool = Field(
        default=True,
        description="If false, return top documents by fusion rerank score only (RRF + max-score), skip MMR diversification.",
    )

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


class KBFuseHitsMMRRequest(BaseModel):
    """Hit lists from separate POST /kb/query calls (one list per query); server dedupes, fusion-reranks, MMR."""

    queries: list[str] = Field(..., min_length=1)
    per_query_hits: list[list[KBQueryHit]] = Field(
        ...,
        description="Same length as queries; each inner list is /kb/query hits in rank order for that query.",
    )
    final_k: int = Field(default=8, ge=1, le=30, description="Documents returned after fusion + MMR.")
    mmr_lambda: float = Field(default=0.55, ge=0.0, le=1.0, description="MMR tradeoff: relevance vs diversity.")
    kb_public_ids: list[str] | None = None
    use_mmr: bool = Field(
        default=True,
        description="If false, fusion rerank only (no MMR diversification).",
    )

    @field_validator("queries")
    @classmethod
    def strip_queries(cls, v: list[str]) -> list[str]:
        out = [q.strip() for q in v if q and str(q).strip()]
        if not out:
            raise ValueError("At least one non-empty query is required")
        return out[:12]

    @model_validator(mode="after")
    def hit_groups_align_with_queries(self) -> KBFuseHitsMMRRequest:
        if len(self.queries) != len(self.per_query_hits):
            raise ValueError("per_query_hits must have exactly one group per query")
        return self


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
    row_index: int | None = Field(default=None, description="When templates were built for a specific results row.")
    row_context: dict[str, Any] | None = Field(
        default=None,
        description="Per-row SHAP highlights and labels when row_index is set.",
    )


class KBLLMShapRetrievalRequest(BaseModel):
    """Draft retrieval text from SHAP templates + row data; optional synthesis prompt for alignment."""

    draft_queries_text: str = Field(..., min_length=1, max_length=12000)
    analyst_synthesis_prompt: str | None = Field(
        default=None,
        max_length=4000,
        description="Usually row_agent_shap_queries.llm_prompt — guides the retrieval rewrite toward later RAG synthesis.",
    )


class KBLLMShapRetrievalResponse(BaseModel):
    retrieval_query: str
    used_llm: bool = Field(description="False when OPENAI_API_KEY is unset and the draft was returned unchanged.")
