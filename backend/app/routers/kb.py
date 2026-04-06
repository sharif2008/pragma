"""RAG knowledge base."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.kb import (
    KBMultiQueryRequest,
    KBMultiQueryResponse,
    KBQueryHit,
    KBQueryRequest,
    KBQueryResponse,
    KBUploadResponse,
    KBRAGLatestPredictionResponse,
    KnowledgeFileOut,
    RAGLLMRequest,
    RAGLLMResponse,
    RAGTemplateItem,
)
from app.services import kb_service
from app.services import llm_service

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


@router.post("/upload", response_model=KBUploadResponse)
async def kb_upload(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
    file: UploadFile = File(...),
) -> KBUploadResponse:
    kb = await kb_service.ingest_kb_document(db, settings, file)
    mf = kb.managed_file
    return KBUploadResponse(
        kb_public_id=kb.public_id,
        managed_file_public_id=mf.public_id,
        chunk_count=kb.chunk_count,
    )


@router.get("/files", response_model=list[KnowledgeFileOut])
def kb_list(db: Annotated[Session, Depends(get_db)]) -> list[KnowledgeFileOut]:
    return kb_service.list_kb_files(db)


@router.delete("/{public_id}", status_code=204)
def kb_delete(
    public_id: str,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    kb_service.delete_kb(db, settings, public_id)


@router.post("/query", response_model=KBQueryResponse)
def kb_query(
    body: KBQueryRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> KBQueryResponse:
    raw = kb_service.query_kb(db, settings, body.query, body.top_k, body.kb_public_ids)
    hits = [
        KBQueryHit(
            score=s,
            text=chunk.get("text", ""),
            source=chunk.get("source"),
            kb_public_id=kb_pid,
        )
        for s, chunk, kb_pid in raw
    ]
    return KBQueryResponse(hits=hits)


@router.get("/rag-templates/latest-prediction", response_model=KBRAGLatestPredictionResponse)
def kb_rag_templates_latest_prediction(
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> KBRAGLatestPredictionResponse:
    """Multi-query + LLM prompt templates derived from the most recent completed prediction job."""
    data = kb_service.latest_prediction_rag_context(db, settings)
    templates = [RAGTemplateItem.model_validate(t) for t in data.get("templates") or []]
    return KBRAGLatestPredictionResponse(
        prediction_job_public_id=data.get("prediction_job_public_id"),
        summary=data.get("summary"),
        templates=templates,
        message=data.get("message"),
    )


@router.post("/query-multi", response_model=KBMultiQueryResponse)
def kb_query_multi(
    body: KBMultiQueryRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> KBMultiQueryResponse:
    """Fuse several retrieval queries (RRF + max-score), then MMR for diverse final documents."""
    raw_hits, meta = kb_service.query_kb_multi_mmr(
        db,
        settings,
        body.queries,
        final_k=body.final_k,
        per_query_k=body.per_query_k,
        mmr_lambda=body.mmr_lambda,
        kb_public_ids=body.kb_public_ids,
    )
    hits = [
        KBQueryHit(
            score=h["score"],
            text=h["text"],
            source=h.get("source"),
            kb_public_id=h.get("kb_public_id"),
            rerank_score=h.get("rerank_score"),
            mmr_margin=h.get("mmr_margin"),
        )
        for h in raw_hits
    ]
    return KBMultiQueryResponse(hits=hits, meta=meta)


@router.post("/rag-llm", response_model=RAGLLMResponse)
async def kb_rag_llm(
    body: RAGLLMRequest,
    db: Annotated[Session, Depends(get_db)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> RAGLLMResponse:
    if body.precomputed_citations:
        hits = list(body.precomputed_citations)
        citations = [
            {
                "text": h.text,
                "source": h.source,
                "score": h.score,
                "kb_public_id": h.kb_public_id,
            }
            for h in hits
        ]
    else:
        raw = kb_service.query_kb(db, settings, body.query, body.top_k, body.kb_public_ids)
        citations = [
            {"text": c.get("text", ""), "source": c.get("source"), "score": s, "kb_public_id": kb}
            for s, c, kb in raw
        ]
        hits = [
            KBQueryHit(score=s, text=c.get("text", ""), source=c.get("source"), kb_public_id=kb)
            for s, c, kb in raw
        ]
    answer = await llm_service.rag_answer(settings, body.query, citations)
    return RAGLLMResponse(answer=answer, citations=hits)
