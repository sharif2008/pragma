import type {
  RAGLLMRequest,
  KBQueryRequest,
  RAGLLMResponse,
  KBQueryResponse,
  KBUploadResponse,
  KnowledgeFileOut,
  KBMultiQueryRequest,
  KBMultiQueryResponse,
  KBFuseHitsMMRRequest,
  KBRAGLatestPredictionResponse,
  KBLLMShapRetrievalRequest,
  KBLLMShapRetrievalResponse,
} from 'src/api/types';

import { paths } from './paths';
import { requestJson, requestVoid, postMultipart } from './http-client';

export async function kbUpload(file: File): Promise<KBUploadResponse> {
  return postMultipart<KBUploadResponse>(paths.kb.upload, file);
}

export async function kbListFiles(): Promise<KnowledgeFileOut[]> {
  return requestJson<KnowledgeFileOut[]>(paths.kb.files);
}

export const kbList = kbListFiles;

export async function kbDelete(publicId: string): Promise<void> {
  return requestVoid(paths.kb.byPublicId(publicId), { method: 'DELETE' });
}

export async function kbQuery(body: KBQueryRequest): Promise<KBQueryResponse> {
  return requestJson<KBQueryResponse>(paths.kb.query, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function kbRagTemplatesLatestPrediction(): Promise<KBRAGLatestPredictionResponse> {
  return requestJson<KBRAGLatestPredictionResponse>(paths.kb.ragTemplatesLatestPrediction);
}

export async function kbRagTemplatesPredictionJob(
  predictionJobPublicId: string,
  opts?: { rowIndex?: number | null }
): Promise<KBRAGLatestPredictionResponse> {
  const q = new URLSearchParams();
  if (opts?.rowIndex != null && opts.rowIndex >= 0) {
    q.set('row_index', String(opts.rowIndex));
  }
  const suffix = q.toString() ? `?${q}` : '';
  return requestJson<KBRAGLatestPredictionResponse>(
    `${paths.kb.ragTemplatesPredictionJob(predictionJobPublicId)}${suffix}`
  );
}

export async function kbQueryMulti(body: KBMultiQueryRequest): Promise<KBMultiQueryResponse> {
  return requestJson<KBMultiQueryResponse>(paths.kb.queryMulti, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function kbFuseHitsMmr(body: KBFuseHitsMMRRequest): Promise<KBMultiQueryResponse> {
  return requestJson<KBMultiQueryResponse>(paths.kb.fuseHitsMmr, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function kbRagLlm(body: RAGLLMRequest): Promise<RAGLLMResponse> {
  return requestJson<RAGLLMResponse>(paths.kb.ragLlm, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function kbLlmShapRetrievalQuery(
  body: KBLLMShapRetrievalRequest
): Promise<KBLLMShapRetrievalResponse> {
  return requestJson<KBLLMShapRetrievalResponse>(paths.kb.llmShapRetrievalQuery, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}
