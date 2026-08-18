import type {
  RAGLLMRequest,
  KBQueryRequest,
  RAGLLMResponse,
  KBQueryResponse,
  KBUploadResponse,
  KnowledgeFileOut,
  KnowledgeFileListResponse,
  KBMultiQueryRequest,
  KBMultiQueryResponse,
  KBFuseHitsMMRRequest,
  KBLLMShapRetrievalRequest,
  KBLLMShapRetrievalResponse,
  KBRAGLatestPredictionResponse,
} from 'src/api/types';

import { paths } from './paths';
import { requestJson, requestVoid, postMultipart } from './http-client';

export async function kbUpload(file: File): Promise<KBUploadResponse> {
  return postMultipart<KBUploadResponse>(paths.kb.upload, file);
}

export function isPipelineKbArtifactName(name?: string | null): boolean {
  const n = (name || '').trim().toLowerCase();
  if (!n) return false;
  return (
    n.startsWith('traffic_run_') ||
    n.startsWith('customer_message_') ||
    n.includes('_traffic_run_') ||
    n.includes('_customer_message_')
  );
}

export async function kbListFiles(opts?: {
  page?: number;
  pageSize?: number;
  order?: 'asc' | 'desc';
}): Promise<KnowledgeFileListResponse> {
  const q = new URLSearchParams();
  q.set('page', String(Math.max(1, opts?.page ?? 1)));
  q.set('page_size', String(Math.max(1, opts?.pageSize ?? 10)));
  q.set('order', opts?.order === 'asc' ? 'asc' : 'desc');
  return requestJson<KnowledgeFileListResponse>(`${paths.kb.files}?${q.toString()}`);
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
