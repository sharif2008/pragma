import type {
  PredictionJobOut,
  FileUploadResponse,
  PredictionStartRequest,
  ManagedFileOut,
  PendingPredictionPurgeOut,
} from 'src/api/types';

export type PredictionJobListItem = Omit<PredictionJobOut, 'results_json'>;

import { paths } from './paths';
import { requestJson, requestVoid, postMultipart } from './http-client';

const PREDICTION_JOBS_PAGE = 500;

export async function listPredictionJobs(limit = 100, offset = 0): Promise<PredictionJobListItem[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<PredictionJobListItem[]>(`${paths.predictions.list}?${q}`);
}

/** Fetch every prediction job (newest first) via repeated pages until the API returns a short batch. */
export async function listAllPredictionJobs(): Promise<PredictionJobListItem[]> {
  const out: PredictionJobListItem[] = [];
  for (let offset = 0; ; offset += PREDICTION_JOBS_PAGE) {
    const batch = await listPredictionJobs(PREDICTION_JOBS_PAGE, offset);
    out.push(...batch);
    if (batch.length < PREDICTION_JOBS_PAGE) break;
  }
  return out;
}

export async function listPredictionInputs(): Promise<ManagedFileOut[]> {
  return requestJson<ManagedFileOut[]>(paths.predictions.inputs);
}

export async function uploadPredictionInput(file: File): Promise<FileUploadResponse> {
  return postMultipart<FileUploadResponse>(paths.predictions.uploadInput, file);
}

export const uploadPredictionCsv = uploadPredictionInput;

export async function startPrediction(body: PredictionStartRequest): Promise<PredictionJobOut> {
  return requestJson<PredictionJobOut>(paths.predictions.start, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function getPrediction(publicId: string): Promise<PredictionJobOut> {
  return getPredictionJob(publicId);
}

export async function getPredictionJob(
  publicId: string,
  opts?: { includeResults?: boolean }
): Promise<PredictionJobOut> {
  const q =
    opts?.includeResults === true
      ? `?include_results=${encodeURIComponent('true')}`
      : '';
  return requestJson<PredictionJobOut>(`${paths.predictions.byPublicId(publicId)}${q}`);
}

export async function deletePredictionJob(publicId: string): Promise<void> {
  return requestVoid(paths.predictions.byPublicId(publicId), { method: 'DELETE' });
}

export async function deleteAllPendingPredictionJobs(): Promise<PendingPredictionPurgeOut> {
  return requestJson<PendingPredictionPurgeOut>(paths.predictions.pendingDelete, { method: 'DELETE' });
}
