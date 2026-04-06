import type {
  PredictionJobOut,
  FileUploadResponse,
  PredictionStartRequest,
  ManagedFileOut,
} from 'src/api/types';

import { paths } from './paths';
import { requestJson, postMultipart } from './http-client';

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

export async function getPredictionJob(publicId: string): Promise<PredictionJobOut> {
  return requestJson<PredictionJobOut>(paths.predictions.byPublicId(publicId));
}
