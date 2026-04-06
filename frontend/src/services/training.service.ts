import type {
  TrainingJobOut,
  ModelVersionOut,
  TrainingStartRequest,
  TrainingStartResponse,
  TrainingRebuildRequest,
} from 'src/api/types';

import { paths } from './paths';
import { requestJson } from './http-client';

export async function listTrainingJobs(limit = 100, offset = 0): Promise<TrainingJobOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<TrainingJobOut[]>(`${paths.training.list}?${q}`);
}

export async function startTraining(body: TrainingStartRequest): Promise<TrainingStartResponse> {
  return requestJson<TrainingStartResponse>(paths.training.start, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function rebuildTraining(body: TrainingRebuildRequest): Promise<TrainingStartResponse> {
  return requestJson<TrainingStartResponse>(paths.training.rebuild, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function getTraining(publicId: string): Promise<TrainingJobOut> {
  return getTrainingJob(publicId);
}

export async function getTrainingJob(publicId: string): Promise<TrainingJobOut> {
  return requestJson<TrainingJobOut>(paths.training.byPublicId(publicId));
}

export async function listModels(): Promise<ModelVersionOut[]> {
  return requestJson<ModelVersionOut[]>(paths.models.list);
}
