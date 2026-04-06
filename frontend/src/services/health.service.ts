import type { HealthResponse, ApiListResponse, RootMetaResponse } from 'src/api/types';

import { paths } from './paths';
import { requestJson } from './http-client';

export async function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>(paths.health);
}

export async function getRootMeta(): Promise<RootMetaResponse> {
  return requestJson<RootMetaResponse>('/');
}

export async function getApiList(): Promise<ApiListResponse> {
  return requestJson<ApiListResponse>('/api-list');
}
