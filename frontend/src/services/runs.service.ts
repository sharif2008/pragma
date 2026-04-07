import type { RunEventOut, RunSummaryOut, RunListItemOut } from 'src/api/types';

import { getApiBaseUrl } from './api-base';
import { requestJson } from './http-client';

export async function listRuns(params?: { status?: string; limit?: number; offset?: number }) {
  const qs = new URLSearchParams();
  if (params?.status) qs.set('status', params.status);
  if (typeof params?.limit === 'number') qs.set('limit', String(params.limit));
  if (typeof params?.offset === 'number') qs.set('offset', String(params.offset));
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return requestJson<RunListItemOut[]>(`/api/v1/runs${suffix}`);
}

export async function getRun(run_id: string) {
  return requestJson<RunSummaryOut>(`/api/v1/runs/${encodeURIComponent(run_id)}`);
}

export async function getRunEvents(run_id: string, limit = 5000) {
  return requestJson<RunEventOut[]>(
    `/api/v1/runs/${encodeURIComponent(run_id)}/events?limit=${encodeURIComponent(String(limit))}`
  );
}

export async function listRunEvents(params?: { limit?: number; offset?: number }) {
  const qs = new URLSearchParams();
  if (typeof params?.limit === 'number') qs.set('limit', String(params.limit));
  if (typeof params?.offset === 'number') qs.set('offset', String(params.offset));
  const suffix = qs.toString() ? `?${qs.toString()}` : '';
  return requestJson<RunEventOut[]>(`/api/v1/runs/events${suffix}`);
}

export function streamRunEvents(run_id: string) {
  const url = `/api/v1/runs/${encodeURIComponent(run_id)}/stream`;
  // Base URL is injected by requestJson; for EventSource we need absolute.
  return new EventSource(`${getApiBaseUrl()}${url}`);
}

