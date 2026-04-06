import { getApiBaseUrl } from './api-base';

// ----------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number;
  readonly body: unknown;

  constructor(message: string, status: number, body?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.body = body;
  }
}

function detailMessage(status: number, body: unknown): string {
  if (body && typeof body === 'object' && 'detail' in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d
        .map((x) => (typeof x === 'object' && x && 'msg' in x ? String((x as { msg: unknown }).msg) : JSON.stringify(x)))
        .join('; ');
    }
  }
  return `Request failed (${status})`;
}

async function parseJsonSafe(res: Response): Promise<unknown> {
  const text = await res.text();
  if (!text) return undefined;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return text;
  }
}

export type RequestInitSubset = Omit<RequestInit, 'body'> & { body?: BodyInit | null };

export async function requestJson<T>(path: string, init?: RequestInitSubset): Promise<T> {
  const url = `${getApiBaseUrl()}${path.startsWith('/') ? path : `/${path}`}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.headers as Record<string, string>),
    },
  });
  const body = await parseJsonSafe(res);
  if (!res.ok) {
    throw new ApiError(detailMessage(res.status, body), res.status, body);
  }
  return body as T;
}

export async function requestVoid(path: string, init?: RequestInitSubset): Promise<void> {
  const url = `${getApiBaseUrl()}${path.startsWith('/') ? path : `/${path}`}`;
  const res = await fetch(url, init);
  if (!res.ok) {
    const body = await parseJsonSafe(res);
    throw new ApiError(detailMessage(res.status, body), res.status, body);
  }
}

export async function postMultipart<T>(
  path: string,
  file: File,
  extra?: Record<string, string | undefined>
): Promise<T> {
  const form = new FormData();
  form.append('file', file);
  if (extra) {
    for (const [key, value] of Object.entries(extra)) {
      if (value !== undefined && value !== '') {
        form.append(key, value);
      }
    }
  }
  return requestJson<T>(path, { method: 'POST', body: form });
}
