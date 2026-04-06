/** Single source for backend origin (FastAPI). Override with `VITE_API_BASE_URL`. */

export const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export function getApiBaseUrl(): string {
  const raw = import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL;
  return String(raw).replace(/\/$/, '');
}
