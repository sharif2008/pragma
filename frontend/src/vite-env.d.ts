/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend origin, e.g. `http://127.0.0.1:8000` (no trailing slash). */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
