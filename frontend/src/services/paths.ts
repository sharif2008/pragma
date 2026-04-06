/**
 * Backend route paths (FastAPI, no global prefix).
 * Use with {@link getApiBaseUrl} from `./api-base`.
 */
export const paths = {
  health: '/health',

  files: {
    upload: '/files/upload',
    list: '/files',
    byPublicId: (publicId: string) => `/files/${encodeURIComponent(publicId)}`,
  },

  datasets: {
    upload: '/datasets/upload',
    list: '/datasets',
    byPublicId: (publicId: string) => `/datasets/${encodeURIComponent(publicId)}`,
    preview: (publicId: string) => `/datasets/${encodeURIComponent(publicId)}/preview`,
  },

  training: {
    list: '/training',
    start: '/training/start',
    rebuild: '/training/rebuild',
    byPublicId: (publicId: string) => `/training/${encodeURIComponent(publicId)}`,
  },

  models: {
    list: '/models',
  },

  predictions: {
    uploadInput: '/predictions/upload-input',
    inputs: '/predictions/inputs',
    start: '/predictions/start',
    byPublicId: (publicId: string) => `/predictions/${encodeURIComponent(publicId)}`,
  },

  agent: {
    decide: '/agent/decide',
    reports: '/agent/reports',
    reportByPublicId: (publicId: string) => `/agent/reports/${encodeURIComponent(publicId)}`,
  },

  kb: {
    upload: '/kb/upload',
    files: '/kb/files',
    byPublicId: (publicId: string) => `/kb/${encodeURIComponent(publicId)}`,
    query: '/kb/query',
    queryMulti: '/kb/query-multi',
    ragTemplatesLatestPrediction: '/kb/rag-templates/latest-prediction',
    ragLlm: '/kb/rag-llm',
  },
} as const;
