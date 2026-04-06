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
    byPublicId: (publicId: string) => `/models/${encodeURIComponent(publicId)}`,
  },

  predictions: {
    list: '/predictions',
    pendingDelete: '/predictions/pending',
    uploadInput: '/predictions/upload-input',
    inputs: '/predictions/inputs',
    start: '/predictions/start',
    byPublicId: (publicId: string) => `/predictions/${encodeURIComponent(publicId)}`,
  },

  agent: {
    decide: '/agent/decide',
    decidePromptPreview: '/agent/decide/prompt-preview',
    jobs: '/agent/jobs',
    reports: '/agent/reports',
    reportByPublicId: (publicId: string) => `/agent/reports/${encodeURIComponent(publicId)}`,
  },

  kb: {
    upload: '/kb/upload',
    files: '/kb/files',
    byPublicId: (publicId: string) => `/kb/${encodeURIComponent(publicId)}`,
    query: '/kb/query',
    queryMulti: '/kb/query-multi',
    fuseHitsMmr: '/kb/fuse-hits-mmr',
    ragTemplatesLatestPrediction: '/kb/rag-templates/latest-prediction',
    ragTemplatesPredictionJob: (publicId: string) =>
      `/kb/rag-templates/prediction-job/${encodeURIComponent(publicId)}`,
    ragLlm: '/kb/rag-llm',
    llmShapRetrievalQuery: '/kb/llm-shap-retrieval-query',
  },
} as const;
