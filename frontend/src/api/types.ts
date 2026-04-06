/** Mirrors backend Pydantic / enum payloads (ChainAgentVFL API). */

export type FileKind =
  | 'upload'
  | 'training_dataset'
  | 'prediction_input'
  | 'knowledge_doc';

export type JobStatus = 'pending' | 'running' | 'completed' | 'failed';

export type AlgorithmName = 'vfl';

export type IsoDateString = string;

export type ManagedFileOut = {
  id: number;
  public_id: string;
  original_name: string;
  storage_path: string;
  mime_type: string | null;
  file_kind: FileKind;
  version: number;
  parent_file_id: number | null;
  size_bytes: number | null;
  created_at: IsoDateString;
};

export type FileUploadResponse = {
  public_id: string;
  original_name: string;
  version: number;
  file_kind: FileKind;
  message?: string;
};

export type DatasetPreviewOut = {
  columns: string[];
  rows: Record<string, unknown>[];
  row_count: number;
  preview_limit: number;
};

export type TrainingStartRequest = {
  dataset_file_public_id: string;
  target_column: string;
  /** Defaults to VFL on the server; included for explicit API parity. */
  algorithm?: AlgorithmName;
  test_size?: number;
  random_state?: number;
  /** Optional path to agentic_features.json (e.g. storage/agentic_features.json). */
  vfl_agent_definitions_path?: string | null;
};

export type TrainingJobOut = {
  id: number;
  public_id: string;
  status: JobStatus;
  dataset_file_id: number;
  dataset_file_public_id?: string | null;
  dataset_original_name?: string | null;
  target_column: string;
  algorithm: string;
  config_json: Record<string, unknown> | null;
  metrics_json: Record<string, unknown> | null;
  model_version_id: number | null;
  model_version_public_id?: string | null;
  error_message: string | null;
  created_at: IsoDateString;
  updated_at: IsoDateString;
};

export type ModelVersionOut = {
  id: number;
  public_id: string;
  version_number: number;
  training_job_id: number | null;
  algorithm: string;
  artifact_path: string;
  metrics_json: Record<string, unknown> | null;
  feature_columns_json: unknown[] | null;
  label_classes_json: unknown[] | null;
  created_at: IsoDateString;
};

export type TrainingStartResponse = {
  job_public_id: string;
  status: JobStatus;
  message?: string;
};

export type TrainingRebuildRequest = {
  from_job_public_id: string;
};

export type PredictionStartRequest = {
  model_version_public_id: string;
  input_file_public_id: string;
  anomaly_probability_threshold?: number | null;
  attack_label_values?: string[] | null;
};

export type PredictionJobOut = {
  id: number;
  public_id: string;
  model_version_id: number;
  input_file_id: number;
  status: JobStatus;
  output_path: string | null;
  rows_total: number | null;
  rows_flagged: number | null;
  config_json: Record<string, unknown> | null;
  error_message: string | null;
  created_at: IsoDateString;
  updated_at: IsoDateString;
};

export type KnowledgeFileOut = {
  id: number;
  public_id: string;
  managed_file_id: number;
  vector_index_dir: string;
  chunk_count: number;
  embedding_model: string;
  created_at: IsoDateString;
};

export type KBUploadResponse = {
  kb_public_id: string;
  managed_file_public_id: string;
  chunk_count: number;
  message?: string;
};

export type KBQueryRequest = {
  query: string;
  top_k?: number;
  kb_public_ids?: string[] | null;
};

export type KBQueryHit = {
  score: number;
  text: string;
  source?: string | null;
  kb_public_id?: string | null;
  rerank_score?: number | null;
  mmr_margin?: number | null;
};

export type AgenticActionPreset = 'standard' | 'containment_focus' | 'fp_review';

export type AgenticDecideRequest = {
  prediction_job_public_id: string;
  use_rag?: boolean;
  feature_notes?: string | null;
  extra_context?: Record<string, unknown> | null;
  kb_citations?: KBQueryHit[] | null;
  agent_action_preset?: AgenticActionPreset | null;
};

export type AgenticReportOut = {
  id: number;
  public_id: string;
  prediction_job_id: number;
  prediction_job_public_id?: string | null;
  summary: string;
  recommended_action: string;
  raw_llm_response: string | null;
  rag_context_used: string | null;
  report_path: string | null;
  created_at: IsoDateString;
};

export type KBQueryResponse = {
  hits: KBQueryHit[];
};

export type RAGLLMRequest = {
  query: string;
  top_k?: number;
  kb_public_ids?: string[] | null;
  precomputed_citations?: KBQueryHit[] | null;
};

export type RAGLLMResponse = {
  answer: string;
  citations: KBQueryHit[];
};

export type KBMultiQueryRequest = {
  queries: string[];
  final_k?: number;
  per_query_k?: number;
  mmr_lambda?: number;
  kb_public_ids?: string[] | null;
};

export type KBMultiQueryResponse = {
  hits: KBQueryHit[];
  meta: Record<string, unknown>;
};

export type RAGTemplateItem = {
  id: string;
  label: string;
  description: string;
  retrieval_queries: string[];
  llm_prompt: string;
};

export type KBRAGLatestPredictionResponse = {
  prediction_job_public_id?: string | null;
  summary?: Record<string, unknown> | null;
  templates: RAGTemplateItem[];
  message?: string | null;
};

export type HealthResponse = {
  status: string;
};

export type RootMetaResponse = {
  message: string;
  swagger_ui: string;
  redoc: string;
  openapi_json: string;
  api_list: string;
};

export type ApiListRouteItem = {
  method: string;
  path: string;
  tag: string;
  summary: string;
};

export type ApiListResponse = {
  title: string;
  version: string;
  swagger_ui: string;
  redoc: string;
  openapi_json: string;
  routes: ApiListRouteItem[];
};
