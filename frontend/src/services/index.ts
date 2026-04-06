export * from './kb.service';
export { paths } from './paths';
export * from './files.service';

export * from './agent.service';
export * from './health.service';
export * from './datasets.service';
export * from './training.service';
export * from './predictions.service';
export { getApiBaseUrl, DEFAULT_API_BASE_URL } from './api-base';
export { ApiError, requestJson, requestVoid, postMultipart, type RequestInitSubset } from './http-client';

import { agentDecide, listAgentReports, getAgentReport } from './agent.service';
import { listFiles, deleteFile, uploadFile } from './files.service';
import { getHealth, getApiList, getRootMeta } from './health.service';
import {
  kbList,
  kbQuery,
  kbQueryMulti,
  kbDelete,
  kbRagLlm,
  kbRagTemplatesLatestPrediction,
  kbUpload,
  kbListFiles,
} from './kb.service';
import { listDatasets, deleteDataset, uploadDataset, getDatasetPreview } from './datasets.service';
import {
  getPrediction,
  startPrediction,
  getPredictionJob,
  uploadPredictionCsv,
  uploadPredictionInput,
  listPredictionInputs,
} from './predictions.service';
import {
  listModels,
  getTraining,
  startTraining,
  getTrainingJob,
  rebuildTraining,
  listTrainingJobs,
} from './training.service';

/** Namespace import: `import { api } from 'src/services'` */
export const api = {
  getHealth,
  getRootMeta,
  getApiList,
  uploadFile,
  listFiles,
  deleteFile,
  uploadDataset,
  listDatasets,
  getDatasetPreview,
  deleteDataset,
  startTraining,
  listTrainingJobs,
  rebuildTraining,
  getTraining,
  getTrainingJob,
  listModels,
  uploadPredictionInput,
  listPredictionInputs,
  uploadPredictionCsv,
  startPrediction,
  getPrediction,
  getPredictionJob,
  agentDecide,
  listAgentReports,
  getAgentReport,
  kbUpload,
  kbListFiles,
  kbList,
  kbDelete,
  kbQuery,
  kbQueryMulti,
  kbRagTemplatesLatestPrediction,
  kbRagLlm,
};
