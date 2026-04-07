export * from './kb.service';
export { paths } from './paths';
export * from './files.service';

export * from './agent.service';
export * from './health.service';
export * from './datasets.service';
export * from './training.service';
export * from './predictions.service';
export * from './runs.service';
export { getApiBaseUrl, DEFAULT_API_BASE_URL } from './api-base';
export { ApiError, requestJson, requestVoid, postMultipart, type RequestInitSubset } from './http-client';

import { getRun, getRunEvents, listRunEvents, listRuns } from './runs.service';
import { listFiles, deleteFile, uploadFile } from './files.service';
import { getHealth, getApiList, getRootMeta } from './health.service';
import { agentDecide, getAgentReport, listAgentReports } from './agent.service';
import { listDatasets, deleteDataset, uploadDataset, getDatasetPreview } from './datasets.service';
import {
  listModels,
  deleteModel,
  getTraining,
  startTraining,
  getTrainingJob,
  rebuildTraining,
  listTrainingJobs,
  deleteTrainingJob,
} from './training.service';
import {
  kbList,
  kbQuery,
  kbDelete,
  kbRagLlm,
  kbUpload,
  kbListFiles,
  kbFuseHitsMmr,
  kbRagTemplatesLatestPrediction,
  kbLlmShapRetrievalQuery,
} from './kb.service';
import {
  getPrediction,
  startPrediction,
  getPredictionJob,
  deletePredictionJob,
  deleteAllPendingPredictionJobs,
  uploadPredictionCsv,
  listPredictionInputs,
  uploadPredictionInput,
} from './predictions.service';

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
  deleteTrainingJob,
  listModels,
  deleteModel,
  uploadPredictionInput,
  listPredictionInputs,
  uploadPredictionCsv,
  startPrediction,
  getPrediction,
  getPredictionJob,
  deletePredictionJob,
  deleteAllPendingPredictionJobs,
  listRuns,
  getRun,
  getRunEvents,
  listRunEvents,
  agentDecide,
  listAgentReports,
  getAgentReport,
  kbUpload,
  kbListFiles,
  kbList,
  kbDelete,
  kbQuery,
  kbFuseHitsMmr,
  kbRagTemplatesLatestPrediction,
  kbRagLlm,
  kbLlmShapRetrievalQuery,
};
