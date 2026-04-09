import type {
  AgenticJobOut,
  AgenticReportOut,
  AgenticDecideRequest,
  TrustAnchorVerifyOut,
  TrustAnchorListItemOut,
  AgenticPromptPreviewOut,
  AgenticJobCreateRequest,
  ExecutionReportDetailOut,
  ExecutionReportListItemOut,
} from 'src/api/types';

import { paths } from './paths';
import { requestJson, requestVoid } from './http-client';

export async function agentDecide(body: AgenticDecideRequest): Promise<AgenticReportOut> {
  return requestJson<AgenticReportOut>(paths.agent.decide, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function agentDecidePromptPreview(body: AgenticDecideRequest): Promise<AgenticPromptPreviewOut> {
  return requestJson<AgenticPromptPreviewOut>(paths.agent.decidePromptPreview, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

const AGENT_REPORTS_PAGE = 500;
const AGENT_JOBS_PAGE = 500;

export async function listAgenticJobs(limit = 100, offset = 0): Promise<AgenticJobOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<AgenticJobOut[]>(`${paths.agent.jobs}?${q}`);
}

/** All agentic jobs (newest first), paged until the API returns a short batch. */
export async function listAllAgenticJobs(): Promise<AgenticJobOut[]> {
  const out: AgenticJobOut[] = [];
  for (let offset = 0; ; offset += AGENT_JOBS_PAGE) {
    const batch = await listAgenticJobs(AGENT_JOBS_PAGE, offset);
    out.push(...batch);
    if (batch.length < AGENT_JOBS_PAGE) break;
  }
  return out;
}

export async function createAgenticJob(body: AgenticJobCreateRequest): Promise<AgenticJobOut> {
  return requestJson<AgenticJobOut>(paths.agent.jobs, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export type ListAgentReportsOpts = {
  /** When set, GET /agent/reports?agentic_job_public_id=… (404 if unknown id). */
  agentic_job_public_id?: string | null;
};

export async function listAgentReports(
  limit = 100,
  offset = 0,
  opts?: ListAgentReportsOpts
): Promise<AgenticReportOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const aid = opts?.agentic_job_public_id?.trim();
  if (aid) q.set('agentic_job_public_id', aid);
  return requestJson<AgenticReportOut[]>(`${paths.agent.reports}?${q}`);
}

/** All agentic reports (newest first), paged until the API returns a short batch. */
export async function listAllAgentReports(opts?: ListAgentReportsOpts): Promise<AgenticReportOut[]> {
  const out: AgenticReportOut[] = [];
  for (let offset = 0; ; offset += AGENT_REPORTS_PAGE) {
    const batch = await listAgentReports(AGENT_REPORTS_PAGE, offset, opts);
    out.push(...batch);
    if (batch.length < AGENT_REPORTS_PAGE) break;
  }
  return out;
}

export async function getAgentReport(publicId: string): Promise<AgenticReportOut> {
  return requestJson<AgenticReportOut>(paths.agent.reportByPublicId(publicId));
}

export async function deleteAgentReport(publicId: string): Promise<void> {
  return requestVoid(paths.agent.reportByPublicId(publicId), { method: 'DELETE' });
}

export async function listTrustAnchors(limit = 100, offset = 0): Promise<TrustAnchorListItemOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<TrustAnchorListItemOut[]>(`${paths.agent.trustAnchors}?${q}`);
}

export async function verifyTrustAnchor(anchorId: number): Promise<TrustAnchorVerifyOut> {
  return requestJson<TrustAnchorVerifyOut>(paths.agent.trustAnchorVerify(anchorId));
}

export async function applyAgenticReport(publicId: string): Promise<ExecutionReportDetailOut> {
  return requestJson<ExecutionReportDetailOut>(paths.agent.applyReport(publicId), { method: 'POST' });
}

export async function listExecutionReports(limit = 100, offset = 0): Promise<ExecutionReportListItemOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<ExecutionReportListItemOut[]>(`${paths.agent.executionReports}?${q}`);
}

export async function getExecutionReport(id: number): Promise<ExecutionReportDetailOut> {
  return requestJson<ExecutionReportDetailOut>(paths.agent.executionReportById(id));
}
