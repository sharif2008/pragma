import type { AgenticReportOut, AgenticDecideRequest } from 'src/api/types';

import { paths } from './paths';
import { requestJson } from './http-client';

export async function agentDecide(body: AgenticDecideRequest): Promise<AgenticReportOut> {
  return requestJson<AgenticReportOut>(paths.agent.decide, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export async function listAgentReports(limit = 100, offset = 0): Promise<AgenticReportOut[]> {
  const q = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return requestJson<AgenticReportOut[]>(`${paths.agent.reports}?${q}`);
}

export async function getAgentReport(publicId: string): Promise<AgenticReportOut> {
  return requestJson<AgenticReportOut>(paths.agent.reportByPublicId(publicId));
}
