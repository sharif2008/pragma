import type { AgenticReportOut } from 'src/api/types';

export type PlanAction = {
  action: string;
  network_tier: string;
  party_evidence_type?: string;
  reasoning: string;
  kind: 'primary' | 'supporting';
};

export function parseStructuredPlan(report: AgenticReportOut): Record<string, unknown> | null {
  const art = report.report_artifact;
  if (art && typeof art === 'object' && art !== null) {
    const sp = (art as { structured_plan?: unknown }).structured_plan;
    if (sp && typeof sp === 'object' && sp !== null) return sp as Record<string, unknown>;
  }
  const raw = report.raw_llm_response;
  if (!raw || typeof raw !== 'string') return null;
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) return null;
  try {
    const parsed = JSON.parse(m[0]);
    return typeof parsed === 'object' && parsed !== null ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export function collectPlanActions(plan: Record<string, unknown> | null): PlanAction[] {
  if (!plan) return [];
  const out: PlanAction[] = [];
  const push = (x: unknown, kind: 'primary' | 'supporting') => {
    if (!x || typeof x !== 'object') return;
    const o = x as Record<string, unknown>;
    out.push({
      action: String(o.action ?? '—'),
      network_tier: String(o.network_tier ?? ''),
      party_evidence_type: o.party_evidence_type != null ? String(o.party_evidence_type) : undefined,
      reasoning: String(o.reasoning ?? ''),
      kind,
    });
  };
  const prim = plan.primary_actions;
  const sup = plan.supporting_actions;
  if (Array.isArray(prim)) prim.forEach((a) => push(a, 'primary'));
  if (Array.isArray(sup)) sup.forEach((a) => push(a, 'supporting'));
  return out;
}

export function actionsFromReport(report: AgenticReportOut): PlanAction[] {
  const fromPlan = collectPlanActions(parseStructuredPlan(report));
  if (fromPlan.length) return fromPlan;
  const fallback = String(report.recommended_action || '').trim();
  if (!fallback) return [];
  return [{ action: fallback, network_tier: '', reasoning: '', kind: 'primary' }];
}

export function attackTypeFromReportRow(report: AgenticReportOut): string | null {
  const ex = report.execution_report;
  if (ex && typeof ex === 'object' && ex.attack_type) return String(ex.attack_type);
  const art = report.report_artifact;
  if (art && typeof art === 'object' && art !== null) {
    const sample = (art as { sample_data?: unknown }).sample_data;
    if (sample && typeof sample === 'object' && sample !== null) {
      const label = (sample as { predicted_label?: unknown }).predicted_label;
      if (label != null && String(label).trim()) return String(label).trim();
    }
  }
  return null;
}

export function execStatusLabel(report: AgenticReportOut): 'applied' | 'failed' | 'none' {
  const ex = report.execution_report;
  if (!ex || typeof ex !== 'object') return 'none';
  const status = String(ex.status ?? '');
  if (status === 'applied') return 'applied';
  if (status === 'failed') return 'failed';
  return 'none';
}
