import type { AgenticReportOut, PredictionJobOut } from 'src/api/types';
import type { PredictionJobListItem } from 'src/services/predictions.service';

import { useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import Checkbox from '@mui/material/Checkbox';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime } from 'src/utils/table-time-sort';

import { useAppSnackbar } from 'src/contexts/app-snackbar-context';
import {
  ApiError,
  agentDecide,
  getAgentReport,
  getPredictionJob,
  deleteAgentReport,
  listAllAgentReports,
  listAllPredictionJobs,
  agentDecidePromptPreview,
} from 'src/services';

import { Iconify } from 'src/components/iconify';
import { GlobalActionLoadingOverlay } from 'src/components/loading/global-action-loading-overlay';

// ----------------------------------------------------------------------

type Notify = (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => void;

type PlanAction = {
  action: string;
  network_tier: string;
  party_evidence_type?: string;
  reasoning: string;
  kind: 'primary' | 'supporting';
};

type ShapChip = { feature: string; value: number };

type DetailState = {
  rowIndex: number;
  predictedLabel?: string;
  confidence?: number;
  contribution?: string;
  shapChips?: ShapChip[];
  ragContext?: string | null;
  prompt?: string | null;
  report?: AgenticReportOut | null;
};

type BusyKind = 'rag' | 'actions' | null;

const TIER_DOMAIN: Record<string, string> = {
  RAN: 'Access / ISP',
  Edge: 'Perimeter / IDS',
  Core: 'Endpoint / EDR',
  'Access / ISP': 'Access / ISP',
  'Perimeter / IDS': 'Perimeter / IDS',
  'Endpoint / EDR': 'Endpoint / EDR',
};

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

function parseRowSelection(text: string, maxRow: number): number[] {
  const raw = text.trim();
  if (!raw) return [];
  const out = new Set<number>();
  for (const part of raw.split(/[,;\s]+/).filter(Boolean)) {
    const range = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (range) {
      let a = Number(range[1]);
      let b = Number(range[2]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
      if (a > b) [a, b] = [b, a];
      for (let i = a; i <= b; i += 1) {
        if (i >= 0 && (maxRow < 0 || i <= maxRow)) out.add(i);
      }
      continue;
    }
    const n = Number(part);
    if (Number.isFinite(n) && n >= 0 && (maxRow < 0 || n <= maxRow)) out.add(Math.floor(n));
  }
  return [...out].sort((a, b) => a - b);
}

function domainLabel(tier: string): string {
  const key = String(tier || '').trim();
  if (!key) return '—';
  return TIER_DOMAIN[key] || key;
}

function isAttackLike(r: { predicted_label?: string; flagged_attack_or_anomaly?: boolean }): boolean {
  if (r.flagged_attack_or_anomaly) return true;
  const label = String(r.predicted_label ?? '')
    .trim()
    .toUpperCase();
  return Boolean(label) && !['BENIGN', 'NORMAL', 'LEGITIMATE', '0', 'FALSE', 'NO', 'NONE', ''].includes(label);
}

function topShapEntries(shap: Record<string, unknown> | undefined, topN = 5): ShapChip[] {
  if (!shap || typeof shap !== 'object') return [];
  const pf = shap.per_feature;
  if (!pf || typeof pf !== 'object') return [];
  return Object.entries(pf as Record<string, unknown>)
    .map(([k, v]) => {
      const n = Number(v);
      return Number.isFinite(n) ? { feature: k, value: n } : null;
    })
    .filter((x): x is ShapChip => !!x)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, topN);
}

function contributionSummary(shap: Record<string, unknown> | undefined): string {
  const top = topShapEntries(shap, 5);
  if (!top.length) {
    if (shap && typeof shap === 'object' && typeof shap.status === 'string') return String(shap.status);
    return 'No contribution features available for this row.';
  }
  return top
    .map((t) => {
      const sign = t.value >= 0 ? '+' : '';
      return `${t.feature} ${sign}${t.value.toFixed(4)}`;
    })
    .join(' · ');
}

function parseStructuredPlan(report: AgenticReportOut): Record<string, unknown> | null {
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

function collectActions(plan: Record<string, unknown> | null): PlanAction[] {
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

/** Split RAG text into chunk-like blocks for display. */
function parseRagChunks(
  rag: string | null | undefined
): { title: string; body: string; source: string | null; sim: string | null; rerank: string | null }[] {
  const text = (rag || '').trim();
  if (!text) return [];

  const metaFrom = (b: string, i: number) => {
    const sim = b.match(/sim=([\d.]+)/)?.[1] ?? null;
    const rerank = b.match(/rerank=([\d.]+)/)?.[1] ?? null;
    const source =
      b.match(/source=([^)\]]+)/)?.[1]?.trim() ||
      b.match(/\[([^\]]+\.(?:pdf|xlsx?|docx?|txt|md|csv))\]/i)?.[1]?.trim() ||
      null;
    const title = sim
      ? `Hit ${i + 1} · sim ${sim}${rerank ? ` · rerank ${rerank}` : ''}${source ? ` · ${source}` : ''}`
      : source
        ? `Hit ${i + 1} · ${source}`
        : `RAG chunk ${i + 1}`;
    const body = b
      .replace(/^- \(sim=[^)]+\)\s*/, '')
      .replace(/^\[\d+\][^\n]*\n?/, '')
      .slice(0, 1200);
    return { title, body, source, sim, rerank };
  };

  const blocks = text
    .split(/\n(?==== |\n- \(sim=|\[(?:\d+)\] )/g)
    .map((s) => s.trim())
    .filter(Boolean);
  if (blocks.length <= 1) {
    const bullets = text
      .split(/\n(?=- \(sim=)/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (bullets.length > 1) {
      return bullets.map((b, i) => metaFrom(b, i));
    }
    return [{ title: 'RAG context', body: text.slice(0, 4000), source: null, sim: null, rerank: null }];
  }
  return blocks.map((b, i) => {
    const parsed = metaFrom(b, i);
    if (parsed.sim || parsed.source) return parsed;
    const head = b.split('\n')[0]?.slice(0, 80) || `Chunk ${i + 1}`;
    return {
      title: head.replace(/^===\s*/, '').replace(/\s*===\s*$/, '') || `Chunk ${i + 1}`,
      body: b.slice(0, 1200),
      source: null,
      sim: null,
      rerank: null,
    };
  });
}

function uniqueRagDocuments(
  chunks: { source: string | null }[]
): { name: string; hitCount: number }[] {
  const counts = new Map<string, number>();
  for (const c of chunks) {
    const name = (c.source || '').trim();
    if (!name) continue;
    counts.set(name, (counts.get(name) || 0) + 1);
  }
  return [...counts.entries()]
    .map(([name, hitCount]) => ({ name, hitCount }))
    .sort((a, b) => b.hitCount - a.hitCount || a.name.localeCompare(b.name));
}

/** Prefer RAG chunks that mention the action name or party features. */
function ragForAction(
  chunks: { title: string; body: string; source: string | null }[],
  action: PlanAction
): { title: string; body: string; source: string | null }[] {
  if (!chunks.length) return [];
  const needles = [action.action, ...(action.party_evidence_type || '').split(/[,;]/).map((s) => s.trim())].filter(
    (s) => s.length > 2
  );
  const scored = chunks.map((c) => {
    const hay = `${c.title}\n${c.body}\n${c.source || ''}`.toLowerCase();
    let score = 0;
    for (const n of needles) {
      if (hay.includes(n.toLowerCase())) score += 1;
    }
    return { c, score };
  });
  scored.sort((a, b) => b.score - a.score);
  const matched = scored.filter((x) => x.score > 0).map((x) => x.c);
  if (matched.length) return matched.slice(0, 3);
  return chunks.slice(0, 2);
}

function promptFromReport(report: AgenticReportOut | null | undefined): string | null {
  if (!report?.report_artifact || typeof report.report_artifact !== 'object') return null;
  const p = report.report_artifact.user_prompt;
  return typeof p === 'string' ? p : null;
}

function ragFromReport(report: AgenticReportOut | null | undefined): string | null {
  if (!report) return null;
  if (report.rag_context_used) return report.rag_context_used;
  const art = report.report_artifact;
  if (art && typeof art.rag_context_used === 'string') return art.rag_context_used;
  return null;
}

// ----------------------------------------------------------------------

function PlanDetailsDialog({
  open,
  onClose,
  detail,
  loading,
}: {
  open: boolean;
  onClose: () => void;
  detail: DetailState | null;
  loading: boolean;
}) {
  const report = detail?.report ?? null;
  const plan = useMemo(() => (report ? parseStructuredPlan(report) : null), [report]);
  const actions = useMemo(() => collectActions(plan), [plan]);
  const ragText = detail?.ragContext ?? ragFromReport(report) ?? '';
  const ragChunks = useMemo(() => parseRagChunks(ragText), [ragText]);
  const ragDocuments = useMemo(() => uniqueRagDocuments(ragChunks), [ragChunks]);
  const llmPrompt = detail?.prompt ?? promptFromReport(report);
  const threat = plan && plan.threat_level != null ? String(plan.threat_level) : '';
  const priority = plan && plan.execution_priority != null ? String(plan.execution_priority) : '';
  const allActions = Array.isArray(plan?.all_actions)
    ? (plan!.all_actions as unknown[]).map(String)
    : actions.map((a) => a.action);
  const shapChips = detail?.shapChips ?? [];
  const confPct =
    detail?.confidence != null && Number.isFinite(detail.confidence)
      ? `${(detail.confidence * 100).toFixed(1)}%`
      : null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ py: 1.5 }}>
        Plan details
        {detail ? (
          <Typography variant="caption" color="text.secondary" display="block">
            Row {detail.rowIndex}
            {report?.prediction_job_public_id
              ? ` · ${(report.prediction_job_public_id || '').slice(0, 8)}…`
              : ''}
            {report?.public_id ? ` · ${report.public_id.slice(0, 8)}…` : ''}
          </Typography>
        ) : null}
      </DialogTitle>
      <DialogContent dividers sx={{ maxHeight: '78vh' }}>
        {loading && (
          <Stack alignItems="center" py={4}>
            <CircularProgress size={28} />
          </Stack>
        )}
        {!loading && !detail && (
          <Alert severity="warning">Could not load plan details.</Alert>
        )}
        {!loading && detail && (
          <Stack spacing={2}>
            <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
              {threat && <Chip size="small" color="error" label={`Threat: ${threat}`} />}
              {priority && <Chip size="small" color="warning" variant="outlined" label={`Priority: ${priority}`} />}
              {report?.trust_commitment && (
                <Chip size="small" color="success" variant="outlined" label="Trust anchored" />
              )}
              <Chip size="small" variant="outlined" label={`Row ${detail.rowIndex}`} />
              {actions.length > 0 && (
                <Chip size="small" variant="outlined" label={`${actions.length} action(s)`} />
              )}
            </Stack>

            {/* Prediction */}
            <Box>
              <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.6 }}>
                Prediction
              </Typography>
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 0.5, mb: 0.75 }}>
                <Chip
                  size="small"
                  label={detail.predictedLabel || '—'}
                  color={isAttackLike({ predicted_label: detail.predictedLabel }) ? 'error' : 'success'}
                  variant={isAttackLike({ predicted_label: detail.predictedLabel }) ? 'filled' : 'outlined'}
                />
                {confPct && <Chip size="small" variant="outlined" label={`Conf. ${confPct}`} />}
              </Stack>
              {detail.contribution && (
                <Typography variant="body2" color="text.secondary" sx={{ mb: 0.75 }}>
                  {detail.contribution}
                </Typography>
              )}
              {shapChips.length > 0 && (
                <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                  {shapChips.map((t) => (
                    <Chip
                      key={`detail-shap-${t.feature}`}
                      size="small"
                      variant="outlined"
                      color={t.value >= 0 ? 'warning' : 'default'}
                      label={`${t.feature}: ${t.value >= 0 ? '+' : ''}${t.value.toFixed(3)}`}
                      sx={{ height: 22 }}
                    />
                  ))}
                </Stack>
              )}
            </Box>

            <Divider />

            {/* Actions */}
            <Box>
              <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.6 }}>
                Actions
              </Typography>
              {!report && (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  Generate actions to produce an LLM plan for this row.
                </Typography>
              )}
              {report && (
                <Box sx={{ mt: 0.5 }}>
                  {report.summary && (
                    <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', mb: 0.75 }}>
                      {typeof plan?.overall_reasoning === 'string' ? plan.overall_reasoning : report.summary}
                    </Typography>
                  )}
                  {report.recommended_action && (
                    <Typography variant="body2" sx={{ mb: 0.75 }}>
                      Recommended: <strong>{report.recommended_action}</strong>
                    </Typography>
                  )}
                  {allActions.length > 0 && (
                    <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mb: 0.75 }}>
                      {allActions.map((a) => (
                        <Chip key={a} size="small" label={a} color="primary" variant="outlined" />
                      ))}
                    </Stack>
                  )}
                  {actions.length === 0 && (
                    <Typography variant="body2" color="text.secondary">
                      No structured actions — see LLM section below.
                    </Typography>
                  )}
                  {actions.map((a, idx) => {
                    const relatedRag = ragForAction(ragChunks, a);
                    return (
                      <Paper key={`${a.action}-${idx}`} variant="outlined" sx={{ p: 1.5, borderRadius: 1.5, mb: 1 }}>
                        <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap sx={{ mb: 0.75 }}>
                          <Chip size="small" label={a.kind} color={a.kind === 'primary' ? 'error' : 'default'} />
                          <Chip size="small" color="primary" label={a.action} />
                          <Chip size="small" variant="outlined" label={domainLabel(a.network_tier)} />
                        </Stack>
                        {a.party_evidence_type && (
                          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                            Evidence: {a.party_evidence_type}
                          </Typography>
                        )}
                        <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', mb: relatedRag.length ? 1 : 0 }}>
                          {a.reasoning || 'No reasoning provided.'}
                        </Typography>
                        {relatedRag.length > 0 && (
                          <>
                            <Typography variant="caption" sx={{ fontWeight: 700, display: 'block', mb: 0.5 }}>
                              Related RAG
                            </Typography>
                            <Stack spacing={0.75}>
                              {relatedRag.map((c, i) => (
                                <Box
                                  key={`${idx}-rag-${i}`}
                                  sx={{
                                    p: 1,
                                    borderRadius: 1,
                                    bgcolor: 'background.neutral',
                                    border: 1,
                                    borderColor: 'divider',
                                  }}
                                >
                                  <Typography variant="caption" sx={{ fontWeight: 700, display: 'block', mb: 0.25 }}>
                                    {c.title}
                                  </Typography>
                                  <Typography
                                    variant="caption"
                                    color="text.secondary"
                                    sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', display: 'block' }}
                                  >
                                    {c.body}
                                  </Typography>
                                </Box>
                              ))}
                            </Stack>
                          </>
                        )}
                      </Paper>
                    );
                  })}
                </Box>
              )}
            </Box>

            <Divider />

            {/* RAG */}
            <Box>
              <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.6 }}>
                RAG context
              </Typography>
              {ragChunks.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  No RAG context retrieved.
                </Typography>
              ) : (
                <Stack spacing={1.25} sx={{ mt: 0.5 }}>
                  <Box>
                    <Typography variant="caption" sx={{ fontWeight: 700, display: 'block', mb: 0.5 }}>
                      Documents ({ragDocuments.length || '—'})
                    </Typography>
                    {ragDocuments.length === 0 ? (
                      <Typography variant="caption" color="text.secondary">
                        Source filenames not present in this RAG blob (re-run RAG details / Generate actions after
                        backend update to include them).
                      </Typography>
                    ) : (
                      <Stack spacing={0.5}>
                        {ragDocuments.map((d) => (
                          <Stack
                            key={d.name}
                            direction="row"
                            alignItems="center"
                            justifyContent="space-between"
                            gap={1}
                            sx={{
                              px: 1,
                              py: 0.5,
                              borderRadius: 1,
                              border: 1,
                              borderColor: 'divider',
                              bgcolor: 'background.paper',
                            }}
                          >
                            <Typography
                              variant="caption"
                              sx={{ fontFamily: 'monospace', wordBreak: 'break-all', flex: 1 }}
                              title={d.name}
                            >
                              {d.name}
                            </Typography>
                            <Chip
                              size="small"
                              variant="outlined"
                              label={`${d.hitCount} chunk${d.hitCount === 1 ? '' : 's'}`}
                              sx={{ height: 20, flexShrink: 0 }}
                            />
                          </Stack>
                        ))}
                      </Stack>
                    )}
                  </Box>

                  <Typography variant="caption" sx={{ fontWeight: 700, display: 'block' }}>
                    Chunks ({ragChunks.length})
                  </Typography>
                  {ragChunks.map((c, i) => (
                    <Box
                      key={`rag-chunk-${i}`}
                      sx={{
                        p: 1,
                        borderRadius: 1,
                        bgcolor: 'background.neutral',
                        border: 1,
                        borderColor: 'divider',
                      }}
                    >
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mb: 0.35 }}>
                        <Typography variant="caption" sx={{ fontWeight: 700 }}>
                          {c.title}
                        </Typography>
                        {c.source && (
                          <Chip size="small" variant="outlined" label={c.source} sx={{ height: 18, maxWidth: 220 }} />
                        )}
                      </Stack>
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', display: 'block' }}
                      >
                        {c.body}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              )}
            </Box>

            <Divider />

            {/* LLM — always at bottom */}
            <Box
              sx={{
                p: 1.5,
                borderRadius: 1.5,
                border: 1,
                borderColor: 'divider',
                bgcolor: (t) => alpha(t.palette.grey[500], 0.06),
              }}
            >
              <Typography variant="overline" color="text.secondary" sx={{ letterSpacing: 0.6, display: 'block', mb: 1 }}>
                LLM
              </Typography>

              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
                Query (prompt)
              </Typography>
              <Box
                component="pre"
                sx={{
                  m: 0,
                  mb: 1.5,
                  p: 1.25,
                  maxHeight: 200,
                  overflow: 'auto',
                  bgcolor: 'background.paper',
                  borderRadius: 1,
                  border: 1,
                  borderColor: 'divider',
                  fontSize: 11,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {llmPrompt || '(prompt not available yet)'}
              </Box>

              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
                Response
              </Typography>
              {!report ? (
                <Typography variant="body2" color="text.secondary">
                  No LLM response yet — use Generate actions.
                </Typography>
              ) : (
                <>
                  <Box
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.25,
                      maxHeight: 260,
                      overflow: 'auto',
                      bgcolor: 'background.paper',
                      borderRadius: 1,
                      border: 1,
                      borderColor: 'divider',
                      fontSize: 12,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                    }}
                  >
                    {report.raw_llm_response || '(empty)'}
                  </Box>
                  {plan && (
                    <Accordion
                      disableGutters
                      elevation={0}
                      sx={{ border: 1, borderColor: 'divider', borderRadius: 1, mt: 1, bgcolor: 'background.paper' }}
                    >
                      <AccordionSummary expandIcon={<Iconify icon="eva:arrow-ios-downward-fill" />}>
                        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                          Parsed structured plan
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <Box
                          component="pre"
                          sx={{
                            m: 0,
                            p: 1.25,
                            maxHeight: 240,
                            overflow: 'auto',
                            bgcolor: 'background.neutral',
                            borderRadius: 1,
                            fontSize: 11,
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                          }}
                        >
                          {JSON.stringify(plan, null, 2)}
                        </Box>
                      </AccordionDetails>
                    </Accordion>
                  )}
                </>
              )}
            </Box>
          </Stack>
        )}
      </DialogContent>
      <DialogActions sx={{ py: 1 }}>
        <Button size="small" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

/** Setting → Agentic planner: job/rows/RAG+decide + plans with per-row details. */
export function SettingAgenticPanel({ onNotify }: { onNotify: Notify }) {
  const toast = useAppSnackbar();
  const notify = useCallback(
    (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => {
      onNotify(b);
      if (!b) return;
      if (b.severity === 'success') toast.showSuccess(b.text);
      else if (b.severity === 'error') toast.showError(b.text);
      else toast.showInfo(b.text);
    },
    [onNotify, toast]
  );

  const [jobs, setJobs] = useState<PredictionJobListItem[]>([]);
  const [jobId, setJobId] = useState('');
  const [jobDetail, setJobDetail] = useState<PredictionJobOut | null>(null);
  const [rowMode, setRowMode] = useState<'flagged' | 'all' | 'custom'>('flagged');
  const [rowSpec, setRowSpec] = useState('');
  const [useRag, setUseRag] = useState(true);
  const [anchorChain, setAnchorChain] = useState(true);
  const [running, setRunning] = useState(false);
  const [busyRow, setBusyRow] = useState<number | null>(null);
  const [busyKind, setBusyKind] = useState<BusyKind>(null);
  const [reports, setReports] = useState<AgenticReportOut[]>([]);
  const [selectedPlanIds, setSelectedPlanIds] = useState<Set<string>>(() => new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [predPage, setPredPage] = useState(0);
  const [predRowsPerPage, setPredRowsPerPage] = useState(10);
  const [loading, setLoading] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [loadingOverlayDismissed, setLoadingOverlayDismissed] = useState(false);
  const [detail, setDetail] = useState<DetailState | null>(null);

  const completedJobs = useMemo(
    () => sortByTime(jobs.filter((j) => j.status === 'completed'), (j) => j.created_at, 'desc'),
    [jobs]
  );

  const predictionRows = useMemo(() => {
    const rows = jobDetail?.results_json?.rows;
    return Array.isArray(rows) ? rows : [];
  }, [jobDetail]);

  const maxRowIndex = useMemo(() => {
    if (!predictionRows.length) return -1;
    return Math.max(...predictionRows.map((r) => Number(r.row_index) || 0));
  }, [predictionRows]);

  const scopedPredictionRows = useMemo(() => {
    if (rowMode === 'all') return predictionRows;
    if (rowMode === 'flagged') return predictionRows.filter((r) => isAttackLike(r));
    const want = new Set(parseRowSelection(rowSpec, maxRowIndex));
    return predictionRows.filter((r) => want.has(Number(r.row_index)));
  }, [predictionRows, rowMode, rowSpec, maxRowIndex]);

  const selectedRows = useMemo(
    () => scopedPredictionRows.map((r) => r.row_index),
    [scopedPredictionRows]
  );

  const predPageRows = useMemo(() => {
    const start = predPage * predRowsPerPage;
    return scopedPredictionRows.slice(start, start + predRowsPerPage);
  }, [scopedPredictionRows, predPage, predRowsPerPage]);

  const allSavedPlans = useMemo(
    () => sortByTime(reports, (r) => r.created_at, 'desc'),
    [reports]
  );

  /** Plans for the currently selected prediction job (row “Plan” column only). */
  const jobPlans = useMemo(() => {
    if (!jobId.trim()) return [];
    return allSavedPlans.filter((r) => r.prediction_job_public_id === jobId);
  }, [allSavedPlans, jobId]);

  const pageRows = useMemo(() => {
    const start = page * rowsPerPage;
    return allSavedPlans.slice(start, start + rowsPerPage);
  }, [allSavedPlans, page, rowsPerPage]);

  const pageSelectedCount = useMemo(
    () => pageRows.filter((r) => selectedPlanIds.has(r.public_id)).length,
    [pageRows, selectedPlanIds]
  );
  const allPageSelected = pageRows.length > 0 && pageSelectedCount === pageRows.length;
  const somePageSelected = pageSelectedCount > 0 && !allPageSelected;

  const toggleSelectPlan = (publicId: string, checked: boolean) => {
    setSelectedPlanIds((prev) => {
      const next = new Set(prev);
      if (checked) next.add(publicId);
      else next.delete(publicId);
      return next;
    });
  };

  const toggleSelectPage = (checked: boolean) => {
    setSelectedPlanIds((prev) => {
      const next = new Set(prev);
      for (const r of pageRows) {
        if (checked) next.add(r.public_id);
        else next.delete(r.public_id);
      }
      return next;
    });
  };

  const deleteSelectedPlans = async () => {
    const ids = [...selectedPlanIds];
    if (!ids.length) return;
    if (!window.confirm(`Delete ${ids.length} action plan(s)? This cannot be undone.`)) return;
    setBulkDeleting(true);
    onNotify(null);
    let ok = 0;
    let failed = 0;
    try {
      for (const id of ids) {
        try {
          await deleteAgentReport(id);
          ok += 1;
        } catch {
          failed += 1;
        }
      }
      setSelectedPlanIds(new Set());
      await refreshReports();
      notify({
        severity: failed ? 'info' : 'success',
        text: `Deleted ${ok} plan(s)${failed ? ` · ${failed} failed` : ''}.`,
      });
    } finally {
      setBulkDeleting(false);
    }
  };

  const planForRow = useCallback(
    (rowIndex: number) =>
      sortByTime(
        jobPlans.filter((r) => r.results_row_index === rowIndex),
        (r) => r.created_at,
        'desc'
      )[0] ?? null,
    [jobPlans]
  );

  const rowMeta = useCallback(
    (rowIndex: number) => {
      const r = predictionRows.find((x) => Number(x.row_index) === rowIndex);
      if (!r) {
        return {
          predictedLabel: undefined as string | undefined,
          confidence: undefined as number | undefined,
          contribution: undefined as string | undefined,
          shapChips: [] as ShapChip[],
        };
      }
      const conf = Number(r.max_class_probability);
      return {
        predictedLabel: r.predicted_label,
        confidence: Number.isFinite(conf) ? conf : undefined,
        contribution: contributionSummary(r.shap),
        shapChips: topShapEntries(r.shap, 5),
      };
    },
    [predictionRows]
  );

  const refreshJobs = useCallback(async () => {
    setLoading(true);
    try {
      const list = await listAllPredictionJobs();
      setJobs(list);
      setJobId((prev) => {
        if (prev && list.some((j) => j.public_id === prev && j.status === 'completed')) return prev;
        const newest = sortByTime(
          list.filter((j) => j.status === 'completed'),
          (j) => j.created_at,
          'desc'
        )[0];
        return newest?.public_id ?? '';
      });
    } catch (e) {
      notify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoading(false);
    }
  }, [notify]);

  const refreshReports = useCallback(async () => {
    try {
      setReports(await listAllAgentReports());
    } catch (e) {
      notify({ severity: 'error', text: formatError(e) });
    }
  }, [notify]);

  useEffect(() => {
    void refreshJobs();
    void refreshReports();
  }, [refreshJobs, refreshReports]);

  useEffect(() => {
    const id = jobId.trim();
    if (!id) {
      setJobDetail(null);
      return () => {};
    }
    let cancelled = false;
    (async () => {
      try {
        const full = await getPredictionJob(id, { includeResults: true });
        if (!cancelled) setJobDetail(full);
      } catch (e) {
        if (!cancelled) notify({ severity: 'error', text: formatError(e) });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId, notify]);

  useEffect(() => {
    setPage(0);
  }, [allSavedPlans.length]);

  useEffect(() => {
    const alive = new Set(allSavedPlans.map((r) => r.public_id));
    setSelectedPlanIds((prev) => {
      let changed = false;
      const next = new Set<string>();
      for (const id of prev) {
        if (alive.has(id)) next.add(id);
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [allSavedPlans]);

  useEffect(() => {
    setPredPage(0);
  }, [jobId, rowMode, rowSpec, scopedPredictionRows.length]);

  const openDetailState = (next: DetailState, showLoading = false) => {
    setDetail(next);
    setDetailOpen(true);
    setDetailLoading(showLoading);
  };

  const openSavedPlanDetails = async (listRow: AgenticReportOut) => {
    const ri = Number(listRow.results_row_index ?? 0);
    const meta = rowMeta(ri);
    openDetailState(
      {
        rowIndex: ri,
        predictedLabel: meta.predictedLabel,
        confidence: meta.confidence,
        contribution: meta.contribution,
        shapChips: meta.shapChips,
        ragContext: ragFromReport(listRow),
        prompt: promptFromReport(listRow),
        report: listRow,
      },
      true
    );
    try {
      const full = await getAgentReport(listRow.public_id);
      const fullMeta = rowMeta(Number(full.results_row_index ?? ri));
      setDetail({
        rowIndex: Number(full.results_row_index ?? ri),
        predictedLabel: fullMeta.predictedLabel,
        confidence: fullMeta.confidence,
        contribution: fullMeta.contribution,
        shapChips: fullMeta.shapChips,
        ragContext: ragFromReport(full),
        prompt: promptFromReport(full),
        report: full,
      });
    } catch (e) {
      notify({ severity: 'error', text: formatError(e) });
    } finally {
      setDetailLoading(false);
    }
  };

  const fetchRagDetails = async (rowIndex: number) => {
    if (!jobId.trim()) return;
    const meta = rowMeta(rowIndex);
    setBusyRow(rowIndex);
    setBusyKind('rag');
    onNotify(null);
    openDetailState(
      {
        rowIndex,
        predictedLabel: meta.predictedLabel,
        confidence: meta.confidence,
        contribution: meta.contribution,
        shapChips: meta.shapChips,
        ragContext: null,
        prompt: null,
        report: null,
      },
      true
    );
    try {
      const preview = await agentDecidePromptPreview({
        prediction_job_public_id: jobId.trim(),
        results_row_index: rowIndex,
        use_rag: useRag,
        agent_action_preset: 'standard',
      });
      const conf =
        preview.confidence != null && Number.isFinite(Number(preview.confidence))
          ? Number(preview.confidence)
          : meta.confidence;
      setDetail({
        rowIndex: preview.results_row_index ?? rowIndex,
        predictedLabel: preview.predicted_label ?? meta.predictedLabel,
        confidence: conf,
        contribution: meta.contribution,
        shapChips: meta.shapChips,
        ragContext: preview.rag_context ?? null,
        prompt: preview.prompt ?? null,
        report: null,
      });
      notify({
        severity: 'success',
        text: `RAG details ready for row ${rowIndex}.`,
      });
    } catch (e) {
      notify({ severity: 'error', text: formatError(e) });
    } finally {
      setDetailLoading(false);
      setBusyRow(null);
      setBusyKind(null);
    }
  };

  const generateActionsForRow = async (rowIndex: number) => {
    if (!jobId.trim()) return;
    const meta = rowMeta(rowIndex);
    setBusyRow(rowIndex);
    setBusyKind('actions');
    setRunning(true);
    onNotify(null);
    try {
      const created = await agentDecide({
        prediction_job_public_id: jobId.trim(),
        results_row_index: rowIndex,
        use_rag: useRag,
        anchor_trust_chain: anchorChain,
        agent_action_preset: 'standard',
      });
      await refreshReports();
      openDetailState(
        {
          rowIndex,
          predictedLabel: meta.predictedLabel,
          confidence: meta.confidence,
          contribution: meta.contribution,
          shapChips: meta.shapChips,
          ragContext: ragFromReport(created),
          prompt: promptFromReport(created),
          report: created,
        },
        true
      );
      const full = await getAgentReport(created.public_id);
      setDetail({
        rowIndex: Number(full.results_row_index ?? rowIndex),
        predictedLabel: meta.predictedLabel,
        confidence: meta.confidence,
        contribution: meta.contribution,
        shapChips: meta.shapChips,
        ragContext: ragFromReport(full),
        prompt: promptFromReport(full),
        report: full,
      });
      notify({
        severity: 'success',
        text: `Generated actions for row ${rowIndex} · plan ${created.public_id.slice(0, 8)}…`,
      });
    } catch (e) {
      notify({ severity: 'error', text: formatError(e) });
    } finally {
      setDetailLoading(false);
      setBusyRow(null);
      setBusyKind(null);
      setRunning(false);
    }
  };

  const generateForFiltered = async () => {
    if (!jobId.trim() || !selectedRows.length) return;
    if (
      !window.confirm(
        `Generate actions for ${selectedRows.length} row(s) one-by-one (separate plan per row)?` +
          (anchorChain ? ' (blockchain trust anchor when enabled)' : '')
      )
    ) {
      return;
    }
    setRunning(true);
    onNotify(null);
    let ok = 0;
    let failed = 0;
    let lastOk: AgenticReportOut | null = null;
    try {
      for (const rowIndex of selectedRows) {
        setBusyRow(rowIndex);
        setBusyKind('actions');
        try {
          lastOk = await agentDecide({
            prediction_job_public_id: jobId.trim(),
            results_row_index: rowIndex,
            use_rag: useRag,
            anchor_trust_chain: anchorChain,
            agent_action_preset: 'standard',
          });
          ok += 1;
        } catch {
          failed += 1;
        }
      }
      await refreshReports();
      notify({
        severity: failed ? 'error' : 'success',
        text: `Generated ${ok} plan(s)${failed ? ` · ${failed} failed` : ''} — open Details per row.`,
      });
      if (lastOk) await openSavedPlanDetails(lastOk);
    } finally {
      setBusyRow(null);
      setBusyKind(null);
      setRunning(false);
    }
  };

  const anyBusy = running || busyRow != null;

  const actionLoading = anyBusy || bulkDeleting || detailLoading;

  useEffect(() => {
    if (actionLoading) setLoadingOverlayDismissed(false);
  }, [actionLoading]);

  const actionLoadingMessage = useMemo(() => {
    if (bulkDeleting) return 'Deleting action plans…';
    if (detailLoading && !anyBusy) return 'Loading plan details…';
    if (busyKind === 'rag') return 'Loading RAG details…';
    if (busyKind === 'actions' || running) return 'Generating detection actions…';
    return 'Processing…';
  }, [bulkDeleting, detailLoading, anyBusy, busyKind, running]);

  const actionLoadingSubmessage = useMemo(() => {
    if (bulkDeleting) return `${selectedPlanIds.size} plan(s)`;
    if (busyRow != null) return `Row ${busyRow}`;
    if (running && selectedRows.length > 1) return `${selectedRows.length} rows in queue`;
    return undefined;
  }, [bulkDeleting, busyRow, running, selectedPlanIds.size, selectedRows.length]);

  return (
    <Stack spacing={1.5}>
      <GlobalActionLoadingOverlay
        open={actionLoading && !loadingOverlayDismissed}
        onClose={() => setLoadingOverlayDismissed(true)}
        message={actionLoadingMessage}
        submessage={actionLoadingSubmessage}
      />
      <Paper variant="outlined" sx={{ p: 1.5, borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
          Agentic planner
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Pick a completed prediction job, review each row, then use <strong>RAG details</strong> or{' '}
          <strong>Generate actions</strong>. Plans save into the bottom table. Turn on <strong>Chain</strong> in
          Prediction rows to attach trust anchors.
        </Typography>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <TextField
            select
            size="small"
            label="Job"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            sx={{ minWidth: 220, flex: 1 }}
            disabled={loading || anyBusy}
          >
            {completedJobs.length === 0 && (
              <MenuItem value="" disabled>
                Score a CSV first
              </MenuItem>
            )}
            {completedJobs.map((j) => (
              <MenuItem key={j.public_id} value={j.public_id}>
                {j.public_id.slice(0, 8)}… · {j.rows_flagged ?? 0}/{j.rows_total ?? '?'} · {fDateTime(j.created_at)}
              </MenuItem>
            ))}
          </TextField>
          <TextField
            select
            size="small"
            label="Filter rows"
            value={rowMode}
            onChange={(e) => setRowMode(e.target.value as typeof rowMode)}
            sx={{ minWidth: 140 }}
            disabled={anyBusy}
          >
            <MenuItem value="flagged">Flagged</MenuItem>
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="custom">Custom</MenuItem>
          </TextField>
          {rowMode === 'custom' && (
            <TextField
              size="small"
              label="e.g. 0,2,5-8"
              value={rowSpec}
              onChange={(e) => setRowSpec(e.target.value)}
              sx={{ minWidth: 140, flex: 1 }}
              disabled={anyBusy}
            />
          )}
          <Chip size="small" label={`${scopedPredictionRows.length} in filter`} variant="outlined" />
          <FormControlLabel
            sx={{ m: 0 }}
            control={
              <Checkbox size="small" checked={useRag} onChange={(_, v) => setUseRag(v)} disabled={anyBusy} />
            }
            label={<Typography variant="caption">RAG</Typography>}
          />
          <Button
            size="small"
            variant="outlined"
            disabled={!jobId || !selectedRows.length || anyBusy}
            onClick={() => void generateForFiltered()}
          >
            {running && busyKind === 'actions' && busyRow != null
              ? `Generating… (${busyRow})`
              : `Generate for filtered (${selectedRows.length})`}
          </Button>
          <Button
            size="small"
            onClick={() => {
              void (async () => {
                await refreshJobs();
                const plans = await listAllAgentReports();
                setReports(plans);
                notify({
                  severity: 'success',
                  text: `Refreshed · ${plans.length} plan(s)`,
                });
              })();
            }}
            disabled={loading || anyBusy}
          >
            Refresh
          </Button>
          {(loading || anyBusy) && <CircularProgress size={18} />}
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" useFlexGap gap={1} sx={{ mb: 0.75 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            Prediction rows
          </Typography>
          <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap" useFlexGap>
            <FormControlLabel
              sx={{ m: 0 }}
              control={
                <Checkbox
                  size="small"
                  checked={anchorChain}
                  onChange={(_, v) => setAnchorChain(v)}
                  disabled={anyBusy}
                />
              }
              label={
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  Chain
                </Typography>
              }
            />
            <Chip
              size="small"
              color={anchorChain ? 'success' : 'default'}
              variant={anchorChain ? 'filled' : 'outlined'}
              label={anchorChain ? 'Chain on · save + trust' : 'Chain off · save only'}
              sx={{ height: 22 }}
            />
            <Typography variant="caption" color="text.secondary">
              {scopedPredictionRows.length} shown
            </Typography>
          </Stack>
        </Stack>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.75 }}>
          Generate actions saves a plan automatically. Keep <strong>Chain</strong> checked to attach a trust anchor for
          Agentic execution.
        </Typography>
        <TableContainer sx={{ maxHeight: 360 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ py: 0.75 }}>Row</TableCell>
                <TableCell sx={{ py: 0.75 }}>Predicted</TableCell>
                <TableCell sx={{ py: 0.75 }}>Conf.</TableCell>
                <TableCell sx={{ py: 0.75 }}>Contribution</TableCell>
                <TableCell sx={{ py: 0.75 }}>Plan</TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  RAG details
                </TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Generate actions
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {predPageRows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} sx={{ py: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {jobId
                        ? 'No rows in this filter — change filter or score a CSV first.'
                        : 'Select a prediction job.'}
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {predPageRows.map((r) => {
                const ri = Number(r.row_index);
                const attack = isAttackLike(r);
                const existing = planForRow(ri);
                const summary = contributionSummary(r.shap);
                const top = topShapEntries(r.shap, 5);
                const rowBusy = busyRow === ri;
                const ragBusy = rowBusy && busyKind === 'rag';
                const actionsBusy = rowBusy && busyKind === 'actions';
                return (
                  <TableRow key={ri} hover selected={rowBusy}>
                    <TableCell sx={{ py: 0.5 }}>{ri}</TableCell>
                    <TableCell sx={{ py: 0.5 }}>
                      <Chip
                        size="small"
                        label={r.predicted_label || '—'}
                        color={attack ? 'error' : 'success'}
                        variant={attack ? 'filled' : 'outlined'}
                        sx={{ height: 22 }}
                      />
                    </TableCell>
                    <TableCell sx={{ py: 0.5, typography: 'caption' }}>
                      {Number.isFinite(Number(r.max_class_probability))
                        ? `${(Number(r.max_class_probability) * 100).toFixed(1)}%`
                        : '—'}
                    </TableCell>
                    <TableCell sx={{ py: 0.5, maxWidth: 320 }}>
                      <Typography variant="caption" color="text.secondary" display="block" noWrap title={summary}>
                        {summary}
                      </Typography>
                      {top.length > 0 && (
                        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 0.35 }}>
                          {top.slice(0, 3).map((t) => (
                            <Chip
                              key={`${ri}-${t.feature}`}
                              size="small"
                              variant="outlined"
                              color={t.value >= 0 ? 'warning' : 'default'}
                              label={`${t.feature}: ${t.value >= 0 ? '+' : ''}${t.value.toFixed(3)}`}
                              sx={{ height: 20, fontSize: 10 }}
                            />
                          ))}
                        </Stack>
                      )}
                    </TableCell>
                    <TableCell sx={{ py: 0.5, fontFamily: 'monospace', fontSize: 11 }}>
                      {existing ? `${existing.public_id.slice(0, 8)}…` : '—'}
                    </TableCell>
                    <TableCell align="right" sx={{ py: 0.5 }}>
                      <Button
                        size="small"
                        variant="outlined"
                        sx={{ minWidth: 0, px: 0.75 }}
                        disabled={!jobId || anyBusy}
                        onClick={() => void fetchRagDetails(ri)}
                      >
                        {ragBusy ? '…' : 'RAG details'}
                      </Button>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 0.5 }}>
                      <Button
                        size="small"
                        variant="contained"
                        sx={{ minWidth: 0, px: 0.75 }}
                        disabled={!jobId || anyBusy}
                        onClick={() => void generateActionsForRow(ri)}
                      >
                        {actionsBusy ? '…' : 'Generate actions'}
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={scopedPredictionRows.length}
          page={predPage}
          onPageChange={(_, p) => setPredPage(p)}
          rowsPerPage={predRowsPerPage}
          onRowsPerPageChange={(e) => {
            setPredRowsPerPage(parseInt(e.target.value, 10));
            setPredPage(0);
          }}
          rowsPerPageOptions={[5, 10, 25]}
          sx={{ minHeight: 40, '& .MuiTablePagination-toolbar': { minHeight: 40, pl: 1 } }}
        />
      </Paper>

      <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 2 }}>
        <Stack spacing={0.75} sx={{ mb: 1 }}>
          <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" useFlexGap gap={1}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Action planner · all saved plans
            </Typography>
            <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap" useFlexGap>
              <Typography variant="caption" color="text.secondary">
                {allSavedPlans.length} plan(s)
                {selectedPlanIds.size ? ` · ${selectedPlanIds.size} selected` : ''}
              </Typography>
              <Button
                size="small"
                color="error"
                variant="outlined"
                disabled={!selectedPlanIds.size || anyBusy || bulkDeleting}
                onClick={() => void deleteSelectedPlans()}
              >
                {bulkDeleting ? 'Deleting…' : `Delete selected (${selectedPlanIds.size})`}
              </Button>
            </Stack>
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block">
            Every generated action plan across all prediction jobs. New Generate actions appear here automatically.
          </Typography>
        </Stack>
        <TableContainer sx={{ maxHeight: 400 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox" sx={{ py: 0.5 }}>
                  <Checkbox
                    size="small"
                    checked={allPageSelected}
                    indeterminate={somePageSelected}
                    disabled={!pageRows.length || anyBusy || bulkDeleting}
                    onChange={(_, v) => toggleSelectPage(v)}
                    inputProps={{ 'aria-label': 'Select all plans on this page' }}
                  />
                </TableCell>
                <TableCell sx={{ py: 0.75 }}>Job</TableCell>
                <TableCell sx={{ py: 0.75 }}>Trust</TableCell>
                <TableCell sx={{ py: 0.75 }}>Created</TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Details
                </TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Del
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {pageRows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} sx={{ py: 1.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      No plans yet — use Generate actions above; they appear here automatically.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {pageRows.map((r) => (
                <TableRow key={r.public_id} hover selected={selectedPlanIds.has(r.public_id)}>
                  <TableCell padding="checkbox" sx={{ py: 0.5 }}>
                    <Checkbox
                      size="small"
                      checked={selectedPlanIds.has(r.public_id)}
                      disabled={anyBusy || bulkDeleting}
                      onChange={(_, v) => toggleSelectPlan(r.public_id, v)}
                      inputProps={{ 'aria-label': `Select plan ${r.public_id.slice(0, 8)}` }}
                    />
                  </TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, py: 0.5 }}>
                    {(r.prediction_job_public_id || String(r.prediction_job_id)).slice(0, 8)}…
                  </TableCell>
                  <TableCell sx={{ py: 0.5 }}>
                    {r.trust_commitment ? (
                      <Chip size="small" color="success" variant="outlined" label="ok" sx={{ height: 22 }} />
                    ) : (
                      '—'
                    )}
                  </TableCell>
                  <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap', py: 0.5 }}>
                    {fDateTime(r.created_at)}
                  </TableCell>
                  <TableCell align="right" sx={{ py: 0.5 }}>
                    <Button
                      size="small"
                      sx={{ minWidth: 0, px: 0.75 }}
                      disabled={anyBusy || bulkDeleting}
                      onClick={() => void openSavedPlanDetails(r)}
                    >
                      Details
                    </Button>
                  </TableCell>
                  <TableCell align="right" sx={{ py: 0.5 }}>
                    <Button
                      size="small"
                      color="error"
                      sx={{ minWidth: 0, px: 0.75 }}
                      disabled={anyBusy || bulkDeleting}
                      onClick={async () => {
                        if (!window.confirm(`Delete plan ${r.public_id.slice(0, 8)}…?`)) return;
                        try {
                          await deleteAgentReport(r.public_id);
                          setSelectedPlanIds((prev) => {
                            const next = new Set(prev);
                            next.delete(r.public_id);
                            return next;
                          });
                          await refreshReports();
                          notify({ severity: 'success', text: 'Plan deleted.' });
                        } catch (e) {
                          notify({ severity: 'error', text: formatError(e) });
                        }
                      }}
                    >
                      Del
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          component="div"
          count={allSavedPlans.length}
          page={page}
          onPageChange={(_, p) => setPage(p)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
          rowsPerPageOptions={[5, 10, 25]}
          sx={{
            minHeight: 40,
            '& .MuiTablePagination-toolbar': { minHeight: 40, pl: 1 },
          }}
        />
      </Paper>

      <PlanDetailsDialog
        open={detailOpen}
        onClose={() => {
          setDetailOpen(false);
          setDetail(null);
          setDetailLoading(false);
        }}
        detail={detail}
        loading={detailLoading}
      />
    </Stack>
  );
}
