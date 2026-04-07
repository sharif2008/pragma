import type { AgenticReportOut } from 'src/api/types';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import CardHeader from '@mui/material/CardHeader';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import Timeline from '@mui/lab/Timeline';
import TimelineDot from '@mui/lab/TimelineDot';
import TimelineItem from '@mui/lab/TimelineItem';
import TimelineContent from '@mui/lab/TimelineContent';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TimelineOppositeContent from '@mui/lab/TimelineOppositeContent';
import { timelineItemClasses } from '@mui/lab/TimelineItem';

import { Iconify } from 'src/components/iconify';
import { AgentReportDetailDialog } from 'src/components/run-monitoring/detail-dialogs';
import { ApiError, getAgentReport, listAllAgentReports } from 'src/services';
import { DashboardContent } from 'src/layouts/dashboard';

// ----------------------------------------------------------------------

type TierAction = {
  action: string;
  network_tier: string;
  reasoning: string;
  party_evidence_type?: string;
};

function parseStructuredPlan(report: AgenticReportOut): Record<string, unknown> | null {
  const art = report.report_artifact;
  if (art && typeof art === 'object' && art !== null) {
    const sp = (art as { structured_plan?: unknown }).structured_plan;
    if (sp && typeof sp === 'object' && sp !== null) {
      return sp as Record<string, unknown>;
    }
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

function collectTieredActions(plan: Record<string, unknown> | null): {
  core: TierAction[];
  edge: TierAction[];
  ran: TierAction[];
  other: TierAction[];
} {
  const out = { core: [] as TierAction[], edge: [] as TierAction[], ran: [] as TierAction[], other: [] as TierAction[] };
  if (!plan) return out;

  const push = (x: unknown) => {
    if (!x || typeof x !== 'object') return;
    const o = x as Record<string, unknown>;
    const tier = String(o.network_tier ?? '').trim();
    const item: TierAction = {
      action: String(o.action ?? '—'),
      network_tier: tier || '—',
      reasoning: String(o.reasoning ?? ''),
      party_evidence_type: o.party_evidence_type != null ? String(o.party_evidence_type) : undefined,
    };
    if (tier === 'Core') out.core.push(item);
    else if (tier === 'Edge') out.edge.push(item);
    else if (tier === 'RAN') out.ran.push(item);
    else out.other.push(item);
  };

  const prim = plan.primary_actions;
  const sup = plan.supporting_actions;
  if (Array.isArray(prim)) prim.forEach(push);
  if (Array.isArray(sup)) sup.forEach(push);
  return out;
}

function ActionColumn({
  title,
  tierKey,
  actions,
  color,
}: {
  title: string;
  tierKey: string;
  actions: TierAction[];
  color: 'primary' | 'secondary' | 'info' | 'warning';
}) {
  return (
    <Card variant="outlined" sx={{ flex: '1 1 280px', minWidth: 280, maxWidth: 420 }}>
      <CardHeader
        title={
          <Stack direction="row" alignItems="center" spacing={1}>
            <Chip size="small" color={color} label={tierKey} sx={{ fontWeight: 700 }} />
            <Typography variant="subtitle2">{title}</Typography>
            <Chip size="small" variant="outlined" label={actions.length} />
          </Stack>
        }
        sx={{ py: 1, '& .MuiCardHeader-title': { width: '100%' } }}
      />
      <Divider />
      <Stack spacing={1} sx={{ p: 1.5 }}>
        {actions.length === 0 && (
          <Typography variant="caption" color="text.secondary">
            No actions tagged for this tier.
          </Typography>
        )}
        {actions.map((a, i) => (
          <Box
            key={`${a.action}-${i}`}
            sx={{
              p: 1,
              borderRadius: 1,
              bgcolor: (theme) => theme.vars.palette.action.hover,
            }}
          >
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              {a.action}
            </Typography>
            {a.party_evidence_type && (
              <Typography variant="caption" color="text.secondary" display="block">
                Evidence: {a.party_evidence_type}
              </Typography>
            )}
            {a.reasoning && (
              <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {a.reasoning}
              </Typography>
            )}
          </Box>
        ))}
      </Stack>
    </Card>
  );
}

// ----------------------------------------------------------------------

export function AgenticActionsView() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [list, setList] = useState<AgenticReportOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [detailById, setDetailById] = useState<Record<string, AgenticReportOut>>({});
  const [detailLoading, setDetailLoading] = useState<Record<string, boolean>>({});
  const [dialogId, setDialogId] = useState<string | null>(null);
  const fetchStartedRef = useRef<Set<string>>(new Set());

  const openDialog = useCallback((publicId: string) => {
    const id = publicId.trim();
    if (!id) return;
    setDialogId(id);
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set('dialog', id);
        return next;
      },
      { replace: true }
    );
  }, [setSearchParams]);

  const closeDialog = useCallback(() => {
    setDialogId(null);
    setSearchParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.delete('dialog');
        return next;
      },
      { replace: true }
    );
  }, [setSearchParams]);

  useEffect(() => {
    const d = searchParams.get('dialog')?.trim();
    if (d) setDialogId(d);
  }, [searchParams]);

  const loadList = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const rows = await listAllAgentReports();
      const sorted = [...rows].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      setList(sorted);
      setDetailById({});
      fetchStartedRef.current = new Set();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadList();
  }, [loadList]);

  const ensureDetail = useCallback(async (publicId: string) => {
    if (detailById[publicId] || fetchStartedRef.current.has(publicId)) return;
    fetchStartedRef.current.add(publicId);
    setDetailLoading((s) => ({ ...s, [publicId]: true }));
    try {
      const full = await getAgentReport(publicId);
      setDetailById((prev) => ({ ...prev, [publicId]: full }));
    } catch {
      fetchStartedRef.current.delete(publicId);
    } finally {
      setDetailLoading((s) => ({ ...s, [publicId]: false }));
    }
  }, [detailById]);

  const handleAccordion = (_event: unknown, expanded: boolean, publicId: string) => {
    if (expanded) void ensureDetail(publicId);
  };

  const mergedReport = (r: AgenticReportOut) => detailById[r.public_id] ?? r;

  const items = useMemo(() => list, [list]);

  return (
    <DashboardContent maxWidth="xl">
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.25 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Agentic actions
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            Timeline of LLM orchestration decisions · actions split by network tier (Core vs Edge; RAN shown separately)
          </Typography>
        </Box>
        <Button size="small" variant="outlined" onClick={() => void loadList()} disabled={loading}>
          Refresh
        </Button>
      </Stack>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Typography variant="body2" color="text.secondary">
          Loading reports…
        </Typography>
      )}

      {!loading && !items.length && (
        <Alert severity="info">No agentic reports yet. Run Agent decide from ML & RAG after a prediction job completes.</Alert>
      )}

      {!loading && items.length > 0 && (
        <Timeline
          sx={{
            m: 0,
            p: 0,
            [`& .${timelineItemClasses.root}:before`]: { flex: 0, padding: 0 },
          }}
        >
          {items.map((r, index) => {
            const full = mergedReport(r);
            const plan = parseStructuredPlan(full);
            const tiers = collectTieredActions(plan);

            return (
              <TimelineItem key={r.public_id}>
                <TimelineOppositeContent sx={{ flex: 0.22, py: 1.5, px: 0 }}>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
                    {new Date(r.created_at).toLocaleString()}
                  </Typography>
                  <Typography variant="caption" sx={{ fontFamily: 'monospace', display: 'block', wordBreak: 'break-all' }}>
                    {r.public_id.slice(0, 8)}…
                  </Typography>
                </TimelineOppositeContent>
                <TimelineSeparator>
                  <TimelineDot color="primary">
                    <Iconify icon="solar:shield-keyhole-bold-duotone" width={18} />
                  </TimelineDot>
                  {index < items.length - 1 ? <TimelineConnector /> : null}
                </TimelineSeparator>
                <TimelineContent sx={{ py: 1.5, px: 0 }}>
                  <Accordion
                    defaultExpanded={index === 0}
                    onChange={(e, exp) => handleAccordion(e, exp, r.public_id)}
                  >
                    <AccordionSummary expandIcon={<Iconify icon="eva:arrow-ios-downward-fill" />}>
                      <Stack direction="row" alignItems="flex-start" spacing={1} sx={{ width: 1, pr: 0.5 }}>
                        <Stack spacing={0.5} sx={{ flex: 1, minWidth: 0, pr: 0.5 }}>
                          <Stack direction="row" flexWrap="wrap" gap={0.75} alignItems="center">
                            <Chip size="small" color="primary" variant="outlined" label={r.recommended_action || '—'} />
                            {r.prediction_job_public_id && (
                              <Chip
                                size="small"
                                variant="outlined"
                                label={`job ${r.prediction_job_public_id.slice(0, 8)}…`}
                              />
                            )}
                            {typeof r.results_row_index === 'number' && (
                              <Chip size="small" variant="outlined" label={`row ${r.results_row_index}`} />
                            )}
                          </Stack>
                          <Typography variant="subtitle2" sx={{ fontWeight: 600, lineHeight: 1.35 }}>
                            {r.summary}
                          </Typography>
                        </Stack>
                        <Stack
                          direction="row"
                          spacing={0.75}
                          flexShrink={0}
                          sx={{ pt: 0.25 }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <Button
                            size="small"
                            variant="contained"
                            color="primary"
                            onClick={() => openDialog(r.public_id)}
                          >
                            View
                          </Button>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={() => navigate(`/agentic/report/${encodeURIComponent(r.public_id)}`)}
                          >
                            Details
                          </Button>
                        </Stack>
                      </Stack>
                    </AccordionSummary>
                    <AccordionDetails>
                      {detailLoading[r.public_id] && (
                        <Typography variant="caption" color="text.secondary">
                          Loading full report…
                        </Typography>
                      )}
                      <Stack spacing={2} sx={{ mt: detailLoading[r.public_id] ? 1 : 0 }}>
                        <Box
                          sx={{
                            overflowX: 'auto',
                            width: 1,
                            maxWidth: 1,
                            pb: 0.5,
                            WebkitOverflowScrolling: 'touch',
                          }}
                        >
                          <Stack direction="row" spacing={1.5} sx={{ flexWrap: 'nowrap', alignItems: 'stretch' }}>
                            <ActionColumn
                              title="Core plane"
                              tierKey="Core"
                              actions={tiers.core}
                              color="primary"
                            />
                            <ActionColumn
                              title="Edge plane"
                              tierKey="Edge"
                              actions={tiers.edge}
                              color="info"
                            />
                          </Stack>
                        </Box>

                        {(tiers.ran.length > 0 || tiers.other.length > 0) && (
                          <Box
                            sx={{
                              overflowX: 'auto',
                              width: 1,
                              maxWidth: 1,
                              pb: 0.5,
                              WebkitOverflowScrolling: 'touch',
                            }}
                          >
                            <Stack direction="row" spacing={1.5} sx={{ flexWrap: 'nowrap', alignItems: 'stretch' }}>
                              {tiers.ran.length > 0 && (
                                <ActionColumn title="Radio access" tierKey="RAN" actions={tiers.ran} color="warning" />
                              )}
                              {tiers.other.length > 0 && (
                                <ActionColumn
                                  title="Other / unspecified tier"
                                  tierKey="Other"
                                  actions={tiers.other}
                                  color="secondary"
                                />
                              )}
                            </Stack>
                          </Box>
                        )}

                        {full.rag_context_used && (
                          <Box>
                            <Typography variant="caption" sx={{ color: 'text.secondary', fontWeight: 700 }}>
                              RAG context (excerpt)
                            </Typography>
                            <Typography
                              variant="caption"
                              component="pre"
                              sx={{
                                display: 'block',
                                mt: 0.5,
                                p: 1,
                                borderRadius: 1,
                                bgcolor: 'action.hover',
                                maxHeight: 140,
                                overflow: 'auto',
                                whiteSpace: 'pre-wrap',
                                wordBreak: 'break-word',
                              }}
                            >
                              {full.rag_context_used.slice(0, 4000)}
                              {full.rag_context_used.length > 4000 ? '…' : ''}
                            </Typography>
                          </Box>
                        )}
                      </Stack>
                    </AccordionDetails>
                  </Accordion>
                </TimelineContent>
              </TimelineItem>
            );
          })}
        </Timeline>
      )}

      <AgentReportDetailDialog open={Boolean(dialogId)} publicId={dialogId} onClose={closeDialog} />
    </DashboardContent>
  );
}
