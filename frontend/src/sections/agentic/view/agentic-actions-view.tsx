import type { AgenticReportOut } from 'src/api/types';

import { useNavigate, useSearchParams } from 'react-router-dom';
import { useRef, useMemo, useState, useEffect, useCallback, type SyntheticEvent } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import Timeline from '@mui/lab/Timeline';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import TableRow from '@mui/material/TableRow';
import TimelineDot from '@mui/lab/TimelineDot';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import CardHeader from '@mui/material/CardHeader';
import TimelineContent from '@mui/lab/TimelineContent';
import TableContainer from '@mui/material/TableContainer';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import TimelineOppositeContent from '@mui/lab/TimelineOppositeContent';
import TimelineItem, { timelineItemClasses } from '@mui/lab/TimelineItem';

import { sortByTime, type TimeSortOrder, toggleTimeSortOrder } from 'src/utils/table-time-sort';

import { DashboardContent } from 'src/layouts/dashboard';
import { ApiError, getAgentReport, listAllAgentReports } from 'src/services';

import { Iconify } from 'src/components/iconify';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import { AgentReportDetailDialog } from 'src/components/run-monitoring/detail-dialogs';

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
    <Card
      variant="outlined"
      sx={{
        height: '100%',
        width: '100%',
        minWidth: 0,
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 1.5,
      }}
    >
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
      <Stack spacing={1} sx={{ p: 1.5, flex: 1, minHeight: 0, overflow: 'auto' }}>
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

type AgenticViewTab = 'table' | 'timeline';

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
  const [reportTimeOrder, setReportTimeOrder] = useState<TimeSortOrder>('desc');
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

  const viewTab: AgenticViewTab = searchParams.get('tab') === 'timeline' ? 'timeline' : 'table';

  const handleViewTabChange = useCallback(
    (_: SyntheticEvent, v: AgenticViewTab) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (v === 'timeline') next.set('tab', 'timeline');
          else next.delete('tab');
          return next;
        },
        { replace: true }
      );
    },
    [setSearchParams]
  );

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

  const items = useMemo(
    () => sortByTime(list, (r) => r.created_at, reportTimeOrder),
    [list, reportTimeOrder]
  );

  const timelineLayout = viewTab === 'timeline';

  return (
    <DashboardContent
      maxWidth={timelineLayout ? false : 'xl'}
      sx={
        timelineLayout
          ? {
              width: 1,
              maxWidth: '100% !important',
              flex: '1 1 auto',
              minHeight: 0,
              display: 'flex',
              flexDirection: 'column',
            }
          : undefined
      }
    >
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        alignItems={{ xs: 'stretch', sm: 'center' }}
        justifyContent="space-between"
        spacing={1.5}
        sx={{
          mb: timelineLayout ? 2 : 1.25,
          flexShrink: 0,
          ...(timelineLayout && {
            p: { xs: 2, sm: 2.5 },
            borderRadius: 2,
            border: 1,
            borderColor: 'divider',
            backgroundImage: (theme) =>
              `linear-gradient(125deg, ${alpha(theme.palette.primary.main, 0.09)} 0%, ${alpha(theme.palette.info.main, 0.05)} 42%, transparent 72%)`,
          }),
        }}
      >
        <Box sx={{ minWidth: 0 }}>
          <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.25 }}>
            <Box
              sx={{
                width: 40,
                height: 40,
                borderRadius: 1.5,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'primary.main',
                bgcolor: (theme) => alpha(theme.palette.primary.main, 0.12),
                flexShrink: 0,
              }}
            >
              <Iconify icon="solar:shield-keyhole-bold-duotone" width={22} />
            </Box>
            <Box>
              <Typography variant={timelineLayout ? 'h4' : 'h5'} sx={{ fontWeight: 800, lineHeight: 1.2 }}>
                {timelineLayout ? 'Agentic timeline' : 'Agentic actions'}
              </Typography>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.25 }}>
                {timelineLayout
                  ? 'Full-width chronological feed. Expand a card for Core / Edge / RAN actions and RAG context.'
                  : 'Table and timeline share sort by created time (newest first by default).'}
              </Typography>
            </Box>
          </Stack>
        </Box>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <Button
            size="small"
            variant="outlined"
            disabled={loading || !list.length}
            onClick={() => setReportTimeOrder((o) => toggleTimeSortOrder(o))}
          >
            Time: {reportTimeOrder === 'desc' ? 'newest first' : 'oldest first'}
          </Button>
          <Button size="small" variant="outlined" onClick={() => void loadList()} disabled={loading}>
            Refresh
          </Button>
        </Stack>
      </Stack>

      <Tabs
        value={viewTab}
        onChange={handleViewTabChange}
        sx={{
          mb: 2,
          minHeight: 44,
          flexShrink: 0,
          '& .MuiTab-root': { minHeight: 44, py: 1, fontSize: '0.875rem', fontWeight: 600 },
        }}
      >
        <Tab value="table" label="Table list" icon={<Iconify icon="eva:done-all-fill" width={18} />} iconPosition="start" />
        <Tab
          value="timeline"
          label="Timeline"
          icon={<Iconify icon="solar:clock-circle-outline" width={18} />}
          iconPosition="start"
        />
      </Tabs>

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

      {!loading && items.length > 0 && viewTab === 'table' && (
        <Card variant="outlined" sx={{ width: 1, minWidth: 0 }}>
          <TableContainer sx={{ maxHeight: { xs: 'none', md: 'calc(100vh - 280px)' } }}>
            <Table size="small" stickyHeader sx={{ minWidth: 720 }}>
              <TableHead>
                <TableRow>
                  <TimeSortHeadCell
                    label="Created"
                    order={reportTimeOrder}
                    onOrderChange={setReportTimeOrder}
                    sx={{ fontWeight: 700 }}
                  />
                  <TableCell sx={{ fontWeight: 700 }}>Report</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Recommended action</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Summary</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Prediction job</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Row</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>
                    Actions
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {items.map((r) => (
                  <TableRow key={r.public_id} hover>
                    <TableCell sx={{ whiteSpace: 'nowrap' }}>{new Date(r.created_at).toLocaleString()}</TableCell>
                    <TableCell
                      sx={{ fontFamily: 'monospace', fontSize: 12, maxWidth: 140 }}
                      title={r.public_id}
                    >
                      {r.public_id.length > 14 ? `${r.public_id.slice(0, 8)}…${r.public_id.slice(-4)}` : r.public_id}
                    </TableCell>
                    <TableCell sx={{ maxWidth: 160 }} title={r.recommended_action || ''}>
                      <Typography variant="body2" noWrap>
                        {r.recommended_action || '—'}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ maxWidth: 280 }} title={r.summary}>
                      <Typography variant="body2" noWrap>
                        {r.summary || '—'}
                      </Typography>
                    </TableCell>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, maxWidth: 120 }} title={r.prediction_job_public_id ?? ''}>
                      {r.prediction_job_public_id
                        ? r.prediction_job_public_id.length > 12
                          ? `${r.prediction_job_public_id.slice(0, 6)}…${r.prediction_job_public_id.slice(-4)}`
                          : r.prediction_job_public_id
                        : '—'}
                    </TableCell>
                    <TableCell sx={{ whiteSpace: 'nowrap' }}>
                      {typeof r.results_row_index === 'number' ? r.results_row_index : '—'}
                    </TableCell>
                    <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>
                      <Button size="small" variant="contained" color="primary" onClick={() => openDialog(r.public_id)} sx={{ mr: 0.5 }}>
                        View
                      </Button>
                      <Button
                        size="small"
                        variant="outlined"
                        onClick={() => navigate(`/agentic/report/${encodeURIComponent(r.public_id)}`)}
                      >
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Card>
      )}

      {!loading && items.length > 0 && viewTab === 'timeline' && (
        <Box
          sx={{
            flex: '1 1 auto',
            minHeight: { xs: 360, md: 'calc(100vh - 248px)' },
            maxHeight: { xs: 'none', md: 'calc(100vh - 168px)' },
            overflow: 'auto',
            pr: { xs: 0, sm: 0.5 },
          }}
        >
          <Timeline
            sx={{
              m: 0,
              p: 0,
              maxWidth: 1120,
              mx: 'auto',
              [`& .${timelineItemClasses.root}:before`]: { flex: 0, padding: 0 },
            }}
          >
            {items.map((r, index) => {
              const full = mergedReport(r);
              const plan = parseStructuredPlan(full);
              const tiers = collectTieredActions(plan);

              return (
                <TimelineItem key={r.public_id}>
                  <TimelineOppositeContent
                    sx={{
                      flex: '0 0 152px',
                      maxWidth: { xs: 104, sm: 152 },
                      py: 2,
                      px: 0,
                      textAlign: 'right',
                    }}
                  >
                    <Typography variant="overline" sx={{ color: 'text.disabled', lineHeight: 1.2, display: 'block' }}>
                      {index === 0 ? 'Latest' : `· ${items.length - index}`}
                    </Typography>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, lineHeight: 1.35, mt: 0.35 }}>
                      {new Date(r.created_at).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                        year: 'numeric',
                      })}
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.25 }}>
                      {new Date(r.created_at).toLocaleTimeString()}
                    </Typography>
                    <Typography
                      variant="caption"
                      sx={{
                        fontFamily: 'monospace',
                        fontSize: 10,
                        color: 'text.disabled',
                        display: 'block',
                        mt: 1,
                        wordBreak: 'break-all',
                      }}
                    >
                      {r.public_id.slice(0, 10)}…
                    </Typography>
                  </TimelineOppositeContent>
                  <TimelineSeparator>
                    <TimelineDot
                      color="primary"
                      sx={{
                        boxShadow: (theme) => `0 0 0 4px ${alpha(theme.palette.primary.main, 0.22)}`,
                      }}
                    >
                      <Iconify icon="solar:shield-keyhole-bold-duotone" width={18} />
                    </TimelineDot>
                    {index < items.length - 1 ? (
                      <TimelineConnector sx={{ bgcolor: (theme) => alpha(theme.palette.primary.main, 0.22) }} />
                    ) : null}
                  </TimelineSeparator>
                  <TimelineContent sx={{ py: 2, px: 0, pl: { xs: 1, sm: 2 } }}>
                    <Card
                      variant="outlined"
                      sx={{
                        borderRadius: 2,
                        overflow: 'hidden',
                        borderColor: (theme) => alpha(theme.palette.divider, 0.95),
                        boxShadow: (theme) => theme.shadows[2],
                      }}
                    >
                      <Accordion
                        defaultExpanded={index === 0}
                        onChange={(e, exp) => handleAccordion(e, exp, r.public_id)}
                        elevation={0}
                        disableGutters
                        sx={{
                          '&:before': { display: 'none' },
                          bgcolor: 'background.paper',
                        }}
                      >
                        <AccordionSummary
                          expandIcon={<Iconify icon="eva:arrow-ios-downward-fill" />}
                          sx={{
                            px: 2,
                            py: 1.5,
                            '& .MuiAccordionSummary-content': { my: 1, alignItems: 'stretch' },
                          }}
                        >
                          <Stack direction="row" alignItems="flex-start" spacing={2} sx={{ width: 1, pr: 1 }}>
                            <Stack spacing={1} sx={{ flex: 1, minWidth: 0 }}>
                              <Stack direction="row" flexWrap="wrap" gap={0.75} alignItems="center">
                                <Chip
                                  size="small"
                                  color="primary"
                                  variant="filled"
                                  label={r.recommended_action || '—'}
                                  sx={{ fontWeight: 700 }}
                                />
                                {r.prediction_job_public_id && (
                                  <Chip
                                    size="small"
                                    variant="outlined"
                                    label={`Job ${r.prediction_job_public_id.slice(0, 8)}…`}
                                  />
                                )}
                                {typeof r.results_row_index === 'number' && (
                                  <Chip size="small" variant="outlined" label={`Row ${r.results_row_index}`} />
                                )}
                              </Stack>
                              <Typography variant="body1" sx={{ fontWeight: 600, lineHeight: 1.45 }}>
                                {r.summary || '—'}
                              </Typography>
                            </Stack>
                            <Stack
                              direction={{ xs: 'column', sm: 'row' }}
                              spacing={0.75}
                              flexShrink={0}
                              onClick={(e) => e.stopPropagation()}
                            >
                              <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                onClick={() => openDialog(r.public_id)}
                              >
                                Quick view
                              </Button>
                              <Button
                                size="small"
                                variant="outlined"
                                onClick={() => navigate(`/agentic/report/${encodeURIComponent(r.public_id)}`)}
                              >
                                Full report
                              </Button>
                            </Stack>
                          </Stack>
                        </AccordionSummary>
                        <AccordionDetails
                          sx={{
                            px: 2,
                            pb: 2.5,
                            pt: 0,
                            bgcolor: (theme) => alpha(theme.palette.grey[500], 0.06),
                          }}
                        >
                          {detailLoading[r.public_id] && (
                            <Typography variant="body2" color="text.secondary" sx={{ py: 1 }}>
                              Loading full report…
                            </Typography>
                          )}
                          <Stack spacing={2.5} sx={{ mt: detailLoading[r.public_id] ? 1 : 0 }}>
                            <Box>
                              <Typography
                                variant="overline"
                                sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: 0.6 }}
                              >
                                Actions by network tier
                              </Typography>
                              <Grid container spacing={1.5} sx={{ mt: 1 }}>
                                <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                                  <ActionColumn title="Core plane" tierKey="Core" actions={tiers.core} color="primary" />
                                </Grid>
                                <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                                  <ActionColumn title="Edge plane" tierKey="Edge" actions={tiers.edge} color="info" />
                                </Grid>
                                {tiers.ran.length > 0 && (
                                  <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                                    <ActionColumn title="Radio access" tierKey="RAN" actions={tiers.ran} color="warning" />
                                  </Grid>
                                )}
                                {tiers.other.length > 0 && (
                                  <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                                    <ActionColumn
                                      title="Other / unspecified"
                                      tierKey="Other"
                                      actions={tiers.other}
                                      color="secondary"
                                    />
                                  </Grid>
                                )}
                              </Grid>
                            </Box>

                            {full.rag_context_used && (
                              <Box>
                                <Typography
                                  variant="overline"
                                  sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: 0.6 }}
                                >
                                  RAG context (excerpt)
                                </Typography>
                                <Typography
                                  variant="body2"
                                  component="pre"
                                  sx={{
                                    display: 'block',
                                    mt: 1,
                                    p: 1.5,
                                    borderRadius: 1.5,
                                    bgcolor: 'background.paper',
                                    border: 1,
                                    borderColor: 'divider',
                                    maxHeight: 240,
                                    overflow: 'auto',
                                    whiteSpace: 'pre-wrap',
                                    wordBreak: 'break-word',
                                    fontSize: '0.8125rem',
                                    fontFamily: 'inherit',
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
                    </Card>
                  </TimelineContent>
                </TimelineItem>
              );
            })}
          </Timeline>
        </Box>
      )}

      <AgentReportDetailDialog open={Boolean(dialogId)} publicId={dialogId} onClose={closeDialog} />
    </DashboardContent>
  );
}
