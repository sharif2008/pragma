import type { AgenticReportOut, TrustAnchorVerifyOut, ExecutionReportDetailOut } from 'src/api/types';

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
import Dialog from '@mui/material/Dialog';
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
import DialogTitle from '@mui/material/DialogTitle';
import TimelineContent from '@mui/lab/TimelineContent';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TableContainer from '@mui/material/TableContainer';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TablePagination from '@mui/material/TablePagination';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import CircularProgress from '@mui/material/CircularProgress';
import TimelineOppositeContent from '@mui/lab/TimelineOppositeContent';
import TimelineItem, { timelineItemClasses } from '@mui/lab/TimelineItem';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime, type TimeSortOrder, toggleTimeSortOrder } from 'src/utils/table-time-sort';
import {
  execStatusLabel,
  actionsFromReport,
  parseStructuredPlan,
  attackTypeFromReportRow,
} from 'src/utils/agentic-plan-actions';

import { DashboardContent } from 'src/layouts/dashboard';
import {
  ApiError,
  getAgentReport,
  verifyTrustAnchor,
  applyAgenticReport,
  getExecutionReport,
  listAllAgentReports,
} from 'src/services';

import { Iconify } from 'src/components/iconify';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import { AgentReportDetailDialog } from 'src/components/run-monitoring/detail-dialogs';
import {
  ExecutionChainSummary,
  ExecutionChainResultsList,
} from 'src/components/agentic/execution-chain-results';
import {
  PlanIdCell,
  ExecStatusChip,
  DetectionActionsList,
} from 'src/components/agentic/detection-actions-list';

// ----------------------------------------------------------------------

type TierAction = {
  action: string;
  network_tier: string;
  reasoning: string;
  party_evidence_type?: string;
};

function collectTieredActions(plan: Record<string, unknown> | null): {
  core: TierAction[];
  edge: TierAction[];
  ran: TierAction[];
  other: TierAction[];
} {
  const out = { core: [] as TierAction[], edge: [] as TierAction[], ran: [] as TierAction[], other: [] as TierAction[] };
  if (!plan) return out;

  const bucket = (tierRaw: string): 'core' | 'edge' | 'ran' | 'other' => {
    const t = tierRaw.trim().toLowerCase().replace(/\s+/g, ' ');
    if (t === 'core' || t === 'endpoint / edr' || t === 'endpoint/edr' || t === 'd3') return 'core';
    if (t === 'edge' || t === 'perimeter / ids' || t === 'perimeter/ids' || t === 'd2') return 'edge';
    if (t === 'ran' || t === 'access / isp' || t === 'access/isp' || t === 'd1') return 'ran';
    return 'other';
  };

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
    const b = bucket(tier);
    if (b === 'core') out.core.push(item);
    else if (b === 'edge') out.edge.push(item);
    else if (b === 'ran') out.ran.push(item);
    else out.other.push(item);
  };

  const prim = plan.primary_actions;
  const sup = plan.supporting_actions;
  if (Array.isArray(prim)) prim.forEach(push);
  if (Array.isArray(sup)) sup.forEach(push);
  return out;
}

function trustAnchorId(report: AgenticReportOut): number | null {
  const ta = report.trust_anchor;
  if (!ta || typeof ta !== 'object') return null;
  // Backend trust anchor list rows use id; report trust_anchor dict currently does not.
  // We resolve by calling verify endpoint via list view button (see apply dialog logic).
  const id = (ta as { id?: unknown }).id;
  if (typeof id === 'number' && Number.isFinite(id)) return id;
  return null;
}

function isApplied(report: AgenticReportOut): boolean {
  const ex = report.execution_report;
  if (!ex || typeof ex !== 'object') return false;
  return String((ex as { status?: unknown }).status ?? '') === 'applied';
}

function executionChip(report: AgenticReportOut) {
  const ex = report.execution_report;
  if (!ex || typeof ex !== 'object') return null;
  const status = String((ex as { status?: unknown }).status ?? '');
  if (status === 'applied') return <Chip size="small" color="success" variant="outlined" label="Applied" />;
  if (status === 'failed') return <Chip size="small" color="error" variant="outlined" label="Apply failed" />;
  return null;
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
          <Stack direction="row" alignItems="center" spacing={0.75}>
            <Chip size="small" color={color} label={title} sx={{ fontWeight: 700, height: 22 }} />
            {tierKey ? (
              <Typography variant="caption" color="text.secondary">
                {tierKey}
              </Typography>
            ) : null}
            <Chip size="small" variant="outlined" label={actions.length} sx={{ height: 20 }} />
          </Stack>
        }
        sx={{ py: 0.5, px: 1, minHeight: 0, '& .MuiCardHeader-title': { width: '100%' } }}
      />
      <Divider />
      <Stack spacing={0.75} sx={{ p: 1, flex: 1, minHeight: 0, overflow: 'auto' }}>
        {actions.length === 0 && (
          <Typography variant="caption" color="text.secondary">
            No actions for this tier.
          </Typography>
        )}
        {actions.map((a, i) => (
          <Box
            key={`${a.action}-${i}`}
            sx={{
              p: 0.75,
              borderRadius: 1,
              bgcolor: (theme) => theme.vars.palette.action.hover,
            }}
          >
            <Typography variant="caption" sx={{ fontWeight: 700, display: 'block' }}>
              {a.action}
            </Typography>
            {a.party_evidence_type && (
              <Typography variant="caption" color="text.secondary" display="block" sx={{ fontSize: 11 }}>
                Evidence: {a.party_evidence_type}
              </Typography>
            )}
            {a.reasoning && (
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ mt: 0.25, display: 'block', whiteSpace: 'pre-wrap', wordBreak: 'break-word', lineHeight: 1.35 }}
              >
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
  const [applyOpen, setApplyOpen] = useState(false);
  const [applyId, setApplyId] = useState<string | null>(null);
  const [applyError, setApplyError] = useState<string>('');
  const [applyLoading, setApplyLoading] = useState(false);
  const [applyVerify, setApplyVerify] = useState<TrustAnchorVerifyOut | null>(null);
  const [applyExec, setApplyExec] = useState<ExecutionReportDetailOut | null>(null);
  const [reportTimeOrder, setReportTimeOrder] = useState<TimeSortOrder>('desc');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
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

  const closeApply = () => {
    setApplyOpen(false);
    setApplyId(null);
    setApplyError('');
    setApplyLoading(false);
    setApplyVerify(null);
    setApplyExec(null);
  };

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

  const openApply = async (publicId: string) => {
    const id = publicId.trim();
    if (!id) return;
    setApplyOpen(true);
    setApplyId(id);
    setApplyError('');
    setApplyVerify(null);
    setApplyExec(null);
    setApplyLoading(true);
    try {
      const full = await getAgentReport(id);
      setDetailById((prev) => ({ ...prev, [id]: full }));

      const ex = full.execution_report;
      if (ex && typeof ex === 'object') {
        const exId = (ex as { id?: unknown }).id;
        if (typeof exId === 'number') {
          const det = await getExecutionReport(exId);
          setApplyExec(det);
          setApplyLoading(false);
          return;
        }
      }

      const taId = trustAnchorId(full);
      if (taId == null) {
        setApplyError('Missing trust anchor id for this report (cannot verify).');
        return;
      }
      const v = await verifyTrustAnchor(taId);
      setApplyVerify(v);
    } catch (e) {
      setApplyError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
    } finally {
      setApplyLoading(false);
    }
  };

  const confirmApply = async () => {
    if (!applyId) return;
    setApplyLoading(true);
    setApplyError('');
    try {
      const out = await applyAgenticReport(applyId);
      setApplyExec(out);
      await loadList();
    } catch (e) {
      setApplyError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
      await loadList();
    } finally {
      setApplyLoading(false);
    }
  };

  const handleAccordion = (_event: unknown, expanded: boolean, publicId: string) => {
    if (expanded) void ensureDetail(publicId);
  };

  const mergedReport = (r: AgenticReportOut) => detailById[r.public_id] ?? r;

  const items = useMemo(
    () => sortByTime(list, (r) => r.created_at, reportTimeOrder),
    [list, reportTimeOrder]
  );

  const pageItems = useMemo(() => {
    const start = page * rowsPerPage;
    return items.slice(start, start + rowsPerPage);
  }, [items, page, rowsPerPage]);

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
        spacing={1}
        sx={{
          mb: 1,
          flexShrink: 0,
          ...(timelineLayout && {
            p: { xs: 1, sm: 1.25 },
            borderRadius: 1.5,
            border: 1,
            borderColor: 'divider',
            backgroundImage: (theme) =>
              `linear-gradient(125deg, ${alpha(theme.palette.primary.main, 0.09)} 0%, ${alpha(theme.palette.info.main, 0.05)} 42%, transparent 72%)`,
          }),
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1} sx={{ minWidth: 0 }}>
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: 'primary.main',
              bgcolor: (theme) => alpha(theme.palette.primary.main, 0.12),
              flexShrink: 0,
            }}
          >
            <Iconify icon="solar:shield-keyhole-bold-duotone" width={18} />
          </Box>
          <Typography variant="h6" sx={{ fontWeight: 800, lineHeight: 1.2 }}>
            {timelineLayout ? 'Agentic timeline' : 'Agentic actions'}
          </Typography>
        </Stack>
        <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
          <Button
            size="small"
            variant="outlined"
            disabled={loading || !list.length}
            onClick={() => setReportTimeOrder((o) => toggleTimeSortOrder(o))}
          >
            {reportTimeOrder === 'desc' ? 'Newest' : 'Oldest'}
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
          mb: 1,
          minHeight: 36,
          flexShrink: 0,
          '& .MuiTab-root': { minHeight: 36, py: 0.5, px: 1.25, fontSize: '0.8125rem', fontWeight: 600 },
        }}
      >
        <Tab value="table" label="Table" icon={<Iconify icon="eva:done-all-fill" width={16} />} iconPosition="start" />
        <Tab
          value="timeline"
          label="Timeline"
          icon={<Iconify icon="solar:clock-circle-outline" width={16} />}
          iconPosition="start"
        />
      </Tabs>

      {error && (
        <Alert severity="error" sx={{ mb: 1, py: 0.5 }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Typography variant="body2" color="text.secondary">
          Loading reports…
        </Typography>
      )}

      {!loading && !items.length && (
        <Alert severity="info">No agentic reports yet. Run Agent decide from Setting after a prediction job completes.</Alert>
      )}

      {!loading && items.length > 0 && viewTab === 'table' && (
        <Card variant="outlined" sx={{ width: 1, minWidth: 0 }}>
          <TableContainer sx={{ maxHeight: { xs: 'none', md: 'calc(100vh - 220px)' } }}>
            <Table size="small" stickyHeader sx={{ minWidth: 900 }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, py: 0.75 }}>Plan</TableCell>
                  <TableCell sx={{ fontWeight: 700, py: 0.75 }}>Attack</TableCell>
                  <TableCell sx={{ fontWeight: 700, py: 0.75 }}>Detection actions</TableCell>
                  <TableCell sx={{ fontWeight: 700, py: 0.75 }}>Status</TableCell>
                  <TimeSortHeadCell
                    label="Created"
                    order={reportTimeOrder}
                    onOrderChange={setReportTimeOrder}
                    sx={{ fontWeight: 700, py: 0.75 }}
                  />
                  <TableCell align="right" sx={{ fontWeight: 700, py: 0.75 }}>
                    View
                  </TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700, py: 0.75 }}>
                    Apply
                  </TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700, py: 0.75 }}>
                    Details
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {pageItems.map((r) => {
                  const full = mergedReport(r);
                  const attackType = attackTypeFromReportRow(full);
                  const actions = actionsFromReport(full);
                  const applied = isApplied(full);
                  return (
                    <TableRow key={r.public_id} hover>
                      <TableCell sx={{ py: 0.5, verticalAlign: 'top' }}>
                        <PlanIdCell publicId={r.public_id} />
                      </TableCell>
                      <TableCell sx={{ py: 0.5, verticalAlign: 'top' }}>
                        {attackType ? (
                          <Chip size="small" variant="outlined" label={attackType} sx={{ height: 20 }} />
                        ) : (
                          '—'
                        )}
                      </TableCell>
                      <TableCell sx={{ py: 0.5, maxWidth: 420, verticalAlign: 'top' }}>
                        <DetectionActionsList actions={actions} compact />
                      </TableCell>
                      <TableCell sx={{ py: 0.5, verticalAlign: 'top' }}>
                        <ExecStatusChip status={execStatusLabel(full)} />
                      </TableCell>
                      <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap', py: 0.5, verticalAlign: 'top' }}>
                        {fDateTime(r.created_at)}
                      </TableCell>
                      <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                        <Button
                          size="small"
                          variant="contained"
                          color="primary"
                          onClick={() => openDialog(r.public_id)}
                          sx={{ minWidth: 0, px: 0.75 }}
                        >
                          View
                        </Button>
                      </TableCell>
                      <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                        <Button
                          size="small"
                          color="success"
                          variant={applied ? 'outlined' : 'contained'}
                          disabled={applied}
                          onClick={() => void openApply(r.public_id)}
                          sx={{ minWidth: 0, px: 0.75 }}
                        >
                          {applied ? 'Applied' : 'Apply'}
                        </Button>
                      </TableCell>
                      <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => navigate(`/agentic/report/${encodeURIComponent(r.public_id)}`)}
                          sx={{ minWidth: 0, px: 0.75 }}
                        >
                          Details
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
            count={items.length}
            page={page}
            onPageChange={(_, p) => setPage(p)}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={(e) => {
              setRowsPerPage(parseInt(e.target.value, 10));
              setPage(0);
            }}
            rowsPerPageOptions={[5, 10, 25, 50]}
            sx={{ minHeight: 44, '& .MuiTablePagination-toolbar': { minHeight: 44, pl: 1 } }}
          />
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
                      flex: '0 0 120px',
                      maxWidth: { xs: 88, sm: 120 },
                      py: 1,
                      px: 0,
                      textAlign: 'right',
                    }}
                  >
                    <Typography variant="overline" sx={{ color: 'text.disabled', lineHeight: 1.1, display: 'block', fontSize: 10 }}>
                      {index === 0 ? 'Latest' : `· ${items.length - index}`}
                    </Typography>
                    <Typography variant="caption" sx={{ fontWeight: 700, lineHeight: 1.3, display: 'block' }}>
                      {new Date(r.created_at).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                      })}
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', fontSize: 11 }}>
                      {new Date(r.created_at).toLocaleTimeString()}
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
                  <TimelineContent sx={{ py: 1, px: 0, pl: { xs: 1, sm: 1.5 } }}>
                    <Card
                      variant="outlined"
                      sx={{
                        borderRadius: 1.5,
                        overflow: 'hidden',
                        borderColor: (theme) => alpha(theme.palette.divider, 0.95),
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
                            px: 1.5,
                            py: 0.75,
                            minHeight: 0,
                            '& .MuiAccordionSummary-content': { my: 0.5, alignItems: 'stretch' },
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
                              {!isApplied(r) ? (
                                <Button size="small" variant="contained" color="success" onClick={() => void openApply(r.public_id)}>
                                  Apply
                                </Button>
                              ) : (
                                <Button size="small" variant="outlined" color="success" onClick={() => void openApply(r.public_id)}>
                                  Applied
                                </Button>
                              )}
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
                          <ActionColumn title="Endpoint / EDR" tierKey="" actions={tiers.core} color="primary" />
                        </Grid>
                        <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                          <ActionColumn title="Perimeter / IDS" tierKey="" actions={tiers.edge} color="info" />
                        </Grid>
                        {tiers.ran.length > 0 && (
                          <Grid size={{ xs: 12, md: 6, xl: 3 }} sx={{ display: 'flex' }}>
                            <ActionColumn title="Access / ISP" tierKey="" actions={tiers.ran} color="warning" />
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

      <Dialog open={applyOpen} onClose={closeApply} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontWeight: 800 }}>Apply agentic actions</DialogTitle>
        <DialogContent dividers>
          {applyError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {applyError}
            </Alert>
          )}

          {applyLoading && (
            <Stack alignItems="center" justifyContent="center" sx={{ py: 4 }}>
              <CircularProgress />
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                Loading integrity and report details…
              </Typography>
            </Stack>
          )}

          {!applyLoading && applyId && (
            (() => {
              const full = detailById[applyId];
              const plan = full ? parseStructuredPlan(full) : null;
              const tiers = collectTieredActions(plan);
              const verifyOk = applyVerify?.overall_integrity === 'valid';

              return (
                <Stack spacing={2}>
                  {full ? (
                    <>
                      <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center">
                        <Chip size="small" variant="outlined" label={`Report ${full.public_id.slice(0, 8)}…`} />
                        {executionChip(full)}
                      </Stack>
                      <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                        {full.summary || '—'}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Recommended: <strong>{full.recommended_action || '—'}</strong>
                      </Typography>
                    </>
                  ) : (
                    <Alert severity="warning">Report details not loaded.</Alert>
                  )}

                  {applyExec ? (
                    <>
                      <Divider />
                      <ExecutionChainSummary exec={applyExec} verify={applyVerify} />
                      <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: 0.6 }}>
                        Apply results by network tier
                      </Typography>
                      <ExecutionChainResultsList exec={applyExec} compact />
                    </>
                  ) : (
                    <>
                      <Divider />
                      <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: 0.6 }}>
                        Integrity (blockchain + report file)
                      </Typography>
                      {applyVerify ? (
                        <Alert severity={verifyOk ? 'success' : 'error'}>
                          Overall: <strong>{applyVerify.overall_integrity}</strong>
                          {applyVerify.tx_hash ? ` · tx ${applyVerify.tx_hash.slice(0, 10)}…` : ''}
                        </Alert>
                      ) : (
                        <Alert severity="warning">Integrity not available.</Alert>
                      )}

                      <Typography variant="overline" sx={{ color: 'text.secondary', fontWeight: 700, letterSpacing: 0.6 }}>
                        Actions by network tier
                      </Typography>
                      <Grid container spacing={1.5}>
                        <Grid size={{ xs: 12, md: 6, xl: 4 }} sx={{ display: 'flex' }}>
                          <ActionColumn title="Endpoint / EDR" tierKey="" actions={tiers.core} color="primary" />
                        </Grid>
                        <Grid size={{ xs: 12, md: 6, xl: 4 }} sx={{ display: 'flex' }}>
                          <ActionColumn title="Perimeter / IDS" tierKey="" actions={tiers.edge} color="info" />
                        </Grid>
                        <Grid size={{ xs: 12, md: 6, xl: 4 }} sx={{ display: 'flex' }}>
                          <ActionColumn title="Access / ISP" tierKey="" actions={tiers.ran} color="warning" />
                        </Grid>
                      </Grid>

                      {!verifyOk && (
                        <Alert severity="error">
                          Apply is blocked: integrity is not <strong>valid</strong>. Fix trust anchor / report integrity first.
                        </Alert>
                      )}
                    </>
                  )}
                </Stack>
              );
            })()
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeApply}>Close</Button>
          {!applyExec && (
            <Button
              variant="contained"
              color="success"
              disabled={applyLoading || applyVerify?.overall_integrity !== 'valid'}
              onClick={() => void confirmApply()}
            >
              Confirm apply
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </DashboardContent>
  );
}
