import type { RunListItemOut, AgenticReportOut } from 'src/api/types';
import type { IconifyName } from 'src/components/iconify/register-icons';

import { useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import { alpha } from '@mui/material/styles';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import CardHeader from '@mui/material/CardHeader';
import CardContent from '@mui/material/CardContent';
import TablePagination from '@mui/material/TablePagination';

import { RouterLink } from 'src/routes/components';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime, type TimeSortOrder } from 'src/utils/table-time-sort';
import {
  execStatusLabel,
  actionsFromReport,
  attackTypeFromReportRow,
} from 'src/utils/agentic-plan-actions';

import { api, ApiError } from 'src/services';
import { DashboardContent } from 'src/layouts/dashboard';
import { useAppSnackbar } from 'src/contexts/app-snackbar-context';

import { Iconify } from 'src/components/iconify';
import { RunDetailDialog } from 'src/components/run-monitoring/detail-dialogs';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import {
  PlanIdCell,
  ExecStatusChip,
} from 'src/components/agentic/detection-actions-list';

const SX_COMPACT_TABLE = {
  '& .MuiTableCell-root': {
    py: 0.625,
    px: 1.125,
    fontSize: '0.8125rem',
    borderColor: 'divider',
  },
  '& .MuiTableCell-head': { fontWeight: 700, fontSize: '0.75rem', letterSpacing: 0.01 },
  width: '100%',
  tableLayout: 'fixed',
} as const;

/** Balanced column weights for fixed-layout tables (sum 100%). */
const RUNS_TABLE_COLGROUP = (
  <colgroup>
    <col style={{ width: '12%' }} />
    <col style={{ width: '14%' }} />
    <col style={{ width: '12%' }} />
    <col style={{ width: '12%' }} />
    <col style={{ width: '18%' }} />
    <col style={{ width: '6%' }} />
  </colgroup>
);

const AGENTIC_TABLE_COLGROUP = (
  <colgroup>
    <col style={{ width: '10%' }} />
    <col style={{ width: '12%' }} />
    <col style={{ width: '36%' }} />
    <col style={{ width: '12%' }} />
    <col style={{ width: '18%' }} />
    <col style={{ width: '6%' }} />
  </colgroup>
);

function headerAvatarIcon(icon: IconifyName, color: 'primary' | 'error' | 'info' | 'success' | 'warning' | 'secondary') {
  return (
    <Box
      sx={(theme) => ({
        width: 40,
        height: 40,
        borderRadius: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: theme.palette[color].main,
        bgcolor: alpha(theme.palette[color].main, 0.12),
      })}
    >
      <Iconify icon={icon} width={22} />
    </Box>
  );
}

function runIdShort(runId: string): string {
  if (!runId) return '—';
  return runId.length > 14 ? `${runId.slice(0, 8)}…` : runId.slice(0, 8);
}

const SX_COMPACT_CARD_HEADER = {
  py: 1,
  px: 1.5,
  '& .MuiCardHeader-title': { fontSize: '0.875rem', fontWeight: 700 },
  '& .MuiCardHeader-subheader': { fontSize: '0.7rem', mt: 0.25 },
} as const;

const CHIP_COMPACT = { height: 22, '& .MuiChip-label': { px: 0.75, fontSize: '0.6875rem' } } as const;

function statusChip(status: string) {
  const sx = CHIP_COMPACT;
  if (status === 'running') return <Chip size="small" sx={sx} color="info" label="running" />;
  if (status === 'completed') return <Chip size="small" sx={sx} color="success" label="completed" />;
  if (status === 'failed') return <Chip size="small" sx={sx} color="error" label="failed" />;
  if (status === 'partial') return <Chip size="small" sx={sx} color="warning" label="partial" />;
  if (status === 'needs_input') return <Chip size="small" sx={sx} color="warning" label="needs_input" />;
  return <Chip size="small" sx={sx} variant="outlined" label={status} />;
}

function severityChip(run: RunListItemOut) {
  const flagged = run.flagged_attack_or_anomaly === true;
  const label = (run.predicted_label || '').toString().trim().toUpperCase();
  const sx = CHIP_COMPACT;

  if (flagged) return <Chip size="small" sx={sx} color="error" label="High" />;
  if (label && label !== 'BENIGN' && label !== 'NORMAL' && label !== 'UNKNOWN') {
    return <Chip size="small" sx={sx} color="warning" label="Medium" />;
  }
  return <Chip size="small" sx={sx} variant="outlined" label="Low" />;
}

function safeNumber(v: unknown): number | null {
  if (typeof v !== 'number' || Number.isNaN(v)) return null;
  return v;
}

function pct(n: number) {
  if (!Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function FullPageCornerLink({ href, title }: { href: string; title: string }) {
  return (
    <Tooltip title={title}>
      <IconButton
        component={RouterLink}
        href={href}
        size="small"
        color="default"
        aria-label={title}
        sx={{ color: 'text.secondary', mr: -0.25, mt: -0.25 }}
      >
        <Iconify icon="eva:arrow-ios-forward-fill" width={20} />
      </IconButton>
    </Tooltip>
  );
}

export function SocDashboardView() {
  const REFRESH_MS = 10_000;
  const toast = useAppSnackbar();
  const [runs, setRuns] = useState<RunListItemOut[]>([]);
  const [agentReports, setAgentReports] = useState<AgenticReportOut[]>([]);
  const [manualBusy, setManualBusy] = useState(false);
  const [error, setError] = useState<string>('');

  const [runsPage, setRunsPage] = useState(0);
  const [runsRowsPerPage, setRunsRowsPerPage] = useState(5);
  const [agentPage, setAgentPage] = useState(0);
  const [agentRowsPerPage, setAgentRowsPerPage] = useState(5);

  const [runDetailId, setRunDetailId] = useState<string | null>(null);

  const [runsTimeOrder, setRunsTimeOrder] = useState<TimeSortOrder>('desc');

  const kpis = useMemo(() => {
    const total = runs.length;
    const flagged = runs.filter((r) => r.flagged_attack_or_anomaly === true).length;
    const running = runs.filter((r) => r.status === 'running').length;
    const failed = runs.filter((r) => r.status === 'failed').length;
    const durations = runs.map((r) => safeNumber(r.duration_ms)).filter((x): x is number => typeof x === 'number');
    const avgDuration = durations.length ? Math.round(durations.reduce((a, b) => a + b, 0) / durations.length) : null;
    const flagRate = total ? flagged / total : 0;
    const failRate = total ? failed / total : 0;
    return { total, flagged, running, failed, avgDuration, flagRate, failRate };
  }, [runs]);

  const topLabels = useMemo(() => {
    const map = new Map<string, number>();
    for (const r of runs) {
      const label = (r.predicted_label || 'UNKNOWN').toString().trim() || 'UNKNOWN';
      map.set(label, (map.get(label) ?? 0) + 1);
    }
    return Array.from(map.entries())
      .map(([label, count]) => ({ label, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 4);
  }, [runs]);

  const runsSorted = useMemo(
    () => sortByTime(runs, (r) => r.updated_at, runsTimeOrder),
    [runs, runsTimeOrder]
  );

  const runsPageRows = useMemo(() => {
    const start = runsPage * runsRowsPerPage;
    return runsSorted.slice(start, start + runsRowsPerPage);
  }, [runsSorted, runsPage, runsRowsPerPage]);

  const agentReportsSorted = useMemo(
    () => sortByTime(agentReports, (r) => r.created_at, 'desc'),
    [agentReports]
  );

  const agentReportsPageRows = useMemo(() => {
    const start = agentPage * agentRowsPerPage;
    return agentReportsSorted.slice(start, start + agentRowsPerPage);
  }, [agentReportsSorted, agentPage, agentRowsPerPage]);

  const loadDashboard = useCallback(
    async (mode: 'silent' | 'manual') => {
      const manual = mode === 'manual';
      if (manual) {
        setManualBusy(true);
        setError('');
      }
      try {
        const [runData, repData] = await Promise.all([
          api.listRuns({ limit: 50 }),
          api.listAgentReports(25, 0),
        ]);
        setRuns(runData);
        setAgentReports(repData);
        setError('');
        if (manual) {
          toast.showSuccess('Dashboard refreshed', { autoHideMs: 3200 });
        } else {
          toast.showSuccess('Dashboard updated', { autoHideMs: REFRESH_MS });
        }
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
        if (manual) setError(msg);
        toast.showError(msg, { autoHideMs: 6000 });
      } finally {
        if (manual) setManualBusy(false);
      }
    },
    [toast]
  );

  useEffect(() => {
    void loadDashboard('silent');
    const t = window.setInterval(() => void loadDashboard('silent'), REFRESH_MS);
    return () => window.clearInterval(t);
  }, [loadDashboard]);

  return (
    <DashboardContent maxWidth="xl">
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.25 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            SOC Dashboard
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            Traffic detection, triage, and agentic actions (last 50 runs)
          </Typography>
        </Box>
        <Button size="small" variant="outlined" onClick={() => void loadDashboard('manual')} disabled={manualBusy}>
          {manualBusy ? 'Refreshing…' : 'Refresh'}
        </Button>
      </Stack>

      {error && (
        <Typography variant="caption" color="error" sx={{ mb: 1, display: 'block' }}>
          {error}
        </Typography>
      )}

      <Grid container spacing={1.25} sx={{ mb: 1.25 }}>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card
            sx={{
              overflow: 'hidden',
              position: 'relative',
              bgcolor: (theme) => theme.vars.palette.background.paper,
              backgroundImage: (theme) =>
                `radial-gradient(800px circle at 10% 10%, ${theme.vars.palette.primary.lighter}22, transparent 55%),` +
                `radial-gradient(800px circle at 90% 30%, ${theme.vars.palette.info.lighter}18, transparent 50%)`,
            }}
          >
            <CardHeader
              avatar={headerAvatarIcon('eva:trending-up-fill', 'primary')}
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs table" />}
              title="Runs"
              subheader="Last 50"
              sx={SX_COMPACT_CARD_HEADER}
            />
            <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
              <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10, lineHeight: 1.2 }}>
                Runs
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{kpis.total}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card
            sx={{
              overflow: 'hidden',
              position: 'relative',
              backgroundImage: (theme) =>
                `radial-gradient(800px circle at 10% 20%, ${theme.vars.palette.error.lighter}1f, transparent 55%),` +
                `radial-gradient(800px circle at 90% 10%, ${theme.vars.palette.warning.lighter}15, transparent 55%)`,
            }}
          >
            <CardHeader
              avatar={headerAvatarIcon('solar:bell-bing-bold-duotone', 'error')}
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs table" />}
              title="Flagged"
              subheader="Triage first"
              sx={SX_COMPACT_CARD_HEADER}
            />
            <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
              <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10, lineHeight: 1.2 }}>
                Flagged
              </Typography>
              <Stack direction="row" spacing={1} alignItems="baseline">
                <Typography variant="h5" sx={{ fontWeight: 700 }}>{kpis.flagged}</Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {pct(kpis.flagRate)}
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card
            sx={{
              overflow: 'hidden',
              position: 'relative',
              backgroundImage: (theme) =>
                `radial-gradient(800px circle at 20% 20%, ${theme.vars.palette.info.lighter}1f, transparent 60%),` +
                `radial-gradient(800px circle at 90% 50%, ${theme.vars.palette.primary.lighter}12, transparent 55%)`,
            }}
          >
            <CardHeader
              avatar={headerAvatarIcon('solar:restart-bold', 'info')}
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs table" />}
              title="Running now"
              subheader="Active pipelines"
              sx={SX_COMPACT_CARD_HEADER}
            />
            <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
              <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10, lineHeight: 1.2 }}>
                Running now
              </Typography>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>{kpis.running}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, sm: 6, md: 3 }}>
          <Card
            sx={{
              overflow: 'hidden',
              position: 'relative',
              backgroundImage: (theme) =>
                `radial-gradient(800px circle at 20% 20%, ${theme.vars.palette.warning.lighter}1a, transparent 60%),` +
                `radial-gradient(800px circle at 90% 50%, ${theme.vars.palette.error.lighter}16, transparent 55%)`,
            }}
          >
            <CardHeader
              avatar={headerAvatarIcon('solar:check-circle-bold', 'success')}
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs table" />}
              title="Reliability"
              subheader="Fail rate + duration"
              sx={SX_COMPACT_CARD_HEADER}
            />
            <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
              <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10, lineHeight: 1.2 }}>
                Fail rate
              </Typography>
              <Stack direction="row" spacing={1} alignItems="baseline">
                <Typography variant="h5" sx={{ fontWeight: 700 }}>{kpis.failed}</Typography>
                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                  {pct(kpis.failRate)}
                </Typography>
              </Stack>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mt: 0.25 }}>
                Avg duration: {typeof kpis.avgDuration === 'number' ? `${kpis.avgDuration}ms` : '—'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={1.25}>
        <Grid size={{ xs: 12 }}>
            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('solar:eye-bold', 'primary')}
                action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs table" />}
                title="Detections & triage"
                subheader="Short view · Run, attack, severity, status · open monitor for full detail"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1, width: 1, minWidth: 0 }}>
                {topLabels.length > 0 && (
                  <Stack direction="row" flexWrap="wrap" useFlexGap spacing={0.5} sx={{ mb: 1, rowGap: 0.5 }}>
                    {topLabels.map((x) => (
                      <Chip
                        key={x.label}
                        size="small"
                        variant="outlined"
                        label={`${x.label}: ${x.count}`}
                        sx={CHIP_COMPACT}
                      />
                    ))}
                  </Stack>
                )}
                <Box sx={{ width: 1, minWidth: 0 }}>
                  <Table size="small" sx={SX_COMPACT_TABLE}>
                    {RUNS_TABLE_COLGROUP}
                    <TableHead>
                      <TableRow>
                        <TableCell>Run</TableCell>
                        <TableCell>Attack</TableCell>
                        <TableCell>Severity</TableCell>
                        <TableCell>Status</TableCell>
                        <TimeSortHeadCell
                          label="Updated"
                          order={runsTimeOrder}
                          onOrderChange={setRunsTimeOrder}
                        />
                        <TableCell align="center" sx={{ px: 0.75 }}>
                          Open
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {runsPageRows.map((r) => {
                        const flagged = r.flagged_attack_or_anomaly === true;
                        return (
                          <TableRow
                            hover
                            key={r.run_id}
                            sx={{
                              ...(flagged && {
                                bgcolor: (theme) => theme.vars.palette.error.lighter,
                              }),
                            }}
                          >
                            <TableCell sx={{ verticalAlign: 'top', fontFamily: 'monospace', fontSize: 11 }} title={r.run_id}>
                              {runIdShort(r.run_id)}
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top', minWidth: 0 }}>
                              <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap" useFlexGap>
                                {r.predicted_label ? (
                                  <Chip size="small" sx={CHIP_COMPACT} variant="outlined" label={r.predicted_label} />
                                ) : (
                                  '—'
                                )}
                                {flagged ? (
                                  <Chip size="small" sx={CHIP_COMPACT} color="error" label="flagged" />
                                ) : null}
                              </Stack>
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top' }}>{severityChip(r)}</TableCell>
                            <TableCell sx={{ verticalAlign: 'top' }}>{statusChip(r.status)}</TableCell>
                            <TableCell sx={{ verticalAlign: 'top', typography: 'caption', whiteSpace: 'nowrap' }}>
                              {fDateTime(r.updated_at)}
                            </TableCell>
                            <TableCell align="center" sx={{ verticalAlign: 'top', px: 0.5 }}>
                              <Tooltip title="Open run details">
                                <IconButton
                                  size="small"
                                  color="primary"
                                  aria-label="View run details"
                                  onClick={() => setRunDetailId(r.run_id)}
                                  sx={{ p: 0.35 }}
                                >
                                  <Iconify icon="solar:eye-bold" width={18} />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                      {!runsPageRows.length && (
                        <TableRow>
                          <TableCell colSpan={6}>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                              No runs yet.
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </Box>
                <TablePagination
                  component="div"
                  size="small"
                  rowsPerPageOptions={[5, 10, 15]}
                  sx={{ '& .MuiTablePagination-toolbar': { minHeight: 44, px: 0.5 } }}
                  count={runsSorted.length}
                  page={runsPage}
                  onPageChange={(_, p) => setRunsPage(p)}
                  rowsPerPage={runsRowsPerPage}
                  onRowsPerPageChange={(e) => {
                    setRunsRowsPerPage(parseInt(e.target.value, 10));
                    setRunsPage(0);
                  }}
                />
              </CardContent>
            </Card>
        </Grid>

        <Grid size={{ xs: 12 }}>
            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('solar:shield-keyhole-bold-duotone', 'secondary')}
                action={<FullPageCornerLink href="/agentic" title="Open Agentic actions — full list" />}
                title="Agentic actions"
                subheader="Short view · Plan, attack, detection actions, status · open full list for apply"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1, width: 1, minWidth: 0 }}>
                <Box sx={{ width: 1, minWidth: 0 }}>
                  <Table size="small" sx={SX_COMPACT_TABLE}>
                    {AGENTIC_TABLE_COLGROUP}
                    <TableHead>
                      <TableRow>
                        <TableCell>Plan</TableCell>
                        <TableCell>Attack</TableCell>
                        <TableCell>Detection actions</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Created</TableCell>
                        <TableCell align="center" sx={{ px: 0.75 }}>
                          Open
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {agentReportsPageRows.map((r) => {
                        const attackType = attackTypeFromReportRow(r);
                        const allActions = actionsFromReport(r);
                        const actionsCsv =
                          allActions
                            .map((a) => a.action)
                            .filter((label) => label && label !== '—')
                            .join(', ') || '—';
                        return (
                          <TableRow key={r.public_id} hover>
                            <TableCell sx={{ verticalAlign: 'top' }}>
                              <PlanIdCell publicId={r.public_id} />
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top' }}>
                              {attackType ? (
                                <Chip size="small" sx={CHIP_COMPACT} variant="outlined" label={attackType} />
                              ) : (
                                '—'
                              )}
                            </TableCell>
                            <TableCell
                              sx={{
                                verticalAlign: 'top',
                                minWidth: 0,
                                typography: 'caption',
                                lineHeight: 1.35,
                                wordBreak: 'break-word',
                              }}
                              title={actionsCsv !== '—' ? actionsCsv : undefined}
                            >
                              {actionsCsv}
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top' }}>
                              <ExecStatusChip status={execStatusLabel(r)} />
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top', typography: 'caption', whiteSpace: 'nowrap' }}>
                              {fDateTime(r.created_at)}
                            </TableCell>
                            <TableCell align="center" sx={{ verticalAlign: 'top', px: 0.5 }}>
                              <Tooltip title="Open full report">
                                <IconButton
                                  component={RouterLink}
                                  href={`/agentic/report/${encodeURIComponent(r.public_id)}`}
                                  size="small"
                                  color="default"
                                  aria-label="Open agentic report"
                                  sx={{ p: 0.35 }}
                                >
                                  <Iconify icon="eva:arrow-ios-forward-fill" width={18} />
                                </IconButton>
                              </Tooltip>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                      {!agentReportsPageRows.length && (
                        <TableRow>
                          <TableCell colSpan={6}>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                              No agent reports yet.
                            </Typography>
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </Box>
                {agentReportsSorted.length > 0 && (
                  <TablePagination
                    component="div"
                    size="small"
                    rowsPerPageOptions={[5, 10, 15]}
                    sx={{ '& .MuiTablePagination-toolbar': { minHeight: 44, px: 0.5 } }}
                    count={agentReportsSorted.length}
                    page={agentPage}
                    onPageChange={(_, p) => setAgentPage(p)}
                    rowsPerPage={agentRowsPerPage}
                    onRowsPerPageChange={(e) => {
                      setAgentRowsPerPage(parseInt(e.target.value, 10));
                      setAgentPage(0);
                    }}
                  />
                )}
              </CardContent>
            </Card>
        </Grid>
      </Grid>

      <RunDetailDialog open={Boolean(runDetailId)} runId={runDetailId} onClose={() => setRunDetailId(null)} />
    </DashboardContent>
  );
}

