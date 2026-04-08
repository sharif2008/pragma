import type { IconifyName } from 'src/components/iconify/register-icons';
import type { RunEventOut, RunListItemOut, AgenticReportOut } from 'src/api/types';

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

import { sortByTime, type TimeSortOrder } from 'src/utils/table-time-sort';

import { api, ApiError } from 'src/services';
import { DashboardContent } from 'src/layouts/dashboard';
import { useAppSnackbar } from 'src/contexts/app-snackbar-context';

import { Iconify } from 'src/components/iconify';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import { RunDetailDialog, RunEventDetailDialog } from 'src/components/run-monitoring/detail-dialogs';

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
    <col style={{ width: '18%' }} />
    <col style={{ width: '17%' }} />
    <col style={{ width: '38%' }} />
    <col style={{ width: '20%' }} />
    <col style={{ width: '7%' }} />
  </colgroup>
);

const EVENTS_TABLE_COLGROUP = (
  <colgroup>
    <col style={{ width: '22%' }} />
    <col style={{ width: '13%' }} />
    <col style={{ width: '60%' }} />
    <col style={{ width: '5%' }} />
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

function clampCellSx(lines: number) {
  return {
    overflow: 'hidden',
    wordBreak: 'break-word' as const,
    '& .MuiTypography-root': {
      display: '-webkit-box',
      WebkitLineClamp: lines,
      WebkitBoxOrient: 'vertical' as const,
      overflow: 'hidden',
    },
  };
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
  const [events, setEvents] = useState<RunEventOut[]>([]);
  const [agentReports, setAgentReports] = useState<AgenticReportOut[]>([]);
  const [manualBusy, setManualBusy] = useState(false);
  const [error, setError] = useState<string>('');

  const [runsPage, setRunsPage] = useState(0);
  const [runsRowsPerPage, setRunsRowsPerPage] = useState(10);
  const [eventsPage, setEventsPage] = useState(0);
  const [eventsRowsPerPage, setEventsRowsPerPage] = useState(10);

  const [runDetailId, setRunDetailId] = useState<string | null>(null);
  const [eventDetail, setEventDetail] = useState<RunEventOut | null>(null);

  const [runsTimeOrder, setRunsTimeOrder] = useState<TimeSortOrder>('desc');
  const [eventsTimeOrder, setEventsTimeOrder] = useState<TimeSortOrder>('desc');

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
      .slice(0, 6);
  }, [runs]);

  const runsSorted = useMemo(
    () => sortByTime(runs, (r) => r.updated_at, runsTimeOrder),
    [runs, runsTimeOrder]
  );

  const runsPageRows = useMemo(() => {
    const start = runsPage * runsRowsPerPage;
    return runsSorted.slice(start, start + runsRowsPerPage);
  }, [runsSorted, runsPage, runsRowsPerPage]);

  /** Newest first, cap for compact SOC summary card */
  const latestAgentReports = useMemo(
    () => sortByTime(agentReports, (r) => r.created_at, 'desc').slice(0, 5),
    [agentReports]
  );

  const sortedEvents = useMemo(
    () => sortByTime(events, (e) => e.timestamp, eventsTimeOrder),
    [events, eventsTimeOrder]
  );

  const eventsSummary = useMemo(() => {
    const total = sortedEvents.length;
    const byLevel = new Map<string, number>();
    const byStep = new Map<string, number>();
    for (const e of sortedEvents) {
      byLevel.set(e.level, (byLevel.get(e.level) ?? 0) + 1);
      byStep.set(e.step_name, (byStep.get(e.step_name) ?? 0) + 1);
    }
    const topSteps = Array.from(byStep.entries())
      .map(([step, count]) => ({ step, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 6);
    return {
      total,
      info: byLevel.get('info') ?? 0,
      warn: byLevel.get('warn') ?? 0,
      error: byLevel.get('error') ?? 0,
      topSteps,
    };
  }, [sortedEvents]);

  const loadDashboard = useCallback(
    async (mode: 'silent' | 'manual') => {
      const manual = mode === 'manual';
      if (manual) {
        setManualBusy(true);
        setError('');
      }
      try {
        const offset = eventsPage * eventsRowsPerPage;
        const [runData, evData, repData] = await Promise.all([
          api.listRuns({ limit: 50 }),
          api.listRunEvents({ limit: eventsRowsPerPage, offset }),
          api.listAgentReports(25, 0),
        ]);
        setRuns(runData);
        setEvents(evData);
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
    [eventsPage, eventsRowsPerPage, toast]
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
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs & events" />}
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
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs & events" />}
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
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs & events" />}
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
              action={<FullPageCornerLink href="/monitor" title="Open Monitor — full runs & events" />}
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
        <Grid size={{ xs: 12, lg: 6 }}>
          <Stack spacing={1.25}>
            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('solar:eye-bold', 'primary')}
                action={<FullPageCornerLink href="/monitor" title="Open Monitor — full table & filters" />}
                title="Detections & triage"
                subheader="Essential fields · open a row for channel, customer, run id, and trace"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1, width: 1, minWidth: 0 }}>
                {topLabels.length > 0 && (
                  <Stack direction="row" flexWrap="wrap" useFlexGap spacing={0.5} sx={{ mb: 1.25, rowGap: 0.5 }}>
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
                        <TableCell>Triage</TableCell>
                        <TableCell>Label</TableCell>
                        <TableCell>Message</TableCell>
                        <TimeSortHeadCell
                          label="Updated"
                          order={runsTimeOrder}
                          onOrderChange={setRunsTimeOrder}
                        />
                        <TableCell align="center" sx={{ px: 0.75 }}>
                          View
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
                            <TableCell sx={{ verticalAlign: 'top' }}>
                              <Stack spacing={0.35} alignItems="flex-start">
                                {severityChip(r)}
                                {statusChip(r.status)}
                              </Stack>
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top', minWidth: 0 }}>
                              <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap" useFlexGap>
                                <Typography variant="body2" component="span" fontWeight={600}>
                                  {r.predicted_label ?? '—'}
                                </Typography>
                                {flagged && <Chip size="small" sx={CHIP_COMPACT} color="error" label="flagged" />}
                              </Stack>
                            </TableCell>
                            <TableCell sx={{ verticalAlign: 'top', minWidth: 0, ...clampCellSx(2) }}>
                              <Typography variant="body2" title={r.message_preview ?? ''}>
                                {r.message_preview ?? '—'}
                              </Typography>
                            </TableCell>
                            <TableCell
                              sx={{
                                verticalAlign: 'top',
                                typography: 'caption',
                                lineHeight: 1.35,
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {new Date(r.updated_at).toLocaleString()}
                            </TableCell>
                            <TableCell align="center" sx={{ verticalAlign: 'top', px: 0.5 }}>
                              <Tooltip title="Details (channel, customer, ids, steps)">
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
                    </TableBody>
                  </Table>
                </Box>
                <TablePagination
                  component="div"
                  size="small"
                  rowsPerPageOptions={[10, 25, 50]}
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

            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('solar:shield-keyhole-bold-duotone', 'secondary')}
                action={<FullPageCornerLink href="/agentic" title="Open Agentic actions — full list" />}
                title="Agentic actions"
                subheader="Recommended action & time · open row for full report page"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1, overflow: 'hidden' }}>
                {agentReports.length ? (
                  <Stack spacing={1} sx={{ width: 1, minWidth: 0 }}>
                    {latestAgentReports.map((r) => (
                      <Stack
                        key={r.public_id}
                        direction="row"
                        alignItems="center"
                        spacing={1.25}
                        sx={{ minWidth: 0, width: 1, py: 0.25 }}
                      >
                        <Box
                          sx={{
                            color: 'primary.main',
                            flexShrink: 0,
                            width: 28,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                          }}
                        >
                          <Iconify icon="solar:chat-round-dots-bold" width={18} />
                        </Box>
                        <Box sx={{ minWidth: 0, flex: '1 1 auto' }}>
                          <Typography
                            variant="body2"
                            sx={{ fontWeight: 600, display: 'block', lineHeight: 1.4 }}
                            title={r.recommended_action || undefined}
                          >
                            {r.recommended_action || '—'}
                          </Typography>
                          <Stack direction="row" alignItems="center" spacing={0.5} sx={{ mt: 0.35, minWidth: 0 }}>
                            <Iconify icon="solar:clock-circle-outline" width={14} style={{ opacity: 0.65, flexShrink: 0 }} />
                            <Typography variant="caption" color="text.secondary" noWrap sx={{ minWidth: 0 }}>
                              {new Date(r.created_at).toLocaleString()}
                            </Typography>
                          </Stack>
                        </Box>
                        <Tooltip title="Open full report">
                          <IconButton
                            component={RouterLink}
                            href={`/agentic/report/${encodeURIComponent(r.public_id)}`}
                            size="small"
                            color="default"
                            aria-label="Open full agentic report"
                            sx={{ color: 'text.secondary', flexShrink: 0, width: 36, height: 36 }}
                          >
                            <Iconify icon="eva:arrow-ios-forward-fill" width={18} />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    ))}
                  </Stack>
                ) : (
                  <Stack direction="row" spacing={1} alignItems="center">
                    <Iconify icon="eva:search-fill" width={20} style={{ opacity: 0.55 }} />
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      No agent reports yet.
                    </Typography>
                  </Stack>
                )}
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        <Grid size={{ xs: 12, lg: 6 }}>
          <Stack spacing={1.25}>
            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('solar:clock-circle-outline', 'info')}
                action={<FullPageCornerLink href="/monitor" title="Open Monitor — full events table" />}
                title="Events"
                subheader="Time, level, step & message · open a row for run id, trace, timing"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1, width: 1, minWidth: 0 }}>
                <Box sx={{ width: 1, minWidth: 0 }}>
                  <Table size="small" sx={SX_COMPACT_TABLE}>
                    {EVENTS_TABLE_COLGROUP}
                    <TableHead>
                      <TableRow>
                        <TimeSortHeadCell
                          label="Time"
                          order={eventsTimeOrder}
                          onOrderChange={setEventsTimeOrder}
                        />
                        <TableCell>Level</TableCell>
                        <TableCell>Step &amp; message</TableCell>
                        <TableCell align="center" sx={{ px: 0.75 }}>
                          View
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sortedEvents.map((e) => (
                        <TableRow key={e.event_id} hover>
                          <TableCell sx={{ typography: 'caption', lineHeight: 1.35, verticalAlign: 'top' }}>
                            {new Date(e.timestamp).toLocaleString()}
                          </TableCell>
                          <TableCell sx={{ verticalAlign: 'top' }}>
                            <Chip
                              size="small"
                              sx={CHIP_COMPACT}
                              color={e.level === 'error' ? 'error' : e.level === 'warn' ? 'warning' : 'default'}
                              label={e.level}
                              variant={e.level === 'info' ? 'outlined' : 'filled'}
                            />
                          </TableCell>
                          <TableCell sx={{ verticalAlign: 'top', minWidth: 0 }}>
                            <Typography
                              variant="caption"
                              fontFamily="monospace"
                              fontSize={11}
                              color="text.secondary"
                              display="block"
                              sx={{ mb: 0.25 }}
                            >
                              {e.step_name}
                            </Typography>
                            <Box sx={{ ...clampCellSx(2) }}>
                              <Typography variant="body2" title={e.message}>
                                {e.message}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell align="center" sx={{ verticalAlign: 'top', px: 0.5 }}>
                            <Tooltip title="Details (run, trace, duration ms)">
                              <IconButton
                                size="small"
                                color="primary"
                                aria-label="View event details"
                                onClick={() => setEventDetail(e)}
                                sx={{ p: 0.35 }}
                              >
                                <Iconify icon="solar:eye-bold" width={18} />
                              </IconButton>
                            </Tooltip>
                          </TableCell>
                        </TableRow>
                      ))}
                      {!sortedEvents.length && (
                        <TableRow>
                          <TableCell colSpan={4}>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                              No events yet.
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
                  rowsPerPageOptions={[10, 25, 50]}
                  sx={{ '& .MuiTablePagination-toolbar': { minHeight: 44, px: 0.5 } }}
                  count={-1}
                  page={eventsPage}
                  onPageChange={(_, p) => setEventsPage(p)}
                  rowsPerPage={eventsRowsPerPage}
                  onRowsPerPageChange={(e) => {
                    setEventsRowsPerPage(parseInt(e.target.value, 10));
                    setEventsPage(0);
                  }}
                  labelDisplayedRows={({ from, to }) => `${from}–${to}`}
                />
              </CardContent>
            </Card>

            <Card sx={{ overflow: 'hidden' }}>
              <CardHeader
                avatar={headerAvatarIcon('ic:round-filter-list', 'warning')}
                action={<FullPageCornerLink href="/monitor" title="Open Monitor — full event history" />}
                title="Event summary"
                subheader="Counts from current page"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
                <Grid container spacing={1}>
                  <Grid size={{ xs: 4 }}>
                    <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10 }}>
                      Errors
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>{eventsSummary.error}</Typography>
                  </Grid>
                  <Grid size={{ xs: 4 }}>
                    <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10 }}>
                      Warn
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>{eventsSummary.warn}</Typography>
                  </Grid>
                  <Grid size={{ xs: 4 }}>
                    <Typography variant="overline" sx={{ color: 'text.secondary', fontSize: 10 }}>
                      Info
                    </Typography>
                    <Typography variant="h6" sx={{ fontWeight: 700 }}>{eventsSummary.info}</Typography>
                  </Grid>
                </Grid>

                <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 1 }}>
                  {eventsSummary.topSteps.map((s) => (
                    <Chip
                      key={s.step}
                      size="small"
                      variant="outlined"
                      label={`${s.step}: ${s.count}`}
                      sx={CHIP_COMPACT}
                    />
                  ))}
                  {!eventsSummary.topSteps.length && (
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      No step stats yet.
                    </Typography>
                  )}
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </Grid>
      </Grid>

      <RunDetailDialog open={Boolean(runDetailId)} runId={runDetailId} onClose={() => setRunDetailId(null)} />
      <RunEventDetailDialog
        open={Boolean(eventDetail)}
        event={eventDetail}
        onClose={() => setEventDetail(null)}
      />
    </DashboardContent>
  );
}

