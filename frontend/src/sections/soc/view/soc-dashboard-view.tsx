import type { AgenticReportOut, RunEventOut, RunListItemOut } from 'src/api/types';

import { useCallback, useEffect, useMemo, useState } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';
import CardHeader from '@mui/material/CardHeader';
import CardContent from '@mui/material/CardContent';
import TablePagination from '@mui/material/TablePagination';
import TableContainer from '@mui/material/TableContainer';

import { Iconify } from 'src/components/iconify';
import {
  AgentReportDetailDialog,
  RunDetailDialog,
  RunEventDetailDialog,
} from 'src/components/run-monitoring/detail-dialogs';
import { useAppSnackbar } from 'src/contexts/app-snackbar-context';
import { api, ApiError } from 'src/services';
import { DashboardContent } from 'src/layouts/dashboard';

const SX_COMPACT_TABLE = {
  '& .MuiTableCell-root': { py: 0.5, px: 1, fontSize: '0.8125rem' },
  '& .MuiTableCell-head': { fontWeight: 700, fontSize: '0.75rem' },
} as const;

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

function shortId(id: string) {
  if (!id) return '—';
  if (id.length <= 14) return id;
  return `${id.slice(0, 8)}…${id.slice(-4)}`;
}

function safeNumber(v: unknown): number | null {
  if (typeof v !== 'number' || Number.isNaN(v)) return null;
  return v;
}

function pct(n: number) {
  if (!Number.isFinite(n)) return '—';
  return `${(n * 100).toFixed(1)}%`;
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
  const [reportDetailId, setReportDetailId] = useState<string | null>(null);

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

  const runsPageRows = useMemo(() => {
    const start = runsPage * runsRowsPerPage;
    return runs.slice(start, start + runsRowsPerPage);
  }, [runs, runsPage, runsRowsPerPage]);

  const eventsSummary = useMemo(() => {
    const total = events.length;
    const byLevel = new Map<string, number>();
    const byStep = new Map<string, number>();
    for (const e of events) {
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
  }, [events]);

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
              title="Runs"
              subheader="Last 50"
              sx={SX_COMPACT_CARD_HEADER}
              action={
                <Box sx={{ color: 'primary.main', mr: 0.25, display: 'flex', alignItems: 'center', opacity: 0.92 }}>
                  <Iconify icon="eva:trending-up-fill" width={22} />
                </Box>
              }
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
              title="Flagged"
              subheader="Triage first"
              sx={SX_COMPACT_CARD_HEADER}
              action={
                <Box sx={{ color: 'error.main', mr: 0.25, display: 'flex', alignItems: 'center', opacity: 0.88 }}>
                  <Iconify icon="solar:bell-bing-bold-duotone" width={24} />
                </Box>
              }
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
              title="Running now"
              subheader="Active pipelines"
              sx={SX_COMPACT_CARD_HEADER}
              action={
                <Box sx={{ color: 'info.main', mr: 0.25, display: 'flex', alignItems: 'center', opacity: 0.9 }}>
                  <Iconify icon="solar:restart-bold" width={22} />
                </Box>
              }
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
              title="Reliability"
              subheader="Fail rate + duration"
              sx={SX_COMPACT_CARD_HEADER}
              action={
                <Box sx={{ color: 'success.main', mr: 0.25, display: 'flex', alignItems: 'center', opacity: 0.9 }}>
                  <Iconify icon="solar:check-circle-bold" width={22} />
                </Box>
              }
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
            <Card>
              <CardHeader
                title="Detections & triage"
                subheader="Last 50 runs (paged)"
                sx={SX_COMPACT_CARD_HEADER}
                action={
                  <Stack direction="row" spacing={0.5} alignItems="center" flexWrap="wrap" useFlexGap>
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
                }
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
                <Box sx={{ overflowX: 'auto' }}>
                  <Table size="small" sx={{ minWidth: 920, ...SX_COMPACT_TABLE }}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Severity</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Label</TableCell>
                        <TableCell>Message</TableCell>
                        <TableCell>Last step</TableCell>
                        <TableCell align="right">Duration</TableCell>
                        <TableCell>Run</TableCell>
                        <TableCell align="right">View</TableCell>
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
                            <TableCell>{severityChip(r)}</TableCell>
                            <TableCell>{statusChip(r.status)}</TableCell>
                            <TableCell>
                              <Stack direction="row" spacing={1} alignItems="center">
                                <Typography variant="body2">{r.predicted_label ?? '—'}</Typography>
                                {flagged && <Chip size="small" sx={CHIP_COMPACT} color="error" label="flagged" />}
                              </Stack>
                            </TableCell>
                            <TableCell sx={{ maxWidth: 320 }}>
                              <Typography variant="body2" noWrap title={r.message_preview ?? ''}>
                                {r.message_preview ?? '—'}
                              </Typography>
                            </TableCell>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{r.last_step ?? '—'}</TableCell>
                            <TableCell align="right">
                              {typeof r.duration_ms === 'number' ? `${r.duration_ms}ms` : '—'}
                            </TableCell>
                            <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{shortId(r.run_id)}</TableCell>
                            <TableCell align="right">
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
                    </TableBody>
                  </Table>
                </Box>
                <TablePagination
                  component="div"
                  size="small"
                  rowsPerPageOptions={[10, 25, 50]}
                  sx={{ '& .MuiTablePagination-toolbar': { minHeight: 44, px: 0.5 } }}
                  count={runs.length}
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

            <Card>
              <CardHeader
                title="Agentic actions summary"
                subheader="Latest saved agent reports (scrollable · View opens full detail modal)"
                sx={SX_COMPACT_CARD_HEADER}
              />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
                {agentReports.length ? (
                  <Box sx={{ overflowX: 'auto' }}>
                    <TableContainer
                      sx={{
                        maxHeight: 320,
                        overflow: 'auto',
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                      }}
                    >
                      <Table size="small" stickyHeader sx={{ minWidth: 640, ...SX_COMPACT_TABLE }}>
                        <TableHead>
                          <TableRow>
                            <TableCell sx={{ bgcolor: 'background.paper' }}>Report</TableCell>
                            <TableCell sx={{ bgcolor: 'background.paper' }}>Recommended</TableCell>
                            <TableCell sx={{ bgcolor: 'background.paper' }}>Summary</TableCell>
                            <TableCell sx={{ bgcolor: 'background.paper' }}>Created</TableCell>
                            <TableCell align="right" sx={{ bgcolor: 'background.paper' }}>
                              View
                            </TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {agentReports.map((r) => (
                            <TableRow key={r.public_id} hover>
                              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{r.public_id}</TableCell>
                              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                                {r.recommended_action || '—'}
                              </TableCell>
                              <TableCell sx={{ maxWidth: 360 }}>
                                <Typography variant="body2" noWrap title={r.summary}>
                                  {r.summary}
                                </Typography>
                              </TableCell>
                              <TableCell sx={{ whiteSpace: 'nowrap' }}>
                                {new Date(r.created_at).toLocaleString()}
                              </TableCell>
                              <TableCell align="right">
                                <Tooltip title="Open full agent report (modal)">
                                  <Button
                                    size="small"
                                    variant="outlined"
                                    color="primary"
                                    aria-label="View agent report"
                                    onClick={() => setReportDetailId(r.public_id)}
                                    startIcon={<Iconify icon="solar:eye-bold" width={18} />}
                                    sx={{ py: 0.25, px: 1, minWidth: 0, fontSize: '0.75rem' }}
                                  >
                                    View
                                  </Button>
                                </Tooltip>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Box>
                ) : (
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    No agent reports found yet.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        <Grid size={{ xs: 12, lg: 6 }}>
          <Stack spacing={1.25}>
            <Card>
              <CardHeader title="Events" subheader="Newest first (across all runs)" sx={SX_COMPACT_CARD_HEADER} />
              <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
                <Box sx={{ overflowX: 'auto' }}>
                  <Table size="small" sx={{ minWidth: 920, ...SX_COMPACT_TABLE }}>
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Level</TableCell>
                        <TableCell>Step</TableCell>
                        <TableCell>Message</TableCell>
                        <TableCell align="right">ms</TableCell>
                        <TableCell>Run</TableCell>
                        <TableCell align="right">View</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {events.map((e) => (
                        <TableRow key={e.event_id} hover>
                          <TableCell sx={{ whiteSpace: 'nowrap' }}>{new Date(e.timestamp).toLocaleString()}</TableCell>
                          <TableCell>
                            <Chip
                              size="small"
                              sx={CHIP_COMPACT}
                              color={e.level === 'error' ? 'error' : e.level === 'warn' ? 'warning' : 'default'}
                              label={e.level}
                              variant={e.level === 'info' ? 'outlined' : 'filled'}
                            />
                          </TableCell>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{e.step_name}</TableCell>
                          <TableCell sx={{ maxWidth: 360 }}>
                            <Typography variant="body2" noWrap title={e.message}>
                              {e.message}
                            </Typography>
                          </TableCell>
                          <TableCell align="right">{typeof e.duration_ms === 'number' ? e.duration_ms : '—'}</TableCell>
                          <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{shortId(e.run_id)}</TableCell>
                          <TableCell align="right">
                            <Tooltip title="Open event details">
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
                      {!events.length && (
                        <TableRow>
                          <TableCell colSpan={7}>
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

            <Card>
              <CardHeader title="Event summary" subheader="Counts from current page" sx={SX_COMPACT_CARD_HEADER} />
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
      <AgentReportDetailDialog
        open={Boolean(reportDetailId)}
        publicId={reportDetailId}
        onClose={() => setReportDetailId(null)}
      />
    </DashboardContent>
  );
}

