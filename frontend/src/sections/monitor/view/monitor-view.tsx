import type { RunEventOut, RunListItemOut } from 'src/api/types';

import { useRef, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Tabs from '@mui/material/Tabs';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import Switch from '@mui/material/Switch';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableHead from '@mui/material/TableHead';
import TableCell from '@mui/material/TableCell';
import CardHeader from '@mui/material/CardHeader';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import Typography from '@mui/material/Typography';
import FormControl from '@mui/material/FormControl';
import CardContent from '@mui/material/CardContent';
import TablePagination from '@mui/material/TablePagination';
import FormControlLabel from '@mui/material/FormControlLabel';

import { sortByTime, type TimeSortOrder } from 'src/utils/table-time-sort';

import { api, ApiError } from 'src/services';
import { DashboardContent } from 'src/layouts/dashboard';
import { useAppSnackbar } from 'src/contexts/app-snackbar-context';

import { Iconify } from 'src/components/iconify';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import { RunDetailDialog, RunEventDetailDialog } from 'src/components/run-monitoring/detail-dialogs';

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

const REFRESH_INTERVAL_OPTIONS: { ms: number; label: string }[] = [
  { ms: 10_000, label: '10 seconds' },
  { ms: 30_000, label: '30 seconds' },
  { ms: 60_000, label: '1 minute' },
  { ms: 90_000, label: '1 minute 30 seconds' },
  { ms: 120_000, label: '2 minutes' },
];

function statusChip(status: string) {
  const sx = CHIP_COMPACT;
  if (status === 'running') return <Chip size="small" sx={sx} color="info" label="running" />;
  if (status === 'completed') return <Chip size="small" sx={sx} color="success" label="completed" />;
  if (status === 'failed') return <Chip size="small" sx={sx} color="error" label="failed" />;
  if (status === 'partial') return <Chip size="small" sx={sx} color="warning" label="partial" />;
  if (status === 'needs_input') return <Chip size="small" sx={sx} color="warning" label="needs_input" />;
  return <Chip size="small" sx={sx} variant="outlined" label={status} />;
}

export function MonitorView() {
  const toast = useAppSnackbar();
  const didShowReady = useRef(false);
  const lastSilentToastAt = useRef(0);
  const [runs, setRuns] = useState<RunListItemOut[]>([]);
  const [events, setEvents] = useState<RunEventOut[]>([]);
  const [manualBusy, setManualBusy] = useState(false);
  const [error, setError] = useState<string>('');
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const [tab, setTab] = useState<'live' | 'events'>('live');

  const [runsPage, setRunsPage] = useState(0);
  const [runsRowsPerPage, setRunsRowsPerPage] = useState(10);

  const [eventsPage, setEventsPage] = useState(0);
  const [eventsRowsPerPage, setEventsRowsPerPage] = useState(10);

  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshIntervalMs, setRefreshIntervalMs] = useState(10_000);

  const [runDetailId, setRunDetailId] = useState<string | null>(null);
  const [eventDetail, setEventDetail] = useState<RunEventOut | null>(null);

  const [runsTimeOrder, setRunsTimeOrder] = useState<TimeSortOrder>('desc');
  const [eventsTimeOrder, setEventsTimeOrder] = useState<TimeSortOrder>('desc');

  const filteredRuns = useMemo(() => {
    if (!flaggedOnly) return runs;
    return runs.filter((r) => r.flagged_attack_or_anomaly === true);
  }, [runs, flaggedOnly]);

  const liveRunsSorted = useMemo(
    () => sortByTime(filteredRuns, (r) => r.updated_at, runsTimeOrder),
    [filteredRuns, runsTimeOrder]
  );

  const runsPageRows = useMemo(() => {
    const start = runsPage * runsRowsPerPage;
    return liveRunsSorted.slice(start, start + runsRowsPerPage);
  }, [liveRunsSorted, runsPage, runsRowsPerPage]);

  const sortedEvents = useMemo(
    () => sortByTime(events, (e) => e.timestamp, eventsTimeOrder),
    [events, eventsTimeOrder]
  );

  const eventsPageRows = sortedEvents;

  const loadMonitoring = useCallback(
    async (mode: 'silent' | 'manual') => {
      const manual = mode === 'manual';
      if (manual) setManualBusy(true);
      try {
        const offset = eventsPage * eventsRowsPerPage;
        const [runData, evData] = await Promise.all([
          api.listRuns({ limit: 50 }),
          api.listRunEvents({ limit: eventsRowsPerPage, offset }),
        ]);
        setRuns(runData);
        setEvents(evData);
        setError('');
        if (manual) {
          toast.showSuccess('Monitoring refreshed', { autoHideMs: 3200 });
        } else if (!didShowReady.current) {
          didShowReady.current = true;
          toast.showSuccess('Monitoring ready', { autoHideMs: 3200 });
        } else if (Date.now() - lastSilentToastAt.current > 45_000) {
          lastSilentToastAt.current = Date.now();
          toast.showInfo('Monitoring updated', { autoHideMs: 2400 });
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
    void loadMonitoring('silent');
    if (!autoRefresh) return undefined;
    const t = window.setInterval(() => void loadMonitoring('silent'), refreshIntervalMs);
    return () => window.clearInterval(t);
  }, [loadMonitoring, autoRefresh, refreshIntervalMs]);

  return (
    <DashboardContent maxWidth="xl">
      <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 1.25 }}>
        <Box>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Monitoring
          </Typography>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block' }}>
            Top recent runs and events ·{' '}
            {autoRefresh
              ? `auto-refresh every ${REFRESH_INTERVAL_OPTIONS.find((o) => o.ms === refreshIntervalMs)?.label.toLowerCase() ?? `${Math.round(refreshIntervalMs / 1000)}s`}`
              : 'auto-refresh off (use Refresh)'}
          </Typography>
        </Box>
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={1}
          alignItems={{ xs: 'stretch', sm: 'center' }}
          flexWrap="wrap"
          useFlexGap
        >
          <FormControlLabel
            sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.8125rem' } }}
            control={
              <Switch size="small" checked={flaggedOnly} onChange={(e) => setFlaggedOnly(e.target.checked)} />
            }
            label="Flagged only"
          />
          <FormControlLabel
            sx={{ mr: 0, '& .MuiFormControlLabel-label': { fontSize: '0.8125rem' } }}
            control={
              <Switch size="small" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            }
            label="Auto refresh"
          />
          <FormControl size="small" sx={{ minWidth: 180 }} disabled={!autoRefresh}>
            <InputLabel id="monitor-refresh-interval-label">Interval</InputLabel>
            <Select
              labelId="monitor-refresh-interval-label"
              label="Interval"
              value={refreshIntervalMs}
              onChange={(e) => setRefreshIntervalMs(Number(e.target.value))}
            >
              {REFRESH_INTERVAL_OPTIONS.map((o) => (
                <MenuItem key={o.ms} value={o.ms}>
                  {o.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button size="small" variant="outlined" onClick={() => void loadMonitoring('manual')} disabled={manualBusy}>
            {manualBusy ? 'Refreshing…' : 'Refresh'}
          </Button>
        </Stack>
      </Stack>

      {error && (
        <Typography variant="caption" color="error" sx={{ mb: 1, display: 'block' }}>
          {error}
        </Typography>
      )}

      <Tabs
        value={tab}
        onChange={(_, v) => setTab(v)}
        sx={{ mb: 1, minHeight: 40, '& .MuiTab-root': { minHeight: 40, py: 0.5, fontSize: '0.8125rem' } }}
      >
        <Tab value="live" label="Live traffic" />
        <Tab value="events" label="All events" />
      </Tabs>

      {tab === 'live' ? (
        <Card sx={{ minWidth: 0, width: 1 }}>
          <CardHeader
            title="Live traffic monitoring"
            subheader="Sort by last update (default: newest first). Use the Updated column header to toggle order."
            sx={SX_COMPACT_CARD_HEADER}
          />
          <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
            <Box sx={{ overflowX: 'auto' }}>
              <Table size="small" sx={{ minWidth: 980, ...SX_COMPACT_TABLE }}>
                <TableHead>
                  <TableRow>
                    <TableCell>Status</TableCell>
                    <TimeSortHeadCell label="Updated" order={runsTimeOrder} onOrderChange={setRunsTimeOrder} />
                    <TableCell>Channel</TableCell>
                    <TableCell>Customer</TableCell>
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
                            bgcolor: (theme) => theme.vars.palette.warning.lighter,
                          }),
                        }}
                      >
                        <TableCell>{statusChip(r.status)}</TableCell>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{new Date(r.updated_at).toLocaleTimeString()}</TableCell>
                        <TableCell>{r.channel ?? '—'}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, whiteSpace: 'nowrap' }}>
                          {r.customer_id ?? '—'}
                        </TableCell>
                        <TableCell>
                          <Stack direction="row" spacing={1} alignItems="center">
                            <Typography variant="body2">{r.predicted_label ?? '—'}</Typography>
                            {flagged && <Chip size="small" sx={CHIP_COMPACT} color="warning" label="flagged" />}
                          </Stack>
                        </TableCell>
                        <TableCell sx={{ maxWidth: 360 }}>
                          <Typography variant="body2" noWrap title={r.message_preview ?? ''}>
                            {r.message_preview ?? '—'}
                          </Typography>
                        </TableCell>
                        <TableCell sx={{ minWidth: 160 }}>{r.last_step ?? '—'}</TableCell>
                        <TableCell align="right">
                          {typeof r.duration_ms === 'number' ? `${r.duration_ms}ms` : '—'}
                        </TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, whiteSpace: 'nowrap' }}>
                          {r.run_id.slice(0, 8)}…{r.run_id.slice(-4)}
                        </TableCell>
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
              count={liveRunsSorted.length}
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
      ) : (
        <Card sx={{ minWidth: 0, width: 1 }}>
          <CardHeader
            title="All events"
            subheader="Sort by time (default: newest first). Toggle via the Time column header."
            sx={SX_COMPACT_CARD_HEADER}
          />
          <CardContent sx={{ pt: 0, px: 1.5, pb: 1 }}>
            <Box sx={{ overflowX: 'auto' }}>
              <Table size="small" sx={{ minWidth: 980, ...SX_COMPACT_TABLE }}>
                <TableHead>
                  <TableRow>
                    <TimeSortHeadCell label="Time" order={eventsTimeOrder} onOrderChange={setEventsTimeOrder} />
                    <TableCell>Level</TableCell>
                    <TableCell>Step</TableCell>
                    <TableCell>Message</TableCell>
                    <TableCell align="right">ms</TableCell>
                    <TableCell>Run</TableCell>
                    <TableCell align="right">View</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {eventsPageRows.map((e) => (
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
                      <TableCell sx={{ maxWidth: 520 }}>
                        <Typography variant="body2" noWrap title={e.message}>
                          {e.message}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">{typeof e.duration_ms === 'number' ? e.duration_ms : '—'}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                        {e.run_id.slice(0, 8)}…{e.run_id.slice(-4)}
                      </TableCell>
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
      )}

      <RunDetailDialog open={Boolean(runDetailId)} runId={runDetailId} onClose={() => setRunDetailId(null)} />
      <RunEventDetailDialog
        open={Boolean(eventDetail)}
        event={eventDetail}
        onClose={() => setEventDetail(null)}
      />
    </DashboardContent>
  );
}

