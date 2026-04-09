import type { ExecutionReportDetailOut, ExecutionReportListItemOut } from 'src/api/types';

import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TableContainer from '@mui/material/TableContainer';
import CircularProgress from '@mui/material/CircularProgress';

import { RouterLink } from 'src/routes/components';

import { DashboardContent } from 'src/layouts/dashboard';
import { ApiError, getExecutionReport, listExecutionReports } from 'src/services';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

function shortId(s: string): string {
  if (!s) return '—';
  return s.length > 14 ? `${s.slice(0, 8)}…${s.slice(-4)}` : s;
}

export function AgenticExecutionReportsView() {
  const [rows, setRows] = useState<ExecutionReportListItemOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');

  const [open, setOpen] = useState(false);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [detail, setDetail] = useState<ExecutionReportDetailOut | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const data = await listExecutionReports(500, 0);
      setRows(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const openDetail = async (id: number) => {
    setOpen(true);
    setActiveId(id);
    setDetail(null);
    setDetailError('');
    setDetailLoading(true);
    try {
      const d = await getExecutionReport(id);
      setDetail(d);
    } catch (e) {
      setDetailError(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
    } finally {
      setDetailLoading(false);
    }
  };

  const close = () => {
    setOpen(false);
    setActiveId(null);
    setDetail(null);
    setDetailError('');
    setDetailLoading(false);
  };

  return (
    <DashboardContent maxWidth="xl">
      <Stack spacing={3}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={2}>
          <Typography variant="h4">Execution reports</Typography>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Iconify icon="solar:restart-bold" />}
            onClick={() => void load()}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>

        {error && <Alert severity="error">{error}</Alert>}

        <Card>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700 }}>Created</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Report</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Integrity</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Error</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>
                    Actions
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={6} align="center" sx={{ py: 6 }}>
                      <CircularProgress size={28} />
                    </TableCell>
                  </TableRow>
                ) : rows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                        No execution reports yet.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  rows.map((r) => (
                    <TableRow key={r.id} hover>
                      <TableCell sx={{ whiteSpace: 'nowrap' }}>{new Date(r.created_at).toLocaleString()}</TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }} title={r.agentic_report_public_id}>
                        <Link
                          component={RouterLink}
                          href={`/agentic/report/${encodeURIComponent(r.agentic_report_public_id)}`}
                          variant="body2"
                        >
                          {shortId(r.agentic_report_public_id)}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          variant="outlined"
                          color={r.status === 'applied' ? 'success' : 'error'}
                          label={r.status}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip size="small" variant="outlined" label={r.integrity_overall} />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {r.error_reason || '—'}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Button size="small" variant="contained" onClick={() => void openDetail(r.id)}>
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </Card>
      </Stack>

      <Dialog open={open} onClose={close} maxWidth="md" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 1 }}>
          Execution report
          <IconButton aria-label="close" onClick={close}>
            <Iconify icon="mingcute:close-line" />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading && (
            <Box display="flex" justifyContent="center" py={4}>
              <CircularProgress />
            </Box>
          )}
          {detailError && <Alert severity="error">{detailError}</Alert>}
          {detail && !detailLoading && (
            <Stack spacing={2}>
              <Stack direction="row" gap={1} flexWrap="wrap" alignItems="center">
                <Chip size="small" variant="outlined" label={`id=${detail.id}`} />
                <Chip size="small" variant="outlined" label={`status=${detail.status}`} />
                <Chip size="small" variant="outlined" label={`integrity=${detail.integrity_overall}`} />
              </Stack>

              {detail.error_reason && (
                <Alert severity="error">
                  <strong>{detail.error_reason}</strong>
                  {detail.error_detail ? ` — ${detail.error_detail}` : ''}
                </Alert>
              )}

              <Divider />
              <Typography variant="subtitle2">Per-tier outcomes (stubbed)</Typography>
              <Typography
                variant="body2"
                sx={{
                  fontFamily: 'monospace',
                  fontSize: 12,
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  border: 1,
                  borderColor: 'divider',
                  borderRadius: 1,
                  p: 1.25,
                }}
              >
                {JSON.stringify(
                  {
                    core: detail.actions_core_json,
                    edge: detail.actions_edge_json,
                    ran: detail.actions_ran_json,
                  },
                  null,
                  2
                )}
              </Typography>
            </Stack>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={close}>Close</Button>
          {activeId != null && (
            <Button
              variant="outlined"
              startIcon={<Iconify icon="solar:restart-bold" />}
              disabled={detailLoading}
              onClick={() => void openDetail(activeId)}
            >
              Re-load
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </DashboardContent>
  );
}

