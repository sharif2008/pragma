import type { ExecutionReportDetailOut, ExecutionReportListItemOut } from 'src/api/types';

import { useMemo, useState, useEffect, useCallback } from 'react';

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
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';

import { RouterLink } from 'src/routes/components';

import { DashboardContent } from 'src/layouts/dashboard';
import { ApiError, getExecutionReport, listExecutionReports } from 'src/services';

import { Iconify } from 'src/components/iconify';
import {
  ExecutionChainSummary,
  ExecutionChainResultsList,
} from 'src/components/agentic/execution-chain-results';

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
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const pageRows = useMemo(() => {
    const start = page * rowsPerPage;
    return rows.slice(start, start + rowsPerPage);
  }, [rows, page, rowsPerPage]);

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
          <Box>
            <Typography variant="h4">Execution reports</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              On-chain whitelist checks and per-action apply outcomes by attack type.
            </Typography>
          </Box>
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
                  <TableCell sx={{ fontWeight: 700 }}>Attack</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Status</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Integrity</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Chain actions</TableCell>
                  <TableCell sx={{ fontWeight: 700 }}>Error</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 700 }}>
                    Actions
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {loading ? (
                  <TableRow>
                    <TableCell colSpan={8} align="center" sx={{ py: 6 }}>
                      <CircularProgress size={28} />
                    </TableCell>
                  </TableRow>
                ) : rows.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8}>
                      <Typography variant="body2" color="text.secondary" sx={{ py: 2 }}>
                        No execution reports yet.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  pageRows.map((r) => (
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
                        {r.attack_type ? (
                          <Chip size="small" variant="outlined" label={r.attack_type} sx={{ height: 22 }} />
                        ) : (
                          '—'
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip
                          size="small"
                          variant="outlined"
                          color={r.status === 'applied' ? 'success' : 'error'}
                          label={r.status === 'applied' ? 'Applied' : 'Not applied'}
                        />
                      </TableCell>
                      <TableCell>
                        <Chip size="small" variant="outlined" label={r.integrity_overall} />
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {r.chain_actions_applied ?? 0}/{r.chain_actions_total ?? 0} applied
                          {(r.chain_actions_whitelisted ?? 0) > 0 ? ` · ${r.chain_actions_whitelisted} whitelisted` : ''}
                        </Typography>
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
          {!loading && rows.length > 0 && (
            <TablePagination
              component="div"
              count={rows.length}
              page={page}
              onPageChange={(_, p) => setPage(p)}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={(e) => {
                setRowsPerPage(parseInt(e.target.value, 10));
                setPage(0);
              }}
              rowsPerPageOptions={[5, 10, 25, 50]}
            />
          )}
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
              <ExecutionChainSummary exec={detail} />

              {detail.error_reason && (
                <Alert severity="error">
                  <strong>{detail.error_reason}</strong>
                  {detail.error_detail ? ` — ${detail.error_detail}` : ''}
                </Alert>
              )}

              <Divider />
              <Box>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
                  Apply results
                </Typography>
                <ExecutionChainResultsList exec={detail} />
              </Box>
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
