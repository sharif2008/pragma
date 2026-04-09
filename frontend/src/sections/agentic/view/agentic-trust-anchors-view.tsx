import type { TrustAnchorVerifyOut, TrustAnchorListItemOut } from 'src/api/types';

import { useNavigate } from 'react-router-dom';
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
import CardHeader from '@mui/material/CardHeader';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TableContainer from '@mui/material/TableContainer';
import CircularProgress from '@mui/material/CircularProgress';

import { RouterLink } from 'src/routes/components';

import { DashboardContent } from 'src/layouts/dashboard';
import { ApiError, listTrustAnchors, verifyTrustAnchor } from 'src/services';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

function shortHex(hex: string, head = 10, tail = 6): string {
  const h = (hex || '').replace(/^0x/i, '');
  if (h.length <= head + tail + 2) return hex || '—';
  return `0x${h.slice(0, head)}…${h.slice(-tail)}`;
}

function integrityChip(overall: TrustAnchorVerifyOut['overall_integrity']) {
  const map = {
    valid: { color: 'success' as const, label: 'Valid' },
    invalid: { color: 'error' as const, label: 'Invalid' },
    unknown: { color: 'warning' as const, label: 'Unknown' },
    anchor_failed: { color: 'default' as const, label: 'Anchor failed' },
  };
  const m = map[overall];
  return <Chip size="small" color={m.color} label={m.label} sx={{ fontWeight: 600 }} />;
}

function triBool(v: boolean | null | undefined): string {
  if (v === true) return 'Yes';
  if (v === false) return 'No';
  return '—';
}

// ----------------------------------------------------------------------

export function AgenticTrustAnchorsView() {
  const navigate = useNavigate();
  const [rows, setRows] = useState<TrustAnchorListItemOut[]>([]);
  const [loading, setLoading] = useState(true);
  const [listError, setListError] = useState<string | null>(null);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);
  const [verify, setVerify] = useState<TrustAnchorVerifyOut | null>(null);
  const [activeAnchorId, setActiveAnchorId] = useState<number | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setListError(null);
    try {
      const data = await listTrustAnchors(500, 0);
      setRows(data);
    } catch (e) {
      setListError(e instanceof ApiError ? e.message : 'Failed to load trust anchors');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const openVerify = async (anchorId: number) => {
    setActiveAnchorId(anchorId);
    setDialogOpen(true);
    setVerify(null);
    setVerifyError(null);
    setVerifyLoading(true);
    try {
      const out = await verifyTrustAnchor(anchorId);
      setVerify(out);
    } catch (e) {
      setVerifyError(e instanceof ApiError ? e.message : 'Verification request failed');
    } finally {
      setVerifyLoading(false);
    }
  };

  const closeDialog = () => {
    setDialogOpen(false);
    setVerify(null);
    setVerifyError(null);
    setActiveAnchorId(null);
  };

  return (
    <DashboardContent maxWidth="xl">
      <Stack spacing={3}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={2}>
          <Typography variant="h4">Trust anchors</Typography>
          <Button
            size="small"
            variant="outlined"
            startIcon={<Iconify icon="solar:restart-bold" />}
            onClick={() => load()}
            disabled={loading}
          >
            Refresh
          </Button>
        </Stack>

        <Typography variant="body2" color="text.secondary">
          Rows from <code>agentic_report_trust_anchors</code>, linked to agentic reports and prediction jobs.
          Use <strong>View</strong> to read the on-chain commitment and compare it with the database and report
          file.
        </Typography>

        {listError && <Alert severity="error">{listError}</Alert>}

        <Card>
          <CardHeader title="Anchored reports" subheader={`${rows.length} row(s)`} />
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Report</TableCell>
                  <TableCell>Prediction job</TableCell>
                  <TableCell sx={{ minWidth: 200 }}>Summary</TableCell>
                  <TableCell>Tx hash</TableCell>
                  <TableCell>Commitment</TableCell>
                  <TableCell>Anchored</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right"> </TableCell>
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
                      <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>
                        No trust anchor rows yet. Enable trust anchoring and create an agentic report with a
                        successful on-chain anchor.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  rows.map((r) => (
                    <TableRow key={r.id} hover>
                      <TableCell>
                        <Link
                          component={RouterLink}
                          href={`/agentic/report/${encodeURIComponent(r.agentic_report_public_id)}`}
                          variant="body2"
                        >
                          {shortHex(r.agentic_report_public_id.replace(/-/g, ''), 6, 4)}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 140 }}>
                          {shortHex(r.prediction_job_public_id.replace(/-/g, ''), 6, 4)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" color="text.secondary">
                          {r.summary_preview}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                          {r.tx_hash ? shortHex(r.tx_hash) : '—'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace' }}>
                          {r.commitment_sha256 ? shortHex(r.commitment_sha256) : '—'}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap>
                          {new Date(r.anchored_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {r.anchor_error ? (
                          <Chip size="small" color="error" label="Error" variant="outlined" />
                        ) : r.tx_hash ? (
                          <Chip size="small" color="success" label="Anchored" variant="outlined" />
                        ) : (
                          <Chip size="small" label="Pending / empty" variant="outlined" />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        <Button size="small" variant="contained" onClick={() => openVerify(r.id)}>
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

      <Dialog open={dialogOpen} onClose={closeDialog} maxWidth="md" fullWidth>
        <DialogTitle sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', pr: 1 }}>
          Trust anchor verification
          <IconButton aria-label="close" onClick={closeDialog}>
            <Iconify icon="mingcute:close-line" />
          </IconButton>
        </DialogTitle>
        <DialogContent dividers>
          {verifyLoading && (
            <Box display="flex" justifyContent="center" py={4}>
              <CircularProgress />
            </Box>
          )}
          {verifyError && <Alert severity="error">{verifyError}</Alert>}
          {verify && !verifyLoading && (
            <Stack spacing={2}>
              <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
                <Typography variant="subtitle2">Overall integrity:</Typography>
                {integrityChip(verify.overall_integrity)}
              </Stack>

              <Alert
                severity={
                  verify.overall_integrity === 'valid'
                    ? 'success'
                    : verify.overall_integrity === 'invalid'
                      ? 'error'
                      : verify.overall_integrity === 'anchor_failed'
                        ? 'info'
                        : 'warning'
                }
              >
                {verify.overall_integrity === 'valid' &&
                  'On-chain commitment matches the database and the report file still reproduces the same hash.'}
                {verify.overall_integrity === 'invalid' &&
                  'At least one check failed: chain vs DB, or file recomputation vs DB.'}
                {verify.overall_integrity === 'unknown' &&
                  'Could not complete all checks (RPC unreachable, missing file, etc.).'}
                {verify.overall_integrity === 'anchor_failed' &&
                  'This row recorded an anchoring error or missing transaction hash.'}
              </Alert>

              <Divider />
              <Typography variant="subtitle2">Report</Typography>
              <Typography variant="body2" color="text.secondary">
                {verify.summary_preview}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Recommended: {verify.recommended_action}
              </Typography>
              <Stack direction="row" gap={1} flexWrap="wrap">
                <Button
                  size="small"
                  variant="outlined"
                  onClick={() => navigate(`/agentic/report/${encodeURIComponent(verify.agentic_report_public_id)}`)}
                >
                  Open report
                </Button>
              </Stack>

              <Divider />
              <Typography variant="subtitle2">Blockchain</Typography>
              <Stack spacing={0.5}>
                <Typography variant="body2">
                  <strong>RPC:</strong> {verify.rpc_connected ? 'connected' : 'not connected'} —{' '}
                  <Box component="span" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                    {verify.rpc_url}
                  </Box>
                </Typography>
                <Typography variant="body2">
                  <strong>Chain ID:</strong> {verify.chain_id}
                </Typography>
                <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                  <strong>Contract:</strong> {verify.contract_address || '—'}
                </Typography>
                <Typography variant="body2" sx={{ wordBreak: 'break-all' }}>
                  <strong>Tx hash:</strong> {verify.tx_hash || '—'}
                </Typography>
              </Stack>

              <Divider />
              <Typography variant="subtitle2">Commitments</Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                DB (anchored): {verify.db_commitment_sha256 || '—'}
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                On-chain: {verify.on_chain_commitment_hex || '—'}
              </Typography>
              <Typography variant="body2">
                Chain matches DB: <strong>{triBool(verify.chain_integrity_valid)}</strong>
                {verify.chain_integrity_detail ? ` — ${verify.chain_integrity_detail}` : ''}
              </Typography>

              <Divider />
              <Typography variant="subtitle2">Report file (recomputed)</Typography>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                Recomputed SHA-256: {verify.recomputed_commitment_sha256 || '—'}
              </Typography>
              <Typography variant="body2">
                Payload matches DB: <strong>{triBool(verify.payload_integrity_valid)}</strong>
                {verify.payload_integrity_detail ? ` — ${verify.payload_integrity_detail}` : ''}
              </Typography>
            </Stack>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closeDialog}>Close</Button>
          {activeAnchorId != null && (
            <Button
              variant="outlined"
              startIcon={<Iconify icon="solar:restart-bold" />}
              disabled={verifyLoading}
              onClick={() => openVerify(activeAnchorId)}
            >
              Re-check
            </Button>
          )}
        </DialogActions>
      </Dialog>
    </DashboardContent>
  );
}
