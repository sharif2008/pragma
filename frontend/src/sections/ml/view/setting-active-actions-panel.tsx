import type {
  AgenticReportOut,
  TrustAnchorVerifyOut,
  ExecutionReportDetailOut,
} from 'src/api/types';

import { useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime } from 'src/utils/table-time-sort';
import {
  type PlanAction,
  actionsFromReport,
} from 'src/utils/agentic-plan-actions';
import {
  applyFailureDetail,
  appliedItemsFromExec,
  attackTypeFromReport,
} from 'src/utils/execution-chain-results';

import {
  ApiError,
  getAgentReport,
  verifyTrustAnchor,
  getExecutionReport,
  applyAgenticReport,
  listAllAgentReports,
} from 'src/services';

import {
  DetectionActionsList,
} from 'src/components/agentic/detection-actions-list';
import {
  ExecutionChainResultsList,
} from 'src/components/agentic/execution-chain-results';
import { GlobalActionLoadingOverlay } from 'src/components/loading/global-action-loading-overlay';

// ----------------------------------------------------------------------

type Notify = (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => void;

type RowResult = {
  verify?: TrustAnchorVerifyOut | null;
  exec?: ExecutionReportDetailOut | null;
  error?: string | null;
  lastAction?: 'validate' | 'apply' | 'result';
};

type BusyAction = 'validate' | 'apply' | 'result';

type AppliedActionItem = {
  index?: number;
  action: string;
  network_tier?: string;
  reasoning?: string;
  result?: string;
  failure_reason?: string;
  bucket?: string;
  whitelisted?: boolean;
  apply_tx_hash?: string;
};

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

function trustAnchorId(report: AgenticReportOut): number | null {
  const ta = report.trust_anchor;
  if (!ta || typeof ta !== 'object') return null;
  const id = (ta as { id?: unknown }).id;
  return typeof id === 'number' && Number.isFinite(id) ? id : null;
}

function execStatus(report: AgenticReportOut): 'applied' | 'failed' | 'none' {
  const ex = report.execution_report;
  if (!ex || typeof ex !== 'object') return 'none';
  const status = String((ex as { status?: unknown }).status ?? '');
  if (status === 'applied') return 'applied';
  if (status === 'failed') return 'failed';
  return 'none';
}

function attackTypeForRow(report: AgenticReportOut, result?: RowResult): string | null {
  return (
    attackTypeFromReport(result?.exec) ||
    (() => {
      const art = report.report_artifact;
      if (art && typeof art === 'object' && art !== null) {
        const sample = (art as { sample_data?: unknown }).sample_data;
        if (sample && typeof sample === 'object' && sample !== null) {
          const label = (sample as { predicted_label?: unknown }).predicted_label;
          if (label != null && String(label).trim()) return String(label).trim();
        }
      }
      const ex = report.execution_report;
      if (ex && typeof ex === 'object' && (ex as { attack_type?: unknown }).attack_type) {
        return String((ex as { attack_type: unknown }).attack_type);
      }
      return null;
    })()
  );
}

function tierLabel(item: AppliedActionItem): string {
  return (item.network_tier || item.bucket || '').trim();
}

function appliedItemsForResult(
  exec: ExecutionReportDetailOut | null | undefined,
  planned: PlanAction[],
  verify?: TrustAnchorVerifyOut | null
): AppliedActionItem[] {
  const failureMsg = applyFailureDetail(exec, verify);
  let items = appliedItemsFromExec(exec).map((it) => ({
    index: it.index,
    action: it.action,
    network_tier: it.network_tier,
    result: it.result,
    failure_reason: it.failure_reason,
    whitelisted: it.whitelisted,
    apply_tx_hash: it.apply_tx_hash,
  }));
  if (!items.length && planned.length && exec?.status === 'failed') {
    items = planned.map((p, i) => ({
      index: i,
      action: p.action,
      network_tier: p.network_tier,
      result: 'failed',
      failure_reason: failureMsg,
      whitelisted: undefined,
      apply_tx_hash: undefined,
    }));
  }
  items = items.map((it, i) => ({
    ...it,
    network_tier: it.network_tier || planned[it.index ?? i]?.network_tier || undefined,
  }));
  if (exec?.status !== 'applied') {
    items = items.map((it) => ({
      ...it,
      result: 'failed',
      failure_reason: failureMsg,
    }));
  }
  return items;
}

function statusChip(report: AgenticReportOut, result?: RowResult) {
  const exec = result?.exec;
  const st = execStatus(report);
  if (st === 'applied' || exec?.status === 'applied') {
    return <Chip size="small" color="success" label="Applied" sx={{ height: 22 }} />;
  }
  if (st === 'failed' || exec?.status === 'failed') {
    return <Chip size="small" color="error" label="Not applied" sx={{ height: 22 }} />;
  }
  if (result?.verify) {
    if (result.verify.overall_integrity === 'valid') {
      return <Chip size="small" color="info" variant="outlined" label="Validated" sx={{ height: 22 }} />;
    }
    const bcLabel = result.verify.overall_integrity === 'anchor_failed' ? 'Not on BC' : 'Invalid BC';
    return <Chip size="small" color="warning" variant="outlined" label={bcLabel} sx={{ height: 22 }} />;
  }
  if (exec?.integrity_overall && exec.integrity_overall !== 'valid') {
    return <Chip size="small" color="warning" variant="outlined" label="Not on BC" sx={{ height: 22 }} />;
  }
  if (result?.error) {
    return <Chip size="small" color="error" variant="outlined" label="Error" sx={{ height: 22 }} />;
  }
  return <Chip size="small" variant="outlined" label="Pending" sx={{ height: 22 }} />;
}

function ResultDialog({
  open,
  onClose,
  report,
  result,
}: {
  open: boolean;
  onClose: () => void;
  report: AgenticReportOut | null;
  result: RowResult | null;
}) {
  const verify = result?.verify;
  const exec = result?.exec;
  const verifyOk = verify?.overall_integrity === 'valid';
  const planned = report ? actionsFromReport(report) : [];
  const appliedItems = appliedItemsForResult(exec, planned, verify);
  const applied = exec?.status === 'applied' || (report ? execStatus(report) === 'applied' : false);
  const applyFailure = !applied && exec ? applyFailureDetail(exec, verify) : null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle sx={{ py: 1.5 }}>
        Active action result
        {report ? (
          <Typography variant="caption" color="text.secondary" display="block">
            Plan {report.public_id.slice(0, 8)}…
          </Typography>
        ) : null}
      </DialogTitle>
      <DialogContent dividers>
        <Stack spacing={1.5}>
          {applied && exec && (
            <Alert severity="success">
              All {appliedItems.length} action(s) applied after blockchain validation
              {exec.applied_at ? ` · ${fDateTime(exec.applied_at)}` : ''}.
            </Alert>
          )}

          {!applied && exec && (
            <Alert severity="error">
              Apply blocked — {exec.integrity_overall || verify?.overall_integrity || 'failed'}
              {applyFailure ? (
                <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap' }}>
                  {applyFailure}
                </Typography>
              ) : null}
            </Alert>
          )}

          {!exec && verify && (
            <Alert severity={verifyOk ? 'success' : 'error'}>
              Blockchain validation: <strong>{verify.overall_integrity}</strong>
              {verify.tx_hash ? ` · tx ${verify.tx_hash.slice(0, 12)}…` : ''}
              {!verifyOk && (
                <Typography variant="body2" sx={{ mt: 0.5, whiteSpace: 'pre-wrap' }}>
                  {verify.chain_integrity_detail || verify.payload_integrity_detail || 'Integrity check did not pass.'}
                </Typography>
              )}
            </Alert>
          )}

          {!verify && !exec && result?.error && (
            <Alert severity="error">
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                {result.error}
              </Typography>
            </Alert>
          )}

          {!verify && !exec && !result?.error && (
            <Alert severity="info">Run Validate or Apply first to see blockchain / apply results here.</Alert>
          )}

          {exec && (
            <>
              <Divider />
              <Box>
                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
                  Apply results ({appliedItems.length})
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  {applied
                    ? 'Per-action outcomes by network tier.'
                    : 'Each action failed — see the error under each action.'}
                </Typography>
                <ExecutionChainResultsList exec={exec} execApplied={applied} />
              </Box>
            </>
          )}

          {report && (
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.75 }}>
                Detection actions ({planned.length})
              </Typography>
              <DetectionActionsList actions={planned} />
              {report.summary ? (
                <>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1.25 }}>
                    Summary
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {report.summary}
                  </Typography>
                </>
              ) : null}
            </Box>
          )}
        </Stack>
      </DialogContent>
      <DialogActions sx={{ py: 1 }}>
        <Button size="small" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

/** Setting → Agentic execution: validate BC integrity, apply plans, inspect results. */
export function SettingActiveActionsPanel({ onNotify }: { onNotify: Notify }) {
  const [reports, setReports] = useState<AgenticReportOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [busyPlanId, setBusyPlanId] = useState('');
  const [busyAction, setBusyAction] = useState<BusyAction | null>(null);
  const [loadingOverlayDismissed, setLoadingOverlayDismissed] = useState(false);
  const [results, setResults] = useState<Record<string, RowResult>>({});
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [resultOpen, setResultOpen] = useState(false);
  const [resultReport, setResultReport] = useState<AgenticReportOut | null>(null);

  const sorted = useMemo(() => sortByTime(reports, (r) => r.created_at, 'desc'), [reports]);
  const pageRows = useMemo(() => {
    const start = page * rowsPerPage;
    return sorted.slice(start, start + rowsPerPage);
  }, [sorted, page, rowsPerPage]);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await listAllAgentReports();
      setReports(rows);
      // Hydrate kept apply results for plans that already have an execution report.
      const withExec = rows.filter((r) => {
        const ex = r.execution_report;
        return !!ex && typeof ex === 'object' && typeof (ex as { id?: unknown }).id === 'number';
      });
      if (withExec.length) {
        const patches = await Promise.all(
          withExec.map(async (r) => {
            const execId = (r.execution_report as { id: number }).id;
            try {
              const exec = await getExecutionReport(execId);
              return { id: r.public_id, exec };
            } catch {
              return null;
            }
          })
        );
        setResults((prev) => {
          const next = { ...prev };
          for (const p of patches) {
            if (!p) continue;
            next[p.id] = { ...next[p.id], exec: p.exec };
          }
          return next;
        });
      }
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (busyPlanId) setLoadingOverlayDismissed(false);
  }, [busyPlanId]);

  const patchResult = (id: string, patch: Partial<RowResult>) => {
    setResults((prev) => ({ ...prev, [id]: { ...prev[id], ...patch } }));
  };

  const loadFull = async (publicId: string) => {
    const full = await getAgentReport(publicId);
    setReports((prev) => prev.map((r) => (r.public_id === publicId ? { ...r, ...full } : r)));
    return full;
  };

  const onValidate = async (row: AgenticReportOut) => {
    const id = row.public_id;
    setBusyPlanId(id);
    setBusyAction('validate');
    onNotify(null);
    try {
      const full = await loadFull(id);
      const taId = trustAnchorId(full);
      if (taId == null) {
        const msg = 'No trust anchor on this plan — re-save with Chain enabled in Agentic planner.';
        patchResult(id, { error: msg, verify: null, lastAction: 'validate' });
        onNotify({ severity: 'error', text: msg });
        return;
      }
      const v = await verifyTrustAnchor(taId);
      patchResult(id, { verify: v, error: null, lastAction: 'validate' });
      const ok = v.overall_integrity === 'valid';
      onNotify({
        severity: ok ? 'success' : 'error',
        text: ok
          ? `Blockchain valid for ${id.slice(0, 8)}… — ready to Apply`
          : `Blockchain validation failed (${v.overall_integrity}) for ${id.slice(0, 8)}…`,
      });
    } catch (e) {
      const msg = formatError(e);
      patchResult(id, { error: msg, lastAction: 'validate' });
      onNotify({ severity: 'error', text: msg });
    } finally {
      setBusyPlanId('');
      setBusyAction(null);
    }
  };

  const onApply = async (row: AgenticReportOut) => {
    const id = row.public_id;
    const actions = actionsFromReport(row);
    if (execStatus(row) === 'applied' || results[id]?.exec?.status === 'applied') {
      onNotify({ severity: 'info', text: 'Already applied.' });
      return;
    }
    setBusyPlanId(id);
    setBusyAction('apply');
    onNotify(null);
    let verifyState = results[id]?.verify ?? null;
    try {
      try {
        const full = await loadFull(id);
        const taId = trustAnchorId(full);
        if (taId != null) {
          const v = await verifyTrustAnchor(taId);
          verifyState = v;
          patchResult(id, { verify: v });
        }
      } catch {
        /* apply still records failure server-side */
      }

      const exec = await applyAgenticReport(id);
      patchResult(id, { exec, lastAction: 'apply', error: null, verify: verifyState });
      const full = await loadFull(id);

      const appliedItems = appliedItemsForResult(exec, actions, verifyState);
      if (exec.status === 'applied') {
        const tiers = [...new Set(appliedItems.map(tierLabel).filter(Boolean))];
        const tierNote = tiers.length ? ` · ${tiers.join(', ')}` : '';
        onNotify({
          severity: 'success',
          text: `Applied ${appliedItems.length || actions.length} action(s) for ${id.slice(0, 8)}… after blockchain validation${tierNote}.`,
        });
      } else {
        const failure = applyFailureDetail(exec, verifyState);
        onNotify({
          severity: 'error',
          text: `${appliedItems.length || actions.length} action(s) not applied — ${exec.integrity_overall}: ${failure}`,
        });
      }
      setResultReport(full);
      setResultOpen(true);
    } catch (e) {
      const msg = formatError(e);
      patchResult(id, { error: msg, lastAction: 'apply' });
      onNotify({ severity: 'error', text: msg });
    } finally {
      setBusyPlanId('');
      setBusyAction(null);
    }
  };

  const onResult = async (row: AgenticReportOut) => {
    const id = row.public_id;
    setBusyPlanId(id);
    setBusyAction('result');
    try {
      const full = await loadFull(id);
      const existing = results[id] || {};
      let exec = existing.exec || null;
      const ex = full.execution_report;
      if (ex && typeof ex === 'object' && typeof (ex as { id?: unknown }).id === 'number') {
        try {
          exec = await getExecutionReport((ex as { id: number }).id);
        } catch {
          /* keep list summary */
        }
      }
      let verify = existing.verify || null;
      const taId = trustAnchorId(full);
      if (taId != null && !verify) {
        try {
          verify = await verifyTrustAnchor(taId);
        } catch {
          /* optional */
        }
      }
      patchResult(id, { exec, verify, lastAction: 'result' });
      setResultReport(full);
      setResultOpen(true);
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setBusyPlanId('');
      setBusyAction(null);
    }
  };

  const actionLoadingMessage = useMemo(() => {
    if (busyAction === 'validate') return 'Validating blockchain…';
    if (busyAction === 'apply') return 'Applying all detection actions…';
    if (busyAction === 'result') return 'Loading apply results…';
    return 'Processing…';
  }, [busyAction]);

  const actionLoadingSubmessage = busyPlanId ? `Plan ${busyPlanId.slice(0, 8)}…` : undefined;

  return (
    <Stack spacing={1.5}>
      <GlobalActionLoadingOverlay
        open={!!busyPlanId && !loadingOverlayDismissed}
        onClose={() => setLoadingOverlayDismissed(true)}
        message={actionLoadingMessage}
        submessage={actionLoadingSubmessage}
      />
      <Paper variant="outlined" sx={{ p: 1.5, borderRadius: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              Agentic execution
            </Typography>
            <Typography variant="caption" color="text.secondary" display="block">
              Validate blockchain, then Apply once to run all detection actions. If the plan is not on-chain,
              status shows Not applied with an error reason.
            </Typography>
          </Box>
          <Stack direction="row" spacing={1} alignItems="center">
            <Button size="small" onClick={() => void refresh()} disabled={loading || !!busyPlanId}>
              Refresh
            </Button>
            {loading && <CircularProgress size={18} />}
          </Stack>
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 1.25, borderRadius: 2 }}>
        <TableContainer sx={{ maxHeight: 440 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell sx={{ py: 0.75 }}>Plan</TableCell>
                <TableCell sx={{ py: 0.75 }}>Attack</TableCell>
                <TableCell sx={{ py: 0.75 }}>Detection actions</TableCell>
                <TableCell sx={{ py: 0.75 }}>Status</TableCell>
                <TableCell sx={{ py: 0.75 }}>Created</TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Validate
                </TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Apply
                </TableCell>
                <TableCell align="right" sx={{ py: 0.75 }}>
                  Result
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {pageRows.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} sx={{ py: 2 }}>
                    <Typography variant="caption" color="text.secondary">
                      No active plans yet — save plans under Agentic planner (with Chain on for BC apply).
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {pageRows.map((r) => {
                const planBusy = busyPlanId === r.public_id;
                const actions = actionsFromReport(r);
                const rowResult = results[r.public_id];
                const applied = execStatus(r) === 'applied' || rowResult?.exec?.status === 'applied';
                return (
                  <TableRow key={r.public_id} hover>
                    <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, py: 0.5, verticalAlign: 'top' }}>
                      {r.public_id.slice(0, 8)}…
                    </TableCell>
                    <TableCell sx={{ py: 0.5, verticalAlign: 'top' }}>
                      {attackTypeForRow(r, rowResult) ? (
                        <Chip size="small" variant="outlined" label={attackTypeForRow(r, rowResult)} sx={{ height: 20 }} />
                      ) : (
                        '—'
                      )}
                    </TableCell>
                    <TableCell sx={{ py: 0.5, maxWidth: 420, verticalAlign: 'top' }}>
                      <DetectionActionsList actions={actions} />
                    </TableCell>
                    <TableCell sx={{ py: 0.5, verticalAlign: 'top' }}>{statusChip(r, rowResult)}</TableCell>
                    <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap', py: 0.5, verticalAlign: 'top' }}>
                      {fDateTime(r.created_at)}
                    </TableCell>
                    <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                      <Button
                        size="small"
                        sx={{ minWidth: 0, px: 0.75 }}
                        disabled={planBusy}
                        onClick={() => void onValidate(r)}
                      >
                        {planBusy && busyAction === 'validate' ? '…' : 'Validate'}
                      </Button>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                      <Button
                        size="small"
                        color="success"
                        variant={applied ? 'outlined' : 'contained'}
                        sx={{ minWidth: 0, px: 0.75 }}
                        disabled={planBusy || applied}
                        onClick={() => void onApply(r)}
                      >
                        {applied ? 'Applied' : planBusy && busyAction === 'apply' ? '…' : 'Apply'}
                      </Button>
                    </TableCell>
                    <TableCell align="right" sx={{ py: 0.5, verticalAlign: 'top' }}>
                      <Button
                        size="small"
                        sx={{ minWidth: 0, px: 0.75 }}
                        disabled={planBusy}
                        onClick={() => void onResult(r)}
                      >
                        {planBusy && busyAction === 'result' ? '…' : 'Result'}
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
          count={sorted.length}
          page={page}
          onPageChange={(_, p) => setPage(p)}
          rowsPerPage={rowsPerPage}
          onRowsPerPageChange={(e) => {
            setRowsPerPage(parseInt(e.target.value, 10));
            setPage(0);
          }}
          rowsPerPageOptions={[5, 10, 25]}
          sx={{ minHeight: 40, '& .MuiTablePagination-toolbar': { minHeight: 40, pl: 1 } }}
        />
      </Paper>

      <ResultDialog
        open={resultOpen}
        onClose={() => {
          setResultOpen(false);
          setResultReport(null);
        }}
        report={resultReport}
        result={resultReport ? results[resultReport.public_id] || null : null}
      />
    </Stack>
  );
}
