import type {
  AgenticReportOut,
  ModelVersionOut,
  PredictionJobOut,
  RunEventOut,
  RunSummaryOut,
} from 'src/api/types';

import { useCallback, useEffect, useState } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import CircularProgress from '@mui/material/CircularProgress';

import { varAlpha } from 'minimal-shared/utils';

import { api, ApiError } from 'src/services';

// ----------------------------------------------------------------------

function JsonBlock({ data }: { data: unknown }) {
  if (data === undefined || data === null) {
    return (
      <Typography variant="body2" sx={{ color: 'text.secondary' }}>
        —
      </Typography>
    );
  }
  return (
    <Box
      component="pre"
      sx={{
        p: 1.5,
        m: 0,
        borderRadius: 1,
        fontSize: 12,
        lineHeight: 1.5,
        overflow: 'auto',
        maxHeight: 360,
        border: (theme) => `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.24)}`,
        bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
      }}
    >
      {JSON.stringify(data, null, 2)}
    </Box>
  );
}

function formatField(v: unknown): string {
  if (v === undefined || v === null || v === '') return '—';
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return JSON.stringify(v);
}

// ----------------------------------------------------------------------

export type RunDetailDialogProps = {
  open: boolean;
  runId: string | null;
  onClose: () => void;
};

export function RunDetailDialog({ open, runId, onClose }: RunDetailDialogProps) {
  const [tab, setTab] = useState<'summary' | 'events' | 'extra'>('summary');
  const [summary, setSummary] = useState<RunSummaryOut | null>(null);
  const [events, setEvents] = useState<RunEventOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    if (!runId) return;
    setLoading(true);
    setError('');
    setSummary(null);
    setEvents([]);
    try {
      const [s, ev] = await Promise.all([api.getRun(runId), api.getRunEvents(runId, 500)]);
      setSummary(s);
      setEvents(ev);
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [runId]);

  useEffect(() => {
    if (!open || !runId) return;
    void load();
  }, [open, runId, load]);

  useEffect(() => {
    if (open) setTab('summary');
  }, [open, runId]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
      <DialogTitle sx={{ pr: 6 }}>
        <Typography variant="h6" component="span">
          Run details
        </Typography>
        {runId && (
          <Typography variant="caption" display="block" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
            {runId}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers>
        {loading && (
          <Stack alignItems="center" py={4}>
            <CircularProgress size={32} />
          </Stack>
        )}
        {!loading && error && <Alert severity="error">{error}</Alert>}
        {!loading && !error && summary && (
          <>
            <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mb: 2 }}>
              <Tab value="summary" label="Summary" />
              <Tab value="events" label={`Timeline (${events.length})`} />
              <Tab value="extra" label="Payloads" />
            </Tabs>

            {tab === 'summary' && (
              <Stack spacing={1.25} divider={<Divider flexItem />}>
                <DetailRow label="Status" value={summary.status} />
                <DetailRow label="Trace ID" value={summary.trace_id} mono />
                <DetailRow label="Customer" value={summary.customer_id} mono />
                <DetailRow label="Channel" value={summary.channel} />
                <DetailRow label="Created" value={new Date(summary.created_at).toLocaleString()} />
                <DetailRow label="Updated" value={new Date(summary.updated_at).toLocaleString()} />
                <DetailRow
                  label="Completed"
                  value={summary.completed_at ? new Date(summary.completed_at).toLocaleString() : null}
                />
                <DetailRow label="Duration" value={typeof summary.duration_ms === 'number' ? `${summary.duration_ms} ms` : null} />
                <DetailRow label="Last step" value={summary.last_step} mono />
                <DetailRow label="Message" value={summary.message_preview} />
                {summary.error && (
                  <Alert severity="error">
                    {summary.error.step_name && (
                      <Typography variant="caption" display="block">
                        Step: {summary.error.step_name}
                      </Typography>
                    )}
                    {summary.error.message}
                  </Alert>
                )}
              </Stack>
            )}

            {tab === 'events' && (
              <Box sx={{ overflowX: 'auto' }}>
                <Table size="small" sx={{ minWidth: 520 }}>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Level</TableCell>
                      <TableCell>Step</TableCell>
                      <TableCell>Message</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {events.map((e) => (
                      <TableRow key={e.event_id}>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{new Date(e.timestamp).toLocaleString()}</TableCell>
                        <TableCell>{e.level}</TableCell>
                        <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{e.step_name}</TableCell>
                        <TableCell sx={{ maxWidth: 420 }}>
                          <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                            {e.message}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {!events.length && (
                  <Typography variant="body2" sx={{ color: 'text.secondary', py: 2 }}>
                    No events returned for this run.
                  </Typography>
                )}
              </Box>
            )}

            {tab === 'extra' && (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Predictions
                  </Typography>
                  <JsonBlock data={summary.predictions ?? null} />
                </Box>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    RAG
                  </Typography>
                  <JsonBlock data={summary.rag ?? null} />
                </Box>
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Actions
                  </Typography>
                  <JsonBlock data={summary.actions ?? null} />
                </Box>
              </Stack>
            )}
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => void load()} disabled={loading || !runId}>
          Reload
        </Button>
        <Button variant="contained" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

function DetailRow({
  label,
  value,
  mono,
}: {
  label: string;
  value: unknown;
  mono?: boolean;
}) {
  return (
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={0.5}>
      <Typography variant="caption" sx={{ color: 'text.secondary', width: { sm: 140 }, flexShrink: 0 }}>
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontFamily: mono ? 'monospace' : 'inherit', wordBreak: 'break-word' }}>
        {formatField(value)}
      </Typography>
    </Stack>
  );
}

// ----------------------------------------------------------------------

export type RunEventDetailDialogProps = {
  open: boolean;
  event: RunEventOut | null;
  onClose: () => void;
};

export function RunEventDetailDialog({ open, event, onClose }: RunEventDetailDialogProps) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth scroll="paper">
      <DialogTitle>Event details</DialogTitle>
      <DialogContent dividers>
        {!event ? (
          <Typography variant="body2" color="text.secondary">
            No event selected.
          </Typography>
        ) : (
          <Stack spacing={1.25} divider={<Divider flexItem />}>
            <DetailRow label="Event ID" value={event.event_id} mono />
            <DetailRow label="Run ID" value={event.run_id} mono />
            <DetailRow label="Trace ID" value={event.trace_id} mono />
            <DetailRow label="Time" value={new Date(event.timestamp).toLocaleString()} />
            <DetailRow label="Level" value={event.level} />
            <DetailRow label="Step" value={event.step_name} mono />
            <DetailRow
              label="Duration"
              value={typeof event.duration_ms === 'number' ? `${event.duration_ms} ms` : null}
            />
            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                Message
              </Typography>
              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                {event.message || '—'}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                Payload
              </Typography>
              <JsonBlock data={event.payload ?? null} />
            </Box>
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button variant="contained" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ----------------------------------------------------------------------

export type AgentReportDetailDialogProps = {
  open: boolean;
  publicId: string | null;
  onClose: () => void;
};

function pickPredictionResultsRow(
  job: PredictionJobOut | null,
  rowIndex: number | null | undefined
): unknown {
  const rj = job?.results_json;
  if (!rj || typeof rj !== 'object' || !('rows' in rj)) return null;
  const rows = (rj as { rows?: unknown }).rows;
  if (!Array.isArray(rows) || !rows.length) return null;
  if (typeof rowIndex === 'number' && rowIndex >= 0) {
    const byField = rows.find(
      (x) =>
        x &&
        typeof x === 'object' &&
        'row_index' in x &&
        Number((x as { row_index: unknown }).row_index) === rowIndex
    );
    if (byField) return byField;
    return rows[rowIndex] ?? null;
  }
  return rows[0] ?? null;
}

export function AgentReportDetailDialog({ open, publicId, onClose }: AgentReportDetailDialogProps) {
  const [tab, setTab] = useState<
    'summary' | 'data' | 'prediction' | 'rag' | 'llm' | 'actions' | 'artifact'
  >('summary');
  const [report, setReport] = useState<AgenticReportOut | null>(null);
  const [predJob, setPredJob] = useState<PredictionJobOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    if (!publicId) return;
    setLoading(true);
    setError('');
    setReport(null);
    setPredJob(null);
    try {
      const r = await api.getAgentReport(publicId);
      setReport(r);
      const pid = r.prediction_job_public_id?.trim();
      if (pid) {
        try {
          setPredJob(await api.getPredictionJob(pid));
        } catch {
          setPredJob(null);
        }
      }
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [publicId]);

  useEffect(() => {
    if (!open || !publicId) return;
    void load();
  }, [open, publicId, load]);

  useEffect(() => {
    if (open) setTab('summary');
  }, [open, publicId]);

  const artifact = report?.report_artifact ?? null;
  const sampleData = artifact && typeof artifact.sample_data !== 'undefined' ? artifact.sample_data : null;
  const userPrompt =
    artifact && typeof artifact.user_prompt === 'string' ? artifact.user_prompt : null;
  const structuredPlan =
    artifact && typeof artifact.structured_plan !== 'undefined' ? artifact.structured_plan : null;
  const trustChainBlock =
    artifact && typeof artifact.trust_chain === 'object' && artifact.trust_chain !== null
      ? artifact.trust_chain
      : null;

  const resultsRow = pickPredictionResultsRow(predJob, report?.results_row_index);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth scroll="paper">
      <DialogTitle sx={{ pr: 6 }}>
        <Typography variant="h6" component="span">
          Agentic report — full detail
        </Typography>
        {publicId && (
          <Typography variant="caption" display="block" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
            {publicId}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers>
        {loading && (
          <Stack alignItems="center" py={4}>
            <CircularProgress size={32} />
          </Stack>
        )}
        {!loading && error && <Alert severity="error">{error}</Alert>}
        {!loading && !error && report && (
          <>
            <Tabs
              value={tab}
              onChange={(_, v) => setTab(v)}
              variant="scrollable"
              scrollButtons="auto"
              sx={{ mb: 2, borderBottom: 1, borderColor: 'divider' }}
            >
              <Tab value="summary" label="Summary" />
              <Tab value="data" label="Data row" />
              <Tab value="prediction" label="Prediction" />
              <Tab value="rag" label="RAG" />
              <Tab value="llm" label="LLM" />
              <Tab value="actions" label="Actions" />
              <Tab value="artifact" label="Raw JSON" />
            </Tabs>

            {tab === 'summary' && (
              <Stack spacing={1.25} divider={<Divider flexItem />}>
                <DetailRow label="Created" value={new Date(report.created_at).toLocaleString()} />
                <DetailRow label="Prediction job" value={report.prediction_job_public_id} mono />
                <DetailRow label="Agentic job" value={report.agentic_job_public_id} mono />
                <DetailRow label="Results row index" value={report.results_row_index ?? '—'} />
                <DetailRow
                  label="Trust (API)"
                  value={report.trust_commitment ?? report.trust_chain_mode ?? '—'}
                />
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                    Executive summary
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                    {report.summary}
                  </Typography>
                </Box>
                <DetailRow label="Recommended action" value={report.recommended_action} mono />
                <DetailRow label="Report file" value={report.report_path} mono />
                {trustChainBlock && (
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                      Trust chain (artifact)
                    </Typography>
                    <JsonBlock data={trustChainBlock} />
                  </Box>
                )}
              </Stack>
            )}

            {tab === 'data' && (
              <Stack spacing={1}>
                <Typography variant="body2" color="text.secondary">
                  <strong>sample_data</strong> passed into the agent (prediction row, SHAP / party context, etc.).
                  Older reports may only have fields on the Prediction tab.
                </Typography>
                <JsonBlock data={sampleData ?? null} />
              </Stack>
            )}

            {tab === 'prediction' && (
              <Stack spacing={1.25} divider={<Divider flexItem />}>
                {!predJob && (
                  <Alert severity="warning">
                    Could not load prediction job {report.prediction_job_public_id ?? '—'}.
                  </Alert>
                )}
                {predJob && (
                  <>
                    <DetailRow label="Job status" value={predJob.status} />
                    <DetailRow label="Rows total" value={predJob.rows_total ?? '—'} />
                    <DetailRow label="Rows flagged" value={predJob.rows_flagged ?? '—'} />
                    <DetailRow label="Model kind" value={predJob.results_model_kind ?? '—'} />
                    <DetailRow label="Output path" value={predJob.output_path} mono />
                    <Box>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                        Results row used for this report
                      </Typography>
                      <JsonBlock data={resultsRow ?? null} />
                    </Box>
                    <Box>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                        Full results_json (job)
                      </Typography>
                      <JsonBlock data={predJob.results_json ?? null} />
                    </Box>
                  </>
                )}
              </Stack>
            )}

            {tab === 'rag' && (
              <Stack spacing={1.25}>
                <Typography variant="body2" color="text.secondary">
                  Text retrieved / formatted for the model (same snapshot as stored after decide).
                </Typography>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                    rag_context_used (DB / API)
                  </Typography>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.25,
                      borderRadius: 1,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
                      border: (theme) => `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.2)}`,
                      maxHeight: 420,
                      overflow: 'auto',
                      fontSize: 12,
                      fontFamily: 'inherit',
                    }}
                  >
                    {report.rag_context_used ?? '—'}
                  </Typography>
                </Box>
                {artifact &&
                  typeof artifact.rag_context_used === 'string' &&
                  artifact.rag_context_used !== report.rag_context_used && (
                    <Box>
                      <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                        rag_context_used (artifact file)
                      </Typography>
                      <Typography
                        variant="body2"
                        component="pre"
                        sx={{
                          m: 0,
                          p: 1.25,
                          borderRadius: 1,
                          whiteSpace: 'pre-wrap',
                          wordBreak: 'break-word',
                          bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
                          border: (theme) => `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.2)}`,
                          maxHeight: 420,
                          overflow: 'auto',
                          fontSize: 12,
                          fontFamily: 'inherit',
                        }}
                      >
                        {artifact.rag_context_used}
                      </Typography>
                    </Box>
                  )}
              </Stack>
            )}

            {tab === 'llm' && (
              <Stack spacing={1.25}>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                    User prompt (orchestration — sent to the LLM)
                  </Typography>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.25,
                      borderRadius: 1,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
                      border: (theme) => `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.2)}`,
                      maxHeight: 420,
                      overflow: 'auto',
                      fontSize: 12,
                      fontFamily: 'inherit',
                    }}
                  >
                    {userPrompt ??
                      '— (not stored in this report file — run a new agent decide to capture it)'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                    Raw LLM response
                  </Typography>
                  <Typography
                    variant="body2"
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.25,
                      borderRadius: 1,
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
                      border: (theme) => `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.2)}`,
                      maxHeight: 420,
                      overflow: 'auto',
                      fontSize: 12,
                      fontFamily: 'inherit',
                    }}
                  >
                    {report.raw_llm_response ?? '—'}
                  </Typography>
                </Box>
                {structuredPlan != null && (
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                      Structured plan (parsed JSON from response, if any)
                    </Typography>
                    <JsonBlock data={structuredPlan} />
                  </Box>
                )}
              </Stack>
            )}

            {tab === 'actions' && (
              <Stack spacing={1.25}>
                <Typography variant="body2" color="text.secondary">
                  Primary / supporting actions from the structured orchestration output (subset), then full plan if
                  needed.
                </Typography>
                {!structuredPlan && (
                  <Alert severity="info">
                    No <code>structured_plan</code> on this report — check the LLM tab for a raw JSON body, or run a
                    newer agent decide.
                  </Alert>
                )}
                <JsonBlock
                  data={(() => {
                    if (!structuredPlan || typeof structuredPlan !== 'object' || structuredPlan === null) {
                      return null;
                    }
                    const p = structuredPlan as Record<string, unknown>;
                    return {
                      primary_actions: p.primary_actions,
                      supporting_actions: p.supporting_actions,
                      all_actions: p.all_actions,
                      execution_priority: p.execution_priority,
                      threat_level: p.threat_level,
                      overall_reasoning: p.overall_reasoning,
                    };
                  })()}
                />
                {structuredPlan && (
                  <Box>
                    <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                      Full structured_plan
                    </Typography>
                    <JsonBlock data={structuredPlan} />
                  </Box>
                )}
              </Stack>
            )}

            {tab === 'artifact' && (
              <Stack spacing={1}>
                <Typography variant="caption" color="text.secondary">
                  Complete on-disk report payload when available from the API.
                </Typography>
                <JsonBlock data={artifact ?? { message: 'No report_artifact on this response' }} />
              </Stack>
            )}
          </>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => void load()} disabled={loading || !publicId}>
          Reload
        </Button>
        <Button variant="contained" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}

// ----------------------------------------------------------------------

export type ModelVersionDetailDialogProps = {
  open: boolean;
  model: ModelVersionOut | null;
  onClose: () => void;
};

/** Registry row detail (no per-model GET API; uses list payload). */
export function ModelVersionDetailDialog({ open, model, onClose }: ModelVersionDetailDialogProps) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth scroll="paper">
      <DialogTitle sx={{ pr: 6 }}>
        <Typography variant="h6" component="span">
          Registered model
        </Typography>
        {model?.public_id && (
          <Typography variant="caption" display="block" sx={{ fontFamily: 'monospace', color: 'text.secondary' }}>
            {model.public_id}
          </Typography>
        )}
      </DialogTitle>
      <DialogContent dividers>
        {!model ? (
          <Typography variant="body2" color="text.secondary">
            No model selected.
          </Typography>
        ) : (
          <Stack spacing={1.25} divider={<Divider flexItem />}>
            <DetailRow label="Internal id" value={model.id} />
            <DetailRow label="Version" value={model.version_number} />
            <DetailRow label="Algorithm" value={model.algorithm} mono />
            <DetailRow label="Training job id" value={model.training_job_id ?? '—'} />
            <DetailRow label="Artifact path" value={model.artifact_path} mono />
            <DetailRow label="Created" value={new Date(model.created_at).toLocaleString()} />
            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                Metrics
              </Typography>
              <JsonBlock data={model.metrics_json} />
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                Feature columns
              </Typography>
              <JsonBlock data={model.feature_columns_json} />
            </Box>
            <Box>
              <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 0.5 }}>
                Label classes
              </Typography>
              <JsonBlock data={model.label_classes_json} />
            </Box>
          </Stack>
        )}
      </DialogContent>
      <DialogActions>
        <Button variant="contained" onClick={onClose}>
          Close
        </Button>
      </DialogActions>
    </Dialog>
  );
}
