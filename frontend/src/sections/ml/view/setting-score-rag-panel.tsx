import type { ModelVersionOut, PredictionJobOut } from 'src/api/types';
import type { PredictionJobListItem } from 'src/services/predictions.service';

import { useMemo, useState, useEffect, useCallback } from 'react';

import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Checkbox from '@mui/material/Checkbox';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import TableContainer from '@mui/material/TableContainer';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime } from 'src/utils/table-time-sort';

import {
  ApiError,
  listModels,
  startPrediction,
  getPredictionJob,
  deletePredictionJob,
  listAllPredictionJobs,
  uploadPredictionInput,
} from 'src/services';

// ----------------------------------------------------------------------

type Notify = (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => void;

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

function modelLabel(m: ModelVersionOut): string {
  const name = m.display_name?.trim();
  if (name) return `${name} · v${m.version_number}`;
  return `${m.algorithm} · v${m.version_number} · ${m.public_id.slice(0, 8)}…`;
}

function topShapEntries(shap: Record<string, unknown> | undefined, topN = 5): { feature: string; value: number }[] {
  if (!shap || typeof shap !== 'object') return [];
  const pf = shap.per_feature;
  if (!pf || typeof pf !== 'object') return [];
  return Object.entries(pf as Record<string, unknown>)
    .map(([k, v]) => {
      const n = Number(v);
      return Number.isFinite(n) ? { feature: k, value: n } : null;
    })
    .filter((x): x is { feature: string; value: number } => !!x)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, topN);
}

function contributionSummary(shap: Record<string, unknown> | undefined): string {
  const top = topShapEntries(shap, 5);
  if (!top.length) {
    if (shap && typeof shap === 'object' && typeof shap.status === 'string') return String(shap.status);
    return 'No contribution features available for this row.';
  }
  return top
    .map((t) => {
      const sign = t.value >= 0 ? '+' : '';
      return `${t.feature} ${sign}${t.value.toFixed(4)}`;
    })
    .join(' · ');
}

const BENIGN_LABELS = new Set(['BENIGN', 'NORMAL', 'LEGITIMATE', '0', 'FALSE', 'NO', 'NONE', '']);

/** True when the row is flagged or the predicted label is a non-benign class. */
function isAttackRow(r: { predicted_label?: string; flagged_attack_or_anomaly?: boolean }): boolean {
  if (r.flagged_attack_or_anomaly) return true;
  const label = String(r.predicted_label ?? '')
    .trim()
    .toUpperCase();
  return Boolean(label) && !BENIGN_LABELS.has(label);
}

/** Setting → Predictions: upload CSV, score, review per-row contribution summaries. */
export function SettingScoreRagPanel({ onNotify }: { onNotify: Notify }) {
  const [models, setModels] = useState<ModelVersionOut[]>([]);
  const [modelId, setModelId] = useState('');
  const [inputId, setInputId] = useState('');
  const [inputName, setInputName] = useState('');
  const [jobs, setJobs] = useState<PredictionJobListItem[]>([]);
  const [selectedJobIds, setSelectedJobIds] = useState<Set<string>>(() => new Set());
  const [activeJobId, setActiveJobId] = useState('');
  const [jobDetail, setJobDetail] = useState<PredictionJobOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [computeShap, setComputeShap] = useState(true);
  const [busy, setBusy] = useState(false);

  const modelsSorted = useMemo(() => sortByTime(models, (m) => m.created_at, 'desc'), [models]);
  const jobsSorted = useMemo(() => sortByTime(jobs, (j) => j.created_at, 'desc'), [jobs]);

  const resultRows = useMemo(() => {
    const rows = jobDetail?.results_json?.rows;
    return Array.isArray(rows) ? rows : [];
  }, [jobDetail]);

  const attackRowCount = useMemo(() => resultRows.filter((r) => isAttackRow(r)).length, [resultRows]);

  const resetForNewPrediction = () => {
    setInputId('');
    setInputName('');
  };

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [m, j] = await Promise.all([listModels(), listAllPredictionJobs()]);
      setModels(m);
      setJobs(j);
      setModelId((prev) => (prev && m.some((x) => x.public_id === prev) ? prev : m[0]?.public_id ?? ''));
      setSelectedJobIds((prev) => {
        const next = new Set<string>();
        for (const id of prev) if (j.some((x) => x.public_id === id)) next.add(id);
        return next;
      });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const loadJobResults = async (id: string) => {
    if (!id.trim()) return;
    setBusy(true);
    try {
      const full = await getPredictionJob(id.trim(), { includeResults: true });
      setActiveJobId(full.public_id);
      setJobDetail(full);
      onNotify({
        severity: 'success',
        text: `Loaded ${full.rows_total ?? '—'} row(s) · flagged ${full.rows_flagged ?? '—'}`,
      });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setBusy(false);
    }
  };

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    setUploading(true);
    onNotify(null);
    try {
      const r = await uploadPredictionInput(file);
      setInputId(r.public_id);
      setInputName(r.original_name || file.name);
      onNotify({ severity: 'success', text: `Uploaded “${r.original_name || file.name}” — ready to predict` });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setUploading(false);
    }
  };

  const runScore = async () => {
    if (!modelId || !inputId) return;
    onNotify(null);
    setBusy(true);
    try {
      const job = await startPrediction({
        model_version_public_id: modelId,
        input_file_public_id: inputId,
        compute_shap: computeShap,
      });
      onNotify({ severity: 'success', text: `Scoring started · ${job.public_id.slice(0, 8)}…` });
      await refresh();
      // poll briefly until completed/failed
      for (let i = 0; i < 60; i += 1) {
        await new Promise((r) => setTimeout(r, 1500));
        const j = await getPredictionJob(job.public_id, { includeResults: true });
        if (j.status === 'completed') {
          setActiveJobId(j.public_id);
          setJobDetail(j);
          resetForNewPrediction();
          onNotify({
            severity: 'success',
            text: `Completed · ${j.rows_total ?? 0} rows · ${j.rows_flagged ?? 0} flagged — upload a new CSV to run again`,
          });
          await refresh();
          return;
        }
        if (j.status === 'failed') {
          resetForNewPrediction();
          onNotify({ severity: 'error', text: j.error_message || 'Prediction failed' });
          await refresh();
          return;
        }
      }
      resetForNewPrediction();
      onNotify({ severity: 'info', text: 'Still running — upload a new CSV when ready for another run.' });
      await refresh();
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setBusy(false);
    }
  };

  const deleteSelected = async () => {
    const ids = [...selectedJobIds];
    if (!ids.length) return;
    if (
      !window.confirm(
        `Delete ${ids.length} prediction job(s) and all related agentic jobs/reports? This cannot be undone.`
      )
    ) {
      return;
    }
    setBusy(true);
    let ok = 0;
    try {
      for (const id of ids) {
        try {
          await deletePredictionJob(id);
          ok += 1;
        } catch (e) {
          onNotify({ severity: 'error', text: formatError(e) });
        }
      }
      if (activeJobId && ids.includes(activeJobId)) {
        setActiveJobId('');
        setJobDetail(null);
      }
      setSelectedJobIds(new Set());
      await refresh();
      onNotify({ severity: 'success', text: `Deleted ${ok} prediction job(s) and cascaded agentic data.` });
    } finally {
      setBusy(false);
    }
  };

  const allSelected = jobsSorted.length > 0 && jobsSorted.every((j) => selectedJobIds.has(j.public_id));

  return (
    <Stack spacing={2.5}>
      <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
          1 · Upload CSV to predict
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
          Choose a model, upload a CSV, run predictions. After results load, the CSV clears so you can upload a new file
          for the next run.
        </Typography>
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={1.5} alignItems={{ md: 'center' }} flexWrap="wrap">
          <TextField
            select
            size="small"
            label="Model"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
            sx={{ minWidth: 260, flex: 1 }}
            disabled={loading || busy}
          >
            {modelsSorted.length === 0 && (
              <MenuItem value="" disabled>
                Train a model first
              </MenuItem>
            )}
            {modelsSorted.map((m) => (
              <MenuItem key={m.public_id} value={m.public_id}>
                {modelLabel(m)}
              </MenuItem>
            ))}
          </TextField>
          <Button component="label" variant="contained" size="small" disabled={uploading || busy}>
            {uploading ? 'Uploading…' : inputId ? 'Replace CSV' : 'Upload CSV to predict'}
            <input type="file" hidden accept=".csv,text/csv" onChange={onUpload} />
          </Button>
          {inputName ? (
            <Chip
              size="small"
              color="info"
              variant="outlined"
              label={inputName}
              onDelete={() => {
                setInputId('');
                setInputName('');
              }}
            />
          ) : (
            <Chip size="small" variant="outlined" label="No CSV selected" sx={{ opacity: 0.7 }} />
          )}
          <FormControlLabel
            control={<Checkbox size="small" checked={computeShap} onChange={(_, v) => setComputeShap(v)} />}
            label="Contribution (SHAP)"
          />
          <Button
            variant="contained"
            color="secondary"
            size="small"
            disabled={!modelId || !inputId || busy}
            onClick={() => void runScore()}
          >
            {busy ? 'Predicting…' : 'Run prediction'}
          </Button>
          <Button size="small" onClick={() => void refresh()} disabled={loading || busy}>
            Refresh
          </Button>
          {(loading || busy) && <CircularProgress size={20} />}
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1} sx={{ mb: 1 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            2 · Prediction jobs
          </Typography>
          <Button size="small" color="error" variant="outlined" disabled={!selectedJobIds.size || busy} onClick={() => void deleteSelected()}>
            Delete selected ({selectedJobIds.size})
          </Button>
        </Stack>
        <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
          Delete removes the job and related agentic plans. Open <strong>Rows</strong> on a completed job for per-row
          contribution summaries.
        </Typography>
        <TableContainer sx={{ maxHeight: 280 }}>
          <Table size="small" stickyHeader>
            <TableHead>
              <TableRow>
                <TableCell padding="checkbox">
                  <Checkbox
                    size="small"
                    checked={allSelected}
                    indeterminate={!!selectedJobIds.size && !allSelected}
                    onChange={() => {
                      if (allSelected) setSelectedJobIds(new Set());
                      else setSelectedJobIds(new Set(jobsSorted.map((j) => j.public_id)));
                    }}
                  />
                </TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Job</TableCell>
                <TableCell>Rows</TableCell>
                <TableCell>Flagged</TableCell>
                <TableCell>Created</TableCell>
                <TableCell align="right">Preview</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobsSorted.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7}>
                    <Typography variant="body2" color="text.secondary">
                      No prediction jobs yet. Upload a CSV above to start.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
              {jobsSorted.map((j) => (
                <TableRow key={j.public_id} hover selected={activeJobId === j.public_id}>
                  <TableCell padding="checkbox">
                    <Checkbox
                      size="small"
                      checked={selectedJobIds.has(j.public_id)}
                      onChange={() => {
                        setSelectedJobIds((prev) => {
                          const next = new Set(prev);
                          if (next.has(j.public_id)) next.delete(j.public_id);
                          else next.add(j.public_id);
                          return next;
                        });
                      }}
                    />
                  </TableCell>
                  <TableCell>
                    <Chip size="small" label={j.status} variant="outlined" />
                  </TableCell>
                  <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{j.public_id.slice(0, 8)}…</TableCell>
                  <TableCell>{j.rows_total ?? '—'}</TableCell>
                  <TableCell>{j.rows_flagged ?? '—'}</TableCell>
                  <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap' }}>{fDateTime(j.created_at)}</TableCell>
                  <TableCell align="right">
                    <Button
                      size="small"
                      disabled={j.status !== 'completed' || busy}
                      onClick={() => void loadJobResults(j.public_id)}
                    >
                      Rows
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2, borderRadius: 2 }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
          3 · Row predictions & contribution summary
          {activeJobId ? ` · ${activeJobId.slice(0, 8)}…` : ''}
        </Typography>
        {!jobDetail && (
          <Alert severity="info" variant="outlined">
            After a job completes, open <strong>Rows</strong> to see each prediction with its contribution summary (top
            features driving the label).
          </Alert>
        )}
        {jobDetail && (
          <>
            <Stack direction="row" spacing={1} flexWrap="wrap" sx={{ mb: 1.5 }}>
              <Chip size="small" label={`${resultRows.length} rows`} />
              <Chip size="small" color="error" label={`${attackRowCount} attack`} />
              <Chip size="small" color="success" variant="outlined" label={`${resultRows.length - attackRowCount} benign`} />
              {jobDetail.results_json?.shap?.status && (
                <Chip size="small" variant="outlined" label={`SHAP: ${jobDetail.results_json.shap.status}`} />
              )}
            </Stack>
            <TableContainer sx={{ maxHeight: 520 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell width={64}>Row</TableCell>
                    <TableCell width={140}>Predicted</TableCell>
                    <TableCell width={100}>Confidence</TableCell>
                    <TableCell width={110}>Attack?</TableCell>
                    <TableCell>Contribution summary</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {resultRows.map((r) => {
                    const summary = contributionSummary(r.shap);
                    const top = topShapEntries(r.shap, 5);
                    const attack = isAttackRow(r);
                    return (
                      <TableRow
                        key={r.row_index}
                        hover
                        sx={{
                          bgcolor: (theme) =>
                            attack
                              ? theme.vars.palette.error.lighter
                              : theme.vars.palette.success.lighter,
                          '&:hover': {
                            bgcolor: (theme) =>
                              attack
                                ? theme.vars.palette.error.light
                                : theme.vars.palette.success.light,
                          },
                        }}
                      >
                        <TableCell>{r.row_index}</TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={r.predicted_label}
                            color={attack ? 'error' : 'success'}
                            variant={attack ? 'filled' : 'outlined'}
                          />
                        </TableCell>
                        <TableCell>{(Number(r.max_class_probability) * 100).toFixed(1)}%</TableCell>
                        <TableCell>
                          <Chip
                            size="small"
                            label={attack ? 'Yes' : 'No'}
                            color={attack ? 'error' : 'success'}
                            variant={attack ? 'filled' : 'outlined'}
                            sx={{ minWidth: 52 }}
                          />
                        </TableCell>
                        <TableCell sx={{ maxWidth: 560 }}>
                          <Typography variant="body2" sx={{ lineHeight: 1.45 }} title={summary}>
                            {summary}
                          </Typography>
                          {top.length > 0 && (
                            <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap sx={{ mt: 0.75 }}>
                              {top.map((t) => (
                                <Chip
                                  key={`${r.row_index}-${t.feature}`}
                                  size="small"
                                  variant="outlined"
                                  color={t.value >= 0 ? 'warning' : 'default'}
                                  label={`${t.feature}: ${t.value >= 0 ? '+' : ''}${t.value.toFixed(3)}`}
                                  sx={{ height: 22, fontSize: 11 }}
                                />
                              ))}
                            </Stack>
                          )}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                  {resultRows.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={5}>
                        <Typography variant="body2" color="text.secondary">
                          No row results stored on this job.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1.5 }}>
              Next: open <strong>Agentic</strong> and choose this prediction job to run plans on selected rows.
            </Typography>
          </>
        )}
      </Paper>
    </Stack>
  );
}
