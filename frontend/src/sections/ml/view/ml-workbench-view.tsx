import type { PredictionJobListItem } from 'src/services/predictions.service';
import type {
  JobStatus,
  KBQueryHit,
  AgenticJobOut,
  ManagedFileOut,
  TrainingJobOut,
  ModelVersionOut,
  RAGTemplateItem,
  PredictionJobOut,
  AgenticReportOut,
  DatasetPreviewOut,
  PredictionResultsJson,
  KBRAGLatestPredictionResponse,
} from 'src/api/types';

import { useRef, useMemo, useState, useEffect, useCallback, type ReactNode } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Tabs from '@mui/material/Tabs';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Tooltip from '@mui/material/Tooltip';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import Checkbox from '@mui/material/Checkbox';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import Collapse from '@mui/material/Collapse';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import Accordion from '@mui/material/Accordion';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';

import { fDateTime } from 'src/utils/format-time';
import { sortByTime, type TimeSortOrder } from 'src/utils/table-time-sort';

import { DashboardContent } from 'src/layouts/dashboard';
import {
  kbQuery,
  ApiError,
  kbDelete,
  kbUpload,
  getHealth,
  listModels,
  agentDecide,
  kbListFiles,
  isPipelineKbArtifactName,
  deleteModel,
  listDatasets,
  kbFuseHitsMmr,
  deleteDataset,
  startTraining,
  uploadDataset,
  getTrainingJob,
  getAgentReport,
  rebuildTraining,
  startPrediction,
  getPredictionJob,
  listTrainingJobs,
  createAgenticJob,
  deleteTrainingJob,
  getDatasetPreview,
  deleteAgentReport,
  listAllAgenticJobs,
  deletePredictionJob,
  listAllAgentReports,
  listPredictionInputs,
  listAllPredictionJobs,
  uploadPredictionInput,
  updateModelDisplayName,
  kbLlmShapRetrievalQuery,
  agentDecidePromptPreview,
  kbRagTemplatesPredictionJob,
  deleteAllPendingPredictionJobs,
} from 'src/services';

import { Iconify } from 'src/components/iconify';
import { TimeSortHeadCell } from 'src/components/table-sort/time-sort-head-cell';
import { ModelVersionDetailDialog } from 'src/components/run-monitoring/detail-dialogs';

import { SettingAgenticPanel } from 'src/sections/ml/view/setting-agentic-panel';
import { SettingScoreRagPanel } from 'src/sections/ml/view/setting-score-rag-panel';
import { SettingActiveActionsPanel } from 'src/sections/ml/view/setting-active-actions-panel';
import {
  readAgenticPrep,
  writeAgenticPrep,
  upsertAgenticJobHandoff,
  type AgenticPrepPayload,
  AGENTIC_PREP_UPDATED_EVENT,
} from 'src/sections/ml/agentic-prep-storage';

// ----------------------------------------------------------------------

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function MlWorkbenchView() {
  /** 0 Overview · 1 Datasets · 2 Training · 3 Knowledge · 4 Predictions · 5 Agentic planner · 6 Agentic execution */
  const [tab, setTab] = useState(0);
  const [banner, setBanner] = useState<{ severity: 'success' | 'error' | 'info'; text: string } | null>(null);
  const [ping, setPing] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setPing(h.status);
      setBanner(null);
    } catch (e) {
      setPing(null);
      setBanner({ severity: 'error', text: formatError(e) });
    }
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  return (
    <DashboardContent maxWidth="xl">
      <Stack spacing={1.5} sx={{ mb: 2.5 }}>
        <Stack direction="row" alignItems="center" spacing={1.5} flexWrap="wrap">
          <Typography variant="h4">Setting</Typography>
          {ping && <Chip size="small" color="success" label={ping} />}
          <Button size="small" variant="text" onClick={refreshHealth}>
            Ping
          </Button>
        </Stack>
        <Typography variant="body2" color="text.secondary">
          Configure data, models, scoring, action plans, and active apply / blockchain validation.
        </Typography>
        {banner && (
          <Alert severity={banner.severity} onClose={() => setBanner(null)}>
            {banner.text}
          </Alert>
        )}
      </Stack>

      <Card
        elevation={0}
        sx={{
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <Tabs
          value={tab}
          onChange={(_, v) => {
            setBanner(null);
            setTab(v);
          }}
          variant="scrollable"
          scrollButtons="auto"
          allowScrollButtonsMobile
          sx={{
            px: 1,
            borderBottom: 1,
            borderColor: 'divider',
            minHeight: 48,
            '& .MuiTab-root': { minHeight: 48, textTransform: 'none', fontWeight: 600 },
          }}
        >
          <Tab label="Overview" />
          <Tab label="Datasets" />
          <Tab label="Training" />
          <Tab label="Knowledge" />
          <Tab label="Predictions" />
          <Tab label="Agentic planner" />
          <Tab label="Agentic execution" />
        </Tabs>
        <CardContent>
          {tab === 0 && <SettingOverviewPanel onGo={setTab} />}
          {tab === 1 && <DatasetsPanel onNotify={setBanner} />}
          {tab === 2 && <TrainingPanel onNotify={setBanner} />}
          {tab === 3 && <KbPanel onNotify={setBanner} />}
          {tab === 4 && <SettingScoreRagPanel onNotify={setBanner} />}
          {tab === 5 && <SettingAgenticPanel onNotify={setBanner} />}
          {tab === 6 && <SettingActiveActionsPanel onNotify={setBanner} />}
        </CardContent>
      </Card>
    </DashboardContent>
  );
}

type PanelProps = { onNotify: (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => void };

const SETTING_SECTIONS: {
  tab: number;
  title: string;
  blurb: string;
  /** MUI palette key used for soft card tint */
  tint: 'info' | 'success' | 'warning' | 'secondary' | 'primary' | 'error';
}[] = [
  { tab: 1, title: 'Datasets', blurb: 'Upload and preview training CSVs.', tint: 'info' },
  { tab: 2, title: 'Training', blurb: 'Train VFL models and set display names.', tint: 'success' },
  { tab: 3, title: 'Knowledge', blurb: 'Upload and manage knowledge documents.', tint: 'warning' },
  { tab: 4, title: 'Predictions', blurb: 'Upload a CSV, score rows, and see contribution summaries.', tint: 'secondary' },
  { tab: 5, title: 'Agentic planner', blurb: 'Pick rows, save action plans, open Details per row.', tint: 'primary' },
  { tab: 6, title: 'Agentic execution', blurb: 'Validate blockchain, apply plans, inspect results.', tint: 'error' },
];

function SettingOverviewPanel({ onGo }: { onGo: (tab: number) => void }) {
  return (
    <Stack spacing={2.5}>
      <Typography variant="body1" color="text.secondary" sx={{ lineHeight: 1.7, maxWidth: 720 }}>
        Flow: datasets → training → predictions → agentic planner → agentic execution (blockchain validate / apply). Knowledge
        docs feed RAG when enabled on decide.
      </Typography>
      <Box
        sx={{
          display: 'grid',
          gap: 1.5,
          gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr', md: '1fr 1fr 1fr' },
        }}
      >
        {SETTING_SECTIONS.map((s) => (
          <Paper
            key={s.tab}
            variant="outlined"
            sx={{
              p: 2,
              borderRadius: 2,
              height: 1,
              cursor: 'pointer',
              borderColor: (t) => alpha(t.palette[s.tint].main, 0.28),
              bgcolor: (t) => alpha(t.palette[s.tint].main, 0.1),
              transition: (t) => t.transitions.create(['border-color', 'background-color', 'box-shadow']),
              '&:hover': {
                borderColor: `${s.tint}.main`,
                bgcolor: (t) => alpha(t.palette[s.tint].main, 0.16),
                boxShadow: 1,
              },
            }}
            onClick={() => onGo(s.tab)}
          >
            <Stack direction="row" alignItems="flex-start" justifyContent="space-between" gap={1}>
              <Box sx={{ minWidth: 0 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: `${s.tint}.dark` }}>
                  {s.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  {s.blurb}
                </Typography>
              </Box>
              <Iconify icon="eva:arrow-ios-forward-fill" width={18} sx={{ color: `${s.tint}.main`, mt: 0.5, opacity: 0.7 }} />
            </Stack>
          </Paper>
        ))}
      </Box>
    </Stack>
  );
}

function formatBytes(n: number | null): string {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function cellPreview(v: unknown): string {
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function DatasetsPanel({ onNotify }: PanelProps) {
  const [rows, setRows] = useState<ManagedFileOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [replacePublicId, setReplacePublicId] = useState('');
  const [previewRow, setPreviewRow] = useState<ManagedFileOut | null>(null);
  const [previewData, setPreviewData] = useState<DatasetPreviewOut | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [datasetTimeOrder, setDatasetTimeOrder] = useState<TimeSortOrder>('desc');

  const rowsSorted = useMemo(
    () => sortByTime(rows, (r) => r.created_at, datasetTimeOrder),
    [rows, datasetTimeOrder]
  );

  const load = useCallback(async () => {
    setLoading(true);
    onNotify(null);
    try {
      const list = await listDatasets();
      setRows(list);
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    load();
  }, [load]);

  const closePreview = useCallback(() => {
    setPreviewRow(null);
    setPreviewData(null);
    setPreviewLoading(false);
  }, []);

  const openPreview = useCallback(
    async (r: ManagedFileOut) => {
      setPreviewRow(r);
      setPreviewData(null);
      setPreviewLoading(true);
      try {
        const data = await getDatasetPreview(r.public_id, 50);
        setPreviewData(data);
      } catch (e) {
        onNotify({ severity: 'error', text: formatError(e) });
        closePreview();
      } finally {
        setPreviewLoading(false);
      }
    },
    [closePreview, onNotify]
  );

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    onNotify(null);
    setUploading(true);
    try {
      const replace = replacePublicId.trim() || null;
      const r = await uploadDataset(file, replace);
      onNotify({
        severity: 'success',
        text: replace
          ? `New version uploaded: ${r.original_name} → v${r.version} (${r.public_id})`
          : `Uploaded ${r.original_name} → ${r.public_id} (v${r.version})`,
      });
      setReplacePublicId('');
      await load();
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setUploading(false);
    }
  };

  return (
    <Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">
        Upload training CSV files. Each upload gets a <strong>version</strong> number. To create a new version of an
        existing dataset, paste its <strong>public_id</strong> below before uploading.
      </Typography>
      <TextField
        size="small"
        fullWidth
        label="Replace / new version of dataset (public_id, optional)"
        placeholder="e.g. uuid of the row you want to supersede"
        value={replacePublicId}
        onChange={(e) => setReplacePublicId(e.target.value)}
        helperText="When set, the new file becomes the next version for that dataset chain."
        disabled={uploading}
      />
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button variant="contained" component="label" disabled={loading || uploading}>
          {uploading ? 'Uploading…' : 'Upload CSV'}
          <input type="file" hidden accept=".csv,text/csv" onChange={onUpload} disabled={loading || uploading} />
        </Button>
        <Button onClick={load} disabled={loading || uploading}>
          Refresh list
        </Button>
        {loading && <CircularProgress size={22} aria-label="Loading list" />}
        {uploading && <CircularProgress size={22} aria-label="Uploading file" />}
      </Stack>

      <Typography variant="subtitle2">Uploaded datasets ({rows.length})</Typography>
      <TableContainer sx={{ maxWidth: 1, overflowX: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>File name</TableCell>
              <TableCell>Version</TableCell>
              <TableCell>public_id</TableCell>
              <TableCell>Parent</TableCell>
              <TableCell>Size</TableCell>
              <TimeSortHeadCell label="Uploaded" order={datasetTimeOrder} onOrderChange={setDatasetTimeOrder} />
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.length === 0 && !loading && (
              <TableRow>
                <TableCell colSpan={7}>
                  <Typography variant="body2" color="text.secondary">
                    No datasets yet. Upload a CSV above.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {rowsSorted.map((r) => (
              <TableRow key={r.public_id} hover>
                <TableCell sx={{ whiteSpace: 'nowrap' }}>{r.original_name}</TableCell>
                <TableCell>
                  <Chip size="small" label={`v${r.version}`} color="primary" variant="outlined" />
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, maxWidth: 200 }}>{r.public_id}</TableCell>
                <TableCell sx={{ typography: 'caption', color: 'text.secondary' }}>
                  {r.parent_file_id != null ? `#${r.parent_file_id}` : '—'}
                </TableCell>
                <TableCell>{formatBytes(r.size_bytes)}</TableCell>
                <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(r.created_at)}</TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={0.5} justifyContent="flex-end" flexWrap="wrap">
                    <Button size="small" variant="outlined" onClick={() => openPreview(r)}>
                      Preview
                    </Button>
                    <Button
                      size="small"
                      onClick={() => {
                        setReplacePublicId(r.public_id);
                        onNotify({ severity: 'info', text: 'Paste is set — choose Upload CSV to add a new version.' });
                      }}
                    >
                      New version
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      onClick={async () => {
                        if (!window.confirm(`Delete dataset ${r.public_id} (v${r.version})?`)) return;
                        try {
                          await deleteDataset(r.public_id);
                          onNotify({ severity: 'info', text: 'Deleted' });
                          if (previewRow?.public_id === r.public_id) closePreview();
                          await load();
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      }}
                    >
                      Delete
                    </Button>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={!!previewRow} onClose={closePreview} maxWidth="lg" fullWidth>
        <DialogTitle>
          Dataset preview
          {previewRow && (
            <Typography variant="caption" display="block" color="text.secondary" sx={{ mt: 0.5 }}>
              {previewRow.original_name} · v{previewRow.version} · {previewRow.public_id}
            </Typography>
          )}
        </DialogTitle>
        <DialogContent dividers>
          {previewLoading && (
            <Box sx={{ py: 4, display: 'flex', justifyContent: 'center' }}>
              <CircularProgress />
            </Box>
          )}
          {!previewLoading && previewData && previewData.columns.length === 0 && (
            <Typography color="text.secondary">No rows in file.</Typography>
          )}
          {!previewLoading && previewData && previewData.columns.length > 0 && (
            <>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
                Showing first {previewData.row_count} row(s) (limit {previewData.preview_limit}).
              </Typography>
              <TableContainer sx={{ maxHeight: 420 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      {previewData.columns.map((col) => (
                        <TableCell key={col} sx={{ fontWeight: 'fontWeightBold', whiteSpace: 'nowrap' }}>
                          {col}
                        </TableCell>
                      ))}
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {previewData.rows.map((row, i) => (
                      <TableRow key={i}>
                        {previewData.columns.map((col) => (
                          <TableCell key={col} sx={{ maxWidth: 200, fontSize: 12 }}>
                            <Box sx={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={cellPreview(row[col])}>
                              {cellPreview(row[col])}
                            </Box>
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={closePreview}>Close</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}

function trainingStatusColor(
  s: TrainingJobOut['status']
): 'default' | 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' {
  switch (s) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'error';
    case 'running':
      return 'info';
    default:
      return 'warning';
  }
}

function TrainingPanel({ onNotify }: PanelProps) {
  const [datasets, setDatasets] = useState<ManagedFileOut[]>([]);
  const [datasetId, setDatasetId] = useState('');
  const [targetColumn, setTargetColumn] = useState('label');
  const [jobs, setJobs] = useState<TrainingJobOut[]>([]);
  const [models, setModels] = useState<Awaited<ReturnType<typeof listModels>>>([]);
  const [modelDetail, setModelDetail] = useState<ModelVersionOut | null>(null);
  const [jobsTimeOrder, setJobsTimeOrder] = useState<TimeSortOrder>('desc');
  const [modelsTimeOrder, setModelsTimeOrder] = useState<TimeSortOrder>('desc');
  const [renameModel, setRenameModel] = useState<ModelVersionOut | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [renaming, setRenaming] = useState(false);

  const datasetsForSelect = useMemo(
    () => sortByTime(datasets, (d) => d.created_at, 'desc'),
    [datasets]
  );

  const jobsSorted = useMemo(
    () => sortByTime(jobs, (j) => j.updated_at, jobsTimeOrder),
    [jobs, jobsTimeOrder]
  );

  const modelsSorted = useMemo(
    () => sortByTime(models, (m) => m.created_at, modelsTimeOrder),
    [models, modelsTimeOrder]
  );

  const loadDatasets = useCallback(async () => {
    try {
      const list = await listDatasets();
      setDatasets(list);
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    }
  }, [onNotify]);

  const loadJobs = useCallback(async () => {
    try {
      setJobs(await listTrainingJobs(100, 0));
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    }
  }, [onNotify]);

  const refreshModels = useCallback(async () => {
    try {
      setModels(await listModels());
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    }
  }, [onNotify]);

  useEffect(() => {
    loadDatasets();
    loadJobs();
    refreshModels();
  }, [loadDatasets, loadJobs, refreshModels]);

  useEffect(() => {
    const active = jobs.some((j) => j.status === 'pending' || j.status === 'running');
    if (!active) {
      return undefined;
    }
    const t = setInterval(() => {
      void listTrainingJobs(100, 0).then(setJobs).catch(() => {});
    }, 2500);
    return () => clearInterval(t);
  }, [jobs]);

  return (
    <Stack spacing={2}>
      <Alert severity="info" variant="outlined">
        <strong>Vertical federated learning (VFL) alignment:</strong> each training run is tied to one uploaded{' '}
        <strong>dataset</strong> (party-held features in your CSV). Pick the dataset below; the server stores the job in
        the database and traces <code>dataset_file_public_id</code> for audit. Retraining creates a new job and a new
        model artifact on disk.
      </Alert>

      <Typography variant="subtitle2">Train VFL model</Typography>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'flex-start' }}>
        <TextField
          select
          label="Training dataset"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          sx={{ minWidth: 280, flex: 1 }}
          helperText={
            datasets.length === 0
              ? 'Upload a CSV under Datasets first.'
              : 'Choose by file name and version'
          }
        >
          {datasets.length === 0 && (
            <MenuItem value="" disabled>
              No datasets
            </MenuItem>
          )}
          {datasetsForSelect.map((d) => (
            <MenuItem key={d.public_id} value={d.public_id}>
              {d.original_name} · v{d.version} · {d.public_id.slice(0, 8)}…
            </MenuItem>
          ))}
        </TextField>
        <TextField label="Target column" value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} sx={{ minWidth: 140 }} />
        <Chip label="VFL only" color="primary" variant="outlined" sx={{ height: 40, fontWeight: 700 }} />
        <Button
          variant="contained"
          disabled={!datasetId.trim()}
          onClick={async () => {
            onNotify(null);
            try {
              const res = await startTraining({
                dataset_file_public_id: datasetId.trim(),
                target_column: targetColumn.trim(),
                algorithm: 'vfl',
                vfl_agent_definitions_path: 'storage/agentic_features.json',
              });
              onNotify({ severity: 'success', text: `VFL training queued · job ${res.job_public_id}` });
              await loadJobs();
              await refreshModels();
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Train VFL model
        </Button>
      </Stack>

      <Stack direction="row" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
        <Typography variant="subtitle2">Training jobs ({jobs.length})</Typography>
        <Stack direction="row" spacing={1}>
          <Button size="small" onClick={loadJobs}>
            Refresh list
          </Button>
          <Button size="small" onClick={loadDatasets}>
            Reload datasets
          </Button>
        </Stack>
      </Stack>
      <TableContainer sx={{ maxWidth: 1, overflowX: 'auto' }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>Status</TableCell>
              <TableCell>Job ID</TableCell>
              <TableCell>Dataset</TableCell>
              <TableCell>Target</TableCell>
              <TableCell>Algorithm</TableCell>
              <TableCell>Model</TableCell>
              <TimeSortHeadCell label="Updated" order={jobsTimeOrder} onOrderChange={setJobsTimeOrder} />
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {jobs.length === 0 && (
              <TableRow>
                <TableCell colSpan={8}>
                  <Typography variant="body2" color="text.secondary">
                    No training jobs yet. Start one above.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {jobsSorted.map((j) => (
              <TableRow key={j.public_id} hover>
                <TableCell>
                  <Chip size="small" label={j.status} color={trainingStatusColor(j.status)} variant="outlined" />
                </TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>{j.public_id}</TableCell>
                <TableCell sx={{ maxWidth: 200 }}>
                  <Typography variant="body2" noWrap title={j.dataset_original_name ?? ''}>
                    {j.dataset_original_name ?? '—'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" noWrap display="block">
                    {j.dataset_file_public_id ?? j.dataset_file_id}
                  </Typography>
                </TableCell>
                <TableCell>{j.target_column}</TableCell>
                <TableCell>{j.algorithm}</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {j.model_version_public_id ?? '—'}
                </TableCell>
                <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap' }}>{fDateTime(j.updated_at)}</TableCell>
                <TableCell align="right">
                  <Stack direction="row" spacing={0.5} justifyContent="flex-end" flexWrap="wrap">
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={async () => {
                        try {
                          const fresh = await getTrainingJob(j.public_id);
                          setJobs((prev) => prev.map((x) => (x.public_id === fresh.public_id ? fresh : x)));
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      }}
                    >
                      Refresh
                    </Button>
                    <Button
                      size="small"
                      onClick={async () => {
                        if (!window.confirm(`Rebuild model from job ${j.public_id.slice(0, 8)}…? (new job, same dataset & settings)`)) return;
                        onNotify(null);
                        try {
                          const res = await rebuildTraining({ from_job_public_id: j.public_id });
                          onNotify({ severity: 'success', text: `Rebuild queued · ${res.job_public_id}` });
                          await loadJobs();
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      }}
                    >
                      Rebuild
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      variant="outlined"
                      disabled={j.status === 'running'}
                      onClick={async () => {
                        if (
                          !window.confirm(
                            `Delete training job ${j.public_id.slice(0, 8)}…? The job record is removed; any registered model stays in the registry (delete it separately if needed).`
                          )
                        ) {
                          return;
                        }
                        onNotify(null);
                        try {
                          await deleteTrainingJob(j.public_id);
                          onNotify({ severity: 'success', text: 'Training job deleted' });
                          await loadJobs();
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      }}
                    >
                      Delete
                    </Button>
                  </Stack>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Stack direction="row" justifyContent="space-between" alignItems="center">
        <Typography variant="subtitle2">Registered models</Typography>
        <Button size="small" onClick={refreshModels}>
          Refresh models
        </Button>
      </Stack>
      <Table
        size="small"
        sx={{
          '& .MuiTableCell-root': { py: 0.5, px: 1, fontSize: '0.8125rem' },
          '& .MuiTableCell-head': { fontWeight: 700, fontSize: '0.75rem' },
        }}
      >
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>public_id</TableCell>
            <TableCell>algorithm</TableCell>
            <TableCell>version</TableCell>
            <TimeSortHeadCell label="Registered" order={modelsTimeOrder} onOrderChange={setModelsTimeOrder} />
            <TableCell align="right">View</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {models.length === 0 && (
            <TableRow>
              <TableCell colSpan={7}>
                <Typography variant="body2" color="text.secondary">
                  No registered models yet.
                </Typography>
              </TableCell>
            </TableRow>
          )}
          {modelsSorted.map((m) => (
            <TableRow key={m.public_id}>
              <TableCell sx={{ maxWidth: 200 }}>
                <Typography variant="body2" noWrap title={m.display_name || undefined} sx={{ fontWeight: 600 }}>
                  {m.display_name?.trim() || '—'}
                </Typography>
              </TableCell>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{m.public_id.slice(0, 8)}…</TableCell>
              <TableCell>{m.algorithm}</TableCell>
              <TableCell>{m.version_number}</TableCell>
              <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(m.created_at)}</TableCell>
              <TableCell align="right">
                <Tooltip title="View model details">
                  <IconButton
                    size="small"
                    color="primary"
                    aria-label="View model details"
                    onClick={() => setModelDetail(m)}
                    sx={{ p: 0.35 }}
                  >
                    <Iconify icon="solar:eye-bold" width={18} />
                  </IconButton>
                </Tooltip>
              </TableCell>
              <TableCell align="right">
                <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                  <Button
                    size="small"
                    onClick={() => {
                      setRenameModel(m);
                      setRenameValue(m.display_name || '');
                    }}
                  >
                    Rename
                  </Button>
                  <Button
                    size="small"
                    color="error"
                    variant="outlined"
                    onClick={async () => {
                      if (
                        !window.confirm(
                          `Delete model ${m.public_id.slice(0, 8)}… (v${m.version_number})? This removes the registry row and the .joblib file. Prediction jobs that used this model must be gone first.`
                        )
                      ) {
                        return;
                      }
                      onNotify(null);
                      try {
                        await deleteModel(m.public_id);
                        onNotify({ severity: 'success', text: 'Model deleted' });
                        await refreshModels();
                        await loadJobs();
                      } catch (e) {
                        onNotify({ severity: 'error', text: formatError(e) });
                      }
                    }}
                  >
                    Delete
                  </Button>
                </Stack>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <Dialog
        open={!!renameModel}
        onClose={() => {
          if (!renaming) setRenameModel(null);
        }}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Rename model</DialogTitle>
        <DialogContent>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1.5 }}>
            {renameModel?.public_id}
          </Typography>
          <TextField
            autoFocus
            fullWidth
            label="Display name"
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            helperText="Shown in Setting lists. Leave blank to clear."
          />
        </DialogContent>
        <DialogActions>
          <Button disabled={renaming} onClick={() => setRenameModel(null)}>
            Cancel
          </Button>
          <Button
            variant="contained"
            disabled={renaming || !renameModel}
            onClick={async () => {
              if (!renameModel) return;
              setRenaming(true);
              try {
                const name = renameValue.trim() || null;
                await updateModelDisplayName(renameModel.public_id, name);
                onNotify({ severity: 'success', text: name ? `Named “${name}”` : 'Name cleared' });
                setRenameModel(null);
                await refreshModels();
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              } finally {
                setRenaming(false);
              }
            }}
          >
            {renaming ? 'Saving…' : 'Save'}
          </Button>
        </DialogActions>
      </Dialog>

      <ModelVersionDetailDialog
        open={Boolean(modelDetail)}
        model={modelDetail}
        onClose={() => setModelDetail(null)}
      />
    </Stack>
  );
}

// Legacy panel — superseded by Setting tabs; kept for reference.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function PredictionsPanel({ onNotify }: PanelProps) {
  const [models, setModels] = useState<ModelVersionOut[]>([]);
  const [inputs, setInputs] = useState<ManagedFileOut[]>([]);
  const [modelId, setModelId] = useState('');
  const [inputId, setInputId] = useState('');
  const [predJobId, setPredJobId] = useState('');
  const [predSummary, setPredSummary] = useState('');
  const [uploadingCsv, setUploadingCsv] = useState(false);
  const [loadingLists, setLoadingLists] = useState(false);
  const [computeShap, setComputeShap] = useState(true);
  const [loadedResults, setLoadedResults] = useState<Awaited<ReturnType<typeof getPredictionJob>>['results_json']>(null);
  const [loadingResults, setLoadingResults] = useState(false);
  const [pollJob, setPollJob] = useState(false);
  const [predictionJobsList, setPredictionJobsList] = useState<PredictionJobListItem[]>([]);
  const [jobsListLoading, setJobsListLoading] = useState(false);
  const [recentJobsExpanded, setRecentJobsExpanded] = useState(true);
  const prevJobsListLen = useRef(-1);
  const [viewJobOpen, setViewJobOpen] = useState(false);
  const [viewJobDetail, setViewJobDetail] = useState<PredictionJobOut | null>(null);
  const [viewJobLoading, setViewJobLoading] = useState(false);
  const lastStatusRef = useRef<string | null>(null);
  const failureNotifiedForId = useRef<string>('');
  const [predJobsTimeOrder, setPredJobsTimeOrder] = useState<TimeSortOrder>('desc');

  const predictionJobsSorted = useMemo(
    () => sortByTime(predictionJobsList, (j) => j.created_at, predJobsTimeOrder),
    [predictionJobsList, predJobsTimeOrder]
  );

  const modelsForPredSelect = useMemo(
    () => sortByTime(models, (m) => m.created_at, 'desc'),
    [models]
  );

  const inputsForPredSelect = useMemo(
    () => sortByTime(inputs, (f) => f.created_at, 'desc'),
    [inputs]
  );

  const refreshPredictionJobsList = useCallback(async () => {
    setJobsListLoading(true);
    try {
      setPredictionJobsList(await listAllPredictionJobs());
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setJobsListLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void refreshPredictionJobsList();
  }, [refreshPredictionJobsList]);

  useEffect(() => {
    const n = predictionJobsList.length;
    if (prevJobsListLen.current === -1) {
      prevJobsListLen.current = n;
      setRecentJobsExpanded(n > 0);
      return;
    }
    if (n > 0 && prevJobsListLen.current === 0) {
      setRecentJobsExpanded(true);
    }
    if (n === 0 && prevJobsListLen.current > 0) {
      setRecentJobsExpanded(false);
    }
    prevJobsListLen.current = n;
  }, [predictionJobsList.length]);

  const refreshLists = useCallback(async () => {
    setLoadingLists(true);
    try {
      const [m, inp] = await Promise.all([listModels(), listPredictionInputs()]);
      setModels(m);
      setInputs(inp);
      setModelId((prev) => {
        if (prev && m.some((x) => x.public_id === prev)) return prev;
        const newest = sortByTime(m, (x) => x.created_at, 'desc')[0];
        return newest?.public_id ?? '';
      });
      // Do not auto-pick an input: user chooses runtime scoring CSV (separate from training data).
      setInputId((prev) => {
        if (prev && inp.some((x) => x.public_id === prev)) return prev;
        return '';
      });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoadingLists(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void refreshLists();
  }, [refreshLists]);

  useEffect(() => {
    lastStatusRef.current = null;
    failureNotifiedForId.current = '';
  }, [predJobId]);

  useEffect(() => {
    const id = predJobId.trim();
    if (!id || !pollJob) {
      return undefined;
    }
    let cancelled = false;

    const pollOnce = async () => {
      try {
        const j = await getPredictionJob(id);
        if (cancelled) return;
        setPredSummary(`Status: ${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'}`);
        if (j.status === 'failed' && j.error_message && failureNotifiedForId.current !== id) {
          failureNotifiedForId.current = id;
          onNotify({ severity: 'error', text: j.error_message });
        }
        if (j.status === 'completed' && lastStatusRef.current !== 'completed') {
          lastStatusRef.current = 'completed';
          try {
            const full = await getPredictionJob(id, { includeResults: true });
            if (!cancelled) {
              setLoadedResults(full.results_json ?? null);
              const sh = full.results_json && typeof full.results_json === 'object' && 'shap' in full.results_json
                ? (full.results_json as { shap?: { status?: string; detail?: string } }).shap
                : undefined;
              onNotify({
                severity: 'success',
                text: sh?.status
                  ? `Prediction completed · SHAP/attribution: ${sh.status}${sh.detail ? ` — ${sh.detail}` : ''}`
                  : 'Prediction completed — per-row results loaded.',
              });
            }
          } catch {
            /* ignore */
          }
        } else if (j.status !== 'completed') {
          lastStatusRef.current = j.status;
        }
        if (j.status === 'completed' || j.status === 'failed') {
          setPollJob(false);
        }
      } catch (e) {
        if (!cancelled) {
          onNotify({ severity: 'error', text: formatError(e) });
          setPollJob(false);
        }
      }
    };

    void pollOnce();
    const t = setInterval(() => void pollOnce(), 2000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [predJobId, pollJob, onNotify]);

  const onUploadCsv: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    onNotify(null);
    setUploadingCsv(true);
    try {
      const r = await uploadPredictionInput(file);
      setInputId(r.public_id);
      const inp = await listPredictionInputs();
      setInputs(inp);
      onNotify({ severity: 'success', text: `Input uploaded → ${r.original_name} (${r.public_id})` });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setUploadingCsv(false);
    }
  };

  return (
    <Stack spacing={2.5}>
      <Alert severity="info" variant="outlined">
        <strong>Scoring is separate from training.</strong> Choose a <strong>registered model</strong> (trained under Data &amp;
        training → Training &amp; models), then at run time pick the <strong>input CSV</strong> you want scored. That file is{' '}
        <strong>not</strong> the training dataset—it is only used to produce predictions for those rows.
      </Alert>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button size="small" variant="outlined" onClick={() => void refreshLists()} disabled={loadingLists || uploadingCsv}>
          {loadingLists ? 'Loading…' : 'Refresh models & inputs'}
        </Button>
        <Button size="small" variant="outlined" onClick={() => void refreshPredictionJobsList()} disabled={jobsListLoading}>
          {jobsListLoading ? 'Loading jobs…' : 'Refresh job list'}
        </Button>
        <Button
          size="small"
          variant="outlined"
          color="warning"
          disabled={jobsListLoading}
          onClick={async () => {
            const pendingCount = predictionJobsList.filter((j) => j.status === 'pending').length;
            if (
              !window.confirm(
                pendingCount > 0
                  ? `Delete all ${pendingCount} pending prediction job(s)? Running, completed, and failed jobs are not removed.`
                  : 'Delete all pending prediction jobs? (None visible in the current list — server will still remove any pending rows.)'
              )
            ) {
              return;
            }
            onNotify(null);
            try {
              const r = await deleteAllPendingPredictionJobs();
              await refreshPredictionJobsList();
              onNotify({
                severity: 'success',
                text: `Removed ${r.deleted} pending prediction job(s).`,
              });
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Delete all pending jobs
        </Button>
      </Stack>

      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 1 }}>
          1 · Model (from registry)
        </Typography>
        <TextField
          select
          fullWidth
          label="Model for this job"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          sx={{ maxWidth: 560 }}
          helperText={
            models.length === 0
              ? 'Train a model under Training first.'
              : 'Dropdown lists registered models. Pick which weights to use for scoring.'
          }
        >
          {models.length === 0 && (
            <MenuItem value="" disabled>
              No models
            </MenuItem>
          )}
          {modelsForPredSelect.map((m) => (
            <MenuItem key={m.public_id} value={m.public_id}>
              v{m.version_number} · {m.algorithm} · {m.public_id.slice(0, 8)}…
            </MenuItem>
          ))}
        </TextField>
      </Box>

      <Divider />

      <Box>
        <Typography variant="subtitle2" sx={{ fontWeight: 800, mb: 0.5 }}>
          2 · Runtime input data (prediction CSV only)
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5, maxWidth: 720, lineHeight: 1.65 }}>
          Upload scoring rows here, then <strong>select which file</strong> this run should use. Switch the dropdown whenever
          you want a different batch scored with the same model.
        </Typography>
        <FormControlLabel
          control={
            <Checkbox
              checked={computeShap}
              onChange={(_, c) => setComputeShap(c)}
              size="small"
            />
          }
          label={
            <Typography variant="body2" color="text.secondary">
              Compute attributions (sklearn: TreeExplainer SHAP; VFL: gradient×input per feature for the predicted class —
              large batches skip on server)
            </Typography>
          }
          sx={{ alignItems: 'flex-start', mb: 1, ml: 0 }}
        />
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ mb: 2 }}>
          <Button variant="outlined" component="label" disabled={uploadingCsv}>
            {uploadingCsv ? 'Uploading…' : 'Upload scoring CSV'}
            <input type="file" hidden accept=".csv,text/csv" onChange={onUploadCsv} disabled={uploadingCsv} />
          </Button>
          {uploadingCsv && <CircularProgress size={22} aria-label="Uploading CSV" />}
        </Stack>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'flex-end' }} flexWrap="wrap">
          <TextField
            select
            required
            label="Input file for this run"
            value={inputId}
            onChange={(e) => setInputId(e.target.value)}
            sx={{ minWidth: 300, flex: 1, maxWidth: 560 }}
            helperText={
              inputs.length === 0
                ? 'Upload a CSV, then choose it here before starting the job.'
                : 'Select the batch to score. New uploads appear in this list; nothing is pre-selected for you.'
            }
          >
            <MenuItem value="">
              <em>Select input for this run…</em>
            </MenuItem>
            {inputsForPredSelect.map((f) => (
              <MenuItem key={f.public_id} value={f.public_id}>
                {f.original_name} · v{f.version} · {f.public_id.slice(0, 8)}…
              </MenuItem>
            ))}
          </TextField>
          <Button
            variant="contained"
            size="large"
            disabled={!modelId.trim() || !inputId.trim()}
            onClick={async () => {
              onNotify(null);
              try {
                const j = await startPrediction({
                  model_version_public_id: modelId.trim(),
                  input_file_public_id: inputId.trim(),
                  compute_shap: computeShap,
                });
                setPredJobId(j.public_id);
                setLoadedResults(null);
                setPredSummary(`Status: ${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'} (auto-refresh every 2s)`);
                setPollJob(true);
                onNotify({
                  severity: 'success',
                  text: `Prediction job ${j.public_id} — status will update until completed or failed.`,
                });
                void refreshPredictionJobsList();
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              }
            }}
          >
            Run prediction
          </Button>
        </Stack>
      </Box>

      <Accordion
        expanded={recentJobsExpanded}
        onChange={(_, expanded) => setRecentJobsExpanded(expanded)}
        disableGutters
        elevation={0}
        sx={{
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          '&:before': { display: 'none' },
        }}
      >
        <AccordionSummary expandIcon={<Iconify width={20} icon="eva:arrow-ios-downward-fill" />}>
          <Stack direction="row" alignItems="center" spacing={1} flexWrap="wrap">
            <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
              Recent prediction jobs
            </Typography>
            <Chip
              size="small"
              variant="outlined"
              label={jobsListLoading ? '…' : predictionJobsList.length}
              sx={{ height: 22 }}
            />
            <Typography variant="caption" color="text.secondary">
              {recentJobsExpanded ? 'Click to hide' : 'Click to show'}
            </Typography>
          </Stack>
        </AccordionSummary>
        <AccordionDetails sx={{ pt: 0 }}>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
            <strong>View</strong> opens job details; <strong>Use as current job</strong> fills the field below for polling /
            results. <strong>Delete</strong> removes the job and its output CSV — allowed for <strong>pending</strong> (cancel
            queued) and finished jobs; not while <strong>running</strong>. <strong>Delete all pending jobs</strong> clears every
            pending row on the server.
          </Typography>
          <TableContainer sx={{ maxWidth: 1, overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell>public_id</TableCell>
                  <TableCell>status</TableCell>
                  <TableCell align="right">rows</TableCell>
                  <TimeSortHeadCell label="Created" order={predJobsTimeOrder} onOrderChange={setPredJobsTimeOrder} />
                  <TableCell align="right">actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {predictionJobsList.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5}>
                      <Typography variant="body2" color="text.secondary">
                        No jobs yet — run a prediction above or refresh.
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  predictionJobsSorted.map((j) => (
                    <TableRow key={j.public_id} hover>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 12, maxWidth: 220, wordBreak: 'break-all' }}>
                        {j.public_id}
                      </TableCell>
                      <TableCell>{j.status}</TableCell>
                      <TableCell align="right">{j.rows_total ?? '—'}</TableCell>
                      <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(j.created_at)}</TableCell>
                      <TableCell align="right">
                        <Stack direction="row" spacing={0.5} justifyContent="flex-end" flexWrap="wrap" useFlexGap>
                          <Button
                            size="small"
                            variant="outlined"
                            onClick={async () => {
                              setViewJobOpen(true);
                              setViewJobDetail(null);
                              setViewJobLoading(true);
                              try {
                                const full = await getPredictionJob(j.public_id);
                                setViewJobDetail(full);
                              } catch (e) {
                                onNotify({ severity: 'error', text: formatError(e) });
                                setViewJobOpen(false);
                              } finally {
                                setViewJobLoading(false);
                              }
                            }}
                          >
                            View
                          </Button>
                          <Button
                            size="small"
                            color="error"
                            disabled={j.status === 'running'}
                            onClick={async () => {
                              if (
                                !window.confirm(
                                  j.status === 'pending'
                                    ? `Remove pending prediction job ${j.public_id}? (Cancels this queued run.)`
                                    : `Delete prediction job ${j.public_id}?`
                                )
                              ) {
                                return;
                              }
                              try {
                                await deletePredictionJob(j.public_id);
                                if (predJobId.trim() === j.public_id) {
                                  setPredJobId('');
                                  setPredSummary('');
                                  setLoadedResults(null);
                                  setPollJob(false);
                                }
                                if (viewJobDetail?.public_id === j.public_id) {
                                  setViewJobOpen(false);
                                  setViewJobDetail(null);
                                }
                                await refreshPredictionJobsList();
                                onNotify({ severity: 'success', text: `Deleted job ${j.public_id}` });
                              } catch (e) {
                                onNotify({ severity: 'error', text: formatError(e) });
                              }
                            }}
                          >
                            Delete
                          </Button>
                        </Stack>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </AccordionDetails>
      </Accordion>

      <Dialog
        open={viewJobOpen}
        onClose={() => {
          setViewJobOpen(false);
          setViewJobDetail(null);
        }}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>Prediction job</DialogTitle>
        <DialogContent dividers>
          {viewJobLoading && (
            <Stack alignItems="center" py={3}>
              <CircularProgress size={32} />
            </Stack>
          )}
          {!viewJobLoading && viewJobDetail && (
            <Stack spacing={2}>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                {viewJobDetail.public_id}
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Chip size="small" label={`Status: ${viewJobDetail.status}`} />
                <Chip size="small" variant="outlined" label={`Rows: ${viewJobDetail.rows_total ?? '—'}`} />
                <Chip size="small" variant="outlined" label={`Flagged: ${viewJobDetail.rows_flagged ?? '—'}`} />
              </Stack>
              <Typography variant="caption" color="text.secondary">
                Updated {fDateTime(viewJobDetail.updated_at)}
              </Typography>
              {viewJobDetail.error_message && (
                <Alert severity="error" variant="outlined">
                  {viewJobDetail.error_message}
                </Alert>
              )}
              {viewJobDetail.output_path && (
                <Typography variant="caption" color="text.secondary">
                  Output: {viewJobDetail.output_path}
                </Typography>
              )}
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                <Button
                  size="small"
                  variant="outlined"
                  disabled={viewJobLoading}
                  onClick={async () => {
                    if (!viewJobDetail) return;
                    setViewJobLoading(true);
                    try {
                      const full = await getPredictionJob(viewJobDetail.public_id, { includeResults: true });
                      setViewJobDetail(full);
                    } catch (e) {
                      onNotify({ severity: 'error', text: formatError(e) });
                    } finally {
                      setViewJobLoading(false);
                    }
                  }}
                >
                  Reload with per-row results &amp; SHAP
                </Button>
              </Stack>
              {viewJobDetail.results_json != null && (
                <Box
                  component="pre"
                  sx={{
                    m: 0,
                    p: 1.5,
                    maxHeight: 360,
                    overflow: 'auto',
                    typography: 'caption',
                    fontFamily: 'monospace',
                    bgcolor: (t) => (t.palette.mode === 'dark' ? 'grey.900' : 'grey.100'),
                    borderRadius: 1,
                  }}
                >
                  {JSON.stringify(viewJobDetail.results_json, null, 2)}
                </Box>
              )}
            </Stack>
          )}
        </DialogContent>
        <DialogActions>
          {viewJobDetail && viewJobDetail.status !== 'running' && (
            <Button
              color="error"
              variant="outlined"
              onClick={async () => {
                if (!viewJobDetail) return;
                const id = viewJobDetail.public_id;
                if (
                  !window.confirm(
                    viewJobDetail.status === 'pending'
                      ? `Remove pending job ${id}? (Cancels this queued run.)`
                      : `Delete prediction job ${id}?`
                  )
                ) {
                  return;
                }
                try {
                  await deletePredictionJob(id);
                  if (predJobId.trim() === id) {
                    setPredJobId('');
                    setPredSummary('');
                    setLoadedResults(null);
                    setPollJob(false);
                  }
                  setViewJobOpen(false);
                  setViewJobDetail(null);
                  await refreshPredictionJobsList();
                  onNotify({ severity: 'success', text: `Removed job ${id}` });
                } catch (e) {
                  onNotify({ severity: 'error', text: formatError(e) });
                }
              }}
            >
              {viewJobDetail.status === 'pending' ? 'Remove pending job' : 'Delete job'}
            </Button>
          )}
          {viewJobDetail && (
            <Button
              variant="contained"
              color="inherit"
              onClick={() => {
                setPredJobId(viewJobDetail.public_id);
                setViewJobOpen(false);
                setViewJobDetail(null);
                onNotify({
                  severity: 'info',
                  text: `Current job set to ${viewJobDetail.public_id.slice(0, 8)}… — use Refresh / Watch / Load results below.`,
                });
              }}
            >
              Use as current job
            </Button>
          )}
          <Button
            onClick={() => {
              setViewJobOpen(false);
              setViewJobDetail(null);
            }}
          >
            Close
          </Button>
        </DialogActions>
      </Dialog>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ xs: 'stretch', md: 'center' }} flexWrap="wrap">
        <Stack spacing={1} sx={{ flex: 1, minWidth: 260 }}>
          <TextField
            select
            fullWidth
            label="Prediction job"
            value={predJobId}
            onChange={(e) => setPredJobId(e.target.value)}
            disabled={jobsListLoading}
            SelectProps={{ displayEmpty: true, MenuProps: PRED_JOB_SELECT_MENU_PROPS }}
            helperText={
              jobsListLoading
                ? 'Loading job list…'
                : `${predictionJobsList.length} job(s) in list · dropdown is all jobs from the server`
            }
          >
            <MenuItem value="">
              <em>Select or clear…</em>
            </MenuItem>
            {renderPredictionJobOrphanMenuItem(predJobId, predictionJobsList)}
            {predictionJobsSorted.map((j) => (
              <MenuItem key={j.public_id} value={j.public_id}>
                <Stack spacing={0.25} alignItems="flex-start" sx={{ py: 0.5, maxWidth: 1 }}>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                    {j.public_id}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {[
                      j.status,
                      formatPredictionModelKind(j.results_model_kind),
                      j.rows_total != null ? `${j.rows_total} rows` : null,
                      fDateTime(j.created_at),
                    ]
                      .filter((x) => x != null && x !== '')
                      .join(' · ')}
                  </Typography>
                </Stack>
              </MenuItem>
            ))}
          </TextField>
          <TextField
            fullWidth
            size="small"
            label="public_id (type or paste)"
            value={predJobId}
            onChange={(e) => setPredJobId(e.target.value)}
            helperText="Same field as the dropdown — use either control."
          />
        </Stack>
        <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ flexShrink: 0 }}>
          <Button
            variant="outlined"
            onClick={async () => {
              if (!predJobId.trim()) return;
              onNotify(null);
              try {
                const j = await getPredictionJob(predJobId.trim());
                lastStatusRef.current = j.status;
                setPredSummary(`Status: ${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'}`);
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              }
            }}
          >
            Refresh job
          </Button>
          <Button
            variant={pollJob ? 'contained' : 'outlined'}
            color={pollJob ? 'info' : 'inherit'}
            size="small"
            disabled={!predJobId.trim()}
            onClick={() => {
              lastStatusRef.current = null;
              setPollJob((p) => !p);
            }}
          >
            {pollJob ? 'Stop watching' : 'Watch job (poll)'}
          </Button>
          <Button
            variant="outlined"
            color="secondary"
            disabled={!predJobId.trim() || loadingResults}
            onClick={async () => {
              if (!predJobId.trim()) return;
              onNotify(null);
              setLoadingResults(true);
              try {
                const j = await getPredictionJob(predJobId.trim(), { includeResults: true });
                setPredSummary(`${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'}`);
                setLoadedResults(j.results_json ?? null);
                if (j.status === 'completed' && !j.results_json) {
                  onNotify({ severity: 'info', text: 'Job completed but no results_json yet (re-run prediction on updated API).' });
                }
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              } finally {
                setLoadingResults(false);
              }
            }}
          >
            {loadingResults ? 'Loading…' : 'Load per-row results & SHAP'}
          </Button>
        </Stack>
      </Stack>
      {predSummary && <Typography variant="body2">{predSummary}</Typography>}
      {loadedResults && (
        <Paper variant="outlined" sx={{ p: 2, borderRadius: 1.5 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
            Stored JSON (database) — {loadedResults.rows?.length ?? 0} row(s)
            {loadedResults.shap?.status && (
              <Chip
                size="small"
                sx={{ ml: 1, verticalAlign: 'middle' }}
                label={`SHAP: ${loadedResults.shap.status}`}
                variant="outlined"
              />
            )}
          </Typography>
          <Box
            component="pre"
            sx={{
              m: 0,
              p: 1.5,
              maxHeight: 420,
              overflow: 'auto',
              typography: 'caption',
              fontFamily: 'monospace',
              bgcolor: (t) => (t.palette.mode === 'dark' ? 'grey.900' : 'grey.100'),
              borderRadius: 1,
            }}
          >
            {JSON.stringify(loadedResults, null, 2)}
          </Box>
        </Paper>
      )}
    </Stack>
  );
}

function KbPanel({ onNotify }: PanelProps) {
  const [rows, setRows] = useState<Awaited<ReturnType<typeof kbListFiles>>['items']>([]);
  const [kbTotal, setKbTotal] = useState(0);
  const [kbPage, setKbPage] = useState(0);
  const [kbRowsPerPage, setKbRowsPerPage] = useState(10);
  const [kbUploading, setKbUploading] = useState(false);
  const [kbDeleting, setKbDeleting] = useState(false);
  const [kbLoading, setKbLoading] = useState(false);
  const [kbTimeOrder, setKbTimeOrder] = useState<TimeSortOrder>('desc');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());

  const allSelected = rows.length > 0 && rows.every((r) => selectedIds.has(r.public_id));
  const someSelected = selectedIds.size > 0;

  const load = useCallback(async () => {
    setKbLoading(true);
    try {
      const res = await kbListFiles({
        page: kbPage + 1,
        pageSize: kbRowsPerPage,
        order: kbTimeOrder,
      });
      const items = res.items.filter((r) => !isPipelineKbArtifactName(r.original_name));
      setRows(items);
      setKbTotal(res.total);
      setSelectedIds((prev) => {
        const next = new Set<string>();
        for (const id of prev) {
          if (items.some((r) => r.public_id === id)) next.add(id);
        }
        return next;
      });
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setKbLoading(false);
    }
  }, [onNotify, kbPage, kbRowsPerPage, kbTimeOrder]);

  useEffect(() => {
    load();
  }, [load]);

  const toggleOne = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (allSelected) {
      setSelectedIds(new Set());
      return;
    }
    setSelectedIds(new Set(rows.map((r) => r.public_id)));
  };

  const deleteSelected = async () => {
    const ids = [...selectedIds];
    if (ids.length === 0) return;
    const labels = ids.map((id) => {
      const row = rows.find((r) => r.public_id === id);
      return row?.original_name?.trim() || id.slice(0, 8);
    });
    if (
      !window.confirm(
        ids.length === 1
          ? `Delete knowledge file “${labels[0]}”? This removes the document and its vector index.`
          : `Delete ${ids.length} knowledge files?\n\n${labels.slice(0, 8).join('\n')}${labels.length > 8 ? '\n…' : ''}`
      )
    ) {
      return;
    }
    onNotify(null);
    setKbDeleting(true);
    let ok = 0;
    const errors: string[] = [];
    try {
      for (const id of ids) {
        try {
          await kbDelete(id);
          ok += 1;
        } catch (e) {
          errors.push(`${id.slice(0, 8)}…: ${formatError(e)}`);
        }
      }
      await load();
      setSelectedIds(new Set());
      if (errors.length === 0) {
        onNotify({ severity: 'success', text: `Deleted ${ok} knowledge file(s).` });
      } else {
        onNotify({
          severity: 'error',
          text: `Deleted ${ok}; ${errors.length} failed. ${errors[0]}`,
        });
      }
    } finally {
      setKbDeleting(false);
    }
  };

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    onNotify(null);
    setKbUploading(true);
    try {
      const r = await kbUpload(file);
      onNotify({
        severity: 'success',
        text: `Indexed “${file.name}” · ${r.chunk_count} chunks`,
      });
      setKbPage(0);
      await load();
      setKbPage(0);
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setKbUploading(false);
    }
  };

  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button variant="contained" component="label" disabled={kbUploading || kbDeleting || kbLoading}>
          {kbUploading ? 'Uploading…' : 'Upload KB document'}
          <input type="file" hidden onChange={onUpload} disabled={kbUploading || kbDeleting || kbLoading} />
        </Button>
        <Button onClick={() => void load()} disabled={kbUploading || kbDeleting || kbLoading}>
          Refresh KB list
        </Button>
        <Button
          color="error"
          variant="outlined"
          disabled={!someSelected || kbUploading || kbDeleting || kbLoading}
          onClick={() => void deleteSelected()}
        >
          {kbDeleting ? 'Deleting…' : `Delete selected${someSelected ? ` (${selectedIds.size})` : ''}`}
        </Button>
        {(kbUploading || kbDeleting || kbLoading) && (
          <CircularProgress size={22} aria-label={kbUploading ? 'Uploading document' : kbDeleting ? 'Deleting documents' : 'Loading documents'} />
        )}
      </Stack>
      <TableContainer component={Paper} variant="outlined">
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell padding="checkbox">
              <Checkbox
                size="small"
                checked={allSelected}
                indeterminate={someSelected && !allSelected}
                disabled={rows.length === 0 || kbDeleting || kbLoading}
                onChange={toggleAll}
                inputProps={{ 'aria-label': 'Select all knowledge files on this page' }}
              />
            </TableCell>
            <TableCell>File name</TableCell>
            <TableCell>Chunks</TableCell>
            <TableCell>Embedding</TableCell>
            <TimeSortHeadCell
              label="Indexed"
              order={kbTimeOrder}
              onOrderChange={(next) => {
                setKbTimeOrder(next);
                setKbPage(0);
              }}
            />
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {!kbLoading && rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={6}>
                <Typography variant="body2" color="text.secondary">
                  No knowledge documents yet. Upload a PDF, TXT, MD, or JSON guide to index it for RAG.
                </Typography>
              </TableCell>
            </TableRow>
          )}
          {kbLoading && (
            <TableRow>
              <TableCell colSpan={6}>
                <Typography variant="body2" color="text.secondary">
                  Loading knowledge documents…
                </Typography>
              </TableCell>
            </TableRow>
          )}
          {!kbLoading &&
            rows.map((r) => {
            const displayName = r.original_name?.trim() || 'Untitled document';
            return (
              <TableRow key={r.public_id} hover selected={selectedIds.has(r.public_id)}>
                <TableCell padding="checkbox">
                  <Checkbox
                    size="small"
                    checked={selectedIds.has(r.public_id)}
                    disabled={kbDeleting}
                    onChange={() => toggleOne(r.public_id)}
                    inputProps={{ 'aria-label': `Select ${displayName}` }}
                  />
                </TableCell>
                <TableCell sx={{ maxWidth: 360 }}>
                  <Typography variant="body2" noWrap title={displayName} sx={{ fontWeight: 600 }}>
                    {displayName}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    noWrap
                    display="block"
                    title={r.public_id}
                    sx={{ fontFamily: 'monospace' }}
                  >
                    {r.public_id.slice(0, 8)}…
                  </Typography>
                </TableCell>
                <TableCell>{r.chunk_count}</TableCell>
                <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }} title={r.embedding_model}>
                  {r.embedding_model}
                </TableCell>
                <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(r.created_at)}</TableCell>
                <TableCell align="right">
                  <Button
                    size="small"
                    color="error"
                    disabled={kbDeleting}
                    onClick={async () => {
                      if (
                        !window.confirm(
                          `Delete knowledge file “${displayName}”? This removes the document and its vector index.`
                        )
                      ) {
                        return;
                      }
                      try {
                        await kbDelete(r.public_id);
                        onNotify({ severity: 'success', text: `Deleted “${displayName}”.` });
                        await load();
                      } catch (e) {
                        onNotify({ severity: 'error', text: formatError(e) });
                      }
                    }}
                  >
                    Delete
                  </Button>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
      <TablePagination
        component="div"
        count={kbTotal}
        page={kbPage}
        onPageChange={(_, p) => setKbPage(p)}
        rowsPerPage={kbRowsPerPage}
        onRowsPerPageChange={(e) => {
          setKbRowsPerPage(parseInt(e.target.value, 10));
          setKbPage(0);
        }}
        rowsPerPageOptions={[5, 10, 25, 50]}
        sx={{ minHeight: 44, '& .MuiTablePagination-toolbar': { minHeight: 44, pl: 1 } }}
      />
      </TableContainer>

      <Alert severity="info" variant="outlined">
        Only uploaded guides (PDF, TXT, MD, JSON) appear here. Pipeline run files such as{' '}
        <code>traffic_run_*.json</code> are not knowledge and are excluded automatically. RAG prep and multi-query retrieval
        live under <strong>RAG prep</strong>.
      </Alert>
    </Stack>
  );
}

function formatAgentShapLines(rowContext: Record<string, unknown> | null | undefined): string {
  if (!rowContext) return '';
  const agentTop = rowContext.agent_top_shap;
  if (!agentTop || typeof agentTop !== 'object') return '';
  const lines: string[] = [];
  for (const [aname, feats] of Object.entries(agentTop as Record<string, { feature?: string; shap?: number }[]>)) {
    if (!Array.isArray(feats) || feats.length === 0) continue;
    const parts = feats.map((f) => {
      const v = Number(f.shap);
      const sign = Number.isFinite(v) && v >= 0 ? '+' : '';
      return `${f.feature ?? '?'} (SHAP ${Number.isFinite(v) ? `${sign}${v.toFixed(4)}` : String(f.shap)})`;
    });
    lines.push(`${aname}: ${parts.join(', ')}`);
  }
  return lines.join('\n');
}

function ragSummaryQueryFromTemplates(templates: RAGTemplateItem[] | undefined): string | null {
  const t = templates?.find((x) => x.id === 'row_summary_rag');
  return t?.retrieval_queries?.[0] ?? null;
}

function ragTemplateById(templates: RAGTemplateItem[] | undefined, id: string): RAGTemplateItem | undefined {
  return templates?.find((x) => x.id === id);
}

/** Fuse backend multi-query lines into one paragraph (one string per track before per-query /kb/query). */
function retrievalQueriesToSingleParagraph(parts: readonly string[] | undefined): string {
  if (!parts?.length) return '';
  const bits = parts.map((p) => p.replace(/\s+/g, ' ').trim()).filter(Boolean);
  return bits.join(' ').replace(/\s+/g, ' ').trim();
}

/** Rephrase / summary-style queries from row_summary_rag (read-only backend template). */
function rephraseRetrievalQueries(templates: RAGTemplateItem[] | undefined): string[] {
  const t = ragTemplateById(templates, 'row_summary_rag');
  if (t?.retrieval_queries?.length) return [...t.retrieval_queries];
  const line = ragSummaryQueryFromTemplates(templates);
  return line ? [line] : [];
}

function llmShapRetrievalQueries(templates: RAGTemplateItem[] | undefined): string[] {
  const t = ragTemplateById(templates, 'row_agent_shap_queries');
  return t?.retrieval_queries?.length ? [...t.retrieval_queries] : [];
}

/** Track 3: SHAP/template draft (backend ``row_agent_shap_queries``); optional LLM refine via ``/kb/llm-shap-retrieval-query``. */
function buildThreeTrackRetrievalParts(opts: {
  useTemplate: boolean;
  useRephrase: boolean;
  templateParagraph: string;
  rephraseParagraph: string;
  llmShapParagraph: string;
}): { template: string; rephrase: string; llmBase: string } {
  const q1 = opts.useTemplate ? opts.templateParagraph.trim() : '';
  const q2 = opts.useRephrase ? opts.rephraseParagraph.trim() : '';
  const q3 = opts.llmShapParagraph.trim();
  return { template: q1, rephrase: q2, llmBase: q3 };
}

/** Non-empty queries only; backend accepts 1–12. Template and/or Rephrase must supply track 1 and/or 2 before submit. */
function threeTrackToApiQueries(parts: { template: string; rephrase: string; llmBase: string }): string[] {
  return [parts.template, parts.rephrase, parts.llmBase].map((s) => s.trim()).filter(Boolean);
}

/** Labels aligned 1:1 with ``threeTrackToApiQueries`` (same inclusion rules). */
function retrievalQueryTrackLabels(
  parts: { template: string; rephrase: string; llmBase: string },
  opts?: { shapTrackLlmRefined?: boolean }
): string[] {
  const labels: string[] = [];
  if (parts.template.trim()) labels.push('Template track');
  if (parts.rephrase.trim()) labels.push('Rephrase (summary) track');
  if (parts.llmBase.trim()) {
    labels.push(
      opts?.shapTrackLlmRefined
        ? 'LLM retrieval (SHAP-driven) track'
        : 'SHAP retrieval (template + data draft)'
    );
  }
  return labels;
}

function buildAgentLlmPromptFromSources(opts: {
  templates: RAGTemplateItem[] | undefined;
  selectedTemplate: RAGTemplateItem | undefined;
  useTemplate: boolean;
  useRephrase: boolean;
  /** When row SHAP template has retrieval text, include its synthesis prompt in handoff. */
  includeShapHandoff: boolean;
}): string {
  const parts: string[] = [];
  if (opts.useTemplate && opts.selectedTemplate?.llm_prompt?.trim()) {
    parts.push(`[Template: ${opts.selectedTemplate.label}]\n${opts.selectedTemplate.llm_prompt.trim()}`);
  }
  const sumT = ragTemplateById(opts.templates, 'row_summary_rag');
  if (opts.useRephrase && sumT?.llm_prompt?.trim()) {
    parts.push(`[Summary / rephrase]\n${sumT.llm_prompt.trim()}`);
  }
  const shapT = ragTemplateById(opts.templates, 'row_agent_shap_queries');
  if (opts.includeShapHandoff && shapT?.llm_prompt?.trim()) {
    parts.push(`[SHAP drivers]\n${shapT.llm_prompt.trim()}`);
  }
  return parts.join('\n\n---\n\n');
}

const MMR_PRESETS = { focused: 0.72, balanced: 0.55, diverse: 0.38 } as const;

function predictionRowMenuLabel(r: NonNullable<PredictionResultsJson['rows']>[number], i: number): string {
  const idx = r.row_index ?? i;
  const p =
    typeof r.max_class_probability === 'number' && Number.isFinite(r.max_class_probability)
      ? `${(r.max_class_probability * 100).toFixed(1)}%`
      : '—';
  const flag = r.flagged_attack_or_anomaly ? ' · flagged' : '';
  return `Row ${idx}: ${r.predicted_label} · max P=${p}${flag}`;
}

const PRED_JOB_SELECT_MENU_PROPS = { PaperProps: { style: { maxHeight: 420 } } } as const;

function formatPredictionModelKind(kind: string | null | undefined): string | null {
  if (kind == null || !String(kind).trim()) return null;
  const k = String(kind).trim();
  if (k === 'sklearn_pipeline') return 'sklearn pipeline';
  if (k === 'vfl_torch') return 'VFL torch';
  return k;
}

function latestCreatedAtIso(reports: readonly { created_at: string }[]): string | null {
  let latest: string | null = null;
  for (const r of reports) {
    if (!latest || r.created_at > latest) latest = r.created_at;
  }
  return latest;
}

/** Reports for a prediction batch + optional analyst row (same rules as the agentic job dropdown). */
function agentReportsMatchingLine(
  predictionJobPublicId: string,
  resultsRowIndex: number | null | undefined,
  reports: readonly AgenticReportOut[]
): AgenticReportOut[] {
  const pid = predictionJobPublicId.trim();
  if (!pid) return [];
  const jr = resultsRowIndex;
  const rowSpecific = jr != null && jr >= 0;
  return reports.filter((r) => {
    if (r.prediction_job_public_id?.trim() !== pid) return false;
    if (!rowSpecific) return true;
    return r.results_row_index === jr;
  });
}

/** Reports saved via POST /agent/decide with this ``agentic_jobs.public_id`` (strict; legacy reports without the id do not count). */
function agentReportsForAgenticJobLine(
  j: AgenticJobOut,
  reports: readonly AgenticReportOut[]
): AgenticReportOut[] {
  return reports.filter((r) => r.agentic_job_public_id?.trim() === j.public_id.trim());
}

function formatAgentReportsLineSummary(
  lineReports: readonly AgenticReportOut[],
  scope: 'row' | 'batch' | 'agentic_job'
): string {
  const scopeNoun =
    scope === 'agentic_job' ? 'agentic job id' : scope === 'row' ? 'handoff row' : 'prediction batch';
  if (lineReports.length === 0) {
    return scope === 'agentic_job'
      ? 'No agent reports yet linked to this agentic job id (run agent with this job selected).'
      : `No agent reports yet for this ${scopeNoun} (same scope as the dropdown).`;
  }
  const latest = latestCreatedAtIso(lineReports) ?? lineReports[0]!.created_at;
  if (lineReports.length === 1) return `1 agent report for this ${scopeNoun} · ${fDateTime(latest)}`;
  return `${lineReports.length} agent reports for this ${scopeNoun} · latest ${fDateTime(latest)}`;
}

/** Keeps MUI Select valid when ``currentId`` is set but not present in ``jobs`` (e.g. after refresh). */
function renderPredictionJobOrphanMenuItem(
  currentId: string,
  jobs: readonly PredictionJobListItem[],
  captionForOrphan?: (publicId: string) => string | undefined
) {
  const t = currentId.trim();
  if (!t || jobs.some((j) => j.public_id === t)) return null;
  const caption =
    captionForOrphan?.(t) ?? 'Not in current list — refresh job list or verify id';
  return (
    <MenuItem value={t} key="__orphan-pred-job">
      <Stack spacing={0.25} alignItems="flex-start" sx={{ py: 0.5, maxWidth: 1 }}>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
          {t}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {caption}
        </Typography>
      </Stack>
    </MenuItem>
  );
}

function agenticJobBatchCaption(j: AgenticJobOut): string {
  const parts = [`prediction ${j.prediction_status}`];
  if (j.rows_total != null && j.rows_total >= 0) {
    const fg = j.rows_flagged;
    parts.push(
      fg != null && fg >= 0 ? `${fg} flagged / ${j.rows_total} rows` : `${j.rows_total} rows`
    );
  }
  const mk = formatPredictionModelKind(j.results_model_kind);
  if (mk) parts.push(mk);
  return parts.join(' · ');
}

/** Orphan select value: handoff id not yet returned by GET /agent/jobs (refresh or re-register). */
function renderAgenticJobOrphanMenuItem(
  currentAgenticJobId: string,
  jobs: readonly AgenticJobOut[],
  captionForOrphan?: (agenticPublicId: string) => string | undefined
) {
  const t = currentAgenticJobId.trim();
  if (!t || jobs.some((j) => j.public_id === t)) return null;
  const caption =
    captionForOrphan?.(t) ?? 'Not in current list — refresh job list or register from RAG prep';
  return (
    <MenuItem value={t} key="__orphan-agentic-job">
      <Stack spacing={0.25} alignItems="flex-start" sx={{ py: 0.5, maxWidth: 1 }}>
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
          {t}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {caption}
        </Typography>
      </Stack>
    </MenuItem>
  );
}

function RagPrepSection({
  kicker,
  title,
  description,
  children,
}: {
  kicker: string;
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: { xs: 1.75, sm: 2.25 },
        borderRadius: 2,
        border: 1,
        borderColor: 'divider',
        bgcolor: (t) => alpha(t.palette.grey[500], 0.04),
      }}
    >
      <Stack spacing={1.75}>
        <Box>
          <Typography
            variant="overline"
            sx={{ color: 'primary.main', fontWeight: 800, letterSpacing: 0.7, display: 'block', lineHeight: 1.3 }}
          >
            {kicker}
          </Typography>
          <Typography variant="subtitle1" sx={{ fontWeight: 800, lineHeight: 1.3 }}>
            {title}
          </Typography>
          {description ? (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, maxWidth: 920, lineHeight: 1.6 }}>
              {description}
            </Typography>
          ) : null}
        </Box>
        <Box sx={{ '& > .MuiAlert-root:first-of-type': { mt: description ? 0 : -0.5 } }}>{children}</Box>
      </Stack>
    </Paper>
  );
}

// Legacy panel — superseded by Setting tabs; kept for reference.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function RagLlmPrepPanel({ onNotify }: PanelProps) {
  const [jobRows, setJobRows] = useState<PredictionJobListItem[]>([]);
  const [listLoading, setListLoading] = useState(false);
  const [selectedJobId, setSelectedJobId] = useState('');
  const [loadedJob, setLoadedJob] = useState<PredictionJobOut | null>(null);
  const [loadJobLoading, setLoadJobLoading] = useState(false);
  const [predCtx, setPredCtx] = useState<KBRAGLatestPredictionResponse | null>(null);
  const [templatesLoading, setTemplatesLoading] = useState(false);
  const [rowIndex, setRowIndex] = useState<number | null>(null);
  const [templateId, setTemplateId] = useState('');
  const [mmrPreset, setMmrPreset] = useState<keyof typeof MMR_PRESETS>('balanced');
  const [retrievalPipeline, setRetrievalPipeline] = useState<'fusion_mmr' | 'fusion_only'>('fusion_mmr');
  const [multiHits, setMultiHits] = useState<KBQueryHit[]>([]);
  const [multiMetaLine, setMultiMetaLine] = useState('');
  const [retrievalLoading, setRetrievalLoading] = useState(false);
  const [useRetrievalTemplate, setUseRetrievalTemplate] = useState(true);
  const [useRetrievalRephrase, setUseRetrievalRephrase] = useState(false);
  const [finalDocCount, setFinalDocCount] = useState(5);
  const [perQueryK, setPerQueryK] = useState(14);
  const [showAllRetrieved, setShowAllRetrieved] = useState(false);
  /** When set, track 3 uses this string for /kb/query instead of the raw SHAP template draft. */
  const [shapRetrievalLlmOverride, setShapRetrievalLlmOverride] = useState<string | null>(null);
  const [shapRetrievalLlmLoading, setShapRetrievalLlmLoading] = useState(false);
  const [prepSavedForAgenticAlert, setPrepSavedForAgenticAlert] = useState<string | null>(null);

  const jobRowsSorted = useMemo(
    () => sortByTime(jobRows, (j) => j.created_at, 'desc'),
    [jobRows]
  );

  const loadJobList = useCallback(async () => {
    setListLoading(true);
    try {
      setJobRows(await listAllPredictionJobs());
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setListLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void loadJobList();
  }, [loadJobList]);

  useEffect(() => {
    if (multiHits.length === 0) setShowAllRetrieved(false);
  }, [multiHits.length]);

  const fetchTemplates = useCallback(
    async (jobPublicId: string, rIdx: number | null) => {
      setTemplatesLoading(true);
      setMultiHits([]);
      setMultiMetaLine('');
      try {
        const ctx = await kbRagTemplatesPredictionJob(jobPublicId, { rowIndex: rIdx });
        setPredCtx(ctx);
        setTemplateId(ctx.templates[0]?.id ?? '');
        if (ctx.message) {
          onNotify({ severity: 'info', text: ctx.message });
        }
      } catch (e) {
        onNotify({ severity: 'error', text: formatError(e) });
      } finally {
        setTemplatesLoading(false);
      }
    },
    [onNotify]
  );

  const loadJobById = useCallback(
    async (publicId: string) => {
      const id = publicId.trim();
      if (!id) {
        setLoadedJob(null);
        setPredCtx(null);
        setRowIndex(null);
        setMultiHits([]);
        setMultiMetaLine('');
        onNotify(null);
        return;
      }
      onNotify(null);
      setLoadJobLoading(true);
      setLoadedJob(null);
      setPredCtx(null);
      setRowIndex(null);
      try {
        const job = await getPredictionJob(id, { includeResults: true });
        setLoadedJob(job);
        if (job.status !== 'completed') {
          onNotify({
            severity: 'info',
            text: 'This job is not completed; RAG templates require a completed prediction.',
          });
        }
        await fetchTemplates(id, null);
        const n = job.results_json?.rows?.length;
        onNotify({
          severity: 'success',
          text:
            n != null && n > 0
              ? `Loaded job ${job.public_id} with full results (${n} row(s)) for batch or per-row prep.`
              : `Loaded job ${job.public_id} (no rows in results_json yet — re-run prediction with SHAP or check API).`,
        });
      } catch (e) {
        onNotify({ severity: 'error', text: formatError(e) });
      } finally {
        setLoadJobLoading(false);
      }
    },
    [onNotify, fetchTemplates]
  );

  const selectedTemplate: RAGTemplateItem | undefined = predCtx?.templates?.find((t) => t.id === templateId);
  const summaryRagLine = ragSummaryQueryFromTemplates(predCtx?.templates);
  const completedJobs = jobRowsSorted.filter((j) => j.status === 'completed');
  const resultRows = loadedJob?.results_json?.rows;

  useEffect(() => {
    if (!predCtx?.templates?.length) return;
    setMultiHits([]);
    setMultiMetaLine('');
    const rq = retrievalQueriesToSingleParagraph(rephraseRetrievalQueries(predCtx.templates));
    setUseRetrievalTemplate(true);
    setUseRetrievalRephrase(Boolean(rq));
    // Intentionally keyed to job + row only so refetches with the same selection do not wipe checkbox state.
  // eslint-disable-next-line react-hooks/exhaustive-deps -- predCtx.templates read inside; omit to avoid reset on every fetch
  }, [predCtx?.prediction_job_public_id, predCtx?.row_index]);

  const templateRetrievalParagraph = useMemo(
    () => retrievalQueriesToSingleParagraph(selectedTemplate?.retrieval_queries),
    [selectedTemplate]
  );
  const rephraseParagraph = useMemo(
    () => retrievalQueriesToSingleParagraph(rephraseRetrievalQueries(predCtx?.templates)),
    [predCtx?.templates]
  );
  const llmShapParagraph = useMemo(
    () => retrievalQueriesToSingleParagraph(llmShapRetrievalQueries(predCtx?.templates)),
    [predCtx?.templates]
  );

  const effectiveLlmShapParagraph = useMemo(() => {
    const over = shapRetrievalLlmOverride?.trim();
    if (over) return over;
    return llmShapParagraph.trim();
  }, [shapRetrievalLlmOverride, llmShapParagraph]);
  const shapTrackLlmRefined = Boolean(shapRetrievalLlmOverride?.trim());

  useEffect(() => {
    setShapRetrievalLlmOverride(null);
  }, [predCtx?.prediction_job_public_id, predCtx?.row_index, llmShapParagraph]);

  const threeTrackRetrieval = useMemo(() => {
    const parts = buildThreeTrackRetrievalParts({
      useTemplate: useRetrievalTemplate,
      useRephrase: useRetrievalRephrase,
      templateParagraph: templateRetrievalParagraph,
      rephraseParagraph,
      llmShapParagraph: effectiveLlmShapParagraph,
    });
    const forApi = threeTrackToApiQueries(parts);
    const trackLabels = retrievalQueryTrackLabels(parts, { shapTrackLlmRefined });
    return { parts, forApi, trackLabels };
  }, [
    useRetrievalTemplate,
    useRetrievalRephrase,
    templateRetrievalParagraph,
    rephraseParagraph,
    effectiveLlmShapParagraph,
    shapTrackLlmRefined,
  ]);

  const templateOrRephraseMandatoryOk =
    (useRetrievalTemplate && Boolean(templateRetrievalParagraph.trim())) ||
    (useRetrievalRephrase && Boolean(rephraseParagraph.trim()));

  const resolvedRetrievalQueries = threeTrackRetrieval.forApi;

  const mergedAgentLlmPrompt = useMemo(
    () =>
      buildAgentLlmPromptFromSources({
        templates: predCtx?.templates,
        selectedTemplate,
        useTemplate: useRetrievalTemplate,
        useRephrase: useRetrievalRephrase,
        includeShapHandoff: Boolean(effectiveLlmShapParagraph.trim()),
      }),
    [
      predCtx?.templates,
      selectedTemplate,
      useRetrievalTemplate,
      useRetrievalRephrase,
      effectiveLlmShapParagraph,
    ]
  );

  const topHitsForLlm = multiHits.slice(0, Math.max(1, Math.min(30, finalDocCount)));

  const onRowSelectionChange = async (value: string) => {
    const id = loadedJob?.public_id;
    if (!id) return;
    const rIdx = value === '' ? null : Number(value);
    setRowIndex(Number.isFinite(rIdx) ? rIdx : null);
    setMultiHits([]);
    setMultiMetaLine('');
    await fetchTemplates(id, Number.isFinite(rIdx) ? rIdx : null);
  };

  return (
    <Stack spacing={2.5}>
      <RagPrepSection
        kicker="Overview"
        title="RAG & LLM prep"
        description="Load a completed prediction job, set row scope, configure retrieval tracks, run vector fusion, then hand hits to Agentic actions."
      >
        <Stack spacing={1.5}>
          <Alert severity="info" variant="outlined">
            Choose a <strong>prediction job</strong> by <strong>public_id</strong> — the app loads{' '}
            <strong>full results for all rows</strong> (<code>include_results</code>). Use <strong>All rows</strong> for batch
            templates, or pick one row for SHAP-aware queries. <strong>Reload all row results</strong> refreshes the payload
            without changing the selection.
          </Alert>

          <Accordion
            defaultExpanded={false}
            disableGutters
            elevation={0}
            sx={{ border: 1, borderColor: 'divider', borderRadius: 1, bgcolor: 'background.paper', '&:before': { display: 'none' } }}
          >
            <AccordionSummary expandIcon={<Iconify width={20} icon="eva:arrow-ios-downward-fill" />}>
              <Typography variant="subtitle2">Enterprise network domains (Access · Perimeter · Endpoint)</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Stack spacing={1.5}>
                <Typography variant="body2" color="text.secondary">
                  Use this mental model when reading SHAP parties and RAG hits: <strong>Access / ISP</strong> is
                  subscriber/ISP-edge volume and rate telemetry; <strong>Perimeter / IDS</strong> is north-south
                  inspection (ports, WAF, scans); <strong>Endpoint / EDR</strong> is host-proximate bidirectional and
                  reverse-path forensics.
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  After you load a <strong>completed prediction job</strong>, the API adds a template pack{' '}
                  <strong>Enterprise network (Access · Perimeter · Endpoint)</strong> with retrieval strings and an LLM
                  synthesis prompt aligned to that batch. Row-level SHAP templates use the same domain framing for SOC
                  triage.
                </Typography>
              </Stack>
            </AccordionDetails>
          </Accordion>
        </Stack>
      </RagPrepSection>

      <RagPrepSection
        kicker="Step 1"
        title="Prediction job"
        description="Select a job from the list or paste an orphan id. Refresh the list after external runs; delete only when you intend to remove stored results."
      >
      <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
        <Button size="small" variant="outlined" onClick={() => void loadJobList()} disabled={listLoading}>
          {listLoading ? 'Loading jobs…' : 'Refresh job list'}
        </Button>
        <Button
          size="small"
          variant="outlined"
          color="error"
          disabled={!selectedJobId.trim() || loadJobLoading || templatesLoading}
          onClick={async () => {
            const id = selectedJobId.trim();
            if (!id) return;
            if (!window.confirm(`Delete prediction job ${id}? This clears RAG prep selection if it was this job.`)) return;
            try {
              await deletePredictionJob(id);
              if (selectedJobId === id) {
                setSelectedJobId('');
                setLoadedJob(null);
                setPredCtx(null);
                setRowIndex(null);
                setMultiHits([]);
                setMultiMetaLine('');
              }
              await loadJobList();
              onNotify({ severity: 'success', text: `Deleted prediction job ${id}` });
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Delete selected prediction job
        </Button>
      </Stack>

      <TextField
        select
        fullWidth
        label="Prediction job (public_id)"
        value={selectedJobId}
        onChange={(e) => {
          const v = e.target.value;
          setSelectedJobId(v);
          void loadJobById(v);
        }}
        SelectProps={{ displayEmpty: true, MenuProps: PRED_JOB_SELECT_MENU_PROPS }}
        helperText={`${jobRows.length} prediction job(s) (${completedJobs.length} completed) · dropdown: newest first · choosing a job loads full results_json (all rows) for batch or row-level RAG`}
      >
        <MenuItem value="">
          <em>Select a prediction job</em>
        </MenuItem>
        {renderPredictionJobOrphanMenuItem(selectedJobId, jobRows)}
        {jobRowsSorted.map((j) => (
          <MenuItem key={j.public_id} value={j.public_id}>
            <Stack spacing={0.25} alignItems="flex-start" sx={{ py: 0.5, maxWidth: 1 }}>
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                {j.public_id}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {[
                  j.status,
                  formatPredictionModelKind(j.results_model_kind),
                  j.rows_total != null ? `${j.rows_total} rows (manifest)` : null,
                ]
                  .filter((x) => x != null && x !== '')
                  .join(' · ')}
              </Typography>
            </Stack>
          </MenuItem>
        ))}
      </TextField>

      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button
          variant="outlined"
          disabled={!selectedJobId.trim() || loadJobLoading}
          onClick={() => void loadJobById(selectedJobId)}
        >
          {loadJobLoading ? 'Loading…' : 'Reload all row results from server'}
        </Button>
        {(loadJobLoading || templatesLoading) && <CircularProgress size={22} aria-label="Loading job or templates" />}
      </Stack>
      </RagPrepSection>

      {loadedJob && (
        <RagPrepSection
          kicker="Step 2"
          title="Row scope"
          description="Use job-level templates for the whole batch, or select one row to unlock SHAP-aware retrieval drafts and rephrase tracks."
        >
          <TextField
            select
            fullWidth
            label="Prediction row (for per-agent SHAP RAG templates)"
            value={rowIndex === null ? '' : String(rowIndex)}
            onChange={(e) => void onRowSelectionChange(e.target.value)}
            disabled={!resultRows?.length}
            helperText={
              resultRows?.length
                ? 'All rows: job-level templates only. Pick one row for top-3 SHAP RAG strings per agent (full results already loaded).'
                : 'No rows in results_json — re-run prediction with SHAP or use “Reload all row results from server”.'
            }
          >
            <MenuItem value="">All rows — batch / job-level templates</MenuItem>
            {resultRows?.map((r, i) => (
              <MenuItem key={i} value={String(i)}>
                {predictionRowMenuLabel(r, i)}
              </MenuItem>
            ))}
          </TextField>
        </RagPrepSection>
      )}

      {predCtx?.templates && predCtx.templates.length > 0 && (
        <RagPrepSection
          kicker="Step 3"
          title="Templates, retrieval & fusion"
          description="Pick a template pack, enable template and/or rephrase tracks (and optional SHAP track 3), tune fusion/MMR, then submit to the KB stack."
        >
          <TextField
            select
            fullWidth
            label="RAG template pack (main retrieval queries)"
            value={templateId}
            onChange={(e) => {
              setTemplateId(e.target.value);
              setMultiHits([]);
              setMultiMetaLine('');
            }}
            helperText={selectedTemplate?.description ?? ' '}
          >
            {predCtx.templates.map((t) => (
              <MenuItem key={t.id} value={t.id}>
                {t.label}
              </MenuItem>
            ))}
          </TextField>

          {selectedTemplate && (
            <>
              <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                Retrieval — <strong>Template</strong> and/or <strong>Rephrase</strong> is required. With a per-row SHAP template pack,
                the <strong>third</strong> <code>/kb/query</code> uses a <strong>SHAP draft</strong> built from template + row data; use{' '}
                <strong>Refine track 3 with LLM</strong> to rewrite that draft into a dense vector-search query (prompt includes the
                draft and the SHAP template&apos;s synthesis instructions). Up to three queries, then <code>/kb/fuse-hits-mmr</code>.
              </Typography>
              {!useRetrievalTemplate && !useRetrievalRephrase && (
                <Alert severity="warning" variant="outlined" sx={{ py: 0.5 }}>
                  Select <strong>Template</strong> and/or <strong>Rephrase</strong> before submitting to RAG.
                </Alert>
              )}
              <Stack direction="row" spacing={2} flexWrap="wrap" alignItems="center">
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={useRetrievalTemplate}
                      disabled={!templateRetrievalParagraph}
                      onChange={(e) => {
                        setUseRetrievalTemplate(e.target.checked);
                        setMultiHits([]);
                        setMultiMetaLine('');
                      }}
                    />
                  }
                  label="Template (selected pack)"
                />
                <FormControlLabel
                  control={
                    <Checkbox
                      size="small"
                      checked={useRetrievalRephrase}
                      disabled={!rephraseParagraph}
                      onChange={(e) => {
                        setUseRetrievalRephrase(e.target.checked);
                        setMultiHits([]);
                        setMultiMetaLine('');
                      }}
                    />
                  }
                  label="Rephrase (row summary template)"
                />
              </Stack>
              {!rephraseParagraph && (
                <Typography variant="caption" color="text.secondary" display="block">
                  Rephrase is available after you pick a prediction row with row-level templates.
                </Typography>
              )}
              {!llmShapParagraph && (
                <Typography variant="caption" color="text.secondary" display="block">
                  Pick a row with SHAP context to enable track 3: a human-readable draft from the SHAP template pack, optionally
                  refined by <code>/kb/llm-shap-retrieval-query</code> before vector search.
                </Typography>
              )}

              {Boolean(llmShapParagraph.trim()) && (
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    3 · SHAP-driven retrieval (draft → optional LLM refine)
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                    Draft strings are fused from <code>row_agent_shap_queries</code> (template + row/SHAP). The LLM sees this draft plus
                    the template&apos;s <code>llm_prompt</code> so the refined query matches your later RAG synthesis task.
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap', mb: 1.5 }}>
                    {llmShapParagraph}
                  </Typography>
                  <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }} flexWrap="wrap">
                    <Button
                      size="small"
                      variant="outlined"
                      disabled={shapRetrievalLlmLoading}
                      onClick={async () => {
                        const draft = llmShapParagraph.trim();
                        if (!draft) return;
                        const shapT = ragTemplateById(predCtx?.templates, 'row_agent_shap_queries');
                        setShapRetrievalLlmLoading(true);
                        try {
                          const res = await kbLlmShapRetrievalQuery({
                            draft_queries_text: draft,
                            analyst_synthesis_prompt: shapT?.llm_prompt?.trim() || null,
                          });
                          setShapRetrievalLlmOverride(res.retrieval_query.trim());
                          setMultiHits([]);
                          setMultiMetaLine('');
                          onNotify({
                            severity: res.used_llm ? 'success' : 'info',
                            text: res.used_llm
                              ? 'Track 3 updated with LLM-refined retrieval query.'
                              : 'OpenAI API key not configured — using normalized draft as track 3 (no LLM).',
                          });
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        } finally {
                          setShapRetrievalLlmLoading(false);
                        }
                      }}
                    >
                      {shapRetrievalLlmLoading ? 'Refining…' : 'Refine track 3 with LLM'}
                    </Button>
                    {shapTrackLlmRefined && (
                      <Button
                        size="small"
                        color="inherit"
                        onClick={() => {
                          setShapRetrievalLlmOverride(null);
                          setMultiHits([]);
                          setMultiMetaLine('');
                        }}
                      >
                        Use draft only
                      </Button>
                    )}
                  </Stack>
                  {shapTrackLlmRefined && (
                    <Typography variant="body2" sx={{ mt: 1.5, whiteSpace: 'pre-wrap', fontFamily: 'monospace', fontSize: 12 }}>
                      <Typography component="span" variant="caption" fontWeight={700} display="block" color="text.primary">
                        Query sent for track 3 (LLM-refined)
                      </Typography>
                      {effectiveLlmShapParagraph}
                    </Typography>
                  )}
                </Paper>
              )}

              {useRetrievalTemplate && (
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    1 · Template — single retrieval paragraph
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap' }}>
                    {templateRetrievalParagraph || '—'}
                  </Typography>
                </Paper>
              )}

              {useRetrievalRephrase && Boolean(rephraseParagraph) && (
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    2 · Rephrase (summary-style) — single retrieval paragraph
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap' }}>
                    {rephraseParagraph}
                  </Typography>
                </Paper>
              )}

              <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.neutral' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Three tracks (non-empty only) → one <code>/kb/query</code> each → <code>/kb/fuse-hits-mmr</code>
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  {resolvedRetrievalQueries.length} non-empty quer
                  {resolvedRetrievalQueries.length === 1 ? 'y' : 'ies'} · final_k=
                  {Math.min(30, Math.max(1, finalDocCount))} · per_query_k={Math.min(50, Math.max(4, perQueryK))} ·{' '}
                  {retrievalPipeline === 'fusion_mmr'
                    ? `MMR on (λ=${MMR_PRESETS[mmrPreset]})`
                    : 'MMR off (CrossEncoder / fusion only)'}
                </Typography>
                <Box component="ol" sx={{ m: 0, pl: 2.5, typography: 'body2', color: 'text.secondary' }}>
                  {resolvedRetrievalQueries.length === 0 ? (
                    <li>—</li>
                  ) : (
                    resolvedRetrievalQueries.map((q, i) => (
                      <li key={i} style={{ marginBottom: 10 }}>
                        <Typography component="span" variant="caption" fontWeight={700} display="block" color="text.primary">
                          {threeTrackRetrieval.trackLabels[i] ?? `Query ${i + 1}`}
                        </Typography>
                        <Typography
                          variant="body2"
                          sx={{
                            mt: 0.5,
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            fontFamily: 'monospace',
                            fontSize: 12,
                            pl: 0.5,
                            borderLeft: 2,
                            borderColor: 'divider',
                          }}
                        >
                          {q}
                        </Typography>
                      </li>
                    ))
                  )}
                </Box>
                {!templateOrRephraseMandatoryOk && (
                  <Typography variant="caption" color="warning.main" display="block" sx={{ mt: 1 }}>
                    Turn on Template and/or Rephrase with content to enable retrieval.
                  </Typography>
                )}
              </Paper>

              <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                Retrieval options (before running vector RAG)
              </Typography>
              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} flexWrap="wrap" alignItems={{ sm: 'flex-start' }}>
                <TextField
                  type="number"
                  label="Top documents (final_k)"
                  value={finalDocCount}
                  onChange={(e) => setFinalDocCount(Math.min(30, Math.max(1, Number(e.target.value) || 5)))}
                  inputProps={{ min: 1, max: 30 }}
                  sx={{ width: 180 }}
                  helperText="Returned after fusion + optional MMR (e.g. 5)."
                />
                <TextField
                  type="number"
                  label="Per-query depth"
                  value={perQueryK}
                  onChange={(e) => setPerQueryK(Math.min(50, Math.max(4, Number(e.target.value) || 14)))}
                  inputProps={{ min: 4, max: 50 }}
                  sx={{ width: 180 }}
                  helperText="FAISS depth per query before fusion."
                />
                <TextField
                  select
                  label="Fusion + MMR"
                  value={retrievalPipeline}
                  onChange={(e) => setRetrievalPipeline(e.target.value as 'fusion_mmr' | 'fusion_only')}
                  sx={{ minWidth: 220 }}
                  helperText="Rerank fusion; enable MMR for diverse top chunks."
                >
                  <MenuItem value="fusion_mmr">CrossEncoder + MMR</MenuItem>
                  <MenuItem value="fusion_only">CrossEncoder only (no MMR)</MenuItem>
                </TextField>
                <TextField
                  select
                  label="MMR balance"
                  value={mmrPreset}
                  onChange={(e) => setMmrPreset(e.target.value as keyof typeof MMR_PRESETS)}
                  sx={{ minWidth: 200 }}
                  disabled={retrievalPipeline !== 'fusion_mmr'}
                  helperText="Only when MMR is on."
                >
                  <MenuItem value="focused">Focused (relevance)</MenuItem>
                  <MenuItem value="balanced">Balanced</MenuItem>
                  <MenuItem value="diverse">Diverse</MenuItem>
                </TextField>
              </Stack>

              <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }} flexWrap="wrap">
                <Button
                  variant="contained"
                  disabled={
                    retrievalLoading ||
                    !templateOrRephraseMandatoryOk ||
                    resolvedRetrievalQueries.length === 0
                  }
                  onClick={async () => {
                    if (!templateOrRephraseMandatoryOk || resolvedRetrievalQueries.length === 0) return;
                    onNotify(null);
                    setPrepSavedForAgenticAlert(null);
                    setRetrievalLoading(true);
                    try {
                      const useMmr = retrievalPipeline === 'fusion_mmr';
                      const fk = Math.min(30, Math.max(1, finalDocCount));
                      const pq = Math.min(50, Math.max(4, perQueryK));
                      const qs = resolvedRetrievalQueries;
                      const perQueryHits: Awaited<ReturnType<typeof kbQuery>>['hits'][] = [];
                      for (const q of qs) {
                        const one = await kbQuery({ query: q, top_k: pq });
                        perQueryHits.push(one.hits);
                      }
                      const res = await kbFuseHitsMmr({
                        queries: qs,
                        per_query_hits: perQueryHits,
                        final_k: fk,
                        mmr_lambda: MMR_PRESETS[mmrPreset],
                        use_mmr: useMmr,
                      });
                      setMultiHits(res.hits);
                      const m = res.meta as Record<string, unknown>;
                      const pipe = String(m.pipeline ?? 'fuse-hits-mmr');
                      setMultiMetaLine(
                        `${qs.length}× /kb/query (top_k=${pq}) → /kb/fuse-hits-mmr · ${pipe} · fused ${String(m.candidates_fused ?? '?')} → pool ${String(m.pool_size ?? '?')} → ${res.hits.length} docs` +
                          (useMmr
                            ? ` (MMR λ=${String(m.mmr_lambda ?? '')}); top ${Math.min(fk, res.hits.length)} for agentic cap`
                            : ` (no MMR); top ${Math.min(fk, res.hits.length)} for agentic cap`)
                      );
                    } catch (e) {
                      onNotify({ severity: 'error', text: formatError(e) });
                    } finally {
                      setRetrievalLoading(false);
                    }
                  }}
                >
                  {retrievalLoading ? 'Retrieving…' : 'Submit to RAG (per-query /kb/query → /kb/fuse-hits-mmr)'}
                </Button>
                {(predCtx?.prediction_job_public_id?.trim() || selectedJobId.trim()) && (
                  <Button
                    variant="outlined"
                    color="primary"
                    onClick={() => {
                      void (async () => {
                        const jid = predCtx?.prediction_job_public_id?.trim() || selectedJobId.trim();
                        if (!jid) return;
                        const r = predCtx?.row_index ?? rowIndex;
                        try {
                          upsertAgenticJobHandoff(jid, r);
                          const aj = await createAgenticJob({
                            prediction_job_public_id: jid,
                            results_row_index: r != null && r >= 0 ? r : null,
                            label: selectedTemplate?.label ?? null,
                          });
                          const p = readAgenticPrep();
                          if (p) {
                            writeAgenticPrep({
                              ...p,
                              agenticJobPublicId: aj.public_id,
                              updatedAt: new Date().toISOString(),
                            });
                          }
                          const msg =
                            'Agentic job saved to the database (agentic_jobs) — refresh Agentic actions or open that tab to load GET /agent/jobs.';
                          setPrepSavedForAgenticAlert(msg);
                          onNotify({ severity: 'success', text: msg });
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      })();
                    }}
                  >
                    Set agentic job for handoff
                  </Button>
                )}
                {topHitsForLlm.length > 0 && predCtx?.prediction_job_public_id && selectedTemplate && (
                  <Button
                    variant="outlined"
                    color="success"
                    onClick={() => {
                      void (async () => {
                        const pjid = predCtx.prediction_job_public_id;
                        if (!pjid) return;
                        const rc = predCtx.row_context as Record<string, unknown> | null | undefined;
                        const handoffPrompt =
                          mergedAgentLlmPrompt.trim() || selectedTemplate.llm_prompt;
                        const rowIx = predCtx.row_index ?? rowIndex;
                        try {
                          const aj = await createAgenticJob({
                            prediction_job_public_id: pjid,
                            results_row_index: rowIx != null && rowIx >= 0 ? rowIx : null,
                            label: selectedTemplate.label,
                          });
                          writeAgenticPrep({
                            predictionJobPublicId: pjid,
                            rowIndex: rowIx,
                            rowPredictedLabel: (rc?.predicted_label as string) ?? null,
                            rowFlagged:
                              typeof rc?.flagged_attack_or_anomaly === 'boolean'
                                ? rc.flagged_attack_or_anomaly
                                : null,
                            templateId: selectedTemplate.id,
                            templateLabel: selectedTemplate.label,
                            llmPrompt: handoffPrompt,
                            summaryRagQueryText: summaryRagLine,
                            citations: topHitsForLlm,
                            retrievalMetaLine: multiMetaLine,
                            retrievalPipeline,
                            shapAgentLines: formatAgentShapLines(rc),
                            llmRagAnswer: null,
                            updatedAt: new Date().toISOString(),
                            retrievalQueriesUsed: resolvedRetrievalQueries,
                            finalDocCount: topHitsForLlm.length,
                            retrievalQuerySources: {
                              template: useRetrievalTemplate,
                              rephrase: useRetrievalRephrase,
                              llmShap: Boolean(effectiveLlmShapParagraph.trim()),
                            },
                            agenticJobPublicId: aj.public_id,
                          });
                          const savedMsg = `Saved top ${topHitsForLlm.length} KB chunk(s) and registered agentic job ${aj.public_id.slice(0, 8)}…`;
                          setPrepSavedForAgenticAlert(savedMsg);
                          onNotify({
                            severity: 'success',
                            text: savedMsg,
                          });
                        } catch (e) {
                          onNotify({ severity: 'error', text: formatError(e) });
                        }
                      })();
                    }}
                  >
                    Save for agentic job (top {topHitsForLlm.length})
                  </Button>
                )}
              </Stack>
              {prepSavedForAgenticAlert && (
                <Alert
                  severity="success"
                  variant="outlined"
                  onClose={() => setPrepSavedForAgenticAlert(null)}
                  sx={{ mt: 2 }}
                >
                  {prepSavedForAgenticAlert}
                </Alert>
              )}
            </>
          )}
        </RagPrepSection>
      )}

      {selectedTemplate && multiHits.length > 0 && (
        <RagPrepSection
          kicker="Step 4"
          title="Retrieval results"
          description="Review fused top chunks, scores, and excerpts before saving citations for Agentic actions."
        >
        <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.paper' }}>
          <Typography variant="subtitle1" gutterBottom>
            Results · top documents for agentic handoff
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1.5 }}>
            Template pack: <strong>{selectedTemplate.label}</strong> — {selectedTemplate.description}. KB synthesis LLM runs under{' '}
            <strong>Agentic</strong> after you save prep.
          </Typography>

          <Box sx={{ width: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Top {topHitsForLlm.length} document(s) (cap {finalDocCount}) —{' '}
              {retrievalPipeline === 'fusion_mmr' ? 'CrossEncoder + MMR' : 'CrossEncoder only'}
            </Typography>
            {multiMetaLine && (
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                {multiMetaLine}
              </Typography>
            )}
            <TableContainer sx={{ maxHeight: 320, border: 1, borderColor: 'divider', borderRadius: 1 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>#</TableCell>
                    <TableCell>sim</TableCell>
                    <TableCell>rerank</TableCell>
                    <TableCell>MMR</TableCell>
                    <TableCell>source</TableCell>
                    <TableCell>excerpt</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {topHitsForLlm.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={6}>
                        <Typography variant="body2" color="text.secondary">
                          Run RAG retrieval above to populate this table.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    topHitsForLlm.map((h, i) => (
                      <TableRow key={`${h.text.slice(0, 40)}-${i}`}>
                        <TableCell>{i + 1}</TableCell>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>{h.score.toFixed(3)}</TableCell>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>
                          {h.rerank_score != null ? h.rerank_score.toFixed(3) : '—'}
                        </TableCell>
                        <TableCell sx={{ whiteSpace: 'nowrap' }}>
                          {h.mmr_margin != null ? h.mmr_margin.toFixed(3) : '—'}
                        </TableCell>
                        <TableCell sx={{ maxWidth: 100 }}>{h.source ?? '—'}</TableCell>
                        <TableCell sx={{ maxWidth: 280, typography: 'caption' }}>
                          {h.text.slice(0, 220)}
                          {h.text.length > 220 ? '…' : ''}
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            {multiHits.length > topHitsForLlm.length && (
              <Box sx={{ mt: 1 }}>
                <Button size="small" onClick={() => setShowAllRetrieved((v) => !v)} sx={{ mb: 1 }}>
                  {showAllRetrieved ? 'Hide' : 'Show'} all {multiHits.length} retrieved (fused pool)
                </Button>
                <Collapse in={showAllRetrieved}>
                  <TableContainer sx={{ maxHeight: 240, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                    <Table size="small">
                      <TableBody>
                        {multiHits.map((h, i) => (
                          <TableRow key={`all-${i}`}>
                            <TableCell sx={{ width: 40 }}>{i + 1}</TableCell>
                            <TableCell sx={{ typography: 'caption' }}>{h.text.slice(0, 160)}…</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Collapse>
              </Box>
            )}
          </Box>
        </Paper>
        </RagPrepSection>
      )}
    </Stack>
  );
}

/** Resolve POST /agent/decide ids from persisted agentic job row and/or session prep. */
function resolveAgenticDecideContext(
  p: AgenticPrepPayload | null,
  rows: readonly AgenticJobOut[],
  selectedAgenticJobId: string
): { jid: string; ri: number | null; predictionStatus: JobStatus | null } {
  const sel = rows.find((j) => j.public_id.trim() === selectedAgenticJobId.trim());
  const jid = sel?.prediction_job_public_id?.trim() || p?.predictionJobPublicId?.trim() || '';
  let ri: number | null = null;
  if (sel?.results_row_index != null && sel.results_row_index >= 0) ri = sel.results_row_index;
  else if (p?.predictionJobPublicId?.trim() === jid && p.rowIndex != null && p.rowIndex >= 0)
    ri = p.rowIndex;
  const predictionStatus: JobStatus | null =
    sel?.prediction_status ??
    (jid
      ? rows.find((j) => j.prediction_job_public_id.trim() === jid)?.prediction_status ?? null
      : null);
  return { jid, ri, predictionStatus };
}

// Legacy panel — superseded by Setting tabs; kept for reference.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function AgenticActionsPanel({ onNotify }: PanelProps) {
  const [prep, setPrep] = useState(() => readAgenticPrep());
  const [selectedAgenticJobPublicId, setSelectedAgenticJobPublicId] = useState(
    () => readAgenticPrep()?.agenticJobPublicId?.trim() ?? ''
  );
  const [agenticJobsRows, setAgenticJobsRows] = useState<AgenticJobOut[]>([]);
  const [agentReportsList, setAgentReportsList] = useState<AgenticReportOut[]>([]);
  const [agenticJobsLoading, setAgenticJobsLoading] = useState(false);
  const [deletingReportId, setDeletingReportId] = useState<string | null>(null);
  const [reportDetailRow, setReportDetailRow] = useState<AgenticReportOut | null>(null);
  const [running, setRunning] = useState(false);
  const [runningTrust, setRunningTrust] = useState(false);
  const [orchestrationPrompt, setOrchestrationPrompt] = useState('');
  const [orchestrationPromptLoading, setOrchestrationPromptLoading] = useState(false);
  const [reportsTableTimeOrder, setReportsTableTimeOrder] = useState<TimeSortOrder>('desc');

  const loadAgenticJobsList = useCallback(async () => {
    setAgenticJobsLoading(true);
    try {
      const [jobs, reports] = await Promise.all([listAllAgenticJobs(), listAllAgentReports()]);
      setAgenticJobsRows(jobs);
      setAgentReportsList(reports);
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setAgenticJobsLoading(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void loadAgenticJobsList();
  }, [loadAgenticJobsList]);

  const refreshPrep = useCallback(() => {
    setPrep(readAgenticPrep());
  }, []);

  /** When RAG prep is saved again, follow its registered agentic job row in the dropdown. */
  useEffect(() => {
    const aid = prep?.agenticJobPublicId?.trim();
    if (aid) setSelectedAgenticJobPublicId(aid);
  }, [prep?.updatedAt, prep?.agenticJobPublicId]);

  useEffect(() => {
    const onHandoffUpdated = () => {
      const next = readAgenticPrep();
      setPrep(next);
      const aid = next?.agenticJobPublicId?.trim();
      if (aid) setSelectedAgenticJobPublicId(aid);
    };
    window.addEventListener(AGENTIC_PREP_UPDATED_EVENT, onHandoffUpdated);
    return () => window.removeEventListener(AGENTIC_PREP_UPDATED_EVENT, onHandoffUpdated);
  }, []);

  const hasCitations = (prep?.citations?.length ?? 0) > 0;
  const jobFromPrep = prep?.predictionJobPublicId?.trim() ?? '';

  const decideContext = useMemo(
    () => resolveAgenticDecideContext(prep, agenticJobsRows, selectedAgenticJobPublicId),
    [prep, agenticJobsRows, selectedAgenticJobPublicId]
  );
  const effectivePredictionJobId = decideContext.jid;
  const effectiveJobId = effectivePredictionJobId;

  const selectedAgenticDbRow = useMemo(
    () => agenticJobsRows.find((j) => j.public_id.trim() === selectedAgenticJobPublicId.trim()),
    [agenticJobsRows, selectedAgenticJobPublicId]
  );

  const reportsForCurrentSelection = useMemo(() => {
    if (selectedAgenticDbRow) {
      return agentReportsForAgenticJobLine(selectedAgenticDbRow, agentReportsList);
    }
    return agentReportsMatchingLine(effectivePredictionJobId, decideContext.ri, agentReportsList);
  }, [selectedAgenticDbRow, agentReportsList, effectivePredictionJobId, decideContext.ri]);

  const reportSummaryScope = useMemo((): 'row' | 'batch' | 'agentic_job' => {
    if (selectedAgenticDbRow) return 'agentic_job';
    return decideContext.ri != null && decideContext.ri >= 0 ? 'row' : 'batch';
  }, [selectedAgenticDbRow, decideContext.ri]);

  /** Saved reports table: when an agentic job is selected, show only reports linked to that id. */
  const agentReportsForTable = useMemo(() => {
    const sel = selectedAgenticJobPublicId.trim();
    if (!sel) return agentReportsList;
    return agentReportsList.filter((r) => r.agentic_job_public_id?.trim() === sel);
  }, [agentReportsList, selectedAgenticJobPublicId]);

  const agentReportsTableSorted = useMemo(
    () => sortByTime(agentReportsForTable, (r) => r.created_at, reportsTableTimeOrder),
    [agentReportsForTable, reportsTableTimeOrder]
  );

  const completedAgenticJobCount = useMemo(
    () => agenticJobsRows.filter((j) => j.prediction_status === 'completed').length,
    [agenticJobsRows]
  );

  const [onlyJobsWithNoAgentReport, setOnlyJobsWithNoAgentReport] = useState(false);

  const agenticJobsForSelect = useMemo(() => {
    if (!onlyJobsWithNoAgentReport) return agenticJobsRows;
    return agenticJobsRows.filter((j) => agentReportsForAgenticJobLine(j, agentReportsList).length === 0);
  }, [onlyJobsWithNoAgentReport, agenticJobsRows, agentReportsList]);

  const noJobsWhenFiltered = onlyJobsWithNoAgentReport && agenticJobsForSelect.length === 0;

  const selectedJobNotCompleted =
    decideContext.predictionStatus != null && decideContext.predictionStatus !== 'completed';

  useEffect(() => {
    if (!onlyJobsWithNoAgentReport) return;
    const id = selectedAgenticJobPublicId.trim();
    if (agenticJobsForSelect.length === 0) {
      if (id) setSelectedAgenticJobPublicId('');
      return;
    }
    if (id && !agenticJobsForSelect.some((j) => j.public_id === id)) {
      setSelectedAgenticJobPublicId('');
    }
  }, [onlyJobsWithNoAgentReport, agenticJobsForSelect, selectedAgenticJobPublicId]);

  useEffect(() => {
    let cancelled = false;
    const p = readAgenticPrep();
    const { jid, ri, predictionStatus } = resolveAgenticDecideContext(
      p,
      agenticJobsRows,
      selectedAgenticJobPublicId
    );
    if (!jid) {
      setOrchestrationPrompt('');
      setOrchestrationPromptLoading(false);
      return () => {
        cancelled = true;
      };
    }
    if (predictionStatus != null && predictionStatus !== 'completed') {
      setOrchestrationPrompt(
        `Prediction job is still "${predictionStatus}". Full orchestration preview needs completed results — refresh the job list when inference finishes.`
      );
      setOrchestrationPromptLoading(false);
      return () => {
        cancelled = true;
      };
    }
    setOrchestrationPromptLoading(true);
    void (async () => {
      try {
        const res = await agentDecidePromptPreview({
          prediction_job_public_id: jid,
          use_rag: true,
          results_row_index: ri,
          feature_notes: null,
          kb_citations: p?.citations?.length ? p.citations : null,
          agent_action_preset: 'standard',
        });
        if (!cancelled) setOrchestrationPrompt(res.prompt);
      } catch (e) {
        if (!cancelled) {
          setOrchestrationPrompt('');
          onNotify({ severity: 'error', text: formatError(e) });
        }
      } finally {
        if (!cancelled) setOrchestrationPromptLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [
    selectedAgenticJobPublicId,
    prep?.updatedAt,
    prep?.predictionJobPublicId,
    prep?.rowIndex,
    prep?.citations?.length,
    agenticJobsRows,
    onNotify,
  ]);

  const reloadOrchestrationPreview = useCallback(() => {
    refreshPrep();
    const p = readAgenticPrep();
    const { jid, ri, predictionStatus } = resolveAgenticDecideContext(
      p,
      agenticJobsRows,
      selectedAgenticJobPublicId
    );
    if (!jid) {
      setOrchestrationPrompt('');
      onNotify({
        severity: 'info',
        text: 'Select an agentic job or save handoff from RAG & LLM prep (includes job id).',
      });
      return;
    }
    if (predictionStatus != null && predictionStatus !== 'completed') {
      setOrchestrationPrompt(
        `Prediction job is still "${predictionStatus}". Full orchestration preview needs completed results — refresh the job list when inference finishes.`
      );
      onNotify({
        severity: 'info',
        text: `Orchestration preview needs a completed prediction job (this one is ${predictionStatus}).`,
      });
      return;
    }
    setOrchestrationPromptLoading(true);
    void (async () => {
      try {
        const res = await agentDecidePromptPreview({
          prediction_job_public_id: jid,
          use_rag: true,
          results_row_index: ri,
          feature_notes: null,
          kb_citations: p?.citations?.length ? p.citations : null,
          agent_action_preset: 'standard',
        });
        setOrchestrationPrompt(res.prompt);
        setPrep(p);
      } catch (e) {
        setOrchestrationPrompt('');
        onNotify({ severity: 'error', text: formatError(e) });
      } finally {
        setOrchestrationPromptLoading(false);
      }
    })();
  }, [onNotify, refreshPrep, selectedAgenticJobPublicId, agenticJobsRows]);

  return (
    <Stack spacing={2}>
      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack direction="row" alignItems="flex-start" justifyContent="space-between" spacing={1} sx={{ mb: 1 }}>
          <Typography variant="overline" color="text.secondary" display="block">
            1 · Agentic job
          </Typography>
          <Button
            size="small"
            variant="text"
            disabled={agenticJobsLoading}
            onClick={() => void loadAgenticJobsList()}
            sx={{ flexShrink: 0, mt: -0.5 }}
          >
            {agenticJobsLoading ? 'Loading…' : 'Refresh job list'}
          </Button>
        </Stack>
        <FormControlLabel
          control={
            <Checkbox
              size="small"
              checked={onlyJobsWithNoAgentReport}
              onChange={(e) => setOnlyJobsWithNoAgentReport(e.target.checked)}
              disabled={agenticJobsLoading}
            />
          }
          label={
            <Typography variant="body2" component="span">
              Show only agentic jobs with <strong>no report linked to this agentic job id</strong> (
              <code>agentic_job_public_id</code>)
            </Typography>
          }
          sx={{ alignItems: 'flex-start', ml: 0, display: 'block', mt: 0.5 }}
        />
        <TextField
          select
          fullWidth
          label="Select agentic job"
          value={selectedAgenticJobPublicId}
          onChange={(e) => setSelectedAgenticJobPublicId(e.target.value)}
          disabled={agenticJobsLoading || noJobsWhenFiltered}
          SelectProps={{
            displayEmpty: true,
            MenuProps: PRED_JOB_SELECT_MENU_PROPS,
            renderValue: (val) => {
              if (val === '' || val == null) {
                return (
                  <Typography variant="body2" component="em" color="text.secondary">
                    Choose an agentic job…
                  </Typography>
                );
              }
              return (
                <Typography variant="body2" component="span" sx={{ fontFamily: 'monospace', fontSize: 12 }}>
                  {String(val)}
                </Typography>
              );
            },
          }}
          helperText={
            agenticJobsLoading
              ? 'Loading jobs…'
              : noJobsWhenFiltered
                ? 'No agentic jobs without an id-linked report — uncheck the filter or refresh the list.'
                : onlyJobsWithNoAgentReport
                  ? `${agenticJobsForSelect.length} with no id-linked report (${agenticJobsRows.length} in agentic_jobs) · ${completedAgenticJobCount} with completed prediction batch`
                  : `${agenticJobsRows.length} agentic job row(s) from GET /agent/jobs · ${completedAgenticJobCount} with completed prediction batch · agent reports from server`
          }
        >
          <MenuItem value="">
            <em>Choose an agentic job…</em>
          </MenuItem>
          {renderAgenticJobOrphanMenuItem(selectedAgenticJobPublicId, agenticJobsForSelect)}
          {agenticJobsForSelect.map((j) => {
            const lineReports = agentReportsForAgenticJobLine(j, agentReportsList);
            const n = lineReports.length;
            const latest = latestCreatedAtIso(lineReports);
            const reportsLabel = 'Reports (linked id)';
            return (
            <MenuItem key={j.public_id} value={j.public_id}>
              <Stack spacing={0.35} alignItems="flex-start" sx={{ py: 0.5, maxWidth: 1 }}>
                <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600, letterSpacing: 0.2 }}>
                  Agentic job (DB)
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                  {j.public_id}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  <Box component="span" sx={{ fontWeight: 600, color: 'text.primary' }}>
                    Prediction batch
                  </Box>
                  {' · '}
                  <Box component="span" sx={{ fontFamily: 'monospace', fontSize: 11 }}>
                    {j.prediction_job_public_id}
                  </Box>
                  {j.results_row_index != null && j.results_row_index >= 0 ? ` · row ${j.results_row_index}` : ''}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {agenticJobBatchCaption(j)} · agentic job {fDateTime(j.updated_at)}
                </Typography>
                <Typography
                  variant="caption"
                  color={n > 0 ? 'success.main' : 'text.secondary'}
                >
                  <Box component="span" sx={{ fontWeight: 600, color: 'inherit' }}>
                    {reportsLabel}
                  </Box>
                  {`: ${n}`}
                  {latest && n > 0 ? ` · latest ${fDateTime(latest)}` : ''}
                </Typography>
              </Stack>
            </MenuItem>
            );
          })}
        </TextField>
        {effectiveJobId ? (
          <Box sx={{ mt: 1.5 }}>
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 12, wordBreak: 'break-all' }}>
                Prediction batch (API): {effectivePredictionJobId}
              </Typography>
              {selectedAgenticJobPublicId.trim() ? (
                <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: 11, wordBreak: 'break-all' }}>
                  Agentic job id (agentic_jobs): {selectedAgenticJobPublicId.trim()}
                </Typography>
              ) : null}
              {jobFromPrep && effectivePredictionJobId === jobFromPrep ? (
                <Chip size="small" label="Matches RAG prep prediction job" color="success" variant="outlined" />
              ) : null}
              {prep?.agenticJobPublicId?.trim() &&
              selectedAgenticJobPublicId.trim() &&
              prep.agenticJobPublicId.trim() === selectedAgenticJobPublicId.trim() ? (
                <Chip size="small" label="Matches RAG prep agentic job id" color="info" variant="outlined" />
              ) : null}
            </Stack>
            <Typography
              variant="caption"
              color={reportsForCurrentSelection.length > 0 ? 'success.main' : 'text.secondary'}
              display="block"
              sx={{ mt: 0.75 }}
            >
              {formatAgentReportsLineSummary(reportsForCurrentSelection, reportSummaryScope)}
            </Typography>
            {decideContext.ri != null && decideContext.ri >= 0 ? (
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
                Agentic API uses <code>results_row_index={decideContext.ri}</code> — prompt <code>sample_data</code> matches this row
                (predicted label, class probabilities, raw <code>shap</code>), not the default first/flagged row.
              </Typography>
            ) : null}
          </Box>
        ) : (
          <Alert severity="warning" variant="outlined" sx={{ py: 0.75, mt: 1.5 }}>
            Select an agentic job above, or use <strong>Set agentic job for handoff</strong> / <strong>Save for agentic job</strong> in RAG
            prep to set the job from handoff.
          </Alert>
        )}
        {prep ? (
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            Prep handoff {prep.updatedAt ? fDateTime(prep.updatedAt) : '—'} ·{' '}
            {prep.finalDocCount ?? prep.citations?.length ?? 0} chunk(s)
            {' · '}
            {prep.templateLabel || prep.templateId || 'Template'}
            {prep.rowIndex != null ? ` · row ${prep.rowIndex}` : ''}
            {prep.retrievalMetaLine ? (
              <>
                <br />
                {prep.retrievalMetaLine}
              </>
            ) : null}
          </Typography>
        ) : null}
      </Paper>

      <Accordion defaultExpanded disableGutters elevation={0} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, '&:before': { display: 'none' } }}>
        <AccordionSummary expandIcon={<Iconify width={20} icon="eva:arrow-ios-downward-fill" />}>
          <Typography variant="subtitle2">2 · Full orchestration prompt (POST /agent/decide user message)</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            {!prep && (
              <Alert severity="warning" variant="outlined">
                No prep in this browser session. Save from <strong>RAG prep</strong>, then <strong>Reload prep</strong>.
              </Alert>
            )}
            {prep && !hasCitations && (
              <Alert severity="warning" variant="outlined">
                Prep has no citations — run retrieval in RAG prep and save again (orchestration prompt still loads; RAG section may
                be sparse).
              </Alert>
            )}
            <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
              <Button
                size="small"
                variant="outlined"
                disabled={orchestrationPromptLoading}
                onClick={() => reloadOrchestrationPreview()}
              >
                {orchestrationPromptLoading ? 'Loading prompt…' : 'Refresh prompt preview'}
              </Button>
              <Typography variant="caption" color="text.secondary">
                Orchestration prompt: prediction summary, <code>sample_data</code> (prediction_row.shap.per_feature = top 5 |value|; stored rows keep top 10),
                allowed actions, agentic tiers, KB excerpts from prep citations. RAG template / prep narrative is not injected into the
                agent LLM.
              </Typography>
            </Stack>
            <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.neutral', maxHeight: 480, overflow: 'auto' }}>
              {orchestrationPromptLoading && !orchestrationPrompt ? (
                <Typography variant="body2" color="text.secondary">
                  Loading…
                </Typography>
              ) : (
                <Typography component="pre" variant="caption" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                  {orchestrationPrompt || '—'}
                </Typography>
              )}
            </Paper>
            <Box>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 0.5 }}>
                RAG document list (prep)
              </Typography>
              {!hasCitations ? (
                <Typography variant="body2" color="text.secondary">
                  No chunks yet.
                </Typography>
              ) : (
                <TableContainer sx={{ maxHeight: 280, border: 1, borderColor: 'divider', borderRadius: 1 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>#</TableCell>
                        <TableCell>sim</TableCell>
                        <TableCell>rerank</TableCell>
                        <TableCell>MMR</TableCell>
                        <TableCell>source</TableCell>
                        <TableCell>excerpt</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {prep!.citations.map((h, i) => (
                        <TableRow key={`${h.text.slice(0, 24)}-${i}`}>
                          <TableCell>{i + 1}</TableCell>
                          <TableCell sx={{ whiteSpace: 'nowrap' }}>{h.score.toFixed(3)}</TableCell>
                          <TableCell sx={{ whiteSpace: 'nowrap' }}>
                            {h.rerank_score != null ? h.rerank_score.toFixed(3) : '—'}
                          </TableCell>
                          <TableCell sx={{ whiteSpace: 'nowrap' }}>
                            {h.mmr_margin != null ? h.mmr_margin.toFixed(3) : '—'}
                          </TableCell>
                          <TableCell sx={{ maxWidth: 100 }}>{h.source ?? '—'}</TableCell>
                          <TableCell sx={{ maxWidth: 280, typography: 'caption' }}>
                            {h.text.slice(0, 220)}
                            {h.text.length > 220 ? '…' : ''}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}
            </Box>
          </Stack>
        </AccordionDetails>
      </Accordion>

      <Stack spacing={1.25}>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }} flexWrap="wrap">
          <Button
            variant="contained"
            size="large"
            disabled={!effectiveJobId || running || runningTrust || selectedJobNotCompleted}
            onClick={async () => {
              onNotify(null);
              setRunning(true);
              try {
                const p = readAgenticPrep();
                const { jid, ri } = resolveAgenticDecideContext(
                  p,
                  agenticJobsRows,
                  selectedAgenticJobPublicId
                );
                if (!jid) {
                  onNotify({
                    severity: 'error',
                    text: 'Select an agentic job from the list (GET /agent/jobs) or save handoff from RAG prep.',
                  });
                  return;
                }
                const r = await agentDecide({
                  prediction_job_public_id: jid,
                  use_rag: true,
                  results_row_index: ri,
                  agentic_job_public_id: selectedAgenticJobPublicId.trim() || null,
                  feature_notes: null,
                  kb_citations: p?.citations?.length ? p.citations : null,
                  agent_action_preset: 'standard',
                  anchor_trust_chain: false,
                });
                onNotify({ severity: 'success', text: `Saved agentic report ${r.public_id}.` });
                await loadAgenticJobsList();
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              } finally {
                setRunning(false);
              }
            }}
          >
            {running ? 'Calling POST /agent/decide…' : 'Run agent & save report'}
          </Button>
          <Button
            variant="outlined"
            size="large"
            color="secondary"
            disabled={!effectiveJobId || running || runningTrust || selectedJobNotCompleted}
            onClick={async () => {
              onNotify(null);
              setRunningTrust(true);
              try {
                const p = readAgenticPrep();
                const { jid, ri } = resolveAgenticDecideContext(
                  p,
                  agenticJobsRows,
                  selectedAgenticJobPublicId
                );
                if (!jid) {
                  onNotify({
                    severity: 'error',
                    text: 'Select an agentic job from the list (GET /agent/jobs) or save handoff from RAG prep.',
                  });
                  return;
                }
                const r = await agentDecide({
                  prediction_job_public_id: jid,
                  use_rag: true,
                  results_row_index: ri,
                  agentic_job_public_id: selectedAgenticJobPublicId.trim() || null,
                  feature_notes: null,
                  kb_citations: p?.citations?.length ? p.citations : null,
                  agent_action_preset: 'standard',
                  anchor_trust_chain: true,
                });
                const tc = r.trust_commitment?.trim();
                const tcShort =
                  tc && tc.length > 40 ? `${tc.slice(0, 16)}…${tc.slice(-12)}` : tc;
                onNotify({
                  severity: 'success',
                  text: tcShort
                    ? `Saved report ${r.public_id} with trust commitment (demo): ${tcShort}`
                    : `Saved report ${r.public_id}.`,
                });
                await loadAgenticJobsList();
              } catch (e) {
                onNotify({ severity: 'error', text: formatError(e) });
              } finally {
                setRunningTrust(false);
              }
            }}
          >
            {runningTrust ? 'Saving report + trust anchor…' : 'Run agent & save to trust chain'}
          </Button>
        </Stack>
        <Typography variant="caption" color="text.secondary" sx={{ maxWidth: 640 }}>
          Same LLM run as above · <strong>Trust chain</strong> adds a demo SHA-256 commitment into the saved JSON (roadmap: replace
          with on-chain / notary). Matches preview · standard preset · KB citations from prep.
          {selectedJobNotCompleted ? (
            <>
              {' '}
              <strong>
                Run agent is disabled until this prediction batch is completed (currently{' '}
                {decideContext.predictionStatus ?? 'unknown'}).
              </strong>
            </>
          ) : null}
        </Typography>
      </Stack>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1} sx={{ mb: 1.5 }} flexWrap="wrap">
          <Typography variant="overline" color="text.secondary">
            Saved agentic reports
          </Typography>
          <Button
            size="small"
            variant="text"
            disabled={agenticJobsLoading}
            onClick={() => void loadAgenticJobsList()}
          >
            {agenticJobsLoading ? 'Loading…' : 'Refresh list'}
          </Button>
        </Stack>
        {agentReportsList.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No reports yet — run <strong>Run agent &amp; save report</strong> or <strong>Run agent &amp; save to trust chain</strong>{' '}
            above (with an agentic job selected to store <code>agentic_job_public_id</code> on the report).
          </Typography>
        ) : agentReportsForTable.length === 0 ? (
          <Typography variant="body2" color="text.secondary">
            No reports linked to the selected agentic job id yet — either run the agent with that job selected, or clear the
            dropdown to see all reports. Legacy reports may have no agentic job id until you re-run.
          </Typography>
        ) : (
          <>
            <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
              {selectedAgenticJobPublicId.trim()
                ? `Showing reports for agentic_jobs id ${selectedAgenticJobPublicId.trim().slice(0, 8)}… — clear the dropdown above to list all.`
                : 'Showing all reports — select an agentic job to filter this table.'}{' '}
              Click a row for full details.
            </Typography>
            <TableContainer sx={{ maxHeight: 360, border: 1, borderColor: 'divider', borderRadius: 1 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Report</TableCell>
                    <TableCell>Agentic job id</TableCell>
                    <TableCell>Prediction batch</TableCell>
                    <TableCell>Row</TableCell>
                    <TableCell>Recommended action</TableCell>
                    <TableCell>Summary</TableCell>
                    <TimeSortHeadCell
                      label="Created"
                      order={reportsTableTimeOrder}
                      onOrderChange={setReportsTableTimeOrder}
                      sx={{ whiteSpace: 'nowrap' }}
                    />
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {agentReportsTableSorted.map((r) => (
                    <TableRow
                      key={r.public_id}
                      hover
                      selected={reportDetailRow?.public_id === r.public_id}
                      onClick={() => {
                        void (async () => {
                          try {
                            setReportDetailRow(await getAgentReport(r.public_id));
                          } catch {
                            setReportDetailRow(r);
                          }
                        })();
                      }}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, maxWidth: 200, wordBreak: 'break-all' }}>
                        {r.public_id}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, maxWidth: 180, wordBreak: 'break-all' }}>
                        {r.agentic_job_public_id?.trim() || '—'}
                      </TableCell>
                      <TableCell sx={{ fontFamily: 'monospace', fontSize: 11, maxWidth: 180, wordBreak: 'break-all' }}>
                        {r.prediction_job_public_id ?? '—'}
                      </TableCell>
                      <TableCell sx={{ typography: 'caption', whiteSpace: 'nowrap' }}>
                        {r.results_row_index != null && r.results_row_index >= 0 ? r.results_row_index : '—'}
                      </TableCell>
                      <TableCell sx={{ maxWidth: 160, typography: 'caption' }}>
                        {r.recommended_action.length > 90 ? `${r.recommended_action.slice(0, 90)}…` : r.recommended_action}
                      </TableCell>
                      <TableCell sx={{ maxWidth: 280, typography: 'caption' }}>
                        {r.summary.length > 140 ? `${r.summary.slice(0, 140)}…` : r.summary}
                      </TableCell>
                      <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(r.created_at)}</TableCell>
                      <TableCell align="right">
                        <Button
                          size="small"
                          color="error"
                          variant="outlined"
                          disabled={deletingReportId === r.public_id}
                          onClick={(e) => {
                            e.stopPropagation();
                            void (async () => {
                              if (!window.confirm(`Delete agentic report ${r.public_id}?`)) return;
                              onNotify(null);
                              setDeletingReportId(r.public_id);
                              try {
                                await deleteAgentReport(r.public_id);
                                if (reportDetailRow?.public_id === r.public_id) setReportDetailRow(null);
                                onNotify({ severity: 'success', text: `Deleted report ${r.public_id}.` });
                                await loadAgenticJobsList();
                              } catch (err) {
                                onNotify({ severity: 'error', text: formatError(err) });
                              } finally {
                                setDeletingReportId(null);
                              }
                            })();
                          }}
                        >
                          {deletingReportId === r.public_id ? '…' : 'Delete'}
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
      </Paper>

      <Dialog open={!!reportDetailRow} onClose={() => setReportDetailRow(null)} maxWidth="md" fullWidth>
        <DialogTitle>Agentic report details</DialogTitle>
        <DialogContent dividers>
          {reportDetailRow ? (
            <Stack spacing={2}>
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  Report public_id
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                  {reportDetailRow.public_id}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  Agentic job id (<code>agentic_jobs</code>)
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                  {reportDetailRow.agentic_job_public_id?.trim() || '—'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  Prediction batch (<code>prediction_jobs</code>)
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                  {reportDetailRow.prediction_job_public_id ?? '—'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  <code>results_row_index</code>
                </Typography>
                <Typography variant="body2">
                  {reportDetailRow.results_row_index != null && reportDetailRow.results_row_index >= 0
                    ? reportDetailRow.results_row_index
                    : '—'}
                </Typography>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary" display="block">
                  Created
                </Typography>
                <Typography variant="body2">{fDateTime(reportDetailRow.created_at)}</Typography>
              </Box>
              {reportDetailRow.report_path ? (
                <Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Stored path (relative)
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all', fontSize: 12 }}>
                    {reportDetailRow.report_path}
                  </Typography>
                </Box>
              ) : null}
              {reportDetailRow.trust_commitment ? (
                <Box>
                  <Typography variant="caption" color="text.secondary" display="block">
                    Trust commitment (demo)
                  </Typography>
                  <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all', fontSize: 12 }}>
                    {reportDetailRow.trust_commitment}
                  </Typography>
                  {reportDetailRow.trust_chain_mode ? (
                    <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5 }}>
                      Mode: {reportDetailRow.trust_chain_mode}
                    </Typography>
                  ) : null}
                </Box>
              ) : null}
              <Divider />
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Recommended action
                </Typography>
                <Typography variant="body2">{reportDetailRow.recommended_action}</Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Summary
                </Typography>
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {reportDetailRow.summary}
                </Typography>
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  RAG context used
                </Typography>
                {reportDetailRow.rag_context_used?.trim() ? (
                  <Paper variant="outlined" sx={{ p: 1.5, maxHeight: 220, overflow: 'auto', bgcolor: 'background.neutral' }}>
                    <Typography component="pre" variant="caption" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {reportDetailRow.rag_context_used}
                    </Typography>
                  </Paper>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    —
                  </Typography>
                )}
              </Box>
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Raw LLM response
                </Typography>
                {reportDetailRow.raw_llm_response?.trim() ? (
                  <Paper variant="outlined" sx={{ p: 1.5, maxHeight: 280, overflow: 'auto', bgcolor: 'background.neutral' }}>
                    <Typography component="pre" variant="caption" sx={{ m: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}>
                      {reportDetailRow.raw_llm_response}
                    </Typography>
                  </Paper>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    —
                  </Typography>
                )}
              </Box>
            </Stack>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setReportDetailRow(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
