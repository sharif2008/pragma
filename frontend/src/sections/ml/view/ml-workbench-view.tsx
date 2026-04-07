import type { IconifyName } from 'src/components/iconify';
import type { PredictionJobListItem } from 'src/services/predictions.service';
import type {
  KBQueryHit,
  ManagedFileOut,
  TrainingJobOut,
  ModelVersionOut,
  RAGTemplateItem,
  PredictionJobOut,
  DatasetPreviewOut,
  PredictionResultsJson,
  KBRAGLatestPredictionResponse,
  AgenticReportOut,
  AgenticJobOut,
  JobStatus,
} from 'src/api/types';

import { useId, useRef, useMemo, Fragment, useState, useEffect, useCallback } from 'react';

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
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import Dialog from '@mui/material/Dialog';
import Divider from '@mui/material/Divider';
import Checkbox from '@mui/material/Checkbox';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import Collapse from '@mui/material/Collapse';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import { alpha, useTheme } from '@mui/material/styles';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';

import { fDateTime } from 'src/utils/format-time';

import { DashboardContent } from 'src/layouts/dashboard';
import {
  kbQuery,
  ApiError,
  kbDelete,
  kbUpload,
  getHealth,
  listModels,
  agentDecide,
  agentDecidePromptPreview,
  kbListFiles,
  deleteModel,
  listDatasets,
  kbFuseHitsMmr,
  deleteDataset,
  getApiBaseUrl,
  startTraining,
  uploadDataset,
  getTrainingJob,
  rebuildTraining,
  startPrediction,
  getPredictionJob,
  listTrainingJobs,
  deleteTrainingJob,
  getDatasetPreview,
  listAllPredictionJobs,
  deletePredictionJob,
  deleteAllPendingPredictionJobs,
  listPredictionInputs,
  uploadPredictionInput,
  kbRagTemplatesPredictionJob,
  kbLlmShapRetrievalQuery,
  listAllAgentReports,
  deleteAgentReport,
  getAgentReport,
  createAgenticJob,
  listAllAgenticJobs,
} from 'src/services';

import { Iconify } from 'src/components/iconify';
import { ModelVersionDetailDialog } from 'src/components/run-monitoring/detail-dialogs';

import {
  AGENTIC_PREP_UPDATED_EVENT,
  readAgenticPrep,
  upsertAgenticJobHandoff,
  writeAgenticPrep,
  type AgenticPrepPayload,
} from 'src/sections/ml/agentic-prep-storage';

// ----------------------------------------------------------------------

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function MlWorkbenchView() {
  /** 0 = Summary · 1 = Data & training (datasets, training, KB) · 2 = Predictions, RAG prep, agent */
  const [group, setGroup] = useState(0);
  const [innerTab, setInnerTab] = useState(0);
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
      <Stack spacing={2} sx={{ mb: 3 }}>
        <Typography variant="h4">ML &amp; RAG workbench</Typography>
        <Stack direction="row" alignItems="center" spacing={2} flexWrap="wrap">
          <Typography variant="body2" color="text.secondary">
            API base: <strong>{getApiBaseUrl()}</strong>
          </Typography>
          <Button size="small" variant="outlined" onClick={refreshHealth}>
            Ping /health
          </Button>
          {ping && <Chip size="small" color="success" label={`health: ${ping}`} />}
        </Stack>
        {banner && <Alert severity={banner.severity}>{banner.text}</Alert>}
      </Stack>

      <Card
        elevation={0}
        sx={{
          border: 1,
          borderColor: 'divider',
          borderRadius: 2,
          overflow: 'hidden',
          boxShadow: (t) => `0 0 0 1px ${alpha(t.palette.divider, 0.6)}, ${t.shadows[2]}`,
        }}
      >
        <Tabs
          value={group}
          onChange={(_, v) => {
            setGroup(v);
            setInnerTab(0);
          }}
          variant="fullWidth"
          sx={{
            px: { xs: 0.5, sm: 1 },
            pt: 1,
            bgcolor: (t) => alpha(t.palette.primary.main, 0.06),
            borderBottom: 1,
            borderColor: 'divider',
            minHeight: 52,
            '& .MuiTab-root': { minHeight: 48, fontWeight: 700, textTransform: 'none', fontSize: '0.9rem' },
          }}
        >
          <Tab label="Summary" />
          <Tab label="Data & training" />
          <Tab label="Predict, RAG & agent" />
        </Tabs>
        {group === 1 && (
          <Tabs
            key="ml-inner-data-training"
            value={innerTab}
            onChange={(_, v) => setInnerTab(v)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              px: { xs: 0.5, sm: 1 },
              pt: 0.5,
              bgcolor: (t) => alpha(t.palette.grey[500], 0.06),
              borderBottom: 1,
              borderColor: 'divider',
              minHeight: 48,
              '& .MuiTab-root': { minHeight: 44, fontWeight: 600, textTransform: 'none', fontSize: '0.8125rem' },
            }}
          >
            <Tab label="Datasets" />
            <Tab label="Training & models" />
            <Tab label="Knowledge base" />
          </Tabs>
        )}
        {group === 2 && (
          <Tabs
            key="ml-inner-predict-rag-agent"
            value={innerTab}
            onChange={(_, v) => setInnerTab(v)}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            sx={{
              px: { xs: 0.5, sm: 1 },
              pt: 0.5,
              bgcolor: (t) => alpha(t.palette.grey[500], 0.06),
              borderBottom: 1,
              borderColor: 'divider',
              minHeight: 48,
              '& .MuiTab-root': { minHeight: 44, fontWeight: 600, textTransform: 'none', fontSize: '0.8125rem' },
            }}
          >
            <Tab label="Predictions" />
            <Tab label="RAG & LLM prep" />
            <Tab label="Agentic actions" />
          </Tabs>
        )}
        <CardContent>
          {group === 0 && <PipelineSummaryPanel />}
          {group === 1 && innerTab === 0 && <DatasetsPanel onNotify={setBanner} />}
          {group === 1 && innerTab === 1 && <TrainingPanel onNotify={setBanner} />}
          {group === 1 && innerTab === 2 && <KbPanel onNotify={setBanner} />}
          {group === 2 && innerTab === 0 && <PredictionsPanel onNotify={setBanner} />}
          {group === 2 && innerTab === 1 && <RagLlmPrepPanel onNotify={setBanner} />}
          {group === 2 && innerTab === 2 && <AgenticActionsPanel onNotify={setBanner} />}
        </CardContent>
      </Card>
    </DashboardContent>
  );
}

type PanelProps = { onNotify: (b: { severity: 'success' | 'error' | 'info'; text: string } | null) => void };

const PIPELINE_STEPS: {
  label: string;
  summary: string;
  bullets: string[];
  apis: string;
}[] = [
  {
    label: 'Datasets',
    summary: 'Versioned training CSVs (public_id + preview).',
    bullets: [
      'Upload CSV → public_id + version for traceability; optional replace public_id for a new version in the same chain.',
      'Preview endpoint: column names and sample rows before training.',
    ],
    apis: 'POST /datasets/upload · GET /datasets · GET /datasets/{public_id}/preview',
  },
  {
    label: 'Training & models',
    summary: 'VFL-only jobs → async metrics and model_version_public_id.',
    bullets: [
      'Algorithm is vfl (not RF/XGBoost): dataset public_id + target column.',
      'Three-party vertical split (embed → fuse → classify); workbench uses storage/agentic_features.json for column ownership; API can omit vfl_agent_definitions_path for heuristics.',
      'Rebuild clones dataset + hyperparameters into a new job.',
    ],
    apis:
      'POST /training/start · GET /training · DELETE /training/{id} · POST /training/rebuild · GET /models · DELETE /models/{id}',
  },
  {
    label: 'Predictions',
    summary: 'Batch score with a registered model → output CSV.',
    bullets: [
      'Upload prediction CSV, then model_version_public_id + input public_id to start the job.',
      'Outputs: predicted_label, max_class_probability, optional flags; poll until completed.',
    ],
    apis: 'POST /predictions/upload-input · POST /predictions/start · GET /predictions/{public_id}',
  },
  {
    label: 'Knowledge base (LLM RAG stack)',
    summary: 'Chunk → embed → FAISS; multi-query + rerank + MMR → LLM.',
    bullets: [
      'Semantic split → chunks → embeddings → per-doc FAISS.',
      'Retrieval: fused multi-query, RRF + max-score rerank, MMR for diverse passages.',
      '/kb/rag-llm with citations; precomputed_citations after /kb/query + /kb/fuse-hits-mmr (or legacy /kb/query-multi). Feed docs + prediction context into Agentic.',
    ],
    apis:
      'GET /kb/rag-templates/latest-prediction · POST /kb/llm-shap-retrieval-query · POST /kb/query + /kb/fuse-hits-mmr · POST /kb/rag-llm · POST /kb/upload · GET /kb/files',
  },
  {
    label: 'Agentic actions',
    summary: 'Policy LLM: predictions + KB → summary + recommended_action.',
    bullets: [
      'Inputs: prediction job stats + optional kb_citations (same RAG stack).',
      'Outputs: agentic_reports with recommended_action (e.g. block_ip, alert_admin, monitor).',
      'Execution mapping: RAN / Edge / CORE roles; optional hash attestation for verification.',
    ],
    apis:
      'POST /agent/decide (optional anchor_trust_chain) · POST /agent/decide/prompt-preview · GET /agent/reports · DELETE /agent/reports/{public_id}',
  },
  {
    label: 'Trust layer (blockchain)',
    summary: 'Anchor for report integrity (roadmap in demo API).',
    bullets: [
      'Attest hash(summary ∥ recommended_action ∥ prediction_job_public_id ∥ timestamp); store tx id with report_path.',
      'Verifiers re-hash and compare to on-chain (or enterprise log) commitment.',
    ],
    apis: '(roadmap) chain notary / Web3 adapter — not in this demo API yet.',
  },
];

const PIPELINE_STAGE_CARDS: {
  id: string;
  title: string;
  sub: string;
  icon: IconifyName;
  tabHint: string;
}[] = [
  {
    id: 'datasets',
    title: 'Datasets',
    sub: 'Versioned CSV',
    icon: 'solar:check-circle-bold',
    tabHint: 'Datasets',
  },
  {
    id: 'training',
    title: 'Training',
    sub: 'Models & VFL',
    icon: 'eva:trending-up-fill',
    tabHint: 'Training & models',
  },
  {
    id: 'predict',
    title: 'Predictions',
    sub: 'Batch scores',
    icon: 'solar:eye-bold',
    tabHint: 'Predictions',
  },
  {
    id: 'kb',
    title: 'Knowledge',
    sub: 'RAG · MMR · LLM',
    icon: 'eva:search-fill',
    tabHint: 'Knowledge base',
  },
  {
    id: 'agent',
    title: 'Agentic',
    sub: 'Policy LLM',
    icon: 'solar:chat-round-dots-bold',
    tabHint: 'Agentic actions',
  },
  {
    id: 'chain',
    title: 'Trust',
    sub: 'Blockchain',
    icon: 'solar:shield-keyhole-bold-duotone',
    tabHint: 'Roadmap / attestation',
  },
];

function PipelineInteractiveFlow() {
  const theme = useTheme();
  const [active, setActive] = useState<number | null>(null);

  return (
    <Box>
      <Stack direction="row" alignItems="center" justifyContent="space-between" flexWrap="wrap" gap={1} sx={{ mb: 2 }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 800, letterSpacing: -0.3 }}>
          Flow at a glance
        </Typography>
        <Chip size="small" variant="outlined" color="primary" label="Tap a stage · jump hint below" sx={{ fontWeight: 600 }} />
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5, maxWidth: 720, lineHeight: 1.65 }}>
        Follow the stages left to right: data and models first, then scoring, retrieval, and the agentic policy. The trust step
        anchors attestation.
      </Typography>
      <Stack
        direction="row"
        spacing={0}
        alignItems="stretch"
        sx={{
          overflowX: 'auto',
          pb: 1,
          pt: 0.5,
          mx: -0.5,
          px: 0.5,
          scrollbarWidth: 'thin',
        }}
      >
        {PIPELINE_STAGE_CARDS.map((n, i) => {
          const isChain = n.id === 'chain';
          const isActive = active === i;
          const accent = isChain ? theme.palette.success.main : theme.palette.primary.main;
          return (
            <Fragment key={n.id}>
              <Card
                elevation={0}
                onClick={() => setActive((v) => (v === i ? null : i))}
                sx={{
                  position: 'relative',
                  minWidth: 132,
                  flex: '0 0 auto',
                  cursor: 'pointer',
                  borderRadius: 2,
                  border: '2px solid',
                  borderColor: isActive ? accent : 'divider',
                  bgcolor: (t) => alpha(accent, isActive ? 0.12 : 0.03),
                  boxShadow: isActive ? `0 8px 24px ${alpha(accent, 0.2)}` : 'none',
                  transition: theme.transitions.create(['border-color', 'box-shadow', 'background-color', 'transform'], {
                    duration: theme.transitions.duration.short,
                  }),
                  '&:hover': {
                    borderColor: isActive ? accent : alpha(accent, 0.35),
                    bgcolor: (t) => alpha(accent, isActive ? 0.14 : 0.06),
                    transform: 'translateY(-3px)',
                    boxShadow: isActive ? `0 12px 28px ${alpha(accent, 0.22)}` : theme.shadows[3],
                  },
                }}
              >
                <Box
                  sx={{
                    position: 'absolute',
                    top: 8,
                    right: 8,
                    width: 22,
                    height: 22,
                    borderRadius: '50%',
                    typography: 'caption',
                    fontWeight: 800,
                    fontSize: 11,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: isActive ? accent : alpha(theme.palette.grey[500], 0.12),
                    color: isActive ? (isChain ? theme.palette.success.contrastText : theme.palette.primary.contrastText) : 'text.secondary',
                  }}
                >
                  {i + 1}
                </Box>
                <CardContent sx={{ py: 2, px: 1.75, '&:last-child': { pb: 2 } }}>
                  <Iconify
                    width={32}
                    icon={n.icon}
                    sx={{ color: accent, mb: 1.25, opacity: isActive ? 1 : 0.88 }}
                  />
                  <Typography variant="subtitle2" sx={{ lineHeight: 1.3, fontWeight: 800, fontSize: 14 }}>
                    {n.title}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.35, lineHeight: 1.4 }}>
                    {n.sub}
                  </Typography>
                  <Chip
                    size="small"
                    label={n.tabHint}
                    sx={{
                      mt: 1.25,
                      height: 22,
                      fontWeight: 600,
                      fontSize: 10,
                      bgcolor: (t) => alpha(accent, 0.12),
                      color: accent,
                      border: '1px solid',
                      borderColor: (t) => alpha(accent, 0.25),
                    }}
                  />
                </CardContent>
              </Card>
              {i < PIPELINE_STAGE_CARDS.length - 1 && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: { xs: 0.25, sm: 0.75 },
                    flexShrink: 0,
                    alignSelf: 'center',
                  }}
                >
                  <Box
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: 32,
                      height: 32,
                      borderRadius: '50%',
                      background: (t) =>
                        `linear-gradient(135deg, ${alpha(t.palette.primary.main, 0.12)} 0%, ${alpha(t.palette.primary.main, 0.04)} 100%)`,
                      border: 1,
                      borderColor: 'divider',
                      color: 'primary.main',
                    }}
                  >
                    <Iconify icon="eva:arrow-ios-forward-fill" width={18} />
                  </Box>
                </Box>
              )}
            </Fragment>
          );
        })}
      </Stack>
      <Collapse in={active !== null} unmountOnExit>
        {active !== null && (
          <Box
            sx={{
              mt: 2.5,
              py: 1.75,
              px: 2.25,
              borderRadius: 2,
              border: 1,
              borderColor: 'primary.main',
              borderLeftWidth: 4,
              borderLeftColor: 'primary.main',
              bgcolor: (t) => alpha(t.palette.primary.main, 0.06),
              boxShadow: (t) => `inset 0 1px 0 ${alpha(t.palette.common.white, 0.06)}`,
            }}
          >
            <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.65 }}>
              <Typography component="span" variant="subtitle2" color="text.primary" sx={{ fontWeight: 800 }}>
                {PIPELINE_STAGE_CARDS[active].title}
              </Typography>
              {' — open '}
              <Typography component="span" variant="subtitle2" color="primary.main" sx={{ fontWeight: 700 }}>
                {PIPELINE_STAGE_CARDS[active].tabHint}
              </Typography>
              {
                ' via the tabs: Summary, Data & training (datasets, training, knowledge), or Predict, RAG & agent (predictions, RAG prep, agentic). Order is left → right.'
              }
            </Typography>
          </Box>
        )}
      </Collapse>
    </Box>
  );
}

function PipelineDecisionTreeDiagram({ embedded = false }: { embedded?: boolean }) {
  const theme = useTheme();
  const uid = useId().replace(/:/g, '');
  const arrowId = `dt-arr-${uid}`;
  const arrowTrustId = `dt-arr-tr-${uid}`;
  const shadowId = `dt-sh-${uid}`;
  const gradStrongId = `dt-gs-${uid}`;
  const gradSoftId = `dt-gf-${uid}`;

  const ff = String(theme.typography.fontFamily ?? 'system-ui, sans-serif').replace(/"/g, '');

  const primary = theme.palette.primary.main;
  const primaryDark = theme.palette.primary.dark;
  const contrast = theme.palette.primary.contrastText;
  const stroke = theme.palette.divider;
  const accent = theme.palette.text.secondary;
  const trust = theme.palette.success.main;
  const laneBg = alpha(primary, 0.06);
  const laneBorder = alpha(primary, 0.14);
  const softNode = alpha(primary, 0.35);

  const svgText = (props: { x: number; y: number; anchor?: 'middle' | 'start'; size: number; weight?: number; fill: string; children: string }) => (
    <text
      x={props.x}
      y={props.y}
      textAnchor={props.anchor ?? 'middle'}
      fill={props.fill}
      style={{ fontSize: props.size, fontWeight: props.weight ?? 500, fontFamily: ff }}
    >
      {props.children}
    </text>
  );

  return (
    <Box
      sx={{
        p: embedded ? 0 : 2,
        pt: embedded ? 0 : 2,
        borderRadius: embedded ? 0 : 2,
        border: embedded ? 0 : 1,
        borderColor: 'divider',
        bgcolor: embedded ? 'transparent' : 'background.neutral',
      }}
    >
      {!embedded && (
        <>
          <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 700 }}>
            Agent decision tree
          </Typography>
          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
            Predictions + RAG → policy LLM → optional attestation → RAN / Edge / CORE.
          </Typography>
        </>
      )}
      <Box
        sx={{
          width: 1,
          overflowX: 'auto',
          borderRadius: 2,
          bgcolor: (t) => alpha(t.palette.grey[500], 0.06),
          border: 1,
          borderColor: 'divider',
          boxShadow: (t) => `inset 0 1px 0 ${alpha(t.palette.common.white, 0.05)}`,
        }}
      >
        <svg
          width="100%"
          height={380}
          viewBox="0 0 720 380"
          role="img"
          aria-label="Decision tree: predictions and RAG into agentic LLM, blockchain attestation, and RAN Edge CORE actions"
          style={{ display: 'block', minWidth: 600 }}
        >
          <defs>
            <linearGradient id={gradStrongId} x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor={primary} stopOpacity={1} />
              <stop offset="100%" stopColor={primaryDark} stopOpacity={1} />
            </linearGradient>
            <linearGradient id={gradSoftId} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor={softNode} stopOpacity={0.9} />
              <stop offset="100%" stopColor={softNode} stopOpacity={0.5} />
            </linearGradient>
            <filter id={shadowId} x="-8%" y="-8%" width="116%" height="116%">
              <feDropShadow dx="0" dy="2" stdDeviation="3" floodColor="#000" floodOpacity="0.12" />
            </filter>
            <marker
              id={arrowId}
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L8,4 L0,8 Z" fill={accent} />
            </marker>
            <marker
              id={arrowTrustId}
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L8,4 L0,8 Z" fill={trust} />
            </marker>
          </defs>

          <rect x="0" y="0" width="720" height="380" rx="12" fill={theme.palette.background.paper} opacity={0.45} />

          {svgText({ x: 132, y: 26, size: 11, weight: 800, fill: accent, children: 'Prediction context' })}
          {svgText({ x: 488, y: 26, size: 11, weight: 800, fill: accent, children: 'Knowledge base → RAG pipeline' })}

          <rect x="16" y="36" width="232" height="148" rx="14" fill={laneBg} stroke={laneBorder} strokeWidth="1.5" />
          <rect x="264" y="36" width="440" height="148" rx="14" fill={laneBg} stroke={laneBorder} strokeWidth="1.5" />

          <rect
            x="36"
            y="56"
            width="192"
            height="108"
            rx="12"
            fill={`url(#${gradStrongId})`}
            filter={`url(#${shadowId})`}
          />
          {svgText({ x: 132, y: 92, size: 14, weight: 800, fill: contrast, children: 'Predictions job' })}
          {svgText({ x: 132, y: 112, size: 10, weight: 500, fill: contrast, children: 'labels · probabilities · flags' })}
          {svgText({ x: 132, y: 132, size: 9, weight: 500, fill: contrast, children: 'stats for agentic prompt', anchor: 'middle' })}

          {[
            { y: 52, h: 26, n: '1', t1: 'Chunk + embed', t2: 'FAISS index' },
            { y: 84, h: 26, n: '2', t1: 'Multi-query fusion', t2: 'RRF · rerank' },
            { y: 116, h: 26, n: '3', t1: 'MMR selection', t2: 'diverse passages' },
            { y: 148, h: 28, n: '4', t1: 'LLM RAG answer', t2: 'citations · grounded' },
          ].map((row, i) => (
            <g key={i}>
              <circle cx="292" cy={row.y + row.h / 2} r="11" fill={primary} opacity={0.92} />
              <text
                x="292"
                y={row.y + row.h / 2 + 4}
                textAnchor="middle"
                fill={contrast}
                style={{ fontSize: 10, fontWeight: 800, fontFamily: ff }}
              >
                {row.n}
              </text>
              <rect
                x="312"
                y={row.y}
                width="376"
                height={row.h}
                rx="8"
                fill={i < 3 ? `url(#${gradSoftId})` : `url(#${gradStrongId})`}
                stroke={stroke}
                strokeWidth="1"
                opacity={i < 3 ? 1 : 1}
                filter={i === 3 ? `url(#${shadowId})` : undefined}
              />
              {svgText({
                x: 500,
                y: row.y + 12,
                size: 10,
                weight: 700,
                fill: i < 3 ? accent : contrast,
                children: row.t1,
              })}
              {svgText({
                x: 500,
                y: row.y + 23,
                size: 8,
                weight: 500,
                fill: i < 3 ? accent : contrast,
                children: row.t2,
              })}
            </g>
          ))}

          <rect
            x="200"
            y="206"
            width="320"
            height="56"
            rx="14"
            fill={`url(#${gradStrongId})`}
            filter={`url(#${shadowId})`}
          />
          {svgText({ x: 360, y: 232, size: 15, weight: 800, fill: contrast, children: 'Agentic LLM — policy' })}
          {svgText({
            x: 360,
            y: 252,
            size: 9,
            weight: 500,
            fill: contrast,
            children: 'merge scores + RAG → summary · recommended_action',
          })}

          <path
            d="M 132 164 C 132 182, 220 198, 280 206"
            fill="none"
            stroke={stroke}
            strokeWidth="2.5"
            markerEnd={`url(#${arrowId})`}
          />
          <path
            d="M 500 176 C 500 188, 430 198 360 206"
            fill="none"
            stroke={stroke}
            strokeWidth="2.5"
            markerEnd={`url(#${arrowId})`}
          />
          {svgText({ x: 360, y: 198, size: 9, weight: 700, fill: accent, children: 'fuse inputs' })}

          <rect
            x="258"
            y="278"
            width="204"
            height="40"
            rx="12"
            fill={alpha(trust, 0.12)}
            stroke={trust}
            strokeWidth="2"
            strokeDasharray="6 4"
          />
          {svgText({ x: 360, y: 298, size: 12, weight: 800, fill: trust, children: 'Trust · attestation' })}
          {svgText({ x: 360, y: 312, size: 8, weight: 600, fill: trust, children: 'hash report · verify on-chain' })}
          <line
            x1="360"
            y1="262"
            x2="360"
            y2="278"
            stroke={trust}
            strokeWidth="2.5"
            strokeDasharray="6 4"
            markerEnd={`url(#${arrowTrustId})`}
          />

          {svgText({ x: 360, y: 338, size: 11, weight: 800, fill: accent, children: 'Map recommended_action → SOC playbooks' })}

          {[
            { x: 72, w: 112, label: 'RAN' },
            { x: 304, w: 112, label: 'Edge' },
            { x: 536, w: 112, label: 'CORE' },
          ].map((b) => (
            <g key={b.label}>
              <rect
                x={b.x}
                y="348"
                width={b.w}
                height="28"
                rx="8"
                fill={alpha(primary, 0.2)}
                stroke={alpha(primary, 0.45)}
                strokeWidth="1.5"
              />
              {svgText({
                x: b.x + b.w / 2,
                y: 366,
                size: 12,
                weight: 800,
                fill: primaryDark,
                children: b.label,
              })}
            </g>
          ))}

          <line
            x1="360"
            y1="318"
            x2="128"
            y2="348"
            stroke={stroke}
            strokeWidth="1.5"
            strokeDasharray="4 3"
            markerEnd={`url(#${arrowId})`}
          />
          <line
            x1="360"
            y1="318"
            x2="360"
            y2="348"
            stroke={stroke}
            strokeWidth="1.5"
            strokeDasharray="4 3"
            markerEnd={`url(#${arrowId})`}
          />
          <line
            x1="360"
            y1="318"
            x2="592"
            y2="348"
            stroke={stroke}
            strokeWidth="1.5"
            strokeDasharray="4 3"
            markerEnd={`url(#${arrowId})`}
          />
        </svg>
      </Box>
      <Stack direction="row" flexWrap="wrap" sx={{ mt: 2, gap: 1.5 }}>
        <Chip
          size="small"
          variant="outlined"
          label="Solid arrows — data & decisions"
          sx={{ fontWeight: 600, borderColor: 'divider' }}
        />
        <Chip
          size="small"
          variant="outlined"
          color="success"
          label="Dashed green — trust path"
          sx={{ fontWeight: 600 }}
        />
      </Stack>
      <Typography variant="body2" color="text.secondary" sx={{ display: 'block', mt: 1.75, lineHeight: 1.65, maxWidth: 720 }}>
        Retrieval runs as a <strong>numbered stack</strong> (chunk → retrieve → MMR → LLM). The <strong>agentic</strong> step
        consumes both prediction stats and grounded answers, then you may <strong>attest</strong> and fan out to{' '}
        <strong>RAN / Edge / CORE</strong> automations.
      </Typography>
    </Box>
  );
}

function PipelineSummaryPanel() {
  const theme = useTheme();
  return (
    <Stack spacing={3} sx={{ maxWidth: 1 }}>
      <Paper
        elevation={0}
        sx={{
          position: 'relative',
          overflow: 'hidden',
          p: { xs: 2.5, sm: 3.5 },
          borderRadius: 2,
          border: 1,
          borderColor: 'divider',
          boxShadow: (t) => `0 12px 40px ${alpha(t.palette.primary.main, 0.08)}`,
          background: (t) =>
            `linear-gradient(135deg, ${alpha(t.palette.primary.main, 0.11)} 0%, ${alpha(t.palette.background.paper, 1)} 42%, ${alpha(t.palette.success.main, 0.06)} 100%)`,
        }}
      >
        <Box
          sx={{
            position: 'absolute',
            top: -48,
            right: -32,
            width: 200,
            height: 200,
            borderRadius: '50%',
            bgcolor: (t) => alpha(t.palette.primary.main, 0.08),
            pointerEvents: 'none',
          }}
        />
        <Box sx={{ position: 'relative' }}>
          <Stack direction="row" alignItems="center" flexWrap="wrap" gap={1} sx={{ mb: 1.5 }}>
            <Chip size="small" color="primary" label="End-to-end" sx={{ fontWeight: 700 }} />
            <Chip size="small" variant="outlined" label="VFL" sx={{ fontWeight: 600 }} />
            <Chip size="small" variant="outlined" label="RAG + MMR" sx={{ fontWeight: 600 }} />
            <Chip size="small" variant="outlined" label="Agentic policy" sx={{ fontWeight: 600 }} />
          </Stack>
          <Typography
            variant="h4"
            sx={{
              fontWeight: 800,
              letterSpacing: -0.5,
              mb: 1.25,
              background: `linear-gradient(90deg, ${theme.palette.primary.main}, ${theme.palette.success.main})`,
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              color: 'primary.main',
            }}
          >
            Pipeline overview
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mb: 2.75, maxWidth: 720, lineHeight: 1.7, fontSize: 16 }}>
            Datasets → VFL training → predictions → knowledge (RAG) → agentic policy → trust attestation. Each stage
            maps to Summary, Data & training, or Predict, RAG & agent—use this page as your map.
          </Typography>
          <Divider sx={{ mb: 3, borderColor: (t) => alpha(t.palette.divider, 0.9) }} />
          <PipelineInteractiveFlow />
        </Box>
      </Paper>

      <Paper
        elevation={0}
        sx={{
          p: { xs: 2, sm: 3 },
          borderRadius: 2,
          border: 1,
          borderColor: (t) => alpha(t.palette.primary.main, 0.22),
          borderLeftWidth: 5,
          borderLeftColor: 'primary.main',
          bgcolor: (t) => alpha(t.palette.primary.main, 0.03),
          boxShadow: (t) => `0 8px 32px ${alpha(t.palette.common.black, 0.06)}`,
        }}
      >
        <Stack direction="row" alignItems="flex-start" spacing={2} sx={{ mb: 2.5 }}>
          <Box
            sx={{
              width: 48,
              height: 48,
              borderRadius: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              background: (t) =>
                `linear-gradient(145deg, ${alpha(t.palette.primary.main, 0.2)} 0%, ${alpha(t.palette.primary.dark, 0.12)} 100%)`,
              color: 'primary.main',
              boxShadow: (t) => `0 4px 14px ${alpha(t.palette.primary.main, 0.25)}`,
            }}
          >
            <Iconify icon="solar:chat-round-dots-bold" width={28} />
          </Box>
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="h6" sx={{ fontWeight: 800, lineHeight: 1.25, letterSpacing: -0.2 }}>
              Agent decision tree
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.75, lineHeight: 1.65, maxWidth: 680 }}>
              Two inputs converge on the policy model: <strong>batch prediction signals</strong> and a <strong>four-step RAG
              stack</strong>. After the decision, you can anchor trust, then route <code>recommended_action</code> to network
              roles.
            </Typography>
          </Box>
        </Stack>
        <PipelineDecisionTreeDiagram embedded />
      </Paper>

      <Box>
        <Typography
          variant="overline"
          sx={{ fontWeight: 700, letterSpacing: 0.8, color: 'text.secondary', mb: 1.5, display: 'block' }}
        >
          Stage reference
        </Typography>
        <Stack spacing={0.75}>
          {PIPELINE_STEPS.map((s) => (
            <Accordion
              key={s.label}
              defaultExpanded={false}
              disableGutters
              sx={{
                border: 1,
                borderColor: 'divider',
                borderRadius: 1.5,
                overflow: 'hidden',
                boxShadow: 'none',
                '&:before': { display: 'none' },
              }}
            >
              <AccordionSummary
                expandIcon={<Iconify icon="eva:arrow-ios-downward-fill" width={18} />}
                sx={{ px: 2, py: 1.25, minHeight: 48, '&.Mui-expanded': { minHeight: 48 } }}
              >
                <Stack spacing={0.25} sx={{ pr: 1, textAlign: 'left' }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                    {s.label}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ lineHeight: 1.4, display: { xs: 'none', sm: 'block' } }}
                  >
                    {s.summary}
                  </Typography>
                </Stack>
              </AccordionSummary>
              <AccordionDetails sx={{ pt: 0, px: 2, pb: 2 }}>
                <Typography
                  variant="caption"
                  color="text.secondary"
                  sx={{ display: { xs: 'block', sm: 'none' }, mb: 1.5, lineHeight: 1.5 }}
                >
                  {s.summary}
                </Typography>
                <Box
                  component="ul"
                  sx={{ m: 0, pl: 2.25, typography: 'body2', color: 'text.secondary', '& li': { mb: 0.5 } }}
                >
                  {s.bullets.map((b, i) => (
                    <li key={`${s.label}-${i}`}>{b}</li>
                  ))}
                </Box>
                <Typography
                  variant="caption"
                  color="text.disabled"
                  sx={{
                    display: 'block',
                    mt: 1.5,
                    fontFamily: 'monospace',
                    fontSize: 11,
                    wordBreak: 'break-all',
                    lineHeight: 1.5,
                  }}
                >
                  {s.apis}
                </Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </Stack>
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

  const load = useCallback(async () => {
    setLoading(true);
    onNotify(null);
    try {
      const list = await listDatasets();
      setRows([...list].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
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
              <TableCell>Uploaded</TableCell>
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
            {rows.map((r) => (
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

  const loadDatasets = useCallback(async () => {
    try {
      const list = await listDatasets();
      setDatasets([...list].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()));
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
              ? 'Upload a CSV under Data & training → Datasets first.'
              : 'Choose by file name and version'
          }
        >
          {datasets.length === 0 && (
            <MenuItem value="" disabled>
              No datasets
            </MenuItem>
          )}
          {datasets.map((d) => (
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
              <TableCell>Updated</TableCell>
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
            {jobs.map((j) => (
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
            <TableCell>public_id</TableCell>
            <TableCell>algorithm</TableCell>
            <TableCell>version</TableCell>
            <TableCell>metrics</TableCell>
            <TableCell align="right">View</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {models.length === 0 && (
            <TableRow>
              <TableCell colSpan={6}>
                <Typography variant="body2" color="text.secondary">
                  No registered models yet.
                </Typography>
              </TableCell>
            </TableRow>
          )}
          {models.map((m) => (
            <TableRow key={m.public_id}>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{m.public_id}</TableCell>
              <TableCell>{m.algorithm}</TableCell>
              <TableCell>{m.version_number}</TableCell>
              <TableCell sx={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {m.metrics_json ? JSON.stringify(m.metrics_json) : '—'}
              </TableCell>
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
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <ModelVersionDetailDialog
        open={Boolean(modelDetail)}
        model={modelDetail}
        onClose={() => setModelDetail(null)}
      />
    </Stack>
  );
}

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
        return m[0]?.public_id ?? '';
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
              ? 'Train a model under Data & training → Training & models first.'
              : 'Dropdown lists registered models. Pick which weights to use for scoring.'
          }
        >
          {models.length === 0 && (
            <MenuItem value="" disabled>
              No models
            </MenuItem>
          )}
          {models.map((m) => (
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
            {inputs.map((f) => (
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
                  <TableCell>created</TableCell>
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
                  predictionJobsList.map((j) => (
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
            {predictionJobsList.map((j) => (
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
  const [rows, setRows] = useState<Awaited<ReturnType<typeof kbListFiles>>>([]);
  const [kbUploading, setKbUploading] = useState(false);

  const load = useCallback(async () => {
    try {
      setRows(await kbListFiles());
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    }
  }, [onNotify]);

  useEffect(() => {
    load();
  }, [load]);

  const onUpload: React.ChangeEventHandler<HTMLInputElement> = async (ev) => {
    const file = ev.target.files?.[0];
    ev.target.value = '';
    if (!file) return;
    onNotify(null);
    setKbUploading(true);
    try {
      const r = await kbUpload(file);
      onNotify({ severity: 'success', text: `Indexed ${r.chunk_count} chunks → ${r.kb_public_id}` });
      await load();
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setKbUploading(false);
    }
  };

  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button variant="contained" component="label" disabled={kbUploading}>
          {kbUploading ? 'Uploading…' : 'Upload KB document'}
          <input type="file" hidden onChange={onUpload} disabled={kbUploading} />
        </Button>
        <Button onClick={load} disabled={kbUploading}>
          Refresh KB list
        </Button>
        {kbUploading && <CircularProgress size={22} aria-label="Uploading document" />}
      </Stack>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>public_id</TableCell>
            <TableCell>chunks</TableCell>
            <TableCell>embedding</TableCell>
            <TableCell align="right">actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((r) => (
            <TableRow key={r.public_id}>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{r.public_id}</TableCell>
              <TableCell>{r.chunk_count}</TableCell>
              <TableCell sx={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.embedding_model}</TableCell>
              <TableCell align="right">
                <Button
                  size="small"
                  color="error"
                  onClick={async () => {
                    if (!window.confirm(`Delete KB ${r.public_id}?`)) return;
                    try {
                      await kbDelete(r.public_id);
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
          ))}
        </TableBody>
      </Table>

      <Alert severity="info" variant="outlined">
        RAG templates, multi-query retrieval, and KB LLM synthesis live under <strong>Predict, RAG &amp; agent → RAG &amp; LLM
        prep</strong>. Upload and curate documents here on <strong>Knowledge base</strong>.
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

/** List-row fields: inference job status plus batch flag counts (no per-row label without full results). */
function predictionJobBatchCaption(j: PredictionJobListItem): string {
  const parts = [`job ${j.status}`];
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
  const completedJobs = jobRows.filter((j) => j.status === 'completed');
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
    <Stack spacing={2}>
      <Alert severity="info" variant="outlined">
        Choose a <strong>prediction job</strong> by <strong>public_id</strong> in the dropdown — the app loads{' '}
        <strong>full results for all rows</strong> (<code>include_results</code>). Use <strong>All rows</strong> for batch
        templates, or pick a single row for SHAP-aware queries. <strong>Reload all row results from server</strong> refreshes that
        payload without changing the selection.
      </Alert>

      <Accordion
        defaultExpanded={false}
        disableGutters
        elevation={0}
        sx={{ border: 1, borderColor: 'divider', borderRadius: 1, '&:before': { display: 'none' } }}
      >
        <AccordionSummary expandIcon={<Iconify width={20} icon="eva:arrow-ios-downward-fill" />}>
          <Typography variant="subtitle2">5G / 6G mobile network context (RAN · Edge · Core)</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={1.5}>
            <Typography variant="body2" color="text.secondary">
              Use this mental model when reading SHAP parties and RAG hits: <strong>RAN</strong> covers radio access
              (gNB, NR air interface, O-RAN near-RT / non-RT RIC), <strong>Edge</strong> is MEC and UPF user-plane
              offload close to subscribers, and <strong>Core</strong> is the 5GC control / user-plane anchor (AMF, SMF,
              UPF, slicing, NEF/NRF). <strong>6G</strong> extends this with IMT-2030 themes (AI-native RAN, ISAC, NTN)—treat
              KB matches as forward-looking research where applicable.
            </Typography>
            <Typography variant="body2" color="text.secondary">
              After you load a <strong>completed prediction job</strong>, the API adds a template pack{' '}
              <strong>5G / 6G mobile network (RAN · Edge · Core)</strong> with retrieval strings and an LLM synthesis
              prompt aligned to that batch. Row-level SHAP templates append the same framing so vectors stay relevant to
              mobile operator SOC work.
            </Typography>
          </Stack>
        </AccordionDetails>
      </Accordion>

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
        helperText={`${jobRows.length} prediction job(s) (${completedJobs.length} completed) · choosing a job loads full results_json (all rows) for batch or row-level RAG`}
      >
        <MenuItem value="">
          <em>Select a prediction job</em>
        </MenuItem>
        {renderPredictionJobOrphanMenuItem(selectedJobId, jobRows)}
        {jobRows.map((j) => (
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

      {loadedJob && (
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
      )}

      {predCtx?.templates && predCtx.templates.length > 0 && (
        <>
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
                    : 'MMR off (fusion rerank only)'}
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
                  <MenuItem value="fusion_mmr">Fusion rerank + MMR</MenuItem>
                  <MenuItem value="fusion_only">Fusion rerank only (no MMR)</MenuItem>
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
        </>
      )}

      {selectedTemplate && multiHits.length > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Results · top documents for agentic handoff
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1.5 }}>
            Template pack: <strong>{selectedTemplate.label}</strong> — {selectedTemplate.description}. KB synthesis LLM runs under{' '}
            <strong>Predict, RAG &amp; agent → Agentic actions</strong> after you save prep.
          </Typography>

          <Box sx={{ width: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              Top {topHitsForLlm.length} document(s) (cap {finalDocCount}) —{' '}
              {retrievalPipeline === 'fusion_mmr' ? 'rerank + MMR' : 'rerank only'}
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
                No prep in this browser session. Save from <strong>RAG &amp; LLM prep</strong>, then <strong>Reload prep</strong>.
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
                Orchestration prompt: prediction summary, <code>sample_data</code> (prediction_row.shap.per_feature = top 10 |value|),
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
                    <TableCell sx={{ whiteSpace: 'nowrap' }}>Created</TableCell>
                    <TableCell align="right">Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {agentReportsForTable.map((r) => (
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
