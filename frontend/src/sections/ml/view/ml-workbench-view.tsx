import type {
  AlgorithmName,
  ManagedFileOut,
  TrainingJobOut,
  DatasetPreviewOut,
  ModelVersionOut,
  RAGTemplateItem,
  KBQueryHit,
  KBRAGLatestPredictionResponse,
  AgenticReportOut,
  AgenticActionPreset,
} from 'src/api/types';
import type { IconifyName } from 'src/components/iconify';

import { Fragment, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Tabs from '@mui/material/Tabs';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Checkbox from '@mui/material/Checkbox';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import Paper from '@mui/material/Paper';
import Collapse from '@mui/material/Collapse';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import { alpha, useTheme } from '@mui/material/styles';

import { Iconify } from 'src/components/iconify';
import { fDateTime } from 'src/utils/format-time';
import { readAgenticPrep, writeAgenticPrep } from 'src/sections/ml/agentic-prep-storage';

import { DashboardContent } from 'src/layouts/dashboard';
import {
  ApiError,
  kbDelete,
  kbRagLlm,
  kbQueryMulti,
  kbRagTemplatesLatestPrediction,
  kbUpload,
  getHealth,
  listModels,
  agentDecide,
  listAgentReports,
  getAgentReport,
  kbListFiles,
  listDatasets,
  deleteDataset,
  getApiBaseUrl,
  startTraining,
  uploadDataset,
  getTrainingJob,
  rebuildTraining,
  startPrediction,
  getPredictionJob,
  listTrainingJobs,
  getDatasetPreview,
  uploadPredictionInput,
  listPredictionInputs,
} from 'src/services';

// ----------------------------------------------------------------------

function formatError(e: unknown): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error) return e.message;
  return String(e);
}

export function MlWorkbenchView() {
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

      <Card>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} variant="scrollable" scrollButtons="auto">
          <Tab label="Pipeline summary" />
          <Tab label="Datasets" />
          <Tab label="Training & models" />
          <Tab label="Predictions" />
          <Tab label="Knowledge base" />
          <Tab label="Agentic actions" />
        </Tabs>
        <CardContent>
          {tab === 0 && <PipelineSummaryPanel />}
          {tab === 1 && <DatasetsPanel onNotify={setBanner} />}
          {tab === 2 && <TrainingPanel onNotify={setBanner} />}
          {tab === 3 && <PredictionsPanel onNotify={setBanner} />}
          {tab === 4 && <KbPanel onNotify={setBanner} />}
          {tab === 5 && <AgenticActionsPanel onNotify={setBanner} />}
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
    summary: 'Register and version the training CSV that anchors every later step.',
    bullets: [
      'Upload a training CSV; the API returns a public_id and version for traceability.',
      'Use optional replace public_id to append a new version to an existing dataset chain.',
      'Open preview to inspect column names and sample rows before you train.',
    ],
    apis: 'POST /datasets/upload · GET /datasets · GET /datasets/{public_id}/preview',
  },
  {
    label: 'Training & models',
    summary: 'Fit a classifier from the selected dataset and register the artifact as a model version.',
    bullets: [
      'Pick dataset public_id, target column, and algorithm (e.g. random_forest, xgboost, or vfl on the server).',
      'Jobs run asynchronously (pending → running → completed); metrics and model_version_public_id appear when done.',
      'VFL mode uses a fixed three-party vertical split (embed → fuse → classify); optional agent-definitions JSON steers column ownership.',
      'Rebuild clones dataset and hyperparameters into a fresh job for retraining.',
    ],
    apis: 'POST /training/start · GET /training · POST /training/rebuild · GET /models',
  },
  {
    label: 'Predictions',
    summary: 'Score new rows with a registered model and store an output CSV.',
    bullets: [
      'Upload a prediction CSV, then pass model_version_public_id and input file public_id to start a batch job.',
      'Outputs include predicted_label, max_class_probability, and optional anomaly / attack flags from job config.',
      'Poll the prediction job until status is completed, then download from storage or use the API summary.',
    ],
    apis: 'POST /predictions/upload-input · POST /predictions/start · GET /predictions/{public_id}',
  },
  {
    label: 'Knowledge base (LLM RAG stack)',
    summary:
      'Semantic document splitting, vector index, multi-query fusion with reranking, MMR-diverse chunks, then LLM synthesis over citations.',
    bullets: [
      'Semantic splitter: PDF/text → overlapping chunks (size + stride from server config) → embedding model → per-document FAISS index.',
      'Multi-query retrieval: several template queries fused with RRF + max-score rerank, then MMR so the LLM sees diverse, non-redundant passages.',
      'LLM RAG: /kb/rag-llm sends retrieved documents + synthesis prompt; optional precomputed_citations skips re-search when you already ran /kb/query-multi.',
      'Hand off ranked documents to Agentic actions together with the prediction job for the policy LLM path.',
    ],
    apis:
      'GET /kb/rag-templates/latest-prediction · POST /kb/query-multi · POST /kb/rag-llm · POST /kb/upload · GET /kb/files',
  },
  {
    label: 'Agentic actions',
    summary:
      'Policy LLM merges prediction batch statistics with KB evidence (MMR-ranked docs); emits recommended_action for downstream orchestration.',
    bullets: [
      'Inputs: prediction job summary (rows, flags, label preview) + optional kb_citations from the RAG tab (same stack: multi-query → rerank → MMR).',
      'Output: structured summary + recommended_action (e.g. block_ip, alert_admin, monitor); stored as agentic_reports with expandable audit tree.',
      'Maps to vertical-FL roles for execution: RAN (access edge), Edge (aggregation), CORE (policy SOC) — wire automations per your deployment.',
      'Pair with the trust layer: once the policy output is fixed, you can commit a hash of the report for independent verification.',
    ],
    apis: 'POST /agent/decide · GET /agent/reports · GET /agent/reports/{public_id}',
  },
  {
    label: 'Trust layer (blockchain)',
    summary:
      'Anchor agentic action records so stakeholders can verify integrity, ordering, and non-repudiation outside the app database alone.',
    bullets: [
      'After an agentic run, publish an attestation: e.g. hash(summary ∥ recommended_action ∥ prediction_job_public_id ∥ report timestamp).',
      'Store the transaction id / ledger anchor alongside report_path so audits can reconcile API rows with an immutable trust layer.',
      'Downstream “actions taken” (RAN / Edge / CORE playbooks) can log execution receipts that optionally reference the same attestation hash.',
      'Verifiers re-hash the same payload and compare to the commitment recorded on-chain (or on an enterprise distributed log).',
    ],
    apis: '(roadmap) chain notary / Web3 adapter — wire to your ledger; not exposed in this demo API yet.',
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
    tabHint: 'Training',
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
      <Typography variant="subtitle2" sx={{ mb: 1.5, fontWeight: 700 }}>
        Interactive flow — tap a stage to highlight (open the matching tab to work there)
      </Typography>
      <Stack
        direction="row"
        spacing={0.5}
        alignItems="stretch"
        sx={{ overflowX: 'auto', pb: 1, px: 0.5, mx: -0.5 }}
      >
        {PIPELINE_STAGE_CARDS.map((n, i) => {
          const isChain = n.id === 'chain';
          const isActive = active === i;
          return (
            <Fragment key={n.id}>
              <Card
                elevation={isActive ? 6 : 0}
                onClick={() => setActive((v) => (v === i ? null : i))}
                sx={{
                  minWidth: 128,
                  flex: '0 0 auto',
                  cursor: 'pointer',
                  borderRadius: 2,
                  border: 2,
                  borderColor: isActive
                    ? isChain
                      ? 'success.main'
                      : 'primary.main'
                    : 'divider',
                  bgcolor: (t) =>
                    alpha(
                      isChain ? t.palette.success.main : t.palette.primary.main,
                      isActive ? 0.14 : 0.04
                    ),
                  transition: 'transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: theme.shadows[isActive ? 12 : 6],
                  },
                }}
              >
                <CardContent sx={{ py: 1.5, px: 1.5, '&:last-child': { pb: 1.5 } }}>
                  <Iconify
                    width={32}
                    icon={n.icon}
                    sx={{ color: isChain ? 'success.main' : 'primary.main', mb: 0.75 }}
                  />
                  <Typography variant="subtitle2" sx={{ lineHeight: 1.25 }}>
                    {n.title}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block">
                    {n.sub}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.disabled"
                    sx={{ display: 'block', mt: 0.75, fontSize: 10 }}
                  >
                    Tab: {n.tabHint}
                  </Typography>
                </CardContent>
              </Card>
              {i < PIPELINE_STAGE_CARDS.length - 1 && (
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    px: 0.25,
                    color: 'text.disabled',
                    flexShrink: 0,
                  }}
                >
                  <Iconify icon="eva:arrow-ios-forward-fill" width={22} />
                </Box>
              )}
            </Fragment>
          );
        })}
      </Stack>
      <Collapse in={active !== null} unmountOnExit>
        {active !== null && (
          <Alert severity="info" variant="outlined" sx={{ mt: 1.5 }}>
            <Typography variant="body2">
              <strong>{PIPELINE_STAGE_CARDS[active].title}</strong> — use the{' '}
              <strong>{PIPELINE_STAGE_CARDS[active].tabHint}</strong> tab. The pipeline runs left → right; trust is an optional
              cap for verifiable decisions.
            </Typography>
          </Alert>
        )}
      </Collapse>
    </Box>
  );
}

function PipelineDecisionTreeDiagram() {
  const theme = useTheme();
  const nodeFill = theme.palette.primary.main;
  const nodeSub = theme.palette.primary.contrastText;
  const stroke = theme.palette.divider;
  const accent = theme.palette.text.secondary;
  const trustStroke = theme.palette.success.main;
  const trustFill = theme.palette.success.main;

  return (
    <Box
      sx={{
        p: 2,
        borderRadius: 2,
        border: 1,
        borderColor: 'divider',
        bgcolor: 'background.neutral',
      }}
    >
      <Typography variant="subtitle2" sx={{ mb: 0.5, fontWeight: 700 }}>
        Agent decision tree (how signals combine)
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1.5 }}>
        Predictions and the full LLM·RAG path feed the agentic policy; optional blockchain attestation and RAN / Edge / CORE
        actions follow.
      </Typography>
      <Box sx={{ width: 1, overflowX: 'auto' }}>
        <svg
          width="100%"
          height={278}
          viewBox="0 0 640 278"
          role="img"
          aria-label="Decision tree: predictions and RAG into agentic LLM, blockchain attestation, and RAN Edge CORE actions"
        >
          <defs>
            <marker
              id="tree-arrowhead"
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L7,3.5 L0,7 Z" fill={accent} />
            </marker>
            <marker
              id="tree-arrowhead-trust"
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L7,3.5 L0,7 Z" fill={trustStroke} />
            </marker>
          </defs>
          <text x={320} y={13} textAnchor="middle" fill={accent} style={{ fontSize: 11, fontFamily: 'inherit' }}>
            LLM RAG (semantic split → multi-query + rerank + MMR → LLM) ∥ Predictions batch
          </text>
          <rect x={16} y={28} width={168} height={52} rx={8} fill={nodeFill} opacity={0.78} />
          <text x={100} y={50} textAnchor="middle" fill={nodeSub} style={{ fontSize: 12, fontWeight: 700, fontFamily: 'inherit' }}>
            Predictions
          </text>
          <text x={100} y={66} textAnchor="middle" fill={nodeSub} opacity={0.9} style={{ fontSize: 9, fontFamily: 'inherit' }}>
            batch job · labels · flags
          </text>
          <rect x={380} y={26} width={244} height={22} rx={4} fill={nodeFill} opacity={0.45} stroke={stroke} strokeWidth={1} />
          <text x={502} y={41} textAnchor="middle" fill={accent} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            ① Semantic splitter + embed (chunks → FAISS)
          </text>
          <line x1={502} y1={48} x2={502} y2={54} stroke={stroke} strokeWidth={1.5} markerEnd="url(#tree-arrowhead)" />
          <rect x={380} y={54} width={244} height={22} rx={4} fill={nodeFill} opacity={0.45} stroke={stroke} strokeWidth={1} />
          <text x={502} y={69} textAnchor="middle" fill={accent} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            ② Multi-query + RRF / max-score rerank
          </text>
          <line x1={502} y1={76} x2={502} y2={82} stroke={stroke} strokeWidth={1.5} markerEnd="url(#tree-arrowhead)" />
          <rect x={380} y={82} width={244} height={22} rx={4} fill={nodeFill} opacity={0.45} stroke={stroke} strokeWidth={1} />
          <text x={502} y={97} textAnchor="middle" fill={accent} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            ③ MMR — diverse documents to LLM
          </text>
          <line x1={502} y1={104} x2={502} y2={110} stroke={stroke} strokeWidth={1.5} markerEnd="url(#tree-arrowhead)" />
          <rect x={380} y={110} width={244} height={26} rx={6} fill={nodeFill} opacity={0.88} />
          <text x={502} y={127} textAnchor="middle" fill={nodeSub} style={{ fontSize: 11, fontWeight: 700, fontFamily: 'inherit' }}>
            LLM RAG synthesis
          </text>
          <text x={502} y={140} textAnchor="middle" fill={nodeSub} opacity={0.85} style={{ fontSize: 8, fontFamily: 'inherit' }}>
            citations + template prompt → answer
          </text>
          <rect x={200} y={152} width={240} height={44} rx={8} fill={nodeFill} />
          <text x={320} y={172} textAnchor="middle" fill={nodeSub} style={{ fontSize: 13, fontWeight: 700, fontFamily: 'inherit' }}>
            Agentic LLM (policy)
          </text>
          <text x={320} y={188} textAnchor="middle" fill={nodeSub} opacity={0.88} style={{ fontSize: 9, fontFamily: 'inherit' }}>
            prediction summary + KB docs → summary · recommended_action
          </text>
          <line
            x1={100}
            y1={80}
            x2={260}
            y2={152}
            stroke={stroke}
            strokeWidth={2}
            markerEnd="url(#tree-arrowhead)"
          />
          <line
            x1={502}
            y1={136}
            x2={380}
            y2={152}
            stroke={stroke}
            strokeWidth={2}
            markerEnd="url(#tree-arrowhead)"
          />
          <rect
            x={238}
            y={208}
            width={164}
            height={32}
            rx={8}
            fill={trustFill}
            opacity={0.2}
            stroke={trustStroke}
            strokeWidth={2}
            strokeDasharray="4 3"
          />
          <text x={320} y={228} textAnchor="middle" fill={trustStroke} style={{ fontSize: 11, fontWeight: 700, fontFamily: 'inherit' }}>
            Blockchain attestation
          </text>
          <line
            x1={320}
            y1={196}
            x2={320}
            y2={208}
            stroke={trustStroke}
            strokeWidth={2}
            strokeDasharray="5 3"
            markerEnd="url(#tree-arrowhead-trust)"
          />
          <text x={320} y={252} textAnchor="middle" fill={accent} style={{ fontSize: 10, fontWeight: 600, fontFamily: 'inherit' }}>
            Actions taken · network roles
          </text>
          <rect x={48} y={258} width={100} height={18} rx={4} fill={nodeFill} opacity={0.55} />
          <text x={98} y={270} textAnchor="middle" fill={nodeSub} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            RAN
          </text>
          <rect x={270} y={258} width={100} height={18} rx={4} fill={nodeFill} opacity={0.55} />
          <text x={320} y={270} textAnchor="middle" fill={nodeSub} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            Edge
          </text>
          <rect x={492} y={258} width={100} height={18} rx={4} fill={nodeFill} opacity={0.55} />
          <text x={542} y={270} textAnchor="middle" fill={nodeSub} style={{ fontSize: 9, fontWeight: 700, fontFamily: 'inherit' }}>
            CORE
          </text>
          <line x1={320} y1={240} x2={98} y2={258} stroke={stroke} strokeWidth={1} strokeDasharray="3 2" markerEnd="url(#tree-arrowhead)" />
          <line x1={320} y1={240} x2={320} y2={258} stroke={stroke} strokeWidth={1} strokeDasharray="3 2" markerEnd="url(#tree-arrowhead)" />
          <line x1={320} y1={240} x2={542} y2={258} stroke={stroke} strokeWidth={1} strokeDasharray="3 2" markerEnd="url(#tree-arrowhead)" />
        </svg>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5 }}>
        Documents are split and embedded; retrieval uses <strong>multi-query fusion + reranking + MMR</strong>; the synthesis LLM
        consumes those passages. That evidence joins <strong>prediction batch outputs</strong> in the <strong>agentic</strong>{' '}
        policy. You can optionally <strong>commit hashes on-chain</strong>, then map <strong>recommended_action</strong> to
        playbooks at <strong>RAN</strong> (access), <strong>Edge</strong> (local SOAR), and <strong>CORE</strong> (SOC / policy).
      </Typography>
    </Box>
  );
}

function PipelineSummaryPanel() {
  const [treeOpen, setTreeOpen] = useState(false);

  return (
    <Stack spacing={3}>
      <Paper
        elevation={0}
        sx={{
          p: { xs: 2, sm: 3 },
          borderRadius: 2,
          overflow: 'hidden',
          position: 'relative',
          border: 1,
          borderColor: 'divider',
          background: (t) =>
            `linear-gradient(125deg, ${alpha(t.palette.primary.main, 0.14)} 0%, ${alpha(t.palette.primary.dark, 0.05)} 42%, ${alpha(t.palette.success.main, 0.1)} 100%)`,
          '&::after': {
            content: '""',
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            background: (t) =>
              `radial-gradient(ellipse 80% 50% at 100% 0%, ${alpha(t.palette.common.white, 0.12)} 0%, transparent 55%)`,
          },
        }}
      >
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          spacing={2}
          alignItems={{ sm: 'flex-start' }}
          justifyContent="space-between"
          sx={{ position: 'relative', zIndex: 1 }}
        >
          <Box sx={{ maxWidth: 640 }}>
            <Typography variant="overline" color="primary" sx={{ fontWeight: 800, letterSpacing: 1.2 }}>
              Pipeline overview
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 800, mt: 0.25, mb: 1, lineHeight: 1.25 }}>
              From datasets to agentic actions
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Ingest versioned data, train and score models, ground answers in your knowledge base, then let the policy LLM
              propose next steps. Optional blockchain attestation links critical outputs to verifiable records.
            </Typography>
          </Box>
          <Chip
            icon={<Iconify icon="solar:share-bold" width={18} />}
            label="Explore stages below"
            color="primary"
            variant="outlined"
            sx={{ alignSelf: { xs: 'flex-start', sm: 'center' }, fontWeight: 600 }}
          />
        </Stack>
      </Paper>

      <Paper variant="outlined" sx={{ p: { xs: 2, sm: 2.5 }, borderRadius: 2 }}>
        <PipelineInteractiveFlow />
      </Paper>

      <Box>
        <Button
          variant="outlined"
          color="inherit"
          size="small"
          endIcon={
            <Iconify icon={treeOpen ? 'eva:arrow-ios-upward-fill' : 'eva:arrow-ios-downward-fill'} width={20} />
          }
          onClick={() => setTreeOpen((v) => !v)}
          sx={{ mb: 1, fontWeight: 600 }}
        >
          {treeOpen ? 'Hide' : 'Show'} decision tree diagram
        </Button>
        <Collapse in={treeOpen}>
          <PipelineDecisionTreeDiagram />
        </Collapse>
      </Box>

      <Typography variant="subtitle2" sx={{ fontWeight: 800 }}>
        Stage details
      </Typography>
      {PIPELINE_STEPS.map((s, idx) => (
        <Accordion
          key={s.label}
          defaultExpanded={idx === 0}
          disableGutters
          sx={{
            border: 1,
            borderColor: 'divider',
            borderRadius: 1,
            overflow: 'hidden',
            '&:before': { display: 'none' },
            mb: 1,
          }}
        >
          <AccordionSummary expandIcon={<Iconify icon="eva:arrow-ios-downward-fill" width={20} />}>
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              {s.label}
            </Typography>
          </AccordionSummary>
          <AccordionDetails sx={{ pt: 0 }}>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
              {s.summary}
            </Typography>
            <Box component="ul" sx={{ m: 0, pl: 2.5, typography: 'body2', color: 'text.secondary' }}>
              {s.bullets.map((b, i) => (
                <li key={`${s.label}-${i}`}>{b}</li>
              ))}
            </Box>
            <Typography
              variant="caption"
              color="text.disabled"
              sx={{ display: 'block', mt: 1.5, fontFamily: 'monospace', wordBreak: 'break-all' }}
            >
              {s.apis}
            </Typography>
          </AccordionDetails>
        </Accordion>
      ))}
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
  const [algorithm, setAlgorithm] = useState<AlgorithmName>('random_forest');
  const [jobs, setJobs] = useState<TrainingJobOut[]>([]);
  const [models, setModels] = useState<Awaited<ReturnType<typeof listModels>>>([]);

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

      <Typography variant="subtitle2">Start training</Typography>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'flex-start' }}>
        <TextField
          select
          label="Training dataset"
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          sx={{ minWidth: 280, flex: 1 }}
          helperText={
            datasets.length === 0
              ? 'Upload a CSV on the Datasets tab (second tab) first.'
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
        <TextField select label="Algorithm" value={algorithm} onChange={(e) => setAlgorithm(e.target.value as AlgorithmName)} sx={{ minWidth: 160 }}>
          <MenuItem value="random_forest">random_forest</MenuItem>
          <MenuItem value="xgboost">xgboost</MenuItem>
        </TextField>
        <Button
          variant="contained"
          disabled={!datasetId.trim()}
          onClick={async () => {
            onNotify(null);
            try {
              const res = await startTraining({
                dataset_file_public_id: datasetId.trim(),
                target_column: targetColumn.trim(),
                algorithm,
              });
              onNotify({ severity: 'success', text: `Training queued · job ${res.job_public_id}` });
              await loadJobs();
              await refreshModels();
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Start training
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
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>public_id</TableCell>
            <TableCell>algorithm</TableCell>
            <TableCell>version</TableCell>
            <TableCell>metrics</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {models.map((m) => (
            <TableRow key={m.public_id}>
              <TableCell sx={{ fontFamily: 'monospace', fontSize: 12 }}>{m.public_id}</TableCell>
              <TableCell>{m.algorithm}</TableCell>
              <TableCell>{m.version_number}</TableCell>
              <TableCell sx={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {m.metrics_json ? JSON.stringify(m.metrics_json) : '—'}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
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

  const refreshLists = useCallback(async () => {
    setLoadingLists(true);
    try {
      const [m, inp] = await Promise.all([listModels(), listPredictionInputs()]);
      setModels(m);
      setInputs(inp);
      setModelId((prev) => prev || (m[0]?.public_id ?? ''));
      setInputId((prev) => prev || (inp[0]?.public_id ?? ''));
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoadingLists(false);
    }
  }, [onNotify]);

  useEffect(() => {
    void refreshLists();
  }, [refreshLists]);

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
    <Stack spacing={2}>
      <Alert severity="info" variant="outlined">
        Pick the <strong>current model</strong> from the registry and the <strong>prediction CSV</strong> to score. After
        upload, that file is selected automatically; you can switch inputs from the dropdown if you have several.
      </Alert>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button variant="outlined" component="label" disabled={uploadingCsv}>
          {uploadingCsv ? 'Uploading…' : 'Upload prediction CSV'}
          <input type="file" hidden accept=".csv,text/csv" onChange={onUploadCsv} disabled={uploadingCsv} />
        </Button>
        {uploadingCsv && <CircularProgress size={22} aria-label="Uploading CSV" />}
        <Button size="small" variant="outlined" onClick={() => void refreshLists()} disabled={loadingLists || uploadingCsv}>
          {loadingLists ? 'Loading…' : 'Refresh models & inputs'}
        </Button>
      </Stack>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems={{ md: 'flex-start' }}>
        <TextField
          select
          label="Model"
          value={modelId}
          onChange={(e) => setModelId(e.target.value)}
          sx={{ minWidth: 280, flex: 1 }}
          helperText={
            models.length === 0
              ? 'Train a model on the Training & models tab first.'
              : 'Newest model is selected by default when you open this tab.'
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
        <TextField
          select
          label="Prediction input CSV"
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
          sx={{ minWidth: 280, flex: 1 }}
          helperText={
            inputs.length === 0
              ? 'Upload a CSV above to create an input file.'
              : 'Uses the file you just uploaded, or choose another from the list.'
          }
        >
          {inputs.length === 0 && (
            <MenuItem value="" disabled>
              No inputs
            </MenuItem>
          )}
          {inputs.map((f) => (
            <MenuItem key={f.public_id} value={f.public_id}>
              {f.original_name} · v{f.version} · {f.public_id.slice(0, 8)}…
            </MenuItem>
          ))}
        </TextField>
        <Button
          variant="contained"
          disabled={!modelId.trim() || !inputId.trim()}
          onClick={async () => {
            onNotify(null);
            try {
              const j = await startPrediction({
                model_version_public_id: modelId.trim(),
                input_file_public_id: inputId.trim(),
              });
              setPredJobId(j.public_id);
              setPredSummary(`${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'}`);
              onNotify({ severity: 'success', text: `Prediction job ${j.public_id}` });
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Start prediction
        </Button>
      </Stack>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="center">
        <TextField label="Prediction job public_id" fullWidth value={predJobId} onChange={(e) => setPredJobId(e.target.value)} />
        <Button
          variant="outlined"
          onClick={async () => {
            if (!predJobId.trim()) return;
            onNotify(null);
            try {
              const j = await getPredictionJob(predJobId.trim());
              setPredSummary(`${j.status} · rows ${j.rows_total ?? '—'} / flagged ${j.rows_flagged ?? '—'}`);
            } catch (e) {
              onNotify({ severity: 'error', text: formatError(e) });
            }
          }}
        >
          Refresh job
        </Button>
      </Stack>
      {predSummary && <Typography variant="body2">{predSummary}</Typography>}
    </Stack>
  );
}

const MMR_PRESETS = { focused: 0.72, balanced: 0.55, diverse: 0.38 } as const;

function KbPanel({ onNotify }: PanelProps) {
  const [rows, setRows] = useState<Awaited<ReturnType<typeof kbListFiles>>>([]);
  const [kbUploading, setKbUploading] = useState(false);
  const [predCtx, setPredCtx] = useState<KBRAGLatestPredictionResponse | null>(null);
  const [loadingCtx, setLoadingCtx] = useState(false);
  const [templateId, setTemplateId] = useState('');
  const [mmrPreset, setMmrPreset] = useState<keyof typeof MMR_PRESETS>('balanced');
  const [multiHits, setMultiHits] = useState<KBQueryHit[]>([]);
  const [multiMetaLine, setMultiMetaLine] = useState('');
  const [retrievalLoading, setRetrievalLoading] = useState(false);
  const [llmLoading, setLlmLoading] = useState(false);
  const [ragAnswer, setRagAnswer] = useState('');

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

  const selectedTemplate: RAGTemplateItem | undefined = predCtx?.templates?.find((t) => t.id === templateId);

  const loadLatestPredictionContext = useCallback(async () => {
    onNotify(null);
    setLoadingCtx(true);
    setMultiHits([]);
    setMultiMetaLine('');
    setRagAnswer('');
    try {
      const ctx = await kbRagTemplatesLatestPrediction();
      setPredCtx(ctx);
      setTemplateId(ctx.templates[0]?.id ?? '');
      if (ctx.message) {
        onNotify({ severity: 'info', text: ctx.message });
      } else if (ctx.prediction_job_public_id) {
        onNotify({
          severity: 'success',
          text: `Loaded context from prediction job ${ctx.prediction_job_public_id.slice(0, 8)}…`,
        });
      }
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    } finally {
      setLoadingCtx(false);
    }
  }, [onNotify]);

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
        RAG and LLM prompts are generated from your <strong>latest completed prediction</strong>—pick a template below. Retrieval
        fuses <strong>multiple queries</strong> (RRF + max-score) then applies <strong>MMR</strong> for diverse documents. No manual
        query typing required.
      </Alert>
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Button variant="outlined" onClick={() => void loadLatestPredictionContext()} disabled={loadingCtx}>
          {loadingCtx ? 'Loading…' : 'Load from latest prediction'}
        </Button>
        {loadingCtx && <CircularProgress size={22} />}
      </Stack>

      {predCtx?.templates && predCtx.templates.length > 0 && (
        <>
          <TextField
            select
            fullWidth
            label="RAG template (from predictions)"
            value={templateId}
            onChange={(e) => {
              setTemplateId(e.target.value);
              setMultiHits([]);
              setMultiMetaLine('');
              setRagAnswer('');
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
            <Stack
              direction={{ xs: 'column', md: 'row' }}
              spacing={2}
              alignItems="stretch"
              sx={{ '& > *': { flex: 1, minWidth: 0 } }}
            >
              <Box sx={{ p: 2, borderRadius: 1, border: 1, borderColor: 'divider' }}>
                <Typography variant="subtitle2" gutterBottom>
                  Final template · multi-query retrieval strings
                </Typography>
                <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1 }}>
                  Fused server-side with RRF + max-score rerank, then MMR (λ = {MMR_PRESETS[mmrPreset]}).
                </Typography>
                <Box component="ol" sx={{ m: 0, pl: 2.5, typography: 'body2', color: 'text.secondary' }}>
                  {selectedTemplate.retrieval_queries.map((rq, i) => (
                    <li key={i} style={{ marginBottom: 8 }}>
                      {rq}
                    </li>
                  ))}
                </Box>
              </Box>
              <Box sx={{ p: 2, borderRadius: 1, border: 1, borderColor: 'divider' }}>
                <Typography variant="subtitle2" gutterBottom>
                  LLM synthesis query (RAG tab + Agentic actions)
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: 'pre-wrap' }}>
                  {selectedTemplate.llm_prompt}
                </Typography>
              </Box>
            </Stack>
          )}

          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} alignItems={{ sm: 'center' }}>
            <TextField
              select
              label="MMR balance"
              value={mmrPreset}
              onChange={(e) => setMmrPreset(e.target.value as keyof typeof MMR_PRESETS)}
              sx={{ minWidth: 200 }}
              helperText="Higher = more relevance; lower = more diversity between chunks."
            >
              <MenuItem value="focused">Focused (relevance)</MenuItem>
              <MenuItem value="balanced">Balanced</MenuItem>
              <MenuItem value="diverse">Diverse (MMR)</MenuItem>
            </TextField>
            <Button
              variant="contained"
              disabled={!selectedTemplate || retrievalLoading}
              onClick={async () => {
                if (!selectedTemplate) return;
                onNotify(null);
                setRetrievalLoading(true);
                setRagAnswer('');
                try {
                  const res = await kbQueryMulti({
                    queries: selectedTemplate.retrieval_queries,
                    final_k: 8,
                    per_query_k: 14,
                    mmr_lambda: MMR_PRESETS[mmrPreset],
                  });
                  setMultiHits(res.hits);
                  const m = res.meta as Record<string, unknown>;
                  setMultiMetaLine(
                    `Fused ${String(m.candidates_fused ?? '?')} candidates → pool ${String(m.pool_size ?? '?')} → ${res.hits.length} docs (MMR λ=${String(m.mmr_lambda ?? '')})`
                  );
                } catch (e) {
                  onNotify({ severity: 'error', text: formatError(e) });
                } finally {
                  setRetrievalLoading(false);
                }
              }}
            >
              {retrievalLoading ? 'Retrieving…' : 'Run multi-query retrieval'}
            </Button>
            <Button
              variant="outlined"
              color="secondary"
              disabled={!selectedTemplate || multiHits.length === 0 || llmLoading}
              onClick={async () => {
                if (!selectedTemplate) return;
                onNotify(null);
                setLlmLoading(true);
                try {
                  const res = await kbRagLlm({
                    query: selectedTemplate.llm_prompt,
                    precomputed_citations: multiHits,
                  });
                  setRagAnswer(res.answer);
                } catch (e) {
                  onNotify({ severity: 'error', text: formatError(e) });
                } finally {
                  setLlmLoading(false);
                }
              }}
            >
              {llmLoading ? 'LLM…' : 'Run LLM on retrieved docs'}
            </Button>
            {multiHits.length > 0 && predCtx?.prediction_job_public_id && selectedTemplate && (
              <Button
                variant="outlined"
                color="success"
                onClick={() => {
                  writeAgenticPrep({
                    predictionJobPublicId: predCtx.prediction_job_public_id ?? null,
                    templateId: selectedTemplate.id,
                    templateLabel: selectedTemplate.label,
                    llmPrompt: selectedTemplate.llm_prompt,
                    citations: multiHits,
                    retrievalMetaLine: multiMetaLine,
                    updatedAt: new Date().toISOString(),
                  });
                  onNotify({
                    severity: 'success',
                    text: 'Context saved for Agentic actions tab (prediction job + MMR documents).',
                  });
                }}
              >
                Send to Agentic actions
              </Button>
            )}
          </Stack>
        </>
      )}

      {selectedTemplate && (multiHits.length > 0 || ragAnswer) && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle1" gutterBottom>
            Consolidated output (template · LLM query · rerank/MMR docs · KB LLM)
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 1.5 }}>
            Final template: <strong>{selectedTemplate.label}</strong> — {selectedTemplate.description}
          </Typography>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            LLM query used for synthesis
          </Typography>
          <Typography variant="body2" sx={{ mb: 2, whiteSpace: 'pre-wrap', pl: 1, borderLeft: 3, borderColor: 'primary.light' }}>
            {selectedTemplate.llm_prompt}
          </Typography>

          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="flex-start">
            <Box sx={{ flex: 1, minWidth: 0, width: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                Rerank + MMR document list
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
                    {multiHits.length === 0 ? (
                      <TableRow>
                        <TableCell colSpan={6}>
                          <Typography variant="body2" color="text.secondary">
                            Run multi-query retrieval to populate this table.
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ) : (
                      multiHits.map((h, i) => (
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
            </Box>
            <Box sx={{ flex: 1, minWidth: 0, width: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                LLM answer (Knowledge base RAG)
              </Typography>
              <Box
                sx={{
                  p: 2,
                  bgcolor: 'background.neutral',
                  borderRadius: 1,
                  minHeight: 160,
                  maxHeight: 320,
                  overflow: 'auto',
                }}
              >
                <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                  {ragAnswer || 'Run “Run LLM on retrieved docs” to fill this panel.'}
                </Typography>
              </Box>
            </Box>
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}

function AgenticResponseTree({ r }: { r: AgenticReportOut }) {
  const block = (title: string, body: string | null | undefined, defaultOpen: boolean) => (
    <Box
      key={title}
      component="details"
      open={defaultOpen}
      sx={{ mb: 1, borderLeft: 2, borderColor: 'divider', pl: 1.5 }}
    >
      <Box
        component="summary"
        sx={{
          cursor: 'pointer',
          fontWeight: 700,
          typography: 'subtitle2',
          listStyle: 'none',
          '&::-webkit-details-marker': { display: 'none' },
        }}
      >
        {title}
      </Box>
      <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', mt: 1, color: 'text.secondary' }}>
        {body?.trim() ? body : '—'}
      </Typography>
    </Box>
  );

  return (
    <Stack spacing={0}>
      {block(
        '1 · Prediction job (fixed input)',
        `public_id: ${r.prediction_job_public_id ?? '—'}\ninternal id: ${r.prediction_job_id}`,
        true
      )}
      {block('2 · Summary', r.summary, true)}
      {block('3 · Recommended action', r.recommended_action, true)}
      {block('4 · RAG / KB context (as seen by agent)', r.rag_context_used, false)}
      {block('5 · Raw LLM response', r.raw_llm_response, false)}
    </Stack>
  );
}

function AgenticActionsPanel({ onNotify }: PanelProps) {
  const [prep, setPrep] = useState(() => readAgenticPrep());
  const [predictionJobId, setPredictionJobId] = useState(() => readAgenticPrep()?.predictionJobPublicId ?? '');
  const [useRag, setUseRag] = useState(true);
  const [usePrepCitations, setUsePrepCitations] = useState(true);
  const [preset, setPreset] = useState<AgenticActionPreset>('standard');
  const [notes, setNotes] = useState('');
  const [reports, setReports] = useState<AgenticReportOut[]>([]);
  const [detail, setDetail] = useState<AgenticReportOut | null>(null);
  const [running, setRunning] = useState(false);

  const loadReports = useCallback(async () => {
    try {
      setReports(await listAgentReports(80, 0));
    } catch (e) {
      onNotify({ severity: 'error', text: formatError(e) });
    }
  }, [onNotify]);

  useEffect(() => {
    void loadReports();
  }, [loadReports]);

  const refreshPrep = useCallback(() => {
    setPrep(readAgenticPrep());
  }, []);

  useEffect(() => {
    if (prep?.predictionJobPublicId && !predictionJobId.trim()) {
      setPredictionJobId(prep.predictionJobPublicId);
    }
  }, [prep, predictionJobId]);

  return (
    <Stack spacing={2}>
      <Alert severity="info" variant="outlined">
        <strong>Agentic actions</strong> combine a <strong>prediction job</strong> with optional <strong>KB documents</strong>. Use{' '}
        <strong>Send to Agentic actions</strong> on the Knowledge base tab after MMR retrieval to fix the document list; then
        start a run here with a <strong>preset</strong> emphasis.
      </Alert>

      <Stack direction="row" spacing={1} flexWrap="wrap" alignItems="center">
        <Button size="small" variant="outlined" onClick={refreshPrep}>
          Reload prepared context
        </Button>
        <Typography variant="caption" color="text.secondary">
          Last KB handoff:{' '}
          {prep?.updatedAt ? fDateTime(prep.updatedAt) : 'none'} · {prep?.citations?.length ?? 0} chunks
        </Typography>
      </Stack>

      {prep && (prep.citations?.length ?? 0) > 0 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Preview — prepared for agent
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Template: <strong>{prep.templateLabel}</strong> · Prediction job:{' '}
            <strong>{prep.predictionJobPublicId?.slice(0, 8) ?? '—'}…</strong>
          </Typography>
          <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
            {prep.retrievalMetaLine}
          </Typography>
        </Paper>
      )}

      <TextField
        select
        label="Agentic action preset (fixed)"
        value={preset}
        onChange={(e) => setPreset(e.target.value as AgenticActionPreset)}
        sx={{ maxWidth: 360 }}
        helperText="Appends a fixed analyst emphasis line to the LLM prompt path."
      >
        <MenuItem value="standard">Standard SOC decision</MenuItem>
        <MenuItem value="containment_focus">Containment & isolation</MenuItem>
        <MenuItem value="fp_review">False-positive & evidence review</MenuItem>
      </TextField>

      <TextField
        label="Prediction job public_id"
        fullWidth
        value={predictionJobId}
        onChange={(e) => setPredictionJobId(e.target.value)}
        helperText="Filled from Knowledge base handoff or paste from Predictions tab."
      />

      <FormControlLabel control={<Checkbox checked={useRag} onChange={(e) => setUseRag(e.target.checked)} />} label="use_rag" />
      <FormControlLabel
        control={
          <Checkbox
            checked={usePrepCitations}
            onChange={(e) => setUsePrepCitations(e.target.checked)}
            disabled={!useRag || !(prep?.citations?.length ?? 0)}
          />
        }
        label="Use KB document list from Knowledge base (MMR chunks)"
      />
      <TextField label="Feature notes (optional)" fullWidth multiline minRows={2} value={notes} onChange={(e) => setNotes(e.target.value)} />

      <Button
        variant="contained"
        disabled={!predictionJobId.trim() || running}
        onClick={async () => {
          onNotify(null);
          setRunning(true);
          try {
            const r = await agentDecide({
              prediction_job_public_id: predictionJobId.trim(),
              use_rag: useRag,
              feature_notes: notes.trim() || null,
              kb_citations:
                useRag && usePrepCitations && prep?.citations?.length ? prep.citations : null,
              agent_action_preset: preset,
            });
            await loadReports();
            setDetail(r);
            onNotify({ severity: 'success', text: `Agentic action saved · ${r.public_id}` });
          } catch (e) {
            onNotify({ severity: 'error', text: formatError(e) });
          } finally {
            setRunning(false);
          }
        }}
      >
        {running ? 'Running…' : 'Start agentic action'}
      </Button>

      <Typography variant="subtitle2">Agentic actions list (preview)</Typography>
      <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: -1, mb: 0.5 }}>
        Click a row to open the full response tree.
      </Typography>
      <TableContainer sx={{ maxWidth: 1, overflowX: 'auto', border: 1, borderColor: 'divider', borderRadius: 1 }}>
        <Table size="small" stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>When</TableCell>
              <TableCell>Prediction job</TableCell>
              <TableCell>Action</TableCell>
              <TableCell>Summary</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {reports.length === 0 && (
              <TableRow>
                <TableCell colSpan={4}>
                  <Typography variant="body2" color="text.secondary">
                    No agentic runs yet.
                  </Typography>
                </TableCell>
              </TableRow>
            )}
            {reports.map((row) => (
              <TableRow
                key={row.public_id}
                hover
                sx={{ cursor: 'pointer' }}
                onClick={async () => {
                  try {
                    const fresh = await getAgentReport(row.public_id);
                    setDetail(fresh);
                  } catch (e) {
                    onNotify({ severity: 'error', text: formatError(e) });
                  }
                }}
              >
                <TableCell sx={{ whiteSpace: 'nowrap', typography: 'caption' }}>{fDateTime(row.created_at)}</TableCell>
                <TableCell sx={{ fontFamily: 'monospace', fontSize: 11 }}>
                  {row.prediction_job_public_id ?? row.prediction_job_id}
                </TableCell>
                <TableCell>
                  <Chip size="small" label={row.recommended_action} variant="outlined" />
                </TableCell>
                <TableCell sx={{ maxWidth: 360 }}>
                  <Typography variant="body2" noWrap title={row.summary}>
                    {row.summary}
                  </Typography>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>

      <Dialog open={detail !== null} onClose={() => setDetail(null)} maxWidth="md" fullWidth>
        <DialogTitle>Agentic action — response tree</DialogTitle>
        <DialogContent dividers>
          {detail && (
            <>
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mb: 2 }}>
                Report <span style={{ fontFamily: 'monospace' }}>{detail.public_id}</span>
              </Typography>
              <AgenticResponseTree r={detail} />
            </>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetail(null)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
