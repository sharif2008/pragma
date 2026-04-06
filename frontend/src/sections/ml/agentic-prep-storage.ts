import type { KBQueryHit } from 'src/api/types';

/** sessionStorage payload from RAG & LLM prep → Agentic actions */
export const AGENTIC_PREP_STORAGE_KEY = 'chainagent_vfl_agentic_prep';

/** Same-window: Agentic tab listens to sync dropdown after handoff writes. */
export const AGENTIC_PREP_UPDATED_EVENT = 'chainagent-vfl-agentic-prep-updated';

export type AgenticPrepPayload = {
  predictionJobPublicId: string | null;
  /** Selected prediction results row (SHAP-aware prep). */
  rowIndex: number | null;
  rowPredictedLabel: string | null;
  rowFlagged: boolean | null;
  templateId: string;
  templateLabel: string;
  llmPrompt: string;
  /** First summary-style retrieval string when using row_summary_rag template (or similar). */
  summaryRagQueryText: string | null;
  citations: KBQueryHit[];
  retrievalMetaLine: string;
  retrievalPipeline: 'fusion_mmr' | 'fusion_only';
  /** Human-readable SHAP lines per agent for the LLM prompt. */
  shapAgentLines: string;
  llmRagAnswer: string | null;
  updatedAt: string;
  /** Queries actually sent to /kb/query-multi (template or rephrased lines). */
  retrievalQueriesUsed?: string[];
  /** Max documents kept for LLM + agentic handoff (e.g. 5). */
  finalDocCount?: number;
  /** Edited LLM RAG prompt if different from template default. */
  llmPromptEdited?: string | null;
  /** Which retrieval query sources were merged for /kb/query-multi (RAG prep checkboxes). */
  retrievalQuerySources?: {
    template: boolean;
    rephrase: boolean;
    llmShap: boolean;
  };
  /** Persisted row in ``agentic_jobs`` after RAG prep registers via POST /agent/jobs. */
  agenticJobPublicId?: string | null;
};

function coercePrep(raw: unknown): AgenticPrepPayload | null {
  if (!raw || typeof raw !== 'object') return null;
  const p = raw as Partial<AgenticPrepPayload>;
  if (!Array.isArray(p.citations)) return null;
  return {
    predictionJobPublicId: p.predictionJobPublicId ?? null,
    rowIndex: p.rowIndex ?? null,
    rowPredictedLabel: p.rowPredictedLabel ?? null,
    rowFlagged: p.rowFlagged ?? null,
    templateId: p.templateId ?? '',
    templateLabel: p.templateLabel ?? '',
    llmPrompt: p.llmPrompt ?? '',
    summaryRagQueryText: p.summaryRagQueryText ?? null,
    citations: p.citations,
    retrievalMetaLine: p.retrievalMetaLine ?? '',
    retrievalPipeline: p.retrievalPipeline === 'fusion_only' ? 'fusion_only' : 'fusion_mmr',
    shapAgentLines: p.shapAgentLines ?? '',
    llmRagAnswer: p.llmRagAnswer ?? null,
    updatedAt: p.updatedAt ?? new Date().toISOString(),
    retrievalQueriesUsed: Array.isArray(p.retrievalQueriesUsed) ? p.retrievalQueriesUsed : undefined,
    finalDocCount: typeof p.finalDocCount === 'number' ? p.finalDocCount : undefined,
    llmPromptEdited: p.llmPromptEdited ?? null,
    retrievalQuerySources:
      p.retrievalQuerySources &&
      typeof p.retrievalQuerySources === 'object' &&
      typeof (p.retrievalQuerySources as { template?: unknown }).template === 'boolean'
        ? {
            template: Boolean((p.retrievalQuerySources as { template: boolean }).template),
            rephrase: Boolean((p.retrievalQuerySources as { rephrase: boolean }).rephrase),
            llmShap: Boolean((p.retrievalQuerySources as { llmShap: boolean }).llmShap),
          }
        : undefined,
    agenticJobPublicId:
      typeof p.agenticJobPublicId === 'string' && p.agenticJobPublicId.trim()
        ? p.agenticJobPublicId.trim()
        : null,
  };
}

export function readAgenticPrep(): AgenticPrepPayload | null {
  try {
    const raw = sessionStorage.getItem(AGENTIC_PREP_STORAGE_KEY);
    if (!raw) return null;
    return coercePrep(JSON.parse(raw));
  } catch {
    return null;
  }
}

export function writeAgenticPrep(payload: AgenticPrepPayload): void {
  sessionStorage.setItem(AGENTIC_PREP_STORAGE_KEY, JSON.stringify(payload));
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(AGENTIC_PREP_UPDATED_EVENT));
  }
}

/** Link or refresh the prediction job id in handoff so Agentic actions → Select agentic job follows this job (optional before full RAG save). */
export function upsertAgenticJobHandoff(predictionJobPublicId: string, rowIndex: number | null): void {
  const prev = readAgenticPrep();
  const now = new Date().toISOString();
  if (!prev) {
    writeAgenticPrep({
      predictionJobPublicId,
      rowIndex,
      rowPredictedLabel: null,
      rowFlagged: null,
      templateId: '',
      templateLabel: '',
      llmPrompt: '',
      summaryRagQueryText: null,
      citations: [],
      retrievalMetaLine:
        'Agentic job linked from RAG & LLM prep — run retrieval and use Save for agentic job for KB citations.',
      retrievalPipeline: 'fusion_mmr',
      shapAgentLines: '',
      llmRagAnswer: null,
      updatedAt: now,
      agenticJobPublicId: null,
    });
    return;
  }
  const predChanged =
    prev.predictionJobPublicId != null && prev.predictionJobPublicId !== predictionJobPublicId;
  writeAgenticPrep({
    ...prev,
    predictionJobPublicId,
    rowIndex: rowIndex ?? prev.rowIndex,
    agenticJobPublicId: predChanged ? null : prev.agenticJobPublicId,
    updatedAt: now,
    retrievalMetaLine:
      prev.retrievalMetaLine?.trim() ||
      'Agentic job updated from RAG & LLM prep — run retrieval and Save for agentic job when ready.',
  });
}

export function clearAgenticPrep(): void {
  sessionStorage.removeItem(AGENTIC_PREP_STORAGE_KEY);
}
