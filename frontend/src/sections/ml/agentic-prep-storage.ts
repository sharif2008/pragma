import type { KBQueryHit } from 'src/api/types';

/** sessionStorage payload from Knowledge base → Agentic actions */
export const AGENTIC_PREP_STORAGE_KEY = 'chainagent_vfl_agentic_prep';

export type AgenticPrepPayload = {
  predictionJobPublicId: string | null;
  templateId: string;
  templateLabel: string;
  llmPrompt: string;
  citations: KBQueryHit[];
  retrievalMetaLine: string;
  updatedAt: string;
};

export function readAgenticPrep(): AgenticPrepPayload | null {
  try {
    const raw = sessionStorage.getItem(AGENTIC_PREP_STORAGE_KEY);
    if (!raw) return null;
    const p = JSON.parse(raw) as AgenticPrepPayload;
    if (!p || !Array.isArray(p.citations)) return null;
    return p;
  } catch {
    return null;
  }
}

export function writeAgenticPrep(payload: AgenticPrepPayload): void {
  sessionStorage.setItem(AGENTIC_PREP_STORAGE_KEY, JSON.stringify(payload));
}

export function clearAgenticPrep(): void {
  sessionStorage.removeItem(AGENTIC_PREP_STORAGE_KEY);
}
