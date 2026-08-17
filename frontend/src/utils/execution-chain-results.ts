import type { TrustAnchorVerifyOut, ExecutionReportDetailOut } from 'src/api/types';

export type ChainActionItem = {
  index?: number;
  attack_type?: string;
  action: string;
  network_tier?: string;
  whitelisted?: boolean;
  whitelist_error?: string;
  apply_tx_hash?: string;
  result?: string;
  failure_reason?: string;
  apply_error?: string;
};

export type AppliedActionView = {
  index?: number;
  action: string;
  network_tier?: string;
  result?: string;
  failure_reason?: string;
  whitelisted?: boolean;
  apply_tx_hash?: string;
};

function asRecord(v: unknown): Record<string, unknown> | null {
  return v && typeof v === 'object' ? (v as Record<string, unknown>) : null;
}

export function chainItemsFromExec(exec: ExecutionReportDetailOut | null | undefined): ChainActionItem[] {
  const chain = asRecord(exec?.actions_chain_json);
  const raw = chain?.items;
  if (!Array.isArray(raw) || !raw.length) return [];
  return raw
    .filter((x): x is Record<string, unknown> => !!x && typeof x === 'object')
    .map((o, i) => ({
      index: typeof o.index === 'number' ? o.index : i,
      attack_type:
        o.attack_type != null
          ? String(o.attack_type)
          : exec?.attack_type ??
            (chain && chain.attack_type != null ? String(chain.attack_type) : undefined),
      action: String(o.action ?? '—'),
      network_tier: o.network_tier != null ? String(o.network_tier) : undefined,
      whitelisted: typeof o.whitelisted === 'boolean' ? o.whitelisted : undefined,
      whitelist_error: o.whitelist_error != null ? String(o.whitelist_error) : undefined,
      apply_tx_hash: o.apply_tx_hash != null ? String(o.apply_tx_hash) : undefined,
      result: o.result != null ? String(o.result) : undefined,
      failure_reason: o.failure_reason != null ? String(o.failure_reason) : undefined,
      apply_error: o.apply_error != null ? String(o.apply_error) : undefined,
    }))
    .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
}

function itemsFromExecBucket(
  bucket: string,
  json: Record<string, unknown> | null | undefined
): AppliedActionView[] {
  if (!json || typeof json !== 'object') return [];
  const items = (json as { items?: unknown }).items;
  if (!Array.isArray(items)) return [];
  return items
    .filter((x): x is Record<string, unknown> => !!x && typeof x === 'object')
    .map((o) => ({
      action: String(o.action ?? '—'),
      network_tier: o.network_tier != null ? String(o.network_tier) : bucket,
      result: o.result != null ? String(o.result) : undefined,
      failure_reason: o.failure_reason != null ? String(o.failure_reason) : undefined,
    }));
}

export function appliedItemsFromExec(exec: ExecutionReportDetailOut | null | undefined): AppliedActionView[] {
  const chain = chainItemsFromExec(exec);
  if (chain.length) {
    return chain.map((it) => ({
      index: it.index,
      action: it.action,
      network_tier: it.network_tier,
      result: it.result,
      failure_reason: it.failure_reason || it.whitelist_error || it.apply_error,
      whitelisted: it.whitelisted,
      apply_tx_hash: it.apply_tx_hash,
    }));
  }

  const per = asRecord(exec?.actions_core_json)?.per_action_items;
  if (Array.isArray(per) && per.length) {
    return per
      .filter((x): x is Record<string, unknown> => !!x && typeof x === 'object')
      .map((o) => ({
        index: typeof o.index === 'number' ? o.index : undefined,
        action: String(o.action ?? '—'),
        network_tier: o.network_tier != null ? String(o.network_tier) : undefined,
        result: o.result != null ? String(o.result) : undefined,
        failure_reason: o.failure_reason != null ? String(o.failure_reason) : undefined,
      }))
      .sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
  }

  return [
    ...itemsFromExecBucket('Access / ISP', asRecord(exec?.actions_ran_json)),
    ...itemsFromExecBucket('Perimeter / IDS', asRecord(exec?.actions_edge_json)),
    ...itemsFromExecBucket('Endpoint / EDR', asRecord(exec?.actions_core_json)),
  ];
}

export function applyFailureDetail(
  exec: ExecutionReportDetailOut | null | undefined,
  verify: TrustAnchorVerifyOut | null | undefined
): string {
  const detail =
    verify?.payload_integrity_detail ||
    verify?.chain_integrity_detail ||
    exec?.payload_detail ||
    exec?.chain_detail ||
    exec?.error_detail ||
    '';
  const integrity = exec?.integrity_overall || verify?.overall_integrity || 'unknown';
  if (integrity === 'anchor_failed') {
    return detail || 'Plan was not anchored on blockchain (no valid trust anchor transaction).';
  }
  if (integrity !== 'valid') {
    return detail || `Blockchain integrity check failed (${integrity}).`;
  }
  return exec?.error_reason || detail || 'Apply failed.';
}

export function attackTypeFromReport(
  exec: ExecutionReportDetailOut | null | undefined,
  reportAttackType?: string | null
): string | null {
  return exec?.attack_type || reportAttackType || exec?.actions_chain_json?.attack_type || null;
}

export function chainSummaryFromExec(exec: ExecutionReportDetailOut | null | undefined) {
  const items = chainItemsFromExec(exec);
  return {
    total: items.length,
    whitelisted: items.filter((x) => x.whitelisted === true).length,
    appliedOnChain: items.filter((x) => x.result === 'success' && x.apply_tx_hash).length,
  };
}
