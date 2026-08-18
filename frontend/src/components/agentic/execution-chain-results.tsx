import type { TrustAnchorVerifyOut, ExecutionReportDetailOut } from 'src/api/types';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

import {
  applyFailureDetail,
  appliedItemsFromExec,
  attackTypeFromReport,
} from 'src/utils/execution-chain-results';

type Props = {
  exec: ExecutionReportDetailOut | null | undefined;
  verify?: TrustAnchorVerifyOut | null;
  execApplied?: boolean;
  compact?: boolean;
};

function resultLabel(
  result: string | undefined,
  execApplied: boolean
): { label: string; color: 'success' | 'error' | 'default' | 'warning' } {
  const ok = execApplied && result === 'success';
  const failed = !execApplied || result === 'failed';
  if (ok) return { label: 'Applied', color: 'success' };
  if (failed) return { label: 'Not applied', color: 'error' };
  if (result === 'skipped') return { label: 'Skipped', color: 'warning' };
  return { label: result || '—', color: 'default' };
}

export function ExecutionChainResultsList({ exec, execApplied, compact = false }: Props) {
  const items = appliedItemsFromExec(exec);
  const applied = execApplied ?? exec?.status === 'applied';
  const attackType = attackTypeFromReport(exec);

  if (!items.length) {
    return (
      <Typography variant="body2" color="text.secondary">
        No per-action results recorded yet.
      </Typography>
    );
  }

  return (
    <Stack spacing={compact ? 0.75 : 1.25}>
      {attackType ? (
        <Chip size="small" variant="outlined" label={`Attack: ${attackType}`} sx={{ alignSelf: 'flex-start' }} />
      ) : null}
      {items.map((it, i) => {
        const { label, color } = resultLabel(it.result, applied);
        const failed = !applied || it.result === 'failed' || it.result === 'skipped';
        return (
          <Box
            key={`${it.index ?? i}-${it.action.slice(0, 20)}`}
            sx={{
              border: 1,
              borderColor: failed ? 'error.light' : 'divider',
              borderRadius: 1,
              px: compact ? 1 : 1.25,
              py: compact ? 0.75 : 1,
              bgcolor: failed ? (theme) => theme.vars.palette.error.lighter : 'background.paper',
            }}
          >
            <Stack direction="row" spacing={0.75} alignItems="center" flexWrap="wrap" useFlexGap>
              <Chip size="small" label={label} color={color} sx={{ height: 20 }} />
              {it.network_tier ? (
                <Chip size="small" variant="outlined" label={it.network_tier} sx={{ height: 20, maxWidth: 160 }} />
              ) : null}
              {typeof it.whitelisted === 'boolean' ? (
                <Chip
                  size="small"
                  variant="outlined"
                  color={it.whitelisted ? 'success' : 'error'}
                  label={it.whitelisted ? 'Whitelisted' : 'Not whitelisted'}
                  sx={{ height: 20 }}
                />
              ) : null}
              <Typography variant="body2" sx={{ flex: 1, fontWeight: 600 }}>
                {it.action}
              </Typography>
            </Stack>
            {it.apply_tx_hash ? (
              <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 0.5, fontFamily: 'monospace' }}>
                tx {it.apply_tx_hash.slice(0, 14)}…
              </Typography>
            ) : null}
            {failed && it.failure_reason ? (
              <Typography variant="caption" color="error.main" display="block" sx={{ mt: 0.5, lineHeight: 1.4 }}>
                {it.failure_reason}
              </Typography>
            ) : null}
          </Box>
        );
      })}
    </Stack>
  );
}

export function ExecutionChainSummary({ exec, verify }: Props) {
  const applied = exec?.status === 'applied';
  const attackType = attackTypeFromReport(exec);
  const failure = !applied && exec ? applyFailureDetail(exec, verify) : null;

  return (
    <Stack spacing={1}>
      <Stack direction="row" gap={0.75} flexWrap="wrap" useFlexGap>
        {attackType ? <Chip size="small" variant="outlined" label={`Attack ${attackType}`} /> : null}
        {exec?.status ? (
          <Chip
            size="small"
            color={applied ? 'success' : 'error'}
            label={exec.status === 'applied' ? 'Applied' : 'Not applied'}
          />
        ) : null}
        {exec?.integrity_overall ? (
          <Chip size="small" variant="outlined" label={`Integrity ${exec.integrity_overall}`} />
        ) : null}
      </Stack>
      {!applied && failure ? (
        <Typography variant="body2" color="error.main" sx={{ whiteSpace: 'pre-wrap' }}>
          {failure}
        </Typography>
      ) : null}
    </Stack>
  );
}
