import type { PlanAction } from 'src/utils/agentic-plan-actions';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';

type Props = {
  actions: PlanAction[];
  compact?: boolean;
};

export function DetectionActionsList({ actions, compact = false }: Props) {
  if (!actions.length) {
    return (
      <Typography variant="caption" color="text.secondary">
        —
      </Typography>
    );
  }

  return (
    <Stack spacing={compact ? 0.25 : 0.35}>
      {actions.map((a, i) => (
        <Stack
          key={`${i}-${a.action.slice(0, 24)}`}
          direction="row"
          spacing={0.75}
          alignItems="flex-start"
        >
          {a.network_tier ? (
            <Chip
              size="small"
              variant="outlined"
              label={a.network_tier}
              sx={{ height: 20, flexShrink: 0, maxWidth: 140 }}
            />
          ) : null}
          <Typography
            variant="caption"
            display="block"
            sx={{ lineHeight: 1.35, wordBreak: 'break-word', flex: 1 }}
            title={a.action}
          >
            {a.action}
          </Typography>
        </Stack>
      ))}
    </Stack>
  );
}

export function ExecStatusChip({ status }: { status: 'applied' | 'failed' | 'none' }) {
  if (status === 'applied') {
    return <Chip size="small" color="success" label="Applied" sx={{ height: 22 }} />;
  }
  if (status === 'failed') {
    return <Chip size="small" color="error" label="Not applied" sx={{ height: 22 }} />;
  }
  return <Chip size="small" variant="outlined" label="Pending" sx={{ height: 22 }} />;
}

export function PlanIdCell({ publicId }: { publicId: string }) {
  return (
    <Box component="span" sx={{ fontFamily: 'monospace', fontSize: 11 }} title={publicId}>
      {publicId.slice(0, 8)}…
    </Box>
  );
}
