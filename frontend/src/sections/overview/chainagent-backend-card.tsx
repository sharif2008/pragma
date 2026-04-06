import { useNavigate } from 'react-router-dom';
import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';

import { ApiError, getHealth, getApiBaseUrl } from 'src/services';

// ----------------------------------------------------------------------

export function ChainagentBackendCard() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<'idle' | 'ok' | 'error'>('idle');
  const [message, setMessage] = useState<string>('');

  const ping = useCallback(async () => {
    setStatus('idle');
    setMessage('');
    try {
      const h = await getHealth();
      setStatus('ok');
      setMessage(h.status);
    } catch (e) {
      setStatus('error');
      setMessage(e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    ping();
  }, [ping]);

  return (
    <Card sx={{ height: 1 }}>
      <CardContent>
        <Typography variant="subtitle2" sx={{ mb: 2 }}>
          Backend API
        </Typography>
        <Stack spacing={1.5}>
          <Typography variant="caption" sx={{ color: 'text.secondary', wordBreak: 'break-all' }}>
            {getApiBaseUrl()}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
            {status === 'ok' && <Chip size="small" color="success" label={`/health → ${message}`} />}
            {status === 'error' && <Chip size="small" color="error" label="Unreachable" />}
            {status === 'idle' && <Chip size="small" variant="outlined" label="Checking…" />}
          </Box>
          {status === 'error' && (
            <Typography variant="caption" color="error">
              {message}
            </Typography>
          )}
          <Stack direction="row" spacing={1} flexWrap="wrap">
            <Button size="small" variant="outlined" onClick={ping}>
              Ping again
            </Button>
            <Button size="small" variant="contained" onClick={() => navigate('/ml')}>
              Open workbench
            </Button>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );
}
