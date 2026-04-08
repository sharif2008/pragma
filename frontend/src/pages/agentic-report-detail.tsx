import { useParams, useNavigate } from 'react-router-dom';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

import { CONFIG } from 'src/config-global';
import { DashboardContent } from 'src/layouts/dashboard';

import { Iconify } from 'src/components/iconify';
import {
  AgentReportDetailContent,
  useAgentReportDetailLoad,
} from 'src/components/run-monitoring/detail-dialogs';

export default function AgenticReportDetailPage() {
  const { publicId } = useParams<{ publicId: string }>();
  const navigate = useNavigate();
  const id = publicId?.trim() ?? null;
  const { report, predJob, loading, error, reload } = useAgentReportDetailLoad(id);

  return (
    <>
      <title>{`Agentic report - ${CONFIG.appName}`}</title>

      <DashboardContent maxWidth="xl">
        <Stack spacing={2}>
          <Box>
            <Button
              size="small"
              variant="outlined"
              startIcon={
                <Iconify icon="eva:arrow-ios-forward-fill" width={18} sx={{ transform: 'scaleX(-1)' }} />
              }
              onClick={() => navigate('/agentic')}
              sx={{ mb: 1 }}
            >
              Back to agentic actions
            </Button>
            <Typography variant="h5" sx={{ fontWeight: 700 }}>
              Agentic report detail
            </Typography>
            {id && (
              <Typography
                variant="caption"
                sx={{ mt: 0.5, fontFamily: 'monospace', color: 'text.secondary', display: 'block', wordBreak: 'break-all' }}
              >
                {id}
              </Typography>
            )}
          </Box>

          <Box sx={{ overflowX: 'auto', width: 1, maxWidth: 1 }}>
            <AgentReportDetailContent
              loading={loading}
              error={error}
              report={report}
              predJob={predJob}
              publicId={id}
            />
          </Box>

          <Button variant="outlined" size="small" onClick={() => void reload()} disabled={loading || !id}>
            Reload
          </Button>
        </Stack>
      </DashboardContent>
    </>
  );
}
