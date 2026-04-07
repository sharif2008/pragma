import type { AgenticReportOut } from 'src/api/types';

import { useCallback, useEffect, useState } from 'react';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Collapse from '@mui/material/Collapse';
import Table from '@mui/material/Table';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import TableContainer from '@mui/material/TableContainer';
import ListItemButton from '@mui/material/ListItemButton';

import { usePathname } from 'src/routes/hooks';
import { RouterLink } from 'src/routes/components';
import { ApiError, listAgentReports } from 'src/services';
import { AgentReportDetailDialog } from 'src/components/run-monitoring/detail-dialogs';
import { Iconify } from 'src/components/iconify';

const MAX = 30;
const TABLE_MAX_HEIGHT = 320;

function shortId(id: string) {
  if (!id) return '—';
  if (id.length <= 12) return id;
  return `${id.slice(0, 6)}…${id.slice(-4)}`;
}

export function AgenticNavReports() {
  const pathname = usePathname();
  const [open, setOpen] = useState(() => pathname.startsWith('/agentic'));
  const [rows, setRows] = useState<AgenticReportOut[]>([]);
  const [dialogId, setDialogId] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await listAgentReports(MAX, 0);
      const sorted = [...r].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      setRows(sorted);
    } catch (e) {
      if (!(e instanceof ApiError)) console.warn(e);
      setRows([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (pathname.startsWith('/agentic')) setOpen(true);
  }, [pathname]);

  return (
    <Box sx={{ pl: 0.5, pr: 0.25, pb: 1, mt: -0.25, width: 1, maxWidth: 1 }}>
      <ListItemButton
        dense
        onClick={() => setOpen((v) => !v)}
        sx={{ borderRadius: 1, py: 0.35, minHeight: 32, px: 1 }}
      >
        <Iconify
          icon="eva:arrow-ios-downward-fill"
          width={16}
          sx={{ mr: 0.5, flexShrink: 0, transform: open ? 'none' : 'rotate(-90deg)' }}
        />
        <Typography variant="caption" fontWeight={700} color="text.secondary" noWrap>
          Agentic actions
        </Typography>
      </ListItemButton>

      <Collapse in={open}>
        <Box sx={{ pl: 0.5, pr: 0, pt: 0.75 }}>
          <TableContainer
            sx={{
              maxHeight: TABLE_MAX_HEIGHT,
              overflow: 'auto',
              width: 1,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Table size="small" stickyHeader sx={{ minWidth: 260 }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 700, fontSize: 10, py: 0.5, bgcolor: 'background.paper' }}>
                    Report
                  </TableCell>
                  <TableCell sx={{ fontWeight: 700, fontSize: 10, py: 0.5, bgcolor: 'background.paper' }}>
                    Action
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{ fontWeight: 700, fontSize: 10, py: 0.5, width: 40, bgcolor: 'background.paper' }}
                  >
                    View
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.map((r) => (
                  <TableRow key={r.public_id} hover sx={{ '& td': { py: 0.35, fontSize: 10 } }}>
                    <TableCell sx={{ fontFamily: 'monospace', maxWidth: 88 }} title={r.public_id}>
                      {shortId(r.public_id)}
                    </TableCell>
                    <TableCell sx={{ maxWidth: 100 }} title={r.recommended_action ?? ''}>
                      <Typography variant="caption" noWrap display="block" sx={{ fontSize: 10 }}>
                        {r.recommended_action || '—'}
                      </Typography>
                    </TableCell>
                    <TableCell align="right" sx={{ px: 0.25 }}>
                      <Tooltip title="Open report details">
                        <IconButton
                          size="small"
                          color="primary"
                          aria-label="View agent report"
                          onClick={() => setDialogId(r.public_id)}
                          sx={{ p: 0.35 }}
                        >
                          <Iconify icon="solar:eye-bold" width={18} />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
                {!rows.length && (
                  <TableRow>
                    <TableCell colSpan={3}>
                      <Typography variant="caption" color="text.disabled" sx={{ py: 0.5, display: 'block' }}>
                        No reports yet
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>

          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ display: 'block', mt: 0.75, px: 0.25, lineHeight: 1.35 }}
            noWrap
            title={rows[0]?.summary}
          >
            {rows[0] ? (
              <>
                Latest:{' '}
                <Typography component="span" variant="caption" sx={{ fontSize: 10 }}>
                  {(rows[0].summary || '—').slice(0, 72)}
                  {(rows[0].summary?.length ?? 0) > 72 ? '…' : ''}
                </Typography>
              </>
            ) : null}
          </Typography>

          <Button
            component={RouterLink}
            href="/agentic"
            size="small"
            variant="text"
            sx={{ mt: 0.25, px: 0.5, fontSize: 10, minHeight: 28 }}
          >
            Full timeline →
          </Button>
        </Box>
      </Collapse>

      <AgentReportDetailDialog open={Boolean(dialogId)} publicId={dialogId} onClose={() => setDialogId(null)} />
    </Box>
  );
}
