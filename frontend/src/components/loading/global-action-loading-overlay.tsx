import Stack from '@mui/material/Stack';
import Dialog from '@mui/material/Dialog';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import DialogContent from '@mui/material/DialogContent';
import CircularProgress from '@mui/material/CircularProgress';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

type Props = {
  open: boolean;
  /** Hides the dialog only — does not cancel the in-flight API request. */
  onClose?: () => void;
  message?: string;
  submessage?: string;
};

/** Closable loading dialog for long-running agentic / API actions. */
export function GlobalActionLoadingOverlay({
  open,
  onClose,
  message = 'Processing…',
  submessage,
}: Props) {
  const handleClose = () => {
    onClose?.();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      aria-labelledby="global-action-loading-title"
      PaperProps={{
        sx: {
          minWidth: 280,
          maxWidth: 420,
          borderRadius: 2,
        },
      }}
    >
      <IconButton
        aria-label="Close loading dialog"
        onClick={handleClose}
        sx={{
          position: 'absolute',
          right: 8,
          top: 8,
          color: 'text.secondary',
        }}
      >
        <Iconify icon="mingcute:close-line" width={20} />
      </IconButton>
      <DialogContent sx={{ pt: 3, pb: 2.5, px: 3 }}>
        <Stack spacing={2} alignItems="center" textAlign="center">
          <CircularProgress size={56} thickness={4} color="primary" />
          <Stack spacing={0.5} alignItems="center">
            <Typography id="global-action-loading-title" variant="subtitle1" sx={{ fontWeight: 700 }}>
              {message}
            </Typography>
            {submessage ? (
              <Typography variant="body2" color="text.secondary">
                {submessage}
              </Typography>
            ) : null}
            <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }}>
              You can close this — the request keeps running in the background.
            </Typography>
          </Stack>
          <Button size="small" variant="outlined" onClick={handleClose} sx={{ mt: 0.5 }}>
            Close
          </Button>
        </Stack>
      </DialogContent>
    </Dialog>
  );
}
