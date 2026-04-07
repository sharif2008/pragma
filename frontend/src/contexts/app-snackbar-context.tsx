import type { AlertColor } from '@mui/material/Alert';
import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import Alert from '@mui/material/Alert';
import Snackbar from '@mui/material/Snackbar';

// ----------------------------------------------------------------------

type ShowOpts = {
  /** ms before auto-hide (default 4000) */
  autoHideMs?: number;
};

type AppSnackbarContextValue = {
  showSuccess: (message: string, opts?: ShowOpts) => void;
  showInfo: (message: string, opts?: ShowOpts) => void;
  showError: (message: string, opts?: ShowOpts) => void;
};

const AppSnackbarContext = createContext<AppSnackbarContextValue | null>(null);

export function AppSnackbarProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState<AlertColor>('success');
  const [autoHide, setAutoHide] = useState(4000);

  const close = useCallback(() => setOpen(false), []);

  const push = useCallback((msg: string, sev: AlertColor, opts?: ShowOpts) => {
    setMessage(msg);
    setSeverity(sev);
    setAutoHide(opts?.autoHideMs ?? 4000);
    setOpen(true);
  }, []);

  const value = useMemo<AppSnackbarContextValue>(
    () => ({
      showSuccess: (msg, opts) => push(msg, 'success', opts),
      showInfo: (msg, opts) => push(msg, 'info', opts),
      showError: (msg, opts) => push(msg, 'error', opts),
    }),
    [push]
  );

  return (
    <AppSnackbarContext.Provider value={value}>
      {children}
      <Snackbar
        open={open}
        autoHideDuration={autoHide}
        onClose={(_, reason) => {
          if (reason === 'clickaway') return;
          close();
        }}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={close}
          severity={severity}
          variant="outlined"
          sx={{
            width: '100%',
            minWidth: 260,
            maxWidth: 420,
            bgcolor: (theme) => theme.vars.palette.background.paper,
            backdropFilter: 'blur(10px)',
            borderColor: (theme) => theme.vars.palette.divider,
            boxShadow: (theme) => theme.shadows[8],
          }}
        >
          {message}
        </Alert>
      </Snackbar>
    </AppSnackbarContext.Provider>
  );
}

export function useAppSnackbar() {
  const ctx = useContext(AppSnackbarContext);
  if (!ctx) {
    throw new Error('useAppSnackbar must be used within AppSnackbarProvider');
  }
  return ctx;
}
