import type { ThemeProviderProps as MuiThemeProviderProps } from '@mui/material/styles';

import { useMemo } from 'react';

import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider as ThemeVarsProvider } from '@mui/material/styles';

import { createAppTheme } from './create-theme';
import { ThemePresetProvider, useThemePreset } from './theme-preset-context';

import type {} from './extend-theme-types';
import type { ThemeOptions } from './types';

// ----------------------------------------------------------------------

export type ThemeProviderProps = Partial<MuiThemeProviderProps> & {
  themeOverrides?: ThemeOptions;
};

function MuiThemeShell({ themeOverrides, children, ...other }: ThemeProviderProps) {
  const { presetId } = useThemePreset();
  const theme = useMemo(
    () => createAppTheme(presetId, themeOverrides),
    [presetId, themeOverrides]
  );

  return (
    <ThemeVarsProvider disableTransitionOnChange theme={theme} {...other}>
      <CssBaseline />
      {children}
    </ThemeVarsProvider>
  );
}

export function ThemeProvider({ children, ...props }: ThemeProviderProps) {
  return (
    <ThemePresetProvider>
      <MuiThemeShell {...props}>{children}</MuiThemeShell>
    </ThemePresetProvider>
  );
}
