import type { CssVarsThemeOptions } from '@mui/material/styles';

import { createTheme as createMuiTheme } from '@mui/material/styles';

import { themeConfig } from './theme-config';
import { components } from './core/components';
import { typography } from './core/typography';
import { buildColorSchemeFromPreset } from './build-color-scheme';
import { THEME_PRESETS, DEFAULT_THEME_PRESET_ID } from './presets';

import type {} from './extend-theme-types';
import type { AppThemePresetId } from './preset-types';
import type { ThemeOptions, ColorSchemeOptionsExtended } from './types';

// ----------------------------------------------------------------------

function resolvePreset(presetId: AppThemePresetId | string) {
  const def = THEME_PRESETS[presetId as AppThemePresetId];
  return def ?? THEME_PRESETS[DEFAULT_THEME_PRESET_ID];
}

export function createAppTheme(
  presetId: AppThemePresetId = DEFAULT_THEME_PRESET_ID,
  userOverrides?: ThemeOptions
) {
  const def = resolvePreset(presetId);
  const { palette, shadows, customShadows } = buildColorSchemeFromPreset(def);

  const lightScheme: ColorSchemeOptionsExtended = { palette, shadows, customShadows };

  const theme: CssVarsThemeOptions = createMuiTheme(
    {
      colorSchemes: {
        light: lightScheme,
      },
      components,
      shape: { borderRadius: 8 },
      cssVariables: themeConfig.cssVariables,
    },
    {
      typography: typography as ThemeOptions['typography'],
      ...userOverrides,
    }
  );

  return theme;
}

export { themeConfig };
