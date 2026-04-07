import type { CssVarsThemeOptions } from '@mui/material/styles';
import { createTheme as createMuiTheme } from '@mui/material/styles';

import { buildColorSchemeFromPreset } from './build-color-scheme';
import { DEFAULT_THEME_PRESET_ID, THEME_PRESETS } from './presets';
import type { AppThemePresetId } from './preset-types';
import { components } from './core/components';
import type { ColorSchemeOptionsExtended, ThemeOptions } from './types';
import { themeConfig } from './theme-config';
import { typography } from './core/typography';

import type {} from './extend-theme-types';

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
