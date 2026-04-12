import type { CommonColors } from '@mui/material/styles';

import type { PaletteColorNoChannels } from './core/palette';

// ----------------------------------------------------------------------

export type AppThemePresetId =
  | 'shamrock'
  | 'sage'
  | 'forest'
  | 'earth'
  | 'midnight'
  | 'light'
  | 'graphite'
  | 'aurora';

export type ThemePresetSurfaces = {
  text: { primary: string; secondary: string; disabled: string };
  background: { paper: string; default: string; neutral: string };
  actionActive: string;
};

export type ThemePresetDefinition = {
  id: AppThemePresetId;
  label: string;
  description?: string;
  palette: {
    primary: PaletteColorNoChannels;
    secondary: PaletteColorNoChannels;
    info: PaletteColorNoChannels;
    success: PaletteColorNoChannels;
    warning: PaletteColorNoChannels;
    error: PaletteColorNoChannels;
    grey: Record<'50' | '100' | '200' | '300' | '400' | '500' | '600' | '700' | '800' | '900', string>;
    common: Pick<CommonColors, 'black' | 'white'>;
  };
  surfaces: ThemePresetSurfaces;
};
