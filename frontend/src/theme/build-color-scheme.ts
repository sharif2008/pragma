import type { ColorSystemOptions } from '@mui/material/styles';

import { varAlpha, createPaletteChannel } from 'minimal-shared/utils';

import { createShadows } from './core/shadows';
import type { ThemePresetDefinition } from './preset-types';

import type { CustomShadows } from './core/custom-shadows';

function createShadowColor(colorChannel: string): string {
  return `0 8px 16px 0 ${varAlpha(colorChannel, 0.24)}`;
}

function buildCustomShadows(
  greyChannel: string,
  blackChannel: string,
  paletteChannels: {
    primary: string;
    secondary: string;
    info: string;
    success: string;
    warning: string;
    error: string;
  }
): CustomShadows {
  return {
    z1: `0 1px 2px 0 ${varAlpha(greyChannel, 0.16)}`,
    z4: `0 4px 8px 0 ${varAlpha(greyChannel, 0.16)}`,
    z8: `0 8px 16px 0 ${varAlpha(greyChannel, 0.16)}`,
    z12: `0 12px 24px -4px ${varAlpha(greyChannel, 0.16)}`,
    z16: `0 16px 32px -4px ${varAlpha(greyChannel, 0.16)}`,
    z20: `0 20px 40px -4px ${varAlpha(greyChannel, 0.16)}`,
    z24: `0 24px 48px 0 ${varAlpha(greyChannel, 0.16)}`,
    dialog: `-40px 40px 80px -8px ${varAlpha(blackChannel, 0.24)}`,
    card: `0 0 2px 0 ${varAlpha(greyChannel, 0.2)}, 0 12px 24px -4px ${varAlpha(greyChannel, 0.12)}`,
    dropdown: `0 0 2px 0 ${varAlpha(greyChannel, 0.24)}, -20px 20px 40px -4px ${varAlpha(greyChannel, 0.24)}`,
    primary: createShadowColor(paletteChannels.primary),
    secondary: createShadowColor(paletteChannels.secondary),
    info: createShadowColor(paletteChannels.info),
    success: createShadowColor(paletteChannels.success),
    warning: createShadowColor(paletteChannels.warning),
    error: createShadowColor(paletteChannels.error),
  };
}

/** @internal exported for rare direct access */
export function buildColorSchemeFromPreset(def: ThemePresetDefinition) {
  const p = def.palette;
  const primary = createPaletteChannel(p.primary);
  const secondary = createPaletteChannel(p.secondary);
  const info = createPaletteChannel(p.info);
  const success = createPaletteChannel(p.success);
  const warning = createPaletteChannel(p.warning);
  const error = createPaletteChannel(p.error);
  const common = createPaletteChannel(p.common);
  const grey = createPaletteChannel(p.grey);

  const baseAction = {
    hover: varAlpha(grey['500Channel'], 0.08),
    selected: varAlpha(grey['500Channel'], 0.16),
    focus: varAlpha(grey['500Channel'], 0.24),
    disabled: varAlpha(grey['500Channel'], 0.8),
    disabledBackground: varAlpha(grey['500Channel'], 0.24),
    hoverOpacity: 0.08,
    disabledOpacity: 0.48,
  };

  const fullPalette: ColorSystemOptions['palette'] = {
    primary,
    secondary,
    info,
    success,
    warning,
    error,
    common,
    grey,
    divider: varAlpha(grey['500Channel'], 0.2),
    text: createPaletteChannel(def.surfaces.text),
    background: createPaletteChannel(def.surfaces.background),
    action: { ...baseAction, active: def.surfaces.actionActive },
  };

  const shadows = createShadows(grey['500Channel']);
  const customShadows = buildCustomShadows(grey['500Channel'], common.blackChannel, {
    primary: primary.mainChannel,
    secondary: secondary.mainChannel,
    info: info.mainChannel,
    success: success.mainChannel,
    warning: warning.mainChannel,
    error: error.mainChannel,
  });

  return { palette: fullPalette, shadows, customShadows };
}
