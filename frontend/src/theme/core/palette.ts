import type { PaletteColor } from '@mui/material/styles';

// ----------------------------------------------------------------------
/**
 * TypeScript (type definition and extension)
 * @to {@link file://./../extend-theme-types.d.ts}
 */

export type PaletteColorKey = 'primary' | 'secondary' | 'info' | 'success' | 'warning' | 'error';

export type PaletteColorNoChannels = Omit<PaletteColor, 'lighterChannel' | 'darkerChannel'>;

export type PaletteColorWithChannels = PaletteColor & {
  lighterChannel?: string;
  darkerChannel?: string;
  mainChannel?: string;
};

export type CommonColorsExtend = {
  whiteChannel: string;
  blackChannel: string;
};

export type TypeTextExtend = {
  disabledChannel: string;
  primaryChannel?: string;
  secondaryChannel?: string;
};

export type TypeBackgroundExtend = {
  neutral: string;
  neutralChannel: string;
  defaultChannel?: string;
  paperChannel?: string;
};

export type PaletteColorExtend = {
  lighter: string;
  darker: string;
  lighterChannel: string;
  darkerChannel: string;
};

export type GreyExtend = {
  '50Channel': string;
  '100Channel': string;
  '200Channel': string;
  '300Channel': string;
  '400Channel': string;
  '500Channel': string;
  '600Channel': string;
  '700Channel': string;
  '800Channel': string;
  '900Channel': string;
};
