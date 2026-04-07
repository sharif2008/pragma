import type { ThemeCssVariables } from './types';

// ----------------------------------------------------------------------

type ThemeConfig = {
  classesPrefix: string;
  cssVariables: ThemeCssVariables;
  fontFamily: Record<'primary' | 'secondary', string>;
};

export const themeConfig: ThemeConfig = {
  classesPrefix: 'minimal',
  fontFamily: {
    primary: 'DM Sans Variable',
    secondary: 'Barlow',
  },
  cssVariables: {
    cssVarPrefix: '',
    colorSchemeSelector: 'data-color-scheme',
  },
};
