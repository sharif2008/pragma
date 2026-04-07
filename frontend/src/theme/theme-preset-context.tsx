import { createContext, useCallback, useContext, useMemo, useState } from 'react';

import type { AppThemePresetId } from './preset-types';
import { DEFAULT_THEME_PRESET_ID, THEME_PRESET_LIST, THEME_PRESETS } from './presets';

// ----------------------------------------------------------------------

const STORAGE_KEY = 'chainagent-theme-preset-v2';
const LEGACY_STORAGE_KEY = 'chainagent-theme-preset-v1';

function readStoredPreset(): AppThemePresetId {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw && raw in THEME_PRESETS) {
      return raw as AppThemePresetId;
    }
    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy && legacy in THEME_PRESETS) {
      localStorage.setItem(STORAGE_KEY, legacy);
      return legacy as AppThemePresetId;
    }
  } catch {
    /* ignore */
  }
  return DEFAULT_THEME_PRESET_ID;
}

export type ThemePresetContextValue = {
  presetId: AppThemePresetId;
  setPresetId: (id: AppThemePresetId) => void;
  presets: typeof THEME_PRESET_LIST;
};

const ThemePresetContext = createContext<ThemePresetContextValue | null>(null);

type ThemePresetProviderProps = {
  children: React.ReactNode;
};

export function ThemePresetProvider({ children }: ThemePresetProviderProps) {
  const [presetId, setPresetIdState] = useState<AppThemePresetId>(readStoredPreset);

  const setPresetId = useCallback((id: AppThemePresetId) => {
    setPresetIdState(id);
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      /* ignore */
    }
  }, []);

  const value = useMemo<ThemePresetContextValue>(
    () => ({
      presetId,
      setPresetId,
      presets: THEME_PRESET_LIST,
    }),
    [presetId, setPresetId]
  );

  return <ThemePresetContext.Provider value={value}>{children}</ThemePresetContext.Provider>;
}

export function useThemePreset() {
  const ctx = useContext(ThemePresetContext);
  if (!ctx) {
    throw new Error('useThemePreset must be used within ThemePresetProvider');
  }
  return ctx;
}
