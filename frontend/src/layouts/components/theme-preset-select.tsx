import type { AppThemePresetId } from 'src/theme/preset-types';

import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';

import { useThemePreset } from 'src/theme/theme-preset-context';

// ----------------------------------------------------------------------

export function ThemePresetSelect() {
  const { presetId, setPresetId, presets } = useThemePreset();

  return (
    <FormControl size="small" sx={{ minWidth: 132 }}>
      <InputLabel id="chainagent-theme-preset-label">Appearance</InputLabel>
      <Select<AppThemePresetId>
        labelId="chainagent-theme-preset-label"
        label="Appearance"
        value={presetId}
        onChange={(e) => setPresetId(e.target.value as AppThemePresetId)}
      >
        {presets.map((p) => (
          <MenuItem key={p.id} value={p.id}>
            {p.label}
          </MenuItem>
        ))}
      </Select>
    </FormControl>
  );
}
