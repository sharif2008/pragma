import Tooltip from '@mui/material/Tooltip';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import OutlinedInput from '@mui/material/OutlinedInput';
import InputAdornment from '@mui/material/InputAdornment';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

type UserTableToolbarProps = {
  numSelected: number;
  filterName: string;
  onFilterName: (event: React.ChangeEvent<HTMLInputElement>) => void;
};

export function UserTableToolbar({ numSelected, filterName, onFilterName }: UserTableToolbarProps) {
  return (
    <Toolbar
      sx={{
        minHeight: 52,
        height: 52,
        display: 'flex',
        justifyContent: 'space-between',
        gap: 1,
        px: { xs: 1.5, sm: 2 },
        py: 0.75,
        ...(numSelected > 0 && {
          color: 'primary.main',
          bgcolor: 'primary.lighter',
        }),
      }}
    >
      {numSelected > 0 ? (
        <Typography component="div" variant="body2" sx={{ fontWeight: 600 }}>
          {numSelected} selected
        </Typography>
      ) : (
        <OutlinedInput
          size="small"
          fullWidth
          value={filterName}
          onChange={onFilterName}
          placeholder="Search user..."
          startAdornment={
            <InputAdornment position="start">
              <Iconify width={18} icon="eva:search-fill" sx={{ color: 'text.disabled' }} />
            </InputAdornment>
          }
          sx={{ maxWidth: 280, '& .MuiInputBase-input': { py: 0.75, fontSize: '0.8125rem' } }}
        />
      )}

      {numSelected > 0 ? (
        <Tooltip title="Delete">
          <IconButton size="small" sx={{ p: 0.5 }}>
            <Iconify icon="solar:trash-bin-trash-bold" width={20} />
          </IconButton>
        </Tooltip>
      ) : (
        <Tooltip title="Filter list">
          <IconButton size="small" sx={{ p: 0.5 }}>
            <Iconify icon="ic:round-filter-list" width={20} />
          </IconButton>
        </Tooltip>
      )}
    </Toolbar>
  );
}
