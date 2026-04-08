import { useState, useCallback } from 'react';

import Box from '@mui/material/Box';
import Avatar from '@mui/material/Avatar';
import Popover from '@mui/material/Popover';
import TableRow from '@mui/material/TableRow';
import Checkbox from '@mui/material/Checkbox';
import MenuList from '@mui/material/MenuList';
import TableCell from '@mui/material/TableCell';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import MenuItem, { menuItemClasses } from '@mui/material/MenuItem';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export type UserProps = {
  id: string;
  name: string;
  role: string;
  status: string;
  company: string;
  avatarUrl: string;
  isVerified: boolean;
};

type UserTableRowProps = {
  row: UserProps;
  selected: boolean;
  onSelectRow: () => void;
};

export function UserTableRow({ row, selected, onSelectRow }: UserTableRowProps) {
  const [openPopover, setOpenPopover] = useState<HTMLButtonElement | null>(null);

  const handleOpenPopover = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    setOpenPopover(event.currentTarget);
  }, []);

  const handleClosePopover = useCallback(() => {
    setOpenPopover(null);
  }, []);

  return (
    <>
      <TableRow hover tabIndex={-1} role="checkbox" selected={selected}>
        <TableCell padding="checkbox" sx={{ py: 0.25 }}>
          <Checkbox size="small" disableRipple checked={selected} onChange={onSelectRow} />
        </TableCell>

        <TableCell component="th" scope="row" sx={{ py: 0.5 }}>
          <Box
            sx={{
              gap: 1.25,
              display: 'flex',
              alignItems: 'center',
              minWidth: 0,
            }}
          >
            <Avatar alt={row.name} src={row.avatarUrl} sx={{ width: 32, height: 32, flexShrink: 0 }} />
            <Typography variant="body2" noWrap sx={{ fontWeight: 500 }}>
              {row.name}
            </Typography>
          </Box>
        </TableCell>

        <TableCell sx={{ py: 0.5 }}>
          <Typography variant="body2" noWrap>
            {row.company}
          </Typography>
        </TableCell>

        <TableCell sx={{ py: 0.5 }}>
          <Typography variant="body2" color="text.secondary" noWrap>
            {row.role}
          </Typography>
        </TableCell>

        <TableCell align="center" sx={{ py: 0.5 }}>
          {row.isVerified ? (
            <Iconify width={18} icon="solar:check-circle-bold" sx={{ color: 'success.main' }} />
          ) : (
            <Typography variant="caption" color="text.disabled">
              —
            </Typography>
          )}
        </TableCell>

        <TableCell sx={{ py: 0.5 }}>
          <Label
            color={(row.status === 'banned' && 'error') || 'success'}
            sx={{
              height: 22,
              px: 0.75,
              fontSize: '0.6875rem',
              fontWeight: 600,
            }}
          >
            {row.status}
          </Label>
        </TableCell>

        <TableCell align="right" sx={{ py: 0.25, width: 48 }}>
          <IconButton size="small" onClick={handleOpenPopover} sx={{ p: 0.35 }}>
            <Iconify icon="eva:more-vertical-fill" width={18} />
          </IconButton>
        </TableCell>
      </TableRow>

      <Popover
        open={!!openPopover}
        anchorEl={openPopover}
        onClose={handleClosePopover}
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        <MenuList
          disablePadding
          sx={{
            p: 0.5,
            gap: 0.5,
            width: 140,
            display: 'flex',
            flexDirection: 'column',
            [`& .${menuItemClasses.root}`]: {
              px: 1,
              gap: 2,
              borderRadius: 0.75,
              [`&.${menuItemClasses.selected}`]: { bgcolor: 'action.selected' },
            },
          }}
        >
          <MenuItem onClick={handleClosePopover}>
            <Iconify icon="solar:pen-bold" />
            Edit
          </MenuItem>

          <MenuItem onClick={handleClosePopover} sx={{ color: 'error.main' }}>
            <Iconify icon="solar:trash-bin-trash-bold" />
            Delete
          </MenuItem>
        </MenuList>
      </Popover>
    </>
  );
}
