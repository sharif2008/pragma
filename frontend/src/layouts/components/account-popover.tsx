import type { IconButtonProps } from '@mui/material/IconButton';

import { useState, useCallback } from 'react';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Popover from '@mui/material/Popover';
import Divider from '@mui/material/Divider';
import MenuList from '@mui/material/MenuList';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import MenuItem, { menuItemClasses } from '@mui/material/MenuItem';

import { useRouter, usePathname } from 'src/routes/hooks';

import { CONFIG } from 'src/config-global';

import { Logo } from 'src/components/logo';

// ----------------------------------------------------------------------

export type AccountPopoverProps = IconButtonProps & {
  data?: {
    label: string;
    href: string;
    icon?: React.ReactNode;
    info?: React.ReactNode;
  }[];
};

export function AccountPopover({ data = [], sx, ...other }: AccountPopoverProps) {
  const router = useRouter();

  const pathname = usePathname();

  const [openPopover, setOpenPopover] = useState<HTMLButtonElement | null>(null);

  const handleOpenPopover = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    setOpenPopover(event.currentTarget);
  }, []);

  const handleClosePopover = useCallback(() => {
    setOpenPopover(null);
  }, []);

  const handleClickItem = useCallback(
    (path: string) => {
      handleClosePopover();
      router.push(path);
    },
    [handleClosePopover, router]
  );

  return (
    <>
      <IconButton
        onClick={handleOpenPopover}
        aria-label={`${CONFIG.appName} menu`}
        sx={{
          p: 0.5,
          width: 44,
          height: 44,
          border: 1,
          borderColor: 'divider',
          borderRadius: 1,
          bgcolor: 'background.paper',
          ...sx,
        }}
        {...other}
      >
        <Box sx={{ width: 34, height: 34, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Logo disabled href="/" sx={{ width: 34, height: 34 }} />
        </Box>
      </IconButton>

      <Popover
        open={!!openPopover}
        anchorEl={openPopover}
        onClose={handleClosePopover}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: { width: 200 },
          },
        }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ p: 2, pb: 1.5 }}>
          <Logo disabled href="/" sx={{ width: 40, height: 40, flexShrink: 0 }} />
          <Box sx={{ minWidth: 0 }}>
            <Typography variant="subtitle1" noWrap fontWeight={700}>
              {CONFIG.appName}
            </Typography>
            <Chip
              size="small"
              label={CONFIG.deploymentLabel}
              color={import.meta.env.PROD ? 'success' : 'default'}
              variant={import.meta.env.PROD ? 'filled' : 'outlined'}
              sx={{ mt: 0.5, height: 22, '& .MuiChip-label': { px: 0.75, fontSize: 11, fontWeight: 700 } }}
            />
          </Box>
        </Stack>

        <Divider sx={{ borderStyle: 'dashed' }} />

        <MenuList
          disablePadding
          sx={{
            p: 1,
            gap: 0.5,
            display: 'flex',
            flexDirection: 'column',
            [`& .${menuItemClasses.root}`]: {
              px: 1,
              gap: 2,
              borderRadius: 0.75,
              color: 'text.secondary',
              '&:hover': { color: 'text.primary' },
              [`&.${menuItemClasses.selected}`]: {
                color: 'text.primary',
                bgcolor: 'action.selected',
                fontWeight: 'fontWeightSemiBold',
              },
            },
          }}
        >
          {data.map((option) => (
            <MenuItem
              key={option.label}
              selected={option.href === pathname}
              onClick={() => handleClickItem(option.href)}
            >
              {option.icon}
              {option.label}
            </MenuItem>
          ))}
        </MenuList>

        <Divider sx={{ borderStyle: 'dashed' }} />

        <Box sx={{ p: 1 }}>
          <Button fullWidth color="inherit" size="medium" variant="text" sx={{ color: 'text.secondary' }}>
            Logout
          </Button>
        </Box>
      </Popover>
    </>
  );
}
