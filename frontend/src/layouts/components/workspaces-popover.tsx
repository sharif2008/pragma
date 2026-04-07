import type { ButtonBaseProps } from '@mui/material/ButtonBase';

import { varAlpha } from 'minimal-shared/utils';
import { useState, useCallback, useEffect } from 'react';

import Box from '@mui/material/Box';
import Popover from '@mui/material/Popover';
import MenuList from '@mui/material/MenuList';
import ButtonBase from '@mui/material/ButtonBase';
import MenuItem, { menuItemClasses } from '@mui/material/MenuItem';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export type WorkspacesPopoverProps = ButtonBaseProps & {
  data?: {
    id: string;
    name: string;
    logo: string;
    plan: string;
  }[];
  /** Icon-only trigger when the dashboard nav is minimized */
  collapsed?: boolean;
};

export function WorkspacesPopover({ data = [], sx, collapsed, ...other }: WorkspacesPopoverProps) {
  const [workspace, setWorkspace] = useState<(typeof data)[number] | undefined>(() => data[0]);

  useEffect(() => {
    if (!data.length) {
      setWorkspace(undefined);
      return;
    }
    setWorkspace((prev) => {
      if (prev && data.some((d) => d.id === prev.id)) {
        return prev;
      }
      return data[0];
    });
  }, [data]);

  const [openPopover, setOpenPopover] = useState<HTMLButtonElement | null>(null);

  const handleOpenPopover = useCallback((event: React.MouseEvent<HTMLButtonElement>) => {
    setOpenPopover(event.currentTarget);
  }, []);

  const handleClosePopover = useCallback(() => {
    setOpenPopover(null);
  }, []);

  const handleChangeWorkspace = useCallback(
    (newValue: (typeof data)[number]) => {
      setWorkspace(newValue);
      handleClosePopover();
    },
    [handleClosePopover]
  );

  const renderAvatar = (alt: string, src: string) => (
    <Box component="img" alt={alt} src={src} sx={{ width: 24, height: 24, borderRadius: '50%' }} />
  );

  const renderLabel = (plan: string) => (
    <Label
      color={
        plan === 'Free' || plan === 'Local'
          ? 'default'
          : plan === 'Production'
            ? 'success'
            : 'info'
      }
    >
      {plan}
    </Label>
  );

  return (
    <>
      <ButtonBase
        disableRipple
        onClick={handleOpenPopover}
        aria-label={collapsed ? 'Workspace menu' : undefined}
        sx={{
          pl: collapsed ? 0 : 2,
          py: collapsed ? 1.5 : 3,
          gap: 1.5,
          pr: collapsed ? 0 : 1.5,
          width: 1,
          borderRadius: 1.5,
          textAlign: collapsed ? 'center' : 'left',
          justifyContent: collapsed ? 'center' : 'flex-start',
          bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
          ...sx,
        }}
        {...other}
      >
        {renderAvatar(workspace?.name ?? 'Workspace', workspace?.logo ?? '')}

        {!collapsed && (
          <>
            <Box
              sx={{
                gap: 1,
                flexGrow: 1,
                display: 'flex',
                alignItems: 'center',
                typography: 'body2',
                fontWeight: 'fontWeightSemiBold',
              }}
            >
              {workspace?.name ?? 'Workspace'}
              {renderLabel(workspace?.plan ?? 'Local')}
            </Box>

            <Iconify width={16} icon="carbon:chevron-sort" sx={{ color: 'text.disabled' }} />
          </>
        )}
      </ButtonBase>

      <Popover open={!!openPopover} anchorEl={openPopover} onClose={handleClosePopover}>
        <MenuList
          disablePadding
          sx={{
            p: 0.5,
            gap: 0.5,
            width: 260,
            display: 'flex',
            flexDirection: 'column',
            [`& .${menuItemClasses.root}`]: {
              p: 1.5,
              gap: 1.5,
              borderRadius: 0.75,
              [`&.${menuItemClasses.selected}`]: {
                bgcolor: 'action.selected',
                fontWeight: 'fontWeightSemiBold',
              },
            },
          }}
        >
          {data.map((option) => (
            <MenuItem
              key={option.id}
              selected={option.id === workspace?.id}
              onClick={() => handleChangeWorkspace(option)}
            >
              {renderAvatar(option.name, option.logo)}

              <Box component="span" sx={{ flexGrow: 1 }}>
                {option.name}
              </Box>

              {renderLabel(option.plan)}
            </MenuItem>
          ))}
        </MenuList>
      </Popover>
    </>
  );
}
