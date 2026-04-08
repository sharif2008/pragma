import type { Theme, SxProps, Breakpoint } from '@mui/material/styles';

import { Fragment, useEffect } from 'react';
import { varAlpha } from 'minimal-shared/utils';

import Box from '@mui/material/Box';
import Tooltip from '@mui/material/Tooltip';
import ListItem from '@mui/material/ListItem';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';
import ListItemButton from '@mui/material/ListItemButton';
import Drawer, { drawerClasses } from '@mui/material/Drawer';

import { usePathname } from 'src/routes/hooks';
import { RouterLink } from 'src/routes/components';

import { Logo } from 'src/components/logo';
import { Iconify } from 'src/components/iconify';
import { Scrollbar } from 'src/components/scrollbar';

import type { NavItem } from '../nav-config-dashboard';

// ----------------------------------------------------------------------

export type NavContentProps = {
  data: NavItem[];
  slots?: {
    topArea?: React.ReactNode;
    bottomArea?: React.ReactNode;
  };
  sx?: SxProps<Theme>;
};

export function NavDesktop({
  sx,
  data,
  slots,
  layoutQuery,
  collapsed = false,
  onToggleCollapsed,
}: NavContentProps & {
  layoutQuery: Breakpoint;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}) {
  const theme = useTheme();

  return (
    <Box
      sx={{
        pt: 2.5,
        px: collapsed ? 1 : 2.5,
        top: 0,
        left: 0,
        height: 1,
        display: 'none',
        position: 'fixed',
        flexDirection: 'column',
        zIndex: 'var(--layout-nav-zIndex)',
        width: 'var(--layout-nav-vertical-width)',
        overflow: 'hidden',
        borderRight: `1px solid ${varAlpha(theme.vars.palette.grey['500Channel'], 0.12)}`,
        transition: theme.transitions.create(['width', 'padding'], {
          easing: 'var(--layout-transition-easing)',
          duration: 'var(--layout-transition-duration)',
        }),
        [theme.breakpoints.up(layoutQuery)]: {
          display: 'flex',
        },
        ...sx,
      }}
    >
      <NavContent
        data={data}
        slots={slots}
        collapsed={collapsed}
        onToggleCollapsed={onToggleCollapsed}
      />
    </Box>
  );
}

// ----------------------------------------------------------------------

export function NavMobile({
  sx,
  data,
  open,
  slots,
  onClose,
}: NavContentProps & { open: boolean; onClose: () => void }) {
  const pathname = usePathname();

  useEffect(() => {
    if (open) {
      onClose();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  return (
    <Drawer
      open={open}
      onClose={onClose}
      sx={{
        [`& .${drawerClasses.paper}`]: {
          pt: 2.5,
          px: 2.5,
          overflow: 'unset',
          width: 'var(--layout-nav-mobile-width)',
          ...sx,
        },
      }}
    >
      <NavContent data={data} slots={slots} />
    </Drawer>
  );
}

// ----------------------------------------------------------------------

export function NavContent({
  data,
  slots,
  sx,
  collapsed = false,
  onToggleCollapsed,
}: NavContentProps & {
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}) {
  const pathname = usePathname();

  return (
    <>
      <Box
        sx={{
          display: 'flex',
          justifyContent: collapsed ? 'center' : 'flex-start',
          width: 1,
          mb: 2,
        }}
      >
        <Logo />
      </Box>

      {slots?.topArea}

      <Scrollbar fillContent>
        <Box
          component="nav"
          sx={[
            {
              display: 'flex',
              flex: '1 1 auto',
              flexDirection: 'column',
            },
            ...(Array.isArray(sx) ? sx : [sx]),
          ]}
        >
          <Box
            component="ul"
            sx={{
              gap: 0.5,
              display: 'flex',
              flexDirection: 'column',
            }}
          >
            {data.map((item) => {
              const isActived =
                item.path === pathname ||
                (item.path === '/agentic' && pathname.startsWith('/agentic'));

              const button = (
                <ListItemButton
                  disableGutters
                  component={RouterLink}
                  href={item.path}
                  aria-label={collapsed ? item.title : undefined}
                  sx={[
                    (theme) => ({
                      pl: collapsed ? 1 : 2,
                      py: 1,
                      gap: collapsed ? 0 : 2,
                      pr: collapsed ? 1 : 1.5,
                      borderRadius: 0.75,
                      typography: 'body2',
                      fontWeight: 'fontWeightMedium',
                      color: theme.vars.palette.text.secondary,
                      minHeight: 44,
                      justifyContent: collapsed ? 'center' : 'flex-start',
                      ...(isActived && {
                        fontWeight: 'fontWeightSemiBold',
                        color: theme.vars.palette.primary.main,
                        bgcolor: varAlpha(theme.vars.palette.primary.mainChannel, 0.08),
                        '&:hover': {
                          bgcolor: varAlpha(theme.vars.palette.primary.mainChannel, 0.16),
                        },
                      }),
                    }),
                  ]}
                >
                  <Box component="span" sx={{ width: 24, height: 24, flexShrink: 0 }}>
                    {item.icon}
                  </Box>

                  {!collapsed && (
                    <>
                      <Box component="span" sx={{ flexGrow: 1 }}>
                        {item.title}
                      </Box>

                      {item.info && item.info}
                    </>
                  )}
                </ListItemButton>
              );

              return (
                <Fragment key={item.title}>
                  <ListItem disableGutters disablePadding>
                    {collapsed ? (
                      <Tooltip title={item.title} placement="right">
                        {button}
                      </Tooltip>
                    ) : (
                      button
                    )}
                  </ListItem>
                </Fragment>
              );
            })}
          </Box>
        </Box>
      </Scrollbar>

      {onToggleCollapsed && (
        <Box sx={{ display: 'flex', justifyContent: collapsed ? 'center' : 'flex-end', pt: 1, pb: 0.5 }}>
          <Tooltip title={collapsed ? 'Expand sidebar' : 'Minimize sidebar'} placement="right">
            <IconButton
              size="small"
              onClick={onToggleCollapsed}
              aria-label={collapsed ? 'Expand sidebar' : 'Minimize sidebar'}
              sx={{ color: 'text.secondary' }}
            >
              <Iconify
                width={20}
                icon="eva:arrow-ios-forward-fill"
                sx={!collapsed ? { transform: 'scaleX(-1)' } : undefined}
              />
            </IconButton>
          </Tooltip>
        </Box>
      )}

      {slots?.bottomArea}
    </>
  );
}
