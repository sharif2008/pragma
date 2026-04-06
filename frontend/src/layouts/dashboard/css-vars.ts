import type { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export const DASHBOARD_NAV_WIDTH_EXPANDED = '300px';
export const DASHBOARD_NAV_WIDTH_COLLAPSED = '88px';

export function dashboardLayoutVars(theme: Theme) {
  return {
    '--layout-transition-easing': 'linear',
    '--layout-transition-duration': '200ms',
    '--layout-nav-vertical-width': DASHBOARD_NAV_WIDTH_EXPANDED,
    '--layout-dashboard-content-pt': theme.spacing(1),
    '--layout-dashboard-content-pb': theme.spacing(8),
    '--layout-dashboard-content-px': theme.spacing(5),
  };
}
