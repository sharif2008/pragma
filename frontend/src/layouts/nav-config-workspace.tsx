import type { WorkspacesPopoverProps } from './components/workspaces-popover';

// ----------------------------------------------------------------------

/** Default workspace shown in the sidebar team switcher (matches demo account branding). */
export const _workspaces: WorkspacesPopoverProps['data'] = [
  {
    id: 'chainagent-default',
    name: 'ChainAgentVFL',
    plan: 'Local',
    logo: '/assets/icons/workspaces/logo-1.webp',
  },
];
