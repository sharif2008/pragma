import packageJson from '../package.json';

// ----------------------------------------------------------------------

export type ConfigValue = {
  appName: string;
  appVersion: string;
  /** Shown in header menu (e.g. Production / Development). Override with `VITE_DEPLOYMENT_LABEL`. */
  deploymentLabel: string;
};

export const CONFIG: ConfigValue = {
  appName: 'ChainAgentVFL',
  appVersion: packageJson.version,
  deploymentLabel:
    import.meta.env.VITE_DEPLOYMENT_LABEL?.trim() ||
    (import.meta.env.PROD ? 'Production' : 'Development'),
};
