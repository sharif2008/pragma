import { CONFIG } from 'src/config-global';

import { SocDashboardView as DashboardView } from 'src/sections/soc/view';

// ----------------------------------------------------------------------

export default function Page() {
  return (
    <>
      <title>{`Dashboard - ${CONFIG.appName}`}</title>
      <meta
        name="description"
        content="SOC dashboard for traffic detection and agentic actions"
      />
      <meta name="keywords" content="soc,monitoring,agentic,rag,traffic,detection" />

      <DashboardView />
    </>
  );
}
