import { CONFIG } from 'src/config-global';

import { MonitorView } from 'src/sections/monitor/view';

export default function Page() {
  return (
    <>
      <title>{`Monitoring - ${CONFIG.appName}`}</title>
      <MonitorView />
    </>
  );
}

