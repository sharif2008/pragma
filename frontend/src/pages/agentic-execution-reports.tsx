import { CONFIG } from 'src/config-global';

import { AgenticExecutionReportsView } from 'src/sections/agentic/view';

export default function AgenticExecutionReportsPage() {
  return (
    <>
      <title>{`Execution reports - ${CONFIG.appName}`}</title>
      <meta name="description" content="Execution reports with on-chain whitelist and per-action apply outcomes" />

      <AgenticExecutionReportsView />
    </>
  );
}

