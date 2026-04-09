import { CONFIG } from 'src/config-global';

import { AgenticExecutionReportsView } from 'src/sections/agentic/view';

export default function AgenticExecutionReportsPage() {
  return (
    <>
      <title>{`Execution reports - ${CONFIG.appName}`}</title>
      <meta name="description" content="Execution reports for applied agentic actions (stubbed)" />

      <AgenticExecutionReportsView />
    </>
  );
}

