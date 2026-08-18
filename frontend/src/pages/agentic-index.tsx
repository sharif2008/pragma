import { CONFIG } from 'src/config-global';

import { AgenticActionsView } from 'src/sections/agentic/view';

export default function AgenticIndexPage() {
  return (
    <>
      <title>{`Agentic actions - ${CONFIG.appName}`}</title>
      <meta
        name="description"
        content="Agentic plans with attack type, network-tier actions, and on-chain whitelist apply results"
      />

      <AgenticActionsView />
    </>
  );
}
