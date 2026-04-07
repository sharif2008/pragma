import { CONFIG } from 'src/config-global';

import { AgenticActionsView } from 'src/sections/agentic/view';

export default function AgenticIndexPage() {
  return (
    <>
      <title>{`Agentic actions - ${CONFIG.appName}`}</title>
      <meta
        name="description"
        content="Timeline of agentic LLM decisions with Core and Edge actions per prediction run"
      />

      <AgenticActionsView />
    </>
  );
}
