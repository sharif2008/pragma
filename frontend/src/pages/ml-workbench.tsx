import { CONFIG } from 'src/config-global';

import { MlWorkbenchView } from 'src/sections/ml/view';

// ----------------------------------------------------------------------

export default function Page() {
  return (
    <>
      <title>{`ML & RAG - ${CONFIG.appName}`}</title>
      <meta name="description" content="ChainAgentVFL datasets, training, predictions, knowledge base, and agent API" />

      <MlWorkbenchView />
    </>
  );
}
