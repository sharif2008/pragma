import { CONFIG } from 'src/config-global';

import { AgenticTrustAnchorsView } from 'src/sections/agentic/view';

export default function AgenticTrustAnchorsPage() {
  return (
    <>
      <title>{`Trust anchors - ${CONFIG.appName}`}</title>
      <meta
        name="description"
        content="On-chain trust anchors for agentic reports: transaction metadata and integrity verification"
      />

      <AgenticTrustAnchorsView />
    </>
  );
}
