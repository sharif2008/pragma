# ChainAgentVFL

## Local Ethereum (`hardhat-blockchain/`)

The `hardhat-blockchain` package is a self-contained [Hardhat](https://hardhat.org/) project: Solidity **^0.8.x**, **@nomicfoundation/hardhat-toolbox**, and **ethers.js**. The local JSON-RPC server uses **`http://127.0.0.1:8545`** and chain ID **`31337`**.

All commands below assume your shell’s working directory is `hardhat-blockchain/` (from the repo root: `cd hardhat-blockchain`).

### Prerequisites

- [Node.js](https://nodejs.org/) (LTS recommended) and npm

### First-time setup

```bash
npm install
```

### Compile contracts

```bash
npm run compile
```

Equivalent: `npx hardhat compile`

### Run a local node

Use a **dedicated terminal** and leave it running:

```bash
npm run node
```

Equivalent: `npx hardhat node`

The process prints dev accounts and private keys (local-only; never use on mainnet).

### Deploy registry to localhost

In a **second terminal** (same `hardhat-blockchain/` directory, with the node still running):

```bash
npm run deploy:local
```

Equivalent: `npx hardhat run scripts/deploy.js --network localhost`

Copy the printed **AgenticTrustRegistry** contract address for the next step.

On deploy, the registry is automatically seeded with **per-attack action whitelists** from `contracts/attack_options.json` (e.g. `DDOS` → `limit rate`, `enable scrubbing`, …). To re-seed an existing deployment:

```bash
TRUST_REGISTRY_ADDRESS=0xYourDeployedAddress npm run seed:whitelist
```

PowerShell:

```powershell
$env:TRUST_REGISTRY_ADDRESS = "0xYourDeployedAddress"
npm run seed:whitelist
```

### What the registry stores

- **Whitelist** (owner, at deploy): `whitelist[attackType][action]`, e.g. `whitelist["PORTSCAN"]["tarpit scan"] = true`.
- **Plan** (authorized agent, once per plan id): job id, prediction id, row index, attack type, threat level, primary and supporting `{action, tier}` units, and `sha256(overall_reasoning)`. `storePlan` rejects unknown attacks, non-whitelisted actions, unknown tiers, bad threat levels, duplicates, and more than 10 actions per list.
- **Apply receipt** (authorized agent): `markApplied(planId, action, tier)` succeeds only for a whitelisted unit that is in the stored plan, and only once.

Only the deployer is an authorized agent by default; add others with `setAgent(address, true)`.

### Store a sample plan (demo)

```powershell
$env:TRUST_REGISTRY_ADDRESS = "0xYourDeployedAddress"
npm run plan:local
```

Equivalent: `npx hardhat run scripts/interact.js --network localhost`. It stores a sample PORTSCAN plan, reads it back, and marks one action applied.

### Tests

```bash
npx hardhat test
```

### npm scripts reference

| Script            | Purpose                          |
| ----------------- | -------------------------------- |
| `npm run compile` | Compile Solidity                 |
| `npm run node`    | Start local chain on `:8545`     |
| `npm run deploy:local` | Deploy `AgenticTrustRegistry` and seed action whitelist |
| `npm run seed:whitelist` | Re-seed whitelist on an existing registry (`TRUST_REGISTRY_ADDRESS`) |
| `npm run plan:local` | Store a sample plan, read it back, mark one action applied |

### MetaMask (optional)

1. Start the local node (`npm run node`).
2. In MetaMask, add a custom network:
   - **RPC URL:** `http://127.0.0.1:8545`
   - **Chain ID:** `31337`
3. Import an account using a **private key** from the Hardhat node output; that account is pre-funded with test ETH on this node only.
