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

### Deploy to localhost

In a **second terminal** (same `hardhat-blockchain/` directory, with the node still running):

```bash
npm run deploy:local
```

Equivalent: `npx hardhat run scripts/deploy.js --network localhost`

Copy the printed **SimpleStorage** contract address for the next step.

### Interact with the deployed contract

The interact script expects **`SIMPLE_STORAGE_ADDRESS`** to be set to that address.

**Linux / macOS:**

```bash
SIMPLE_STORAGE_ADDRESS=0xYourDeployedAddress npm run interact:local
```

**Windows (PowerShell):**

```powershell
$env:SIMPLE_STORAGE_ADDRESS = "0xYourDeployedAddress"
npm run interact:local
```

Equivalent: `npx hardhat run scripts/interact.js --network localhost` (with the same environment variable).

### npm scripts reference

| Script            | Purpose                          |
| ----------------- | -------------------------------- |
| `npm run compile` | Compile Solidity                 |
| `npm run node`    | Start local chain on `:8545`     |
| `npm run deploy:local` | Deploy `SimpleStorage`    |
| `npm run interact:local` | Call `set` / `get` via script |

### MetaMask (optional)

1. Start the local node (`npm run node`).
2. In MetaMask, add a custom network:
   - **RPC URL:** `http://127.0.0.1:8545`
   - **Chain ID:** `31337`
3. Import an account using a **private key** from the Hardhat node output; that account is pre-funded with test ETH on this node only.
