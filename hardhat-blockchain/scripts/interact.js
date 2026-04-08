const hre = require("hardhat");

async function main() {
  const address = process.env.TRUST_REGISTRY_ADDRESS;
  if (!address) {
    throw new Error("Set TRUST_REGISTRY_ADDRESS to the deployed contract address.");
  }

  const [signer] = await hre.ethers.getSigners();
  console.log("Using signer:", signer.address);

  const registry = await hre.ethers.getContractAt("AgenticTrustRegistry", address, signer);

  const agentKey = process.env.AGENT_KEY_BYTES32;
  const reportKey = process.env.REPORT_KEY_BYTES32;
  const commitment = process.env.COMMITMENT_BYTES32;
  if (!agentKey || !reportKey || !commitment) {
    throw new Error(
      "Set AGENT_KEY_BYTES32, REPORT_KEY_BYTES32, COMMITMENT_BYTES32 (all 0x-prefixed 32-byte hex)."
    );
  }

  const tx = await registry.anchor(agentKey, reportKey, commitment);
  console.log("anchor tx hash:", tx.hash);
  await tx.wait();

  const stored = await registry.getCommitment(agentKey, reportKey);
  console.log("Stored commitment:", stored);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
