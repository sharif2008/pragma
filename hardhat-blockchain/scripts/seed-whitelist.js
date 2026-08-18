const hre = require("hardhat");
const { seedActionWhitelist } = require("./attack-options");

async function main() {
  const address = process.env.TRUST_REGISTRY_ADDRESS;
  if (!address) {
    throw new Error("Set TRUST_REGISTRY_ADDRESS to the deployed AgenticTrustRegistry address.");
  }

  const [signer] = await hre.ethers.getSigners();
  console.log("Seeding whitelist with owner/signer:", signer.address);
  console.log("Registry:", address);

  const registry = await hre.ethers.getContractAt("AgenticTrustRegistry", address, signer);
  await seedActionWhitelist(registry);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
