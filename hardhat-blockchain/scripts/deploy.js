const hre = require("hardhat");
const { seedActionWhitelist } = require("./attack-options");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const AgenticTrustRegistry = await hre.ethers.getContractFactory("AgenticTrustRegistry");
  const registry = await AgenticTrustRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log("AgenticTrustRegistry deployed to:", address);

  console.log("Seeding action whitelist from contracts/attack_options.json …");
  await seedActionWhitelist(registry);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
