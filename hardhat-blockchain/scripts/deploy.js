const hre = require("hardhat");
const { seedActionWhitelist } = require("./attack-options");
const { configureRoles } = require("./roles");

async function main() {
  const signers = await hre.ethers.getSigners();
  const [deployer] = signers;
  console.log("Deploying with:", deployer.address);

  const AgenticTrustRegistry = await hre.ethers.getContractFactory("AgenticTrustRegistry");
  const registry = await AgenticTrustRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log("AgenticTrustRegistry deployed to:", address);

  console.log("Seeding action whitelist from contracts/attack_options.json ...");
  await seedActionWhitelist(registry);

  console.log("Assigning reviewer and tier executor roles ...");
  await configureRoles(registry, signers);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
