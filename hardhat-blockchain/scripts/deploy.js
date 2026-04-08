const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying with:", deployer.address);

  const AgenticTrustRegistry = await hre.ethers.getContractFactory("AgenticTrustRegistry");
  const registry = await AgenticTrustRegistry.deploy();
  await registry.waitForDeployment();

  const address = await registry.getAddress();
  console.log("AgenticTrustRegistry deployed to:", address);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
