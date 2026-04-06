const hre = require("hardhat");

async function main() {
  const address = process.env.SIMPLE_STORAGE_ADDRESS;
  if (!address) {
    throw new Error("Set SIMPLE_STORAGE_ADDRESS to the deployed contract address.");
  }

  const [signer] = await hre.ethers.getSigners();
  console.log("Using signer:", signer.address);

  const simpleStorage = await hre.ethers.getContractAt("SimpleStorage", address, signer);

  const tx = await simpleStorage.set(42n);
  console.log("set(42) tx hash:", tx.hash);
  await tx.wait();

  const value = await simpleStorage.get();
  console.log("Stored value:", value.toString());
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
