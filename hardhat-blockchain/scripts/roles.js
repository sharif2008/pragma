const TIERS = ["Access / ISP", "Perimeter / IDS", "Endpoint / EDR"];

/**
 * Hardhat signer layout used by deploy, tests, and the backend .env:
 *   #0 owner + planner agent, #1 human reviewer,
 *   #2 Access / ISP executor, #3 Perimeter / IDS executor, #4 Endpoint / EDR executor.
 */
async function configureRoles(registry, signers) {
  const [, reviewer, ...executors] = signers;
  await (await registry.setReviewer(reviewer.address, true)).wait();
  console.log(`  reviewer: ${reviewer.address}`);
  for (let i = 0; i < TIERS.length; i++) {
    await (await registry.setTierExecutor(TIERS[i], executors[i].address)).wait();
    console.log(`  executor ${TIERS[i]}: ${executors[i].address}`);
  }
}

module.exports = { TIERS, configureRoles };
