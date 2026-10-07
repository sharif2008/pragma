const hre = require("hardhat");
const crypto = require("crypto");

/**
 * Store one sample PORTSCAN plan, read it back, and mark one action applied
 * with the Perimeter / IDS executor key (signer #3).
 * Requires TRUST_REGISTRY_ADDRESS and a registry deployed by deploy.js (whitelist + roles).
 */
async function main() {
  const address = process.env.TRUST_REGISTRY_ADDRESS;
  if (!address) {
    throw new Error("Set TRUST_REGISTRY_ADDRESS to the deployed contract address.");
  }

  const [signer, , , perimeterExecutor] = await hre.ethers.getSigners();
  console.log("Using signer:", signer.address);

  const registry = await hre.ethers.getContractAt("AgenticTrustRegistry", address, signer);

  const planId = process.env.PLAN_ID || `demo_${Date.now()}`;
  const reasoning =
    "Perimeter share dominates; scan originates outside the boundary, so the perimeter acts first and Access tightens the ACL.";
  const reasoningHash = "0x" + crypto.createHash("sha256").update(reasoning, "utf8").digest("hex");

  const plan = {
    jobId: "demo_job",
    predictionId: "demo_prediction",
    rowIndex: 0,
    attackType: "PORTSCAN",
    threatLevel: "High",
    primaryActions: [
      { action: "tarpit scan", tier: "Perimeter / IDS" },
      { action: "harden ports", tier: "Perimeter / IDS" },
    ],
    supportingActions: [{ action: "update ACL", tier: "Access / ISP" }],
    reasoningHash,
  };

  const tx = await registry.storePlan(planId, plan);
  console.log("storePlan tx hash:", tx.hash);
  await tx.wait();

  const stored = await registry.getPlan(planId);
  console.log("Stored plan:", stored);

  const applyTx = await registry
    .connect(perimeterExecutor)
    .markApplied(planId, "tarpit scan", "Perimeter / IDS");
  await applyTx.wait();
  console.log("isApplied(tarpit scan):", await registry.isApplied(planId, "tarpit scan", "Perimeter / IDS"));
  console.log("isInPlan(limit rate):", await registry.isInPlan(planId, "limit rate", "Perimeter / IDS"));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
