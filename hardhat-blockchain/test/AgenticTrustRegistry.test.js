const { expect } = require("chai");
const { ethers } = require("hardhat");
const { seedActionWhitelist } = require("../scripts/attack-options");
const { configureRoles } = require("../scripts/roles");

const PERIMETER = "Perimeter / IDS";
const ACCESS = "Access / ISP";
const ENDPOINT = "Endpoint / EDR";
const REASONING_HASH = "0x" + "ab".repeat(32);

function portscanPlan(overrides = {}) {
  return {
    jobId: "job_7f3a9c21",
    predictionId: "pred_42",
    rowIndex: 1187,
    attackType: "PORTSCAN",
    threatLevel: "High",
    primaryActions: [
      { action: "tarpit scan", tier: PERIMETER },
      { action: "harden ports", tier: PERIMETER },
    ],
    supportingActions: [{ action: "update ACL", tier: ACCESS }],
    reasoningHash: REASONING_HASH,
    ...overrides,
  };
}

describe("AgenticTrustRegistry", function () {
  let registry, owner, reviewer, accessExec, perimeterExec, endpointExec, agent, outsider;

  beforeEach(async function () {
    const signers = await ethers.getSigners();
    [owner, reviewer, accessExec, perimeterExec, endpointExec, agent, outsider] = signers;
    const Factory = await ethers.getContractFactory("AgenticTrustRegistry");
    registry = await Factory.deploy();
    await registry.waitForDeployment();
    await seedActionWhitelist(registry);
    await configureRoles(registry, signers);
  });

  it("seeds a readable plain-text whitelist", async function () {
    expect(await registry.whitelist("PORTSCAN", "tarpit scan")).to.equal(true);
    expect(await registry.whitelist("PORTSCAN", "enable scrubbing")).to.equal(false);
    expect(await registry.getAllowedActions("PORTSCAN")).to.include("harden ports");
  });

  it("stores and returns a plan with tiers and detection link", async function () {
    await registry.storePlan("report_1", portscanPlan());
    const p = await registry.getPlan("report_1");
    expect(p.jobId).to.equal("job_7f3a9c21");
    expect(p.predictionId).to.equal("pred_42");
    expect(p.rowIndex).to.equal(1187n);
    expect(p.attackType).to.equal("PORTSCAN");
    expect(p.threatLevel).to.equal("High");
    expect(p.primaryActions[0].action).to.equal("tarpit scan");
    expect(p.primaryActions[0].tier).to.equal(PERIMETER);
    expect(p.supportingActions[0].tier).to.equal(ACCESS);
    expect(p.reasoningHash).to.equal(REASONING_HASH);
    expect(p.storedBy).to.equal(owner.address);
    expect(p.origin).to.equal("agent");
    expect(p.parentPlanId).to.equal("");
    expect(p.superseded).to.equal(false);
  });

  it("only authorized agents can store plans", async function () {
    await expect(registry.connect(outsider).storePlan("r", portscanPlan())).to.be.revertedWith(
      "not_authorized_agent"
    );
    await registry.setAgent(agent.address, true);
    await registry.connect(agent).storePlan("r", portscanPlan());
  });

  it("rejects a second write for the same plan id", async function () {
    await registry.storePlan("r", portscanPlan());
    await expect(registry.storePlan("r", portscanPlan())).to.be.revertedWith("already_stored");
  });

  it("rejects actions not whitelisted for the attack", async function () {
    const bad = portscanPlan({ primaryActions: [{ action: "enable scrubbing", tier: PERIMETER }] });
    await expect(registry.storePlan("r", bad)).to.be.revertedWith("action_not_whitelisted");
  });

  it("rejects unknown attacks, bad tiers, bad threat levels, duplicates, and oversized lists", async function () {
    await expect(
      registry.storePlan("r1", portscanPlan({ attackType: "PORTSCN", primaryActions: [], supportingActions: [] }))
    ).to.be.revertedWith("unknown_attack");
    await expect(
      registry.storePlan("r2", portscanPlan({ primaryActions: [{ action: "tarpit scan", tier: "Core" }] }))
    ).to.be.revertedWith("bad_tier");
    await expect(registry.storePlan("r3", portscanPlan({ threatLevel: "Severe" }))).to.be.revertedWith(
      "bad_threat_level"
    );
    await expect(
      registry.storePlan(
        "r4",
        portscanPlan({ supportingActions: [{ action: "tarpit scan", tier: PERIMETER }] })
      )
    ).to.be.revertedWith("duplicate_action");
    const many = Array.from({ length: 11 }, () => ({ action: "tarpit scan", tier: PERIMETER }));
    await expect(registry.storePlan("r5", portscanPlan({ primaryActions: many }))).to.be.revertedWith(
      "too_many_primary"
    );
  });

  it("allows the same action on two different tiers", async function () {
    const twoTiers = portscanPlan({
      primaryActions: [{ action: "block IP", tier: PERIMETER }],
      supportingActions: [{ action: "block IP", tier: ACCESS }],
    });
    await registry.storePlan("r", twoTiers);
    expect(await registry.isInPlan("r", "block IP", ACCESS)).to.equal(true);
  });

  it("binds apply to the planned action and tier, once", async function () {
    await registry.storePlan("r", portscanPlan());
    const perim = registry.connect(perimeterExec);
    expect(await registry.isInPlan("r", "tarpit scan", PERIMETER)).to.equal(true);
    expect(await registry.isInPlan("r", "tarpit scan", ENDPOINT)).to.equal(false);

    await expect(perim.markApplied("r", "tarpit scan", ENDPOINT)).to.be.revertedWith("action_plan_mismatch");
    await expect(perim.markApplied("r", "limit rate", PERIMETER)).to.be.revertedWith("action_plan_mismatch");
    await expect(perim.markApplied("r", "enable scrubbing", PERIMETER)).to.be.revertedWith(
      "action_not_whitelisted"
    );

    await perim.markApplied("r", "tarpit scan", PERIMETER);
    expect(await registry.isApplied("r", "tarpit scan", PERIMETER)).to.equal(true);
    await expect(perim.markApplied("r", "tarpit scan", PERIMETER)).to.be.revertedWith("already_applied");
  });

  it("only the tier's executor key can apply an action on that tier", async function () {
    await registry.storePlan("r", portscanPlan());
    await expect(registry.markApplied("r", "tarpit scan", PERIMETER)).to.be.revertedWith("not_tier_executor");
    await expect(
      registry.connect(accessExec).markApplied("r", "tarpit scan", PERIMETER)
    ).to.be.revertedWith("not_tier_executor");
    await registry.connect(accessExec).markApplied("r", "update ACL", ACCESS);
    expect(await registry.tierExecutor(ENDPOINT)).to.equal(endpointExec.address);

    await registry.setTierExecutor(PERIMETER, ethers.ZeroAddress);
    await expect(
      registry.connect(perimeterExec).markApplied("r", "tarpit scan", PERIMETER)
    ).to.be.revertedWith("not_tier_executor");
    await expect(registry.connect(outsider).setTierExecutor(PERIMETER, outsider.address)).to.be.revertedWith(
      "not_owner"
    );
  });

  it("agent re-plans once; the parent is superseded and cannot be applied", async function () {
    await registry.storePlan("r", portscanPlan());
    const replan = portscanPlan({ primaryActions: [{ action: "block IP", tier: PERIMETER }] });
    await registry.revisePlan("r_v2", "r", replan);

    const parent = await registry.getPlan("r");
    const child = await registry.getPlan("r_v2");
    expect(parent.superseded).to.equal(true);
    expect(child.origin).to.equal("replan");
    expect(child.parentPlanId).to.equal("r");

    await expect(
      registry.connect(perimeterExec).markApplied("r", "tarpit scan", PERIMETER)
    ).to.be.revertedWith("plan_superseded");
    await registry.connect(perimeterExec).markApplied("r_v2", "block IP", PERIMETER);

    await expect(registry.revisePlan("r_v3", "r_v2", replan)).to.be.revertedWith("replan_limit");
    await expect(registry.revisePlan("r_v3", "r", replan)).to.be.revertedWith("parent_superseded");
  });

  it("a reviewer commits a human correction on top of a re-plan", async function () {
    await registry.storePlan("r", portscanPlan());
    await registry.revisePlan("r_v2", "r", portscanPlan());
    const human = portscanPlan({ primaryActions: [{ action: "limit rate", tier: PERIMETER }] });
    await registry.connect(reviewer).revisePlan("r_v3", "r_v2", human);

    const p = await registry.getPlan("r_v3");
    expect(p.origin).to.equal("human");
    expect(p.storedBy).to.equal(reviewer.address);
    await registry.connect(perimeterExec).markApplied("r_v3", "limit rate", PERIMETER);
  });

  it("revisions must stay on the same detection and attack", async function () {
    await registry.storePlan("r", portscanPlan());
    await expect(
      registry.connect(reviewer).revisePlan("x", "r", portscanPlan({ predictionId: "pred_99" }))
    ).to.be.revertedWith("detection_mismatch");
    await expect(
      registry.connect(reviewer).revisePlan(
        "x",
        "r",
        portscanPlan({ attackType: "DDOS", primaryActions: [], supportingActions: [] })
      )
    ).to.be.revertedWith("attack_mismatch");
    await expect(registry.connect(outsider).revisePlan("x", "r", portscanPlan())).to.be.revertedWith(
      "not_authorized"
    );
    await expect(registry.revisePlan("x", "missing", portscanPlan())).to.be.revertedWith("parent_not_stored");
  });

  it("owner can remove a whitelisted action, which blocks later applies", async function () {
    await registry.storePlan("r", portscanPlan());
    await registry.removeAllowedAction("PORTSCAN", "harden ports");
    expect(await registry.whitelist("PORTSCAN", "harden ports")).to.equal(false);
    expect(await registry.getAllowedActions("PORTSCAN")).to.not.include("harden ports");
    await expect(
      registry.connect(perimeterExec).markApplied("r", "harden ports", PERIMETER)
    ).to.be.revertedWith("action_not_whitelisted");
    await expect(registry.connect(outsider).removeAllowedAction("PORTSCAN", "tarpit scan")).to.be.revertedWith(
      "not_owner"
    );
  });

  it("transfers ownership", async function () {
    await registry.transferOwnership(agent.address);
    expect(await registry.owner()).to.equal(agent.address);
    await expect(registry.setAgent(outsider.address, true)).to.be.revertedWith("not_owner");
  });
});
