const fs = require("fs");
const path = require("path");

/** Path to contracts/attack_options.json (bundled with the registry). */
function attackOptionsPath() {
  return path.join(__dirname, "..", "contracts", "attack_options.json");
}

function loadAttackOptions() {
  const filePath = attackOptionsPath();
  const raw = fs.readFileSync(filePath, "utf8");
  const data = JSON.parse(raw);
  if (!data.attacks || typeof data.attacks !== "object") {
    throw new Error(`attack_options.json missing "attacks" object: ${filePath}`);
  }
  return data;
}

/**
 * Seed on-chain whitelist from attack_options.json (plain-text labels):
 * attacks[ATTACK_TYPE] => [ "limit rate", "block IP", ... ]
 */
async function seedActionWhitelist(registry) {
  const data = loadAttackOptions();
  const attacks = data.attacks;
  let total = 0;

  for (const [attackType, actions] of Object.entries(attacks)) {
    if (!Array.isArray(actions) || actions.length === 0) {
      console.warn(`Skipping ${attackType}: no actions`);
      continue;
    }
    const tx = await registry.addAllowedActions(attackType, actions.map(String));
    await tx.wait();
    total += actions.length;
    console.log(`Whitelisted ${actions.length} action(s) for ${attackType}`);
  }

  console.log(`Whitelist seed complete: ${Object.keys(attacks).length} attack type(s), ${total} action slot(s).`);
  return total;
}

module.exports = {
  attackOptionsPath,
  loadAttackOptions,
  seedActionWhitelist,
};
