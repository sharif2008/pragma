// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * AgenticTrustRegistry
 *
 * Plain-text governance record for agentic mitigation plans.
 *
 * Stored once at deploy (owner):
 *   - whitelist: allowed actions per attack type (seeded from contracts/attack_options.json)
 *   - roles: planner agents, human reviewers, one executor key per network tier
 *
 * Stored once per plan (planner agent, or reviewer for a human correction):
 *   - jobId, predictionId, rowIndex      (link to the detection that produced the plan)
 *   - attackType, threatLevel
 *   - primaryActions[], supportingActions[]   each {action, tier}
 *   - reasoningHash = sha256(overall_reasoning); the text itself stays off-chain
 *   - parentPlanId, origin ("agent" | "replan" | "human"), superseded
 *
 * Recorded per executed action (the executor key of that action's tier):
 *   - applied[planId][action, tier]
 *
 * Revision policy: an agent may re-plan an original ("agent") plan once; any further
 * correction must come from a reviewer. Revising a plan supersedes its parent, so the
 * parent can no longer be applied.
 *
 * planId is the agentic report id: one agentic job can produce several plans,
 * so jobId alone is not unique.
 */
contract AgenticTrustRegistry {
    uint256 public constant MAX_ACTIONS_PER_LIST = 10;

    string private constant TIER_ACCESS = "Access / ISP";
    string private constant TIER_PERIMETER = "Perimeter / IDS";
    string private constant TIER_ENDPOINT = "Endpoint / EDR";

    string private constant ORIGIN_AGENT = "agent";
    string private constant ORIGIN_REPLAN = "replan";
    string private constant ORIGIN_HUMAN = "human";

    struct ActionUnit {
        string action;
        string tier;
    }

    struct PlanInput {
        string jobId;
        string predictionId;
        int256 rowIndex; // -1 when the plan is not tied to a single row
        string attackType;
        string threatLevel;
        ActionUnit[] primaryActions;
        ActionUnit[] supportingActions;
        bytes32 reasoningHash;
    }

    struct Plan {
        string jobId;
        string predictionId;
        int256 rowIndex;
        string attackType;
        string threatLevel;
        ActionUnit[] primaryActions;
        ActionUnit[] supportingActions;
        bytes32 reasoningHash;
        address storedBy;
        uint256 storedAt;
        bool exists;
        string parentPlanId;
        string origin;
        bool superseded;
    }

    address public owner;
    mapping(address => bool) public authorizedAgents;
    mapping(address => bool) public reviewers;
    // tier label => the only address allowed to markApplied actions on that tier
    mapping(string => address) public tierExecutor;

    // attackType => action => allowed
    mapping(string => mapping(string => bool)) public whitelist;
    // attackType => allowed actions (readable catalog)
    mapping(string => string[]) private allowedActions;
    // attackType => action => 1-based position in allowedActions (0 = absent)
    mapping(string => mapping(string => uint256)) private allowedIndex;

    // planId => plan
    mapping(string => Plan) private plans;
    // planId => keccak256(action, tier) => in plan
    mapping(string => mapping(bytes32 => bool)) private inPlan;
    // planId => keccak256(action, tier) => applied
    mapping(string => mapping(bytes32 => bool)) private applied;

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event AgentAuthorized(address indexed agent, bool allowed);
    event ReviewerAuthorized(address indexed reviewer, bool allowed);
    event TierExecutorSet(string tier, address indexed executor);
    event ActionWhitelisted(string attackType, string action, bool allowed);
    event PlanStored(string planId, string jobId, string attackType, string threatLevel, address storedBy);
    event PlanRevised(string planId, string parentPlanId, string origin, address storedBy);
    event ActionApplied(string planId, string action, string tier, address executor);

    modifier onlyOwner() {
        require(msg.sender == owner, "not_owner");
        _;
    }

    modifier onlyAgent() {
        require(authorizedAgents[msg.sender], "not_authorized_agent");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedAgents[msg.sender] = true;
        emit OwnershipTransferred(address(0), msg.sender);
        emit AgentAuthorized(msg.sender, true);
    }

    // ---------------------------------------------------------------- admin

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "owner=0");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    function setAgent(address agent, bool allowed) external onlyOwner {
        require(agent != address(0), "agent=0");
        authorizedAgents[agent] = allowed;
        emit AgentAuthorized(agent, allowed);
    }

    function setReviewer(address reviewer, bool allowed) external onlyOwner {
        require(reviewer != address(0), "reviewer=0");
        reviewers[reviewer] = allowed;
        emit ReviewerAuthorized(reviewer, allowed);
    }

    /// @param executor address(0) disables execution on that tier.
    function setTierExecutor(string calldata tier, address executor) external onlyOwner {
        require(_validTier(tier), "bad_tier");
        tierExecutor[tier] = executor;
        emit TierExecutorSet(tier, executor);
    }

    // ------------------------------------------------------------ whitelist

    function addAllowedActions(string calldata attackType, string[] calldata actions) external onlyOwner {
        require(bytes(attackType).length > 0, "empty_attack");
        for (uint256 i = 0; i < actions.length; i++) {
            string calldata a = actions[i];
            require(bytes(a).length > 0, "empty_action");
            if (whitelist[attackType][a]) continue;
            whitelist[attackType][a] = true;
            allowedActions[attackType].push(a);
            allowedIndex[attackType][a] = allowedActions[attackType].length;
            emit ActionWhitelisted(attackType, a, true);
        }
    }

    function removeAllowedAction(string calldata attackType, string calldata action) external onlyOwner {
        uint256 pos = allowedIndex[attackType][action];
        require(pos != 0, "not_whitelisted");
        string[] storage list = allowedActions[attackType];
        uint256 last = list.length;
        if (pos != last) {
            string memory moved = list[last - 1];
            list[pos - 1] = moved;
            allowedIndex[attackType][moved] = pos;
        }
        list.pop();
        delete allowedIndex[attackType][action];
        whitelist[attackType][action] = false;
        emit ActionWhitelisted(attackType, action, false);
    }

    function getAllowedActions(string calldata attackType) external view returns (string[] memory) {
        return allowedActions[attackType];
    }

    function isKnownAttack(string calldata attackType) public view returns (bool) {
        return allowedActions[attackType].length > 0;
    }

    // ----------------------------------------------------------------- plan

    function storePlan(string calldata planId, PlanInput calldata p) external onlyAgent {
        _storePlan(planId, p, "", ORIGIN_AGENT);
    }

    /**
     * Store a corrected plan for the same detection and supersede the parent.
     * Reviewers may revise any live plan (origin "human"); agents may revise only an
     * original agent plan (origin "replan"), so the agent gets a single re-plan.
     */
    function revisePlan(string calldata planId, string calldata parentPlanId, PlanInput calldata p) external {
        bool isReviewer = reviewers[msg.sender];
        require(isReviewer || authorizedAgents[msg.sender], "not_authorized");
        Plan storage parent = plans[parentPlanId];
        require(parent.exists, "parent_not_stored");
        require(!parent.superseded, "parent_superseded");
        require(_eq(parent.predictionId, p.predictionId) && parent.rowIndex == p.rowIndex, "detection_mismatch");
        require(_eq(parent.attackType, p.attackType), "attack_mismatch");
        if (!isReviewer) {
            require(_eq(parent.origin, ORIGIN_AGENT), "replan_limit");
        }

        string memory origin = isReviewer ? ORIGIN_HUMAN : ORIGIN_REPLAN;
        _storePlan(planId, p, parentPlanId, origin);
        parent.superseded = true;
        emit PlanRevised(planId, parentPlanId, origin, msg.sender);
    }

    function getPlan(string calldata planId) external view returns (Plan memory) {
        require(plans[planId].exists, "not_stored");
        return plans[planId];
    }

    function isInPlan(string calldata planId, string calldata action, string calldata tier)
        external
        view
        returns (bool)
    {
        return inPlan[planId][_unitKey(action, tier)];
    }

    // ---------------------------------------------------------------- apply

    function markApplied(string calldata planId, string calldata action, string calldata tier) external {
        Plan storage s = plans[planId];
        require(s.exists, "not_stored");
        require(!s.superseded, "plan_superseded");
        require(whitelist[s.attackType][action], "action_not_whitelisted");
        bytes32 k = _unitKey(action, tier);
        require(inPlan[planId][k], "action_plan_mismatch");
        address executor = tierExecutor[tier];
        require(executor != address(0) && msg.sender == executor, "not_tier_executor");
        require(!applied[planId][k], "already_applied");
        applied[planId][k] = true;
        emit ActionApplied(planId, action, tier, msg.sender);
    }

    function isApplied(string calldata planId, string calldata action, string calldata tier)
        external
        view
        returns (bool)
    {
        return applied[planId][_unitKey(action, tier)];
    }

    // -------------------------------------------------------------- helpers

    function _storePlan(
        string calldata planId,
        PlanInput calldata p,
        string memory parentPlanId,
        string memory origin
    ) private {
        require(bytes(planId).length > 0, "empty_plan_id");
        require(!plans[planId].exists, "already_stored");
        require(bytes(p.jobId).length > 0, "empty_job");
        require(bytes(p.predictionId).length > 0, "empty_prediction");
        require(p.rowIndex >= -1, "bad_row_index");
        require(isKnownAttack(p.attackType), "unknown_attack");
        require(_validThreatLevel(p.threatLevel), "bad_threat_level");
        require(p.reasoningHash != bytes32(0), "empty_reasoning");
        require(p.primaryActions.length <= MAX_ACTIONS_PER_LIST, "too_many_primary");
        require(p.supportingActions.length <= MAX_ACTIONS_PER_LIST, "too_many_supporting");

        Plan storage s = plans[planId];
        s.jobId = p.jobId;
        s.predictionId = p.predictionId;
        s.rowIndex = p.rowIndex;
        s.attackType = p.attackType;
        s.threatLevel = p.threatLevel;
        s.reasoningHash = p.reasoningHash;
        s.storedBy = msg.sender;
        s.storedAt = block.timestamp;
        s.exists = true;
        s.parentPlanId = parentPlanId;
        s.origin = origin;

        for (uint256 i = 0; i < p.primaryActions.length; i++) {
            _addUnit(planId, p.attackType, p.primaryActions[i], s.primaryActions);
        }
        for (uint256 i = 0; i < p.supportingActions.length; i++) {
            _addUnit(planId, p.attackType, p.supportingActions[i], s.supportingActions);
        }

        emit PlanStored(planId, p.jobId, p.attackType, p.threatLevel, msg.sender);
    }

    function _addUnit(
        string calldata planId,
        string calldata attackType,
        ActionUnit calldata u,
        ActionUnit[] storage target
    ) private {
        require(whitelist[attackType][u.action], "action_not_whitelisted");
        require(_validTier(u.tier), "bad_tier");
        bytes32 k = _unitKey(u.action, u.tier);
        require(!inPlan[planId][k], "duplicate_action");
        inPlan[planId][k] = true;
        target.push(ActionUnit(u.action, u.tier));
    }

    function _unitKey(string calldata action, string calldata tier) private pure returns (bytes32) {
        return keccak256(abi.encode(action, tier));
    }

    function _eq(string memory a, string memory b) private pure returns (bool) {
        return keccak256(bytes(a)) == keccak256(bytes(b));
    }

    function _validTier(string calldata t) private pure returns (bool) {
        bytes32 h = keccak256(bytes(t));
        return h == keccak256(bytes(TIER_ACCESS)) || h == keccak256(bytes(TIER_PERIMETER))
            || h == keccak256(bytes(TIER_ENDPOINT));
    }

    function _validThreatLevel(string calldata t) private pure returns (bool) {
        bytes32 h = keccak256(bytes(t));
        return h == keccak256("Critical") || h == keccak256("High") || h == keccak256("Medium")
            || h == keccak256("Low");
    }
}
