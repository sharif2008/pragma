// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * AgenticTrustRegistry
 *
 * Hash-only trust anchoring for agentic reports plus on-chain whitelists for
 * mitigation actions per attack type (seeded from contracts/attack_options.json).
 *
 * Keys are bytes32 so the backend can pre-hash identifiers:
 * - agentic_job_public_id (or "unlinked")  -> agentKey   (sha256 off-chain, bytes32 on-chain)
 * - agentic_report.public_id               -> reportKey  (sha256 off-chain, bytes32 on-chain)
 * - attack type label e.g. "DDOS"          -> attackKey  (keccak256(bytes(label)))
 * - action string e.g. "limit rate"        -> actionKey  (keccak256(bytes(action)))
 *
 * Value is a bytes32 commitment hash (sha256 of canonical JSON payload).
 */
contract AgenticTrustRegistry {
    address public owner;

    // agentKey => reportKey => commitment
    mapping(bytes32 => mapping(bytes32 => bytes32)) private commitments;

    // attackKey => actionKey => allowed (from attack_options.json per attack type)
    mapping(bytes32 => mapping(bytes32 => bool)) private actionWhitelist;

    // agentKey => reportKey => actionKey => applied on-chain
    mapping(bytes32 => mapping(bytes32 => mapping(bytes32 => bool))) private appliedActions;

    event TrustAnchored(bytes32 indexed agentKey, bytes32 indexed reportKey, bytes32 commitment);
    event ActionWhitelisted(bytes32 indexed attackKey, bytes32 indexed actionKey, bool allowed);
    event ActionApplied(
        bytes32 indexed attackKey,
        bytes32 indexed actionKey,
        bytes32 indexed agentKey,
        bytes32 reportKey,
        address executor
    );
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    modifier onlyOwner() {
        require(msg.sender == owner, "not_owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        emit OwnershipTransferred(address(0), msg.sender);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "owner=0");
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /// @dev Canonical attack-type key (matches deploy seed + backend keccak256 UTF-8 label).
    function attackKey(string calldata attackType) external pure returns (bytes32) {
        return keccak256(bytes(attackType));
    }

    /// @dev Canonical mitigation-action key (matches deploy seed + backend keccak256 UTF-8 action).
    function actionKey(string calldata action) external pure returns (bytes32) {
        return keccak256(bytes(action));
    }

    function setActionWhitelisted(bytes32 attackKey_, bytes32 actionKey_, bool allowed) external onlyOwner {
        require(attackKey_ != bytes32(0), "attackKey=0");
        require(actionKey_ != bytes32(0), "actionKey=0");
        actionWhitelist[attackKey_][actionKey_] = allowed;
        emit ActionWhitelisted(attackKey_, actionKey_, allowed);
    }

    /// @dev Batch whitelist actions for one attack type (used at deploy from attack_options.json).
    function batchWhitelistActions(bytes32 attackKey_, bytes32[] calldata actionKeys_) external onlyOwner {
        require(attackKey_ != bytes32(0), "attackKey=0");
        for (uint256 i = 0; i < actionKeys_.length; i++) {
            bytes32 ak = actionKeys_[i];
            require(ak != bytes32(0), "actionKey=0");
            actionWhitelist[attackKey_][ak] = true;
            emit ActionWhitelisted(attackKey_, ak, true);
        }
    }

    function isActionWhitelisted(bytes32 attackKey_, bytes32 actionKey_) external view returns (bool) {
        return actionWhitelist[attackKey_][actionKey_];
    }

    function anchor(bytes32 agentKey, bytes32 reportKey, bytes32 commitment) external {
        require(agentKey != bytes32(0), "agentKey=0");
        require(reportKey != bytes32(0), "reportKey=0");
        require(commitment != bytes32(0), "commitment=0");
        require(commitments[agentKey][reportKey] == bytes32(0), "already_anchored");
        commitments[agentKey][reportKey] = commitment;
        emit TrustAnchored(agentKey, reportKey, commitment);
    }

    function getCommitment(bytes32 agentKey, bytes32 reportKey) external view returns (bytes32) {
        return commitments[agentKey][reportKey];
    }

    /// @dev Record an applied mitigation action after BC anchor + whitelist check.
    function applyAction(
        bytes32 attackKey_,
        bytes32 actionKey_,
        bytes32 agentKey,
        bytes32 reportKey
    ) external {
        require(actionWhitelist[attackKey_][actionKey_], "action_not_whitelisted");
        require(commitments[agentKey][reportKey] != bytes32(0), "not_anchored");
        require(!appliedActions[agentKey][reportKey][actionKey_], "already_applied");
        appliedActions[agentKey][reportKey][actionKey_] = true;
        emit ActionApplied(attackKey_, actionKey_, agentKey, reportKey, msg.sender);
    }

    function isActionApplied(bytes32 agentKey, bytes32 reportKey, bytes32 actionKey_) external view returns (bool) {
        return appliedActions[agentKey][reportKey][actionKey_];
    }
}
