// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * AgenticTrustRegistry
 *
 * Hash-only trust anchoring for agentic reports.
 *
 * Keys are bytes32 so the backend can pre-hash (sha256) identifiers like:
 * - agentic_job_public_id (or "unlinked")
 * - agentic_report.public_id
 *
 * Value is a bytes32 commitment hash (sha256 of canonical JSON payload).
 */
contract AgenticTrustRegistry {
    // agentKey => reportKey => commitment
    mapping(bytes32 => mapping(bytes32 => bytes32)) private commitments;

    event TrustAnchored(bytes32 indexed agentKey, bytes32 indexed reportKey, bytes32 commitment);

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
}

