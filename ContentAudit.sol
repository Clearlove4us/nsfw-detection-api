// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ContentAudit {
    struct AuditRecord {
        string contentHash;
        bool isUnsafe;
        uint256 score;
        uint256 timestamp;
        address auditor;
    }

    mapping(string => AuditRecord) public records;

    event DetectionLogged(string indexed contentHash, bool isUnsafe, uint256 score, uint256 timestamp);

    function logDetection(string memory _contentHash, bool _isUnsafe, uint256 _score) public {
        require(records[_contentHash].timestamp == 0, "Record already exists");

        records[_contentHash] = AuditRecord({
            contentHash: _contentHash,
            isUnsafe: _isUnsafe,
            score: _score,
            timestamp: block.timestamp,
            auditor: msg.sender
        });

        emit DetectionLogged(_contentHash, _isUnsafe, _score, block.timestamp);
    }
}
