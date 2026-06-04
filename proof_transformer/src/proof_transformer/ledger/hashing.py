from __future__ import annotations

import hashlib
import json


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_payload(value: object) -> str:
    return sha256_hex(canonical_json(value))


def compute_policy_hash(policy: dict[str, float]) -> str:
    return hash_payload(policy)


def compute_block_hash(
    *,
    height: int,
    parent_hash: str | None,
    candidate_id: str,
    validator_summary: dict,
    consensus_score: float,
    policy_version: int | None,
    adapter_version: str | None,
    ranker_version: str | None,
    memory_diff: dict,
    created_at: str,
) -> str:
    payload = {
        "height": height,
        "parent_hash": parent_hash,
        "candidate_id": candidate_id,
        "validator_summary": validator_summary,
        "consensus_score": round(float(consensus_score), 8),
        "policy_version": policy_version,
        "adapter_version": adapter_version,
        "ranker_version": ranker_version,
        "memory_diff": memory_diff,
        "created_at": created_at,
    }
    return hash_payload(payload)
