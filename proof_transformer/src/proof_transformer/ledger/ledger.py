from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..ids import new_id
from ..timeutils import utc_now
from .hashing import compute_block_hash
from .policy import PolicyStore


class LearningLedger:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.policy_store = PolicyStore(conn)

    def create_genesis_if_missing(self) -> dict:
        latest = self.get_latest_block()
        if latest:
            return latest
        policy_version, _policy = self.policy_store.ensure_default()
        created_at = utc_now()
        block_hash = compute_block_hash(
            height=0,
            parent_hash=None,
            candidate_id="genesis",
            validator_summary={"genesis": True},
            consensus_score=1.0,
            policy_version=policy_version,
            adapter_version=None,
            ranker_version=None,
            memory_diff={"type": "genesis", "message": "Proof-of-Improvement Persistent Transformer initialized."},
            created_at=created_at,
        )
        block_id = new_id("block")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learning_blocks(
                    id, height, parent_id, parent_hash, candidate_id, policy_version, adapter_version,
                    ranker_version, memory_diff_json, validator_summary_json, consensus_score,
                    status, block_hash, created_at
                ) VALUES(?, 0, NULL, NULL, ?, ?, NULL, NULL, ?, ?, ?, 'accepted', ?, ?)
                """,
                (
                    block_id,
                    "genesis",
                    policy_version,
                    json.dumps({"type": "genesis", "message": "Proof-of-Improvement Persistent Transformer initialized."}),
                    json.dumps({"genesis": True}),
                    1.0,
                    block_hash,
                    created_at,
                ),
            )
        return self.get_latest_block()

    def get_latest_block(self) -> dict | None:
        row = self.conn.execute("SELECT * FROM learning_blocks ORDER BY height DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def append_block(
        self,
        *,
        candidate_id: str,
        status: str,
        memory_diff: dict,
        validator_summary: dict,
        consensus_score: float,
        policy_version: int | None = None,
        adapter_version: str | None = None,
        ranker_version: str | None = None,
        created_at: str | None = None,
    ) -> dict:
        parent = self.create_genesis_if_missing()
        height = int(parent["height"]) + 1
        created_at = created_at or utc_now()
        block_hash = compute_block_hash(
            height=height,
            parent_hash=parent["block_hash"],
            candidate_id=candidate_id,
            validator_summary=validator_summary,
            consensus_score=consensus_score,
            policy_version=policy_version,
            adapter_version=adapter_version,
            ranker_version=ranker_version,
            memory_diff=memory_diff,
            created_at=created_at,
        )
        block_id = new_id("block")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO learning_blocks(
                    id, height, parent_id, parent_hash, candidate_id, policy_version, adapter_version,
                    ranker_version, memory_diff_json, validator_summary_json, consensus_score,
                    status, block_hash, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block_id,
                    height,
                    parent["id"],
                    parent["block_hash"],
                    candidate_id,
                    policy_version,
                    adapter_version,
                    ranker_version,
                    json.dumps(memory_diff),
                    json.dumps(validator_summary),
                    float(consensus_score),
                    status,
                    block_hash,
                    created_at,
                ),
            )
        row = self.conn.execute("SELECT * FROM learning_blocks WHERE id = ?", (block_id,)).fetchone()
        return dict(row)

    def verify_chain(self) -> dict:
        rows = self.conn.execute("SELECT * FROM learning_blocks ORDER BY height ASC").fetchall()
        errors: list[str] = []
        previous: sqlite3.Row | None = None
        expected_height = 0
        row_by_id = {row["id"]: row for row in rows}

        for row in rows:
            height = int(row["height"])
            if height != expected_height:
                errors.append(f"height discontinuity at {row['id']}: expected {expected_height}, got {height}")
                expected_height = height
            if height == 0:
                if row["parent_id"] or row["parent_hash"]:
                    errors.append("genesis block must not have parent")
            else:
                parent_id = row["parent_id"]
                if not parent_id or parent_id not in row_by_id:
                    errors.append(f"missing parent for block {row['id']}")
                else:
                    parent = row_by_id[parent_id]
                    if row["parent_hash"] != parent["block_hash"]:
                        errors.append(f"bad parent hash for block {row['id']}")
                    if previous is not None and parent["id"] != previous["id"]:
                        errors.append(f"height discontinuity at block {row['id']}: parent does not match previous height")

            recomputed = compute_block_hash(
                height=height,
                parent_hash=row["parent_hash"],
                candidate_id=row["candidate_id"],
                validator_summary=json.loads(row["validator_summary_json"]),
                consensus_score=float(row["consensus_score"]),
                policy_version=row["policy_version"],
                adapter_version=row["adapter_version"],
                ranker_version=row["ranker_version"],
                memory_diff=json.loads(row["memory_diff_json"]),
                created_at=row["created_at"],
            )
            if recomputed != row["block_hash"]:
                errors.append(f"recomputed hash mismatch for block {row['id']}")
            previous = row
            expected_height = height + 1

        return {
            "ok": not errors,
            "block_count": len(rows),
            "errors": errors,
            "latest_block": dict(rows[-1]) if rows else None,
        }

    def export_audit_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        counts = self.conn.execute(
            "SELECT status, COUNT(*) AS count FROM learning_blocks GROUP BY status"
        ).fetchall()
        report = {
            "verification": self.verify_chain(),
            "status_counts": {row["status"]: row["count"] for row in counts},
            "latest_block": self.get_latest_block(),
        }
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return target
