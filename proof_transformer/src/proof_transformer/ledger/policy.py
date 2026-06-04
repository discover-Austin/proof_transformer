from __future__ import annotations

import json
import sqlite3

from ..ids import new_id
from ..timeutils import utc_now
from .hashing import compute_policy_hash
from .models import DecoderPolicy


class PolicyStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_default(self) -> tuple[int, DecoderPolicy]:
        row = self.conn.execute(
            "SELECT version, policy_json FROM policy_snapshots ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if row:
            return int(row["version"]), DecoderPolicy(**json.loads(row["policy_json"]))
        policy = DecoderPolicy()
        version = 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO policy_snapshots(id, version, policy_json, policy_hash, created_at) VALUES(?, ?, ?, ?, ?)",
                (
                    new_id("policy"),
                    version,
                    json.dumps(policy.as_dict()),
                    compute_policy_hash(policy.as_dict()),
                    utc_now(),
                ),
            )
        return version, policy

    def get_current(self) -> tuple[int, DecoderPolicy]:
        return self.ensure_default()

    def snapshot(self, policy: DecoderPolicy) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM policy_snapshots").fetchone()
        version = int(row["version"]) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO policy_snapshots(id, version, policy_json, policy_hash, created_at) VALUES(?, ?, ?, ?, ?)",
                (
                    new_id("policy"),
                    version,
                    json.dumps(policy.as_dict()),
                    compute_policy_hash(policy.as_dict()),
                    utc_now(),
                ),
            )
        return version

    def apply_delta(self, updates: dict[str, float]) -> tuple[int, DecoderPolicy]:
        _version, policy = self.get_current()
        updated = policy.apply_updates(updates)
        if updated.as_dict() == policy.as_dict():
            current_version, _ = self.get_current()
            return current_version, policy
        version = self.snapshot(updated)
        return version, updated

    def list_snapshots(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM policy_snapshots ORDER BY version ASC").fetchall()
        return [dict(row) for row in rows]
