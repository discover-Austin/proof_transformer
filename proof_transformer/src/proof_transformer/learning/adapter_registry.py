from __future__ import annotations

import sqlite3
from pathlib import Path

from ..ids import new_id
from ..timeutils import utc_now


class AdapterRegistry:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def register_adapter(
        self,
        *,
        name: str,
        version: str,
        path: str | Path,
        parent_version: str | None = None,
        dataset_hash: str | None = None,
        eval_report_hash: str | None = None,
        status: str = "pending",
    ) -> dict:
        adapter_path = Path(path)
        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)
        adapter_id = new_id("adapter")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO adapter_registry(
                    id, name, version, path, parent_version, dataset_hash,
                    eval_report_hash, status, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    adapter_id,
                    name,
                    version,
                    str(adapter_path),
                    parent_version,
                    dataset_hash,
                    eval_report_hash,
                    status,
                    utc_now(),
                ),
            )
        return self.get_adapter(name=name, version=version)

    def get_adapter(self, *, name: str, version: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM adapter_registry WHERE name = ? AND version = ? LIMIT 1",
            (name, version),
        ).fetchone()
        return dict(row) if row else None

    def list_adapters(self, name: str | None = None) -> list[dict]:
        if name:
            rows = self.conn.execute("SELECT * FROM adapter_registry WHERE name = ? ORDER BY created_at DESC", (name,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM adapter_registry ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]

    def mark_status(self, *, name: str, version: str, status: str) -> dict:
        with self.conn:
            self.conn.execute(
                "UPDATE adapter_registry SET status = ? WHERE name = ? AND version = ?",
                (status, name, version),
            )
        adapter = self.get_adapter(name=name, version=version)
        if not adapter:
            raise ValueError(f"Unknown adapter {name}:{version}")
        return adapter

    def accept_adapter(self, *, name: str, version: str, activate: bool = True) -> dict:
        adapter = self.mark_status(name=name, version=version, status="accepted")
        return self.activate_adapter(name=name, version=version) if activate else adapter

    def reject_adapter(self, *, name: str, version: str) -> dict:
        return self.mark_status(name=name, version=version, status="rejected")

    def activate_adapter(self, *, name: str, version: str) -> dict:
        adapter = self.get_adapter(name=name, version=version)
        if not adapter:
            raise ValueError(f"Unknown adapter {name}:{version}")
        if adapter["status"] not in {"accepted", "active"}:
            raise ValueError("Only accepted adapters can be activated")
        with self.conn:
            self.conn.execute(
                "UPDATE adapter_registry SET status = 'accepted' WHERE name = ? AND status = 'active'",
                (name,),
            )
            self.conn.execute(
                "UPDATE adapter_registry SET status = 'active' WHERE name = ? AND version = ?",
                (name, version),
            )
        return self.get_adapter(name=name, version=version)

    def get_active_adapter(self, name: str | None = None) -> dict | None:
        if name:
            row = self.conn.execute(
                "SELECT * FROM adapter_registry WHERE name = ? AND status = 'active' ORDER BY created_at DESC LIMIT 1",
                (name,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT * FROM adapter_registry WHERE status = 'active' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def rollback(self, *, name: str) -> dict | None:
        active = self.get_active_adapter(name=name)
        accepted = self.conn.execute(
            "SELECT * FROM adapter_registry WHERE name = ? AND status = 'accepted' ORDER BY created_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        if active:
            self.mark_status(name=name, version=active["version"], status="rolled_back")
        if not accepted:
            return None
        return self.activate_adapter(name=name, version=accepted["version"])
