from __future__ import annotations

import json
import sqlite3

from ..ids import new_id
from ..timeutils import utc_now


TASK_TYPES = {
    "ingest",
    "index_workspace",
    "ask",
    "assist",
    "learn",
    "feedback",
    "plan_repair",
    "apply_repair_plan",
    "autonomous_repair",
    "run_command",
    "review_execution",
    "health",
    "verify_ledger",
    "export_audit",
    "status",
    "demo",
}


class TaskQueue:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def enqueue(self, task_type: str, payload: dict | None = None, priority: int = 100) -> dict:
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unsupported task type: {task_type}")
        task_id = new_id("task")
        now = utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO task_queue(
                    id, task_type, payload_json, status, priority,
                    result_json, error_text, created_at, started_at, finished_at
                ) VALUES(?, ?, ?, 'pending', ?, NULL, NULL, ?, NULL, NULL)
                """,
                (task_id, task_type, json.dumps(payload or {}), int(priority), now),
            )
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM task_queue WHERE id = ?", (task_id,)).fetchone()
        return self._decode_task(row) if row else None

    def list_tasks(self, status: str | None = None, limit: int = 50) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM task_queue WHERE status = ? ORDER BY priority ASC, created_at ASC LIMIT ?",
                (status, int(limit)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM task_queue ORDER BY priority ASC, created_at ASC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._decode_task(row) for row in rows]

    def run_next(self, orchestrator) -> dict | None:
        row = self.conn.execute(
            "SELECT id FROM task_queue WHERE status = 'pending' ORDER BY priority ASC, created_at ASC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return self.run_task(orchestrator, row["id"])

    def run_pending(self, orchestrator, limit: int | None = None) -> dict:
        results: list[dict] = []
        remaining = None if limit is None else max(int(limit), 0)
        while remaining is None or remaining > 0:
            result = self.run_next(orchestrator)
            if result is None:
                break
            results.append(result)
            if remaining is not None:
                remaining -= 1
        return {"ran": len(results), "tasks": results}

    def run_task(self, orchestrator, task_id: str) -> dict:
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Unknown task: {task_id}")
        if task["status"] not in {"pending", "failed"}:
            return task

        run_id = new_id("taskrun")
        started_at = utc_now()
        with self.conn:
            self.conn.execute(
                "UPDATE task_queue SET status = 'running', started_at = ?, error_text = NULL WHERE id = ?",
                (started_at, task_id),
            )
            self.conn.execute(
                "INSERT INTO task_runs(id, task_id, status, output_json, error_text, created_at, finished_at) VALUES(?, ?, 'running', NULL, NULL, ?, NULL)",
                (run_id, task_id, started_at),
            )
        try:
            result = self._dispatch(orchestrator, task["task_type"], task["payload"])
            serializable = result if isinstance(result, (dict, list, str, int, float, bool)) or result is None else {"value": str(result)}
            finished_at = utc_now()
            with self.conn:
                self.conn.execute(
                    "UPDATE task_queue SET status = 'completed', result_json = ?, finished_at = ? WHERE id = ?",
                    (json.dumps(serializable), finished_at, task_id),
                )
                self.conn.execute(
                    "UPDATE task_runs SET status = 'completed', output_json = ?, finished_at = ? WHERE id = ?",
                    (json.dumps(serializable), finished_at, run_id),
                )
        except Exception as exc:
            finished_at = utc_now()
            with self.conn:
                self.conn.execute(
                    "UPDATE task_queue SET status = 'failed', error_text = ?, finished_at = ? WHERE id = ?",
                    (str(exc), finished_at, task_id),
                )
                self.conn.execute(
                    "UPDATE task_runs SET status = 'failed', error_text = ?, finished_at = ? WHERE id = ?",
                    (str(exc), finished_at, run_id),
                )
        return self.get_task(task_id)

    def status_counts(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) AS count FROM task_queue GROUP BY status").fetchall()
        return {row["status"]: int(row["count"]) for row in rows}

    def _dispatch(self, orchestrator, task_type: str, payload: dict):
        if task_type == "ingest":
            return orchestrator.ingest(payload["path"])
        if task_type == "index_workspace":
            return orchestrator.index_workspace(
                payload["path"],
                max_file_bytes=int(payload.get("max_file_bytes", 1_000_000)),
                extensions=set(payload["extensions"]) if payload.get("extensions") else None,
            )
        if task_type == "ask":
            return orchestrator.ask(payload["query"])
        if task_type == "assist":
            return orchestrator.assist(
                payload["query"],
                workspace_path=payload.get("workspace_path"),
                index_workspace=bool(payload.get("index_workspace", False)),
            )
        if task_type == "learn":
            return orchestrator.learn(payload["source_text"], payload["event_type"], payload.get("context"))
        if task_type == "feedback":
            return orchestrator.feedback(
                payload["retrieval_event_id"],
                payload.get("useful_chunk_ids", []),
                payload.get("irrelevant_chunk_ids", []),
            )
        if task_type == "plan_repair":
            return orchestrator.plan_repair(
                payload["workspace_path"],
                payload["command"],
                execution_id=payload.get("execution_id"),
                timeout=payload.get("timeout"),
            )
        if task_type == "apply_repair_plan":
            return orchestrator.apply_repair_plan(
                payload["plan_id"],
                run_post_commands=bool(payload.get("run_post_commands", True)),
            )
        if task_type == "autonomous_repair":
            return orchestrator.autonomous_repair(
                payload["workspace_path"],
                payload["command"],
                max_iterations=int(payload.get("max_iterations", 1)),
                timeout=payload.get("timeout"),
            )
        if task_type == "run_command":
            return orchestrator.run_command(
                payload["command"],
                cwd=payload.get("cwd"),
                timeout=payload.get("timeout"),
                ingest_output=bool(payload.get("ingest_output", True)),
            )
        if task_type == "review_execution":
            return orchestrator.review_execution(payload["execution_id"], payload["question"])
        if task_type == "health":
            return orchestrator.health()
        if task_type == "verify_ledger":
            return orchestrator.verify_ledger()
        if task_type == "export_audit":
            return {"path": orchestrator.export_audit(payload["path"])}
        if task_type == "status":
            return orchestrator.status()
        if task_type == "demo":
            return orchestrator.demo()
        raise ValueError(f"Unsupported task type: {task_type}")

    def _decode_task(self, row: sqlite3.Row) -> dict:
        task = dict(row)
        task["payload"] = json.loads(task.pop("payload_json"))
        task["result"] = json.loads(task["result_json"]) if task.get("result_json") else None
        task.pop("result_json", None)
        return task
