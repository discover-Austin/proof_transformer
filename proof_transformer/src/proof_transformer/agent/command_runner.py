from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..config import load_settings
from ..ids import new_id
from ..memory.documents import DocumentStore
from ..paths import get_brain_paths
from ..timeutils import utc_now


class CommandRunner:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.settings = load_settings()
        self.paths = get_brain_paths()
        self.document_store = DocumentStore(conn)

    def run(
        self,
        command: list[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: int | None = None,
        ingest_output: bool = True,
    ) -> dict:
        command_list = command if isinstance(command, list) else [command]
        if not command_list:
            raise ValueError("Command cannot be empty")
        run_id = new_id("exec")
        started_at = utc_now()
        timeout_seconds = int(timeout or self.settings.command_timeout)
        run_dir = self.paths.logs / "executions" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        combined_path = run_dir / "combined.txt"
        cwd_path = Path(cwd) if cwd else None
        if cwd_path and not cwd_path.exists():
            raise FileNotFoundError(cwd_path)

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO execution_runs(
                    id, command_json, cwd, status, exit_code, stdout_path, stderr_path,
                    output_document_id, started_at, finished_at, metadata_json
                ) VALUES(?, ?, ?, 'running', NULL, ?, ?, NULL, ?, NULL, ?)
                """,
                (
                    run_id,
                    json.dumps(command_list),
                    str(cwd_path) if cwd_path else None,
                    str(stdout_path),
                    str(stderr_path),
                    started_at,
                    json.dumps({"timeout_seconds": timeout_seconds}),
                ),
            )

        exit_code: int | None = None
        status = "completed"
        stdout_text = ""
        stderr_text = ""
        try:
            completed = subprocess.run(
                command_list,
                cwd=str(cwd_path) if cwd_path else None,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
            exit_code = int(completed.returncode)
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            if exit_code != 0:
                status = "failed"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            exit_code = None
            stdout_text = exc.stdout or ""
            stderr_text = exc.stderr or ""
            if stderr_text:
                stderr_text += "\n"
            stderr_text += f"Command timed out after {timeout_seconds} seconds."
        except Exception as exc:
            status = "error"
            exit_code = None
            stderr_text = str(exc)

        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        combined_text = self._build_combined_text(command_list, cwd_path, status, exit_code, stdout_text, stderr_text)
        combined_path.write_text(combined_text, encoding="utf-8")

        output_document_id = None
        if ingest_output:
            document = self.document_store.ingest_text(
                title=f"execution_{run_id}",
                text=combined_text,
                path=str(combined_path),
                source_uri=str(combined_path),
                metadata={
                    "execution_id": run_id,
                    "command": command_list,
                    "cwd": str(cwd_path) if cwd_path else None,
                    "status": status,
                    "exit_code": exit_code,
                },
            )
            output_document_id = document["id"]

        finished_at = utc_now()
        with self.conn:
            self.conn.execute(
                """
                UPDATE execution_runs
                SET status = ?, exit_code = ?, output_document_id = ?, finished_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    status,
                    exit_code,
                    output_document_id,
                    finished_at,
                    json.dumps({"timeout_seconds": timeout_seconds, "ingest_output": ingest_output}),
                    run_id,
                ),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM execution_runs WHERE id = ?", (run_id,)).fetchone()
        return self._decode_row(row) if row else None

    def list_runs(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM execution_runs ORDER BY started_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [self._decode_row(row) for row in rows]

    def _decode_row(self, row) -> dict:
        payload = dict(row)
        payload["command"] = json.loads(payload.pop("command_json"))
        payload["metadata"] = json.loads(payload.pop("metadata_json")) if payload.get("metadata_json") else {}
        return payload

    def _build_combined_text(
        self,
        command_list: list[str],
        cwd_path: Path | None,
        status: str,
        exit_code: int | None,
        stdout_text: str,
        stderr_text: str,
    ) -> str:
        lines = [
            f"Command: {' '.join(command_list)}",
            f"Working directory: {cwd_path if cwd_path else ''}",
            f"Status: {status}",
            f"Exit code: {exit_code}",
            "",
            "STDOUT:",
            stdout_text,
            "",
            "STDERR:",
            stderr_text,
        ]
        return "\n".join(lines).strip() + "\n"
