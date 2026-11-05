from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.peds.actuator_providers import (  # noqa: E402
    detect_actuator_providers,
    run_actuator_task,
)
from proof_transformer.peds.brain import (  # noqa: E402
    DEFAULT_BRAIN_ROOT,
    claim_next_codex_task,
    derive_current_task_queue,
    ensure_codex_task_queued,
    load_brain_state,
    mark_task_running,
    record_codex_task_result,
)

DEFAULT_TIMEOUT_SECONDS = 1500


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as exc:  # noqa: BLE001 - CLI should return JSON errors, not tracebacks.
        result = {"ok": False, "error": str(exc)}
    indent = None if args.json else 2
    print(json.dumps(result, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0 if result.get("ok") else 1


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root) if args.root is not None else DEFAULT_BRAIN_ROOT
    project_root = Path(args.project_root) if args.project_root is not None else PROJECT_ROOT
    provider = str(args.provider or "auto")
    if args.providers:
        status = detect_actuator_providers()
        return {"ok": True, **status}
    if args.status:
        queue = derive_current_task_queue(root=root)
        providers = detect_actuator_providers()
        return {
            "ok": True,
            "brain_root": str(root),
            "state": load_brain_state(root),
            "queue": queue,
            "queue_by_state": split_queue_by_state(queue),
            "actuator_providers": providers,
            "codex_support": inspect_codex_support(),
        }
    if args.claim_next:
        task = claim_next_codex_task(root=root)
        return {"ok": True, "task": task, "queue": derive_current_task_queue(root=root)}
    if args.run_next:
        return run_next_task(root=root, project_root=project_root, timeout_seconds=args.timeout_seconds, provider=provider)
    if args.run_task is not None:
        return run_task_path(args.run_task, root=root, project_root=project_root, timeout_seconds=args.timeout_seconds, provider=provider)
    if args.record_result is not None:
        result = record_codex_task_result(args.record_result, root=root)
        return {"ok": True, "task_result": result, "queue": derive_current_task_queue(root=root)}
    return {"ok": False, "error": "one actuator action is required"}


def split_queue_by_state(queue: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    states = {
        "pending": [],
        "claimed": [],
        "running": [],
        "completed": [],
        "failed": [],
        "manual_actuator_required": [],
        "superseded": [],
    }
    for item in queue:
        status = str(item.get("status") or "")
        states.setdefault(status, []).append(item)
    return states


def run_next_task(
    *,
    root: Path | None = None,
    project_root: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    provider: str = "auto",
) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    task = claim_next_codex_task(root=brain_root)
    if task is None:
        claimed = [item for item in derive_current_task_queue(root=brain_root) if item.get("status") == "claimed" and item.get("task_path")]
        task = sorted(claimed, key=lambda row: str(row.get("timestamp") or ""))[0] if claimed else None
    if task is None:
        return {
            "ok": True,
            "ran": False,
            "status": "no_pending_task",
            "summary": "No pending or claimed PEDS actuator task is available.",
            "actuator_providers": detect_actuator_providers(),
            "queue": derive_current_task_queue(root=brain_root),
        }
    running = mark_task_running(task, root=brain_root) or task
    result_path, actuator_result = execute_task(running, root=brain_root, project_root=project_root, timeout_seconds=timeout_seconds, provider=provider)
    recorded = record_codex_task_result(result_path, root=brain_root)
    return {
        "ok": actuator_result.get("status") == "completed",
        "ran": True,
        "task": running,
        "result_path": str(result_path),
        "actuator_result": actuator_result,
        "recorded_task_result": recorded,
        "queue": derive_current_task_queue(root=brain_root),
    }


def run_task_path(
    task_path: str | Path,
    *,
    root: Path | None = None,
    project_root: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    provider: str = "auto",
) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    path = Path(task_path)
    if not path.exists():
        return {"ok": False, "error": f"task_path does not exist: {path}"}
    queued = ensure_codex_task_queued(task_path=path, action="manual_run_task", root=brain_root)
    running = mark_task_running(queued, root=brain_root) or queued
    result_path, actuator_result = execute_task(running, root=brain_root, project_root=project_root, timeout_seconds=timeout_seconds, provider=provider)
    recorded = record_codex_task_result(result_path, root=brain_root)
    return {
        "ok": actuator_result.get("status") == "completed",
        "ran": True,
        "task": running,
        "result_path": str(result_path),
        "actuator_result": actuator_result,
        "recorded_task_result": recorded,
        "queue": derive_current_task_queue(root=brain_root),
    }


def execute_task(
    task: dict[str, Any],
    *,
    root: Path | None = None,
    project_root: Path | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    provider: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    _ = project_root  # Provider runner owns provider subprocess execution; kept for API compatibility.
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    task_path = Path(str(task.get("task_path") or ""))
    task_id = str(task.get("task_id") or task.get("queue_id") or task_path.stem)
    run_dir = brain_root / "actuator_runs" / _run_id(task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_actuator_task(task_path=task_path, provider=provider, timeout_seconds=timeout_seconds, output_dir=run_dir)
    result.update(task_id=task_id, task_path=str(task_path))
    result_path = Path(str(result.get("result_path") or (run_dir / "actuator_result.json")))
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result_path, result


def inspect_codex_support() -> dict[str, Any]:
    providers = detect_actuator_providers()
    codex = next((item for item in providers.get("providers", []) if isinstance(item, dict) and item.get("name") == "codex"), None)
    if not isinstance(codex, dict):
        return {"found": False, "exec_supported": False, "reason": "codex_probe_missing"}
    return {
        "found": bool(codex.get("available")),
        "exec_supported": bool(codex.get("supports_noninteractive")),
        "reason": codex.get("reason"),
        "probe_commands": codex.get("probe_commands", []),
    }


def _run_help(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except FileNotFoundError:
        return {"found": False, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": "command_not_found"}
    except subprocess.TimeoutExpired:
        return {"found": True, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": "help_timeout"}
    except OSError as exc:
        return {"found": False, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": str(exc)}
    return {"found": True, "command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "reason": None if completed.returncode == 0 else "help_returncode_nonzero"}


def _git_status_lines(repo_root: Path) -> set[str]:
    try:
        completed = subprocess.run(["git", "status", "--short"], cwd=str(repo_root), text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if completed.returncode != 0:
        return set()
    return {line.rstrip() for line in completed.stdout.splitlines() if line.strip()}


def _changed_files(before: set[str], after: set[str]) -> list[str]:
    changed = after - before
    if not changed:
        return []
    files = []
    for line in sorted(changed):
        files.append(line[3:] if len(line) > 3 else line)
    return files


def _run_id(task_id: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in task_id).strip("_") or "task"
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{safe}"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local PEDS actuator for PEDS-decided tasks.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--providers", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--claim-next", action="store_true")
    actions.add_argument("--run-next", action="store_true")
    actions.add_argument("--run-task", type=Path)
    actions.add_argument("--record-result", type=Path)
    parser.add_argument("--provider", choices=("auto", "codex", "pi", "manual"), default="auto")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
