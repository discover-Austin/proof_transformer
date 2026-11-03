from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Import-safe: no Tk root is created at import time.
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - depends on local Python installation.
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
STATUS_DIR = PROJECT_ROOT / "temp" / "operator_status"
CONTINUATION_POLICIES = ("last-valid-parent", "fixed-baseline", "fresh-baseline", "stop")
DEFAULT_GENERATIONS = 5
DEFAULT_RUNS = 2
DEFAULT_MAX_ATTEMPTS_PER_GENERATION = 5
DEFAULT_OBJECTIVE = "Improve Codex patch reliability in this repo."
AUTOPILOT_OUTPUT_ROOT = PROJECT_ROOT / "temp" / "peds_autopilot_outputs"
PEDS_BRAIN_ROOT = PROJECT_ROOT / "temp" / "peds_brain"
BRAIN_ACTION_FLAGS = {
    "init": "--init",
    "status": "--status",
    "think": "--think",
    "step": "--step",
    "run": "--run",
    "claim-codex-task": "--claim-codex-task",
    "list-codex-queue": "--list-codex-queue",
}

LINEAGE_COLUMNS = (
    "generation",
    "parent_candidate_id",
    "source_candidate_id",
    "attempt_candidate_id",
    "winning_candidate_id",
    "generated_child_candidate_id",
    "selected_parent_for_next_generation",
    "parent_selection_reason",
)
ATTEMPT_COLUMNS = (
    "generation",
    "attempt_index",
    "outcome",
    "diagnosis_category",
    "correction",
    "mutation_target",
    "mutation_seed",
    "replay_mean_delta",
    "adapter_loaded",
    "training_trained",
)
MUTATION_COLUMNS = (
    "generation",
    "attempt_index",
    "diagnosis_category",
    "mutation_target",
    "mutation_seed",
    "delta_description",
)
SUMMARY_FIELDS = (
    "campaign_id",
    "campaign_root",
    "generations_requested",
    "total_generations_completed",
    "total_candidate_attempts",
    "total_winning_attempts",
    "total_rejected_attempts",
    "exhausted_generation_count",
    "corrections_attempted",
    "corrections_followed_by_winner",
    "correction_success_rate",
    "collapse_detected",
    "adapter_loaded_rate",
    "trained_rate",
)


def build_generation_command(generations: int, repo_root: Path | None = None) -> list[str]:
    """Compatibility builder for the original trained-local-generations demo."""
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    return [
        sys.executable,
        str(root / "examples" / "peds_trained_local_generations.py"),
        "--generations",
        str(generations),
    ]


def build_campaign_command(
    *,
    generations: int = DEFAULT_GENERATIONS,
    runs: int = DEFAULT_RUNS,
    max_attempts_per_generation: int = DEFAULT_MAX_ATTEMPTS_PER_GENERATION,
    continuation_policy: str = "last-valid-parent",
    strict: bool = False,
    repo_root: Path | None = None,
) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    command = [
        sys.executable,
        str(root / "scripts" / "peds_training_campaign.py"),
        "--generations",
        str(generations),
        "--runs",
        str(runs),
        "--max-attempts-per-generation",
        str(max_attempts_per_generation),
        "--continuation-policy",
        continuation_policy,
        "--json",
    ]
    if strict:
        command.append("--strict")
    return command


def build_autopilot_command(
    *,
    objective: str,
    campaign: str = "5x2",
    use_latest: bool = False,
    activate_best: bool = False,
    write: bool = True,
    artifact_type: str = "all",
    repo_root: Path | None = None,
) -> list[str]:
    objective_text = str(objective or "").strip()
    if not objective_text:
        raise ValueError("objective must not be blank")
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    command = [
        sys.executable,
        str(root / "scripts" / "peds_autopilot.py"),
        "--objective",
        objective_text,
        "--campaign",
        campaign,
        "--artifact-type",
        artifact_type,
        "--json",
    ]
    if use_latest:
        command.append("--use-latest")
    if activate_best:
        command.append("--activate-best")
    if write:
        command.append("--write")
    return command


def build_watch_autopilot_command(
    *,
    objective: str,
    campaign: str = "5x2",
    use_latest: bool = False,
    activate_best: bool = False,
    write: bool = True,
    artifact_type: str = "all",
    status_dir: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    command = build_autopilot_command(
        objective=objective,
        campaign=campaign,
        use_latest=use_latest,
        activate_best=activate_best,
        write=write,
        artifact_type=artifact_type,
        repo_root=repo_root,
    )
    return build_watch_command(name="peds_autopilot", command=command, status_dir=status_dir, repo_root=repo_root)


def build_brain_command(
    action: str,
    *,
    objective: str | None = None,
    max_iterations: int | None = None,
    execute_actuator: bool = False,
    repo_root: Path | None = None,
) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    if action not in BRAIN_ACTION_FLAGS:
        raise ValueError(f"Unsupported brain action: {action!r}")
    command = [sys.executable, str(root / "scripts" / "peds_brain.py"), BRAIN_ACTION_FLAGS[action], "--json"]
    if objective is not None:
        objective_text = str(objective or "").strip()
        if not objective_text:
            raise ValueError("objective must not be blank")
        command.extend(["--objective", objective_text])
    if max_iterations is not None:
        command.extend(["--max-iterations", str(int(max_iterations))])
    if execute_actuator:
        command.append("--execute-actuator")
    return command


def build_actuator_command(
    action: str,
    *,
    task_path: str | Path | None = None,
    result_path: str | Path | None = None,
    provider: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    flags = {
        "providers": "--providers",
        "status": "--status",
        "claim-next": "--claim-next",
        "run-next": "--run-next",
        "run-task": "--run-task",
        "record-result": "--record-result",
    }
    if action not in flags:
        raise ValueError(f"Unsupported actuator action: {action!r}")
    command = [sys.executable, str(root / "scripts" / "peds_actuator.py"), flags[action]]
    if action == "run-task":
        if task_path is None:
            raise ValueError("task_path is required for run-task")
        command.append(str(task_path))
    if action == "record-result":
        if result_path is None:
            raise ValueError("result_path is required for record-result")
        command.append(str(result_path))
    if provider is not None:
        command.extend(["--provider", str(provider)])
    command.append("--json")
    return command


def build_brain_record_task_result_command(
    result_path: str | Path,
    *,
    repo_root: Path | None = None,
) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    path = Path(result_path)
    if not str(path).strip():
        raise ValueError("result_path must not be blank")
    return [
        sys.executable,
        str(root / "scripts" / "peds_brain.py"),
        "--record-task-result",
        str(path),
        "--json",
    ]


def build_watch_brain_command(
    action: str,
    *,
    objective: str | None = None,
    max_iterations: int | None = None,
    execute_actuator: bool = False,
    status_dir: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    command = build_brain_command(action, objective=objective, max_iterations=max_iterations, execute_actuator=execute_actuator, repo_root=repo_root)
    return build_watch_command(name=f"peds_brain_{action}", command=command, status_dir=status_dir, repo_root=repo_root)


def build_watch_actuator_command(
    action: str,
    *,
    task_path: str | Path | None = None,
    result_path: str | Path | None = None,
    status_dir: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    command = build_actuator_command(action, task_path=task_path, result_path=result_path, repo_root=repo_root)
    return build_watch_command(name=f"peds_actuator_{action}", command=command, status_dir=status_dir, repo_root=repo_root)


def build_validation_command(kind: str, repo_root: Path | None = None) -> list[str]:
    _ = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    patterns = {
        "mutation": "test_peds_mutation_strategies.py",
        "campaign": "test_peds_training_campaign.py",
        "operator": "test_peds_operator_tools.py",
        "broad_peds": "test_peds_*.py",
    }
    if kind not in patterns:
        raise ValueError(f"Unsupported validation kind: {kind!r}")
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", patterns[kind]]


def build_broad_validation_command(repo_root: Path | None = None) -> list[str]:
    return build_validation_command("broad_peds", repo_root=repo_root)


def build_latest_report_command(repo_root: Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    return [sys.executable, str(root / "scripts" / "peds_latest_report.py"), "--json"]


def build_latest_campaign_report_command(repo_root: Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    return [sys.executable, str(root / "scripts" / "peds_latest_campaign_report.py"), "--json"]


def build_watch_command(
    *,
    name: str,
    command: list[str],
    status_dir: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    output_dir = Path(status_dir) if status_dir is not None else STATUS_DIR
    return [
        sys.executable,
        str(root / "scripts" / "watch_command.py"),
        "--name",
        name,
        "--cwd",
        str(root),
        "--status-dir",
        str(output_dir),
        "--",
        *command,
    ]


def build_watcher_command(
    *,
    name: str,
    command: list[str],
    repo_root: Path | None = None,
    status_dir: Path | None = None,
) -> list[str]:
    """Compatibility alias for older operator tests/tools."""
    return build_watch_command(
        name=name,
        command=command,
        status_dir=str(status_dir) if status_dir is not None else None,
        repo_root=repo_root,
    )


def load_latest_report(repo_root: Path | None = None) -> dict[str, Any]:
    """Compatibility helper for the older generation-only dashboard pane."""
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    result = subprocess.run(
        build_latest_report_command(root),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"found": False, "passed": False, "blocker": result.stderr.strip() or "report command failed"}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"found": False, "passed": False, "blocker": "latest report command returned invalid JSON"}


def load_latest_campaign_report(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root)
    result = subprocess.run(
        build_latest_campaign_report_command(root),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return {"found": False, "reason": result.stderr.strip() or "latest campaign report command failed"}
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"found": False, "reason": "latest campaign report command returned invalid JSON"}
    return decoded if isinstance(decoded, dict) else {"found": False, "reason": "latest campaign report was not a JSON object"}


def load_latest_watcher_status(status_dir: Path | None = None) -> dict[str, Any] | None:
    output_dir = Path(status_dir) if status_dir is not None else STATUS_DIR
    if not output_dir.exists():
        return None
    candidates = list(output_dir.glob("*.status.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    try:
        status = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "path": str(latest), "blocker": "invalid status JSON"}
    if isinstance(status, dict):
        status["path"] = str(latest)
        status.setdefault("stdout_log_path", str(latest.with_name(latest.name.replace(".status.json", ".stdout.log"))))
        status.setdefault("stderr_log_path", str(latest.with_name(latest.name.replace(".status.json", ".stderr.log"))))
        return status
    return {"status": "failed", "path": str(latest), "blocker": "status JSON was not an object"}


def load_latest_autopilot_summary(output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root) if output_root is not None else AUTOPILOT_OUTPUT_ROOT
    if not root.exists():
        return {"ok": False, "found": False, "reason": f"No autopilot outputs found under {root}"}
    candidates = list(root.glob("*/autopilot_summary.json"))
    if not candidates:
        return {"ok": False, "found": False, "reason": f"No autopilot_summary.json found under {root}"}
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.parent.name))
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ok": False, "found": False, "reason": f"Invalid autopilot summary JSON: {latest}", "summary_path": str(latest)}
    summary = parse_autopilot_summary(payload if isinstance(payload, dict) else {})
    summary["summary_path"] = str(latest)
    return summary


def parse_autopilot_summary(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return {"ok": False, "found": False, "reason": "invalid autopilot JSON"}
        payload = decoded if isinstance(decoded, dict) else {"ok": False, "found": False, "reason": "autopilot JSON was not an object"}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    coverage = payload.get("agentic_capability_coverage") if isinstance(payload.get("agentic_capability_coverage"), dict) else {}
    return {
        "ok": bool(payload.get("ok")),
        "found": bool(payload.get("ok") or payload.get("candidate_id") or payload.get("output_dir")),
        "objective": payload.get("objective"),
        "campaign": payload.get("campaign"),
        "campaign_ran": bool(payload.get("campaign_ran", False)),
        "survivor_selected": bool(payload.get("survivor_selected", False)),
        "candidate_id": payload.get("candidate_id"),
        "useful_output_ready": bool(payload.get("useful_output_ready", False)),
        "artifacts_written": bool(payload.get("artifacts_written", bool(artifacts))),
        "artifacts": artifacts,
        "output_dir": payload.get("output_dir"),
        "active_policy": payload.get("active_policy"),
        "latest_active_policy": payload.get("latest_active_policy"),
        "safety": payload.get("safety") if isinstance(payload.get("safety"), dict) else {},
        "what_improved": payload.get("what_improved") if isinstance(payload.get("what_improved"), list) else [],
        "recommended_uses": payload.get("recommended_uses") if isinstance(payload.get("recommended_uses"), list) else [],
        "agentic_capability_coverage": coverage,
        "agentic_next_eval_tasks": payload.get("agentic_next_eval_tasks") if isinstance(payload.get("agentic_next_eval_tasks"), list) else [],
        "codex_prompt_preview": payload.get("codex_prompt_preview") or "",
        "error": payload.get("error"),
        "reason": payload.get("reason"),
        "summary_path": payload.get("summary_path"),
    }


def summarize_agentic_capability_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    coverage = summary.get("agentic_capability_coverage") if isinstance(summary.get("agentic_capability_coverage"), dict) else {}
    if not coverage:
        return {
            "found": False,
            "status": "not_reported",
            "pressured_capabilities": [],
            "untested_capabilities": [],
            "next_eval_tasks": [],
            "coverage_warning": None,
        }
    pressured = coverage.get("pressured_capabilities") if isinstance(coverage.get("pressured_capabilities"), list) else []
    untested = coverage.get("untested_capabilities") if isinstance(coverage.get("untested_capabilities"), list) else []
    next_tasks = coverage.get("next_eval_tasks") if isinstance(coverage.get("next_eval_tasks"), list) else []
    return {
        "found": True,
        "status": coverage.get("status") or ("broadening_needed" if untested else "broad_curriculum_pressured"),
        "pressured_capabilities": [str(item) for item in pressured],
        "untested_capabilities": [str(item) for item in untested],
        "next_eval_tasks": [str(item) for item in next_tasks],
        "coverage_warning": coverage.get("coverage_warning"),
    }


def evaluate_ui_browser_operation_capability(
    brain_status: dict[str, Any],
    autopilot_summary: dict[str, Any],
) -> dict[str, Any]:
    queue_rows = brain_status.get("codex_queue") if isinstance(brain_status.get("codex_queue"), list) else []
    next_action = brain_status.get("next_action") if isinstance(brain_status.get("next_action"), dict) else {}
    task_result = (
        brain_status.get("last_codex_task_result")
        if isinstance(brain_status.get("last_codex_task_result"), dict)
        else {}
    )
    coverage = summarize_agentic_capability_coverage(parse_autopilot_summary(autopilot_summary))
    terminal_statuses = {"completed", "failed", "manual_actuator_required", "superseded"}
    checks = [
        {
            "name": "brain_status_visible",
            "passed": bool(brain_status.get("found") and brain_status.get("system") == "PEDS" and brain_status.get("status")),
        },
        {
            "name": "brain_decision_visible",
            "passed": bool(next_action.get("action") or brain_status.get("last_decision_id")),
            "source": "brain_state.next_action" if next_action.get("action") else "brain_state.last_decision_id",
        },
        {
            "name": "task_queue_visible",
            "passed": isinstance(queue_rows, list),
            "queue_count": len(queue_rows),
        },
        {
            "name": "task_result_visible",
            "passed": bool(task_result or any(row.get("status") in terminal_statuses for row in queue_rows if isinstance(row, dict))),
        },
        {
            "name": "agentic_gap_visible",
            "passed": bool(coverage.get("found") and "ui_or_browser_operation" in coverage.get("untested_capabilities", [])),
        },
        {
            "name": "dashboard_does_not_own_decision",
            "passed": "dashboard_decision" not in brain_status,
            "decision_source": "PEDS brain",
        },
    ]
    passed = all(bool(check.get("passed")) for check in checks)
    return {
        "capability": "ui_or_browser_operation",
        "status": "passed" if passed else "needs_evidence",
        "passed": passed,
        "checks": checks,
        "mirrors_brain_state": bool(checks[0]["passed"] and checks[1]["passed"]),
        "decision_owner": "PEDS brain",
        "coverage_status": coverage.get("status"),
        "untested_capabilities": coverage.get("untested_capabilities", []),
        "next_eval_tasks": coverage.get("next_eval_tasks", []),
    }


def prompt_preview_from_summary(summary: dict[str, Any]) -> str:
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    prompt_path = artifacts.get("codex_prompt")
    if prompt_path and Path(str(prompt_path)).exists():
        return Path(str(prompt_path)).read_text(encoding="utf-8", errors="replace")
    preview = summary.get("codex_prompt_preview")
    if preview:
        return str(preview)
    if summary.get("error"):
        return f"Autopilot did not produce a prompt: {summary.get('error')}"
    return "No Codex prompt has been generated yet."


def load_actuator_provider_status(repo_root: Path | None = None) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else PROJECT_ROOT
    result = subprocess.run(
        build_actuator_command("providers", repo_root=root),
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    try:
        decoded = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "selected_provider": "manual", "providers": [], "reason": "provider command returned invalid JSON"}
    if isinstance(decoded, dict):
        decoded.setdefault("ok", result.returncode == 0)
        return decoded
    return {"ok": False, "selected_provider": "manual", "providers": [], "reason": "provider command did not return a JSON object"}


def summarize_actuator_provider_status(status: dict[str, Any]) -> dict[str, Any]:
    providers = status.get("providers") if isinstance(status.get("providers"), list) else []
    noninteractive = [item for item in providers if isinstance(item, dict) and item.get("available") and item.get("supports_noninteractive")]
    return {
        "selected_provider": status.get("selected_provider") or "manual",
        "noninteractive_available": bool(noninteractive),
        "available_provider_names": [str(item.get("name")) for item in providers if isinstance(item, dict) and item.get("available")],
    }


def load_brain_status(brain_root: Path | None = None) -> dict[str, Any]:
    root = Path(brain_root) if brain_root is not None else PEDS_BRAIN_ROOT
    path = root / "brain_state.json"
    if not path.exists():
        return {"found": False, "reason": f"No PEDS brain state found under {root}"}
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"found": False, "reason": f"Invalid PEDS brain state JSON: {path}"}
    if not isinstance(state, dict):
        return {"found": False, "reason": f"PEDS brain state was not a JSON object: {path}"}
    state["found"] = True
    state["state_path"] = str(path)
    state["codex_queue"] = load_codex_queue_status(root)
    state["actuator_providers"] = load_actuator_provider_status()
    return state


def load_codex_queue_status(brain_root: Path | None = None) -> list[dict[str, Any]]:
    root = Path(brain_root) if brain_root is not None else PEDS_BRAIN_ROOT
    path = root / "task_queue.jsonl"
    if not path.exists():
        path = root / "codex_actuator_queue.jsonl"
    if not path.exists():
        return []
    latest_by_id: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and (decoded.get("task_id") or decoded.get("queue_id")):
            key = str(decoded.get("task_id") or decoded.get("queue_id"))
            row = dict(decoded)
            row.setdefault("task_id", key)
            row.setdefault("queue_id", key)
            latest_by_id[key] = row
    latest_by_path: dict[str, dict[str, Any]] = {}
    for row in latest_by_id.values():
        key = str(row.get("task_path") or row.get("task_id") or row.get("queue_id"))
        existing = latest_by_path.get(key)
        if existing is None or _prefer_queue_row(row, existing):
            latest_by_path[key] = row
    return sorted(latest_by_path.values(), key=lambda row: str(row.get("timestamp") or ""))


def _prefer_queue_row(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    terminal = {"completed", "failed", "manual_actuator_required", "superseded"}
    candidate_terminal = candidate.get("status") in terminal
    incumbent_terminal = incumbent.get("status") in terminal
    if candidate_terminal != incumbent_terminal:
        return candidate_terminal
    return str(candidate.get("timestamp") or "") >= str(incumbent.get("timestamp") or "")


def split_task_queue_by_status(queue: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "pending": [],
        "claimed": [],
        "running": [],
        "completed": [],
        "failed": [],
        "manual_actuator_required": [],
        "superseded": [],
    }
    for item in queue:
        groups.setdefault(str(item.get("status") or "unknown"), []).append(item)
    return groups


def latest_manual_actuator_command(state: dict[str, Any]) -> str | None:
    result = state.get("last_codex_task_result") if isinstance(state.get("last_codex_task_result"), dict) else {}
    evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    command = evidence.get("manual_command") or result.get("manual_command") or evidence.get("next_recommendation")
    return str(command) if command else None


def load_latest_brain_decision(brain_root: Path | None = None) -> dict[str, Any] | None:
    root = Path(brain_root) if brain_root is not None else PEDS_BRAIN_ROOT
    path = root / "decisions.jsonl"
    if not path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            latest = decoded
    return latest


def latest_brain_task_path(brain_root: Path | None = None) -> Path:
    root = Path(brain_root) if brain_root is not None else PEDS_BRAIN_ROOT
    task_dir = root / "tasks"
    candidates = list(task_dir.glob("*_codex_task.md")) if task_dir.exists() else []
    if candidates:
        return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))
    return task_dir / "_codex_task.md"


def parse_latest_campaign_summary(payload: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError:
            return summarize_campaign_report({"found": False, "reason": "invalid JSON"})
        payload = decoded if isinstance(decoded, dict) else {"found": False, "reason": "JSON payload was not an object"}
    return summarize_campaign_report(payload)


def summarize_campaign_report(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
    winning_attempts = report.get("winning_attempts") if isinstance(report.get("winning_attempts"), list) else []
    rejected_attempts = report.get("rejected_attempts") if isinstance(report.get("rejected_attempts"), list) else []
    corrections = report.get("corrections") if isinstance(report.get("corrections"), list) else []
    exhausted_generations = report.get("exhausted_generations") if isinstance(report.get("exhausted_generations"), list) else []
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}

    summary = {
        "found": bool(report.get("found", bool(report.get("campaign_id") or report.get("campaign_root") or aggregate))),
        "campaign_id": report.get("campaign_id"),
        "campaign_root": report.get("campaign_root"),
        "report_path": report.get("report_path"),
        "generations_requested": report.get("generations_requested"),
        "total_generations_completed": _first_present(aggregate, "total_generations_completed", default=0),
        "total_candidate_attempts": _first_present(aggregate, "total_candidate_attempts", default=len(attempts)),
        "total_winning_attempts": _first_present(aggregate, "total_winning_attempts", default=len(winning_attempts)),
        "total_rejected_attempts": _first_present(aggregate, "total_rejected_attempts", default=len(rejected_attempts)),
        "exhausted_generation_count": _first_present(aggregate, "exhausted_generation_count", default=len(exhausted_generations)),
        "corrections_attempted": _first_present(aggregate, "corrections_attempted", default=len(corrections)),
        "corrections_followed_by_winner": _first_present(aggregate, "corrections_followed_by_winner", default=0),
        "correction_success_rate": _first_present(aggregate, "correction_success_rate", default=0.0),
        "collapse_detected": bool(aggregate.get("collapse_detected", False)),
        "adapter_loaded_rate": _first_present(aggregate, "adapter_loaded_rate", default=0.0),
        "trained_rate": _first_present(aggregate, "trained_rate", default=0.0),
        "network_attempts": list(safety.get("network_attempts") or []),
        "model_modules_loaded": list(safety.get("model_modules_loaded") or []),
        "fake_weights_written": bool(safety.get("fake_weights_written", False)),
        "safety_clean": _safety_clean(safety),
        "reason": report.get("reason") or report.get("blocker"),
    }
    return summary


def extract_lineage_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    lineage = report.get("lineage") if isinstance(report.get("lineage"), list) else []
    if not lineage and isinstance(report.get("winning_attempts"), list):
        lineage = report["winning_attempts"]
    rows: list[dict[str, Any]] = []
    for item in lineage:
        if not isinstance(item, dict):
            continue
        rows.append({key: item.get(key) for key in LINEAGE_COLUMNS})
    return rows


def extract_attempt_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = report.get("attempts") if isinstance(report.get("attempts"), list) else []
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        metric = attempt.get("metric") if isinstance(attempt.get("metric"), dict) else {}
        mutation = attempt.get("mutation") if isinstance(attempt.get("mutation"), dict) else {}
        rows.append(
            {
                "generation": attempt.get("generation"),
                "attempt_index": attempt.get("attempt_index"),
                "outcome": attempt.get("outcome"),
                "diagnosis_category": _diagnosis_category(attempt),
                "correction": _correction_name(attempt.get("correction")),
                "mutation_target": attempt.get("mutation_target") or mutation.get("mutation_target"),
                "mutation_seed": attempt.get("mutation_seed") or mutation.get("mutation_seed"),
                "replay_mean_delta": metric.get("replay_mean_delta") if metric else attempt.get("replay_mean_delta"),
                "adapter_loaded": metric.get("adapter_loaded") if metric else attempt.get("adapter_loaded"),
                "training_trained": metric.get("training_trained") if metric else attempt.get("training_trained"),
            }
        )
    return rows


def extract_mutation_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = report.get("mutations") if isinstance(report.get("mutations"), list) else []
    if not source_rows:
        source_rows = [attempt for attempt in report.get("attempts", []) if isinstance(attempt, dict) and attempt.get("mutation")]
    rows: list[dict[str, Any]] = []
    for item in source_rows:
        if not isinstance(item, dict):
            continue
        mutation = item.get("mutation") if isinstance(item.get("mutation"), dict) else item
        rows.append(
            {
                "generation": item.get("generation"),
                "attempt_index": item.get("attempt_index"),
                "diagnosis_category": item.get("diagnosis_category") or mutation.get("diagnosis_category") or _diagnosis_category(item),
                "mutation_target": item.get("mutation_target") or mutation.get("mutation_target"),
                "mutation_seed": item.get("mutation_seed") or mutation.get("mutation_seed"),
                "delta_description": item.get("delta_description") or mutation.get("delta_description"),
            }
        )
    return rows


def build_evolution_story(report: dict[str, Any]) -> list[str]:
    attempts = _attempts_with_run_index(report)
    if not attempts:
        lineage_rows = extract_lineage_rows(report)
        if not lineage_rows:
            return ["No campaign story is available yet. Run a learning campaign to create one."]
        lines = ["Evolution Chain"]
        for row in lineage_rows:
            generation = row.get("generation") or "?"
            winner = _short_id(row.get("winning_candidate_id"))
            next_parent = _short_id(row.get("selected_parent_for_next_generation"))
            if winner:
                lines.append(f"G{generation} ✅ winner → next parent {next_parent or winner}")
            else:
                lines.append(f"G{generation} ❌ no winner recorded")
        return lines

    lines: list[str] = []
    current_run: Any = object()
    rejected_generations = {
        (attempt.get("run_index"), attempt.get("generation"))
        for attempt in attempts
        if attempt.get("outcome") == "rejected"
    }
    for attempt in attempts:
        run_index = attempt.get("run_index") or 1
        if run_index != current_run:
            if lines:
                lines.append("")
            lines.append(f"Run {run_index}")
            current_run = run_index
        generation = attempt.get("generation") or "?"
        attempt_index = attempt.get("attempt_index") or "?"
        outcome = str(attempt.get("outcome") or "unknown")
        diagnosis = _diagnosis_category(attempt)
        mutation_target = _attempt_mutation_target(attempt)
        next_parent = _short_id(attempt.get("selected_parent_for_next_generation") or attempt.get("winning_candidate_id"))
        correction = _correction_name(attempt.get("correction"))
        corrected = outcome == "winner" and (
            _coerce_int(attempt.get("attempt_index"), default=1) > 1
            or bool(correction)
            or (attempt.get("run_index"), attempt.get("generation")) in rejected_generations
        )
        if outcome == "winner":
            label = "✅ corrected winner" if corrected else "✅ winner"
            line = f"G{generation} A{attempt_index} {label} → next parent {next_parent}"
            if mutation_target:
                line += f" — 🧬 mutation {mutation_target}"
        elif outcome == "rejected":
            line = f"G{generation} A{attempt_index} ❌ rejected"
            if diagnosis:
                line += f" — diagnosis {diagnosis}"
            if mutation_target:
                line += f" — 🧬 mutation {mutation_target}"
        else:
            line = f"G{generation} A{attempt_index} [{outcome}]"
            if diagnosis:
                line += f" — diagnosis {diagnosis}"
            if mutation_target:
                line += f" — 🧬 mutation {mutation_target}"
        lines.append(line)
    return lines


def summarize_peds_learning(report: dict[str, Any]) -> str:
    summary = summarize_campaign_report(report)
    if not summary.get("found"):
        return "No campaign has been loaded yet. Run a 5x2 adaptive campaign to let PEDS attempt, reject, correct, and advance a lineage."

    completed = summary.get("total_generations_completed", 0)
    attempts = summary.get("total_candidate_attempts", 0)
    rejected = summary.get("total_rejected_attempts", 0)
    corrections = summary.get("corrections_attempted", 0)
    corrected_winners = summary.get("corrections_followed_by_winner", 0)
    dominant_diagnosis = _dominant_diagnosis(report) or "none recorded"
    dominant_mutation = _dominant_mutation_label(report) or "none recorded"
    lineage_advanced = _lineage_advanced(report)
    safety_clean = bool(summary.get("safety_clean"))
    families = _mutation_families(report)
    diversity_text = (
        "Mutation diversity is still narrow, so the next useful run should stress mutation diversity."
        if len(families) <= 1 and families
        else "Mutation diversity is broad enough to compare multiple correction families."
        if families
        else "No mutation family has been recorded yet."
    )
    lineage_text = "Lineage advanced after each available winner." if lineage_advanced else "Lineage advancement needs inspection."
    safety_text = "Safety remained clean." if safety_clean else "Safety was not clean; inspect safety before scaling."
    return (
        f"PEDS completed {completed} generations across {attempts} attempts. "
        f"{rejected} attempts were rejected, and {corrected_winners} of {corrections} corrections were followed by winners. "
        f"The dominant diagnosis was {dominant_diagnosis}, and the dominant mutation was {dominant_mutation}. "
        f"{lineage_text} {safety_text} {diversity_text}"
    )


def recommend_next_action(report: dict[str, Any], validation_state: dict | None = None) -> dict[str, Any]:
    summary = summarize_campaign_report(report)
    if not summary.get("found"):
        command = build_campaign_command(generations=5, runs=2, max_attempts_per_generation=5)
        return {
            "label": "Run Learning Campaign 5x2",
            "reason": "No campaign report is loaded yet; start with the standard adaptive run.",
            "command_kind": "campaign_5x2",
            "command": command,
        }
    if not summary.get("safety_clean"):
        return {
            "label": "Inspect safety report",
            "reason": "Safety fields are not clean; do not scale the campaign until network/model/fake-weight contamination is understood.",
            "command_kind": "inspect_safety",
            "command": build_latest_campaign_report_command(),
        }
    if summary.get("collapse_detected") or summary.get("exhausted_generation_count", 0):
        return {
            "label": "Inspect exhausted/rejected generations",
            "reason": "The latest campaign reported collapse or exhausted generations; inspect evidence before launching a larger run.",
            "command_kind": "inspect_campaign",
            "command": build_latest_campaign_report_command(),
        }
    if validation_state is not None and validation_state.get("broad_peds") not in (True, "passed"):
        return {
            "label": "Validate PEDS Core",
            "reason": "Broad PEDS validation is not known to have passed in this session.",
            "command_kind": "validate_broad_peds",
            "command": build_validation_command("broad_peds"),
        }
    correction_success_rate = float(summary.get("correction_success_rate") or 0.0)
    rejected = int(summary.get("total_rejected_attempts") or 0)
    families = _mutation_families(report)
    if correction_success_rate >= 0.8 and rejected > 0:
        if len(families) <= 1 and families:
            return {
                "label": "Run Mutation Diversity Stress 10x3",
                "reason": "Corrections worked, but the successful mutation family is narrow; stress mutation diversity at larger scale.",
                "command_kind": "mutation_stress_campaign",
                "command": build_campaign_command(generations=10, runs=3, max_attempts_per_generation=5),
            }
        return {
            "label": "Run Bigger Campaign 10x3",
            "reason": "Corrections are working and safety is clean; scale the adaptive campaign.",
            "command_kind": "campaign_10x3",
            "command": build_campaign_command(generations=10, runs=3, max_attempts_per_generation=5),
        }
    if rejected > 0 and correction_success_rate < 0.8:
        return {
            "label": "Inspect mutations and corrections",
            "reason": "Rejected attempts are present, but corrections are not reliably producing winners.",
            "command_kind": "inspect_mutations",
            "command": build_latest_campaign_report_command(),
        }
    return {
        "label": "Validate PEDS Core",
        "reason": "No urgent campaign action is indicated; validate the core before the next experiment.",
        "command_kind": "validate_broad_peds",
        "command": build_validation_command("broad_peds"),
    }


def build_brain_status(report: dict[str, Any], *, current_mode: str = "Idle") -> dict[str, str]:
    summary = summarize_campaign_report(report)
    recommendation = recommend_next_action(report)
    survivor = _current_survivor(report)
    if not summary.get("found"):
        last_campaign = "No campaign loaded"
    elif summary.get("collapse_detected") or summary.get("exhausted_generation_count", 0):
        last_campaign = "Needs inspection"
    else:
        last_campaign = "Successful adaptive run"
    safety_status = "✅ Clean" if summary.get("safety_clean") else "❌ Safety issue"
    corrections = f"{summary.get('corrections_followed_by_winner', 0)} / {summary.get('corrections_attempted', 0)} worked"
    return {
        "Current Mode": current_mode,
        "Last Campaign Result": last_campaign,
        "Generations Completed": f"{summary.get('total_generations_completed', 0)} completed",
        "Total Attempts": f"{summary.get('total_candidate_attempts', 0)} total",
        "Winners": str(summary.get("total_winning_attempts", 0)),
        "Rejected Attempts": str(summary.get("total_rejected_attempts", 0)),
        "Corrections Worked": corrections,
        "Correction Success Rate": str(summary.get("correction_success_rate", 0.0)),
        "Current Survivor": _short_id(survivor) or "none",
        "Dominant Diagnosis": _dominant_diagnosis(report) or "none recorded",
        "Dominant Mutation": _dominant_mutation_label(report) or "none recorded",
        "Safety Status": safety_status,
        "Recommended Next Action": recommendation["label"],
    }


def open_campaign_folder_from_report(report: dict[str, Any]) -> bool:
    root = report.get("campaign_root") if isinstance(report, dict) else None
    return open_report_path(root)


def open_campaign_json_from_report(report: dict[str, Any]) -> bool:
    path = report.get("report_path") if isinstance(report, dict) else None
    if not path and report.get("campaign_root"):
        path = str(Path(str(report["campaign_root"])) / "campaign_report.json")
    return open_report_path(path)


def open_report_path(path: str | Path | None) -> bool:
    if not path:
        return False
    target = Path(path)
    if not target.exists():
        return False
    _open_path(target)
    return True


class PEDSDashboard:
    def __init__(self, root: "tk.Tk", repo_root: Path = PROJECT_ROOT, *, auto_refresh: bool = True) -> None:
        if tk is None or ttk is None:  # pragma: no cover - guarded by create_app/tests.
            raise RuntimeError("tkinter is not available in this Python environment")
        self.root = root
        self.repo_root = Path(repo_root)
        self.status_dir = STATUS_DIR
        self.current_process: subprocess.Popen[str] | None = None
        self.current_status_name: str | None = None
        self.current_status_path: Path | None = None
        self.current_stdout_path: Path | None = None
        self.current_stderr_path: Path | None = None
        self.command_started_monotonic: float | None = None
        self.output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.latest_report: dict[str, Any] = {}
        self.autopilot_summary: dict[str, Any] = {}
        self.brain_status: dict[str, Any] = {}
        self.polling_started = False
        self.last_status_tail_signature: tuple[tuple[str, ...], tuple[str, ...]] | None = None

        self.root.title("PEDS Operator Cockpit")
        self.root.geometry("1280x820")

        self.status_var = tk.StringVar(value="idle")
        self.operator_message_var = tk.StringVar(value="Ready.")
        self.last_action_var = tk.StringVar(value="none")
        self.current_command_var = tk.StringVar(value="")
        self.started_at_var = tk.StringVar(value="")
        self.elapsed_var = tk.StringVar(value="0")
        self.returncode_var = tk.StringVar(value="")
        self.status_path_var = tk.StringVar(value="")
        self.stdout_path_var = tk.StringVar(value="")
        self.stderr_path_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="No campaign report loaded.")
        self.safety_var = tk.StringVar(value="Safety: unknown")
        self.brain_card_vars: dict[str, Any] = {}
        self.recommended_label_var = tk.StringVar(value="Recommended Next Action: unknown")
        self.recommended_reason_var = tk.StringVar(value="Refresh brain state to compute the next action.")
        self.artifact_type_var = tk.StringVar(value="all")
        self.best_survivor_var = tk.StringVar(value="none")
        self.what_improved_var = tk.StringVar(value="No autopilot output yet.")
        self.useful_output_ready_var = tk.StringVar(value="No")
        self.active_policy_var = tk.StringVar(value="none")
        self.autopilot_safety_var = tk.StringVar(value="unknown")
        self.recommended_use_var = tk.StringVar(value="Run Improve and Produce Useful Output.")
        self.autopilot_status_var = tk.StringVar(value="No autopilot run loaded.")
        self.output_folder_var = tk.StringVar(value="")
        self.brain_objective_var = tk.StringVar(value="No PEDS brain initialized.")
        self.brain_focus_var = tk.StringVar(value="unknown")
        self.brain_active_candidate_var = tk.StringVar(value="none")
        self.brain_status_var = tk.StringVar(value="unknown")
        self.brain_next_action_var = tk.StringVar(value="none")
        self.brain_latest_decision_var = tk.StringVar(value="none")
        self.brain_latest_output_var = tk.StringVar(value="none")
        self.brain_latest_task_var = tk.StringVar(value="none")
        self.brain_task_result_var = tk.StringVar(value="none")
        self.brain_queue_var = tk.StringVar(value="none")
        self.brain_manual_command_var = tk.StringVar(value="none")
        self.actuator_provider_var = tk.StringVar(value="manual")
        self.actuator_noninteractive_var = tk.StringVar(value="No")

        self.generations_var = tk.StringVar(value=str(DEFAULT_GENERATIONS))
        self.runs_var = tk.StringVar(value=str(DEFAULT_RUNS))
        self.max_attempts_var = tk.StringVar(value=str(DEFAULT_MAX_ATTEMPTS_PER_GENERATION))
        self.policy_var = tk.StringVar(value="last-valid-parent")
        self.strict_var = tk.BooleanVar(value=False)

        self._build_ui()
        if auto_refresh:
            self.refresh_latest_report()
            self._schedule_poll()

    def _build_ui(self) -> None:
        self._configure_styles()
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        self._build_controls(outer)
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))
        self._build_brain_tab()
        self._build_use_peds_tab()
        self._build_overview_tab()
        self._build_story_tab()
        self._build_learning_tab()
        self._build_safety_tab()
        self._build_status_tab()
        self._build_logs_tab()
        self._build_summary_tab()
        self._build_raw_json_tab()
        self.lineage_tree = self._build_table_tab("Lineage", LINEAGE_COLUMNS)
        self.attempts_tree = self._build_table_tab("Attempts", ATTEMPT_COLUMNS)
        self.mutations_tree = self._build_table_tab("Mutations", MUTATION_COLUMNS)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("SafetyClean.TLabel", foreground="green")
        style.configure("SafetyWarning.TLabel", foreground="red", font=("TkDefaultFont", 10, "bold"))
        style.configure("CardTitle.TLabel", font=("TkDefaultFont", 9, "bold"))
        style.configure("CardValue.TLabel", font=("TkDefaultFont", 12, "bold"))
        style.configure("MissionHeader.TLabel", font=("TkDefaultFont", 15, "bold"))

    def _build_controls(self, parent: "ttk.Frame") -> None:
        frame = ttk.LabelFrame(parent, text="PEDS Control Center", padding=8)
        frame.pack(fill="x")
        for index in range(4):
            frame.columnconfigure(index, weight=1)

        status = ttk.Frame(frame)
        status.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        status.columnconfigure(1, weight=0)
        status.columnconfigure(3, weight=0)
        status.columnconfigure(5, weight=1)
        ttk.Label(status, text="State", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))
        ttk.Label(status, textvariable=self.status_var, width=12).grid(row=0, column=1, sticky="w", padx=(0, 12))
        ttk.Label(status, text="Action", style="CardTitle.TLabel").grid(row=0, column=2, sticky="w", padx=(0, 4))
        ttk.Label(status, textvariable=self.last_action_var, width=28).grid(row=0, column=3, sticky="w", padx=(0, 12))
        ttk.Label(status, text="Message", style="CardTitle.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 4))
        ttk.Label(status, textvariable=self.operator_message_var, wraplength=720).grid(row=0, column=5, sticky="ew")

        self._build_button_group(
            frame,
            "Use PEDS",
            [
                ("Improve + Output", self.run_autopilot_improve),
                ("Use Latest Survivor", self.run_autopilot_use_latest),
                ("Activate Survivor", self.run_autopilot_activate_best),
                ("Copy Prompt", self.copy_codex_prompt),
                ("Open Output", self.open_autopilot_output_folder),
            ],
            row=1,
            column=0,
        )
        self._build_button_group(
            frame,
            "PEDS Brain",
            [
                ("Initialize", self.run_brain_init),
                ("Think", self.run_brain_think),
                ("One Step", self.run_brain_step),
                ("Loop 5", self.run_brain_loop_5),
                ("Run Actuator Next", self.run_actuator_next),
                ("Brain Step + Actuator", self.run_brain_step_with_actuator),
                ("Refresh Actuator Providers", self.refresh_actuator_providers),
                ("Claim Codex Task", self.run_brain_claim_codex_task),
                ("List Queue", self.run_brain_list_codex_queue),
                ("Record Task Result", self.run_brain_record_task_result),
                ("Open Task", self.open_latest_brain_task),
            ],
            row=1,
            column=1,
        )
        self._build_button_group(
            frame,
            "Campaign Engine",
            [
                ("Recommended", self.run_recommended_next_action),
                ("5x2 Learning", self.run_5x2_campaign),
                ("10x3 Bigger", self.run_bigger_campaign),
                ("Mutation Stress", self.run_mutation_stress_campaign),
                ("3 Gen Smoke", self.run_three_generation_campaign),
                ("Custom", self.run_custom_campaign),
            ],
            row=1,
            column=2,
        )
        self._build_button_group(
            frame,
            "Validation + Evidence",
            [
                ("Validate Core", lambda: self.run_validation("broad_peds")),
                ("Validate Mutation", lambda: self.run_validation("mutation")),
                ("Validate GUI", self.run_gui_validation),
                ("Evidence Check", lambda: self.run_validation("campaign")),
                ("Trainer Check", self.run_trainer_signal_check),
                ("Refresh", self.refresh_latest_report),
                ("Open Evidence", self.open_latest_campaign_folder),
                ("Open JSON", self.open_latest_campaign_json),
                ("Stop Run", self.stop_running_command),
            ],
            row=1,
            column=3,
        )

        custom = ttk.Frame(frame)
        custom.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        for index in range(12):
            custom.columnconfigure(index, weight=0)
        custom.columnconfigure(11, weight=1)
        ttk.Label(custom, text="generations").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(custom, from_=1, to=999, width=5, textvariable=self.generations_var).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(custom, text="runs").grid(row=0, column=2, sticky="w")
        ttk.Spinbox(custom, from_=1, to=999, width=5, textvariable=self.runs_var).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(custom, text="max attempts/gen").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(custom, from_=1, to=999, width=5, textvariable=self.max_attempts_var).grid(row=0, column=5, sticky="w", padx=(4, 12))
        ttk.Label(custom, text="continuation").grid(row=0, column=6, sticky="w")
        ttk.Combobox(custom, values=CONTINUATION_POLICIES, textvariable=self.policy_var, width=18, state="readonly").grid(
            row=0, column=7, sticky="w", padx=(4, 12)
        )
        ttk.Checkbutton(custom, text="strict", variable=self.strict_var).grid(row=0, column=8, sticky="w")

    def _build_button_group(
        self,
        parent: "ttk.Frame",
        title: str,
        buttons: list[tuple[str, Any]],
        *,
        row: int,
        column: int,
    ) -> None:
        group = ttk.LabelFrame(parent, text=title, padding=6)
        group.grid(row=row, column=column, sticky="nsew", padx=4)
        group.columnconfigure(0, weight=1)
        group.columnconfigure(1, weight=1)
        for index, (label, command) in enumerate(buttons):
            ttk.Button(group, text=label, command=command).grid(row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)

    def _build_brain_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="PEDS Brain")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(3, weight=1)
        ttk.Label(tab, text="PEDS Brain Control Loop", style="MissionHeader.TLabel").grid(row=0, column=0, sticky="w")

        actions = ttk.LabelFrame(tab, text="Brain Actions", padding=8)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        buttons = [
            ("Initialize Brain", self.run_brain_init),
            ("Think", self.run_brain_think),
            ("Run One Brain Step", self.run_brain_step),
            ("Run Brain Loop 5 Steps", self.run_brain_loop_5),
            ("Run Actuator Next", self.run_actuator_next),
            ("Brain Step + Actuator", self.run_brain_step_with_actuator),
            ("Refresh Actuator Providers", self.refresh_actuator_providers),
            ("Claim Codex Task", self.run_brain_claim_codex_task),
            ("List Queue", self.run_brain_list_codex_queue),
            ("Record Task Result", self.run_brain_record_task_result),
            ("Open Latest PEDS-Decided Task", self.open_latest_brain_task),
        ]
        for index, (label, command) in enumerate(buttons):
            ttk.Button(actions, text=label, command=command).grid(row=index // 4, column=index % 4, sticky="ew", padx=3, pady=3)
            actions.columnconfigure(index % 4, weight=1)

        cards = ttk.Frame(tab)
        cards.grid(row=2, column=0, sticky="ew")
        card_specs = [
            ("Current Objective", self.brain_objective_var),
            ("Current Focus", self.brain_focus_var),
            ("Active Policy Candidate", self.brain_active_candidate_var),
            ("Status", self.brain_status_var),
            ("Next Action", self.brain_next_action_var),
            ("Latest Decision", self.brain_latest_decision_var),
            ("Latest Useful Output", self.brain_latest_output_var),
            ("Latest Task", self.brain_latest_task_var),
            ("Task Result", self.brain_task_result_var),
            ("Codex Queue", self.brain_queue_var),
            ("Selected Actuator Provider", self.actuator_provider_var),
            ("Noninteractive Actuator", self.actuator_noninteractive_var),
            ("Manual Actuator Command", self.brain_manual_command_var),
        ]
        for index, (label, var) in enumerate(card_specs):
            card = ttk.LabelFrame(cards, text=label, padding=8)
            card.grid(row=index // 3, column=index % 3, sticky="nsew", padx=4, pady=4)
            ttk.Label(card, textvariable=var, wraplength=340).pack(anchor="w")
            cards.columnconfigure(index % 3, weight=1)

        queue_frame = ttk.LabelFrame(tab, text="Codex Actuator Queue", padding=8)
        queue_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        queue_frame.rowconfigure(0, weight=1)
        queue_frame.columnconfigure(0, weight=1)
        self.brain_queue_text = tk.Text(queue_frame, height=8, wrap="word")
        queue_scroll = ttk.Scrollbar(queue_frame, orient="vertical", command=self.brain_queue_text.yview)
        self.brain_queue_text.configure(yscrollcommand=queue_scroll.set)
        self.brain_queue_text.grid(row=0, column=0, sticky="nsew")
        queue_scroll.grid(row=0, column=1, sticky="ns")

    def _build_use_peds_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Use PEDS")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)
        ttk.Label(tab, text="PEDS: Local Self-Improvement Engine", style="MissionHeader.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(tab, text="What do you want the improved local assistant to help you do?").grid(row=1, column=0, sticky="w", pady=(8, 2))
        self.objective_text = tk.Text(tab, height=3, wrap="word")
        self.objective_text.grid(row=2, column=0, sticky="ew")
        self.objective_text.insert("1.0", DEFAULT_OBJECTIVE)

        actions = ttk.LabelFrame(tab, text="Main Actions", padding=8)
        actions.grid(row=3, column=0, sticky="ew", pady=8)
        actions.columnconfigure(7, weight=1)
        ttk.Button(actions, text="Improve and Produce Useful Output", command=self.run_autopilot_improve).grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        ttk.Button(actions, text="Use Latest Survivor", command=self.run_autopilot_use_latest).grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        ttk.Button(actions, text="Activate Best Survivor", command=self.run_autopilot_activate_best).grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        ttk.Button(actions, text="Copy Codex Prompt", command=self.copy_codex_prompt).grid(row=0, column=3, padx=3, pady=3, sticky="ew")
        ttk.Button(actions, text="Open Output Folder", command=self.open_autopilot_output_folder).grid(row=0, column=4, padx=3, pady=3, sticky="ew")
        ttk.Label(actions, text="Output type").grid(row=0, column=5, padx=(10, 3), sticky="e")
        ttk.Combobox(actions, values=("all", "prompt", "profile", "skill", "plan"), textvariable=self.artifact_type_var, width=10, state="readonly").grid(row=0, column=6, padx=3, sticky="w")

        body = ttk.Frame(tab)
        body.grid(row=4, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        cards = ttk.LabelFrame(body, text="Current Result", padding=8)
        cards.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        result_cards = [
            ("Best Survivor", self.best_survivor_var),
            ("What Improved", self.what_improved_var),
            ("Useful Output Ready", self.useful_output_ready_var),
            ("Active Policy", self.active_policy_var),
            ("Safety", self.autopilot_safety_var),
            ("Recommended Use", self.recommended_use_var),
            ("Output Folder", self.output_folder_var),
            ("Autopilot Status", self.autopilot_status_var),
        ]
        for row, (label, var) in enumerate(result_cards):
            ttk.Label(cards, text=label, style="CardTitle.TLabel").grid(row=row * 2, column=0, sticky="w", pady=(4, 0))
            ttk.Label(cards, textvariable=var, wraplength=330).grid(row=row * 2 + 1, column=0, sticky="w")

        preview_frame = ttk.LabelFrame(body, text="Codex Prompt Preview", padding=8)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.prompt_preview_text = tk.Text(preview_frame, wrap="word", height=20)
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.prompt_preview_text.yview)
        self.prompt_preview_text.configure(yscrollcommand=preview_scroll.set)
        self.prompt_preview_text.grid(row=0, column=0, sticky="nsew")
        preview_scroll.grid(row=0, column=1, sticky="ns")

    def _build_overview_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Brain Status")
        ttk.Label(tab, text="PEDS Brain Mission Control", style="MissionHeader.TLabel").pack(anchor="w", pady=(0, 8))
        cards = ttk.Frame(tab)
        cards.pack(fill="x")
        card_names = [
            "Current Mode",
            "Last Campaign Result",
            "Generations Completed",
            "Total Attempts",
            "Winners",
            "Rejected Attempts",
            "Corrections Worked",
            "Correction Success Rate",
            "Current Survivor",
            "Dominant Diagnosis",
            "Dominant Mutation",
            "Safety Status",
            "Recommended Next Action",
        ]
        for index, name in enumerate(card_names):
            var = tk.StringVar(value="—")
            self.brain_card_vars[name] = var
            card = ttk.LabelFrame(cards, text=name, padding=8)
            card.grid(row=index // 4, column=index % 4, sticky="nsew", padx=4, pady=4)
            ttk.Label(card, textvariable=var, style="CardValue.TLabel", wraplength=245).pack(anchor="w")
            cards.columnconfigure(index % 4, weight=1)

        recommendation = ttk.LabelFrame(tab, text="Recommended Next Action", padding=8)
        recommendation.pack(fill="x", pady=(10, 0))
        ttk.Label(recommendation, textvariable=self.recommended_label_var, style="CardValue.TLabel", wraplength=1080).pack(anchor="w")
        ttk.Label(recommendation, textvariable=self.recommended_reason_var, wraplength=1120).pack(anchor="w", pady=(3, 6))
        ttk.Button(recommendation, text="Run Recommended Next Action", command=self.run_recommended_next_action).pack(anchor="w")

    def _build_story_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Evolution Story")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.story_text = tk.Text(tab, wrap="word", height=24)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.story_text.yview)
        self.story_text.configure(yscrollcommand=scrollbar.set)
        self.story_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_learning_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="What PEDS Learned")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.learning_text = tk.Text(tab, wrap="word", height=20)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.learning_text.yview)
        self.learning_text.configure(yscrollcommand=scrollbar.set)
        self.learning_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_safety_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Safety")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.safety_text = tk.Text(tab, wrap="word", height=20)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.safety_text.yview)
        self.safety_text.configure(yscrollcommand=scrollbar.set)
        self.safety_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_status_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Command Status")
        labels = [
            ("current command", self.current_command_var),
            ("state", self.status_var),
            ("started_at", self.started_at_var),
            ("elapsed_seconds", self.elapsed_var),
            ("returncode", self.returncode_var),
            ("status JSON path", self.status_path_var),
            ("stdout log path", self.stdout_path_var),
            ("stderr log path", self.stderr_path_var),
        ]
        for row, (label, variable) in enumerate(labels):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="nw", padx=(0, 8), pady=3)
            ttk.Label(tab, textvariable=variable, wraplength=1050).grid(row=row, column=1, sticky="nw", pady=3)
        tab.columnconfigure(1, weight=1)

    def _build_logs_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Live Logs")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.log_text = tk.Text(tab, wrap="word", height=24)
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_summary_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Campaign Summary")
        self.safety_label = ttk.Label(tab, textvariable=self.safety_var, style="SafetyWarning.TLabel", wraplength=1120)
        self.safety_label.pack(fill="x", pady=(0, 8))
        text_frame = ttk.Frame(tab)
        text_frame.pack(fill="both", expand=True)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.summary_text = tk.Text(text_frame, wrap="word", height=24)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        self.summary_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_raw_json_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Raw JSON")
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        self.raw_json_text = tk.Text(tab, wrap="none", height=20)
        y_scroll = ttk.Scrollbar(tab, orient="vertical", command=self.raw_json_text.yview)
        x_scroll = ttk.Scrollbar(tab, orient="horizontal", command=self.raw_json_text.xview)
        self.raw_json_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.raw_json_text.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

    def _build_table_tab(self, title: str, columns: tuple[str, ...]) -> "ttk.Treeview":
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text=title)
        tab.rowconfigure(0, weight=1)
        tab.columnconfigure(0, weight=1)
        tree = ttk.Treeview(tab, columns=columns, show="headings", height=20)
        y_scroll = ttk.Scrollbar(tab, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(tab, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        for column in columns:
            tree.heading(column, text=column)
            width = 110 if column in {"generation", "attempt_index", "outcome"} else 220
            tree.column(column, width=width, stretch=True, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        return tree

    def run_brain_init(self) -> None:
        try:
            command = build_brain_command("init", objective=self._objective_value(), max_iterations=10, repo_root=self.repo_root)
        except ValueError as exc:
            self._notify_error("Brain input rejected", str(exc))
            return
        self._launch_watched_command("peds_brain_init", command)

    def run_brain_think(self) -> None:
        self._launch_watched_command("peds_brain_think", build_brain_command("think", repo_root=self.repo_root))

    def run_brain_step(self) -> None:
        self._launch_watched_command("peds_brain_step", build_brain_command("step", repo_root=self.repo_root))

    def run_brain_step_with_actuator(self) -> None:
        self._launch_watched_command("peds_brain_step_with_actuator", build_brain_command("step", execute_actuator=True, repo_root=self.repo_root))

    def run_brain_loop_5(self) -> None:
        self._launch_watched_command("peds_brain_loop_5", build_brain_command("run", max_iterations=5, repo_root=self.repo_root))

    def run_actuator_next(self) -> None:
        self._launch_watched_command("peds_actuator_run_next", build_actuator_command("run-next", repo_root=self.repo_root))

    def refresh_actuator_providers(self) -> None:
        self._launch_watched_command("peds_actuator_providers", build_actuator_command("providers", repo_root=self.repo_root))

    def run_brain_claim_codex_task(self) -> None:
        self._launch_watched_command("peds_brain_claim_codex_task", build_brain_command("claim-codex-task", repo_root=self.repo_root))

    def run_brain_list_codex_queue(self) -> None:
        self._launch_watched_command("peds_brain_list_codex_queue", build_brain_command("list-codex-queue", repo_root=self.repo_root))

    def run_brain_record_task_result(self) -> None:
        result_path = self._select_task_result_path()
        if result_path is None:
            self._notify_info("Record task result cancelled.")
            return
        try:
            command = build_brain_record_task_result_command(result_path, repo_root=self.repo_root)
        except ValueError as exc:
            self._notify_error("Invalid task result", str(exc))
            return
        self._launch_watched_command("peds_brain_record_task_result", command)

    def open_latest_brain_task(self) -> None:
        path = latest_brain_task_path()
        if open_report_path(path):
            self._notify_info(f"Opened PEDS task: {path}")
        else:
            self._notify_error("No PEDS task", f"No PEDS-decided Codex task found at {path}")

    def run_autopilot_improve(self) -> None:
        self._run_autopilot(campaign="5x2", use_latest=False, activate_best=False)

    def run_autopilot_use_latest(self) -> None:
        self._run_autopilot(campaign="none", use_latest=True, activate_best=False)

    def run_autopilot_activate_best(self) -> None:
        self._run_autopilot(campaign="none", use_latest=True, activate_best=True)

    def _run_autopilot(self, *, campaign: str, use_latest: bool, activate_best: bool) -> None:
        try:
            objective = self._objective_value()
            command = build_autopilot_command(
                objective=objective,
                campaign=campaign,
                use_latest=use_latest,
                activate_best=activate_best,
                write=True,
                artifact_type=self.artifact_type_var.get() or "all",
                repo_root=self.repo_root,
            )
        except ValueError as exc:
            self._notify_error("Autopilot input rejected", str(exc))
            self.autopilot_status_var.set(str(exc))
            self._append_log("ui", f"Autopilot input rejected: {exc}\n")
            return
        self._launch_watched_command("peds_autopilot", command)

    def copy_codex_prompt(self) -> None:
        if not hasattr(self, "prompt_preview_text"):
            return
        text = self.prompt_preview_text.get("1.0", "end").strip()
        if not text or text == "No Codex prompt has been generated yet.":
            self._notify_error("No Codex prompt", "No Codex prompt has been generated yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._notify_info("Codex prompt copied.")

    def open_autopilot_output_folder(self) -> None:
        output_dir = self.autopilot_summary.get("output_dir") if isinstance(self.autopilot_summary, dict) else None
        if open_report_path(output_dir):
            self._notify_info(f"Opened output folder: {output_dir}")
        else:
            self._notify_error("No output folder", "No autopilot output folder is available yet.")

    def _objective_value(self) -> str:
        if not hasattr(self, "objective_text"):
            return DEFAULT_OBJECTIVE
        value = self.objective_text.get("1.0", "end").strip()
        if not value:
            raise ValueError("objective must not be blank")
        return value

    def run_validation(self, kind: str) -> None:
        command = build_validation_command(kind, repo_root=self.repo_root)
        self._launch_watched_command(f"peds_{kind}_validation", command)

    def run_three_generation_campaign(self) -> None:
        command = build_campaign_command(
            generations=3,
            runs=1,
            max_attempts_per_generation=3,
            continuation_policy="last-valid-parent",
            strict=False,
            repo_root=self.repo_root,
        )
        self._launch_watched_command("peds_3_generation_campaign", command)

    def run_5x2_campaign(self) -> None:
        command = build_campaign_command(
            generations=5,
            runs=2,
            max_attempts_per_generation=5,
            continuation_policy="last-valid-parent",
            strict=False,
            repo_root=self.repo_root,
        )
        self._launch_watched_command("peds_5x2_adaptive_campaign", command)

    def run_bigger_campaign(self) -> None:
        command = build_campaign_command(
            generations=10,
            runs=3,
            max_attempts_per_generation=5,
            continuation_policy="last-valid-parent",
            strict=False,
            repo_root=self.repo_root,
        )
        self._launch_watched_command("peds_10x3_bigger_campaign", command)

    def run_mutation_stress_campaign(self) -> None:
        command = build_campaign_command(
            generations=10,
            runs=3,
            max_attempts_per_generation=5,
            continuation_policy="last-valid-parent",
            strict=False,
            repo_root=self.repo_root,
        )
        self._launch_watched_command("peds_mutation_stress_campaign", command)

    def run_gui_validation(self) -> None:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_peds_dashboard.py"]
        self._launch_watched_command("peds_gui_validation", command)

    def run_trainer_signal_check(self) -> None:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_peds_trainer.py"]
        self._launch_watched_command("peds_trainer_signal_check", command)

    def run_recommended_next_action(self) -> None:
        recommendation = recommend_next_action(self.latest_report or {})
        command = list(recommendation.get("command") or [])
        if not command:
            self._notify_error("No recommended command", str(recommendation.get("reason") or "Recommended action did not include a command."))
            return
        self._launch_watched_command(str(recommendation.get("command_kind") or "peds_recommended_action"), command)

    def run_custom_campaign(self) -> None:
        try:
            generations = _positive_int(self.generations_var.get(), "generations")
            runs = _positive_int(self.runs_var.get(), "runs")
            max_attempts = _positive_int(self.max_attempts_var.get(), "max_attempts_per_generation")
        except ValueError as exc:
            self._append_log("ui", f"Invalid custom campaign input: {exc}\n")
            self._notify_error("Invalid custom campaign input", str(exc))
            return
        policy = self.policy_var.get() or "last-valid-parent"
        if policy not in CONTINUATION_POLICIES:
            self._append_log("ui", f"Invalid continuation policy: {policy}\n")
            self._notify_error("Invalid continuation policy", policy)
            return
        command = build_campaign_command(
            generations=generations,
            runs=runs,
            max_attempts_per_generation=max_attempts,
            continuation_policy=policy,
            strict=bool(self.strict_var.get()),
            repo_root=self.repo_root,
        )
        self._launch_watched_command("peds_custom_campaign", command)

    def refresh_latest_report(self) -> None:
        self.latest_report = load_latest_campaign_report(self.repo_root)
        self.autopilot_summary = load_latest_autopilot_summary()
        self.brain_status = load_brain_status()
        summary = summarize_campaign_report(self.latest_report)
        self._render_brain(self.brain_status)
        self._render_use_peds(self.autopilot_summary)
        self._render_overview(self.latest_report)
        self._render_story(self.latest_report)
        self._render_learning(self.latest_report)
        self._render_safety(self.latest_report, self.autopilot_summary)
        self._render_summary(summary)
        self._render_raw_json(self.latest_report, self.autopilot_summary)
        self._replace_tree_rows(self.lineage_tree, LINEAGE_COLUMNS, extract_lineage_rows(self.latest_report))
        self._replace_tree_rows(self.attempts_tree, ATTEMPT_COLUMNS, extract_attempt_rows(self.latest_report))
        self._replace_tree_rows(self.mutations_tree, MUTATION_COLUMNS, extract_mutation_rows(self.latest_report))
        if summary.get("found"):
            self.status_var.set("loaded")
            self._set_operator_message("Latest PEDS state loaded.")
        else:
            self.status_var.set("missing report")
            self._set_operator_message(str(summary.get("reason") or "No latest campaign report found."))

    def open_latest_campaign_folder(self) -> None:
        report = self.latest_report or load_latest_campaign_report(self.repo_root)
        if open_campaign_folder_from_report(report):
            self._notify_info(f"Opened evidence folder: {report.get('campaign_root')}")
        else:
            self._notify_error("No evidence folder", str(report.get("reason") or "No latest campaign folder is available."))

    def open_latest_campaign_json(self) -> None:
        report = self.latest_report or load_latest_campaign_report(self.repo_root)
        if open_campaign_json_from_report(report):
            self._notify_info(f"Opened campaign JSON: {report.get('report_path')}")
        else:
            self._notify_error("No campaign JSON", str(report.get("reason") or "No latest campaign JSON is available."))

    def stop_running_command(self) -> None:
        if self.current_process is None or self.current_process.poll() is not None:
            self._notify_info("No running command.")
            return
        self.current_process.terminate()
        self.status_var.set("stopped")
        self._set_mode_card("Stopped")
        self._set_operator_message("Stop requested.")
        self._append_log("ui", "Stop requested: terminated watcher process only.\n")

    def _select_task_result_path(self) -> Path | None:
        if filedialog is None:
            self._notify_error("File picker unavailable", "Tk file dialog is not available in this Python environment.")
            return None
        selected = filedialog.askopenfilename(
            title="Select PEDS task result",
            initialdir=str(PEDS_BRAIN_ROOT / "task_results"),
            filetypes=(("JSON or Markdown", "*.json *.md *.txt"), ("All files", "*.*")),
        )
        return Path(selected) if selected else None

    def _set_operator_message(self, message: str, *, action: str | None = None) -> None:
        self.operator_message_var.set(str(message))
        if action is not None:
            self.last_action_var.set(str(action))

    def _notify_info(self, message: str, *, action: str | None = None) -> None:
        self._set_operator_message(message, action=action)
        self._append_log("ui", f"{message}\n")

    def _notify_error(self, title: str, message: str) -> None:
        text = str(message)
        self.status_var.set("error")
        self._set_operator_message(f"{title}: {text}")
        self._append_log("ui", f"{title}: {text}\n")
        if messagebox is not None:
            try:
                messagebox.showerror(title, text, parent=self.root)
            except Exception:  # noqa: BLE001 - messageboxes are best-effort UI feedback.
                pass

    def _validate_launch_command(self, command: list[str]) -> str | None:
        if not command:
            return "Command is empty."
        if len(command) >= 2 and str(command[1]).endswith(".py"):
            script_path = Path(str(command[1]))
            if not script_path.exists():
                return f"Command script does not exist: {script_path}"
        return None

    def _launch_watched_command(self, name: str, command: list[str]) -> None:
        if self.current_process is not None and self.current_process.poll() is None:
            self._notify_error("Command already running", "Stop the current command before launching another.")
            self._append_log("ui", "A command is already running; stop it before launching another.\n")
            return
        blocker = self._validate_launch_command(command)
        if blocker:
            self._notify_error("Cannot launch command", blocker)
            return
        self.status_dir.mkdir(parents=True, exist_ok=True)
        status_name = f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
        safe_name = _safe_status_name(status_name)
        self.current_status_name = safe_name
        self.current_status_path = self.status_dir / f"{safe_name}.status.json"
        self.current_stdout_path = self.status_dir / f"{safe_name}.stdout.log"
        self.current_stderr_path = self.status_dir / f"{safe_name}.stderr.log"
        watch_command = build_watch_command(
            name=status_name,
            command=command,
            status_dir=str(self.status_dir),
            repo_root=self.repo_root,
        )
        self.current_command_var.set(" ".join(command))
        self.status_path_var.set(str(self.current_status_path))
        self.stdout_path_var.set(str(self.current_stdout_path))
        self.stderr_path_var.set(str(self.current_stderr_path))
        self.returncode_var.set("")
        self.started_at_var.set(_utc_now())
        self.elapsed_var.set("0")
        self.command_started_monotonic = time.monotonic()
        self.status_var.set("running")
        self._set_operator_message(f"Running {name}.", action=name)
        self._set_mode_card("Running")
        self._append_log("ui", f"Launching via watcher: {' '.join(watch_command)}\n")
        self.current_process = subprocess.Popen(
            watch_command,
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self._start_reader_thread(self.current_process.stdout, "stdout")
        self._start_reader_thread(self.current_process.stderr, "stderr")
        self._schedule_poll()

    def _start_reader_thread(self, stream: Any, label: str) -> None:
        def reader() -> None:
            if stream is None:
                return
            try:
                for line in stream:
                    self.output_queue.put((label, line))
            finally:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001 - closing best effort for UI thread reader.
                    pass

        threading.Thread(target=reader, daemon=True).start()

    def _schedule_poll(self) -> None:
        if self.polling_started:
            return
        self.polling_started = True
        self.root.after(250, self._poll_background_state)

    def _poll_background_state(self) -> None:
        self._drain_output_queue()
        self._poll_status_json()
        self._poll_process_exit()
        self._update_elapsed_time()
        self.root.after(500, self._poll_background_state)

    def _drain_output_queue(self) -> None:
        while True:
            try:
                label, line = self.output_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(label, line)

    def _poll_status_json(self) -> None:
        status_path = self.current_status_path
        if status_path is None or not status_path.exists():
            return
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if not isinstance(status, dict):
            return
        self.status_var.set(str(status.get("status") or "running"))
        self.started_at_var.set(str(status.get("started_at") or self.started_at_var.get()))
        self.elapsed_var.set(str(status.get("elapsed_seconds") or self.elapsed_var.get()))
        returncode = status.get("returncode")
        self.returncode_var.set("" if returncode is None else str(returncode))
        self._render_status_tail(status)

    def _render_status_tail(self, status: dict[str, Any]) -> None:
        stdout_lines = status.get("last_stdout_lines") if isinstance(status.get("last_stdout_lines"), list) else []
        stderr_lines = status.get("last_stderr_lines") if isinstance(status.get("last_stderr_lines"), list) else []
        signature = (tuple(str(line) for line in stdout_lines[-5:]), tuple(str(line) for line in stderr_lines[-5:]))
        if signature == self.last_status_tail_signature:
            return
        self.last_status_tail_signature = signature
        if stdout_lines or stderr_lines:
            tail = ["\n--- watcher status tail ---\n"]
            tail.extend(f"stdout: {line}\n" for line in stdout_lines[-5:])
            tail.extend(f"stderr: {line}\n" for line in stderr_lines[-5:])
            self._append_log("status", "".join(tail))

    def _poll_process_exit(self) -> None:
        if self.current_process is None:
            return
        returncode = self.current_process.poll()
        if returncode is None:
            return
        if self.returncode_var.get() == "":
            self.returncode_var.set(str(returncode))
        state = self.status_var.get()
        if state == "running":
            self.status_var.set("passed" if returncode == 0 else "failed")
        self._set_mode_card("Idle")
        command_name = self.current_status_name or "command"
        self.current_process = None
        if returncode == 0:
            self.refresh_latest_report()
            self._set_operator_message(f"Completed: {command_name}.", action=command_name)
        else:
            detail = self._latest_failure_detail()
            self._notify_error("Command failed", f"{command_name} exited with {returncode}. {detail}".strip())

    def _update_elapsed_time(self) -> None:
        if self.current_process is None or self.current_process.poll() is not None:
            return
        if self.command_started_monotonic is not None:
            self.elapsed_var.set(str(round(time.monotonic() - self.command_started_monotonic, 1)))

    def _latest_failure_detail(self) -> str:
        status_path = self.current_status_path
        if status_path is not None and status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            if isinstance(status, dict):
                stderr_lines = status.get("last_stderr_lines") if isinstance(status.get("last_stderr_lines"), list) else []
                stdout_lines = status.get("last_stdout_lines") if isinstance(status.get("last_stdout_lines"), list) else []
                tail = stderr_lines[-3:] or stdout_lines[-3:]
                if tail:
                    return " ".join(str(line) for line in tail)
        return ""

    def _render_brain(self, state: dict[str, Any]) -> None:
        if not state.get("found"):
            self.brain_objective_var.set(str(state.get("reason") or "No PEDS brain initialized."))
            self.brain_focus_var.set("uninitialized")
            self.brain_active_candidate_var.set("none")
            self.brain_status_var.set("idle")
            self.brain_next_action_var.set("Initialize Brain")
            self.brain_latest_decision_var.set("none")
            self.brain_latest_output_var.set("none")
            self.brain_latest_task_var.set("none")
            self.brain_task_result_var.set("none")
            self.brain_queue_var.set("none")
            self.brain_manual_command_var.set("none")
            self.actuator_provider_var.set("manual")
            self.actuator_noninteractive_var.set("No")
            self._render_brain_queue([])
            return
        self.brain_objective_var.set(str(state.get("current_objective") or ""))
        self.brain_focus_var.set(str(state.get("current_focus") or "unknown"))
        self.brain_active_candidate_var.set(_short_id(state.get("current_active_candidate_id")) or "none")
        self.brain_status_var.set(str(state.get("status") or "unknown"))
        next_action = state.get("next_action") if isinstance(state.get("next_action"), dict) else {}
        self.brain_next_action_var.set(str(next_action.get("action") or "none"))
        decision = load_latest_brain_decision()
        if isinstance(decision, dict):
            self.brain_latest_decision_var.set(f"{decision.get('action')} - {decision.get('reason')}")
        else:
            self.brain_latest_decision_var.set(str(state.get("last_decision_id") or "none"))
        self.brain_latest_output_var.set(_display_path(state.get("last_autopilot_output") or state.get("last_campaign_report")))
        self.brain_latest_task_var.set(_display_path(state.get("last_codex_task")))
        task_result = state.get("last_codex_task_result") if isinstance(state.get("last_codex_task_result"), dict) else {}
        if task_result:
            self.brain_task_result_var.set(f"{task_result.get('status')} / {task_result.get('brain_evaluation') or 'pending evaluation'}")
        else:
            self.brain_task_result_var.set("none")
        manual_command = latest_manual_actuator_command(state)
        self.brain_manual_command_var.set(manual_command or "none")
        provider_status = state.get("actuator_providers") if isinstance(state.get("actuator_providers"), dict) else {}
        provider_summary = summarize_actuator_provider_status(provider_status)
        self.actuator_provider_var.set(str(provider_summary.get("selected_provider") or "manual"))
        self.actuator_noninteractive_var.set("Yes" if provider_summary.get("noninteractive_available") else "No")
        queue = state.get("codex_queue") if isinstance(state.get("codex_queue"), list) else []
        active_queue = [item for item in queue if item.get("status") in {"pending", "claimed", "running", "manual_actuator_required", "failed"}]
        if active_queue:
            latest = active_queue[-1]
            self.brain_queue_var.set(f"{latest.get('status')} {_short_id(latest.get('queue_id'))}")
        elif queue:
            latest = queue[-1]
            self.brain_queue_var.set(f"{latest.get('status')} {_short_id(latest.get('queue_id'))}")
        else:
            self.brain_queue_var.set("none")
        self._render_brain_queue(queue)

    def _render_brain_queue(self, queue: list[dict[str, Any]]) -> None:
        if not hasattr(self, "brain_queue_text"):
            return
        lines: list[str] = []
        groups = split_task_queue_by_status(queue)
        for label in ("pending", "claimed", "running", "manual_actuator_required", "failed", "completed", "superseded"):
            items = groups.get(label) or []
            lines.append(f"[{label}] {len(items)}")
            for item in items[-5:]:
                task_path = _display_path(item.get("task_path"), limit=110)
                manual = ""
                if label == "manual_actuator_required":
                    manual = f"\nmanual: {latest_manual_actuator_command({'last_codex_task_result': item}) or item.get('next_recommendation') or ''}"
                lines.append(
                    f"{item.get('timestamp', '')}  {item.get('task_id') or item.get('queue_id', '')}  {item.get('action', '')}\n{task_path}{manual}"
                )
        self.brain_queue_text.delete("1.0", "end")
        self.brain_queue_text.insert("1.0", "\n\n".join(lines) if queue else "No Codex queue entries.")

    def _render_use_peds(self, summary: dict[str, Any]) -> None:
        parsed = parse_autopilot_summary(summary) if summary else {"found": False}
        candidate = _short_id(parsed.get("candidate_id")) or "none"
        self.best_survivor_var.set(candidate)
        what_improved = parsed.get("what_improved") if isinstance(parsed.get("what_improved"), list) else []
        self.what_improved_var.set("; ".join(str(item) for item in what_improved[:3]) if what_improved else "No autopilot output yet.")
        self.useful_output_ready_var.set("Yes" if parsed.get("useful_output_ready") else "No")
        active = parsed.get("active_policy") or parsed.get("latest_active_policy")
        if isinstance(active, dict) and active.get("candidate_id"):
            self.active_policy_var.set(_short_id(active.get("candidate_id")))
        else:
            self.active_policy_var.set("none")
        safety = parsed.get("safety") if isinstance(parsed.get("safety"), dict) else {}
        self.autopilot_safety_var.set("✅ Clean" if _safety_clean(safety) else "❌ Safety issue")
        recommended_uses = parsed.get("recommended_uses") if isinstance(parsed.get("recommended_uses"), list) else []
        self.recommended_use_var.set(str(recommended_uses[0]) if recommended_uses else "Use the generated Codex prompt on the next repo task.")
        self.output_folder_var.set(str(parsed.get("output_dir") or ""))
        if parsed.get("ok"):
            self.autopilot_status_var.set(
                f"campaign={parsed.get('campaign')} ran={parsed.get('campaign_ran')} survivor={parsed.get('survivor_selected')} artifacts={parsed.get('artifacts_written')}"
            )
        else:
            self.autopilot_status_var.set(str(parsed.get("reason") or parsed.get("error") or "No autopilot run loaded."))
        if hasattr(self, "prompt_preview_text"):
            self.prompt_preview_text.delete("1.0", "end")
            self.prompt_preview_text.insert("1.0", prompt_preview_from_summary(parsed))

    def _render_overview(self, report: dict[str, Any]) -> None:
        mode = "Running" if self.current_process is not None and self.current_process.poll() is None else "Idle"
        status = build_brain_status(report, current_mode=mode)
        for name, value in status.items():
            if name in self.brain_card_vars:
                self.brain_card_vars[name].set(value)
        recommendation = recommend_next_action(report)
        self.recommended_label_var.set(str(recommendation.get("label") or "No recommendation"))
        self.recommended_reason_var.set(str(recommendation.get("reason") or ""))

    def _render_story(self, report: dict[str, Any]) -> None:
        if not hasattr(self, "story_text"):
            return
        self.story_text.delete("1.0", "end")
        self.story_text.insert("1.0", "\n".join(build_evolution_story(report)))

    def _render_learning(self, report: dict[str, Any]) -> None:
        if not hasattr(self, "learning_text"):
            return
        recommendation = recommend_next_action(report)
        text = summarize_peds_learning(report)
        text += "\n\nRecommended next action:\n"
        text += f"{recommendation.get('label')} — {recommendation.get('reason')}"
        self.learning_text.delete("1.0", "end")
        self.learning_text.insert("1.0", text)

    def _render_safety(self, report: dict[str, Any], autopilot_summary: dict[str, Any]) -> None:
        if not hasattr(self, "safety_text"):
            return
        campaign_safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
        autopilot_safety = autopilot_summary.get("safety") if isinstance(autopilot_summary.get("safety"), dict) else {}
        lines = [
            "Campaign safety:",
            json.dumps(campaign_safety, ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "Autopilot safety:",
            json.dumps(autopilot_safety, ensure_ascii=False, indent=2, sort_keys=True),
        ]
        if not _safety_clean(campaign_safety) or (autopilot_safety and not _safety_clean(autopilot_safety)):
            lines.insert(0, "SAFETY WARNING: inspect before running larger campaigns.\n")
        else:
            lines.insert(0, "Safety clean: no network/model/fake-weight contamination reported.\n")
        self.safety_text.delete("1.0", "end")
        self.safety_text.insert("1.0", "\n".join(lines))

    def _render_raw_json(self, report: dict[str, Any], autopilot_summary: dict[str, Any]) -> None:
        if not hasattr(self, "raw_json_text"):
            return
        payload = {"latest_campaign_report": report, "latest_autopilot_summary": autopilot_summary}
        self.raw_json_text.delete("1.0", "end")
        self.raw_json_text.insert("1.0", json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    def _render_summary(self, summary: dict[str, Any]) -> None:
        lines = []
        if not summary.get("found"):
            lines.append(f"No latest campaign report found: {summary.get('reason') or 'not found'}")
        else:
            for field in SUMMARY_FIELDS:
                lines.append(f"{field}: {summary.get(field)}")
            lines.extend(
                [
                    "",
                    "Safety fields:",
                    f"network_attempts: {json.dumps(summary.get('network_attempts', []), ensure_ascii=False)}",
                    f"model_modules_loaded: {json.dumps(summary.get('model_modules_loaded', []), ensure_ascii=False)}",
                    f"fake_weights_written: {summary.get('fake_weights_written')}",
                    "",
                    "Process success is command/report success; rejected candidate attempts are search evidence, not process failure.",
                ]
            )
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", "\n".join(lines))
        if summary.get("safety_clean"):
            self.safety_label.configure(style="SafetyClean.TLabel")
            self.safety_var.set("Safety clean: no network attempts, model modules, or fake weights reported.")
        else:
            self.safety_label.configure(style="SafetyWarning.TLabel")
            self.safety_var.set(
                "SAFETY WARNING: "
                f"network_attempts={summary.get('network_attempts')}, "
                f"model_modules_loaded={summary.get('model_modules_loaded')}, "
                f"fake_weights_written={summary.get('fake_weights_written')}"
            )

    def _replace_tree_rows(self, tree: "ttk.Treeview", columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            tree.insert("", "end", values=[_cell(row.get(column)) for column in columns])

    def _set_mode_card(self, value: str) -> None:
        var = self.brain_card_vars.get("Current Mode")
        if var is not None:
            var.set(value)

    def _append_log(self, label: str, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        prefix = f"[{label}] " if label else ""
        self.log_text.insert("end", prefix + text)
        self.log_text.see("end")


# Backward-compatible class name used by older tests/docs.
PEDSOperatorCockpit = PEDSDashboard


def create_app(repo_root: Path = PROJECT_ROOT) -> "tk.Tk":
    if tk is None or ttk is None:
        raise RuntimeError("tkinter is not available in this Python environment")
    root = tk.Tk()
    PEDSDashboard(root, repo_root=repo_root)
    return root


def main() -> int:
    root = create_app()
    root.mainloop()
    return 0


def _attempts_with_run_index(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    runs = report.get("runs") if isinstance(report.get("runs"), list) else []
    if runs:
        for run in runs:
            if not isinstance(run, dict):
                continue
            run_index = run.get("run_index") or 1
            for attempt in run.get("attempts") or []:
                if isinstance(attempt, dict):
                    row = dict(attempt)
                    row.setdefault("run_index", run_index)
                    rows.append(row)
    else:
        for attempt in report.get("attempts") or []:
            if isinstance(attempt, dict):
                row = dict(attempt)
                row.setdefault("run_index", attempt.get("run_index") or 1)
                rows.append(row)
    return sorted(rows, key=lambda item: (_coerce_int(item.get("run_index"), default=1), _coerce_int(item.get("generation"), default=0), _coerce_int(item.get("attempt_index"), default=0)))


def _current_survivor(report: dict[str, Any]) -> Any:
    winning_attempts = report.get("winning_attempts") if isinstance(report.get("winning_attempts"), list) else []
    if winning_attempts:
        last = winning_attempts[-1]
        if isinstance(last, dict):
            return last.get("selected_parent_for_next_generation") or last.get("winning_candidate_id")
    lineage = report.get("lineage") if isinstance(report.get("lineage"), list) else []
    if lineage:
        last = lineage[-1]
        if isinstance(last, dict):
            return last.get("selected_parent_for_next_generation") or last.get("winning_candidate_id")
    return None


def _dominant_diagnosis(report: dict[str, Any]) -> str | None:
    values: list[str] = []
    for attempt in _attempts_with_run_index(report):
        category = _diagnosis_category(attempt)
        if category:
            values.append(str(category))
    if not values:
        for item in report.get("diagnoses") or []:
            if isinstance(item, dict):
                category = item.get("category") or _diagnosis_category(item)
                if category:
                    values.append(str(category))
    return Counter(values).most_common(1)[0][0] if values else None


def _dominant_mutation_target(report: dict[str, Any]) -> str | None:
    mutation_rows = extract_mutation_rows(report)
    diagnosed = [row for row in mutation_rows if row.get("diagnosis_category") not in (None, "", "unknown")]
    values = [str(row["mutation_target"]) for row in (diagnosed or mutation_rows) if row.get("mutation_target")]
    if not values:
        for attempt in _attempts_with_run_index(report):
            target = _attempt_mutation_target(attempt)
            if target:
                values.append(str(target))
    return Counter(values).most_common(1)[0][0] if values else None


def _dominant_mutation_label(report: dict[str, Any]) -> str | None:
    labels: list[str] = []
    attempts = _attempts_with_run_index(report)
    corrected_or_diagnosed = [
        attempt
        for attempt in attempts
        if _coerce_int(attempt.get("attempt_index"), default=1) > 1
        or (_attempt_mutation_diagnosis(attempt) not in (None, "", "unknown"))
    ]
    for attempt in corrected_or_diagnosed or attempts:
        target = _attempt_mutation_target(attempt)
        if not target:
            continue
        correction = _correction_name(attempt.get("correction"))
        if correction and correction != "no_safe_correction_available":
            labels.append(f"{target} {str(correction).replace('_', ' ')}")
        else:
            diagnosis = _attempt_mutation_diagnosis(attempt)
            labels.append(f"{target} {str(diagnosis).replace('_', ' ')}" if diagnosis and diagnosis != "unknown" else str(target))
    if not labels:
        for row in extract_mutation_rows(report):
            target = row.get("mutation_target")
            if target:
                labels.append(str(target))
    return Counter(labels).most_common(1)[0][0] if labels else None


def _mutation_families(report: dict[str, Any]) -> set[str]:
    rows = extract_mutation_rows(report)
    diagnosed = [row for row in rows if row.get("diagnosis_category") not in (None, "", "unknown")]
    families: set[str] = set()
    for row in diagnosed or rows:
        target = row.get("mutation_target")
        diagnosis = row.get("diagnosis_category")
        if target:
            families.add(f"{target}:{diagnosis or 'unknown'}")
    return families


def _attempt_mutation_target(attempt: dict[str, Any]) -> Any:
    mutation = attempt.get("mutation") if isinstance(attempt.get("mutation"), dict) else {}
    return attempt.get("mutation_target") or mutation.get("mutation_target")


def _attempt_mutation_diagnosis(attempt: dict[str, Any]) -> Any:
    mutation = attempt.get("mutation") if isinstance(attempt.get("mutation"), dict) else {}
    return mutation.get("diagnosis_category") or _diagnosis_category(attempt)


def _lineage_advanced(report: dict[str, Any]) -> bool:
    winning_attempts = report.get("winning_attempts") if isinstance(report.get("winning_attempts"), list) else []
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for item in winning_attempts:
        if isinstance(item, dict):
            grouped.setdefault(item.get("run_index") or 1, []).append(item)
    if not grouped:
        lineage = report.get("lineage") if isinstance(report.get("lineage"), list) else []
        grouped[1] = [item for item in lineage if isinstance(item, dict)]
    saw_pair = False
    for rows in grouped.values():
        rows = sorted(rows, key=lambda item: _coerce_int(item.get("generation"), default=0))
        for previous, current in zip(rows, rows[1:]):
            previous_winner = previous.get("winning_candidate_id") or previous.get("selected_parent_for_next_generation")
            current_parent = current.get("parent_candidate_id") or current.get("source_candidate_id")
            if previous_winner and current_parent:
                saw_pair = True
                if current_parent != previous_winner:
                    return False
    return saw_pair or bool(winning_attempts or report.get("lineage"))


def _short_id(value: Any, limit: int = 12) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _display_path(value: Any, limit: int = 96) -> str:
    if value in (None, ""):
        return "none"
    text = str(value)
    try:
        path = Path(text)
        if path.name and len(text) > limit:
            parent = path.parent.name
            text = f".../{parent}/{path.name}" if parent else path.name
    except (TypeError, ValueError):
        pass
    if len(text) <= limit:
        return text
    return "..." + text[-max(0, limit - 3) :]


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_present(mapping: dict[str, Any], key: str, *, default: Any) -> Any:
    value = mapping.get(key) if isinstance(mapping, dict) else None
    return default if value is None else value


def _safety_clean(safety: dict[str, Any]) -> bool:
    return not bool(
        safety.get("network_attempts")
        or safety.get("model_modules_loaded")
        or safety.get("fake_weights_written")
    )


def _diagnosis_category(attempt: dict[str, Any]) -> Any:
    diagnosis = attempt.get("diagnosis")
    if isinstance(diagnosis, dict):
        return diagnosis.get("category")
    return attempt.get("diagnosis_category")


def _correction_name(correction: Any) -> Any:
    if isinstance(correction, dict):
        return correction.get("correction") or correction.get("action")
    return correction


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer") from None
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _safe_status_name(name: str) -> str:
    cleaned = []
    for char in str(name):
        cleaned.append(char if char.isalnum() or char in "_.-" else "_")
    value = "".join(cleaned).strip("._")
    return value or "command"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _open_path(path: Path) -> None:
    if os.name == "nt" and hasattr(os, "startfile"):
        os.startfile(str(path))  # type: ignore[attr-defined]
    else:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    raise SystemExit(main())
