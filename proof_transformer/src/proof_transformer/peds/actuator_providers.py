from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROVIDER_NAMES = {"codex", "pi", "manual", "auto"}


def detect_actuator_providers() -> dict[str, Any]:
    providers = [_probe_codex(), _probe_pi(), _manual_provider()]
    selected = "manual"
    for provider in providers:
        if provider["available"] and provider["supports_noninteractive"]:
            selected = str(provider["name"])
            break
    return {"providers": providers, "selected_provider": selected}


def build_actuator_command(*, provider: str, task_path: Path) -> list[str] | None:
    provider = _normalize_provider(provider)
    path = Path(task_path)
    if provider == "manual":
        return None
    if not path.exists():
        return None
    task_text = path.read_text(encoding="utf-8", errors="replace")
    if provider == "codex":
        return [*_codex_command(), "exec", task_text]
    if provider == "pi":
        return ["pi", "exec", task_text]
    return None


def run_actuator_task(
    *,
    task_path: Path,
    provider: str | None = None,
    timeout_seconds: int = 1800,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_path = Path(task_path)
    detection = detect_actuator_providers()
    provider_name = _select_provider(provider, detection)
    provider_info = _provider_info(detection, provider_name)
    result_path = output_dir / "actuator_result.json"

    if provider_info is None:
        provider_info = {
            "name": provider_name,
            "available": False,
            "supports_noninteractive": False,
            "probe_commands": [],
            "reason": "unknown_provider",
        }

    command = build_actuator_command(provider=provider_name, task_path=task_path) if provider_info.get("supports_noninteractive") else None
    if command is None:
        result = _manual_required_result(
            task_path=task_path,
            output_dir=output_dir,
            provider_info=provider_info,
            detection=detection,
            reason=str(provider_info.get("reason") or "provider_does_not_support_noninteractive"),
        )
        result["result_path"] = str(result_path)
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return result

    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8", errors="replace")
        stderr_path.write_text(completed.stderr, encoding="utf-8", errors="replace")
        status = "completed" if completed.returncode == 0 else "failed"
        result = _base_result(task_path=task_path, output_dir=output_dir, provider_info=provider_info, detection=detection)
        result.update(
            ok=status == "completed",
            status=status,
            actuator=provider_name,
            command=command,
            returncode=completed.returncode,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            summary=f"{provider_name} provider returned {completed.returncode}.",
            issues_found=[] if status == "completed" else [f"{provider_name}_returncode_nonzero"],
            next_recommendation="PEDS should evaluate the recorded actuator result.",
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8", errors="replace")
        result = _base_result(task_path=task_path, output_dir=output_dir, provider_info=provider_info, detection=detection)
        result.update(
            ok=False,
            status="failed",
            actuator=provider_name,
            command=command,
            returncode=None,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            summary=f"{provider_name} provider timed out after {timeout_seconds} seconds.",
            issues_found=[f"{provider_name}_timeout"],
            next_recommendation="Inspect stdout/stderr and record a follow-up result for PEDS.",
        )
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _codex_command() -> list[str]:
    """Executable argv prefix for the Codex CLI.

    On Windows, npm installs Codex as a .cmd shim that Python's
    subprocess/CreateProcess cannot execute directly: shutil.which finds
    the shim, but running it raises FileNotFoundError, so the probe
    reports command_not_found. The real program is the node script at
    <npm dir>/node_modules/@openai/codex/bin/codex.js, so run it via
    node. Non-Windows hosts and hosts without a resolvable codex.js
    (or node) keep the plain "codex" command, preserving prior behaviour.
    """
    if sys.platform != "win32":
        return ["codex"]
    codex_js = _find_codex_js()
    if codex_js is None:
        return ["codex"]
    node = shutil.which("node")
    if node is None:
        return ["codex"]
    return [node, str(codex_js)]


def _find_codex_js() -> Path | None:
    candidates: list[Path] = []
    shim = shutil.which("codex")
    if shim:
        candidates.append(Path(shim).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(Path(appdata) / "npm" / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    for path_entry in os.environ.get("PATH", "").split(os.pathsep):
        if path_entry:
            candidates.append(Path(path_entry) / "node_modules" / "@openai" / "codex" / "bin" / "codex.js")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _probe_codex() -> dict[str, Any]:
    base = _codex_command()
    probes = [base + ["--help"], base + ["exec", "--help"]]
    help_result = _run_probe(probes[0])
    if not help_result.get("found") or help_result.get("returncode") != 0:
        return _provider("codex", False, False, probes, str(help_result.get("reason") or "codex_help_unavailable"))
    exec_result = _run_probe(probes[1])
    supports = bool(exec_result.get("found") and exec_result.get("returncode") == 0)
    return _provider("codex", True, supports, probes, None if supports else str(exec_result.get("reason") or "codex_exec_unavailable"))


def _probe_pi() -> dict[str, Any]:
    probes = [["pi", "--help"]]
    help_result = _run_probe(probes[0])
    if not help_result.get("found") or help_result.get("returncode") != 0:
        return _provider("pi", False, False, probes, str(help_result.get("reason") or "pi_help_unavailable"))
    help_text = f"{help_result.get('stdout') or ''}\n{help_result.get('stderr') or ''}".lower()
    if "exec" not in help_text:
        return _provider("pi", True, False, probes, "no_safe_noninteractive_mode_detected")
    exec_probe = ["pi", "exec", "--help"]
    probes.append(exec_probe)
    exec_result = _run_probe(exec_probe)
    exec_text = f"{exec_result.get('stdout') or ''}\n{exec_result.get('stderr') or ''}".lower()
    supports = bool(
        exec_result.get("found")
        and exec_result.get("returncode") == 0
        and ("noninteractive" in exec_text or "non-interactive" in exec_text or "execute" in exec_text or "stdin" in exec_text)
    )
    return _provider("pi", True, supports, probes, None if supports else "pi_exec_help_did_not_confirm_safe_noninteractive_mode")


def _manual_provider() -> dict[str, Any]:
    return _provider("manual", True, False, [], None)


def _provider(name: str, available: bool, supports: bool, probes: list[list[str]], reason: str | None) -> dict[str, Any]:
    return {
        "name": name,
        "available": bool(available),
        "supports_noninteractive": bool(supports),
        "probe_commands": probes,
        "reason": reason,
    }


def _run_probe(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except FileNotFoundError:
        return {"found": False, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": "command_not_found"}
    except subprocess.TimeoutExpired:
        return {"found": True, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": "probe_timeout"}
    except OSError as exc:
        return {"found": False, "command": command, "returncode": None, "stdout": "", "stderr": "", "reason": str(exc)}
    return {
        "found": True,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "reason": None if completed.returncode == 0 else "probe_returncode_nonzero",
    }


def _select_provider(provider: str | None, detection: dict[str, Any]) -> str:
    provider = _normalize_provider(provider or "auto")
    if provider == "auto":
        return str(detection.get("selected_provider") or "manual")
    return provider


def _provider_info(detection: dict[str, Any], provider: str) -> dict[str, Any] | None:
    for item in detection.get("providers", []) if isinstance(detection.get("providers"), list) else []:
        if isinstance(item, dict) and item.get("name") == provider:
            return item
    return None


def _normalize_provider(provider: str) -> str:
    provider = str(provider or "auto").strip().lower()
    if provider not in PROVIDER_NAMES:
        raise ValueError(f"Unsupported actuator provider: {provider!r}")
    return provider


def _manual_required_result(
    *,
    task_path: Path,
    output_dir: Path,
    provider_info: dict[str, Any],
    detection: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    result = _base_result(task_path=task_path, output_dir=output_dir, provider_info=provider_info, detection=detection)
    manual_command = (
        f"Open {task_path}, run the task in the available coding agent, write a result report, "
        "then record it with: python scripts/peds_brain.py --record-task-result <result_path> --json"
    )
    result.update(
        ok=False,
        status="manual_actuator_required",
        actuator="manual",
        command=None,
        returncode=None,
        stdout_path=None,
        stderr_path=None,
        summary=f"Provider {provider_info.get('name')} cannot execute non-interactively: {reason}",
        issues_found=[reason],
        next_recommendation=manual_command,
        manual_command=manual_command,
    )
    return result


def _base_result(*, task_path: Path, output_dir: Path, provider_info: dict[str, Any], detection: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_path": str(task_path),
        "ok": False,
        "status": "failed",
        "provider": provider_info.get("name"),
        "provider_available": bool(provider_info.get("available")),
        "provider_supports_noninteractive": bool(provider_info.get("supports_noninteractive")),
        "actuator": provider_info.get("name") or "manual",
        "command": None,
        "returncode": None,
        "stdout_path": None,
        "stderr_path": None,
        "summary": "",
        "files_changed": [],
        "validation": [],
        "issues_found": [],
        "issues_fixed": [],
        "next_recommendation": None,
        "manual_command": None,
        "probe_summary": detection,
        "run_dir": str(output_dir),
        "recorded_at": _utc_now(),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
