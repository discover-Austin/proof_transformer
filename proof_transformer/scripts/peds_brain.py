from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peds_latest_campaign_report import build_latest_campaign_report  # noqa: E402
from proof_transformer.peds.actuator_providers import detect_actuator_providers  # noqa: E402
from proof_transformer.peds.brain import (  # noqa: E402
    CODE_ACTUATOR_ACTIONS,
    DEFAULT_BRAIN_ROOT,
    append_brain_journal,
    claim_next_codex_task,
    choose_next_action,
    ensure_codex_task_queued,
    generate_codex_task,
    initialize_brain,
    latest_active_policy,
    latest_decision,
    list_codex_actuator_queue,
    load_brain_state,
    observe_latest_evidence,
    propose_next_hypotheses,
    record_codex_task_result,
    record_decision,
    record_hypotheses,
    update_active_policy_from_survivor,
    write_brain_state,
)

DEFAULT_AUTOPILOT_OUTPUT_ROOT = PROJECT_ROOT / "temp" / "peds_autopilot_outputs"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as exc:  # noqa: BLE001 - CLI must return JSON errors, not tracebacks.
        result = {"ok": False, "error": str(exc)}
    indent = None if args.json else 2
    print(json.dumps(result, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0 if result.get("ok") else 1


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root) if args.root is not None else DEFAULT_BRAIN_ROOT
    if args.init:
        if not args.objective:
            return {"ok": False, "error": "--objective is required with --init"}
        state = initialize_brain(objective=args.objective, root=root, max_iterations=args.max_iterations)
        return {"ok": True, "state": _compact_state(state), "brain_root": str(root)}
    if args.status:
        return {"ok": True, "state": _compact_state(load_brain_state(root)), "latest_decision": latest_decision(root), "latest_active_policy": latest_active_policy(root), "task_queue": list_codex_actuator_queue(root=root), "actuator_providers": detect_actuator_providers()}
    if args.think:
        return think_once(root=root)
    if args.step:
        return run_brain_step(root=root, execute_actuator=args.execute_actuator)
    if args.run:
        return run_brain_loop(root=root, max_iterations=args.max_iterations, execute_actuator=args.execute_actuator)
    if args.record_task_result:
        result = record_codex_task_result(args.record_task_result, root=root)
        return {"ok": True, "task_result": result, "state": _compact_state(load_brain_state(root))}
    if args.activate_profile:
        profile = _read_json_file(args.activate_profile)
        if not isinstance(profile, dict):
            return {"ok": False, "error": f"could not read survivor profile JSON: {args.activate_profile}"}
        profile = dict(profile, profile_path=str(args.activate_profile))
        active = update_active_policy_from_survivor(profile, root=root)
        return {"ok": True, "active_policy": active, "state": _compact_state(load_brain_state(root))}
    if args.claim_codex_task:
        task = claim_next_codex_task(root=root)
        return {"ok": True, "codex_task": task, "queue": list_codex_actuator_queue(root=root)}
    if args.list_codex_queue:
        return {"ok": True, "queue": list_codex_actuator_queue(root=root)}
    return {"ok": False, "error": "one of --init, --status, --think, --step, --run, --record-task-result, --activate-profile, --claim-codex-task, or --list-codex-queue is required"}


def think_once(*, root: Path | None = None, execute_actuator: bool = False) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    state = load_brain_state(brain_root)
    campaign_report = build_latest_campaign_report()
    autopilot_summary = load_latest_autopilot_summary()
    evidence = observe_latest_evidence(campaign_report=campaign_report, autopilot_summary=autopilot_summary, root=brain_root)
    _update_state_from_evidence(brain_root, evidence, status="planning")
    append_brain_journal(
        "observation",
        "Observed latest local campaign/autopilot evidence.",
        evidence=evidence,
        root=brain_root,
    )
    hypotheses = propose_next_hypotheses(state, evidence)
    record_hypotheses(hypotheses, root=brain_root)
    decision = record_decision(choose_next_action(state, evidence, hypotheses, execute_actuator=execute_actuator), root=brain_root)
    return {
        "ok": True,
        "state": _compact_state(load_brain_state(brain_root)),
        "evidence": evidence,
        "hypotheses": hypotheses,
        "decision": decision,
    }


def run_brain_step(*, root: Path | None = None, execute_actuator: bool = False) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    state = load_brain_state(brain_root)
    if not state.get("current_objective"):
        return {"ok": False, "error": "PEDS brain is not initialized."}
    max_iterations = int(state.get("max_iterations") if state.get("max_iterations") is not None else 1)
    iteration = int(state.get("iteration") or 0)
    if iteration >= max_iterations:
        state["status"] = "stopped"
        state["stop_reason"] = "max_iterations_reached"
        write_brain_state(state, root=brain_root)
        append_brain_journal("stop", "Stopped because max_iterations was reached.", evidence={"max_iterations": max_iterations}, root=brain_root)
        return {"ok": True, "stopped": True, "state": _compact_state(state), "stop_reason": "max_iterations_reached"}
    state["iteration"] = iteration + 1
    state["status"] = "planning"
    write_brain_state(state, root=brain_root)
    thought = think_once(root=brain_root, execute_actuator=execute_actuator)
    decision = thought["decision"]
    execution = execute_decision(decision=decision, evidence=thought["evidence"], hypotheses=thought["hypotheses"], root=brain_root)
    return {
        "ok": bool(thought.get("ok") and execution.get("ok", True)),
        "state": _compact_state(load_brain_state(brain_root)),
        "decision": decision,
        "execution": execution,
    }


def run_brain_loop(*, root: Path | None = None, max_iterations: int = 5, execute_actuator: bool = False) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    limit = max(0, int(max_iterations))
    steps: list[dict[str, Any]] = []
    for _ in range(limit):
        step = run_brain_step(root=brain_root, execute_actuator=execute_actuator)
        steps.append(step)
        decision = step.get("decision") if isinstance(step.get("decision"), dict) else {}
        if step.get("stopped") or decision.get("action") == "stop_and_report":
            break
    return {"ok": all(step.get("ok", False) for step in steps), "steps_run": len(steps), "steps": steps, "state": _compact_state(load_brain_state(brain_root))}


def execute_decision(
    *,
    decision: dict[str, Any],
    evidence: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    brain_root = Path(root) if root is not None else DEFAULT_BRAIN_ROOT
    action = decision.get("action")
    state = load_brain_state(brain_root)
    objective = state.get("current_objective") or "Improve proof_transformer/PEDS."
    if action == "stop_and_report":
        state["status"] = "stopped"
        state["stop_reason"] = decision.get("stop_reason") or decision.get("reason")
        write_brain_state(state, root=brain_root)
        append_brain_journal("stop", str(decision.get("reason") or "PEDS stopped."), decision=decision, root=brain_root)
        return {"ok": True, "action": action, "stopped": True, "stop_reason": state["stop_reason"]}
    if action in CODE_ACTUATOR_ACTIONS:
        hypothesis = _matching_hypothesis(hypotheses, str(decision.get("hypothesis_kind")))
        task = generate_codex_task(decision=decision, evidence=evidence, hypothesis=hypothesis, root=brain_root)
        return {"ok": True, "action": action, "codex_task": task}
    if action == "wait_for_codex_task_result":
        queue_record = None
        if decision.get("task_path"):
            queue_record = ensure_codex_task_queued(
                task_path=str(decision.get("task_path")),
                decision=decision,
                action=str(decision.get("action")),
                root=brain_root,
            )
        state["status"] = "idle"
        write_brain_state(state, root=brain_root)
        append_brain_journal(
            "decision",
            "PEDS is waiting for Codex actuator task result evidence before continuing.",
            evidence={"task_path": decision.get("task_path"), "codex_queue": queue_record},
            decision=decision,
            root=brain_root,
        )
        return {"ok": True, "action": action, "waiting": True, "task_path": decision.get("task_path"), "codex_queue": queue_record}
    if action == "accept_result":
        result = state.get("last_codex_task_result") if isinstance(state.get("last_codex_task_result"), dict) else {}
        if result:
            result = dict(result, brain_evaluation="accepted", evaluated_at=_utc_now())
            state["last_codex_task_result"] = result
        state["status"] = "idle"
        write_brain_state(state, root=brain_root)
        append_brain_journal(
            "decision",
            "PEDS accepted the latest Codex actuator task result.",
            evidence={"codex_task_result": result},
            decision=decision,
            root=brain_root,
        )
        return {"ok": True, "action": action, "accepted": True, "task_result": result}
    if action == "manual_actuator_required":
        state["status"] = "manual_actuator_required"
        state["stop_reason"] = "manual_actuator_required"
        write_brain_state(state, root=brain_root)
        append_brain_journal(
            "decision",
            "PEDS exposed a manual actuator requirement without marking the task successful.",
            evidence={"manual_command": decision.get("manual_command"), "task_path": decision.get("task_path")},
            decision=decision,
            root=brain_root,
        )
        return {"ok": True, "action": action, "manual_actuator_required": True, "manual_command": decision.get("manual_command"), "task_path": decision.get("task_path")}
    if action == "execute_actuator_task":
        try:
            from peds_actuator import run_next_task  # noqa: WPS433 - script-level optional actuator import.
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "action": action, "error": f"could not import peds_actuator: {exc}"}
        result = run_next_task(root=brain_root, project_root=PROJECT_ROOT)
        append_brain_journal(
            "actuator_result",
            "PEDS executed at most one pending actuator task for this brain step.",
            evidence={"actuator_result": result},
            decision=decision,
            root=brain_root,
        )
        return {"ok": bool(result.get("ok", True)), "action": action, "actuator_result": result}

    command = _command_for_action(str(action), str(objective), brain_root)
    if command is None:
        return {"ok": False, "action": action, "error": f"unsupported brain action: {action}"}
    state["status"] = _status_for_action(str(action))
    write_brain_state(state, root=brain_root)
    completed = _run_local_command(command)
    payload = completed.get("json") if isinstance(completed.get("json"), dict) else {}
    _update_state_after_payload(brain_root, str(action), payload)
    event_type = "campaign_result" if "campaign" in str(action) else ("activation" if action == "activate_best_survivor" else "output_generation")
    append_brain_journal(
        event_type,
        f"Executed PEDS brain action {action}.",
        evidence={"command": command, "returncode": completed.get("returncode"), "payload": payload},
        decision=decision,
        root=brain_root,
    )
    if action == "activate_best_survivor" and payload.get("ok"):
        profile_path = ((payload.get("artifacts") or {}).get("survivor_profile") if isinstance(payload.get("artifacts"), dict) else None)
        profile = _read_json_file(profile_path)
        if isinstance(profile, dict):
            if profile_path:
                profile = dict(profile, profile_path=profile_path)
            active = update_active_policy_from_survivor(profile, root=brain_root)
            completed["brain_active_policy"] = active
    return {"ok": completed.get("returncode") == 0, "action": action, "command": command, "result": completed}


def load_latest_autopilot_summary(output_root: Path | None = None) -> dict[str, Any]:
    root = Path(output_root) if output_root is not None else DEFAULT_AUTOPILOT_OUTPUT_ROOT
    if not root.exists():
        return {"ok": False, "found": False, "reason": f"No autopilot outputs found under {root}"}
    candidates = list(root.glob("*/autopilot_summary.json"))
    if not candidates:
        return {"ok": False, "found": False, "reason": f"No autopilot_summary.json found under {root}"}
    latest = max(candidates, key=lambda path: (path.stat().st_mtime, path.parent.name))
    payload = _read_json_file(latest)
    if not isinstance(payload, dict):
        return {"ok": False, "found": False, "reason": f"Invalid autopilot summary JSON: {latest}", "summary_path": str(latest)}
    payload.setdefault("summary_path", str(latest))
    payload.setdefault("found", True)
    return payload


def _command_for_action(action: str, objective: str, brain_root: Path | None = None) -> list[str] | None:
    active_policy_args = _active_policy_args(brain_root)
    if action == "run_campaign_5x2":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_training_campaign.py"),
            "--generations",
            "5",
            "--runs",
            "2",
            "--max-attempts-per-generation",
            "5",
            "--json",
            *active_policy_args,
        ]
    if action == "run_campaign_10x3":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_training_campaign.py"),
            "--generations",
            "10",
            "--runs",
            "3",
            "--max-attempts-per-generation",
            "5",
            "--json",
            *active_policy_args,
        ]
    if action == "run_mutation_stress":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_autopilot.py"),
            "--objective",
            objective,
            "--campaign",
            "mutation-stress",
            "--write",
            "--json",
            *active_policy_args,
        ]
    if action == "run_autopilot_use_latest":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_autopilot.py"),
            "--objective",
            objective,
            "--use-latest",
            "--write",
            "--json",
            *active_policy_args,
        ]
    if action == "activate_best_survivor":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_autopilot.py"),
            "--objective",
            objective,
            "--use-latest",
            "--activate-best",
            "--write",
            "--json",
            *active_policy_args,
        ]
    if action == "generate_useful_outputs":
        return [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "peds_autopilot.py"),
            "--objective",
            objective,
            "--use-latest",
            "--write",
            "--json",
            *active_policy_args,
        ]
    if action == "rerun_validation":
        return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_peds_brain.py"]
    return None


def _active_policy_args(brain_root: Path | None = None) -> list[str]:
    root = Path(brain_root) if brain_root is not None else DEFAULT_BRAIN_ROOT
    policy_path = root / "active_policy.jsonl"
    policy = latest_active_policy(root)
    if isinstance(policy, dict) and policy.get("candidate_id") and policy_path.exists():
        return ["--active-policy-file", str(policy_path)]
    return []


def _run_local_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
    payload: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            decoded = json.loads(completed.stdout)
            payload = decoded if isinstance(decoded, dict) else None
        except json.JSONDecodeError:
            payload = None
    return {
        "returncode": completed.returncode,
        "stdout_preview": completed.stdout[-4000:],
        "stderr_preview": completed.stderr[-4000:],
        "json": payload,
    }


def _update_state_from_evidence(root: Path, evidence: dict[str, Any], *, status: str) -> None:
    state = load_brain_state(root)
    campaign = evidence.get("campaign") if isinstance(evidence.get("campaign"), dict) else {}
    autopilot = evidence.get("autopilot") if isinstance(evidence.get("autopilot"), dict) else {}
    if campaign.get("report_path"):
        state["last_campaign_report"] = campaign.get("report_path")
    if autopilot.get("summary_path") or autopilot.get("output_dir"):
        state["last_autopilot_output"] = autopilot.get("summary_path") or autopilot.get("output_dir")
    active = evidence.get("active_policy") if isinstance(evidence.get("active_policy"), dict) else {}
    if active.get("active_policy_id"):
        state["current_active_policy_id"] = active.get("active_policy_id")
    if active.get("candidate_id"):
        state["current_active_candidate_id"] = active.get("candidate_id")
    codex_task = evidence.get("codex_task") if isinstance(evidence.get("codex_task"), dict) else {}
    if codex_task.get("path"):
        state["last_codex_task"] = codex_task.get("path")
    codex_task_result = evidence.get("codex_task_result") if isinstance(evidence.get("codex_task_result"), dict) else {}
    if codex_task_result.get("found"):
        state["last_codex_task_result"] = codex_task_result
    state["status"] = status
    write_brain_state(state, root=root)


def _update_state_after_payload(root: Path, action: str, payload: dict[str, Any]) -> None:
    state = load_brain_state(root)
    if payload.get("report_path"):
        state["last_campaign_report"] = payload.get("report_path")
    if payload.get("summary_path") or payload.get("output_dir"):
        state["last_autopilot_output"] = payload.get("summary_path") or payload.get("output_dir")
    if payload.get("candidate_id"):
        state["current_active_candidate_id"] = state.get("current_active_candidate_id") or payload.get("candidate_id")
    state["status"] = "idle"
    if action == "activate_best_survivor":
        state["status"] = "activating"
    write_brain_state(state, root=root)


def _status_for_action(action: str) -> str:
    if action.startswith("run_campaign") or action == "run_mutation_stress":
        return "running_campaign"
    if action in {"run_autopilot_use_latest", "generate_useful_outputs"}:
        return "generating_output"
    if action == "activate_best_survivor":
        return "activating"
    return "planning"


def _matching_hypothesis(hypotheses: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    for hypothesis in hypotheses:
        if hypothesis.get("kind") == kind:
            return hypothesis
    return hypotheses[0] if hypotheses else None


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "schema_version",
        "system",
        "seeded_objective",
        "current_objective",
        "current_active_policy_id",
        "current_active_candidate_id",
        "current_focus",
        "iteration",
        "max_iterations",
        "status",
        "last_campaign_report",
        "last_autopilot_output",
        "last_codex_task",
        "last_codex_task_result",
        "last_actuator_run",
        "last_decision_id",
        "next_action",
        "open_questions",
        "stop_reason",
    ]
    return {key: state.get(key) for key in keys}


def _read_json_file(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    try:
        decoded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Operate the local PEDS Brain control loop.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--init", action="store_true")
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--think", action="store_true")
    actions.add_argument("--step", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--record-task-result", type=Path)
    actions.add_argument("--activate-profile", type=Path)
    actions.add_argument("--claim-codex-task", action="store_true")
    actions.add_argument("--list-codex-queue", action="store_true")
    parser.add_argument("--objective")
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--execute-actuator", action="store_true", help="Allow --step/--run to execute at most one pending actuator task per brain step.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())
