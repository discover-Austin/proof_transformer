from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BRAIN_ROOT = Path(__file__).resolve().parents[3] / "temp" / "peds_brain"
SCHEMA_VERSION = "1.0"
SYSTEM_NAME = "PEDS"
SEEDED_OBJECTIVE = (
    "Make proof_transformer/PEDS into a local self-improving repo assistant engine "
    "where the user can enter an objective and receive useful, evidence-backed output "
    "while PEDS handles experiments, correction, survivor selection, activation, and "
    "next-step planning behind the scenes across broad agentic task capabilities."
)

BRAIN_STATE_FILE = "brain_state.json"
BRAIN_JOURNAL_FILE = "brain_journal.jsonl"
GOALS_FILE = "goals.jsonl"
HYPOTHESES_FILE = "hypotheses.jsonl"
DECISIONS_FILE = "decisions.jsonl"
ACTIVE_POLICY_FILE = "active_policy.jsonl"
TASK_QUEUE_FILE = "task_queue.jsonl"
TASK_RESULTS_FILE = "task_results.jsonl"
CODEX_ACTUATOR_QUEUE_FILE = "codex_actuator_queue.jsonl"  # Backward-compatible mirror of task_queue.jsonl.
TERMINAL_TASK_STATES = {"completed", "failed", "manual_actuator_required", "superseded"}

CODE_ACTUATOR_ACTIONS = {
    "inspect_artifact_quality",
    "improve_trainer_signal",
    "improve_gui_use_peds_path",
    "improve_survivor_output_quality",
    "add_active_policy_campaign_start",
    "expand_agentic_capability_curriculum",
    "request_followup",
}


def initialize_brain(
    *,
    objective: str,
    root: Path | None = None,
    max_iterations: int = 10,
) -> dict[str, Any]:
    objective = _require_text(objective, "objective")
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    for filename in (
        BRAIN_JOURNAL_FILE,
        GOALS_FILE,
        HYPOTHESES_FILE,
        DECISIONS_FILE,
        ACTIVE_POLICY_FILE,
        TASK_QUEUE_FILE,
        TASK_RESULTS_FILE,
        CODEX_ACTUATOR_QUEUE_FILE,
    ):
        (brain_root / filename).touch(exist_ok=True)
    (brain_root / "tasks").mkdir(exist_ok=True)
    (brain_root / "actuator_runs").mkdir(exist_ok=True)
    state = {
        "schema_version": SCHEMA_VERSION,
        "system": SYSTEM_NAME,
        "seeded_objective": SEEDED_OBJECTIVE,
        "current_objective": objective,
        "current_active_policy_id": None,
        "current_active_candidate_id": None,
        "current_focus": "establish_evidence_backed_improvement_loop",
        "iteration": 0,
        "max_iterations": int(max_iterations),
        "status": "idle",
        "last_campaign_report": None,
        "last_autopilot_output": None,
        "last_codex_task": None,
        "last_codex_task_result": None,
        "last_actuator_run": None,
        "last_decision_id": None,
        "next_action": None,
        "open_questions": [],
        "stop_reason": None,
    }
    _write_state(state, brain_root)
    goal = {
        "goal_id": _stable_id("goal", objective, _utc_now()),
        "timestamp": _utc_now(),
        "objective": objective,
        "seeded_objective": SEEDED_OBJECTIVE,
        "max_iterations": int(max_iterations),
    }
    _append_jsonl(brain_root / GOALS_FILE, goal)
    append_brain_journal(
        "goal",
        f"Initialized PEDS brain objective: {objective}",
        evidence={"goal_id": goal["goal_id"]},
        root=brain_root,
    )
    return load_brain_state(brain_root)


def load_brain_state(root: Path | None = None) -> dict[str, Any]:
    brain_root = _root(root)
    path = brain_root / BRAIN_STATE_FILE
    if not path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "system": SYSTEM_NAME,
            "seeded_objective": SEEDED_OBJECTIVE,
            "current_objective": "",
            "current_active_policy_id": None,
            "current_active_candidate_id": None,
            "current_focus": "uninitialized",
            "iteration": 0,
            "max_iterations": 10,
            "status": "idle",
            "last_campaign_report": None,
            "last_autopilot_output": None,
            "last_codex_task": None,
            "last_codex_task_result": None,
            "last_actuator_run": None,
            "last_decision_id": None,
            "next_action": None,
            "open_questions": ["Brain has not been initialized."],
            "stop_reason": None,
        }
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "schema_version": SCHEMA_VERSION,
            "system": SYSTEM_NAME,
            "seeded_objective": SEEDED_OBJECTIVE,
            "current_objective": "",
            "current_active_policy_id": None,
            "current_active_candidate_id": None,
            "current_focus": "state_unreadable",
            "iteration": 0,
            "max_iterations": 10,
            "status": "stopped",
            "last_campaign_report": None,
            "last_autopilot_output": None,
            "last_codex_task": None,
            "last_codex_task_result": None,
            "last_actuator_run": None,
            "last_decision_id": None,
            "next_action": None,
            "open_questions": ["Brain state JSON could not be parsed."],
            "stop_reason": "invalid_brain_state_json",
        }
    return _normalize_state(decoded) if isinstance(decoded, dict) else {}


def append_brain_journal(
    event_type: str,
    summary: str,
    *,
    evidence: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    next_action: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    event_type = _require_text(event_type, "event_type")
    summary = _require_text(summary, "summary")
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    state = load_brain_state(brain_root)
    entry = {
        "timestamp": _utc_now(),
        "iteration": _coerce_int(state.get("iteration"), default=0),
        "event_type": event_type,
        "summary": summary,
        "evidence": _compact(deepcopy(evidence or {})),
        "decision": deepcopy(decision) if isinstance(decision, dict) else None,
        "next_action": deepcopy(next_action) if isinstance(next_action, dict) else None,
    }
    _append_jsonl(brain_root / BRAIN_JOURNAL_FILE, entry)
    return entry


def observe_latest_evidence(
    *,
    campaign_report: dict[str, Any] | None = None,
    autopilot_summary: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    brain_root = _root(root)
    state = load_brain_state(brain_root)
    brain_policy = _latest_jsonl(brain_root / ACTIVE_POLICY_FILE)
    campaign = _campaign_evidence(campaign_report)
    autopilot = _autopilot_evidence(autopilot_summary)
    unsafe = _unsafe_reasons(campaign, autopilot)
    active_policy = brain_policy or _state_active_policy(state)
    external_active_policy = autopilot.get("active_policy") or autopilot.get("latest_active_policy")
    useful_artifacts = bool(
        autopilot.get("useful_output_ready")
        or autopilot.get("artifacts_written")
        or autopilot.get("output_dir")
    )
    queue = derive_current_task_queue(root=brain_root)
    latest_task = _state_codex_task_path(state, brain_root) or _latest_queue_task_path(queue) or _latest_codex_task_path(brain_root)
    latest_result = state.get("last_codex_task_result") if isinstance(state.get("last_codex_task_result"), dict) else load_latest_task_result(root=brain_root)
    if not isinstance(latest_result, dict):
        latest_result = _queue_result_for_task(queue, latest_task)
    task_result = _codex_task_result_evidence(latest_result, latest_task=latest_task)
    artifact_quality = _artifact_quality_evidence(
        autopilot,
        useful_artifacts=useful_artifacts,
        latest_task=latest_task,
        task_result=task_result,
    )
    agentic_capability = _apply_agentic_task_result(
        _agentic_capability_evidence(autopilot),
        task_result,
    )
    actuator_provider_status = _actuator_provider_evidence(state)
    evidence = {
        "observed_at": _utc_now(),
        "brain_state": {
            "iteration": state.get("iteration", 0),
            "current_objective": state.get("current_objective", ""),
            "current_active_policy_id": state.get("current_active_policy_id"),
            "current_active_candidate_id": state.get("current_active_candidate_id"),
            "status": state.get("status", "idle"),
        },
        "campaign": campaign,
        "autopilot": autopilot,
        "active_policy": {
            "found": isinstance(active_policy, dict) and bool(active_policy.get("candidate_id")),
            "active_policy_id": active_policy.get("active_policy_id") if isinstance(active_policy, dict) else None,
            "candidate_id": active_policy.get("candidate_id") if isinstance(active_policy, dict) else state.get("current_active_candidate_id"),
            "source": "brain" if brain_policy else ("brain_state" if isinstance(active_policy, dict) else None),
        },
        "external_active_policy": {
            "found": isinstance(external_active_policy, dict) and bool(external_active_policy.get("candidate_id")),
            "active_policy_id": external_active_policy.get("active_policy_id") if isinstance(external_active_policy, dict) else None,
            "candidate_id": external_active_policy.get("candidate_id") if isinstance(external_active_policy, dict) else None,
            "source": "autopilot" if isinstance(external_active_policy, dict) else None,
        },
        "has_campaign_report": bool(campaign.get("found")),
        "has_survivor": bool(autopilot.get("survivor_selected") or autopilot.get("candidate_id")),
        "has_active_policy": bool(isinstance(active_policy, dict) and active_policy.get("candidate_id")),
        "useful_artifacts": useful_artifacts,
        "codex_task": {
            "found": latest_task is not None,
            "path": str(latest_task) if latest_task is not None else None,
            "pending_result": latest_task is not None and not task_result.get("found"),
            "task_id": _task_id_for_path(queue, latest_task),
        },
        "task_queue": {
            "current": queue,
            "pending": [item for item in queue if item.get("status") == "pending"],
            "claimed": [item for item in queue if item.get("status") == "claimed"],
            "running": [item for item in queue if item.get("status") == "running"],
            "completed": [item for item in queue if item.get("status") == "completed"],
            "failed": [item for item in queue if item.get("status") == "failed"],
            "manual_actuator_required": [item for item in queue if item.get("status") == "manual_actuator_required"],
        },
        "active_policy_usage": _active_policy_usage_evidence(active_policy),
        "codex_task_result": task_result,
        "actuator_providers": actuator_provider_status,
        "artifact_quality": artifact_quality,
        "agentic_capability": agentic_capability,
        "artifact_quality_weak": artifact_quality.get("status") == "weak",
        "artifact_quality_unknown": artifact_quality.get("status") == "unknown",
        "safety": {"clean": not unsafe, "unsafe_reasons": unsafe},
    }
    return evidence


def propose_next_hypotheses(brain_state: dict[str, Any], evidence: dict[str, Any]) -> list[dict[str, Any]]:
    campaign = evidence.get("campaign") if isinstance(evidence.get("campaign"), dict) else {}
    autopilot = evidence.get("autopilot") if isinstance(evidence.get("autopilot"), dict) else {}
    hypotheses = [
        _hypothesis(
            brain_state,
            "evidence_integrity",
            "PEDS must keep campaign and safety evidence current before selecting or activating survivors.",
            "Without clean evidence, survivor promotion becomes noise.",
            {
                "campaign_found": evidence.get("has_campaign_report", False),
                "safety_clean": (evidence.get("safety") or {}).get("clean", True),
            },
            {"action": "run_campaign_5x2" if not evidence.get("has_campaign_report") else "run_autopilot_use_latest"},
            "Current evidence summary and a safe next action.",
        ),
        _hypothesis(
            brain_state,
            "active_policy",
            "A selected survivor should become an explicit active PEDS policy before it guides future work.",
            "PEDS needs durable policy memory, not just a one-off prompt.",
            {
                "survivor_selected": evidence.get("has_survivor", False),
                "active_policy_found": evidence.get("has_active_policy", False),
                "candidate_id": autopilot.get("candidate_id"),
            },
            {"action": "activate_best_survivor"},
            "Append-only active policy record tied to survivor evidence.",
        ),
        _hypothesis(
            brain_state,
            "mutation_diversity",
            "High correction success with narrow mutation diversity should be stress-tested before scaling trust.",
            "A single successful mutation family can overfit local evidence.",
            {
                "correction_success_rate": campaign.get("correction_success_rate"),
                "mutation_diversity": campaign.get("mutation_diversity"),
            },
            {"action": "run_mutation_stress"},
            "Mutation-diversity evidence that either confirms or challenges the current survivor family.",
        ),
        _hypothesis(
            brain_state,
            "artifact_quality",
            "Useful outputs should be concrete enough for an actuator to execute and for PEDS to evaluate later.",
            "Weak artifacts leave the loop centered on explanation instead of measurable work.",
            {
                "useful_artifacts": evidence.get("useful_artifacts", False),
                "artifact_quality_status": (evidence.get("artifact_quality") or {}).get("status"),
                "artifact_quality_weak": evidence.get("artifact_quality_weak", False),
                "artifact_quality_unknown": evidence.get("artifact_quality_unknown", False),
            },
            {"action": "inspect_artifact_quality"},
            "A PEDS-decided Codex task artifact with files, constraints, validation, and report contract.",
        ),
        _hypothesis(
            brain_state,
            "agentic_capability_curriculum",
            "PEDS must improve broad agentic task capabilities, not only its own improvement mechanics.",
            "Self-improvement evidence can overfit the loop unless distinct agentic skills get local verifier pressure.",
            {
                "coverage_found": (evidence.get("agentic_capability") or {}).get("found", False),
                "coverage_status": (evidence.get("agentic_capability") or {}).get("status"),
                "untested_capabilities": (evidence.get("agentic_capability") or {}).get("untested_capabilities", []),
            },
            {"action": "expand_agentic_capability_curriculum"},
            "A local curriculum/eval task that pressures untested agentic capabilities with evidence.",
        ),
        _hypothesis(
            brain_state,
            "trainer_signal",
            "Trainer signal should remain auditable and tied to survivor behavior rather than metadata shortcuts.",
            "The learning loop only compounds if training evidence remains real.",
            {
                "trained_rate": campaign.get("trained_rate"),
                "adapter_loaded_rate": campaign.get("adapter_loaded_rate"),
            },
            {"action": "improve_trainer_signal"},
            "A focused actuator task when trainer signal becomes the limiting factor.",
        ),
        _hypothesis(
            brain_state,
            "gui_usability",
            "The GUI should operate the PEDS brain while leaving decision ownership inside PEDS.",
            "The interface is the face, not the core control loop.",
            {
                "brain_status": brain_state.get("status"),
                "next_action": brain_state.get("next_action"),
            },
            {"action": "improve_gui_use_peds_path"},
            "A GUI task only when evidence shows the interface blocks the PEDS brain loop.",
        ),
    ]
    return hypotheses


def choose_next_action(
    brain_state: dict[str, Any],
    evidence: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    *,
    execute_actuator: bool = False,
) -> dict[str, Any]:
    _ = hypotheses
    safety = evidence.get("safety") if isinstance(evidence.get("safety"), dict) else {"clean": True}
    campaign = evidence.get("campaign") if isinstance(evidence.get("campaign"), dict) else {}
    iteration = _coerce_int(brain_state.get("iteration"), default=0)
    max_iterations = _coerce_int(brain_state.get("max_iterations"), default=10)
    if max_iterations >= 0 and iteration >= max_iterations:
        return _action("stop_and_report", "PEDS reached max_iterations.", "evidence_integrity", stop_reason="max_iterations_reached")
    if not evidence.get("has_campaign_report"):
        return _action("run_campaign_5x2", "No campaign report exists yet.", "evidence_integrity")
    if not safety.get("clean", True):
        return _action("stop_and_report", "Safety evidence is not clean.", "evidence_integrity", stop_reason="unsafe_evidence")
    if not evidence.get("has_survivor"):
        return _action("run_autopilot_use_latest", "Campaign evidence exists but no survivor is selected.", "active_policy")
    if not evidence.get("has_active_policy"):
        return _action("activate_best_survivor", "A survivor exists but PEDS has no active policy record.", "active_policy")

    task_result = evidence.get("codex_task_result") if isinstance(evidence.get("codex_task_result"), dict) else {}
    if task_result.get("found") and not task_result.get("brain_evaluation"):
        return _action_for_task_result(task_result)

    codex_task = evidence.get("codex_task") if isinstance(evidence.get("codex_task"), dict) else {}
    task_queue = evidence.get("task_queue") if isinstance(evidence.get("task_queue"), dict) else {}
    pending_tasks = task_queue.get("pending") if isinstance(task_queue.get("pending"), list) else []
    if codex_task.get("pending_result"):
        if execute_actuator and pending_tasks:
            next_task = sorted(pending_tasks, key=lambda row: str(row.get("timestamp") or ""))[0]
            return _action(
                "execute_actuator_task",
                "PEDS has a pending actuator task and execution was explicitly requested for this bounded step.",
                "artifact_quality",
                task_id=next_task.get("task_id") or next_task.get("queue_id"),
                task_path=next_task.get("task_path") or codex_task.get("path"),
            )
        return _action(
            "wait_for_codex_task_result",
            "PEDS has issued a Codex actuator task and no result has been recorded yet.",
            "artifact_quality",
            task_id=codex_task.get("task_id"),
            task_path=codex_task.get("path"),
        )

    if not evidence.get("useful_artifacts"):
        return _action("generate_useful_outputs", "An active policy exists but useful artifacts are missing.", "artifact_quality")

    artifact_quality = evidence.get("artifact_quality") if isinstance(evidence.get("artifact_quality"), dict) else {}
    if artifact_quality.get("status") in {"weak", "unknown"}:
        return _action("inspect_artifact_quality", f"Generated artifact quality is {artifact_quality.get('status')}; PEDS needs an actuator inspection task.", "artifact_quality")

    active_usage = evidence.get("active_policy_usage") if isinstance(evidence.get("active_policy_usage"), dict) else {}
    active_policy_start_supported = bool(
        active_usage.get("campaign_start_supported") and active_usage.get("autopilot_start_supported")
    )
    if active_usage.get("needs_codex_task") and not active_policy_start_supported and not _has_action_task(evidence, "add_active_policy_campaign_start"):
        return _action(
            "add_active_policy_campaign_start",
            "An active policy exists, but campaign/autopilot start-from-active-policy is not wired; PEDS must not fake usage.",
            "active_policy",
        )

    agentic_capability = evidence.get("agentic_capability") if isinstance(evidence.get("agentic_capability"), dict) else {}
    if (
        agentic_capability.get("needs_curriculum_task")
        and not _has_action_task(evidence, "expand_agentic_capability_curriculum")
    ):
        return _action(
            "expand_agentic_capability_curriculum",
            "Survivor evidence is still narrow; PEDS needs local eval pressure across broader agentic capabilities.",
            "agentic_capability_curriculum",
        )

    if _high_correction_success(campaign) and _narrow_mutation_diversity(campaign):
        return _action("run_mutation_stress", "Correction success is high but mutation diversity is narrow.", "mutation_diversity")
    if _trainer_signal_weak(campaign):
        return _action("improve_trainer_signal", "Trainer/adapter signal is below the desired local evidence threshold.", "trainer_signal")
    return _action(
        "stop_and_report",
        "Acceptance criteria are met or a bounded actuator task is already available for the next improvement.",
        "evidence_integrity",
        stop_reason="success",
    )


def record_decision(
    decision: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError("decision must be a dict")
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    state = load_brain_state(brain_root)
    decision_id = decision.get("decision_id") or _stable_id(
        "decision",
        state.get("iteration", 0),
        decision.get("action"),
        decision.get("reason"),
    )
    record = {
        "decision_id": decision_id,
        "timestamp": _utc_now(),
        "iteration": _coerce_int(state.get("iteration"), default=0),
        **deepcopy(decision),
    }
    record["decision_id"] = decision_id
    _append_jsonl(brain_root / DECISIONS_FILE, record)
    state["last_decision_id"] = decision_id
    state["next_action"] = deepcopy(record)
    state["current_focus"] = record.get("hypothesis_kind") or state.get("current_focus")
    state["status"] = "planning"
    if record.get("action") != "stop_and_report":
        state["stop_reason"] = None
    _write_state(state, brain_root)
    append_brain_journal(
        "decision",
        f"PEDS chose next action: {record.get('action')}",
        decision=record,
        next_action=record,
        root=brain_root,
    )
    return record


def record_hypotheses(
    hypotheses: list[dict[str, Any]],
    *,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        record = {"timestamp": _utc_now(), **deepcopy(hypothesis)}
        _append_jsonl(brain_root / HYPOTHESES_FILE, record)
        records.append(record)
    if records:
        append_brain_journal(
            "hypothesis",
            f"Recorded {len(records)} deterministic PEDS hypotheses.",
            evidence={"hypothesis_ids": [item.get("hypothesis_id") for item in records]},
            root=brain_root,
        )
    return records


def update_active_policy_from_survivor(
    profile: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(profile, dict) or not profile.get("candidate_id"):
        raise ValueError("profile must include candidate_id")
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    record = {
        "active_policy_id": _stable_id("brain_policy", profile.get("candidate_id"), _utc_now()),
        "timestamp": _utc_now(),
        "candidate_id": profile.get("candidate_id"),
        "objective": profile.get("objective"),
        "selection_reason": profile.get("selection_reason"),
        "profile_path": profile.get("profile_path"),
        "safety": deepcopy(profile.get("safety") if isinstance(profile.get("safety"), dict) else {}),
        "evidence_summary": deepcopy(profile.get("evidence_summary") if isinstance(profile.get("evidence_summary"), dict) else {}),
    }
    _append_jsonl(brain_root / ACTIVE_POLICY_FILE, record)
    state = load_brain_state(brain_root)
    state["current_active_policy_id"] = record["active_policy_id"]
    state["current_active_candidate_id"] = record["candidate_id"]
    state["status"] = "idle"
    _write_state(state, brain_root)
    append_brain_journal(
        "activation",
        f"Activated survivor policy for candidate {record['candidate_id']}.",
        evidence={"active_policy_id": record["active_policy_id"], "candidate_id": record["candidate_id"]},
        root=brain_root,
    )
    return record


def generate_codex_task(
    *,
    decision: dict[str, Any],
    evidence: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    brain_root = _root(root)
    task_dir = brain_root / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    state = load_brain_state(brain_root)
    action = decision.get("action")
    task_timestamp = _task_run_id()
    task_id = _stable_id("task", task_timestamp, action, decision.get("decision_id"), state.get("current_objective"))
    path = task_dir / f"{task_timestamp}_{_safe_name(str(action))}_{task_id}_codex_task.md"
    hypothesis = hypothesis or {}
    likely_files = _likely_files_for_action(str(action))
    validation = [
        "python -m unittest discover -s tests -p 'test_peds_brain.py'",
        "python -m unittest discover -s tests -p 'test_peds_dashboard.py'",
        "python -m unittest discover -s tests -p 'test_peds_*.py'",
    ]
    content = "\n".join(
        [
            "# PEDS-Decided Codex Task",
            "",
            "## Brain objective",
            str(state.get("current_objective") or ""),
            "",
            "## Evidence observed",
            json.dumps(_compact(evidence), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "## Hypothesis",
            json.dumps(_compact(hypothesis), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "## Decision",
            json.dumps(_compact(decision), ensure_ascii=False, indent=2, sort_keys=True),
            "",
            "## Task",
            _task_text_for_action(str(action)),
            "",
            "## Files likely touched",
            *(f"- {item}" for item in likely_files),
            "",
            "## Constraints",
            "- Preserve replay/canary/shadow/promotion/training gates.",
            "- Do not fake winners or bypass rejected states.",
            "- Do not add network/model API calls.",
            "- Do not read secrets.",
            "- Keep Codex as actuator; PEDS owns the decision.",
            "",
            "## Validation commands",
            *(f"- {item}" for item in validation),
            "",
            "## Expected useful output",
            str(hypothesis.get("expected_useful_output") or "A verified local improvement artifact PEDS can evaluate."),
            "",
            "## Final report contract",
            "Return Summary, Files changed, Validation, Evidence observed, Decision honored, and Remaining risks.",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")
    queue_record = ensure_codex_task_queued(
        task_path=path,
        decision=decision,
        action=str(action),
        task_id=task_id,
        root=brain_root,
    )
    state["last_codex_task"] = str(path)
    state["last_codex_task_result"] = None
    state["status"] = "idle"
    _write_state(state, brain_root)
    append_brain_journal(
        "decision",
        f"Generated Codex actuator task for {action}.",
        evidence={"task_path": str(path)},
        decision=decision,
        next_action=decision,
        root=brain_root,
    )
    return {
        "task_id": task_id,
        "task_path": str(path),
        "action": action,
        "queue_id": queue_record.get("queue_id"),
        "queue_status": queue_record.get("status"),
        "likely_files": likely_files,
        "validation_commands": validation,
    }


def write_codex_task_artifact(
    *,
    decision: dict[str, Any],
    evidence: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    return generate_codex_task(decision=decision, evidence=evidence, hypothesis=hypothesis, root=root)


def create_codex_task(
    *,
    decision: dict[str, Any],
    evidence: dict[str, Any],
    hypothesis: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    return generate_codex_task(decision=decision, evidence=evidence, hypothesis=hypothesis, root=root)


def record_codex_task_result(
    result_path: str | Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    brain_root = _root(root)
    path = Path(result_path)
    if not path.exists():
        raise FileNotFoundError(f"Codex task result does not exist: {path}")
    raw = path.read_text(encoding="utf-8", errors="replace")
    decoded: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw)
        decoded = parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        decoded = None
    state = load_brain_state(brain_root)
    latest_task = state.get("last_codex_task") or str(_latest_codex_task_path(brain_root) or "")
    task_path = str((decoded or {}).get("task_path") or latest_task)
    task_id = str((decoded or {}).get("task_id") or _task_id_for_path(derive_current_task_queue(root=brain_root), Path(task_path)) or _stable_id("task", task_path))
    status = _infer_task_result_status(raw, decoded)
    ok = _infer_task_result_ok(raw, decoded)
    if status == "manual_actuator_required":
        ok = False
    record = {
        "task_result_id": _stable_id("task_result", path.resolve(), _utc_now()),
        "timestamp": _utc_now(),
        "path": str(path),
        "task_id": task_id,
        "task_path": task_path,
        "summary": _task_result_summary(raw, decoded),
        "status": status,
        "ok": ok,
        "evidence": _compact(decoded or {}),
    }
    _append_jsonl(brain_root / TASK_RESULTS_FILE, record)
    state["last_codex_task"] = task_path or state.get("last_codex_task")
    state["last_codex_task_result"] = record
    state["last_actuator_run"] = (decoded or {}).get("run_dir") or (decoded or {}).get("result_path") or str(path)
    state["status"] = "evaluating"
    _write_state(state, brain_root)
    queue_record = complete_codex_task_for_result(record, root=brain_root)
    append_brain_journal(
        "observation",
        "Recorded Codex actuator task result for PEDS evaluation.",
        evidence={"codex_task_result": record, "task_queue": queue_record},
        root=brain_root,
    )
    return record


def ensure_codex_task_queued(
    *,
    task_path: str | Path,
    decision: dict[str, Any] | None = None,
    action: str | None = None,
    task_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    brain_root = _root(root)
    brain_root.mkdir(parents=True, exist_ok=True)
    (brain_root / "tasks").mkdir(exist_ok=True)
    (brain_root / "actuator_runs").mkdir(exist_ok=True)
    path = str(Path(task_path))
    for item in derive_current_task_queue(root=brain_root):
        if item.get("task_path") == path and item.get("status") != "superseded":
            return item
    decision = decision if isinstance(decision, dict) else {}
    task_id = task_id or _stable_id("task", path, action or decision.get("action"), _utc_now())
    record = {
        "task_id": task_id,
        "queue_id": task_id,
        "timestamp": _utc_now(),
        "event": "queued",
        "status": "pending",
        "task_path": path,
        "action": action or decision.get("action"),
        "decision_id": decision.get("decision_id"),
        "hypothesis_kind": decision.get("hypothesis_kind"),
        "reason": decision.get("reason"),
    }
    _append_task_event(brain_root, record)
    append_brain_journal(
        "decision",
        f"Queued Codex actuator task: {record.get('action') or path}",
        evidence={"task_queue": record},
        decision=decision,
        root=brain_root,
    )
    return record


def claim_next_codex_task(*, root: Path | None = None) -> dict[str, Any] | None:
    brain_root = _root(root)
    pending = [
        item
        for item in derive_current_task_queue(root=brain_root)
        if item.get("status") == "pending" and item.get("task_path")
    ]
    if not pending:
        return None
    item = sorted(pending, key=lambda row: str(row.get("timestamp") or ""))[0]
    record = {
        **item,
        "timestamp": _utc_now(),
        "event": "claimed",
        "status": "claimed",
        "claimed_by": "codex",
    }
    _append_task_event(brain_root, record)
    state = load_brain_state(brain_root)
    state["last_codex_task"] = record.get("task_path") or state.get("last_codex_task")
    state["status"] = "idle"
    _write_state(state, brain_root)
    append_brain_journal(
        "decision",
        f"Codex claimed PEDS actuator task {record.get('task_id') or record.get('queue_id')}.",
        evidence={"task_queue": record},
        root=brain_root,
    )
    return record


def mark_task_running(
    task: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
    task_path: str | Path | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    brain_root = _root(root)
    item = dict(task) if isinstance(task, dict) else None
    if item is None:
        for candidate in derive_current_task_queue(root=brain_root):
            if (task_id and candidate.get("task_id") == task_id) or (task_path and candidate.get("task_path") == str(Path(task_path))):
                item = candidate
                break
    if item is None:
        return None
    record = {
        **item,
        "timestamp": _utc_now(),
        "event": "running",
        "status": "running",
    }
    _append_task_event(brain_root, record)
    state = load_brain_state(brain_root)
    state["last_codex_task"] = record.get("task_path") or state.get("last_codex_task")
    state["status"] = "running_actuator"
    _write_state(state, brain_root)
    append_brain_journal("decision", f"Marked PEDS actuator task {record.get('task_id')} running.", evidence={"task_queue": record}, root=brain_root)
    return record


def complete_codex_task_for_result(
    result: dict[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    brain_root = _root(root)
    task_path = result.get("task_path")
    task_id = result.get("task_id")
    if not task_path and not task_id:
        return None
    matching = [
        item
        for item in derive_current_task_queue(root=brain_root)
        if ((task_id and item.get("task_id") == task_id) or (task_path and item.get("task_path") == task_path))
        and item.get("status") != "superseded"
    ]
    if not matching:
        return None
    item = sorted(matching, key=lambda row: str(row.get("timestamp") or ""))[0]
    status = str(result.get("status") or "").lower()
    if status not in TERMINAL_TASK_STATES:
        status = "completed" if result.get("ok") is True else "failed"
    record = {
        **item,
        "timestamp": _utc_now(),
        "event": status,
        "status": status,
        "task_result_id": result.get("task_result_id"),
        "result_path": result.get("path"),
        "summary": result.get("summary"),
    }
    _append_task_event(brain_root, record)
    return record


def derive_current_task_queue(*, root: Path | None = None) -> list[dict[str, Any]]:
    return sorted(
        _task_queue_status(_root(root)).values(),
        key=lambda row: str(row.get("timestamp") or ""),
    )


def list_task_queue(*, root: Path | None = None) -> list[dict[str, Any]]:
    return derive_current_task_queue(root=root)


def list_codex_actuator_queue(*, root: Path | None = None) -> list[dict[str, Any]]:
    return derive_current_task_queue(root=root)


def load_latest_task_result(root: Path | None = None) -> dict[str, Any] | None:
    return _latest_jsonl(_root(root) / TASK_RESULTS_FILE)


def evaluate_task_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return _action("rerun_validation", "No task result evidence exists to accept.", "artifact_quality")
    return _action_for_task_result({"found": True, **result})


def write_brain_state(state: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    brain_root = _root(root)
    _write_state(state, brain_root)
    return load_brain_state(brain_root)


def latest_decision(root: Path | None = None) -> dict[str, Any] | None:
    return _latest_jsonl(_root(root) / DECISIONS_FILE)


def latest_active_policy(root: Path | None = None) -> dict[str, Any] | None:
    return _latest_jsonl(_root(root) / ACTIVE_POLICY_FILE)


def _actuator_provider_evidence(state: dict[str, Any]) -> dict[str, Any]:
    try:
        from proof_transformer.peds.actuator_providers import detect_actuator_providers

        detected = detect_actuator_providers()
    except Exception as exc:  # noqa: BLE001 - provider evidence must not break brain observation.
        detected = {
            "providers": [
                {
                    "name": "manual",
                    "available": True,
                    "supports_noninteractive": False,
                    "probe_commands": [],
                    "reason": f"provider_detection_failed:{exc}",
                }
            ],
            "selected_provider": "manual",
        }
    providers = detected.get("providers") if isinstance(detected.get("providers"), list) else []
    noninteractive = [item for item in providers if isinstance(item, dict) and item.get("available") and item.get("supports_noninteractive")]
    latest_result = state.get("last_codex_task_result") if isinstance(state.get("last_codex_task_result"), dict) else {}
    evidence = latest_result.get("evidence") if isinstance(latest_result.get("evidence"), dict) else {}
    return {
        "providers": providers,
        "selected_provider": detected.get("selected_provider") or "manual",
        "noninteractive_available": bool(noninteractive),
        "available_provider_names": [str(item.get("name")) for item in providers if isinstance(item, dict) and item.get("available")],
        "latest_result_provider": latest_result.get("provider") or evidence.get("provider"),
    }


def _active_policy_usage_evidence(active_policy: dict[str, Any] | None) -> dict[str, Any]:
    campaign_supported = _script_supports_arg("peds_training_campaign.py", "--active-policy-file")
    autopilot_supported = _script_supports_arg("peds_autopilot.py", "--active-policy-file")
    has_active_policy = bool(isinstance(active_policy, dict) and active_policy.get("candidate_id"))
    return {
        "campaign_start_supported": campaign_supported,
        "autopilot_start_supported": autopilot_supported,
        "source": "script_arg_probe" if campaign_supported or autopilot_supported else "not_wired_yet",
        "needs_codex_task": has_active_policy and not (campaign_supported and autopilot_supported),
    }


def _script_supports_arg(script_name: str, flag: str) -> bool:
    script_path = Path(__file__).resolve().parents[3] / "scripts" / script_name
    try:
        return flag in script_path.read_text(encoding="utf-8")
    except OSError:
        return False


def _campaign_evidence(report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get("found", bool(report.get("campaign_id"))):
        return {"found": False, "reason": (report or {}).get("reason") if isinstance(report, dict) else "missing"}
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    safety = report.get("safety") if isinstance(report.get("safety"), dict) else {}
    mutations = report.get("mutations") if isinstance(report.get("mutations"), list) else []
    mutation_targets = sorted(
        {
            str((item.get("mutation") if isinstance(item.get("mutation"), dict) else item).get("mutation_target"))
            for item in mutations
            if isinstance(item, dict)
            and (item.get("mutation") if isinstance(item.get("mutation"), dict) else item).get("mutation_target")
        }
    )
    return {
        "found": True,
        "campaign_id": report.get("campaign_id"),
        "report_path": report.get("report_path"),
        "campaign_root": report.get("campaign_root"),
        "total_generations_completed": aggregate.get("total_generations_completed", 0),
        "total_candidate_attempts": aggregate.get("total_candidate_attempts", 0),
        "total_winning_attempts": aggregate.get("total_winning_attempts", 0),
        "total_rejected_attempts": aggregate.get("total_rejected_attempts", 0),
        "corrections_attempted": aggregate.get("corrections_attempted", 0),
        "corrections_followed_by_winner": aggregate.get("corrections_followed_by_winner", 0),
        "correction_success_rate": aggregate.get("correction_success_rate", 0.0),
        "collapse_detected": bool(aggregate.get("collapse_detected", False)),
        "trained_rate": aggregate.get("trained_rate"),
        "adapter_loaded_rate": aggregate.get("adapter_loaded_rate"),
        "mutation_targets": mutation_targets,
        "mutation_diversity": len(mutation_targets),
        "safety": deepcopy(safety),
    }


def _autopilot_evidence(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, dict) or not summary.get("ok", summary.get("found", False)):
        return {"found": False, "reason": (summary or {}).get("error") if isinstance(summary, dict) else "missing"}
    active_policy = summary.get("active_policy") if isinstance(summary.get("active_policy"), dict) else None
    latest_policy = summary.get("latest_active_policy") if isinstance(summary.get("latest_active_policy"), dict) else None
    artifacts = summary.get("artifacts") if isinstance(summary.get("artifacts"), dict) else {}
    return {
        "found": True,
        "summary_path": summary.get("summary_path"),
        "output_dir": summary.get("output_dir"),
        "candidate_id": summary.get("candidate_id"),
        "survivor_selected": bool(summary.get("survivor_selected")),
        "active_policy": deepcopy(active_policy),
        "latest_active_policy": deepcopy(latest_policy),
        "useful_output_ready": bool(summary.get("useful_output_ready")),
        "artifacts_written": bool(summary.get("artifacts_written", bool(artifacts))),
        "artifacts": deepcopy(artifacts),
        "codex_prompt_preview": summary.get("codex_prompt_preview") or "",
        "what_improved": deepcopy(summary.get("what_improved") if isinstance(summary.get("what_improved"), list) else []),
        "agentic_capability_coverage": deepcopy(
            summary.get("agentic_capability_coverage") if isinstance(summary.get("agentic_capability_coverage"), dict) else {}
        ),
        "agentic_next_eval_tasks": deepcopy(
            summary.get("agentic_next_eval_tasks") if isinstance(summary.get("agentic_next_eval_tasks"), list) else []
        ),
        "safety": deepcopy(summary.get("safety") if isinstance(summary.get("safety"), dict) else {}),
    }


def _unsafe_reasons(campaign: dict[str, Any], autopilot: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for label, payload in (("campaign", campaign.get("safety")), ("autopilot", autopilot.get("safety"))):
        if not isinstance(payload, dict):
            continue
        if payload.get("network_attempts"):
            reasons.append(f"{label}:network_attempts")
        if payload.get("model_modules_loaded"):
            reasons.append(f"{label}:model_modules_loaded")
        if payload.get("fake_weights_written"):
            reasons.append(f"{label}:fake_weights_written")
    return reasons


def _state_active_policy(state: dict[str, Any]) -> dict[str, Any] | None:
    if state.get("current_active_candidate_id"):
        return {
            "active_policy_id": state.get("current_active_policy_id"),
            "candidate_id": state.get("current_active_candidate_id"),
        }
    return None


def _artifact_quality_weak(autopilot: dict[str, Any]) -> bool:
    if not autopilot.get("found"):
        return False
    preview = str(autopilot.get("codex_prompt_preview") or "")
    return bool(autopilot.get("useful_output_ready")) and len(preview.strip()) < 240


def _artifact_quality_evidence(
    autopilot: dict[str, Any],
    *,
    useful_artifacts: bool,
    latest_task: Path | None,
    task_result: dict[str, Any],
) -> dict[str, Any]:
    preview = str(autopilot.get("codex_prompt_preview") or "")
    artifacts = autopilot.get("artifacts") if isinstance(autopilot.get("artifacts"), dict) else {}
    if not useful_artifacts:
        status = "missing"
    elif _artifact_quality_weak(autopilot):
        status = "weak"
    elif task_result.get("found"):
        status = "task_result_recorded"
    elif latest_task is not None:
        status = "task_generated"
    else:
        status = "unknown"
    return {
        "status": status,
        "prompt_preview_length": len(preview.strip()),
        "artifact_count": len(artifacts),
        "latest_task_path": str(latest_task) if latest_task is not None else None,
        "latest_task_result_id": task_result.get("task_result_id"),
    }


def _agentic_capability_evidence(autopilot: dict[str, Any]) -> dict[str, Any]:
    if not autopilot.get("found"):
        return {
            "found": False,
            "status": "missing_autopilot",
            "pressured_capabilities": [],
            "untested_capabilities": [],
            "next_eval_tasks": [],
            "needs_curriculum_task": False,
        }
    coverage = autopilot.get("agentic_capability_coverage")
    if not isinstance(coverage, dict) or not coverage:
        return {
            "found": False,
            "status": "not_reported",
            "pressured_capabilities": [],
            "untested_capabilities": [],
            "next_eval_tasks": [],
            "needs_curriculum_task": False,
        }
    untested = coverage.get("untested_capabilities") if isinstance(coverage.get("untested_capabilities"), list) else []
    next_tasks = coverage.get("next_eval_tasks") if isinstance(coverage.get("next_eval_tasks"), list) else []
    return {
        "found": True,
        "status": coverage.get("status") or ("broadening_needed" if untested else "broad_curriculum_pressured"),
        "intent": coverage.get("intent"),
        "coverage_warning": coverage.get("coverage_warning"),
        "pressured_capabilities": deepcopy(
            coverage.get("pressured_capabilities") if isinstance(coverage.get("pressured_capabilities"), list) else []
        ),
        "untested_capabilities": deepcopy(untested),
        "next_eval_tasks": deepcopy(next_tasks),
        "needs_curriculum_task": bool(untested),
    }


def _apply_agentic_task_result(agentic_capability: dict[str, Any], task_result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(task_result, dict) or not task_result.get("found"):
        return agentic_capability
    if task_result.get("ok") is not True and str(task_result.get("status") or "").lower() not in {"completed", "accepted", "success"}:
        return agentic_capability
    evidence = task_result.get("evidence") if isinstance(task_result.get("evidence"), dict) else {}
    observed = evidence.get("evidence_observed") if isinstance(evidence.get("evidence_observed"), dict) else {}
    if observed.get("agentic_capability_action") != "expand_agentic_capability_curriculum":
        return agentic_capability
    if observed.get("dashboard_eval_passed") is not True:
        return agentic_capability

    updated = deepcopy(agentic_capability)
    pressured = {
        str(item)
        for item in updated.get("pressured_capabilities", [])
        if item not in (None, "")
    }
    pressured.add("ui_or_browser_operation")
    untested = [
        str(item)
        for item in updated.get("untested_capabilities", [])
        if item not in (None, "", "ui_or_browser_operation")
    ]
    updated["found"] = True
    updated["pressured_capabilities"] = sorted(pressured)
    updated["untested_capabilities"] = untested
    updated["next_eval_tasks"] = [] if not untested else updated.get("next_eval_tasks", [])
    updated["status"] = "broad_curriculum_pressured" if not untested else "broadening_needed"
    updated["needs_curriculum_task"] = bool(untested)
    updated["latest_curriculum_task_result_id"] = task_result.get("task_result_id")
    updated["latest_curriculum_evidence"] = "accepted local dashboard/UI-browser capability eval"
    return updated


def _state_codex_task_path(state: dict[str, Any], root: Path) -> Path | None:
    value = state.get("last_codex_task")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None


def _latest_codex_task_path(root: Path) -> Path | None:
    task_dir = root / "tasks"
    if not task_dir.exists():
        return None
    candidates = list(task_dir.glob("*_codex_task.md"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _codex_task_result_evidence(result: Any, *, latest_task: Path | None) -> dict[str, Any]:
    default = {
        "found": False,
        "task_id": None,
        "task_path": str(latest_task) if latest_task is not None else None,
        "brain_evaluation": None,
    }
    if not isinstance(result, dict):
        return default
    result_task_path = result.get("task_path")
    if latest_task is not None and result_task_path and str(result_task_path) != str(latest_task):
        return default
    return {
        "found": True,
        "task_result_id": result.get("task_result_id"),
        "task_id": result.get("task_id"),
        "path": result.get("path"),
        "task_path": result_task_path or (str(latest_task) if latest_task is not None else None),
        "summary": result.get("summary"),
        "status": result.get("status"),
        "ok": result.get("ok"),
        "brain_evaluation": result.get("brain_evaluation"),
        "evidence": _compact(result.get("evidence") if isinstance(result.get("evidence"), dict) else {}),
    }


def _hypothesis(
    brain_state: dict[str, Any],
    kind: str,
    claim: str,
    why_it_matters: str,
    evidence: dict[str, Any],
    proposed_experiment: dict[str, Any],
    expected_useful_output: str,
) -> dict[str, Any]:
    return {
        "hypothesis_id": _stable_id("hypothesis", brain_state.get("iteration", 0), kind, claim),
        "kind": kind,
        "claim": claim,
        "why_it_matters": why_it_matters,
        "evidence": _compact(evidence),
        "proposed_experiment": proposed_experiment,
        "expected_useful_output": expected_useful_output,
    }


def _action(action: str, reason: str, hypothesis_kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "decision_id": _stable_id("decision", action, reason, hypothesis_kind),
        "action": action,
        "reason": reason,
        "hypothesis_kind": hypothesis_kind,
        **extra,
    }


def _action_for_task_result(task_result: dict[str, Any]) -> dict[str, Any]:
    status = str(task_result.get("status") or "").lower()
    ok = task_result.get("ok")
    evidence = task_result.get("evidence") if isinstance(task_result.get("evidence"), dict) else {}
    if status == "manual_actuator_required":
        return _action(
            "manual_actuator_required",
            "Codex CLI non-interactive execution was not available; expose the exact manual actuator instructions without marking success.",
            "artifact_quality",
            task_result_id=task_result.get("task_result_id"),
            task_id=task_result.get("task_id"),
            task_path=task_result.get("task_path"),
            manual_command=evidence.get("manual_command") or evidence.get("next_recommendation"),
        )
    if status in {"needs_validation", "validation_required"}:
        return _action(
            "rerun_validation",
            "A Codex task result exists and asks PEDS to rerun validation.",
            "artifact_quality",
            task_result_id=task_result.get("task_result_id"),
            task_id=task_result.get("task_id"),
        )
    if ok is True or status in {"completed", "accepted", "success"}:
        return _action(
            "accept_result",
            "A Codex task result exists with completed evidence; PEDS should accept it into memory.",
            "artifact_quality",
            task_result_id=task_result.get("task_result_id"),
            task_id=task_result.get("task_id"),
        )
    if ok is False or status in {"failed", "blocked", "needs_followup"}:
        return _action(
            "request_followup",
            "A Codex task result exists but indicates failure or a blocker; PEDS should issue a follow-up actuator task.",
            "artifact_quality",
            task_result_id=task_result.get("task_result_id"),
            task_id=task_result.get("task_id"),
        )
    return _action(
        "rerun_validation",
        "A Codex task result exists but its outcome is unclear; PEDS should validate before accepting it.",
        "artifact_quality",
        task_result_id=task_result.get("task_result_id"),
        task_id=task_result.get("task_id"),
    )


def _high_correction_success(campaign: dict[str, Any]) -> bool:
    return float(campaign.get("correction_success_rate") or 0.0) >= 0.8 and int(campaign.get("corrections_attempted") or 0) > 0


def _narrow_mutation_diversity(campaign: dict[str, Any]) -> bool:
    return int(campaign.get("mutation_diversity") or 0) <= 1


def _trainer_signal_weak(campaign: dict[str, Any]) -> bool:
    trained = campaign.get("trained_rate")
    loaded = campaign.get("adapter_loaded_rate")
    return (trained is not None and float(trained) < 0.5) or (loaded is not None and float(loaded) < 0.5)


def _likely_files_for_action(action: str) -> list[str]:
    mapping = {
        "inspect_artifact_quality": [
            "src/proof_transformer/peds/survivor_outputs.py",
            "scripts/peds_autopilot.py",
            "tests/test_peds_autopilot.py",
        ],
        "improve_trainer_signal": [
            "src/proof_transformer/peds/trainer.py",
            "examples/peds_trained_local_generations.py",
            "tests/test_peds_trainer.py",
        ],
        "improve_gui_use_peds_path": [
            "scripts/peds_dashboard.py",
            "tests/test_peds_dashboard.py",
        ],
        "improve_survivor_output_quality": [
            "src/proof_transformer/peds/survivor_outputs.py",
            "tests/test_peds_survivor_outputs.py",
        ],
        "add_active_policy_campaign_start": [
            "scripts/peds_autopilot.py",
            "scripts/peds_training_campaign.py",
            "src/proof_transformer/peds/brain.py",
            "tests/test_peds_autopilot.py",
        ],
        "expand_agentic_capability_curriculum": [
            "src/proof_transformer/peds/survivor_outputs.py",
            "src/proof_transformer/peds/brain.py",
            "scripts/peds_autopilot.py",
            "tests/test_peds_survivor_outputs.py",
            "tests/test_peds_brain.py",
        ],
    }
    return mapping.get(action, ["src/proof_transformer/peds/brain.py", "tests/test_peds_brain.py"])


def _task_text_for_action(action: str) -> str:
    mapping = {
        "inspect_artifact_quality": "Inspect and strengthen PEDS useful-output artifacts so they are concrete, evidence-backed, and directly usable.",
        "improve_trainer_signal": "Improve trainer-signal evidence while preserving audited local replay/canary/shadow/promotion gates.",
        "improve_gui_use_peds_path": "Improve the GUI path only as an operator face for PEDS Brain decisions, not as the source of decisions.",
        "improve_survivor_output_quality": "Strengthen survivor profile, Codex prompt, repo plan, skill, and autopilot summary quality without weakening proof gates.",
        "add_active_policy_campaign_start": "Wire campaign/autopilot startup from the active PEDS policy, or expose a truthful unsupported-path result; do not pretend active-policy usage.",
        "expand_agentic_capability_curriculum": "Add or strengthen local evaluations for broad agentic capabilities beyond PEDS self-improvement mechanics, with verifier-backed evidence for each capability.",
        "request_followup": "Resolve the blocker or failed evidence from the latest Codex actuator task result, then report back in a result artifact PEDS can evaluate.",
    }
    return mapping.get(action, f"Execute the PEDS-decided action: {action}")


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"attempts", "rejected_attempts", "diagnoses", "corrections", "mutations", "lineage", "runs"}:
                compacted[key] = f"<omitted {len(item) if isinstance(item, list) else 'large'} records>"
            else:
                compacted[key] = _compact(item)
        return compacted
    if isinstance(value, list):
        if len(value) > 12:
            return [_compact(item) for item in value[:12]] + [f"<omitted {len(value) - 12} items>"]
        return [_compact(item) for item in value]
    return value


def _write_state(state: dict[str, Any], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / BRAIN_STATE_FILE).write_text(json.dumps(_normalize_state(state), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "schema_version": SCHEMA_VERSION,
        "system": SYSTEM_NAME,
        "seeded_objective": SEEDED_OBJECTIVE,
        "current_objective": "",
        "current_active_policy_id": None,
        "current_active_candidate_id": None,
        "current_focus": "uninitialized",
        "iteration": 0,
        "max_iterations": 10,
        "status": "idle",
        "last_campaign_report": None,
        "last_autopilot_output": None,
        "last_codex_task": None,
        "last_codex_task_result": None,
        "last_actuator_run": None,
        "last_decision_id": None,
        "next_action": None,
        "open_questions": [],
        "stop_reason": None,
    }
    normalized = {**defaults, **state}
    if not isinstance(normalized.get("open_questions"), list):
        normalized["open_questions"] = []
    return normalized


def _task_result_summary(raw: str, decoded: dict[str, Any] | None) -> str:
    if isinstance(decoded, dict):
        for key in ("summary", "result", "message", "final_report"):
            value = decoded.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:800]
    compact = " ".join(raw.strip().split())
    return compact[:800] if compact else "Empty Codex task result."


def _infer_task_result_status(raw: str, decoded: dict[str, Any] | None) -> str:
    if isinstance(decoded, dict):
        status = decoded.get("status") or decoded.get("outcome")
        if isinstance(status, str) and status.strip():
            return status.strip().lower()
        if decoded.get("requires_validation") or decoded.get("validation_required"):
            return "needs_validation"
        if decoded.get("ok") is True:
            return "completed"
        if decoded.get("ok") is False:
            return "failed"
    lowered = raw.lower()
    if "manual_actuator_required" in lowered or "manual actuator required" in lowered:
        return "manual_actuator_required"
    if "validation required" in lowered or "needs validation" in lowered:
        return "needs_validation"
    if "blocked" in lowered:
        return "blocked"
    if "failed" in lowered or "failure" in lowered:
        return "failed"
    return "completed"


def _infer_task_result_ok(raw: str, decoded: dict[str, Any] | None) -> bool | None:
    if isinstance(decoded, dict) and isinstance(decoded.get("ok"), bool):
        return bool(decoded["ok"])
    status = _infer_task_result_status(raw, decoded)
    if status in {"completed", "accepted", "success"}:
        return True
    if status in {"failed", "blocked", "needs_followup", "manual_actuator_required"}:
        return False
    return None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _latest_jsonl(path: Path) -> dict[str, Any] | None:
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


def _append_task_event(root: Path, payload: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    if record.get("task_id") and not record.get("queue_id"):
        record["queue_id"] = record["task_id"]
    if record.get("queue_id") and not record.get("task_id"):
        record["task_id"] = record["queue_id"]
    _append_jsonl(root / TASK_QUEUE_FILE, record)
    _append_jsonl(root / CODEX_ACTUATOR_QUEUE_FILE, record)


def _task_queue_status(root: Path) -> dict[str, dict[str, Any]]:
    paths = [root / TASK_QUEUE_FILE]
    if not paths[0].exists() and (root / CODEX_ACTUATOR_QUEUE_FILE).exists():
        paths = [root / CODEX_ACTUATOR_QUEUE_FILE]
    status: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(decoded, dict):
                continue
            task_key = decoded.get("task_id") or decoded.get("queue_id") or decoded.get("task_path")
            if task_key:
                row = dict(decoded)
                row.setdefault("task_id", task_key)
                row.setdefault("queue_id", task_key)
                status[str(task_key)] = row
    return _collapse_task_status_by_path(status)


def _collapse_task_status_by_path(status: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    collapsed: dict[str, dict[str, Any]] = {}
    for row in status.values():
        key = str(row.get("task_path") or row.get("task_id") or row.get("queue_id"))
        existing = collapsed.get(key)
        if existing is None or _prefer_task_row(row, existing):
            collapsed[key] = row
    return collapsed


def _prefer_task_row(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
    candidate_terminal = candidate.get("status") in TERMINAL_TASK_STATES
    incumbent_terminal = incumbent.get("status") in TERMINAL_TASK_STATES
    if candidate_terminal != incumbent_terminal:
        return candidate_terminal
    return str(candidate.get("timestamp") or "") >= str(incumbent.get("timestamp") or "")


def _codex_queue_status(root: Path) -> dict[str, dict[str, Any]]:
    return _task_queue_status(root)


def _latest_queue_task_path(queue: list[dict[str, Any]]) -> Path | None:
    task_rows = [item for item in queue if item.get("task_path") and item.get("status") not in {"superseded"}]
    if not task_rows:
        return None
    latest = sorted(task_rows, key=lambda row: str(row.get("timestamp") or ""))[-1]
    path = Path(str(latest["task_path"]))
    return path if path.exists() else None


def _task_id_for_path(queue: list[dict[str, Any]], path: Path | None) -> str | None:
    if path is None:
        return None
    for item in queue:
        if item.get("task_path") == str(path):
            return str(item.get("task_id") or item.get("queue_id"))
    return None


def _queue_result_for_task(queue: list[dict[str, Any]], path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    for item in queue:
        if item.get("task_path") == str(path) and item.get("status") in TERMINAL_TASK_STATES:
            return {
                "task_result_id": item.get("task_result_id") or _stable_id("task_result", item.get("task_id"), item.get("status")),
                "task_id": item.get("task_id") or item.get("queue_id"),
                "path": item.get("result_path"),
                "task_path": item.get("task_path"),
                "summary": item.get("summary") or f"Task queue reached terminal status: {item.get('status')}",
                "status": item.get("status"),
                "ok": item.get("status") == "completed",
                "evidence": {"queue_event": item},
            }
    return None


def _has_action_task(evidence: dict[str, Any], action: str) -> bool:
    queue = evidence.get("task_queue") if isinstance(evidence.get("task_queue"), dict) else {}
    for item in queue.get("current", []) if isinstance(queue.get("current"), list) else []:
        if item.get("action") == action and item.get("status") not in {"failed", "superseded"}:
            return True
    return False


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else DEFAULT_BRAIN_ROOT


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _safe_name(value: str) -> str:
    safe = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_") or "action"


def _task_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
