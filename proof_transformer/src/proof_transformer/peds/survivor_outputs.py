from __future__ import annotations

import json
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ACTIVE_POLICY_ROOT = Path(__file__).resolve().parents[3] / "temp" / "peds_active_policy"

AGENTIC_CAPABILITY_CURRICULUM = [
    {
        "capability": "repo_navigation",
        "objective": "Find the smallest truthful code, test, config, and artifact surface for a user goal.",
        "local_eval": "Given a repo objective, produce the likely files, evidence paths, and first verifier command before editing.",
        "success_signal": "The chosen files include the real implementation and nearest tests, with no broad unrelated crawl required.",
    },
    {
        "capability": "code_editing",
        "objective": "Make scoped code changes that preserve existing work and improve the requested behavior.",
        "local_eval": "Apply one bounded patch for a PEDS-decided task and keep unrelated dirty worktree changes intact.",
        "success_signal": "The diff only touches expected files and focused tests pass.",
    },
    {
        "capability": "test_diagnosis",
        "objective": "Use failing or targeted tests to identify the next discriminating change.",
        "local_eval": "Run a focused test, classify the failure or pass signal, and choose the next smallest validation step.",
        "success_signal": "The report names the exact command, outcome, and remaining unverified risk.",
    },
    {
        "capability": "tool_orchestration",
        "objective": "Coordinate local scripts, queues, actuator handoffs, and result recording without fake completion.",
        "local_eval": "Claim or generate a PEDS task, route it through the available actuator path, and record the result.",
        "success_signal": "Task queue state moves through pending, running or manual, and completed/failed with evidence.",
    },
    {
        "capability": "planning_decomposition",
        "objective": "Break vague agentic goals into executable local experiments with acceptance criteria.",
        "local_eval": "Turn a broad objective into three verifiable repo tasks and one current action.",
        "success_signal": "Each task has likely files, constraints, validation commands, and a report contract.",
    },
    {
        "capability": "memory_provenance",
        "objective": "Carry forward useful lineage, survivor, and task-result evidence without overwriting history.",
        "local_eval": "Activate a survivor policy append-only and cite the evidence used for the next task.",
        "success_signal": "The active policy and journal retain candidate, objective, safety, and evidence summary fields.",
    },
    {
        "capability": "artifact_writing",
        "objective": "Produce operator-usable prompts, plans, skills, and summaries from local evidence.",
        "local_eval": "Write survivor profile, Codex prompt, repo plan, and skill artifacts from the selected survivor.",
        "success_signal": "Artifacts include concrete actions, constraints, validation, and do-not-do rules.",
    },
    {
        "capability": "bounded_autonomy",
        "objective": "Keep autonomous loops measurable, local, and stop-capable while still pushing useful work forward.",
        "local_eval": "Run a bounded brain loop with max-iteration, safety, and pending-task checks.",
        "success_signal": "The loop stops, waits, or queues a task for explicit evidence instead of spinning or overclaiming.",
    },
    {
        "capability": "data_extraction",
        "objective": "Read structured local artifacts and extract compact, decision-useful evidence.",
        "local_eval": "Summarize campaign, autopilot, task queue, and task-result JSON into a stable evidence object.",
        "success_signal": "Large attempt lists are compacted while decision-critical aggregate fields remain visible.",
    },
    {
        "capability": "ui_or_browser_operation",
        "objective": "Operate a local interface as an actuator surface while preserving PEDS as the decision owner.",
        "local_eval": "Verify a dashboard or browser-facing flow can expose brain status, decisions, and task results.",
        "success_signal": "The UI path mirrors PEDS state and does not invent decisions outside the brain loop.",
    },
]


def select_current_survivor(report: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(report, dict) or not _safety_clean(report.get("safety") if isinstance(report.get("safety"), dict) else {}):
        return None

    attempts = _report_attempts(report)
    winning_attempts = sorted(
        [attempt for attempt in attempts if attempt.get("outcome") == "winner" and attempt.get("winning_candidate_id")],
        key=_attempt_sort_key,
    )
    preferred = [attempt for attempt in winning_attempts if _attempt_trained_and_loaded(attempt)]
    if preferred:
        return _survivor_from_attempt(preferred[-1], "latest winning attempt with trained adapter and clean safety")
    if winning_attempts:
        return _survivor_from_attempt(winning_attempts[-1], "latest winning attempt with clean safety")

    public_winners = sorted(
        [item for item in report.get("winning_attempts", []) if isinstance(item, dict) and item.get("winning_candidate_id")],
        key=_attempt_sort_key,
    )
    if public_winners:
        return _survivor_from_public_winner(public_winners[-1], "latest public winning attempt with clean safety")

    lineage_rows = [item for item in report.get("lineage", []) if isinstance(item, dict)]
    for row in reversed(lineage_rows):
        candidate_id = row.get("selected_parent_for_next_generation") or row.get("winning_candidate_id")
        if candidate_id:
            return {
                "candidate_id": candidate_id,
                "generation": row.get("generation"),
                "attempt_index": row.get("winning_attempt_number"),
                "parent_candidate_id": row.get("parent_candidate_id"),
                "source_candidate_id": row.get("source_candidate_id"),
                "mutation_target": row.get("mutation_target"),
                "mutation_seed": row.get("mutation_seed"),
                "diagnosis_category": _mutation_value(row, "diagnosis_category"),
                "delta_description": _mutation_value(row, "delta_description"),
                "replay_mean_delta": None,
                "adapter_loaded": None,
                "training_trained": None,
                "selection_reason": "latest lineage selected parent with clean safety",
            }
    return None


def build_survivor_profile(report: dict[str, Any], objective: str, survivor: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = survivor if survivor is not None else select_current_survivor(report)
    if selected is None:
        raise ValueError("No usable PEDS survivor is available from clean campaign evidence.")
    mutation_summary = _mutation_summary(report)
    evidence_summary = _evidence_summary(report)
    agentic_capability_coverage = build_agentic_capability_coverage(report, objective)
    winning_strategy = {
        "candidate_id": selected["candidate_id"],
        "generation": selected.get("generation"),
        "attempt_index": selected.get("attempt_index"),
        "mutation_target": selected.get("mutation_target"),
        "mutation_seed": selected.get("mutation_seed"),
        "diagnosis_category": selected.get("diagnosis_category"),
        "delta_description": selected.get("delta_description"),
        "replay_mean_delta": selected.get("replay_mean_delta"),
        "adapter_loaded": selected.get("adapter_loaded"),
        "training_trained": selected.get("training_trained"),
    }
    return {
        "profile_type": "peds_survivor_profile",
        "objective": objective,
        "candidate_id": selected["candidate_id"],
        "selected_at": _utc_now(),
        "selection_reason": selected.get("selection_reason"),
        "generation": selected.get("generation"),
        "attempt_index": selected.get("attempt_index"),
        "what_improved": _what_improved(report, selected),
        "winning_strategy": winning_strategy,
        "mutation_summary": mutation_summary,
        "evidence_summary": evidence_summary,
        "agentic_capability_coverage": agentic_capability_coverage,
        "agentic_capability_curriculum": agentic_capability_coverage["curriculum"],
        "safety": deepcopy(report.get("safety") if isinstance(report.get("safety"), dict) else {}),
        "recommended_uses": [
            "Use this survivor profile to guide the next local repo-assistant task.",
            "Use this profile to pressure broad agentic capabilities, not only the improvement loop itself.",
            "Prefer patch plans that preserve replay/canary/shadow/promotion/training gates.",
            "Continue from the selected survivor evidence instead of restarting from rejected attempts.",
        ],
        "next_tasks": [
            *agentic_capability_coverage["next_eval_tasks"],
            "Run python scripts/peds_brain.py --think --json and follow the PEDS-decided next action.",
            "If PEDS queues a task, run python scripts/peds_actuator.py --run-next --json or manually execute the task and record the result.",
            "Validate with focused PEDS tests before broad test_peds_*.py validation.",
        ],
        "validation_commands": [
            "python -m unittest discover -s tests -p 'test_peds_brain.py'",
            "python -m unittest discover -s tests -p 'test_peds_actuator.py'",
            "python -m unittest discover -s tests -p 'test_peds_*.py'",
        ],
        "do_not_do": [
            "Do not force-promote or mark rejected attempts as winners.",
            "Do not weaken replay, canary, shadow, promotion, or training gates.",
            "Do not call network or model APIs as part of local PEDS operation.",
            "Do not edit candidate rows or hide safety contamination.",
        ],
    }


def build_codex_prompt_from_survivor(profile: dict[str, Any]) -> str:
    strategy = profile.get("winning_strategy") if isinstance(profile.get("winning_strategy"), dict) else {}
    mutation_summary = profile.get("mutation_summary") if isinstance(profile.get("mutation_summary"), dict) else {}
    evidence_summary = profile.get("evidence_summary") if isinstance(profile.get("evidence_summary"), dict) else {}
    safety = profile.get("safety") if isinstance(profile.get("safety"), dict) else {}
    what_improved = profile.get("what_improved") if isinstance(profile.get("what_improved"), list) else []
    do_not_do = profile.get("do_not_do") if isinstance(profile.get("do_not_do"), list) else []
    validation_commands = profile.get("validation_commands") if isinstance(profile.get("validation_commands"), list) else []
    capability_coverage = profile.get("agentic_capability_coverage") if isinstance(profile.get("agentic_capability_coverage"), dict) else {}
    capability_tasks = capability_coverage.get("next_eval_tasks") if isinstance(capability_coverage.get("next_eval_tasks"), list) else []
    return "\n".join(
        [
            "You are Codex operating with the selected PEDS survivor profile.",
            "",
            f"Objective: {profile.get('objective')}",
            f"Survivor candidate: {profile.get('candidate_id')}",
            f"Selection reason: {profile.get('selection_reason')}",
            "",
            "Winning strategy:",
            f"- generation: {strategy.get('generation')}",
            f"- attempt: {strategy.get('attempt_index')}",
            f"- mutation target: {strategy.get('mutation_target')}",
            f"- mutation seed: {strategy.get('mutation_seed')}",
            f"- diagnosis category: {strategy.get('diagnosis_category')}",
            f"- delta description: {strategy.get('delta_description')}",
            f"- replay mean delta: {strategy.get('replay_mean_delta')}",
            f"- adapter_loaded: {strategy.get('adapter_loaded')}",
            f"- training_trained: {strategy.get('training_trained')}",
            "",
            "Mutation/correction evidence:",
            f"- dominant diagnosis: {mutation_summary.get('dominant_diagnosis')}",
            f"- dominant mutation target: {mutation_summary.get('dominant_mutation_target')}",
            f"- mutation targets observed: {mutation_summary.get('mutation_targets')}",
            f"- correction success: {evidence_summary.get('corrections_followed_by_winner')} / {evidence_summary.get('corrections_attempted')}",
            "",
            "Validation/evidence:",
            f"- completed generations: {evidence_summary.get('total_generations_completed')}",
            f"- total attempts: {evidence_summary.get('total_candidate_attempts')}",
            f"- winners: {evidence_summary.get('total_winning_attempts')}",
            f"- rejected attempts: {evidence_summary.get('total_rejected_attempts')}",
            f"- collapse_detected: {evidence_summary.get('collapse_detected')}",
            f"- safety: {json.dumps(safety, ensure_ascii=False, sort_keys=True)}",
            "",
            "What improved:",
            *[f"- {item}" for item in what_improved],
            "",
            "Agentic capability coverage:",
            f"- status: {capability_coverage.get('status')}",
            f"- pressured capabilities: {capability_coverage.get('pressured_capabilities')}",
            f"- untested capabilities: {capability_coverage.get('untested_capabilities')}",
            "- Do not count improvement-loop progress as broad agentic progress until a local eval pressures that capability.",
            "",
            "Next agentic eval tasks:",
            *(f"- {item}" for item in capability_tasks),
            "",
            "How to continue from the survivor:",
            "- Use the survivor's winning strategy as the default local patch policy for this objective.",
            "- Prefer tasks that expand verified agentic coverage across repo navigation, code editing, tests, tools, memory, artifacts, planning, data extraction, UI/browser operation, and bounded autonomy.",
            "- Preserve parent protection: rejected attempts are expendable, winners advance lineage.",
            "- Keep evidence isolated by attempt when judging future changes.",
            "- Validate with focused tests first, then broad PEDS validation.",
            "- If PEDS has queued a Codex actuator task, inspect/claim it with: python scripts/peds_brain.py --claim-codex-task --json",
            "- Then execute it with: python scripts/peds_actuator.py --run-next --json",
            "- If automatic execution is unsupported, follow the manual command in the actuator result and then record it with: python scripts/peds_actuator.py --record-result <result_report_path> --json",
            "- Legacy/manual brain recording remains: python scripts/peds_brain.py --record-task-result <result_report_path> --json",
            "",
            "Validation ladder:",
            *(f"- {item}" for item in validation_commands),
            "",
            "Do not do:",
            *[f"- {item}" for item in do_not_do],
            "",
            "Final report contract:",
            "- Summary",
            "- Files changed",
            "- Validation commands and outcomes",
            "- Evidence observed",
            "- PEDS task/result state",
            "- Remaining risks and next recommendation",
        ]
    )


def build_repo_improvement_plan(profile: dict[str, Any]) -> str:
    strategy = profile.get("winning_strategy") if isinstance(profile.get("winning_strategy"), dict) else {}
    what_improved = profile.get("what_improved") if isinstance(profile.get("what_improved"), list) else []
    capability_coverage = profile.get("agentic_capability_coverage") if isinstance(profile.get("agentic_capability_coverage"), dict) else {}
    capability_tasks = capability_coverage.get("next_eval_tasks") if isinstance(capability_coverage.get("next_eval_tasks"), list) else []
    return "\n".join(
        [
            "# PEDS Repo Improvement Plan",
            "",
            f"Objective: {profile.get('objective')}",
            f"Selected survivor: `{profile.get('candidate_id')}`",
            "",
            "## What PEDS improved",
            *(f"- {item}" for item in what_improved),
            "",
            "## How to use the survivor",
            "- Start the next repo task from the survivor profile, not from a rejected attempt.",
            f"- Preserve the winning mutation target: `{strategy.get('mutation_target')}`.",
            "- Keep correction evidence auditable and attempt-scoped.",
            "- Let PEDS issue the next task, let the actuator claim/run it when available, then record the result back into PEDS.",
            "",
            "## Agentic capability curriculum",
            f"- Coverage status: `{capability_coverage.get('status')}`.",
            f"- Pressured capabilities: `{capability_coverage.get('pressured_capabilities')}`.",
            f"- Untested capabilities: `{capability_coverage.get('untested_capabilities')}`.",
            "- Treat this as the bridge from self-improvement mechanics to broad agentic skill pressure.",
            "- A capability counts only when a local eval or artifact proves it under repo constraints.",
            "",
            "## Next agentic eval tasks",
            *(f"- {item}" for item in capability_tasks),
            "",
            "## PEDS-to-Codex actuator handoff",
            "1. python scripts/peds_brain.py --think --json",
            "2. python scripts/peds_brain.py --step --json",
            "3. python scripts/peds_brain.py --claim-codex-task --json (optional inspection) or python scripts/peds_actuator.py --run-next --json",
            "4. If the actuator reports manual_actuator_required, run the surfaced manual Codex command.",
            "5. python scripts/peds_actuator.py --record-result <result_report_path> --json",
            "6. python scripts/peds_brain.py --think --json",
            "",
            "## Next 3 concrete repo tasks",
            "1. Claim and complete the latest PEDS-decided Codex actuator task.",
            "2. Add or run the next broad agentic capability eval before calling the loop broadly improved.",
            "3. Record the task result so PEDS can accept, reject, or request follow-up.",
            "",
            "## Validation ladder",
            "1. python -m unittest discover -s tests -p 'test_peds_dashboard.py'",
            "2. python -m unittest discover -s tests -p 'test_peds_training_campaign.py'",
            "3. python -m unittest discover -s tests -p 'test_peds_*.py'",
            "",
            "## Expected useful output",
            "A safer, more reliable local repo-assistant workflow that keeps PEDS proof gates intact while using the latest survivor strategy.",
        ]
    )


def build_codex_skill_from_survivor(profile: dict[str, Any]) -> str:
    strategy = profile.get("winning_strategy") if isinstance(profile.get("winning_strategy"), dict) else {}
    do_not_do = profile.get("do_not_do") if isinstance(profile.get("do_not_do"), list) else []
    capability_coverage = profile.get("agentic_capability_coverage") if isinstance(profile.get("agentic_capability_coverage"), dict) else {}
    capability_tasks = capability_coverage.get("next_eval_tasks") if isinstance(capability_coverage.get("next_eval_tasks"), list) else []
    return "\n".join(
        [
            "# PEDS Survivor Skill",
            "",
            "## Name",
            "peds-survivor-local-repo-assistant",
            "",
            "## When to use",
            f"Use when the objective is: {profile.get('objective')}",
            "Use when you want Codex to continue from the latest clean PEDS survivor instead of raw baseline behavior.",
            "",
            "## Operating rules",
            "- Treat rejected attempts as search evidence, not as winners.",
            "- Protect lineage parents and advance only winning children.",
            "- Keep attempt evidence isolated and auditable.",
            f"- Prefer the survivor mutation target: {strategy.get('mutation_target')}.",
            "- Expand verified agentic capability coverage instead of only improving PEDS mechanics.",
            "",
            "## Workflow",
            "1. Ask PEDS Brain to think or step before editing.",
            "2. Let the actuator run one PEDS-decided task, or follow the manual actuator command exactly.",
            "3. Record the result so PEDS can accept, validate, fail, or request follow-up.",
            "",
            "## Agentic capability curriculum",
            f"- coverage_status: {capability_coverage.get('status')}",
            f"- pressured_capabilities: {capability_coverage.get('pressured_capabilities')}",
            f"- untested_capabilities: {capability_coverage.get('untested_capabilities')}",
            *(f"- next_eval: {item}" for item in capability_tasks),
            "",
            "## Validation ladder / Validation commands",
            "- python -m unittest discover -s tests -p 'test_peds_training_campaign.py'",
            "- python -m unittest discover -s tests -p 'test_peds_mutation_strategies.py'",
            "- python -m unittest discover -s tests -p 'test_peds_*.py'",
            "",
            "## Survivor strategy summary",
            f"- candidate_id: {profile.get('candidate_id')}",
            f"- selection_reason: {profile.get('selection_reason')}",
            f"- mutation_target: {strategy.get('mutation_target')}",
            f"- mutation_seed: {strategy.get('mutation_seed')}",
            f"- diagnosis_category: {strategy.get('diagnosis_category')}",
            "",
            "## Do-not-do rules",
            *(f"- {item}" for item in do_not_do),
            "",
            "## Output contract",
            "Return Summary, Files changed, Validation, Evidence observed, PEDS task/result state, and Next recommendation.",
        ]
    )


def write_autopilot_artifacts(profile: dict[str, Any], output_dir: Path, artifact_type: str = "all") -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_type = artifact_type or "all"
    if artifact_type not in {"all", "profile", "prompt", "plan", "skill"}:
        raise ValueError("artifact_type must be one of all, profile, prompt, plan, skill")
    paths: dict[str, str] = {}
    if artifact_type in {"all", "profile"}:
        profile_path = output_dir / "survivor_profile.json"
        profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        paths["survivor_profile"] = str(profile_path)
    if artifact_type in {"all", "prompt"}:
        prompt_path = output_dir / "codex_prompt.txt"
        prompt_path.write_text(build_codex_prompt_from_survivor(profile), encoding="utf-8")
        paths["codex_prompt"] = str(prompt_path)
    if artifact_type in {"all", "plan"}:
        plan_path = output_dir / "repo_improvement_plan.md"
        plan_path.write_text(build_repo_improvement_plan(profile), encoding="utf-8")
        paths["repo_improvement_plan"] = str(plan_path)
    if artifact_type in {"all", "skill"}:
        skill_path = output_dir / "SKILL.md"
        skill_path.write_text(build_codex_skill_from_survivor(profile), encoding="utf-8")
        paths["skill"] = str(skill_path)
    return paths


def append_active_policy(profile: dict[str, Any], *, registry_root: Path | None = None) -> dict[str, Any]:
    root = registry_root or DEFAULT_ACTIVE_POLICY_ROOT
    root.mkdir(parents=True, exist_ok=True)
    registry_path = root / "active_policy.jsonl"
    record = {
        "active_policy_id": f"peds_active_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{uuid.uuid4().hex[:8]}",
        "candidate_id": profile.get("candidate_id"),
        "objective": profile.get("objective"),
        "selected_at": _utc_now(),
        "selection_reason": profile.get("selection_reason"),
        "profile_path": profile.get("profile_path"),
        "safety": deepcopy(profile.get("safety") if isinstance(profile.get("safety"), dict) else {}),
        "evidence_summary": deepcopy(profile.get("evidence_summary") if isinstance(profile.get("evidence_summary"), dict) else {}),
        "agentic_capability_coverage": deepcopy(
            profile.get("agentic_capability_coverage") if isinstance(profile.get("agentic_capability_coverage"), dict) else {}
        ),
    }
    with registry_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    record["registry_path"] = str(registry_path)
    return record


def load_latest_active_policy(registry_root: Path | None = None) -> dict[str, Any] | None:
    root = registry_root or DEFAULT_ACTIVE_POLICY_ROOT
    registry_path = root / "active_policy.jsonl"
    if not registry_path.exists():
        return None
    latest: dict[str, Any] | None = None
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            latest = decoded
    if latest is not None:
        latest["registry_path"] = str(registry_path)
    return latest


def build_agentic_capability_coverage(report: dict[str, Any], objective: str) -> dict[str, Any]:
    objective_text = str(objective or "").lower()
    evidence = _evidence_summary(report)
    mutation = _mutation_summary(report)
    pressured = set()
    reasons: dict[str, str] = {}

    if any(token in objective_text for token in ("repo", "patch", "code", "codex", "test", "file")):
        for capability in ("repo_navigation", "code_editing", "test_diagnosis", "artifact_writing"):
            pressured.add(capability)
            reasons[capability] = "objective asks for repo/code/test work"
    if evidence.get("total_candidate_attempts"):
        for capability in ("bounded_autonomy", "data_extraction"):
            pressured.add(capability)
            reasons[capability] = "campaign report produced attempt and aggregate evidence"
    if evidence.get("corrections_attempted"):
        pressured.add("planning_decomposition")
        reasons["planning_decomposition"] = "correction attempts required bounded next-action selection"
    if evidence.get("total_winning_attempts"):
        pressured.add("memory_provenance")
        reasons["memory_provenance"] = "survivor lineage and active policy can carry winner evidence forward"
    if mutation.get("mutation_diversity"):
        pressured.add("tool_orchestration")
        reasons["tool_orchestration"] = "campaign/autopilot tooling produced mutation-target evidence"

    curriculum = []
    for item in AGENTIC_CAPABILITY_CURRICULUM:
        capability = item["capability"]
        status = "pressured_by_current_survivor" if capability in pressured else "not_yet_pressured"
        curriculum.append(
            {
                **item,
                "status": status,
                "evidence_reason": reasons.get(capability),
            }
        )
    untested = [item["capability"] for item in curriculum if item["status"] == "not_yet_pressured"]
    next_eval_tasks = _next_agentic_eval_tasks(curriculum)
    return {
        "status": "broadening_needed" if untested else "broad_curriculum_pressured",
        "intent": "Improve agentic capability across task classes, not only PEDS self-improvement mechanics.",
        "pressured_capabilities": sorted(pressured),
        "untested_capabilities": untested,
        "curriculum": curriculum,
        "next_eval_tasks": next_eval_tasks,
        "coverage_warning": "Do not claim all-agentic improvement until each capability has local verifier evidence.",
    }


def _survivor_from_attempt(attempt: dict[str, Any], reason: str) -> dict[str, Any]:
    metric = attempt.get("metric") if isinstance(attempt.get("metric"), dict) else {}
    mutation = attempt.get("mutation") if isinstance(attempt.get("mutation"), dict) else {}
    return {
        "candidate_id": attempt.get("winning_candidate_id") or attempt.get("generated_child_candidate_id") or metric.get("candidate_id"),
        "generation": attempt.get("generation"),
        "attempt_index": attempt.get("attempt_index"),
        "parent_candidate_id": attempt.get("parent_candidate_id"),
        "source_candidate_id": attempt.get("source_candidate_id"),
        "mutation_target": attempt.get("mutation_target") or mutation.get("mutation_target"),
        "mutation_seed": attempt.get("mutation_seed") or mutation.get("mutation_seed"),
        "diagnosis_category": mutation.get("diagnosis_category") or _diagnosis_category(attempt),
        "delta_description": mutation.get("delta_description"),
        "replay_mean_delta": metric.get("replay_mean_delta"),
        "adapter_loaded": metric.get("adapter_loaded"),
        "training_trained": metric.get("training_trained"),
        "selection_reason": reason,
    }


def _survivor_from_public_winner(item: dict[str, Any], reason: str) -> dict[str, Any]:
    mutation = item.get("mutation") if isinstance(item.get("mutation"), dict) else {}
    return {
        "candidate_id": item.get("winning_candidate_id"),
        "generation": item.get("generation"),
        "attempt_index": item.get("winning_attempt_number") or item.get("attempt_index"),
        "parent_candidate_id": item.get("parent_candidate_id"),
        "source_candidate_id": item.get("source_candidate_id"),
        "mutation_target": item.get("mutation_target") or mutation.get("mutation_target"),
        "mutation_seed": item.get("mutation_seed") or mutation.get("mutation_seed"),
        "diagnosis_category": mutation.get("diagnosis_category"),
        "delta_description": mutation.get("delta_description"),
        "replay_mean_delta": None,
        "adapter_loaded": None,
        "training_trained": None,
        "selection_reason": reason,
    }


def _attempt_trained_and_loaded(attempt: dict[str, Any]) -> bool:
    metric = attempt.get("metric") if isinstance(attempt.get("metric"), dict) else {}
    return metric.get("adapter_loaded") is True and metric.get("training_trained") is True


def _what_improved(report: dict[str, Any], survivor: dict[str, Any]) -> list[str]:
    evidence = _evidence_summary(report)
    improved = []
    if evidence.get("total_winning_attempts"):
        improved.append(f"{evidence['total_winning_attempts']} winning attempts advanced the lineage.")
    if evidence.get("total_rejected_attempts"):
        improved.append(f"{evidence['total_rejected_attempts']} rejected attempts were kept out of the survivor chain.")
    if evidence.get("corrections_followed_by_winner"):
        improved.append(
            f"{evidence['corrections_followed_by_winner']} corrections were followed by winners without weakening gates."
        )
    if survivor.get("mutation_target"):
        improved.append(f"The selected survivor carries a {survivor['mutation_target']} mutation strategy.")
    if not improved:
        improved.append("A clean survivor was selected from available lineage evidence.")
    return improved


def _mutation_summary(report: dict[str, Any]) -> dict[str, Any]:
    mutations = []
    for item in report.get("mutations", []):
        if not isinstance(item, dict):
            continue
        mutation = item.get("mutation") if isinstance(item.get("mutation"), dict) else item
        mutations.append(mutation)
    if not mutations:
        for attempt in _report_attempts(report):
            mutation = attempt.get("mutation") if isinstance(attempt.get("mutation"), dict) else {}
            if mutation:
                mutations.append(mutation)
    targets = [mutation.get("mutation_target") for mutation in mutations if mutation.get("mutation_target")]
    diagnoses = [mutation.get("diagnosis_category") for mutation in mutations if mutation.get("diagnosis_category")]
    diagnosed_mutations = [
        mutation
        for mutation in mutations
        if mutation.get("diagnosis_category") not in (None, "", "unknown")
    ]
    diagnosed_targets = [mutation.get("mutation_target") for mutation in diagnosed_mutations if mutation.get("mutation_target")]
    diagnosed_categories = [mutation.get("diagnosis_category") for mutation in diagnosed_mutations if mutation.get("diagnosis_category")]
    return {
        "mutation_count": len(mutations),
        "mutation_targets": sorted(set(str(target) for target in targets)),
        "dominant_mutation_target": _most_common(diagnosed_targets) or _most_common(targets),
        "dominant_diagnosis": _most_common(diagnosed_categories) or _most_common(diagnoses),
        "mutation_diversity": len(set(str(target) for target in targets)),
    }


def _evidence_summary(report: dict[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate") if isinstance(report.get("aggregate"), dict) else {}
    attempts = _report_attempts(report)
    winning_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "winner"]
    rejected_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "rejected"]
    return {
        "campaign_id": report.get("campaign_id"),
        "generations_requested": report.get("generations_requested"),
        "total_generations_completed": aggregate.get("total_generations_completed", len({attempt.get("generation") for attempt in winning_attempts})),
        "total_candidate_attempts": aggregate.get("total_candidate_attempts", len(attempts)),
        "total_winning_attempts": aggregate.get("total_winning_attempts", len(winning_attempts)),
        "total_rejected_attempts": aggregate.get("total_rejected_attempts", len(rejected_attempts)),
        "corrections_attempted": aggregate.get("corrections_attempted", 0),
        "corrections_followed_by_winner": aggregate.get("corrections_followed_by_winner", 0),
        "correction_success_rate": aggregate.get("correction_success_rate", 0.0),
        "collapse_detected": bool(aggregate.get("collapse_detected", False)),
        "exhausted_generation_count": aggregate.get("exhausted_generation_count", 0),
    }


def _diagnosis_category(attempt: dict[str, Any]) -> str | None:
    diagnosis = attempt.get("diagnosis")
    if isinstance(diagnosis, dict) and diagnosis.get("category"):
        return str(diagnosis["category"])
    if attempt.get("diagnosis_category"):
        return str(attempt["diagnosis_category"])
    return None


def _mutation_value(item: dict[str, Any], key: str) -> Any:
    mutation = item.get("mutation") if isinstance(item.get("mutation"), dict) else {}
    return item.get(key) or mutation.get(key)


def _report_attempts(report: dict[str, Any]) -> list[dict[str, Any]]:
    attempts = [dict(attempt) for attempt in report.get("attempts", []) if isinstance(attempt, dict)]
    if attempts:
        return attempts
    rows: list[dict[str, Any]] = []
    for run in report.get("runs", []) if isinstance(report.get("runs"), list) else []:
        if not isinstance(run, dict):
            continue
        run_index = run.get("run_index")
        for attempt in run.get("attempts", []) if isinstance(run.get("attempts"), list) else []:
            if isinstance(attempt, dict):
                row = dict(attempt)
                row.setdefault("run_index", run_index)
                rows.append(row)
        for generation in run.get("generation_summaries", []) if isinstance(run.get("generation_summaries"), list) else []:
            if not isinstance(generation, dict):
                continue
            for attempt in generation.get("attempts", []) if isinstance(generation.get("attempts"), list) else []:
                if isinstance(attempt, dict):
                    row = dict(attempt)
                    row.setdefault("run_index", run_index)
                    row.setdefault("generation", generation.get("generation"))
                    rows.append(row)
    return rows


def _attempt_sort_key(attempt: dict[str, Any]) -> tuple[int, int, int]:
    return (
        _coerce_int(attempt.get("run_index"), default=1),
        _coerce_int(attempt.get("generation"), default=0),
        _coerce_int(attempt.get("attempt_index") or attempt.get("winning_attempt_number"), default=0),
    )


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safety_clean(safety: dict[str, Any]) -> bool:
    return not bool(
        safety.get("network_attempts")
        or safety.get("model_modules_loaded")
        or safety.get("fake_weights_written")
    )


def _most_common(values: list[Any]) -> Any:
    filtered = [str(value) for value in values if value not in (None, "")]
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]


def _next_agentic_eval_tasks(curriculum: list[dict[str, Any]]) -> list[str]:
    tasks = []
    for item in curriculum:
        if item.get("status") == "not_yet_pressured":
            tasks.append(
                f"Add a local {item.get('capability')} eval: {item.get('local_eval')} Success signal: {item.get('success_signal')}"
            )
        if len(tasks) >= 3:
            break
    if tasks:
        return tasks
    return [
        "Run the broad agentic curriculum suite and record per-capability evidence before scaling autonomy.",
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
