from __future__ import annotations

import inspect
import json
import sqlite3
from copy import deepcopy
from typing import Any

from .adapter_bridge import create_candidate_from_training_run
from .canary import create_canary_job, run_canary_job
from .distillation import write_distillation_manifest
from .evaluation_pipeline import (
    create_post_training_replay_job,
    plan_post_training_evaluation,
    select_replay_episodes_for_candidate,
)
from .promotion import mark_distillation_candidate, promote_candidate
from .replay import run_replay_job
from .shadow import complete_shadow_run, record_shadow_event, start_shadow_run, summarize_shadow_run
from .store import get_candidate_current_state
from .trainer import run_prepared_training_run
from .training import get_training_run, prepare_training_run


_STATE_NEXT_STEPS = {
    "draft_candidate": "create_replay_job",
    "replay_winner": "create_canary_job",
    "canary_passed": "start_shadow_run",
    "shadow_started": "collect_shadow_events",
    "shadow_passed": "promote_or_distill",
    "promoted": "mark_distillation_candidate",
    "shadow_deployment": "mark_distillation_candidate",
    "distillation_candidate": "build_distillation_manifest",
}


def plan_peds_cycle(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    candidate = _load_candidate(conn, candidate_id)
    current_state = candidate["current_state"]
    latest_completed_training_run = _latest_completed_training_run(conn, candidate_id)
    next_step = _next_step_for_state(current_state, latest_completed_training_run)

    blockers: list[str] = []
    details: dict[str, Any] = {}

    if current_state == "draft_candidate":
        existing_replay_job = _latest_replay_job_for_candidate(conn, candidate_id, split="replay")
        if existing_replay_job is not None:
            next_step = "run_replay_job"
            details["replay_job"] = existing_replay_job
    elif current_state == "replay_winner":
        existing_canary_job = _latest_replay_job_for_candidate(conn, candidate_id, split="canary")
        if existing_canary_job is not None:
            next_step = "run_canary_job"
            details["canary_job"] = existing_canary_job
    elif current_state in {"canary_passed", "shadow_started"}:
        active_shadow_run = _active_shadow_run_for_candidate(conn, candidate_id)
        if active_shadow_run is not None:
            next_step = "collect_shadow_events"
            details["shadow_run"] = active_shadow_run

    if next_step == "create_replay_job":
        replay_plan = plan_post_training_evaluation(conn, candidate_id)
        details["replay_plan"] = replay_plan
        blockers.extend(replay_plan.get("blockers") or [])
    elif next_step == "run_replay_job":
        replay_job = details.get("replay_job") or _latest_replay_job_for_candidate(conn, candidate_id, split="replay")
        details["replay_job"] = replay_job
        if replay_job is None:
            blockers.append("replay job required")
        elif replay_job["status"] != "queued":
            blockers.append(f"queued replay job required, got {replay_job['status']!r}")
    elif next_step == "create_canary_job":
        incumbent = _latest_incumbent_from_decision(conn, candidate_id, "replay_winner")
        details["incumbent"] = incumbent
        if incumbent["incumbent_id"] is None:
            blockers.append("cannot infer incumbent from replay_winner evidence")
        details["canary_episode_count"] = _episode_count(conn, "canary")
        if details["canary_episode_count"] == 0:
            blockers.append("no canary episodes available")
    elif next_step == "run_canary_job":
        canary_job = details.get("canary_job") or _latest_replay_job_for_candidate(conn, candidate_id, split="canary")
        details["canary_job"] = canary_job
        if canary_job is None:
            blockers.append("canary job required")
        elif canary_job["status"] != "queued":
            blockers.append(f"queued canary job required, got {canary_job['status']!r}")
    elif next_step == "start_shadow_run":
        incumbent = _latest_incumbent_from_decision(conn, candidate_id, "canary_passed")
        details["incumbent"] = incumbent
        if incumbent["incumbent_id"] is None:
            blockers.append("cannot infer incumbent from canary_passed evidence")
    elif next_step == "collect_shadow_events":
        shadow_run = details.get("shadow_run") or _active_shadow_run_for_candidate(conn, candidate_id)
        details["shadow_run"] = shadow_run
        if shadow_run is None:
            blockers.append("active shadow run required")
    elif next_step == "promote_or_distill":
        details["promotion"] = "ready_for_promotion"
    elif next_step == "mark_distillation_candidate":
        details["distillation"] = "candidate can be marked for distillation using existing promotion evidence"
    elif next_step == "build_distillation_manifest":
        details["distillation"] = "build a distillation manifest from existing evidence"
    elif next_step == "create_candidate_from_training_run":
        details["training_run"] = latest_completed_training_run
        blockers.append("adapter_name option required to create candidate from completed training run")
    else:
        blockers.append(f"no automated cycle step for state {current_state!r}")

    return {
        "candidate_id": candidate_id,
        "current_state": current_state,
        "next_step": next_step,
        "would_advance": not blockers,
        "blockers": blockers,
        "details": details,
    }


def advance_peds_cycle(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    operator: str,
    mode: str = "manual",
    auto_promote: bool = False,
    dry_run: bool = False,
    options: dict[str, Any] | None = None,
    replay_evaluator=None,
    canary_evaluator=None,
    shadow_event_provider=None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    option_values = _normalize_options(options)
    plan = plan_peds_cycle(conn, candidate_id)
    if dry_run:
        return {
            "candidate_id": candidate_id,
            "dry_run": True,
            "mode": mode,
            "plan": plan,
            "action": "dry_run",
            "changed": {},
            "blocked": False,
            "blockers": [],
        }

    next_step = plan["next_step"]
    if plan["blockers"] and not _can_override_plan_blockers(next_step, option_values):
        return _blocked(candidate_id, plan, _blocker_action(next_step), plan["blockers"])

    if next_step in {"create_replay_job", "run_replay_job"}:
        return _advance_replay_stage(conn, candidate_id, plan, option_values, replay_evaluator)
    if next_step in {"create_canary_job", "run_canary_job"}:
        return _advance_canary_stage(conn, candidate_id, plan, option_values, canary_evaluator)
    if next_step in {"start_shadow_run", "collect_shadow_events"}:
        return _advance_shadow_stage(conn, candidate_id, plan, option_values, shadow_event_provider)
    if next_step == "promote_or_distill":
        if not auto_promote:
            return {
                "candidate_id": candidate_id,
                "state_before": plan["current_state"],
                "state_after": get_candidate_current_state(conn, candidate_id),
                "action": "ready_for_promotion",
                "changed": {},
                "blocked": True,
                "blockers": ["explicit auto_promote=True required to promote shadow_passed candidate"],
                "plan": plan,
            }
        reason = str(option_values.get("promotion_reason") or "PEDS cycle explicit promotion after shadow_passed evidence.")
        promotion = promote_candidate(conn, candidate_id, operator=operator_text, reason=reason)
        return _changed(conn, candidate_id, plan, "promote_candidate", {"promotion": promotion})
    if next_step == "mark_distillation_candidate":
        reason = str(option_values.get("distillation_reason") or "PEDS cycle marked promoted survivor for distillation.")
        metadata = option_values.get("distillation_metadata")
        decision = mark_distillation_candidate(
            conn,
            candidate_id,
            operator=operator_text,
            reason=reason,
            metadata=metadata if isinstance(metadata, dict) else None,
        )
        return _changed(conn, candidate_id, plan, "mark_distillation_candidate", {"promotion_decision": decision})
    if next_step == "build_distillation_manifest":
        return _advance_distillation_or_training_bridge(conn, candidate_id, plan, operator_text, option_values)
    if next_step == "create_candidate_from_training_run":
        return _advance_training_bridge(conn, candidate_id, plan, operator_text, option_values)

    return _blocked(candidate_id, plan, "unsupported_cycle_step", plan["blockers"] or [f"unsupported next_step: {next_step}"])


def run_peds_cycle_until_blocked(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    operator: str,
    auto_promote: bool = False,
    max_steps: int = 10,
    options: dict[str, Any] | None = None,
    replay_evaluator=None,
    canary_evaluator=None,
    shadow_event_provider=None,
) -> dict[str, Any]:
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 0:
        raise ValueError("max_steps must be a non-negative int")

    step_log: list[dict[str, Any]] = []
    current_candidate_id = candidate_id
    stopped_reason = "max_steps reached"

    for _ in range(max_steps):
        result = advance_peds_cycle(
            conn,
            current_candidate_id,
            operator=operator,
            auto_promote=auto_promote,
            options=options,
            replay_evaluator=replay_evaluator,
            canary_evaluator=canary_evaluator,
            shadow_event_provider=shadow_event_provider,
        )
        step_log.append(result)

        created_candidate = (result.get("changed") or {}).get("candidate")
        if isinstance(created_candidate, dict) and created_candidate.get("candidate_id"):
            current_candidate_id = created_candidate["candidate_id"]

        if result.get("blocked"):
            stopped_reason = "; ".join(result.get("blockers") or [str(result.get("action") or "stopped")])
            break
    else:
        if max_steps == 0:
            stopped_reason = "max_steps reached"

    return {
        "candidate_id": candidate_id,
        "final_candidate_id": current_candidate_id,
        "steps_run": len(step_log),
        "stopped_reason": stopped_reason,
        "step_log": step_log,
    }


def _advance_replay_stage(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    options: dict[str, Any],
    replay_evaluator,
) -> dict[str, Any]:
    job = _latest_replay_job_for_candidate(conn, candidate_id, split="replay", statuses=("queued",))
    created_job = None
    if job is None:
        job = _latest_replay_job_for_candidate(conn, candidate_id, split="replay")

    if job is None:
        try:
            created_job = create_post_training_replay_job(
                conn,
                candidate_id,
                incumbent_id=options.get("incumbent_id"),
                episode_ids=options.get("episode_ids") or options.get("replay_episode_ids"),
                limit=options.get("limit"),
                policy_tags=options.get("policy_tags"),
                evaluator_version=str(options.get("evaluator_version") or "post_training_replay_v1"),
                runtime=options.get("runtime") if isinstance(options.get("runtime"), dict) else None,
            )
        except ValueError as exc:
            return _blocked(candidate_id, plan, "create_replay_job", [str(exc)])
        job = _load_replay_job(conn, created_job["job_id"])

    changed = {"replay_job": job}
    if created_job is not None:
        changed["created_replay_job"] = created_job

    if replay_evaluator is None:
        return _changed(
            conn,
            candidate_id,
            plan,
            "create_replay_job" if created_job is not None else "await_replay_evaluator",
            changed,
            blocked=True,
            blockers=["replay_evaluator required"],
        )

    try:
        run = run_replay_job(
            conn,
            job["job_id"],
            replay_evaluator,
            write_decision=True,
            thresholds=options.get("replay_thresholds") if isinstance(options.get("replay_thresholds"), dict) else None,
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "run_replay_job", [str(exc)])
    changed["replay_run"] = run
    return _changed(conn, candidate_id, plan, "run_replay_job", changed)


def _advance_canary_stage(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    options: dict[str, Any],
    canary_evaluator,
) -> dict[str, Any]:
    job = _latest_replay_job_for_candidate(conn, candidate_id, split="canary", statuses=("queued",))
    created_job = None
    if job is None:
        job = _latest_replay_job_for_candidate(conn, candidate_id, split="canary")

    if job is not None:
        changed = {"canary_job_id": job["job_id"], "canary_job": job}
        if canary_evaluator is None:
            return _changed(
                conn,
                candidate_id,
                plan,
                "await_canary_evaluator",
                changed,
                blocked=True,
                blockers=["canary_evaluator required"],
            )
        try:
            run = run_canary_job(
                conn,
                job["job_id"],
                canary_evaluator,
                write_decision=True,
                thresholds=options.get("canary_thresholds") if isinstance(options.get("canary_thresholds"), dict) else None,
            )
        except ValueError as exc:
            return _blocked(candidate_id, plan, "run_canary_job", [str(exc)])
        changed["canary_run"] = run
        return _changed(conn, candidate_id, plan, "run_canary_job", changed)

    incumbent_id = options.get("incumbent_id") or (plan.get("details", {}).get("incumbent") or {}).get("incumbent_id")
    episode_ids = options.get("canary_episode_ids") or options.get("episode_ids")
    if not episode_ids:
        episode_ids = _select_episode_ids_by_split(conn, "canary", limit=options.get("canary_limit"))
    if not incumbent_id:
        return _blocked(candidate_id, plan, "create_canary_job", ["incumbent_id is required for canary job"])
    if not episode_ids:
        return _blocked(candidate_id, plan, "create_canary_job", ["at least one canary episode is required"])

    try:
        job_id = create_canary_job(
            conn,
            candidate_id=candidate_id,
            incumbent_id=str(incumbent_id),
            episode_ids=list(episode_ids),
            thresholds=options.get("canary_thresholds") if isinstance(options.get("canary_thresholds"), dict) else None,
            metadata=options.get("canary_metadata") if isinstance(options.get("canary_metadata"), dict) else None,
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "create_canary_job", [str(exc)])
    created_job = _load_replay_job(conn, job_id)
    changed = {"canary_job_id": job_id, "canary_job": created_job, "created_canary_job": created_job}

    if canary_evaluator is None:
        return _changed(
            conn,
            candidate_id,
            plan,
            "create_canary_job",
            changed,
            blocked=True,
            blockers=["canary_evaluator required"],
        )

    try:
        run = run_canary_job(
            conn,
            job_id,
            canary_evaluator,
            write_decision=True,
            thresholds=options.get("canary_thresholds") if isinstance(options.get("canary_thresholds"), dict) else None,
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "run_canary_job", [str(exc)])
    changed["canary_run"] = run
    return _changed(conn, candidate_id, plan, "run_canary_job", changed)


def _advance_shadow_stage(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    options: dict[str, Any],
    shadow_event_provider,
) -> dict[str, Any]:
    run = _active_shadow_run_for_candidate(conn, candidate_id)
    created_run = None
    if run is None:
        incumbent_id = options.get("incumbent_id") or (plan.get("details", {}).get("incumbent") or {}).get("incumbent_id")
        if not incumbent_id:
            return _blocked(candidate_id, plan, "start_shadow_run", ["incumbent_id is required for shadow run"])
        try:
            created_run = start_shadow_run(
                conn,
                candidate_id=candidate_id,
                incumbent_id=str(incumbent_id),
                thresholds=options.get("shadow_thresholds") if isinstance(options.get("shadow_thresholds"), dict) else None,
                metadata=options.get("shadow_metadata") if isinstance(options.get("shadow_metadata"), dict) else None,
                write_decision=bool(options.get("write_shadow_started", True)),
            )
        except ValueError as exc:
            return _blocked(candidate_id, plan, "start_shadow_run", [str(exc)])
        run = created_run

    changed: dict[str, Any] = {"shadow_run": run}
    if created_run is not None:
        changed["created_shadow_run"] = created_run

    if shadow_event_provider is None:
        return _changed(
            conn,
            candidate_id,
            plan,
            "start_shadow_run" if created_run is not None else "await_shadow_event_provider",
            changed,
            blocked=True,
            blockers=["shadow_event_provider required"],
        )

    try:
        event_payload = _call_shadow_event_provider(
            shadow_event_provider,
            conn=conn,
            candidate_id=candidate_id,
            incumbent_id=run["incumbent_id"],
            shadow_run=deepcopy(run),
            candidate=deepcopy(_load_candidate(conn, candidate_id)),
            incumbent=deepcopy(_load_candidate(conn, run["incumbent_id"])),
        )
        event_specs = _normalize_shadow_event_specs(event_payload, shadow_run_id=run["shadow_run_id"])
        recorded_events = [
            record_shadow_event(conn, shadow_run_id=run["shadow_run_id"], **event_spec)
            for event_spec in event_specs
        ]
    except (TypeError, ValueError) as exc:
        return _blocked(candidate_id, plan, "record_shadow_event", [str(exc)])

    summary = summarize_shadow_run(conn, run["shadow_run_id"])
    changed["shadow_events"] = recorded_events
    changed["shadow_summary"] = summary

    if bool(options.get("complete_shadow")) or _shadow_has_enough_events(summary):
        try:
            completion = complete_shadow_run(
                conn,
                run["shadow_run_id"],
                thresholds=options.get("shadow_thresholds") if isinstance(options.get("shadow_thresholds"), dict) else None,
                write_decision=True,
            )
        except ValueError as exc:
            return _blocked(candidate_id, plan, "complete_shadow_run", [str(exc)])
        changed["shadow_completion"] = completion
        return _changed(conn, candidate_id, plan, "complete_shadow_run", changed)

    blockers = ["not enough shadow events"]
    if not recorded_events:
        blockers.insert(0, "shadow_event_provider produced no events")
    return _changed(
        conn,
        candidate_id,
        plan,
        "record_shadow_event",
        changed,
        blocked=True,
        blockers=blockers,
    )


def _advance_distillation_or_training_bridge(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    operator: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    training_run_id = options.get("training_run_id")
    latest_completed = _latest_completed_training_run(conn, candidate_id)
    if training_run_id or latest_completed:
        training_run = get_training_run(conn, str(training_run_id)) if training_run_id else latest_completed
        if training_run["status"] == "completed":
            bridge_plan = dict(plan)
            bridge_plan["next_step"] = "create_candidate_from_training_run"
            bridge_plan.setdefault("details", {})["training_run"] = training_run
            return _advance_training_bridge(conn, candidate_id, bridge_plan, operator, options)

    try:
        manifest = write_distillation_manifest(
            conn,
            candidate_id,
            output_path=options.get("output_path"),
            max_examples=options.get("max_examples"),
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "build_distillation_manifest", [str(exc)])

    changed: dict[str, Any] = {"distillation_manifest": manifest, "output_path": options.get("output_path")}
    training_run = get_training_run(conn, str(training_run_id)) if training_run_id else None
    if training_run is None:
        trainer_name = options.get("trainer_name")
        if not trainer_name:
            return _changed(
                conn,
                candidate_id,
                plan,
                "build_distillation_manifest",
                changed,
                blocked=True,
                blockers=["trainer_name required"],
            )
        try:
            training_run = prepare_training_run(
                conn,
                manifest,
                trainer_name=str(trainer_name),
                trainer_config=options.get("trainer_config") if isinstance(options.get("trainer_config"), dict) else None,
                output_dir=options.get("training_output_dir"),
            )
        except ValueError as exc:
            return _blocked(candidate_id, plan, "prepare_training_run", [str(exc)])
        changed["training_run"] = training_run
    else:
        changed["training_run"] = training_run

    if not options.get("run_training"):
        return _changed(
            conn,
            candidate_id,
            plan,
            "prepare_training_run",
            changed,
            blocked=True,
            blockers=["run_training=True required"],
        )

    try:
        training_result = run_prepared_training_run(
            conn,
            training_run["training_run_id"],
            operator=operator,
            mode=str(options.get("trainer_mode") or "dry_run"),
            artifact_dir=options.get("artifact_dir") or options.get("training_artifact_dir"),
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "run_prepared_training_run", [str(exc)])
    changed["training_result"] = training_result

    if not options.get("adapter_name"):
        return _changed(
            conn,
            candidate_id,
            plan,
            "run_prepared_training_run",
            changed,
            blocked=True,
            blockers=["adapter_name option required"],
        )

    bridge_plan = dict(plan)
    bridge_plan["next_step"] = "create_candidate_from_training_run"
    bridge_plan.setdefault("details", {})["training_run"] = training_result["training_run"]
    bridge_result = _advance_training_bridge(conn, candidate_id, bridge_plan, operator, options)
    merged_changed = dict(changed)
    merged_changed.update(bridge_result["changed"])
    return {
        **bridge_result,
        "changed": merged_changed,
    }


def _advance_training_bridge(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    operator: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    training_run = plan.get("details", {}).get("training_run") or _latest_completed_training_run(conn, candidate_id)
    if training_run is None:
        return _blocked(candidate_id, plan, "create_candidate_from_training_run", ["completed training run is required"])
    adapter_name = options.get("adapter_name")
    if not adapter_name:
        return _blocked(candidate_id, plan, "create_candidate_from_training_run", ["adapter_name option required"])

    replay_probe = select_replay_episodes_for_candidate(
        conn,
        candidate_id,
        limit=options.get("limit"),
        policy_tags=options.get("policy_tags"),
    )
    if not replay_probe["episode_ids"] and not (options.get("episode_ids") or options.get("replay_episode_ids")):
        return _blocked(candidate_id, plan, "create_candidate_from_training_run", ["at least one replay episode is required"])

    try:
        candidate = create_candidate_from_training_run(
            conn,
            training_run["training_run_id"],
            parent_candidate_id=candidate_id,
            adapter_name=str(adapter_name),
            operator=operator,
            require_trained=bool(options.get("require_trained", False)),
            delta_description=options.get("delta_description"),
        )
        replay_job = create_post_training_replay_job(
            conn,
            candidate["candidate_id"],
            episode_ids=options.get("episode_ids") or options.get("replay_episode_ids"),
            limit=options.get("limit"),
            policy_tags=options.get("policy_tags"),
            evaluator_version=str(options.get("evaluator_version") or "post_training_replay_v1"),
            runtime=options.get("runtime") if isinstance(options.get("runtime"), dict) else None,
        )
    except ValueError as exc:
        return _blocked(candidate_id, plan, "create_candidate_from_training_run", [str(exc)])

    return _changed(
        conn,
        candidate_id,
        plan,
        "create_candidate_from_training_run",
        {"candidate": candidate, "replay_job": replay_job},
        blocked=True,
        blockers=["post-training candidate created; start a new cycle for generated candidate"],
    )


def _next_step_for_state(current_state: str | None, latest_completed_training_run: dict[str, Any] | None) -> str:
    if current_state == "distillation_candidate" and latest_completed_training_run is not None:
        return "create_candidate_from_training_run"
    if current_state is None:
        return "unknown"
    return _STATE_NEXT_STEPS.get(current_state, "unsupported_state")


def _load_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    candidate = dict(row)
    candidate["manifest"] = json.loads(candidate["manifest_json"])
    candidate["current_state"] = get_candidate_current_state(conn, candidate_id)
    return candidate


def _latest_completed_training_run(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT training_run_id
        FROM peds_training_runs
        WHERE candidate_id = ? AND status = 'completed'
        ORDER BY finished_at DESC, updated_at DESC, created_at DESC, training_run_id DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if not row:
        return None
    return get_training_run(conn, row["training_run_id"])


def _latest_incumbent_from_decision(conn: sqlite3.Connection, candidate_id: str, decision: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT incumbent_id, job_id, shadow_run_id, decision_id
        FROM promotion_decisions
        WHERE candidate_id = ? AND decision = ?
        ORDER BY created_at DESC, decision_id DESC
        LIMIT 1
        """,
        (candidate_id, decision),
    ).fetchone()
    if not row:
        return {
            "candidate_id": candidate_id,
            "incumbent_id": None,
            "strategy": f"latest_{decision}",
            "warnings": [f"missing {decision} decision"],
        }
    return {
        "candidate_id": candidate_id,
        "incumbent_id": row["incumbent_id"],
        "strategy": f"latest_{decision}",
        "decision_id": row["decision_id"],
        "job_id": row["job_id"],
        "shadow_run_id": row["shadow_run_id"],
        "warnings": [],
    }


def _episode_count(conn: sqlite3.Connection, split: str) -> int:
    row = conn.execute("SELECT COUNT(*) AS count FROM episodes WHERE split = ?", (split,)).fetchone()
    return int(row["count"])


def _select_episode_ids_by_split(conn: sqlite3.Connection, split: str, *, limit: int | None = None) -> list[str]:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 0):
        raise ValueError("limit must be a non-negative int or None")
    rows = conn.execute(
        """
        SELECT episode_id
        FROM episodes
        WHERE split = ?
        ORDER BY timestamp ASC, episode_id ASC
        """,
        (split,),
    ).fetchall()
    episode_ids = [row["episode_id"] for row in rows]
    return episode_ids[:limit] if limit is not None else episode_ids


def _load_replay_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ? LIMIT 1", (job_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown replay job: {job_id}")
    job = dict(row)
    job["runtime"] = json.loads(job.pop("runtime_json"))
    job["episode_ids"] = [
        episode_row["episode_id"]
        for episode_row in conn.execute(
            "SELECT episode_id FROM replay_job_episodes WHERE job_id = ? ORDER BY episode_order",
            (job_id,),
        ).fetchall()
    ]
    return job


def _latest_replay_job_for_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    split: str,
    statuses: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    parameters: list[Any] = [candidate_id, split]
    status_clause = ""
    if statuses is not None:
        if not statuses:
            return None
        status_clause = " AND status IN (" + ", ".join("?" for _ in statuses) + ")"
        parameters.extend(statuses)
    row = conn.execute(
        f"""
        SELECT job_id
        FROM replay_jobs
        WHERE candidate_id = ? AND split = ?{status_clause}
        ORDER BY created_at DESC, job_id DESC
        LIMIT 1
        """,
        parameters,
    ).fetchone()
    if not row:
        return None
    return _load_replay_job(conn, row["job_id"])


def _active_shadow_run_for_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT shadow_run_id
        FROM shadow_runs
        WHERE candidate_id = ? AND status = 'active'
        ORDER BY started_at DESC, shadow_run_id DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if not row:
        return None
    return _load_shadow_run(conn, row["shadow_run_id"])


def _load_shadow_run(conn: sqlite3.Connection, shadow_run_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM shadow_runs WHERE shadow_run_id = ? LIMIT 1", (shadow_run_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown shadow run: {shadow_run_id}")
    run = dict(row)
    run["thresholds"] = json.loads(run.pop("thresholds_json"))
    run["metadata"] = json.loads(run.pop("metadata_json"))
    return run


def _call_shadow_event_provider(shadow_event_provider, **kwargs: Any) -> Any:
    if not callable(shadow_event_provider):
        raise ValueError("shadow_event_provider must be callable")
    try:
        signature = inspect.signature(shadow_event_provider)
    except (TypeError, ValueError):
        return shadow_event_provider(**kwargs)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return shadow_event_provider(**kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in parameters
    }
    return shadow_event_provider(**accepted)


def _normalize_shadow_event_specs(value: Any, *, shadow_run_id: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict) and "events" in value:
        raw_events = value["events"]
    elif isinstance(value, dict):
        raw_events = [value]
    elif isinstance(value, (list, tuple)):
        raw_events = list(value)
    else:
        try:
            raw_events = list(value)
        except TypeError as exc:
            raise ValueError("shadow_event_provider must return an event dict or iterable of event dicts") from exc

    normalized: list[dict[str, Any]] = []
    allowed_keys = {
        "input",
        "incumbent_output",
        "candidate_output",
        "divergence",
        "candidate_would_act",
        "incumbent_did_act",
        "accepted_by_incumbent",
        "task_ref",
        "episode_ref",
        "metadata",
    }
    required_keys = {"input", "incumbent_output", "candidate_output"}
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, dict):
            raise ValueError(f"shadow event {index} must be a dict")
        event = dict(raw_event)
        supplied_shadow_run_id = event.pop("shadow_run_id", shadow_run_id)
        if supplied_shadow_run_id != shadow_run_id:
            raise ValueError(f"shadow event {index} references a different shadow_run_id")
        missing = sorted(required_keys - set(event))
        if missing:
            raise ValueError(f"shadow event {index} missing required keys: {', '.join(missing)}")
        unknown = sorted(set(event) - allowed_keys)
        if unknown:
            raise ValueError(f"shadow event {index} has unsupported keys: {', '.join(unknown)}")
        normalized.append(event)
    return normalized


def _shadow_has_enough_events(summary: dict[str, Any]) -> bool:
    thresholds = summary.get("thresholds") or {}
    min_events = int(thresholds.get("min_events", 10))
    return int(summary.get("n_events", 0)) >= min_events


def _can_override_plan_blockers(next_step: str, options: dict[str, Any]) -> bool:
    if next_step == "create_replay_job":
        return True
    if next_step == "create_canary_job" and (options.get("episode_ids") or options.get("canary_episode_ids")):
        return True
    if next_step == "create_candidate_from_training_run" and options.get("adapter_name"):
        return True
    return False


def _blocker_action(next_step: str) -> str:
    if next_step in {"unknown", "unsupported_state"}:
        return "unsupported_candidate_state"
    return f"await_{next_step}"


def _normalize_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("options must be a dict or None")
    return dict(options)


def _require_non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _changed(
    conn: sqlite3.Connection,
    candidate_id: str,
    plan: dict[str, Any],
    action: str,
    changed: dict[str, Any],
    *,
    blocked: bool = False,
    blockers: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "state_before": plan.get("current_state"),
        "state_after": get_candidate_current_state(conn, candidate_id),
        "action": action,
        "changed": changed,
        "blocked": blocked,
        "blockers": blockers or [],
        "plan": plan,
    }


def _blocked(
    candidate_id: str,
    plan: dict[str, Any],
    action: str,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "state_before": plan.get("current_state"),
        "state_after": plan.get("current_state"),
        "action": action,
        "changed": {},
        "blocked": True,
        "blockers": blockers,
        "plan": plan,
    }
