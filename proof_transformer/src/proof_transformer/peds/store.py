from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from ..ids import new_id
from ..timeutils import utc_now


def _canonical_json(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            pass
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row else None


def _require_candidate(conn: sqlite3.Connection, candidate_id: str) -> None:
    row = conn.execute(
        "SELECT candidate_id FROM candidates WHERE candidate_id = ? LIMIT 1",
        (candidate_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")


def _require_episode(conn: sqlite3.Connection, episode_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT episode_id, split FROM episodes WHERE episode_id = ? LIMIT 1",
        (episode_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown episode: {episode_id}")
    return row


def _require_job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT job_id, candidate_id, incumbent_id, split FROM replay_jobs WHERE job_id = ? LIMIT 1",
        (job_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown replay job: {job_id}")
    return row


def _require_shadow_run(conn: sqlite3.Connection, shadow_run_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT shadow_run_id, candidate_id, incumbent_id, status FROM shadow_runs WHERE shadow_run_id = ? LIMIT 1",
        (shadow_run_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown shadow run: {shadow_run_id}")
    return row


def _unique_episode_ids(episode_ids: Sequence[str]) -> list[str]:
    if not episode_ids:
        raise ValueError("episode_ids must not be empty")
    seen: set[str] = set()
    ordered: list[str] = []
    for episode_id in episode_ids:
        if episode_id in seen:
            raise ValueError(f"Duplicate episode_id in replay set: {episode_id}")
        seen.add(episode_id)
        ordered.append(episode_id)
    return ordered


def _ordered_append_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{stamp}_{new_id('evt').split('_', 1)[1]}"


def create_candidate(conn: sqlite3.Connection, manifest: dict[str, Any]) -> dict:
    payload = deepcopy(manifest)
    candidate_id = str(payload.get("candidate_id") or new_id("candidate"))
    parent_candidate_id = payload.get("parent_candidate_id")
    created_at = payload.get("created_at") or utc_now()
    mutation_target = payload["mutation_target"]
    phi_version = payload["phi_version"]
    rho_version = payload["rho_version"]
    theta_version = payload.get("theta_version")
    psi_version = payload["psi_version"]
    delta_description = payload["delta_description"]

    if parent_candidate_id:
        _require_candidate(conn, parent_candidate_id)

    manifest_body = payload.get("manifest_json")
    if manifest_body is None:
        manifest_body = {k: v for k, v in payload.items() if k != "manifest_json"}
        manifest_body.setdefault("candidate_id", candidate_id)
        manifest_body.setdefault("created_at", created_at)

    with conn:
        conn.execute(
            """
            INSERT INTO candidates(
                candidate_id, parent_candidate_id, created_at, mutation_target,
                phi_version, rho_version, theta_version, psi_version,
                delta_description, manifest_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                parent_candidate_id,
                created_at,
                mutation_target,
                phi_version,
                rho_version,
                theta_version,
                psi_version,
                delta_description,
                _canonical_json(manifest_body),
            ),
        )

    row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
    result = dict(row)
    result["manifest"] = _decode_json(result["manifest_json"])
    result["current_state"] = get_candidate_current_state(conn, candidate_id)
    return result


def record_episode(conn: sqlite3.Connection, episode: dict[str, Any]) -> dict:
    payload = deepcopy(episode)

    task = payload.get("task") or {}
    snapshot = payload.get("snapshot") or {}
    candidate = payload.get("candidate") or {}
    run = payload.get("run") or {}
    outcome = payload.get("outcome") or {}
    metrics = payload.get("metrics") or {}
    privacy = payload.get("privacy") or {}

    episode_id = str(payload.get("episode_id") or new_id("episode"))
    schema_version = payload.get("schema_version") or "peds.episode.v1"
    split = payload.get("split") or "online"
    timestamp = payload.get("timestamp") or utc_now()
    task_id = payload.get("task_id") or task.get("task_id") or new_id("task")
    task_title = payload.get("task_title") or task.get("title")
    task_prompt = payload.get("task_prompt") or task.get("prompt")
    if not task_prompt:
        raise ValueError("episode task prompt is required")

    candidate_id = payload.get("candidate_id") or candidate.get("candidate_id")
    if candidate_id:
        _require_candidate(conn, candidate_id)

    acceptance_tests = payload.get("acceptance_tests") or task.get("acceptance_tests")
    policy_tags = payload.get("policy_tags") or task.get("policy_tags")
    model_lock = payload.get("model_lock") or snapshot.get("model_lock") or {}
    context_selected = payload.get("context_selected") or run.get("context_selected") or []
    actions = payload.get("actions") or run.get("actions") or []

    raw_payload = deepcopy(payload)
    raw_payload.setdefault("episode_id", episode_id)
    raw_payload.setdefault("schema_version", schema_version)
    raw_payload.setdefault("split", split)
    raw_payload.setdefault("timestamp", timestamp)

    with conn:
        conn.execute(
            """
            INSERT INTO episodes(
                episode_id, schema_version, split, timestamp,
                task_id, task_title, task_prompt, acceptance_tests_json, policy_tags_json,
                repo_commit, repo_tree_hash, retrieval_index_hash, env_image_hash,
                tool_profile, model_lock_json, candidate_id, context_json, actions_json,
                patch, final_message, judge_verdict, user_verdict,
                failure_reason, repair_trace, metrics_json, privacy_json, raw_episode_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                schema_version,
                split,
                timestamp,
                task_id,
                task_title,
                task_prompt,
                _canonical_json(acceptance_tests) if acceptance_tests is not None else None,
                _canonical_json(policy_tags) if policy_tags is not None else None,
                payload.get("repo_commit") or snapshot.get("repo_commit"),
                payload.get("repo_tree_hash") or snapshot.get("repo_tree_hash"),
                payload.get("retrieval_index_hash") or snapshot.get("retrieval_index_hash"),
                payload.get("env_image_hash") or snapshot.get("env_image_hash"),
                payload.get("tool_profile") or snapshot.get("tool_profile"),
                _canonical_json(model_lock),
                candidate_id,
                _canonical_json(context_selected),
                _canonical_json(actions),
                payload.get("patch") or run.get("patch"),
                payload.get("final_message") or run.get("final_message"),
                payload.get("judge_verdict") or outcome.get("judge_verdict"),
                payload.get("user_verdict") or outcome.get("user_verdict"),
                payload.get("failure_reason") or outcome.get("failure_reason"),
                payload.get("repair_trace") or outcome.get("repair_trace"),
                _canonical_json(metrics),
                _canonical_json(privacy),
                _canonical_json(raw_payload),
            ),
        )

    row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
    result = dict(row)
    result["acceptance_tests"] = _decode_json(result["acceptance_tests_json"]) if result["acceptance_tests_json"] else None
    result["policy_tags"] = _decode_json(result["policy_tags_json"]) if result["policy_tags_json"] else None
    result["model_lock"] = _decode_json(result["model_lock_json"])
    result["context"] = _decode_json(result["context_json"])
    result["actions"] = _decode_json(result["actions_json"])
    result["metrics"] = _decode_json(result["metrics_json"])
    result["privacy"] = _decode_json(result["privacy_json"])
    result["raw_episode"] = _decode_json(result["raw_episode_json"])
    return result


def create_replay_job(
    conn: sqlite3.Connection,
    candidate_id: str,
    incumbent_id: str,
    episode_ids: Sequence[str],
    *,
    split: str = "replay",
    evaluator_version: str = "eval_v1",
    runtime: dict[str, Any] | None = None,
    job_id: str | None = None,
    created_at: str | None = None,
    status: str = "queued",
) -> dict:
    _require_candidate(conn, candidate_id)
    _require_candidate(conn, incumbent_id)
    ordered_episode_ids = _unique_episode_ids(episode_ids)
    for episode_id in ordered_episode_ids:
        episode = _require_episode(conn, episode_id)
        if episode["split"] != split:
            raise ValueError(
                f"Replay job split {split!r} cannot include episode {episode_id!r} with split {episode['split']!r}"
            )

    replay_job_id = job_id or new_id("replay_job")
    created = created_at or utc_now()
    runtime_payload = runtime or {}

    with conn:
        conn.execute(
            """
            INSERT INTO replay_jobs(
                job_id, created_at, started_at, finished_at, split,
                candidate_id, incumbent_id, evaluator_version, runtime_json, status
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                replay_job_id,
                created,
                None,
                None,
                split,
                candidate_id,
                incumbent_id,
                evaluator_version,
                _canonical_json(runtime_payload),
                status,
            ),
        )
        conn.executemany(
            "INSERT INTO replay_job_episodes(job_id, episode_id, episode_order) VALUES(?, ?, ?)",
            [(replay_job_id, episode_id, index) for index, episode_id in enumerate(ordered_episode_ids)],
        )

    row = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ?", (replay_job_id,)).fetchone()
    result = dict(row)
    result["runtime"] = _decode_json(result["runtime_json"])
    result["episode_ids"] = ordered_episode_ids
    return result


def record_replay_result(conn: sqlite3.Connection, result: dict[str, Any]) -> dict:
    payload = deepcopy(result)
    result_id = str(payload.get("result_id") or new_id("replay_result"))
    job_id = payload["job_id"]
    episode_id = payload["episode_id"]
    _require_job(conn, job_id)
    membership = conn.execute(
        "SELECT 1 FROM replay_job_episodes WHERE job_id = ? AND episode_id = ? LIMIT 1",
        (job_id, episode_id),
    ).fetchone()
    if not membership:
        raise ValueError(f"Episode {episode_id} is not frozen into replay job {job_id}")

    candidate_score = float(payload["candidate_score"])
    incumbent_score = float(payload["incumbent_score"])
    delta = float(payload.get("delta", candidate_score - incumbent_score))
    judge_verdict = payload["judge_verdict"]
    canary_failures = payload.get("canary_failures") or []
    created_at = payload.get("created_at") or utc_now()

    with conn:
        conn.execute(
            """
            INSERT INTO replay_results(
                result_id, job_id, episode_id, candidate_score, incumbent_score,
                delta, judge_verdict, canary_failures_json, audit_log_ref, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result_id,
                job_id,
                episode_id,
                candidate_score,
                incumbent_score,
                delta,
                judge_verdict,
                _canonical_json(canary_failures),
                payload.get("audit_log_ref"),
                created_at,
            ),
        )

    row = conn.execute("SELECT * FROM replay_results WHERE result_id = ?", (result_id,)).fetchone()
    stored = dict(row)
    stored["canary_failures"] = _decode_json(stored["canary_failures_json"])
    return stored


def record_promotion_decision(conn: sqlite3.Connection, decision: dict[str, Any]) -> dict:
    payload = deepcopy(decision)
    decision_id = str(payload.get("decision_id") or _ordered_append_id("promotion_decision"))
    candidate_id = payload["candidate_id"]
    incumbent_id = payload["incumbent_id"]
    job_id = payload.get("job_id")
    shadow_run_id = payload.get("shadow_run_id")

    if bool(job_id) == bool(shadow_run_id):
        raise ValueError("Promotion decision must reference exactly one of job_id or shadow_run_id")

    if job_id:
        job = _require_job(conn, job_id)
        if job["candidate_id"] != candidate_id or job["incumbent_id"] != incumbent_id:
            raise ValueError("Promotion decision candidate/incumbent must match the replay job")
    if shadow_run_id:
        shadow_run = _require_shadow_run(conn, shadow_run_id)
        if shadow_run["candidate_id"] != candidate_id or shadow_run["incumbent_id"] != incumbent_id:
            raise ValueError("Promotion decision candidate/incumbent must match the shadow run")

    created_at = payload.get("created_at") or utc_now()

    with conn:
        conn.execute(
            """
            INSERT INTO promotion_decisions(
                decision_id, candidate_id, incumbent_id, job_id, shadow_run_id, decision,
                mean_delta, lcb95, regression_rate, canary_failures,
                token_ratio, latency_ratio, reason, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                candidate_id,
                incumbent_id,
                job_id,
                shadow_run_id,
                payload["decision"],
                payload.get("mean_delta"),
                payload.get("lcb95"),
                payload.get("regression_rate"),
                payload.get("canary_failures"),
                payload.get("token_ratio"),
                payload.get("latency_ratio"),
                payload["reason"],
                created_at,
            ),
        )

    row = conn.execute("SELECT * FROM promotion_decisions WHERE decision_id = ?", (decision_id,)).fetchone()
    stored = dict(row)
    stored["current_state"] = get_candidate_current_state(conn, candidate_id)
    return stored


def get_candidate_current_state(conn: sqlite3.Connection, candidate_id: str) -> str | None:
    row = conn.execute(
        "SELECT current_state FROM candidate_current_state WHERE candidate_id = ? LIMIT 1",
        (candidate_id,),
    ).fetchone()
    return row["current_state"] if row else None
