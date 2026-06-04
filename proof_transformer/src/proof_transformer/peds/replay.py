from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from ..timeutils import utc_now
from .scoring import evaluate_replay_job
from .store import get_candidate_current_state, record_replay_result

ReplayEvaluator = Callable[..., dict[str, Any]]

_DEFAULT_THRESHOLDS = {
    "epsilon": 0.02,
    "max_regression_rate": 0.25,
    "min_episodes": 3,
    "max_token_ratio": None,
    "max_latency_ratio": None,
}
_VALID_JUDGE_VERDICTS = {"pass", "fail", "unsafe", "timeout"}
_SENTINEL = object()


def load_replay_job_bundle(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    job = _load_job(conn, job_id)
    candidate = _load_candidate(conn, job["candidate_id"])
    incumbent = _load_candidate(conn, job["incumbent_id"])

    rows = conn.execute(
        """
        SELECT rje.episode_order, e.*
        FROM replay_job_episodes rje
        JOIN episodes e ON e.episode_id = rje.episode_id
        WHERE rje.job_id = ?
        ORDER BY rje.episode_order, e.episode_id
        """,
        (job_id,),
    ).fetchall()

    episodes: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        record = dict(row)
        episodes.append(
            {
                "episode_id": record["episode_id"],
                "ordinal": ordinal,
                "episode_order": record["episode_order"],
                "episode": json.loads(record["raw_episode_json"]),
                "record": _decode_episode_record(record),
            }
        )

    return {
        "job": job,
        "candidate": candidate,
        "incumbent": incumbent,
        "episodes": episodes,
    }


def run_replay_job(
    conn: sqlite3.Connection,
    job_id: str,
    evaluator: ReplayEvaluator,
    *,
    write_decision: bool = False,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    threshold_values = _normalize_thresholds(thresholds)
    execution = _execute_frozen_job(conn, job_id, evaluator, required_split="replay")
    evaluation = evaluate_replay_job(
        conn,
        job_id,
        write_decision=write_decision,
        **threshold_values,
    )
    promotion_decision = evaluation["promotion_decision"]

    return {
        "job_id": job_id,
        "candidate_id": execution["candidate"]["candidate_id"],
        "incumbent_id": execution["incumbent"]["candidate_id"],
        "episodes_run": len(execution["episodes"]),
        "results_written": len(execution["results"]),
        "aggregate": evaluation["aggregate"],
        "gate": evaluation["gate"],
        "promotion_decision_id": promotion_decision["decision_id"] if promotion_decision else None,
        "status": execution["job"]["status"],
    }


def _execute_frozen_job(
    conn: sqlite3.Connection,
    job_id: str,
    evaluator: ReplayEvaluator,
    *,
    required_split: str,
) -> dict[str, Any]:
    bundle = load_replay_job_bundle(conn, job_id)
    _assert_job_runnable(conn, bundle["job"], bundle["episodes"], required_split=required_split)

    _mark_job_status(conn, job_id, "running", started_at=utc_now(), finished_at=None)

    normalized_results: list[dict[str, Any]] = []
    try:
        for episode_entry in bundle["episodes"]:
            evaluator_result = evaluator(
                episode=deepcopy(episode_entry["episode"]),
                candidate=deepcopy(bundle["candidate"]),
                incumbent=deepcopy(bundle["incumbent"]),
                job=deepcopy(bundle["job"]),
                ordinal=episode_entry["ordinal"],
            )
            normalized_results.append(
                _normalize_evaluator_result(
                    evaluator_result,
                    job_id=job_id,
                    episode_id=episode_entry["episode_id"],
                )
            )
    except Exception:
        _mark_job_status(conn, job_id, "failed", finished_at=utc_now())
        raise

    stored_results: list[dict[str, Any]] = []
    try:
        for result in normalized_results:
            stored_results.append(record_replay_result(conn, result))
    except Exception:
        _mark_job_status(conn, job_id, "failed", finished_at=utc_now())
        raise

    _mark_job_status(conn, job_id, "completed", finished_at=utc_now())
    return {
        "job": _load_job(conn, job_id),
        "candidate": bundle["candidate"],
        "incumbent": bundle["incumbent"],
        "episodes": bundle["episodes"],
        "results": stored_results,
    }


def _assert_job_runnable(
    conn: sqlite3.Connection,
    job: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    required_split: str,
) -> None:
    if job["split"] != required_split:
        raise ValueError(f"Replay job {job['job_id']} must have split {required_split!r}, got {job['split']!r}")
    if job["status"] != "queued":
        raise ValueError(f"Replay job {job['job_id']} is not runnable from status {job['status']!r}")
    existing = conn.execute(
        "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
        (job["job_id"],),
    ).fetchone()
    if existing and int(existing["count"]) > 0:
        raise ValueError(f"Replay job {job['job_id']} already has recorded results")
    if not episodes:
        raise ValueError(f"Replay job {job['job_id']} has no frozen episodes")


def _load_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ? LIMIT 1", (job_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown replay job: {job_id}")
    job = dict(row)
    job["runtime"] = json.loads(job.pop("runtime_json"))
    return job


def _mark_job_status(
    conn: sqlite3.Connection,
    job_id: str,
    status: str,
    *,
    started_at: str | None | object = _SENTINEL,
    finished_at: str | None | object = _SENTINEL,
) -> None:
    assignments = ["status = ?"]
    parameters: list[Any] = [status]

    if started_at is not _SENTINEL:
        assignments.append("started_at = ?")
        parameters.append(started_at)
    if finished_at is not _SENTINEL:
        assignments.append("finished_at = ?")
        parameters.append(finished_at)

    parameters.append(job_id)
    with conn:
        conn.execute(f"UPDATE replay_jobs SET {', '.join(assignments)} WHERE job_id = ?", parameters)


def _load_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    candidate = dict(row)
    candidate["manifest"] = json.loads(candidate["manifest_json"])
    candidate["current_state"] = get_candidate_current_state(conn, candidate_id)
    return candidate


def _decode_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(record)
    decoded["acceptance_tests"] = json.loads(decoded["acceptance_tests_json"]) if decoded.get("acceptance_tests_json") else None
    decoded["policy_tags"] = json.loads(decoded["policy_tags_json"]) if decoded.get("policy_tags_json") else None
    decoded["model_lock"] = json.loads(decoded["model_lock_json"])
    decoded["context"] = json.loads(decoded["context_json"])
    decoded["actions"] = json.loads(decoded["actions_json"])
    decoded["metrics"] = json.loads(decoded["metrics_json"])
    decoded["privacy"] = json.loads(decoded["privacy_json"])
    decoded["raw_episode"] = json.loads(decoded["raw_episode_json"])
    return decoded


def _normalize_evaluator_result(result: dict[str, Any], *, job_id: str, episode_id: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("Replay evaluator must return a dict")

    missing = [key for key in ("candidate_score", "incumbent_score") if key not in result]
    if missing:
        raise ValueError(f"Replay evaluator result missing required keys: {', '.join(missing)}")

    candidate_score = _coerce_float(result["candidate_score"], name="candidate_score")
    incumbent_score = _coerce_float(result["incumbent_score"], name="incumbent_score")
    delta = _coerce_float(result.get("delta", candidate_score - incumbent_score), name="delta")

    judge_verdict = str(result.get("judge_verdict") or "pass")
    if judge_verdict not in _VALID_JUDGE_VERDICTS:
        raise ValueError(f"Unsupported judge_verdict: {judge_verdict}")

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "delta": delta,
        "judge_verdict": judge_verdict,
        "canary_failures": _normalize_canary_failures(result.get("canary_failures")),
        "audit_log_ref": result.get("audit_log_ref"),
    }


def _normalize_canary_failures(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return deepcopy(value)
    if isinstance(value, dict):
        return [deepcopy(value)]
    if isinstance(value, bool):
        raise ValueError("Replay evaluator canary_failures must not be a boolean")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("Replay evaluator canary_failures must not be negative")
        return [f"canary_failure_{index + 1}" for index in range(value)]
    raise ValueError("Replay evaluator canary_failures must be a list, dict, int, or None")


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Replay evaluator {name} must be numeric") from exc


def _normalize_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    if thresholds is None:
        return dict(_DEFAULT_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict or None")

    unknown = sorted(set(thresholds) - set(_DEFAULT_THRESHOLDS))
    if unknown:
        raise ValueError(f"Unknown replay threshold keys: {', '.join(unknown)}")

    normalized = dict(_DEFAULT_THRESHOLDS)
    normalized.update(thresholds)
    return normalized
