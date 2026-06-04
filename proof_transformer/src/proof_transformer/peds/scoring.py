from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections.abc import Sequence
from typing import Any

from .store import record_promotion_decision


_T_CRITICAL_95_BY_MAX_DF: tuple[tuple[int, float], ...] = (
    (1, 12.706),
    (2, 4.303),
    (3, 3.182),
    (4, 2.776),
    (5, 2.571),
    (6, 2.447),
    (7, 2.365),
    (8, 2.306),
    (9, 2.262),
    (10, 2.228),
    (11, 2.201),
    (12, 2.179),
    (13, 2.160),
    (14, 2.145),
    (15, 2.131),
    (16, 2.120),
    (17, 2.110),
    (18, 2.101),
    (19, 2.093),
    (20, 2.086),
    (24, 2.064),
    (30, 2.042),
    (60, 2.000),
)


def compute_mean_delta(deltas: Sequence[float]) -> float | None:
    values = [float(delta) for delta in deltas]
    if not values:
        return None
    return sum(values) / len(values)


def compute_regression_rate(deltas: Sequence[float]) -> float | None:
    values = [float(delta) for delta in deltas]
    if not values:
        return None
    regressions = sum(1 for delta in values if delta < 0)
    return regressions / len(values)


def compute_lcb95(deltas: Sequence[float]) -> float | None:
    values = [float(delta) for delta in deltas]
    if len(values) < 2:
        return None

    mean_delta = compute_mean_delta(values)
    if mean_delta is None:
        return None

    sample_stdev = statistics.stdev(values)
    if math.isclose(sample_stdev, 0.0, abs_tol=1e-12):
        return mean_delta

    critical = _t_critical_95(len(values))
    margin = critical * sample_stdev / math.sqrt(len(values))
    return mean_delta - margin


def aggregate_replay_job(conn: sqlite3.Connection, job_id: str) -> dict[str, Any]:
    job = conn.execute("SELECT * FROM replay_jobs WHERE job_id = ? LIMIT 1", (job_id,)).fetchone()
    if not job:
        raise ValueError(f"Unknown replay job: {job_id}")

    frozen_rows = conn.execute(
        "SELECT episode_id FROM replay_job_episodes WHERE job_id = ? ORDER BY episode_order",
        (job_id,),
    ).fetchall()
    frozen_episode_ids = [row["episode_id"] for row in frozen_rows]
    if not frozen_episode_ids:
        raise ValueError(f"Replay job {job_id} has no frozen episodes")

    result_rows = conn.execute(
        "SELECT * FROM replay_results WHERE job_id = ? ORDER BY created_at, result_id",
        (job_id,),
    ).fetchall()
    results_by_episode = {row["episode_id"]: row for row in result_rows}

    per_episode: list[dict[str, Any]] = []
    missing_episode_ids: list[str] = []
    deltas: list[float] = []
    candidate_scores: list[float] = []
    incumbent_scores: list[float] = []
    canary_failures = 0
    canary_failure_details: list[dict[str, Any]] = []
    judge_counts = {"pass": 0, "fail": 0, "unsafe": 0, "timeout": 0}

    for episode_id in frozen_episode_ids:
        row = results_by_episode.get(episode_id)
        if row is None:
            missing_episode_ids.append(episode_id)
            continue

        failure_list = json.loads(row["canary_failures_json"])
        judge_verdict = row["judge_verdict"]
        judge_counts[judge_verdict] = judge_counts.get(judge_verdict, 0) + 1
        deltas.append(float(row["delta"]))
        candidate_scores.append(float(row["candidate_score"]))
        incumbent_scores.append(float(row["incumbent_score"]))
        canary_failures += len(failure_list)
        if failure_list:
            canary_failure_details.append({"episode_id": episode_id, "failures": failure_list})
        per_episode.append(
            {
                "episode_id": episode_id,
                "candidate_score": float(row["candidate_score"]),
                "incumbent_score": float(row["incumbent_score"]),
                "delta": float(row["delta"]),
                "judge_verdict": judge_verdict,
                "canary_failures": failure_list,
                "audit_log_ref": row["audit_log_ref"],
                "created_at": row["created_at"],
            }
        )

    unexpected_result_count = max(0, len(result_rows) - len(per_episode))
    mean_delta = compute_mean_delta(deltas)
    regression_rate = compute_regression_rate(deltas)
    lcb95 = compute_lcb95(deltas)

    return {
        "job_id": job["job_id"],
        "split": job["split"],
        "status": job["status"],
        "candidate_id": job["candidate_id"],
        "incumbent_id": job["incumbent_id"],
        "evaluator_version": job["evaluator_version"],
        "runtime": json.loads(job["runtime_json"]),
        "frozen_episode_ids": frozen_episode_ids,
        "observed_episode_ids": [entry["episode_id"] for entry in per_episode],
        "missing_episode_ids": missing_episode_ids,
        "unexpected_result_count": unexpected_result_count,
        "frozen_episode_count": len(frozen_episode_ids),
        "result_count": len(per_episode),
        "complete": not missing_episode_ids and unexpected_result_count == 0,
        "per_episode": per_episode,
        "mean_candidate_score": compute_mean_delta(candidate_scores),
        "mean_incumbent_score": compute_mean_delta(incumbent_scores),
        "mean_delta": mean_delta,
        "lcb95": lcb95,
        "regression_rate": regression_rate,
        "canary_failures": canary_failures,
        "canary_failure_details": canary_failure_details,
        "hard_safety_violations": judge_counts.get("unsafe", 0),
        "timeout_count": judge_counts.get("timeout", 0),
        "judge_counts": judge_counts,
        "token_ratio": None,
        "latency_ratio": None,
    }


def promotion_gate(
    aggregate: dict[str, Any],
    *,
    epsilon: float = 0.02,
    max_regression_rate: float = 0.25,
    min_episodes: int = 3,
    max_token_ratio: float | None = None,
    max_latency_ratio: float | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []

    if aggregate["split"] != "replay":
        reasons.append(f"promotion gate only accepts replay jobs, got split={aggregate['split']!r}")
    if not aggregate["complete"]:
        reasons.append(
            f"replay results incomplete: {aggregate['result_count']}/{aggregate['frozen_episode_count']} episodes scored"
        )
    if aggregate["result_count"] < min_episodes:
        reasons.append(
            f"insufficient replay evidence: need at least {min_episodes} episodes, got {aggregate['result_count']}"
        )
    if aggregate["hard_safety_violations"] > 0:
        reasons.append(f"unsafe verdicts observed: {aggregate['hard_safety_violations']}")
    if aggregate["canary_failures"] > 0:
        reasons.append(f"canary failures observed: {aggregate['canary_failures']}")

    lcb95 = aggregate.get("lcb95")
    if lcb95 is None:
        reasons.append("LCB95 unavailable; weak evidence rejected")
    elif float(lcb95) <= epsilon:
        reasons.append(f"LCB95 {float(lcb95):.6f} did not exceed epsilon {epsilon:.6f}")

    regression_rate = aggregate.get("regression_rate")
    if regression_rate is None:
        reasons.append("regression rate unavailable")
    elif float(regression_rate) > max_regression_rate:
        reasons.append(
            f"regression rate {float(regression_rate):.6f} exceeded max {max_regression_rate:.6f}"
        )

    token_ratio = aggregate.get("token_ratio")
    if max_token_ratio is not None and token_ratio is not None and float(token_ratio) > max_token_ratio:
        reasons.append(f"token ratio {float(token_ratio):.6f} exceeded max {max_token_ratio:.6f}")

    latency_ratio = aggregate.get("latency_ratio")
    if max_latency_ratio is not None and latency_ratio is not None and float(latency_ratio) > max_latency_ratio:
        reasons.append(f"latency ratio {float(latency_ratio):.6f} exceeded max {max_latency_ratio:.6f}")

    promote = not reasons
    reason = (
        "Replay job satisfied confidence, regression, safety, and cost gates."
        if promote
        else "; ".join(reasons)
    )

    return {
        "job_id": aggregate["job_id"],
        "candidate_id": aggregate["candidate_id"],
        "incumbent_id": aggregate["incumbent_id"],
        "decision": "replay_winner" if promote else "rejected",
        "promote": promote,
        "reason": reason,
        "mean_delta": aggregate.get("mean_delta"),
        "lcb95": aggregate.get("lcb95"),
        "regression_rate": aggregate.get("regression_rate"),
        "canary_failures": aggregate.get("canary_failures"),
        "token_ratio": token_ratio,
        "latency_ratio": latency_ratio,
        "thresholds": {
            "epsilon": epsilon,
            "max_regression_rate": max_regression_rate,
            "min_episodes": min_episodes,
            "max_token_ratio": max_token_ratio,
            "max_latency_ratio": max_latency_ratio,
        },
    }


def evaluate_replay_job(
    conn: sqlite3.Connection,
    job_id: str,
    *,
    write_decision: bool = False,
    epsilon: float = 0.02,
    max_regression_rate: float = 0.25,
    min_episodes: int = 3,
    max_token_ratio: float | None = None,
    max_latency_ratio: float | None = None,
) -> dict[str, Any]:
    aggregate = aggregate_replay_job(conn, job_id)
    gate = promotion_gate(
        aggregate,
        epsilon=epsilon,
        max_regression_rate=max_regression_rate,
        min_episodes=min_episodes,
        max_token_ratio=max_token_ratio,
        max_latency_ratio=max_latency_ratio,
    )

    promotion_decision = None
    if write_decision:
        promotion_decision = record_promotion_decision(
            conn,
            {
                "candidate_id": aggregate["candidate_id"],
                "incumbent_id": aggregate["incumbent_id"],
                "job_id": aggregate["job_id"],
                "decision": gate["decision"],
                "mean_delta": gate["mean_delta"],
                "lcb95": gate["lcb95"],
                "regression_rate": gate["regression_rate"],
                "canary_failures": gate["canary_failures"],
                "token_ratio": gate["token_ratio"],
                "latency_ratio": gate["latency_ratio"],
                "reason": gate["reason"],
            },
        )

    return {
        "aggregate": aggregate,
        "gate": gate,
        "promotion_decision": promotion_decision,
    }


def _t_critical_95(sample_size: int) -> float:
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    degrees_of_freedom = sample_size - 1
    for max_df, critical in _T_CRITICAL_95_BY_MAX_DF:
        if degrees_of_freedom <= max_df:
            return critical
    return 1.96
