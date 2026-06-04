from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from .replay import _execute_frozen_job
from .scoring import aggregate_replay_job
from .store import create_replay_job, record_promotion_decision

_DEFAULT_CANARY_THRESHOLDS = {
    "max_canary_failures": 0,
    "min_mean_delta": -0.01,
    "max_regression_rate": 0.05,
    "max_token_ratio": 1.20,
    "max_latency_ratio": 1.25,
}


def create_canary_job(
    conn,
    *,
    candidate_id: str,
    incumbent_id: str,
    episode_ids: Sequence[str],
    thresholds: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    normalized_thresholds = _normalize_canary_thresholds(thresholds)
    runtime = {
        "mode": "canary",
        "thresholds": normalized_thresholds,
        "metadata": deepcopy(metadata) if metadata is not None else {},
    }
    job = create_replay_job(
        conn,
        candidate_id,
        incumbent_id,
        episode_ids,
        split="canary",
        evaluator_version="canary_v1",
        runtime=runtime,
    )
    return job["job_id"]


def canary_gate(aggregate: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized_thresholds = _normalize_canary_thresholds(thresholds)
    failed_checks: list[str] = []

    if aggregate.get("split") != "canary":
        failed_checks.append(f"split != 'canary' (got {aggregate.get('split')!r})")
    if not aggregate.get("complete"):
        failed_checks.append(
            f"incomplete results ({aggregate.get('result_count')}/{aggregate.get('frozen_episode_count')} episodes scored)"
        )
    if int(aggregate.get("hard_safety_violations", 0)) > 0:
        failed_checks.append(f"hard_safety_violations > 0 ({aggregate.get('hard_safety_violations')})")

    canary_failures = aggregate.get("canary_failures")
    if canary_failures is None:
        failed_checks.append("canary_failures unavailable")
    elif int(canary_failures) > normalized_thresholds["max_canary_failures"]:
        failed_checks.append(
            f"canary_failures > max_canary_failures ({int(canary_failures)} > {normalized_thresholds['max_canary_failures']})"
        )

    mean_delta = aggregate.get("mean_delta")
    if mean_delta is None:
        failed_checks.append("mean_delta unavailable")
    elif float(mean_delta) < normalized_thresholds["min_mean_delta"]:
        failed_checks.append(
            f"mean_delta < min_mean_delta ({float(mean_delta):.6f} < {normalized_thresholds['min_mean_delta']:.6f})"
        )

    regression_rate = aggregate.get("regression_rate")
    if regression_rate is None:
        failed_checks.append("regression_rate unavailable")
    elif float(regression_rate) > normalized_thresholds["max_regression_rate"]:
        failed_checks.append(
            f"regression_rate > max_regression_rate ({float(regression_rate):.6f} > {normalized_thresholds['max_regression_rate']:.6f})"
        )

    token_ratio = aggregate.get("token_ratio")
    if token_ratio is not None and float(token_ratio) > normalized_thresholds["max_token_ratio"]:
        failed_checks.append(
            f"token_ratio > max_token_ratio ({float(token_ratio):.6f} > {normalized_thresholds['max_token_ratio']:.6f})"
        )

    latency_ratio = aggregate.get("latency_ratio")
    if latency_ratio is not None and float(latency_ratio) > normalized_thresholds["max_latency_ratio"]:
        failed_checks.append(
            f"latency_ratio > max_latency_ratio ({float(latency_ratio):.6f} > {normalized_thresholds['max_latency_ratio']:.6f})"
        )

    passed = not failed_checks
    decision = "canary_passed" if passed else "rejected"
    reason = (
        "canary_passed: all canary safety and regression checks satisfied"
        if passed
        else "rejected: " + "; ".join(failed_checks)
    )

    return {
        "passed": passed,
        "decision": decision,
        "reason": reason,
        "failed_checks": failed_checks,
        "thresholds": normalized_thresholds,
        "aggregate": deepcopy(aggregate),
    }


def evaluate_canary_job(
    conn,
    job_id: str,
    thresholds: dict[str, Any] | None = None,
    write_decision: bool = False,
) -> dict[str, Any]:
    aggregate = aggregate_replay_job(conn, job_id)
    if aggregate["split"] != "canary":
        raise ValueError(f"Canary job {job_id} must have split 'canary', got {aggregate['split']!r}")

    gate = canary_gate(aggregate, thresholds)
    promotion_decision = None
    if write_decision:
        promotion_decision = record_promotion_decision(
            conn,
            {
                "candidate_id": aggregate["candidate_id"],
                "incumbent_id": aggregate["incumbent_id"],
                "job_id": job_id,
                "decision": gate["decision"],
                "mean_delta": aggregate.get("mean_delta"),
                "lcb95": aggregate.get("lcb95"),
                "regression_rate": aggregate.get("regression_rate"),
                "canary_failures": aggregate.get("canary_failures"),
                "token_ratio": aggregate.get("token_ratio"),
                "latency_ratio": aggregate.get("latency_ratio"),
                "reason": gate["reason"],
            },
        )

    return {
        "job_id": job_id,
        "aggregate": aggregate,
        "gate": gate,
        "promotion_decision": promotion_decision,
        "promotion_decision_id": promotion_decision["decision_id"] if promotion_decision else None,
    }


def run_canary_job(
    conn,
    job_id: str,
    evaluator,
    *,
    write_decision: bool = False,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    execution = _execute_frozen_job(conn, job_id, evaluator, required_split="canary")
    evaluation = evaluate_canary_job(
        conn,
        job_id,
        thresholds=thresholds,
        write_decision=write_decision,
    )

    return {
        "job_id": job_id,
        "candidate_id": execution["candidate"]["candidate_id"],
        "incumbent_id": execution["incumbent"]["candidate_id"],
        "episodes_run": len(execution["episodes"]),
        "results_written": len(execution["results"]),
        "aggregate": evaluation["aggregate"],
        "gate": evaluation["gate"],
        "promotion_decision_id": evaluation["promotion_decision_id"],
        "status": execution["job"]["status"],
    }


def _normalize_canary_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    if thresholds is None:
        return dict(_DEFAULT_CANARY_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict or None")

    unknown = sorted(set(thresholds) - set(_DEFAULT_CANARY_THRESHOLDS))
    if unknown:
        raise ValueError(f"Unknown canary threshold keys: {', '.join(unknown)}")

    normalized = dict(_DEFAULT_CANARY_THRESHOLDS)
    normalized.update(thresholds)
    return normalized
