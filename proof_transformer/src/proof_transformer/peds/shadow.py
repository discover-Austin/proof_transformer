from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from typing import Any

from ..ids import new_id
from ..timeutils import utc_now
from .store import get_candidate_current_state, record_promotion_decision

_DEFAULT_SHADOW_THRESHOLDS = {
    "min_events": 10,
    "max_divergence_rate": 0.30,
    "min_accepted_by_incumbent_rate": 0.70,
}


def start_shadow_run(
    conn,
    *,
    candidate_id: str,
    incumbent_id: str,
    thresholds: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    write_decision: bool = False,
) -> dict[str, Any]:
    _require_candidate(conn, candidate_id)
    _require_candidate(conn, incumbent_id)

    current_state = get_candidate_current_state(conn, candidate_id)
    if current_state != "canary_passed":
        raise ValueError("Candidate must be canary_passed before shadow mode")

    shadow_run_id = new_id("shadow_run")
    started_at = utc_now()
    threshold_payload = _normalize_shadow_thresholds(thresholds)
    metadata_payload = deepcopy(metadata) if metadata is not None else {}

    with conn:
        conn.execute(
            """
            INSERT INTO shadow_runs(
                shadow_run_id, candidate_id, incumbent_id, status,
                started_at, completed_at, thresholds_json, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shadow_run_id,
                candidate_id,
                incumbent_id,
                "active",
                started_at,
                None,
                _canonical_json(threshold_payload),
                _canonical_json(metadata_payload),
            ),
        )

    promotion_decision = None
    if write_decision:
        promotion_decision = record_promotion_decision(
            conn,
            {
                "candidate_id": candidate_id,
                "incumbent_id": incumbent_id,
                "shadow_run_id": shadow_run_id,
                "decision": "shadow_started",
                "reason": "Candidate entered non-authoritative shadow mode.",
            },
        )

    run = _load_shadow_run(conn, shadow_run_id)
    run["promotion_decision_id"] = promotion_decision["decision_id"] if promotion_decision else None
    return run


def record_shadow_event(
    conn,
    *,
    shadow_run_id: str,
    input: dict,
    incumbent_output: dict,
    candidate_output: dict,
    divergence: dict | None = None,
    candidate_would_act: dict | None = None,
    incumbent_did_act: dict | None = None,
    accepted_by_incumbent: bool | None = None,
    task_ref: str | None = None,
    episode_ref: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    run = _load_shadow_run(conn, shadow_run_id)
    if run["status"] != "active":
        raise ValueError(f"Shadow run {shadow_run_id} is not active")

    ordinal_row = conn.execute(
        "SELECT COALESCE(MAX(ordinal), -1) + 1 AS next_ordinal FROM shadow_events WHERE shadow_run_id = ?",
        (shadow_run_id,),
    ).fetchone()
    ordinal = int(ordinal_row["next_ordinal"])
    shadow_event_id = new_id("shadow_event")
    created_at = utc_now()

    with conn:
        conn.execute(
            """
            INSERT INTO shadow_events(
                shadow_event_id, shadow_run_id, ordinal, task_ref, episode_ref,
                input_json, incumbent_output_json, candidate_output_json,
                divergence_json, candidate_would_act_json, incumbent_did_act_json,
                accepted_by_incumbent, created_at, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                shadow_event_id,
                shadow_run_id,
                ordinal,
                task_ref,
                episode_ref,
                _canonical_json(input),
                _canonical_json(incumbent_output),
                _canonical_json(candidate_output),
                _canonical_json(divergence) if divergence is not None else None,
                _canonical_json(candidate_would_act) if candidate_would_act is not None else None,
                _canonical_json(incumbent_did_act) if incumbent_did_act is not None else None,
                _encode_bool(accepted_by_incumbent),
                created_at,
                _canonical_json(metadata or {}),
            ),
        )

    return _load_shadow_event(conn, shadow_event_id)


def summarize_shadow_run(conn, shadow_run_id: str) -> dict[str, Any]:
    run = _load_shadow_run(conn, shadow_run_id)
    events = _list_shadow_events(conn, shadow_run_id)

    n_events = len(events)
    n_divergences = sum(1 for event in events if _has_divergence(event["divergence"]))
    n_candidate_would_act = sum(1 for event in events if event["candidate_would_act"] is not None)
    n_incumbent_did_act = sum(1 for event in events if event["incumbent_did_act"] is not None)

    accepted_values = [event["accepted_by_incumbent"] for event in events if event["accepted_by_incumbent"] is not None]
    accepted_by_incumbent_rate = None
    if accepted_values:
        accepted_by_incumbent_rate = sum(1 for value in accepted_values if value) / len(accepted_values)

    return {
        "shadow_run_id": shadow_run_id,
        "candidate_id": run["candidate_id"],
        "incumbent_id": run["incumbent_id"],
        "status": run["status"],
        "n_events": n_events,
        "n_divergences": n_divergences,
        "divergence_rate": (n_divergences / n_events) if n_events else 0.0,
        "accepted_by_incumbent_rate": accepted_by_incumbent_rate,
        "n_candidate_would_act": n_candidate_would_act,
        "n_incumbent_did_act": n_incumbent_did_act,
        "thresholds": deepcopy(run["thresholds"]),
    }


def shadow_gate(summary: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    threshold_payload = _normalize_shadow_thresholds(thresholds)
    failed_checks: list[str] = []

    if int(summary.get("n_events", 0)) < threshold_payload["min_events"]:
        failed_checks.append(
            f"n_events < min_events ({int(summary.get('n_events', 0))} < {threshold_payload['min_events']})"
        )

    divergence_rate = float(summary.get("divergence_rate", 0.0))
    if divergence_rate > threshold_payload["max_divergence_rate"]:
        failed_checks.append(
            f"divergence_rate > max_divergence_rate ({divergence_rate:.6f} > {threshold_payload['max_divergence_rate']:.6f})"
        )

    accepted_rate = summary.get("accepted_by_incumbent_rate")
    if accepted_rate is not None and float(accepted_rate) < threshold_payload["min_accepted_by_incumbent_rate"]:
        failed_checks.append(
            "accepted_by_incumbent_rate < min_accepted_by_incumbent_rate "
            f"({float(accepted_rate):.6f} < {threshold_payload['min_accepted_by_incumbent_rate']:.6f})"
        )

    passed = not failed_checks
    decision = "shadow_passed" if passed else "shadow_failed"
    reason = (
        "shadow_passed: non-authoritative shadow evidence satisfied all thresholds"
        if passed
        else "shadow_failed: " + "; ".join(failed_checks)
    )

    return {
        "passed": passed,
        "decision": decision,
        "reason": reason,
        "failed_checks": failed_checks,
        "thresholds": threshold_payload,
        "summary": deepcopy(summary),
    }


def complete_shadow_run(
    conn,
    shadow_run_id: str,
    *,
    thresholds: dict[str, Any] | None = None,
    write_decision: bool = False,
) -> dict[str, Any]:
    run = _load_shadow_run(conn, shadow_run_id)
    if run["status"] != "active":
        raise ValueError(f"Shadow run {shadow_run_id} is not active")

    threshold_payload = _normalize_shadow_thresholds(thresholds or run["thresholds"])
    summary = summarize_shadow_run(conn, shadow_run_id)
    gate = shadow_gate(summary, threshold_payload)

    completed_at = utc_now()
    with conn:
        conn.execute(
            "UPDATE shadow_runs SET status = ?, completed_at = ? WHERE shadow_run_id = ?",
            ("completed", completed_at, shadow_run_id),
        )

    promotion_decision = None
    if write_decision:
        promotion_decision = record_promotion_decision(
            conn,
            {
                "candidate_id": run["candidate_id"],
                "incumbent_id": run["incumbent_id"],
                "shadow_run_id": shadow_run_id,
                "decision": gate["decision"],
                "reason": gate["reason"],
            },
        )

    return {
        "shadow_run": _load_shadow_run(conn, shadow_run_id),
        "summary": summary,
        "gate": gate,
        "promotion_decision_id": promotion_decision["decision_id"] if promotion_decision else None,
    }


def _load_shadow_run(conn, shadow_run_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM shadow_runs WHERE shadow_run_id = ? LIMIT 1", (shadow_run_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown shadow run: {shadow_run_id}")
    run = dict(row)
    run["thresholds"] = _decode_json(run.pop("thresholds_json"))
    run["metadata"] = _decode_json(run.pop("metadata_json"))
    return run


def _load_shadow_event(conn, shadow_event_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM shadow_events WHERE shadow_event_id = ? LIMIT 1", (shadow_event_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown shadow event: {shadow_event_id}")
    event = dict(row)
    event["input"] = _decode_json(event.pop("input_json"))
    event["incumbent_output"] = _decode_json(event.pop("incumbent_output_json"))
    event["candidate_output"] = _decode_json(event.pop("candidate_output_json"))
    event["divergence"] = _decode_json(event.pop("divergence_json")) if event.get("divergence_json") else None
    event["candidate_would_act"] = _decode_json(event.pop("candidate_would_act_json")) if event.get("candidate_would_act_json") else None
    event["incumbent_did_act"] = _decode_json(event.pop("incumbent_did_act_json")) if event.get("incumbent_did_act_json") else None
    event["accepted_by_incumbent"] = _decode_bool(event.pop("accepted_by_incumbent"))
    event["metadata"] = _decode_json(event.pop("metadata_json"))
    return event


def _list_shadow_events(conn, shadow_run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT shadow_event_id FROM shadow_events WHERE shadow_run_id = ? ORDER BY ordinal, created_at, shadow_event_id",
        (shadow_run_id,),
    ).fetchall()
    return [_load_shadow_event(conn, row["shadow_event_id"]) for row in rows]


def _require_candidate(conn, candidate_id: str) -> None:
    row = conn.execute("SELECT candidate_id FROM candidates WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")


def _normalize_shadow_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    if thresholds is None:
        return dict(_DEFAULT_SHADOW_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict or None")

    unknown = sorted(set(thresholds) - set(_DEFAULT_SHADOW_THRESHOLDS))
    if unknown:
        raise ValueError(f"Unknown shadow threshold keys: {', '.join(unknown)}")

    normalized = dict(_DEFAULT_SHADOW_THRESHOLDS)
    normalized.update(thresholds)
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _decode_json(value: str) -> Any:
    return json.loads(value)


def _encode_bool(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0


def _decode_bool(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _has_divergence(value: Any) -> bool:
    return value not in (None, {}, [], "")
