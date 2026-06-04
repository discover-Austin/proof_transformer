from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from typing import Any

from .shadow import summarize_shadow_run
from .store import get_candidate_current_state, record_promotion_decision

_DEFAULT_PROMOTION_ELIGIBILITY_THRESHOLDS = {
    "required_current_state": "shadow_passed",
    "require_replay_winner": True,
    "require_canary_passed": True,
    "require_shadow_passed": True,
    "min_shadow_events": 10,
    "max_shadow_divergence_rate": 0.30,
    "min_shadow_acceptance_rate": 0.70,
    "fail_on_warnings": False,
}
_DECISION_ORDER = "created_at DESC, decision_id DESC"
_INVALIDATING_AFTER_SHADOW_PASS = {"rejected", "shadow_failed"}
_DISTILLATION_READY_STATES = {"promoted", "shadow_deployment"}


def summarize_promotion_evidence(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    _require_candidate(conn, candidate_id)

    timeline = _decision_timeline(conn, candidate_id)
    latest_replay_winner = _latest_from_timeline(timeline, "replay_winner")
    latest_canary_passed = _latest_from_timeline(timeline, "canary_passed")
    latest_shadow_started = _latest_from_timeline(timeline, "shadow_started")
    latest_shadow_passed = _latest_from_timeline(timeline, "shadow_passed")
    latest_shadow_failed = _latest_from_timeline(timeline, "shadow_failed")
    latest_shadow_decision = _latest_from_timeline(timeline, "shadow_passed", "shadow_failed")
    latest_rejected = _latest_from_timeline(timeline, "rejected")
    latest_promoted = _latest_from_timeline(timeline, "promoted")

    latest_shadow_run = _latest_shadow_run(conn, candidate_id, completed_only=False)
    latest_completed_shadow_run = _latest_shadow_run(conn, candidate_id, completed_only=True)
    latest_shadow_summary = None
    if latest_completed_shadow_run is not None:
        latest_shadow_summary = summarize_shadow_run(conn, latest_completed_shadow_run["shadow_run_id"])

    invalidating_decisions_after_shadow_pass = _decisions_after(
        timeline,
        latest_shadow_passed,
        _INVALIDATING_AFTER_SHADOW_PASS,
    )

    incumbent_id, warnings = _infer_incumbent_id(timeline, [latest_shadow_run, latest_completed_shadow_run])

    return {
        "candidate_id": candidate_id,
        "current_state": get_candidate_current_state(conn, candidate_id),
        "decision_timeline": timeline,
        "latest_replay_winner": latest_replay_winner,
        "latest_canary_passed": latest_canary_passed,
        "latest_shadow_started": latest_shadow_started,
        "latest_shadow_passed": latest_shadow_passed,
        "latest_shadow_failed": latest_shadow_failed,
        "latest_shadow_decision": latest_shadow_decision,
        "latest_rejected": latest_rejected,
        "latest_promoted": latest_promoted,
        "latest_shadow_run": latest_shadow_run,
        "latest_shadow_summary": latest_shadow_summary,
        "latest_completed_shadow_run": latest_completed_shadow_run,
        "latest_completed_shadow_summary": latest_shadow_summary,
        "invalidating_decisions_after_shadow_pass": invalidating_decisions_after_shadow_pass,
        "incumbent_id": incumbent_id,
        "warnings": warnings,
        # Backwards-compatible aliases for the first promotion-eligibility layer.
        "latest_replay_winner_decision": latest_replay_winner,
        "latest_canary_passed_decision": latest_canary_passed,
        "latest_shadow_passed_decision": latest_shadow_passed,
        "latest_rejection_after_shadow_evidence": _latest_from_list(
            invalidating_decisions_after_shadow_pass,
            "rejected",
        ),
        "latest_rejection_after_shadow_passed": _latest_from_list(
            invalidating_decisions_after_shadow_pass,
            "rejected",
        ),
        "latest_shadow_failure_after_shadow_passed": _latest_from_list(
            invalidating_decisions_after_shadow_pass,
            "shadow_failed",
        ),
        "rejected_after_shadow_evidence": any(
            decision["decision"] == "rejected" for decision in invalidating_decisions_after_shadow_pass
        ),
        "rejected_after_shadow_passed": any(
            decision["decision"] == "rejected" for decision in invalidating_decisions_after_shadow_pass
        ),
        "shadow_failed_after_shadow_passed": any(
            decision["decision"] == "shadow_failed" for decision in invalidating_decisions_after_shadow_pass
        ),
    }


def promotion_eligibility_gate(evidence: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    threshold_values = _normalize_thresholds(thresholds)
    failed_checks: list[str] = []

    required_state = threshold_values["required_current_state"]
    current_state = evidence.get("current_state")
    if required_state is not None and current_state != required_state:
        failed_checks.append(f"current_state != {required_state!r} (got {current_state!r})")

    if threshold_values["require_replay_winner"] and not _get_evidence(evidence, "latest_replay_winner"):
        failed_checks.append("missing replay_winner evidence")

    if threshold_values["require_canary_passed"] and not _get_evidence(evidence, "latest_canary_passed"):
        failed_checks.append("missing canary_passed evidence")

    if threshold_values["require_shadow_passed"] and not _get_evidence(evidence, "latest_shadow_passed"):
        failed_checks.append("missing shadow_passed evidence")

    latest_shadow_decision = evidence.get("latest_shadow_decision")
    if latest_shadow_decision and latest_shadow_decision.get("decision") == "shadow_failed":
        failed_checks.append("latest shadow decision is shadow_failed")

    invalidating_decisions = _invalidating_decisions(evidence)
    if invalidating_decisions:
        failed_checks.append(
            "invalidating decision after latest shadow_passed: "
            + ", ".join(decision["decision"] for decision in invalidating_decisions)
        )

    shadow_summary = _get_evidence(evidence, "latest_shadow_summary")
    if shadow_summary is None:
        failed_checks.append("missing latest shadow_summary")
    else:
        _check_shadow_summary(shadow_summary, threshold_values, failed_checks)

    warnings = list(evidence.get("warnings") or [])
    if _has_inconsistent_incumbent_warning(warnings):
        failed_checks.append("incumbent evidence is inconsistent")
    elif _required_evidence_present(evidence, threshold_values) and not evidence.get("incumbent_id"):
        failed_checks.append("incumbent_id is not inferable from evidence")

    if threshold_values["fail_on_warnings"] and warnings:
        failed_checks.append("evidence warnings present: " + "; ".join(warnings))

    eligible = not failed_checks
    reason = (
        "promotion_eligible: replay, canary, and shadow evidence satisfy final eligibility gates"
        if eligible
        else "promotion_ineligible: " + "; ".join(failed_checks)
    )

    return {
        "eligible": eligible,
        "decision": "promotion_eligible" if eligible else "promotion_ineligible",
        "reason": reason,
        "failed_checks": failed_checks,
        "thresholds": threshold_values,
        "evidence": deepcopy(evidence),
    }


def evaluate_promotion_eligibility(
    conn: sqlite3.Connection,
    candidate_id: str,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = summarize_promotion_evidence(conn, candidate_id)
    return promotion_eligibility_gate(evidence, thresholds)


def assert_promotion_eligible(
    conn: sqlite3.Connection,
    candidate_id: str,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = evaluate_promotion_eligibility(conn, candidate_id, thresholds)
    if not gate["eligible"]:
        raise ValueError(gate["reason"])
    return gate


def promote_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    operator: str,
    reason: str,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    reason_text = _require_non_empty_text(reason, "reason")
    eligibility = assert_promotion_eligible(conn, candidate_id, thresholds)
    evidence = eligibility["evidence"]
    incumbent_id = _require_incumbent_id(evidence)
    shadow_run_id = _require_shadow_reference(evidence)

    promotion_decision = record_promotion_decision(
        conn,
        {
            "candidate_id": candidate_id,
            "incumbent_id": incumbent_id,
            "shadow_run_id": shadow_run_id,
            "decision": "promoted",
            "reason": _operator_reason("Explicit promotion", operator_text, reason_text),
        },
    )

    return {
        "candidate_id": candidate_id,
        "incumbent_id": incumbent_id,
        "promotion_decision": promotion_decision,
        "eligibility": eligibility,
        "operator": operator_text,
        "reason": reason_text,
    }


def mark_shadow_deployment(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    operator: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    reason_text = _require_non_empty_text(reason, "reason")
    current_state = get_candidate_current_state(conn, candidate_id)
    if current_state != "promoted":
        raise ValueError(f"Candidate must be promoted before shadow deployment, got {current_state!r}")

    evidence = summarize_promotion_evidence(conn, candidate_id)
    incumbent_id = _require_incumbent_id(evidence)
    shadow_run_id = _require_shadow_reference(evidence)

    return record_promotion_decision(
        conn,
        {
            "candidate_id": candidate_id,
            "incumbent_id": incumbent_id,
            "shadow_run_id": shadow_run_id,
            "decision": "shadow_deployment",
            "reason": _operator_reason("Shadow deployment marker", operator_text, reason_text, metadata),
        },
    )


def mark_distillation_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    operator: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    reason_text = _require_non_empty_text(reason, "reason")
    evidence = summarize_promotion_evidence(conn, candidate_id)
    current_state = evidence.get("current_state")
    if current_state not in _DISTILLATION_READY_STATES:
        raise ValueError(
            "Candidate must be promoted or shadow_deployment before distillation selection, "
            f"got {current_state!r}"
        )
    _require_distillation_evidence(evidence)
    incumbent_id = _require_incumbent_id(evidence)
    shadow_run_id = _require_shadow_reference(evidence)

    return record_promotion_decision(
        conn,
        {
            "candidate_id": candidate_id,
            "incumbent_id": incumbent_id,
            "shadow_run_id": shadow_run_id,
            "decision": "distillation_candidate",
            "reason": _operator_reason("Distillation candidate marker", operator_text, reason_text, metadata),
        },
    )


def _check_shadow_summary(
    shadow_summary: dict[str, Any],
    thresholds: dict[str, Any],
    failed_checks: list[str],
) -> None:
    n_events = int(shadow_summary.get("n_events", 0))
    if n_events < thresholds["min_shadow_events"]:
        failed_checks.append(
            f"shadow n_events < min_shadow_events ({n_events} < {thresholds['min_shadow_events']})"
        )

    divergence_rate = float(shadow_summary.get("divergence_rate", 0.0))
    if divergence_rate > thresholds["max_shadow_divergence_rate"]:
        failed_checks.append(
            "shadow divergence_rate > max_shadow_divergence_rate "
            f"({divergence_rate:.6f} > {thresholds['max_shadow_divergence_rate']:.6f})"
        )

    accepted_rate = shadow_summary.get("accepted_by_incumbent_rate")
    min_acceptance_rate = thresholds["min_shadow_acceptance_rate"]
    if accepted_rate is not None and float(accepted_rate) < min_acceptance_rate:
        failed_checks.append(
            "shadow accepted_by_incumbent_rate < min_shadow_acceptance_rate "
            f"({float(accepted_rate):.6f} < {min_acceptance_rate:.6f})"
        )


def _require_candidate(conn: sqlite3.Connection, candidate_id: str) -> None:
    row = conn.execute(
        "SELECT candidate_id FROM candidates WHERE candidate_id = ? LIMIT 1",
        (candidate_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")


def _decision_timeline(conn: sqlite3.Connection, candidate_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM promotion_decisions
        WHERE candidate_id = ?
        ORDER BY {_DECISION_ORDER}
        """,
        (candidate_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def _latest_from_timeline(timeline: list[dict[str, Any]], *decisions: str) -> dict[str, Any] | None:
    wanted = set(decisions)
    for decision in timeline:
        if decision["decision"] in wanted:
            return deepcopy(decision)
    return None


def _latest_from_list(items: list[dict[str, Any]], decision: str) -> dict[str, Any] | None:
    for item in items:
        if item.get("decision") == decision:
            return deepcopy(item)
    return None


def _decisions_after(
    timeline: list[dict[str, Any]],
    after_decision: dict[str, Any] | None,
    decisions: set[str],
) -> list[dict[str, Any]]:
    if after_decision is None:
        return []

    result: list[dict[str, Any]] = []
    for row in timeline:
        if row["decision_id"] == after_decision["decision_id"]:
            break
        if row["decision"] in decisions:
            result.append(deepcopy(row))
    return result


def _latest_shadow_run(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    completed_only: bool,
) -> dict[str, Any] | None:
    status_clause = "AND status = 'completed'" if completed_only else ""
    rows = conn.execute(
        f"""
        SELECT *
        FROM shadow_runs
        WHERE candidate_id = ? {status_clause}
        ORDER BY COALESCE(completed_at, started_at) DESC, started_at DESC, shadow_run_id DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchall()
    if not rows:
        return None
    record = dict(rows[0])
    record["thresholds"] = json.loads(record.pop("thresholds_json"))
    record["metadata"] = json.loads(record.pop("metadata_json"))
    return record


def _infer_incumbent_id(
    timeline: list[dict[str, Any]],
    shadow_runs: list[dict[str, Any] | None],
) -> tuple[str | None, list[str]]:
    incumbent_ids = {row["incumbent_id"] for row in timeline if row.get("incumbent_id")}
    incumbent_ids.update(run["incumbent_id"] for run in shadow_runs if run and run.get("incumbent_id"))

    warnings: list[str] = []
    if not incumbent_ids:
        warnings.append("incumbent_id not inferable from evidence")
        return None, warnings
    if len(incumbent_ids) > 1:
        ordered = sorted(incumbent_ids)
        warnings.append("inconsistent incumbent_ids: " + ", ".join(ordered))
        return None, warnings
    return next(iter(incumbent_ids)), warnings


def _get_evidence(evidence: dict[str, Any], key: str) -> Any:
    if key in evidence:
        return evidence[key]
    aliases = {
        "latest_replay_winner": "latest_replay_winner_decision",
        "latest_canary_passed": "latest_canary_passed_decision",
        "latest_shadow_passed": "latest_shadow_passed_decision",
        "latest_shadow_summary": "latest_completed_shadow_summary",
    }
    alias = aliases.get(key)
    if alias:
        return evidence.get(alias)
    return None


def _invalidating_decisions(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = evidence.get("invalidating_decisions_after_shadow_pass")
    if explicit is not None:
        return list(explicit)

    synthetic: list[dict[str, Any]] = []
    if evidence.get("rejected_after_shadow_passed") or evidence.get("rejected_after_shadow_evidence"):
        synthetic.append({"decision": "rejected"})
    if evidence.get("shadow_failed_after_shadow_passed"):
        synthetic.append({"decision": "shadow_failed"})
    return synthetic


def _has_inconsistent_incumbent_warning(warnings: list[str]) -> bool:
    return any(warning.startswith("inconsistent incumbent_ids:") for warning in warnings)


def _required_evidence_present(evidence: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    if thresholds["require_replay_winner"] and not _get_evidence(evidence, "latest_replay_winner"):
        return False
    if thresholds["require_canary_passed"] and not _get_evidence(evidence, "latest_canary_passed"):
        return False
    if thresholds["require_shadow_passed"] and not _get_evidence(evidence, "latest_shadow_passed"):
        return False
    return _get_evidence(evidence, "latest_shadow_summary") is not None


def _require_incumbent_id(evidence: dict[str, Any]) -> str:
    incumbent_id = evidence.get("incumbent_id")
    if not incumbent_id:
        raise ValueError("Cannot infer a consistent incumbent_id from promotion evidence")
    return str(incumbent_id)


def _require_shadow_reference(evidence: dict[str, Any]) -> str:
    latest_shadow_passed = _get_evidence(evidence, "latest_shadow_passed") or {}
    shadow_run_id = latest_shadow_passed.get("shadow_run_id")
    if shadow_run_id:
        return str(shadow_run_id)

    latest_shadow_summary = _get_evidence(evidence, "latest_shadow_summary") or {}
    shadow_run_id = latest_shadow_summary.get("shadow_run_id")
    if shadow_run_id:
        return str(shadow_run_id)

    latest_shadow_run = evidence.get("latest_shadow_run") or evidence.get("latest_completed_shadow_run") or {}
    shadow_run_id = latest_shadow_run.get("shadow_run_id")
    if shadow_run_id:
        return str(shadow_run_id)

    raise ValueError("Cannot infer a shadow_run_id reference from promotion evidence")


def _require_distillation_evidence(evidence: dict[str, Any]) -> None:
    missing: list[str] = []
    if not _get_evidence(evidence, "latest_replay_winner"):
        missing.append("replay_winner")
    if not _get_evidence(evidence, "latest_canary_passed"):
        missing.append("canary_passed")
    if not _get_evidence(evidence, "latest_shadow_passed"):
        missing.append("shadow_passed")
    if not _get_evidence(evidence, "latest_shadow_summary"):
        missing.append("shadow_summary")
    if _invalidating_decisions(evidence):
        missing.append("no invalidating rejected/shadow_failed after shadow_passed")
    if evidence.get("warnings"):
        missing.append("warning-free incumbent chain")

    if missing:
        raise ValueError("Candidate lacks required distillation evidence: " + ", ".join(missing))


def _require_non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _operator_reason(
    prefix: str,
    operator: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    text = f"{prefix} by {operator}: {reason}"
    if metadata is not None:
        text += " metadata=" + _canonical_json(metadata)
    return text


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalize_thresholds(thresholds: dict[str, Any] | None) -> dict[str, Any]:
    if thresholds is None:
        return dict(_DEFAULT_PROMOTION_ELIGIBILITY_THRESHOLDS)
    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dict or None")

    unknown = sorted(set(thresholds) - set(_DEFAULT_PROMOTION_ELIGIBILITY_THRESHOLDS))
    if unknown:
        raise ValueError(f"Unknown promotion eligibility threshold keys: {', '.join(unknown)}")

    normalized = dict(_DEFAULT_PROMOTION_ELIGIBILITY_THRESHOLDS)
    normalized.update(thresholds)
    return normalized
