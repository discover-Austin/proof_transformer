from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from typing import Any


def acceptance_checklist_replay_evaluator(
    *,
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    ordinal: int = 0,
    **_: Any,
) -> dict[str, Any]:
    return _acceptance_checklist_result(
        episode=episode,
        candidate=candidate,
        incumbent=incumbent,
        ordinal=ordinal,
        split="replay",
    )


def acceptance_checklist_canary_evaluator(
    *,
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    ordinal: int = 0,
    **_: Any,
) -> dict[str, Any]:
    result = _acceptance_checklist_result(
        episode=episode,
        candidate=candidate,
        incumbent=incumbent,
        ordinal=ordinal,
        split="canary",
    )
    result["canary_failures"] = _canary_failures(episode, result)
    return result


def make_local_episode_shadow_event_provider(
    *,
    split: str = "online",
    limit: int = 10,
    policy_tags: list[str] | None = None,
):
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("limit must be a non-negative int")
    tag_filter = _normalize_policy_tags(policy_tags)

    def provider(
        *,
        conn: sqlite3.Connection,
        candidate: dict[str, Any],
        incumbent: dict[str, Any],
        shadow_run: dict[str, Any],
        **_: Any,
    ) -> list[dict[str, Any]]:
        return local_episode_shadow_events(
            conn,
            candidate=candidate,
            incumbent=incumbent,
            shadow_run=shadow_run,
            split=split,
            limit=limit,
            policy_tags=tag_filter,
        )

    return provider


def local_episode_shadow_events(
    conn: sqlite3.Connection,
    *,
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    shadow_run: dict[str, Any],
    split: str = "online",
    limit: int = 10,
    policy_tags: list[str] | None = None,
) -> list[dict[str, Any]]:
    tag_filter = set(_normalize_policy_tags(policy_tags) or [])
    rows = conn.execute(
        """
        SELECT
            episode_id,
            task_id,
            task_title,
            task_prompt,
            acceptance_tests_json,
            policy_tags_json,
            final_message,
            judge_verdict,
            user_verdict,
            metrics_json,
            privacy_json,
            raw_episode_json
        FROM episodes
        WHERE split = ?
        ORDER BY timestamp ASC, episode_id ASC
        """,
        (split,),
    ).fetchall()

    events: list[dict[str, Any]] = []
    for row in rows:
        episode = _episode_from_row(row)
        episode_tags = set(episode["policy_tags"])
        if tag_filter and not (tag_filter & episode_tags):
            continue
        accepted = _episode_is_accepted(episode)
        events.append(
            {
                "input": _shadow_input(episode),
                "incumbent_output": _shadow_output(episode, incumbent),
                "candidate_output": _shadow_output(episode, candidate),
                "divergence": None if accepted else _shadow_divergence(episode, candidate, incumbent),
                "candidate_would_act": _would_act(candidate),
                "incumbent_did_act": _would_act(incumbent),
                "accepted_by_incumbent": accepted,
                "episode_ref": episode["episode_id"],
                "task_ref": episode["task_id"],
                "metadata": {
                    "provider": "local_episode_shadow_events",
                    "source_split": split,
                    "source_episode_id": episode["episode_id"],
                    "shadow_run_id": shadow_run["shadow_run_id"],
                    "judge_verdict": episode["judge_verdict"],
                    "user_verdict": episode["user_verdict"],
                    "candidate_theta_strategy": _theta_strategy(candidate),
                    "incumbent_theta_strategy": _theta_strategy(incumbent),
                },
            }
        )
        if len(events) >= limit:
            break
    return events


def _acceptance_checklist_result(
    *,
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    ordinal: int,
    split: str,
) -> dict[str, Any]:
    candidate_score = _score_candidate_on_episode(episode, candidate)
    incumbent_score = _score_candidate_on_episode(episode, incumbent)
    return {
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "judge_verdict": _judge_verdict(episode),
        "audit_log_ref": f"local:acceptance_checklist:{split}:{ordinal}",
    }


def _score_candidate_on_episode(episode: dict[str, Any], candidate: dict[str, Any]) -> float:
    acceptance_tests = _acceptance_tests(episode)
    metrics = _metrics(episode)
    privacy = _privacy(episode)
    score = 0.56

    if _judge_verdict(episode) == "pass":
        score += 0.10
    if _user_verdict(episode) == "accepted":
        score += 0.06
    score += min(len(acceptance_tests), 4) * 0.015
    score += _clamp_float(metrics.get("tests_passed"), default=0.0, low=0.0, high=1.0) * 0.06

    score -= _clamp_float(metrics.get("failed_tool_calls"), default=0.0, low=0.0, high=10.0) * 0.01
    score -= _clamp_float(metrics.get("hallucinated_claims"), default=0.0, low=0.0, high=10.0) * 0.03
    score -= _clamp_float(metrics.get("unrelated_edit_ratio"), default=0.0, low=0.0, high=1.0) * 0.04
    if privacy.get("contains_secret"):
        score -= 0.20

    strategy = _theta_strategy(candidate)
    if strategy == "acceptance_checklist_first":
        score += 0.035 + min(len(acceptance_tests), 4) * 0.025
    elif strategy == "baseline_final_answer":
        score += 0.005

    return round(max(0.0, min(1.0, score)), 6)


def _canary_failures(episode: dict[str, Any], result: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    if result["candidate_score"] < result["incumbent_score"]:
        failures.append({"kind": "regression", "reason": "candidate score below incumbent"})
    if _judge_verdict(episode) in {"fail", "unsafe", "timeout"}:
        failures.append({"kind": "judge_verdict", "reason": _judge_verdict(episode)})
    if _privacy(episode).get("contains_secret"):
        failures.append({"kind": "privacy", "reason": "episode contains secret"})
    return failures


def _episode_from_row(row: sqlite3.Row) -> dict[str, Any]:
    raw_episode = _decode_json(row["raw_episode_json"]) if row["raw_episode_json"] else {}
    return {
        "episode_id": row["episode_id"],
        "task_id": row["task_id"],
        "task_title": row["task_title"],
        "task_prompt": row["task_prompt"],
        "acceptance_tests": _decode_json(row["acceptance_tests_json"]) if row["acceptance_tests_json"] else [],
        "policy_tags": _decode_json(row["policy_tags_json"]) if row["policy_tags_json"] else [],
        "final_message": row["final_message"],
        "judge_verdict": row["judge_verdict"],
        "user_verdict": row["user_verdict"],
        "metrics": _decode_json(row["metrics_json"]) if row["metrics_json"] else {},
        "privacy": _decode_json(row["privacy_json"]) if row["privacy_json"] else {},
        "raw_episode": raw_episode,
    }


def _shadow_input(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": episode["task_id"],
        "title": episode["task_title"],
        "prompt": episode["task_prompt"],
        "acceptance_tests": deepcopy(episode["acceptance_tests"]),
        "policy_tags": deepcopy(episode["policy_tags"]),
    }


def _shadow_output(episode: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    output = {
        "final_message": episode.get("final_message") or "Local episode had no final message.",
        "theta_strategy": _theta_strategy(candidate),
    }
    if _theta_strategy(candidate) == "acceptance_checklist_first":
        output["acceptance_checklist"] = deepcopy(episode["acceptance_tests"])
    return output


def _shadow_divergence(
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> dict[str, Any] | None:
    candidate_strategy = _theta_strategy(candidate)
    incumbent_strategy = _theta_strategy(incumbent)
    if candidate_strategy == incumbent_strategy:
        return None
    return {
        "kind": "theta_strategy",
        "candidate_theta_strategy": candidate_strategy,
        "incumbent_theta_strategy": incumbent_strategy,
        "acceptance_test_count": len(episode["acceptance_tests"]),
    }


def _would_act(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "local_final_answer",
        "theta_strategy": _theta_strategy(candidate),
        "uses_acceptance_checklist": _theta_strategy(candidate) == "acceptance_checklist_first",
    }


def _episode_is_accepted(episode: dict[str, Any]) -> bool:
    return (
        episode["judge_verdict"] == "pass"
        and episode["user_verdict"] == "accepted"
        and not bool(episode["privacy"].get("contains_secret"))
    )


def _acceptance_tests(episode: dict[str, Any]) -> list[Any]:
    task = episode.get("task") or {}
    return list(episode.get("acceptance_tests") or task.get("acceptance_tests") or [])


def _metrics(episode: dict[str, Any]) -> dict[str, Any]:
    return dict(episode.get("metrics") or {})


def _privacy(episode: dict[str, Any]) -> dict[str, Any]:
    return dict(episode.get("privacy") or {})


def _judge_verdict(episode: dict[str, Any]) -> str:
    outcome = episode.get("outcome") or {}
    return str(episode.get("judge_verdict") or outcome.get("judge_verdict") or "pass")


def _user_verdict(episode: dict[str, Any]) -> str | None:
    outcome = episode.get("outcome") or {}
    value = episode.get("user_verdict") or outcome.get("user_verdict")
    return str(value) if value is not None else None


def _theta_strategy(candidate: dict[str, Any]) -> str | None:
    manifest = candidate.get("manifest") or {}
    value = manifest.get("theta_strategy")
    return str(value) if value is not None else None


def _normalize_policy_tags(policy_tags: list[str] | None) -> list[str] | None:
    if policy_tags is None:
        return None
    if not isinstance(policy_tags, list):
        raise ValueError("policy_tags must be a list[str] or None")
    tags: list[str] = []
    for tag in policy_tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("policy_tags entries must be non-empty strings")
        tags.append(tag.strip())
    return tags


def _clamp_float(value: Any, *, default: float, low: float, high: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def _decode_json(value: str) -> Any:
    return json.loads(value)
