from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from typing import Any

from .store import create_replay_job, get_candidate_current_state


def select_replay_episodes_for_candidate(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    split: str = "replay",
    limit: int | None = None,
    policy_tags: list[str] | None = None,
) -> dict[str, Any]:
    _load_candidate(conn, candidate_id)
    limit_value = _normalize_limit(limit)
    policy_tag_values = _normalize_policy_tags(policy_tags)
    policy_tag_set = set(policy_tag_values) if policy_tag_values is not None else None

    rows = conn.execute(
        """
        SELECT episode_id, policy_tags_json
        FROM episodes
        WHERE split = ?
        ORDER BY timestamp ASC, episode_id ASC
        """,
        (split,),
    ).fetchall()

    episode_ids: list[str] = []
    warnings: list[str] = []
    for row in rows:
        if policy_tag_set is not None:
            episode_tags = _decode_policy_tags(row["policy_tags_json"])
            if not (set(episode_tags) & policy_tag_set):
                continue
        episode_ids.append(row["episode_id"])
        if limit_value is not None and len(episode_ids) >= limit_value:
            break

    if not episode_ids:
        warnings.append("No replay episodes matched selection criteria.")

    return {
        "candidate_id": candidate_id,
        "episode_ids": episode_ids,
        "selection": {
            "split": split,
            "limit": limit_value,
            "policy_tags": policy_tag_values,
            "count": len(episode_ids),
        },
        "warnings": warnings,
    }


def infer_incumbent_for_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    candidate = _load_candidate(conn, candidate_id)
    warnings: list[str] = []

    parent_candidate_id = candidate.get("parent_candidate_id")
    if parent_candidate_id:
        if _candidate_exists(conn, str(parent_candidate_id)):
            return {
                "candidate_id": candidate_id,
                "incumbent_id": str(parent_candidate_id),
                "strategy": "parent_candidate",
                "warnings": warnings,
            }
        warnings.append(f"Parent candidate does not exist: {parent_candidate_id}")

    promoted = conn.execute(
        """
        SELECT candidate_id
        FROM promotion_decisions
        WHERE decision = 'promoted'
        ORDER BY created_at DESC, decision_id DESC
        LIMIT 1
        """,
    ).fetchone()
    if promoted:
        return {
            "candidate_id": candidate_id,
            "incumbent_id": promoted["candidate_id"],
            "strategy": "latest_promoted",
            "warnings": warnings,
        }

    warnings.append("No parent_candidate_id or promoted incumbent candidate is available.")
    return {
        "candidate_id": candidate_id,
        "incumbent_id": None,
        "strategy": "none",
        "warnings": warnings,
    }


def plan_post_training_evaluation(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    incumbent_id: str | None = None,
    episode_ids: list[str] | None = None,
    limit: int | None = None,
    policy_tags: list[str] | None = None,
    split: str = "replay",
) -> dict[str, Any]:
    candidate = _load_candidate(conn, candidate_id)
    blockers: list[str] = []

    if candidate["current_state"] != "draft_candidate":
        blockers.append(f"candidate current_state must be 'draft_candidate', got {candidate['current_state']!r}")

    if incumbent_id is None:
        incumbent = infer_incumbent_for_candidate(conn, candidate_id)
    else:
        if _candidate_exists(conn, incumbent_id):
            incumbent = {
                "candidate_id": candidate_id,
                "incumbent_id": incumbent_id,
                "strategy": "provided",
                "warnings": [],
            }
        else:
            incumbent = {
                "candidate_id": candidate_id,
                "incumbent_id": incumbent_id,
                "strategy": "provided",
                "warnings": [f"Provided incumbent candidate does not exist: {incumbent_id}"],
            }
    if incumbent["incumbent_id"] is None:
        blockers.append("incumbent_id is required")
    elif not _candidate_exists(conn, str(incumbent["incumbent_id"])):
        blockers.append(f"Unknown incumbent: {incumbent['incumbent_id']}")

    if episode_ids is None:
        episode_selection = select_replay_episodes_for_candidate(
            conn,
            candidate_id,
            split=split,
            limit=limit,
            policy_tags=policy_tags,
        )
    else:
        explicit_episode_ids = list(episode_ids)
        episode_selection = {
            "candidate_id": candidate_id,
            "episode_ids": explicit_episode_ids,
            "selection": {
                "split": split,
                "limit": None,
                "policy_tags": None,
                "count": len(explicit_episode_ids),
                "source": "provided_episode_ids",
            },
            "warnings": [],
        }
        seen: set[str] = set()
        for episode_id in explicit_episode_ids:
            if episode_id in seen:
                blockers.append(f"Duplicate episode_id in replay set: {episode_id}")
            seen.add(episode_id)
            episode = conn.execute(
                "SELECT split FROM episodes WHERE episode_id = ? LIMIT 1",
                (episode_id,),
            ).fetchone()
            if not episode:
                blockers.append(f"Unknown episode: {episode_id}")
            elif episode["split"] != split:
                blockers.append(
                    f"Replay job split {split!r} cannot include episode {episode_id!r} with split {episode['split']!r}"
                )

    if not episode_selection["episode_ids"]:
        blockers.append("at least one replay episode is required")

    return {
        "candidate_id": candidate_id,
        "candidate_current_state": candidate["current_state"],
        "incumbent": incumbent,
        "episode_selection": episode_selection,
        "would_create_job": not blockers,
        "blockers": blockers,
    }


def create_post_training_replay_job(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    incumbent_id: str | None = None,
    episode_ids: list[str] | None = None,
    limit: int | None = None,
    policy_tags: list[str] | None = None,
    evaluator_version: str = "post_training_replay_v1",
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = plan_post_training_evaluation(
        conn,
        candidate_id,
        incumbent_id=incumbent_id,
        episode_ids=episode_ids,
        limit=limit,
        policy_tags=policy_tags,
        split="replay",
    )
    if not plan["would_create_job"]:
        raise ValueError("; ".join(plan["blockers"]))

    resolved_incumbent_id = plan["incumbent"]["incumbent_id"]
    if resolved_incumbent_id is None:
        raise ValueError("incumbent_id is required")

    runtime_payload = dict(deepcopy(runtime) if runtime is not None else {})
    runtime_payload.update(
        {
            "mode": "post_training_candidate_evaluation",
            "candidate_id": candidate_id,
            "incumbent_id": resolved_incumbent_id,
            "episode_selection": deepcopy(plan["episode_selection"]["selection"]),
        }
    )

    job = create_replay_job(
        conn,
        candidate_id,
        resolved_incumbent_id,
        plan["episode_selection"]["episode_ids"],
        split="replay",
        evaluator_version=evaluator_version,
        runtime=runtime_payload,
    )
    job["plan"] = plan
    return job


def _load_candidate(conn: sqlite3.Connection, candidate_id: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM candidates WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")
    candidate = dict(row)
    candidate["manifest"] = json.loads(candidate["manifest_json"])
    candidate["current_state"] = get_candidate_current_state(conn, candidate_id)
    return candidate


def _candidate_exists(conn: sqlite3.Connection, candidate_id: str) -> bool:
    return (
        conn.execute("SELECT 1 FROM candidates WHERE candidate_id = ? LIMIT 1", (candidate_id,)).fetchone()
        is not None
    )


def _normalize_limit(limit: int | None) -> int | None:
    if limit is None:
        return None
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an int or None")
    if limit < 0:
        raise ValueError("limit must not be negative")
    return limit


def _normalize_policy_tags(policy_tags: list[str] | None) -> list[str] | None:
    if policy_tags is None:
        return None
    if not isinstance(policy_tags, list):
        raise ValueError("policy_tags must be a list[str] or None")
    normalized: list[str] = []
    for tag in policy_tags:
        if not isinstance(tag, str) or not tag.strip():
            raise ValueError("policy_tags entries must be non-empty strings")
        normalized.append(tag.strip())
    return normalized


def _decode_policy_tags(value: str | None) -> list[str]:
    if not value:
        return []
    decoded = json.loads(value)
    if isinstance(decoded, list):
        return [str(tag) for tag in decoded]
    return []
