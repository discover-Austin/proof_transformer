from __future__ import annotations

import json
import re
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def acceptance_checklist_replay_evaluator(
    *,
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    conn: sqlite3.Connection | None = None,
    ordinal: int = 0,
    **_: Any,
) -> dict[str, Any]:
    return _acceptance_checklist_result(
        conn=conn,
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
    conn: sqlite3.Connection | None = None,
    ordinal: int = 0,
    **_: Any,
) -> dict[str, Any]:
    result = _acceptance_checklist_result(
        conn=conn,
        episode=episode,
        candidate=candidate,
        incumbent=incumbent,
        ordinal=ordinal,
        split="canary",
    )
    result["canary_failures"] = _canary_failures(episode, result)
    return result


def load_candidate_local_adapter(
    conn: sqlite3.Connection | None,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    manifest = _candidate_manifest(candidate)
    for artifact_ref in _candidate_adapter_artifact_refs(conn, candidate, manifest):
        artifact = _load_adapter_artifact_json(artifact_ref)
        if _is_trained_local_json_adapter(artifact):
            return artifact
    return None


def local_adapter_behavior_profile(
    conn: sqlite3.Connection | None,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    artifact = load_candidate_local_adapter(conn, candidate)
    if artifact is None:
        return {
            "theta_strategy": _theta_strategy(candidate),
            "adapter": {
                "loaded": False,
                "trained": False,
                "mode": None,
            },
            "learned_parameters": {},
        }

    learned_parameters = artifact.get("learned_parameters") or {}
    return {
        "theta_strategy": _theta_strategy(candidate),
        "adapter": {
            "loaded": True,
            "trained": True,
            "mode": artifact.get("mode"),
            "training_run_id": artifact.get("training_run_id"),
            "dataset_hash": artifact.get("dataset_hash"),
            "n_examples": artifact.get("n_examples"),
            "learned_parameter_count": _count_learned_parameters(learned_parameters),
        },
        "learned_parameters": _learned_parameters_summary(learned_parameters),
    }


def local_adapter_replay_signal(
    conn: sqlite3.Connection,
    candidate: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    artifact = load_candidate_local_adapter(conn, candidate)
    if artifact is None:
        return {
            "adapter_loaded": False,
            "adapter_trained": False,
            "adapter_mode": None,
            "episode_count": 0,
            "behavior_delta_count": 0,
            "mean_behavior_delta": None,
            "max_behavior_delta": None,
            "training_run_id": None,
        }

    rows = conn.execute(
        """
        SELECT e.raw_episode_json
        FROM replay_job_episodes rje
        JOIN episodes e ON e.episode_id = rje.episode_id
        WHERE rje.job_id = ?
        ORDER BY rje.episode_order
        """,
        (job_id,),
    ).fetchall()
    deltas: list[float] = []
    for row in rows:
        try:
            episode = _decode_json(row["raw_episode_json"])
        except (TypeError, json.JSONDecodeError):
            episode = {}
        deltas.append(_trained_adapter_score_adjustment(conn, episode, candidate))

    positive_deltas = [delta for delta in deltas if delta > 0.0]
    return {
        "adapter_loaded": True,
        "adapter_trained": True,
        "adapter_mode": artifact.get("mode"),
        "episode_count": len(deltas),
        "behavior_delta_count": len(positive_deltas),
        "mean_behavior_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "max_behavior_delta": max(deltas) if deltas else None,
        "training_run_id": artifact.get("training_run_id"),
    }


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
    candidate_profile = local_adapter_behavior_profile(conn, candidate)
    incumbent_profile = local_adapter_behavior_profile(conn, incumbent)
    for row in rows:
        episode = _episode_from_row(row)
        episode_tags = set(episode["policy_tags"])
        if tag_filter and not (tag_filter & episode_tags):
            continue
        accepted = _episode_is_accepted(episode)
        events.append(
            {
                "input": _shadow_input(episode),
                "incumbent_output": _shadow_output(episode, incumbent, adapter_profile=incumbent_profile),
                "candidate_output": _shadow_output(episode, candidate, adapter_profile=candidate_profile),
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
    conn: sqlite3.Connection | None,
    episode: dict[str, Any],
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
    ordinal: int,
    split: str,
) -> dict[str, Any]:
    candidate_score = _score_candidate_on_episode(conn, episode, candidate)
    incumbent_score = _score_candidate_on_episode(conn, episode, incumbent)
    return {
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "judge_verdict": _judge_verdict(episode),
        "audit_log_ref": f"local:acceptance_checklist:{split}:{ordinal}",
        "candidate_adapter_behavior_profile": local_adapter_behavior_profile(conn, candidate),
        "incumbent_adapter_behavior_profile": local_adapter_behavior_profile(conn, incumbent),
    }


def _score_candidate_on_episode(
    conn: sqlite3.Connection | None,
    episode: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
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

    score += _trained_adapter_score_adjustment(conn, episode, candidate)
    score += _corrected_attempt_score_adjustment(episode, candidate)

    return round(max(0.0, min(1.0, score)), 6)


def _corrected_attempt_score_adjustment(episode: dict[str, Any], candidate: dict[str, Any]) -> float:
    manifest = _candidate_manifest(candidate)
    if manifest.get("role") != "candidate_attempt":
        return 0.0
    generation_options = manifest.get("generation_options")
    if not isinstance(generation_options, dict):
        generation_options = {}
    correction_action = str(generation_options.get("correction_action") or manifest.get("correction_action") or "")
    theta_suffix = str(manifest.get("theta_strategy_suffix") or generation_options.get("theta_strategy_suffix") or "")
    if not correction_action and theta_suffix in ("", "initial"):
        return 0.0
    attempt_suffix = str(
        manifest.get("attempt_artifact_suffix")
        or generation_options.get("attempt_artifact_suffix")
        or generation_options.get("evidence_suffix")
        or ""
    )
    episode_tags = set(str(tag) for tag in _episode_policy_tags(episode))
    if attempt_suffix and attempt_suffix not in episode_tags:
        return 0.0
    return 0.02


def _episode_policy_tags(episode: dict[str, Any]) -> list[Any]:
    task = episode.get("task") or {}
    return list(episode.get("policy_tags") or task.get("policy_tags") or [])


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


def _shadow_output(
    episode: dict[str, Any],
    candidate: dict[str, Any],
    *,
    adapter_profile: dict[str, Any],
) -> dict[str, Any]:
    output = {
        "final_message": episode.get("final_message") or "Local episode had no final message.",
        "theta_strategy": _theta_strategy(candidate),
        "metadata": {
            "adapter_behavior_profile": deepcopy(adapter_profile),
        },
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


def _prompt_text(episode: dict[str, Any]) -> str:
    task = episode.get("task") or {}
    value = episode.get("task_prompt") or task.get("prompt") or episode.get("prompt")
    return str(value) if value is not None else ""


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
    manifest = _candidate_manifest(candidate)
    value = manifest.get("theta_strategy")
    return str(value) if value is not None else None


def _candidate_manifest(candidate: dict[str, Any]) -> dict[str, Any]:
    manifest = candidate.get("manifest")
    if isinstance(manifest, dict):
        return manifest
    manifest_json = candidate.get("manifest_json")
    if isinstance(manifest_json, str):
        try:
            decoded = json.loads(manifest_json)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _candidate_adapter_artifact_refs(
    conn: sqlite3.Connection | None,
    candidate: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    refs: list[str] = []

    adapter_registry_id = manifest.get("adapter_registry_id")
    if adapter_registry_id is not None:
        registry_path = _adapter_registry_path_by_id(conn, str(adapter_registry_id))
        if registry_path:
            refs.append(registry_path)

    adapter_ref = manifest.get("adapter_ref")
    if isinstance(adapter_ref, dict):
        name = adapter_ref.get("name")
        version = adapter_ref.get("version")
        if name and version:
            registry_path = _adapter_registry_path_by_name_version(conn, str(name), str(version))
            if registry_path:
                refs.append(registry_path)
        if adapter_ref.get("path"):
            refs.append(str(adapter_ref["path"]))

    if manifest.get("artifact_ref"):
        refs.append(str(manifest["artifact_ref"]))

    registration = candidate.get("adapter_registration")
    if isinstance(registration, dict):
        adapter = registration.get("adapter")
        if isinstance(adapter, dict) and adapter.get("path"):
            refs.append(str(adapter["path"]))

    return _dedupe_text(refs)


def _adapter_registry_path_by_id(conn: sqlite3.Connection | None, adapter_id: str) -> str | None:
    if conn is None:
        return None
    try:
        row = conn.execute("SELECT path FROM adapter_registry WHERE id = ? LIMIT 1", (adapter_id,)).fetchone()
    except sqlite3.Error:
        return None
    return str(row["path"]) if row and row["path"] else None


def _adapter_registry_path_by_name_version(
    conn: sqlite3.Connection | None,
    name: str,
    version: str,
) -> str | None:
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT path FROM adapter_registry WHERE name = ? AND version = ? LIMIT 1",
            (name, version),
        ).fetchone()
    except sqlite3.Error:
        return None
    return str(row["path"]) if row and row["path"] else None


def _load_adapter_artifact_json(artifact_ref: str) -> dict[str, Any] | None:
    artifact_path = Path(artifact_ref)
    if not artifact_path.is_file():
        return None
    try:
        decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _is_trained_local_json_adapter(artifact: dict[str, Any] | None) -> bool:
    if not isinstance(artifact, dict) or artifact.get("trained") is not True:
        return False
    metrics = artifact.get("metrics") if isinstance(artifact.get("metrics"), dict) else {}
    if artifact.get("mode") != "local_json_trainer" and metrics.get("mode") != "local_json_trainer":
        return False
    return isinstance(artifact.get("learned_parameters"), dict)


def _trained_adapter_score_adjustment(
    conn: sqlite3.Connection | None,
    episode: dict[str, Any],
    candidate: dict[str, Any],
) -> float:
    artifact = load_candidate_local_adapter(conn, candidate)
    if artifact is None:
        return 0.0

    learned_parameters = artifact.get("learned_parameters")
    if not isinstance(learned_parameters, dict):
        return 0.0

    learned_tokens = _learned_token_set(learned_parameters)
    if not learned_tokens:
        return 0.0

    acceptance_tokens = set(_tokens(_acceptance_tests(episode)))
    prompt_tokens = set(_tokens(_prompt_text(episode)))
    acceptance_overlap = _set_overlap_ratio(acceptance_tokens, learned_tokens)
    prompt_overlap = _set_overlap_ratio(prompt_tokens, learned_tokens)
    if acceptance_overlap <= 0.0 and prompt_overlap <= 0.0:
        return 0.0

    feature_weights = learned_parameters.get("feature_weights")
    if not isinstance(feature_weights, dict):
        return 0.0

    acceptance_weight = _clamp_float(
        feature_weights.get("acceptance_token_overlap"),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    prompt_weight = _clamp_float(
        feature_weights.get("prompt_token_overlap"),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    shadow_signal = _clamp_float(
        feature_weights.get("shadow_acceptance_signal"),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    replay_signal = _clamp_float(
        feature_weights.get("replay_win_signal"),
        default=0.0,
        low=0.0,
        high=1.0,
    )
    support_signal = max(shadow_signal, replay_signal)

    token_signal = (acceptance_overlap * acceptance_weight) + (prompt_overlap * prompt_weight * 0.5)
    if token_signal <= 0.0:
        return 0.0
    return round(min(0.10, token_signal * (0.06 + support_signal * 0.04)), 6)


def _learned_token_set(learned_parameters: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key in ("top_input_tokens", "top_chosen_tokens"):
        entries = learned_parameters.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, dict):
                token = entry.get("token")
            else:
                token = entry
            if token is not None:
                tokens.update(_tokens(token))
    return tokens


def _learned_parameters_summary(learned_parameters: Any) -> dict[str, Any]:
    if not isinstance(learned_parameters, dict):
        return {}
    feature_weights = learned_parameters.get("feature_weights")
    return {
        "feature_weights": deepcopy(feature_weights) if isinstance(feature_weights, dict) else {},
        "source_type_weights": deepcopy(learned_parameters.get("source_type_weights") or {}),
        "tag_weights": deepcopy(learned_parameters.get("tag_weights") or {}),
        "top_input_tokens": deepcopy(learned_parameters.get("top_input_tokens") or [])[:10],
        "top_chosen_tokens": deepcopy(learned_parameters.get("top_chosen_tokens") or [])[:10],
    }


def _count_learned_parameters(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_learned_parameters(nested) for nested in value.values())
    if isinstance(value, list):
        return sum(_count_learned_parameters(nested) for nested in value)
    return 1


def _tokens(value: Any) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(_stringify(value))]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_stringify(nested) for _, nested in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(nested) for nested in value)
    return str(value)


def _set_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


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
