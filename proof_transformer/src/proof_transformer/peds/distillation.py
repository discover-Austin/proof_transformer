from __future__ import annotations

import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..timeutils import utc_now
from .promotion import summarize_promotion_evidence
from .shadow import summarize_shadow_run
from .store import get_candidate_current_state

_READY_STATES = {"distillation_candidate", "shadow_deployment", "promoted"}
_UNSAFE_VERDICTS = {"unsafe", "timeout", "rejected", "fail"}
_EXAMPLE_KEYS = {
    "example_id",
    "candidate_id",
    "source_type",
    "source_ref",
    "input",
    "chosen",
    "tags",
    "provenance",
    "safety_exclusions",
    "metadata",
}


def collect_distillation_sources(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    include_shadow: bool = True,
    include_replay: bool = True,
) -> dict[str, Any]:
    _require_candidate(conn, candidate_id)
    current_state = get_candidate_current_state(conn, candidate_id)
    if current_state not in _READY_STATES:
        raise ValueError(
            "Candidate must be promoted, shadow_deployment, or distillation_candidate before "
            f"distillation source collection, got {current_state!r}"
        )

    promotion_evidence = summarize_promotion_evidence(conn, candidate_id)
    decision_timeline = promotion_evidence["decision_timeline"]
    warnings = list(promotion_evidence.get("warnings") or [])
    distillation_candidate_decision = _latest_decision(decision_timeline, "distillation_candidate")

    replay_jobs = _load_jobs_with_results(conn, candidate_id, split="replay", decision="replay_winner") if include_replay else []
    canary_jobs = _load_jobs_with_results(conn, candidate_id, split="canary", decision="canary_passed")
    shadow_runs = _load_shadow_runs_with_events(conn, candidate_id) if include_shadow else []

    if not promotion_evidence.get("latest_replay_winner"):
        warnings.append("missing replay_winner decision")
    if include_replay and not replay_jobs:
        warnings.append("missing replay winner job/results")
    if not promotion_evidence.get("latest_canary_passed"):
        warnings.append("missing canary_passed decision")
    if not canary_jobs:
        warnings.append("missing canary job/results")
    if not promotion_evidence.get("latest_shadow_passed"):
        warnings.append("missing shadow_passed decision")
    if include_shadow and not shadow_runs:
        warnings.append("missing shadow runs/events")

    hard_missing = [warning for warning in warnings if warning.startswith("missing ")]
    if hard_missing:
        raise ValueError("Candidate lacks required distillation evidence: " + "; ".join(hard_missing))

    warnings.extend(_canary_warnings(canary_jobs))
    if current_state in {"promoted", "shadow_deployment"} and distillation_candidate_decision is None:
        warnings.append("distillation_candidate decision absent; collecting from promoted/deployment survivor")

    return {
        "candidate_id": candidate_id,
        "current_state": current_state,
        "promotion_evidence": promotion_evidence,
        "decision_timeline": decision_timeline,
        "replay_jobs": replay_jobs,
        "canary_jobs": canary_jobs,
        "shadow_runs": shadow_runs,
        "distillation_candidate_decision": distillation_candidate_decision,
        "include_shadow": include_shadow,
        "include_replay": include_replay,
        "warnings": warnings,
    }


def build_distillation_examples(source_packet: dict[str, Any], *, max_examples: int | None = None) -> list[dict[str, Any]]:
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative or None")

    examples: list[dict[str, Any]] = []
    candidate_id = source_packet["candidate_id"]

    for run in sorted(source_packet.get("shadow_runs") or [], key=_shadow_run_sort_key):
        if run.get("status") != "completed":
            continue
        for event in sorted(run.get("events") or [], key=_shadow_event_sort_key):
            example = _build_shadow_example(candidate_id, run, event)
            if example is not None:
                examples.append(example)

    for job in sorted(source_packet.get("replay_jobs") or [], key=_job_sort_key):
        for result in sorted(job.get("results") or [], key=_result_sort_key):
            example = _build_replay_example(candidate_id, job, result)
            if example is not None:
                examples.append(example)

    for index, example in enumerate(examples):
        example["example_id"] = _example_id(index, example)

    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def write_distillation_manifest(
    conn: sqlite3.Connection,
    candidate_id: str,
    *,
    output_path: str | None = None,
    max_examples: int | None = None,
) -> dict[str, Any]:
    source_packet = collect_distillation_sources(conn, candidate_id)
    examples = build_distillation_examples(source_packet, max_examples=max_examples)
    manifest = {
        "candidate_id": candidate_id,
        "created_at": utc_now(),
        "dataset_hash": _dataset_hash(examples),
        "n_examples": len(examples),
        "examples": examples,
        "source_summary": _source_summary(source_packet),
        "warnings": list(source_packet.get("warnings") or []),
    }

    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json_pretty(manifest), encoding="utf-8")

    return manifest


def validate_distillation_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    failed_checks: list[str] = []

    examples = manifest.get("examples")
    if not isinstance(examples, list):
        failed_checks.append("examples must be a list")
        examples = []

    if int(manifest.get("n_examples", 0)) <= 0:
        failed_checks.append("n_examples must be > 0")
    if manifest.get("n_examples") != len(examples):
        failed_checks.append("n_examples does not match examples length")

    expected_hash = _dataset_hash(examples)
    if manifest.get("dataset_hash") != expected_hash:
        failed_checks.append("dataset_hash does not match canonical examples")

    seen_ids: set[str] = set()
    for index, example in enumerate(examples):
        missing_keys = sorted(_EXAMPLE_KEYS - set(example))
        if missing_keys:
            failed_checks.append(f"example {index} missing keys: {', '.join(missing_keys)}")
        example_id = example.get("example_id")
        if not example_id:
            failed_checks.append(f"example {index} missing example_id")
        elif example_id in seen_ids:
            failed_checks.append(f"duplicate example_id: {example_id}")
        seen_ids.add(str(example_id))

        if example.get("source_type") not in {"shadow_event", "replay_result"}:
            failed_checks.append(f"example {index} has unsupported source_type")
        if not example.get("provenance"):
            failed_checks.append(f"example {index} missing provenance")
        if not example.get("chosen"):
            failed_checks.append(f"example {index} missing chosen output")
        if example.get("safety_exclusions"):
            failed_checks.append(f"example {index} has safety exclusions")
        if _example_is_unsafe(example):
            failed_checks.append(f"example {index} is unsafe")

    try:
        _canonical_json(manifest)
    except (TypeError, ValueError) as exc:
        failed_checks.append(f"manifest is not canonical-json serializable: {exc}")

    valid = not failed_checks
    return {
        "valid": valid,
        "reason": "distillation manifest is valid" if valid else "; ".join(failed_checks),
        "failed_checks": failed_checks,
        "manifest": deepcopy(manifest),
    }


def _load_jobs_with_results(conn: sqlite3.Connection, candidate_id: str, *, split: str, decision: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT DISTINCT rj.*
        FROM promotion_decisions pd
        JOIN replay_jobs rj ON rj.job_id = pd.job_id
        WHERE pd.candidate_id = ? AND pd.decision = ? AND rj.split = ?
        ORDER BY rj.created_at, rj.job_id
        """,
        (candidate_id, decision, split),
    ).fetchall()

    jobs: list[dict[str, Any]] = []
    for row in rows:
        job = dict(row)
        job["runtime"] = _decode_json(job.pop("runtime_json"))
        job["results"] = _load_job_results(conn, job["job_id"])
        jobs.append(job)
    return jobs


def _load_job_results(conn: sqlite3.Connection, job_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            rje.episode_order,
            rr.*,
            e.task_prompt,
            e.task_title,
            e.acceptance_tests_json,
            e.policy_tags_json,
            e.raw_episode_json
        FROM replay_job_episodes rje
        JOIN replay_results rr ON rr.job_id = rje.job_id AND rr.episode_id = rje.episode_id
        JOIN episodes e ON e.episode_id = rje.episode_id
        WHERE rje.job_id = ?
        ORDER BY rje.episode_order, rr.created_at, rr.result_id
        """,
        (job_id,),
    ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        result = dict(row)
        result["canary_failures"] = _decode_json(result.pop("canary_failures_json"))
        result["acceptance_tests"] = _decode_json(result.pop("acceptance_tests_json")) if result.get("acceptance_tests_json") else None
        result["policy_tags"] = _decode_json(result.pop("policy_tags_json")) if result.get("policy_tags_json") else None
        result["raw_episode"] = _decode_json(result.pop("raw_episode_json"))
        results.append(result)
    return results


def _load_shadow_runs_with_events(conn: sqlite3.Connection, candidate_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM shadow_runs
        WHERE candidate_id = ?
        ORDER BY started_at, shadow_run_id
        """,
        (candidate_id,),
    ).fetchall()

    runs: list[dict[str, Any]] = []
    for row in rows:
        run = dict(row)
        run["thresholds"] = _decode_json(run.pop("thresholds_json"))
        run["metadata"] = _decode_json(run.pop("metadata_json"))
        run["summary"] = _shadow_summary_for_loaded_run(conn, run)
        run["events"] = _load_shadow_events(conn, run["shadow_run_id"])
        runs.append(run)
    return runs


def _shadow_summary_for_loaded_run(conn: sqlite3.Connection, run: dict[str, Any]) -> dict[str, Any] | None:
    if run.get("status") != "completed":
        return None
    return summarize_shadow_run(conn, run["shadow_run_id"])


def _load_shadow_events(conn: sqlite3.Connection, shadow_run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM shadow_events
        WHERE shadow_run_id = ?
        ORDER BY ordinal, created_at, shadow_event_id
        """,
        (shadow_run_id,),
    ).fetchall()

    events: list[dict[str, Any]] = []
    for row in rows:
        event = dict(row)
        event["input"] = _decode_json(event.pop("input_json"))
        event["incumbent_output"] = _decode_json(event.pop("incumbent_output_json"))
        event["candidate_output"] = _decode_json(event.pop("candidate_output_json"))
        event["divergence"] = _decode_json(event.pop("divergence_json")) if event.get("divergence_json") else None
        event["candidate_would_act"] = _decode_json(event.pop("candidate_would_act_json")) if event.get("candidate_would_act_json") else None
        event["incumbent_did_act"] = _decode_json(event.pop("incumbent_did_act_json")) if event.get("incumbent_did_act_json") else None
        event["accepted_by_incumbent"] = _decode_bool(event.pop("accepted_by_incumbent"))
        event["metadata"] = _decode_json(event.pop("metadata_json"))
        events.append(event)
    return events


def _build_shadow_example(candidate_id: str, run: dict[str, Any], event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("accepted_by_incumbent") is not True:
        return None
    if _shadow_event_is_unsafe(event):
        return None
    chosen = event.get("candidate_output")
    if not chosen:
        return None

    rejected = event.get("incumbent_output") if event.get("incumbent_output") != chosen else None
    example = {
        "example_id": "",
        "candidate_id": candidate_id,
        "source_type": "shadow_event",
        "source_ref": event["shadow_event_id"],
        "input": deepcopy(event.get("input")),
        "chosen": deepcopy(chosen),
        "tags": ["peds", "shadow_event", "accepted_by_incumbent"],
        "provenance": {
            "source_table": "shadow_events",
            "shadow_run_id": run["shadow_run_id"],
            "shadow_event_id": event["shadow_event_id"],
            "ordinal": event["ordinal"],
            "created_at": event["created_at"],
            "accepted_by_incumbent": True,
        },
        "safety_exclusions": [],
        "metadata": {
            "task_ref": event.get("task_ref"),
            "episode_ref": event.get("episode_ref"),
            "divergence": deepcopy(event.get("divergence")),
            "candidate_would_act": deepcopy(event.get("candidate_would_act")),
            "incumbent_did_act": deepcopy(event.get("incumbent_did_act")),
            "event_metadata": deepcopy(event.get("metadata") or {}),
            "run_metadata": deepcopy(run.get("metadata") or {}),
        },
    }
    if rejected is not None:
        example["rejected"] = deepcopy(rejected)
    return example


def _build_replay_example(candidate_id: str, job: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
    if str(result.get("judge_verdict")) in _UNSAFE_VERDICTS:
        return None
    if result.get("canary_failures"):
        return None
    if float(result["candidate_score"]) <= float(result["incumbent_score"]):
        return None

    chosen = _chosen_from_raw_episode(result)
    if not chosen:
        return None

    raw_episode = result.get("raw_episode") or {}
    task = raw_episode.get("task") or {}
    tags = ["peds", "replay_result", "replay_win"]
    tags.extend(result.get("policy_tags") or task.get("policy_tags") or [])

    return {
        "example_id": "",
        "candidate_id": candidate_id,
        "source_type": "replay_result",
        "source_ref": result["result_id"],
        "input": {
            "task_id": raw_episode.get("task_id") or task.get("task_id"),
            "title": raw_episode.get("task_title") or task.get("title") or result.get("task_title"),
            "prompt": raw_episode.get("task_prompt") or task.get("prompt") or result.get("task_prompt"),
            "acceptance_tests": result.get("acceptance_tests") or task.get("acceptance_tests"),
            "context": (raw_episode.get("run") or {}).get("context_selected", []),
        },
        "chosen": chosen,
        "tags": _unique_strings(tags),
        "provenance": {
            "source_table": "replay_results",
            "job_id": job["job_id"],
            "result_id": result["result_id"],
            "episode_id": result["episode_id"],
            "episode_order": result["episode_order"],
            "split": job["split"],
            "evaluator_version": job["evaluator_version"],
            "candidate_score": float(result["candidate_score"]),
            "incumbent_score": float(result["incumbent_score"]),
            "delta": float(result["delta"]),
        },
        "safety_exclusions": [],
        "metadata": {
            "judge_verdict": result.get("judge_verdict"),
            "audit_log_ref": result.get("audit_log_ref"),
            "created_at": result.get("created_at"),
        },
    }


def _chosen_from_raw_episode(result: dict[str, Any]) -> dict[str, Any] | None:
    raw_episode = result.get("raw_episode") or {}
    run = raw_episode.get("run") or {}
    chosen = {
        "final_message": raw_episode.get("final_message") or run.get("final_message"),
        "patch": raw_episode.get("patch") or run.get("patch"),
    }
    actions = raw_episode.get("actions") or run.get("actions")
    if actions:
        chosen["actions"] = deepcopy(actions)
    return {key: value for key, value in chosen.items() if value not in (None, [], {}, "")}


def _source_summary(source_packet: dict[str, Any]) -> dict[str, Any]:
    replay_jobs = source_packet.get("replay_jobs") or []
    canary_jobs = source_packet.get("canary_jobs") or []
    shadow_runs = source_packet.get("shadow_runs") or []
    return {
        "candidate_id": source_packet["candidate_id"],
        "current_state": source_packet["current_state"],
        "n_decisions": len(source_packet.get("decision_timeline") or []),
        "n_replay_jobs": len(replay_jobs),
        "n_replay_results": sum(len(job.get("results") or []) for job in replay_jobs),
        "n_canary_jobs": len(canary_jobs),
        "n_canary_results": sum(len(job.get("results") or []) for job in canary_jobs),
        "n_shadow_runs": len(shadow_runs),
        "n_shadow_events": sum(len(run.get("events") or []) for run in shadow_runs),
        "distillation_candidate_decision_id": (source_packet.get("distillation_candidate_decision") or {}).get("decision_id"),
    }


def _canary_warnings(canary_jobs: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for job in canary_jobs:
        for result in job.get("results") or []:
            if result.get("judge_verdict") in {"unsafe", "timeout"}:
                warnings.append(f"unsafe canary result excluded: {result['result_id']}")
            if result.get("canary_failures"):
                warnings.append(f"canary failures present in source evidence: {result['result_id']}")
    return warnings


def _shadow_event_is_unsafe(event: dict[str, Any]) -> bool:
    metadata = event.get("metadata") or {}
    divergence = event.get("divergence") or {}
    values = [
        metadata.get("judge_verdict"),
        metadata.get("safety"),
        metadata.get("status"),
        divergence.get("judge_verdict") if isinstance(divergence, dict) else None,
        divergence.get("safety") if isinstance(divergence, dict) else None,
    ]
    return any(str(value).lower() in _UNSAFE_VERDICTS for value in values if value is not None)


def _example_is_unsafe(example: dict[str, Any]) -> bool:
    metadata = example.get("metadata") or {}
    provenance = example.get("provenance") or {}
    values = [
        metadata.get("judge_verdict"),
        metadata.get("safety"),
        provenance.get("judge_verdict"),
    ]
    tags = [str(tag).lower() for tag in example.get("tags") or []]
    return any(str(value).lower() in _UNSAFE_VERDICTS for value in values if value is not None) or any(
        tag in _UNSAFE_VERDICTS for tag in tags
    )


def _require_candidate(conn: sqlite3.Connection, candidate_id: str) -> None:
    row = conn.execute(
        "SELECT candidate_id FROM candidates WHERE candidate_id = ? LIMIT 1",
        (candidate_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")


def _latest_decision(timeline: list[dict[str, Any]], decision: str) -> dict[str, Any] | None:
    for row in timeline:
        if row.get("decision") == decision:
            return deepcopy(row)
    return None


def _job_sort_key(job: dict[str, Any]) -> tuple[str, str]:
    return (str(job.get("created_at") or ""), str(job.get("job_id") or ""))


def _result_sort_key(result: dict[str, Any]) -> tuple[int, str, str]:
    return (int(result.get("episode_order", 0)), str(result.get("created_at") or ""), str(result.get("result_id") or ""))


def _shadow_run_sort_key(run: dict[str, Any]) -> tuple[str, str]:
    return (str(run.get("started_at") or ""), str(run.get("shadow_run_id") or ""))


def _shadow_event_sort_key(event: dict[str, Any]) -> tuple[int, str, str]:
    return (int(event.get("ordinal", 0)), str(event.get("created_at") or ""), str(event.get("shadow_event_id") or ""))


def _example_id(index: int, example: dict[str, Any]) -> str:
    identity = {
        "candidate_id": example["candidate_id"],
        "source_type": example["source_type"],
        "source_ref": example["source_ref"],
        "index": index,
    }
    digest = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:16]
    return f"distill_example_{index:06d}_{digest}"


def _dataset_hash(examples: list[dict[str, Any]]) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(examples).encode("utf-8")).hexdigest()


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _decode_bool(value: int | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
