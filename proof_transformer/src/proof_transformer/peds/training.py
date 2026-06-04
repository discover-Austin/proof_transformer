from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..ids import new_id
from ..paths import get_brain_paths
from ..timeutils import utc_now
from .distillation import validate_distillation_manifest

_RESULT_STATUSES = {"completed", "failed", "aborted"}
_UNSAFE_TAGS = {"unsafe", "timeout", "rejected", "fail"}


def validate_training_readiness(
    conn: sqlite3.Connection,
    manifest: dict[str, Any],
    trainer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    manifest_validation = validate_distillation_manifest(manifest)
    if not manifest_validation["valid"]:
        failed_checks.extend(f"manifest: {check}" for check in manifest_validation["failed_checks"])

    if not manifest.get("dataset_hash"):
        failed_checks.append("dataset_hash is required")
    if int(manifest.get("n_examples", 0)) <= 0:
        failed_checks.append("n_examples must be > 0")

    examples = manifest.get("examples") if isinstance(manifest.get("examples"), list) else []
    for index, example in enumerate(examples):
        if not example.get("provenance"):
            failed_checks.append(f"example {index} missing provenance")
        if _example_is_unsafe(example):
            failed_checks.append(f"example {index} is unsafe")

    config = trainer_config or {}
    if not isinstance(config, dict):
        failed_checks.append("trainer_config must be a dict or None")
        config = {}
    try:
        _canonical_json(config)
    except (TypeError, ValueError) as exc:
        failed_checks.append(f"trainer_config is not JSON-serializable: {exc}")

    output_dir = config.get("output_dir")
    if output_dir is not None and not _is_safe_output_dir(str(output_dir)):
        failed_checks.append("output_dir must be relative or inside the configured brain root")

    ready = not failed_checks
    return {
        "ready": ready,
        "reason": "training manifest is ready" if ready else "; ".join(failed_checks),
        "failed_checks": failed_checks,
        "manifest_summary": _manifest_summary(manifest),
    }


def prepare_training_run(
    conn: sqlite3.Connection,
    manifest: dict[str, Any],
    *,
    trainer_name: str,
    trainer_config: dict[str, Any] | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    trainer_name_text = _require_non_empty_text(trainer_name, "trainer_name")
    config = _normalize_trainer_config(trainer_config)
    if output_dir is not None:
        config = dict(config)
        config["output_dir"] = output_dir
    output_dir_value = output_dir if output_dir is not None else config.get("output_dir")

    readiness = validate_training_readiness(conn, manifest, config)
    if not readiness["ready"]:
        raise ValueError(readiness["reason"])

    candidate_id = str(manifest["candidate_id"])
    _require_candidate(conn, candidate_id)

    training_run_id = new_id("peds_training_run")
    created_at = utc_now()
    manifest_payload = deepcopy(manifest)
    manifest_summary = readiness["manifest_summary"]

    with conn:
        conn.execute(
            """
            INSERT INTO peds_training_runs(
                training_run_id, candidate_id, created_at, updated_at, finished_at,
                status, dataset_hash, manifest_json, manifest_summary_json,
                trainer_name, trainer_config_json, output_dir, metrics_json,
                artifact_ref, error
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                training_run_id,
                candidate_id,
                created_at,
                created_at,
                None,
                "prepared",
                manifest_payload["dataset_hash"],
                _canonical_json(manifest_payload),
                _canonical_json(manifest_summary),
                trainer_name_text,
                _canonical_json(config),
                output_dir_value,
                _canonical_json({}),
                None,
                None,
            ),
        )

    run = get_training_run(conn, training_run_id)
    run["readiness"] = readiness
    return run


def record_training_result(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    status: str,
    metrics: dict[str, Any] | None = None,
    artifact_ref: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    if status not in _RESULT_STATUSES:
        raise ValueError(f"Unsupported training result status: {status}")
    get_training_run(conn, training_run_id)

    metrics_payload = metrics or {}
    if not isinstance(metrics_payload, dict):
        raise ValueError("metrics must be a dict or None")
    try:
        metrics_json = _canonical_json(metrics_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"metrics must be JSON-serializable: {exc}") from exc

    updated_at = utc_now()
    with conn:
        conn.execute(
            """
            UPDATE peds_training_runs
            SET status = ?, updated_at = ?, finished_at = ?, metrics_json = ?,
                artifact_ref = ?, error = ?
            WHERE training_run_id = ?
            """,
            (
                status,
                updated_at,
                updated_at,
                metrics_json,
                artifact_ref,
                error,
                training_run_id,
            ),
        )
    return get_training_run(conn, training_run_id)


def get_training_run(conn: sqlite3.Connection, training_run_id: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM peds_training_runs WHERE training_run_id = ? LIMIT 1",
        (training_run_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown PEDS training run: {training_run_id}")

    run = dict(row)
    run["manifest"] = _decode_json(run.pop("manifest_json"))
    run["manifest_summary"] = _decode_json(run.pop("manifest_summary_json"))
    run["trainer_config"] = _decode_json(run.pop("trainer_config_json"))
    run["metrics"] = _decode_json(run.pop("metrics_json"))
    return run


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": manifest.get("candidate_id"),
        "dataset_hash": manifest.get("dataset_hash"),
        "n_examples": manifest.get("n_examples"),
        "source_summary": deepcopy(manifest.get("source_summary") or {}),
        "warnings": deepcopy(manifest.get("warnings") or []),
    }


def _normalize_trainer_config(trainer_config: dict[str, Any] | None) -> dict[str, Any]:
    if trainer_config is None:
        return {}
    if not isinstance(trainer_config, dict):
        raise ValueError("trainer_config must be a dict or None")
    # Round-trip through canonical JSON so callers get only JSON-safe values.
    return json.loads(_canonical_json(trainer_config))


def _require_candidate(conn: sqlite3.Connection, candidate_id: str) -> None:
    row = conn.execute(
        "SELECT candidate_id FROM candidates WHERE candidate_id = ? LIMIT 1",
        (candidate_id,),
    ).fetchone()
    if not row:
        raise ValueError(f"Unknown candidate: {candidate_id}")


def _require_non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _is_safe_output_dir(value: str) -> bool:
    path = Path(value)
    if not path.is_absolute():
        return True
    brain_root = get_brain_paths().root.resolve()
    try:
        path.resolve().relative_to(brain_root)
        return True
    except ValueError:
        return False


def _example_is_unsafe(example: dict[str, Any]) -> bool:
    if example.get("safety_exclusions"):
        return True
    tags = [str(tag).lower() for tag in example.get("tags") or []]
    metadata = example.get("metadata") or {}
    provenance = example.get("provenance") or {}
    values = [metadata.get("judge_verdict"), metadata.get("safety"), provenance.get("judge_verdict")]
    return any(tag in _UNSAFE_TAGS for tag in tags) or any(
        str(value).lower() in _UNSAFE_TAGS for value in values if value is not None
    )


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
