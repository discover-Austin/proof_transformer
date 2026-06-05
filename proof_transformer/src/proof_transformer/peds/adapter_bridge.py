from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..learning.adapter_registry import AdapterRegistry
from .store import create_candidate, get_candidate_current_state
from .training import get_training_run


def validate_training_artifact_for_candidate(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    require_trained: bool = False,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    try:
        training_run = get_training_run(conn, training_run_id)
    except ValueError as exc:
        return {
            "valid": False,
            "reason": str(exc),
            "failed_checks": [str(exc)],
            "training_run": None,
        }

    if training_run["status"] != "completed":
        failed_checks.append(f"training run status must be completed, got {training_run['status']!r}")
    if not training_run.get("dataset_hash"):
        failed_checks.append("dataset_hash is required")

    manifest = training_run.get("manifest") or {}
    manifest_candidate_id = manifest.get("candidate_id")
    if not manifest_candidate_id:
        failed_checks.append("manifest candidate_id is required")
    elif not _candidate_exists(conn, str(manifest_candidate_id)):
        failed_checks.append(f"manifest candidate_id does not exist: {manifest_candidate_id}")

    metrics = training_run.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        failed_checks.append("training metrics are required")
        metrics = {}

    artifact_ref = training_run.get("artifact_ref")
    artifact_manifest = None
    if artifact_ref:
        artifact_path = Path(str(artifact_ref))
        if not artifact_path.exists():
            failed_checks.append(f"artifact_ref path does not exist: {artifact_ref}")
        else:
            try:
                artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                failed_checks.append(f"artifact_ref is not valid JSON: {exc}")
    else:
        failed_checks.append("artifact_ref is required for adapter registry bridge")

    trained = _training_run_trained(training_run, artifact_manifest)
    if require_trained and not trained:
        failed_checks.append("training artifact is metadata-only / trained=false")

    valid = not failed_checks
    return {
        "valid": valid,
        "reason": "training artifact is usable for candidate generation" if valid else "; ".join(failed_checks),
        "failed_checks": failed_checks,
        "training_run": training_run,
        "artifact_manifest": artifact_manifest,
        "trained": trained,
    }


def register_training_artifact(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    adapter_name: str,
    adapter_version: str | None = None,
    operator: str,
    require_trained: bool = False,
) -> dict[str, Any]:
    adapter_name_text = _require_non_empty_text(adapter_name, "adapter_name")
    operator_text = _require_non_empty_text(operator, "operator")
    validation = validate_training_artifact_for_candidate(
        conn,
        training_run_id,
        require_trained=require_trained,
    )
    if not validation["valid"]:
        raise ValueError(validation["reason"])

    training_run = validation["training_run"]
    version = adapter_version or _default_adapter_version(training_run_id)
    artifact_ref = training_run["artifact_ref"]
    trained = bool(validation["trained"])
    registry = AdapterRegistry(conn)
    existing = registry.get_adapter(name=adapter_name_text, version=version)
    if existing is not None:
        if existing.get("dataset_hash") != training_run["dataset_hash"]:
            raise ValueError("existing adapter version is registered to a different dataset_hash")
        if existing.get("path") != str(Path(str(artifact_ref))):
            raise ValueError("existing adapter version is registered to a different artifact_ref")
        adapter = existing
    else:
        adapter = registry.register_adapter(
            name=adapter_name_text,
            version=version,
            path=artifact_ref,
            parent_version=None,
            dataset_hash=training_run["dataset_hash"],
            eval_report_hash=None,
            status="trained" if trained else "metadata_only",
        )

    return {
        "training_run_id": training_run_id,
        "candidate_id": training_run["candidate_id"],
        "operator": operator_text,
        "adapter": adapter,
        "adapter_name": adapter_name_text,
        "adapter_version": version,
        "artifact_ref": artifact_ref,
        "trained": trained,
        "status": adapter["status"],
        "validation": validation,
    }


def create_candidate_from_training_run(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    parent_candidate_id: str,
    adapter_name: str,
    operator: str,
    require_trained: bool = False,
    delta_description: str | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    parent = _load_candidate(conn, parent_candidate_id)
    validation = validate_training_artifact_for_candidate(
        conn,
        training_run_id,
        require_trained=require_trained,
    )
    if not validation["valid"]:
        raise ValueError(validation["reason"])
    training_run = validation["training_run"]
    if training_run["candidate_id"] != parent_candidate_id:
        raise ValueError(
            "parent_candidate_id must match the candidate_id that produced the training run"
        )
    registration = register_training_artifact(
        conn,
        training_run_id,
        adapter_name=adapter_name,
        operator=operator_text,
        require_trained=require_trained,
    )
    adapter = registration["adapter"]
    trained = bool(registration["trained"])
    trainer_mode = (training_run.get("metrics") or {}).get("mode")
    theta_version = f"adapter:{adapter['name']}:{adapter['version']}"
    inherited_behavior = _inherited_behavior_manifest(parent["manifest"])

    manifest = {
        "source": "peds_training_run",
        "source_training_run_id": training_run_id,
        "parent_candidate_id": parent_candidate_id,
        **inherited_behavior,
        "dataset_hash": training_run["dataset_hash"],
        "trainer_name": training_run["trainer_name"],
        "trainer_mode": trainer_mode,
        "trained": trained,
        "adapter_registry_id": adapter.get("id"),
        "adapter_ref": {
            "name": adapter.get("name"),
            "version": adapter.get("version"),
            "path": adapter.get("path"),
            "status": adapter.get("status"),
        },
        "artifact_ref": training_run.get("artifact_ref"),
        "operator": operator_text,
    }
    candidate = create_candidate(
        conn,
        {
            "parent_candidate_id": parent_candidate_id,
            "mutation_target": "theta",
            "phi_version": parent["phi_version"],
            "rho_version": parent["rho_version"],
            "theta_version": theta_version,
            "psi_version": parent["psi_version"],
            "delta_description": delta_description
            or f"PEDS adapter candidate from training run {training_run_id}",
            "manifest_json": manifest,
        },
    )
    candidate["adapter_registration"] = registration
    return candidate


def _training_run_trained(training_run: dict[str, Any], artifact_manifest: dict[str, Any] | None) -> bool:
    metrics = training_run.get("metrics") or {}
    if metrics.get("trained") is True:
        return True
    if artifact_manifest and artifact_manifest.get("trained") is True:
        return True
    return False


def _default_adapter_version(training_run_id: str) -> str:
    suffix = training_run_id.removeprefix("peds_training_run_")[:16]
    return f"peds-{suffix}"


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


def _inherited_behavior_manifest(parent_manifest: dict[str, Any]) -> dict[str, Any]:
    inherited: dict[str, Any] = {}
    theta_strategy = parent_manifest.get("theta_strategy")
    if theta_strategy is not None:
        inherited["theta_strategy"] = str(theta_strategy)
    return inherited


def _require_non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()
