from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ..learning.adapter_registry import AdapterRegistry
from .local_signals import local_adapter_replay_signal
from .promotion import summarize_promotion_evidence
from .scoring import aggregate_replay_job
from .store import create_candidate, get_candidate_current_state, record_promotion_decision
from .training import get_training_run

_PROOF_SOURCE_STATES = {"distillation_candidate", "promoted", "shadow_deployment"}
_INVALIDATING_DECISIONS = {"rejected", "shadow_failed"}


def validate_training_artifact_for_candidate(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    require_trained: bool = False,
    proof_mode: bool = False,
    expected_candidate_id: str | None = None,
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
    elif expected_candidate_id is not None and str(manifest_candidate_id) != str(expected_candidate_id):
        failed_checks.append(
            f"manifest candidate_id {manifest_candidate_id!r} does not match expected {expected_candidate_id!r}"
        )

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

    strict_trained = bool(require_trained or proof_mode)
    trained = _training_run_trained(training_run, artifact_manifest, strict=strict_trained)
    if strict_trained and not trained:
        failed_checks.append("training artifact is metadata-only / trained=false")
    if proof_mode:
        _validate_training_artifact_manifest_for_proof(training_run, artifact_manifest, failed_checks)
        _validate_training_source_for_proof(conn, training_run, failed_checks)

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
    proof_mode: bool = False,
    allow_metadata_only: bool = False,
) -> dict[str, Any]:
    adapter_name_text = _require_non_empty_text(adapter_name, "adapter_name")
    operator_text = _require_non_empty_text(operator, "operator")
    validation = validate_training_artifact_for_candidate(
        conn,
        training_run_id,
        require_trained=require_trained or proof_mode,
        proof_mode=proof_mode,
    )
    if not validation["valid"]:
        raise ValueError(validation["reason"])

    training_run = validation["training_run"]
    version = adapter_version or _default_adapter_version(training_run_id)
    artifact_ref = training_run["artifact_ref"]
    trained = bool(validation["trained"])
    if not trained and not allow_metadata_only:
        raise ValueError("allow_metadata_only=True is required to register metadata-only adapter artifacts")
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
    proof_mode: bool = True,
    allow_metadata_only: bool = False,
    delta_description: str | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    parent = _load_candidate(conn, parent_candidate_id)
    validation = validate_training_artifact_for_candidate(
        conn,
        training_run_id,
        require_trained=require_trained or proof_mode,
        proof_mode=proof_mode,
        expected_candidate_id=parent_candidate_id,
    )
    if not validation["valid"]:
        raise ValueError(validation["reason"])
    training_run = validation["training_run"]
    if training_run["candidate_id"] != parent_candidate_id:
        raise ValueError(
            "parent_candidate_id must match the candidate_id that produced the training run"
        )
    if not bool(validation["trained"]) and not allow_metadata_only:
        raise ValueError("allow_metadata_only=True is required to create metadata-only adapter candidates")
    registration = register_training_artifact(
        conn,
        training_run_id,
        adapter_name=adapter_name,
        operator=operator_text,
        require_trained=require_trained or proof_mode,
        proof_mode=proof_mode,
        allow_metadata_only=allow_metadata_only,
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
        "training_trained": trained,
        "metadata_only": not trained,
        "proof_mode": bool(proof_mode),
        "adapter_registry_id": adapter.get("id"),
        "adapter_ref": {
            "name": adapter.get("name"),
            "version": adapter.get("version"),
            "path": adapter.get("path"),
            "status": adapter.get("status"),
        },
        "artifact_ref": training_run.get("artifact_ref"),
        "adapter_proof": {
            "proof_mode": bool(proof_mode),
            "training_trained": trained,
            "artifact_ref": training_run.get("artifact_ref"),
            "adapter_registry_id": adapter.get("id"),
            "adapter_registry_status": adapter.get("status"),
        },
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


def validate_post_training_replay_proof(
    conn: sqlite3.Connection,
    candidate_id: str,
    job_id: str,
) -> dict[str, Any]:
    failed_checks: list[str] = []
    candidate = _load_candidate(conn, candidate_id)
    manifest = candidate.get("manifest") or {}
    if manifest.get("source") != "peds_training_run":
        failed_checks.append("candidate is not a post-training adapter candidate")
    if manifest.get("proof_mode") is not True:
        failed_checks.append("candidate was not generated in proof_mode")
    if manifest.get("training_trained") is not True or manifest.get("trained") is not True:
        failed_checks.append("candidate manifest training_trained=true is required")

    _validate_candidate_adapter_registry_linkage(conn, candidate, failed_checks)

    aggregate: dict[str, Any] | None = None
    try:
        aggregate = aggregate_replay_job(conn, job_id)
    except ValueError as exc:
        failed_checks.append(str(exc))

    if aggregate is not None:
        if aggregate["split"] != "replay":
            failed_checks.append(f"post-training proof requires replay split, got {aggregate['split']!r}")
        if aggregate["candidate_id"] != candidate_id:
            failed_checks.append("replay job candidate_id does not match generated candidate")
        if not aggregate.get("incumbent_id"):
            failed_checks.append("replay job incumbent_id is required")
        if aggregate["status"] != "completed":
            failed_checks.append(f"replay job must be completed, got {aggregate['status']!r}")
        if not aggregate["complete"]:
            failed_checks.append("replay result set must cover every frozen episode exactly once")
        if aggregate.get("mean_delta") is None:
            failed_checks.append("replay_mean_delta is required")
        elif float(aggregate["mean_delta"]) <= 0.0:
            failed_checks.append("replay_mean_delta must be positive")
        if int(aggregate.get("hard_safety_violations") or 0) > 0:
            failed_checks.append("replay proof has hard safety violations")
        if int(aggregate.get("canary_failures") or 0) > 0:
            failed_checks.append("replay proof has canary failures")
        for decision in _candidate_invalidating_decisions(conn, candidate_id):
            failed_checks.append(f"candidate has invalidating decision: {decision['decision']}")

    behavior_signal = local_adapter_replay_signal(conn, candidate, job_id)
    if behavior_signal.get("adapter_loaded") is not True:
        failed_checks.append("adapter_loaded=true is required for post-training replay proof")
    if behavior_signal.get("adapter_mode") != "local_json_trainer":
        failed_checks.append("adapter replay proof requires local_json_trainer artifact")
    if int(behavior_signal.get("behavior_delta_count") or 0) <= 0:
        failed_checks.append("behavior-dependent adapter replay delta is required")

    valid = not failed_checks
    return {
        "valid": valid,
        "reason": "post-training replay proof is valid" if valid else "; ".join(failed_checks),
        "failed_checks": failed_checks,
        "candidate_id": candidate_id,
        "job_id": job_id,
        "aggregate": aggregate,
        "adapter_loaded": behavior_signal.get("adapter_loaded") is True,
        "behavior_signal": behavior_signal,
        "replay_mean_delta": aggregate.get("mean_delta") if aggregate is not None else None,
    }


def record_post_training_replay_proof_rejection(
    conn: sqlite3.Connection,
    proof: dict[str, Any],
) -> dict[str, Any] | None:
    aggregate = proof.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    existing = conn.execute(
        """
        SELECT decision_id
        FROM promotion_decisions
        WHERE candidate_id = ? AND job_id = ? AND decision = 'rejected'
        LIMIT 1
        """,
        (aggregate["candidate_id"], aggregate["job_id"]),
    ).fetchone()
    if existing:
        return None
    return record_promotion_decision(
        conn,
        {
            "candidate_id": aggregate["candidate_id"],
            "incumbent_id": aggregate["incumbent_id"],
            "job_id": aggregate["job_id"],
            "decision": "rejected",
            "mean_delta": aggregate.get("mean_delta"),
            "lcb95": aggregate.get("lcb95"),
            "regression_rate": aggregate.get("regression_rate"),
            "canary_failures": aggregate.get("canary_failures"),
            "token_ratio": aggregate.get("token_ratio"),
            "latency_ratio": aggregate.get("latency_ratio"),
            "reason": "post-training adapter proof rejected: " + str(proof.get("reason") or "invalid proof"),
        },
    )


def _training_run_trained(
    training_run: dict[str, Any],
    artifact_manifest: dict[str, Any] | None,
    *,
    strict: bool = False,
) -> bool:
    metrics = training_run.get("metrics") or {}
    artifact_trained = isinstance(artifact_manifest, dict) and artifact_manifest.get("trained") is True
    metrics_trained = metrics.get("trained") is True
    return metrics_trained and artifact_trained if strict else metrics_trained or artifact_trained


def _validate_training_artifact_manifest_for_proof(
    training_run: dict[str, Any],
    artifact_manifest: dict[str, Any] | None,
    failed_checks: list[str],
) -> None:
    if not isinstance(artifact_manifest, dict):
        failed_checks.append("proof mode requires a readable local JSON adapter artifact")
        return

    metrics = training_run.get("metrics") if isinstance(training_run.get("metrics"), dict) else {}
    artifact_metrics = artifact_manifest.get("metrics") if isinstance(artifact_manifest.get("metrics"), dict) else {}
    provenance = artifact_manifest.get("provenance") if isinstance(artifact_manifest.get("provenance"), dict) else {}

    if artifact_manifest.get("artifact_type") != "peds_adapter_artifact_manifest":
        failed_checks.append("artifact_type must be peds_adapter_artifact_manifest")
    if artifact_manifest.get("training_run_id") != training_run["training_run_id"]:
        failed_checks.append("artifact training_run_id does not match training run")
    if artifact_manifest.get("candidate_id") != training_run["candidate_id"]:
        failed_checks.append("artifact candidate_id does not match training run candidate_id")
    if artifact_manifest.get("dataset_hash") != training_run["dataset_hash"]:
        failed_checks.append("artifact dataset_hash does not match training run dataset_hash")
    if artifact_manifest.get("mode") != "local_json_trainer" or artifact_metrics.get("mode") != "local_json_trainer":
        failed_checks.append("proof mode requires local_json_trainer artifact mode")
    if artifact_manifest.get("trained") is not True:
        failed_checks.append("artifact trained=true is required")
    if artifact_metrics.get("trained") is not True or metrics.get("trained") is not True:
        failed_checks.append("training metrics trained=true is required")
    if provenance.get("metadata_only") is True:
        failed_checks.append("metadata_only artifacts cannot satisfy proof mode")
    if provenance.get("trained") is not True:
        failed_checks.append("artifact provenance trained=true is required")
    learned_parameters = artifact_manifest.get("learned_parameters")
    if not isinstance(learned_parameters, dict) or not learned_parameters:
        failed_checks.append("learned_parameters are required for proof mode")
    if _has_fake_weights_marker(training_run) or _has_fake_weights_marker(artifact_manifest):
        failed_checks.append("fake_weights_written cannot satisfy proof mode")


def _validate_training_source_for_proof(
    conn: sqlite3.Connection,
    training_run: dict[str, Any],
    failed_checks: list[str],
) -> None:
    candidate_id = str(training_run["candidate_id"])
    current_state = get_candidate_current_state(conn, candidate_id)
    if current_state not in _PROOF_SOURCE_STATES:
        failed_checks.append(
            "proof training source must be a promoted/shadow_deployment/distillation_candidate survivor, "
            f"got {current_state!r}"
        )

    manifest = training_run.get("manifest") if isinstance(training_run.get("manifest"), dict) else {}
    warnings = manifest.get("warnings") if isinstance(manifest.get("warnings"), list) else []
    if warnings:
        failed_checks.append("proof training source manifest has warnings: " + "; ".join(str(warning) for warning in warnings))

    source_summary = manifest.get("source_summary") if isinstance(manifest.get("source_summary"), dict) else {}
    summary_candidate_id = source_summary.get("candidate_id")
    if summary_candidate_id is not None and str(summary_candidate_id) != candidate_id:
        failed_checks.append("source_summary candidate_id does not match training run candidate_id")
    summary_state = source_summary.get("current_state")
    if summary_state is not None and str(summary_state) not in _PROOF_SOURCE_STATES:
        failed_checks.append(f"source_summary current_state is not proof-trainable: {summary_state!r}")

    evidence = summarize_promotion_evidence(conn, candidate_id)
    if not evidence.get("latest_replay_winner"):
        failed_checks.append("proof training source is missing replay_winner evidence")
    if not evidence.get("latest_canary_passed"):
        failed_checks.append("proof training source is missing canary_passed evidence")
    if not evidence.get("latest_shadow_passed"):
        failed_checks.append("proof training source is missing shadow_passed evidence")
    latest_shadow_decision = evidence.get("latest_shadow_decision")
    if latest_shadow_decision and latest_shadow_decision.get("decision") == "shadow_failed":
        failed_checks.append("proof training source latest shadow decision is shadow_failed")
    invalidating = evidence.get("invalidating_decisions_after_shadow_pass") or []
    if invalidating:
        failed_checks.append(
            "proof training source has invalidating decision after shadow_passed: "
            + ", ".join(str(decision.get("decision")) for decision in invalidating)
        )
    inconsistent = [warning for warning in evidence.get("warnings") or [] if "incumbent" in str(warning).lower()]
    if inconsistent:
        failed_checks.append("proof training source has inconsistent incumbent evidence")


def _validate_candidate_adapter_registry_linkage(
    conn: sqlite3.Connection,
    candidate: dict[str, Any],
    failed_checks: list[str],
) -> None:
    manifest = candidate.get("manifest") or {}
    adapter_registry_id = manifest.get("adapter_registry_id")
    adapter_ref = manifest.get("adapter_ref") if isinstance(manifest.get("adapter_ref"), dict) else {}
    artifact_ref = manifest.get("artifact_ref")
    if not adapter_registry_id:
        failed_checks.append("adapter_registry_id is required")
        return

    row = conn.execute(
        "SELECT * FROM adapter_registry WHERE id = ? LIMIT 1",
        (str(adapter_registry_id),),
    ).fetchone()
    if not row:
        failed_checks.append("adapter registry record is required")
        return

    adapter = dict(row)
    if adapter.get("status") != "trained":
        failed_checks.append(f"adapter registry status must be trained, got {adapter.get('status')!r}")
    if adapter_ref.get("name") and adapter_ref.get("name") != adapter.get("name"):
        failed_checks.append("adapter_ref name does not match adapter registry")
    if adapter_ref.get("version") and adapter_ref.get("version") != adapter.get("version"):
        failed_checks.append("adapter_ref version does not match adapter registry")
    if artifact_ref and str(Path(str(artifact_ref))) != str(Path(str(adapter.get("path")))):
        failed_checks.append("artifact_ref does not match adapter registry path")
    if adapter_ref.get("path") and str(Path(str(adapter_ref.get("path")))) != str(Path(str(adapter.get("path")))):
        failed_checks.append("adapter_ref path does not match adapter registry path")
    if adapter_ref.get("status") and adapter_ref.get("status") != adapter.get("status"):
        failed_checks.append("adapter_ref status does not match adapter registry status")


def _candidate_invalidating_decisions(conn: sqlite3.Connection, candidate_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT decision_id, decision
        FROM promotion_decisions
        WHERE candidate_id = ? AND decision IN ('rejected', 'shadow_failed')
        ORDER BY created_at DESC, decision_id DESC
        """,
        (candidate_id,),
    ).fetchall()
    return [dict(row) for row in rows if row["decision"] in _INVALIDATING_DECISIONS]


def _has_fake_weights_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if str(key) == "fake_weights_written" and bool(nested):
                return True
            if _has_fake_weights_marker(nested):
                return True
    elif isinstance(value, list):
        return any(_has_fake_weights_marker(nested) for nested in value)
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
