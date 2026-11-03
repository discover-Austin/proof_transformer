from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
EXAMPLES = ROOT / "examples"
for path in (SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peds_first_local_experiment import (  # noqa: E402
    _contains_weight_payload,
    _create_experiment_candidate,
    _deny_network_calls,
    _record_local_episode,
)
from proof_transformer.db import get_connection  # noqa: E402
from proof_transformer.paths import get_brain_paths, reset_cached_paths  # noqa: E402
from proof_transformer.peds.cycle import advance_peds_cycle, run_peds_cycle_until_blocked  # noqa: E402
from proof_transformer.peds.canary import create_canary_job  # noqa: E402
from proof_transformer.peds.local_signals import (  # noqa: E402
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
    load_candidate_local_adapter,
    make_local_episode_shadow_event_provider,
)
from proof_transformer.peds.mutation_strategies import validate_single_mutation  # noqa: E402
from proof_transformer.peds.store import create_candidate, get_candidate_current_state  # noqa: E402


def main() -> None:
    args = _parse_args()
    run_id = _run_id()
    brain_root = args.run_root or ROOT / "temp" / "peds_trained_local_generations" / run_id
    report = run_generations(
        brain_root=brain_root,
        run_id=run_id,
        generations=args.generations,
    )
    report_path = Path(report["brain_root"]) / "peds_generation_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def run_generations(
    *,
    brain_root: Path,
    run_id: str,
    generations: int,
    strict: bool = True,
    parent_candidate_id: str | None = None,
    generation: int | None = None,
    generation_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(generations, int) or isinstance(generations, bool) or generations <= 0:
        raise ValueError("generations must be a positive int")
    if generation is not None and (not isinstance(generation, int) or isinstance(generation, bool) or generation <= 0):
        raise ValueError("generation must be a positive int when supplied")
    if generation is not None and generations != 1:
        raise ValueError("generation override is only supported for one-generation runs")

    previous_root = os.environ.get("AUSTIN_AI_BRAIN_ROOT")
    os.environ["AUSTIN_AI_BRAIN_ROOT"] = str(brain_root)
    reset_cached_paths()
    conn = get_connection()
    try:
        paths = get_brain_paths(refresh=True)
        with _deny_network_calls() as network_attempts:
            options = deepcopy(generation_options or {})
            input_parent_state_before = (
                get_candidate_current_state(conn, parent_candidate_id) if parent_candidate_id else None
            )
            if parent_candidate_id:
                attempt_candidate = _create_lineage_attempt_candidate(
                    conn,
                    run_id=run_id,
                    generation=generation or 1,
                    parent_candidate_id=parent_candidate_id,
                    options=options,
                )
                current_candidate_id = attempt_candidate["candidate_id"]
            else:
                attempt_candidate = _seed_initial_chain(conn, run_id=run_id)
                current_candidate_id = attempt_candidate["candidate_id"]
            generation_reports: list[dict[str, Any]] = []
            generated_payloads: list[Any] = []
            generation_numbers = [generation] if generation is not None else list(range(1, generations + 1))

            for current_generation in generation_numbers:
                artifact_suffix = _artifact_suffix(options, current_generation)
                episode_ids = _record_generation_episodes(
                    conn,
                    run_id=run_id,
                    generation=current_generation,
                    tag_suffix=artifact_suffix,
                )
                _create_attempt_canary_job(
                    conn,
                    candidate_id=current_candidate_id,
                    episode_ids=episode_ids["canary"],
                    options=options,
                )
                generation_dir = paths.root / "experiments" / artifact_suffix
                trainer_config = {"epochs": 1, "generation": current_generation}
                trainer_config.update(options.get("trainer_config") or {})
                cycle_result = run_peds_cycle_until_blocked(
                    conn,
                    current_candidate_id,
                    operator="owner",
                    auto_promote=True,
                    max_steps=10,
                    replay_evaluator=acceptance_checklist_replay_evaluator,
                    canary_evaluator=acceptance_checklist_canary_evaluator,
                    shadow_event_provider=make_local_episode_shadow_event_provider(
                        split="online",
                        limit=10,
                        policy_tags=[episode_ids["generation_tag"]],
                    ),
                    options={
                        "promotion_reason": f"trained local PEDS generation {current_generation} promotion",
                        "distillation_reason": f"trained local PEDS generation {current_generation} selected for distillation",
                        "trainer_name": "local_json_trainer",
                        "trainer_config": trainer_config,
                        "run_training": True,
                        "trainer_mode": "local_json_trainer",
                        "adapter_name": f"acceptance_checklist_theta_gen_{current_generation:02d}",
                        "output_path": str(generation_dir / "distillation_manifest.json"),
                        "artifact_dir": str(generation_dir / "artifacts"),
                        "require_trained": True,
                        "replay_episode_ids": episode_ids["replay"],
                        "canary_episode_ids": episode_ids["canary"],
                        "policy_tags": [episode_ids["generation_tag"]],
                    },
                )
                try:
                    generated = _generated_child_from_cycle(cycle_result)
                except AssertionError as exc:
                    if strict:
                        raise
                    report = _partial_generation_report(
                        conn,
                        paths_root=paths.root,
                        run_id=run_id,
                        generations=generations,
                        generation_reports=generation_reports,
                        current_candidate_id=current_candidate_id,
                        input_parent_candidate_id=parent_candidate_id,
                        input_parent_state_before_attempt=input_parent_state_before,
                        input_source_candidate_id=current_candidate_id,
                        generation=current_generation,
                        blocker=f"{type(exc).__name__}: {exc}",
                        cycle_result=cycle_result,
                        network_attempts=network_attempts,
                        generated_payloads=generated_payloads,
                        episode_ids=episode_ids,
                        options=options,
                    )
                    return report
                generated_candidate = generated["candidate"]
                generated_replay_job = generated["replay_job"]
                training_result = generated["training_result"]
                generated_payloads.extend([training_result, generated_candidate["manifest"]])

                replay_probe = _run_generated_replay_probe(
                    conn,
                    generated_candidate,
                    replay_episode_ids=episode_ids["replay"],
                    policy_tags=[episode_ids["generation_tag"]],
                )
                generated_payloads.append(replay_probe["artifact"])
                generation_report = _generation_report(
                    conn,
                    generation=current_generation,
                    source_candidate_id=current_candidate_id,
                    cycle_result=cycle_result,
                    generated_candidate=generated_candidate,
                    generated_replay_job=generated_replay_job,
                    training_result=training_result,
                    replay_probe=replay_probe,
                    network_attempts=network_attempts,
                    episode_ids=episode_ids,
                    options=options,
                )
                try:
                    _assert_generation_report(generation_report)
                except AssertionError as exc:
                    if strict:
                        raise
                    generation_reports.append(generation_report)
                    report = _partial_generation_report(
                        conn,
                        paths_root=paths.root,
                        run_id=run_id,
                        generations=generations,
                        generation_reports=generation_reports,
                        current_candidate_id=generated_candidate["candidate_id"],
                        input_parent_candidate_id=parent_candidate_id,
                        input_parent_state_before_attempt=input_parent_state_before,
                        input_source_candidate_id=generation_report["source_candidate_id"],
                        generation=current_generation,
                        blocker=f"{type(exc).__name__}: generated generation report failed assertions",
                        cycle_result=cycle_result,
                        network_attempts=network_attempts,
                        generated_payloads=generated_payloads,
                        episode_ids=episode_ids,
                        options=options,
                    )
                    return report
                generation_reports.append(generation_report)
                current_candidate_id = generated_candidate["candidate_id"]

            model_modules_loaded = sorted(name for name in sys.modules if name.startswith("proof_transformer.model"))
            fake_weights_written = any(_contains_weight_payload(value) for value in generated_payloads)
            if network_attempts:
                raise AssertionError(f"Network attempts were blocked during local generation run: {network_attempts!r}")
            if model_modules_loaded:
                raise AssertionError(f"Model modules loaded during local generation run: {model_modules_loaded!r}")
            if fake_weights_written:
                raise AssertionError("Local trainer wrote a model-weight-shaped payload")

            return {
                "brain_root": str(paths.root),
                "fake_weights_written": fake_weights_written,
                "final_candidate_id": current_candidate_id,
                "final_candidate_state": get_candidate_current_state(conn, current_candidate_id),
                "generation_reports": generation_reports,
                "generations_completed": len(generation_reports),
                "generations_requested": generations,
                "input_parent_candidate_id": parent_candidate_id,
                "input_parent_state_after_attempt": (
                    get_candidate_current_state(conn, parent_candidate_id) if parent_candidate_id else None
                ),
                "input_parent_state_before_attempt": input_parent_state_before,
                "input_source_candidate_id": generation_reports[0]["source_candidate_id"] if generation_reports else current_candidate_id,
                "input_source_manifest": _candidate_manifest(conn, generation_reports[0]["source_candidate_id"] if generation_reports else current_candidate_id),
                "model_modules_loaded": model_modules_loaded,
                "network_attempts": list(network_attempts),
                "run_id": run_id,
                "trainer_mode": "local_json_trainer",
            }
    finally:
        conn.close()
        if previous_root is None:
            os.environ.pop("AUSTIN_AI_BRAIN_ROOT", None)
        else:
            os.environ["AUSTIN_AI_BRAIN_ROOT"] = previous_root
        reset_cached_paths()


def _artifact_suffix(options: dict[str, Any], generation: int) -> str:
    suffix = str(options.get("attempt_artifact_suffix") or f"trained_generation_{generation:02d}")
    cleaned = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in suffix)
    return cleaned or f"trained_generation_{generation:02d}"


def _create_attempt_canary_job(
    conn,
    *,
    candidate_id: str,
    episode_ids: list[str],
    options: dict[str, Any],
) -> str | None:
    if not options.get("force_new_canary_job"):
        return None
    if get_candidate_current_state(conn, candidate_id) != "replay_winner":
        return None
    incumbent_id = _latest_incumbent_id_from_decision(conn, candidate_id, "replay_winner")
    if not incumbent_id:
        return None
    metadata = {
        "adaptive_attempt": True,
        "attempt_artifact_suffix": options.get("attempt_artifact_suffix"),
        "canary_selection_policy": options.get("canary_selection_policy"),
        "correction_action": options.get("correction_action"),
        "generation": options.get("generation"),
    }
    thresholds = options.get("canary_thresholds") if isinstance(options.get("canary_thresholds"), dict) else None
    return create_canary_job(
        conn,
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        episode_ids=episode_ids,
        thresholds=thresholds,
        metadata=metadata,
    )


def _latest_incumbent_id_from_decision(conn, candidate_id: str, decision: str) -> str | None:
    row = conn.execute(
        """
        SELECT incumbent_id
        FROM promotion_decisions
        WHERE candidate_id = ?
          AND decision = ?
        ORDER BY created_at DESC, decision_id DESC
        LIMIT 1
        """,
        (candidate_id, decision),
    ).fetchone()
    return row["incumbent_id"] if row else None


def _partial_generation_report(
    conn,
    *,
    paths_root: Path,
    run_id: str,
    generations: int,
    generation_reports: list[dict[str, Any]],
    current_candidate_id: str,
    input_parent_candidate_id: str | None,
    input_parent_state_before_attempt: str | None,
    input_source_candidate_id: str,
    generation: int,
    blocker: str,
    cycle_result: dict[str, Any],
    network_attempts: list[Any],
    generated_payloads: list[Any],
    episode_ids: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    model_modules_loaded = sorted(name for name in sys.modules if name.startswith("proof_transformer.model"))
    fake_weights_written = any(_contains_weight_payload(value) for value in generated_payloads)
    step_log = cycle_result.get("step_log") or []
    return {
        "blocked_generation": generation,
        "blocker": blocker,
        "brain_root": str(paths_root),
        "cycle_actions": [step.get("action") for step in step_log if isinstance(step, dict) and step.get("action")],
        "cycle_blockers": [
            blocker_text
            for step in step_log
            if isinstance(step, dict)
            for blocker_text in (step.get("blockers") or [])
        ],
        "cycle_state_transitions": [
            {"action": step.get("action"), "state_before": step.get("state_before"), "state_after": step.get("state_after")}
            for step in step_log
            if isinstance(step, dict)
        ],
        "cycle_stopped_reason": cycle_result.get("stopped_reason"),
        "evidence": _evidence_report(
            conn,
            source_candidate_id=input_source_candidate_id,
            generated_replay_job_id=None,
            episode_ids=episode_ids,
            options=options,
        ),
        "fake_weights_written": fake_weights_written,
        "final_candidate_id": current_candidate_id,
        "final_candidate_state": get_candidate_current_state(conn, current_candidate_id),
        "generation_reports": generation_reports,
        "generations_completed": len(generation_reports),
        "generations_requested": generations,
        "input_parent_candidate_id": input_parent_candidate_id,
        "input_parent_state_after_attempt": (
            get_candidate_current_state(conn, input_parent_candidate_id) if input_parent_candidate_id else None
        ),
        "input_parent_state_before_attempt": input_parent_state_before_attempt,
        "input_source_candidate_id": input_source_candidate_id,
        "input_source_manifest": _candidate_manifest(conn, input_source_candidate_id),
        "last_cycle_action": (cycle_result.get("step_log") or [{}])[-1].get("action"),
        "last_cycle_blockers": (cycle_result.get("step_log") or [{}])[-1].get("blockers", []),
        "model_modules_loaded": model_modules_loaded,
        "network_attempts": list(network_attempts),
        "run_id": run_id,
        "trainer_mode": "local_json_trainer",
    }


def _seed_initial_chain(conn, *, run_id: str) -> dict[str, Any]:
    incumbent = _create_experiment_candidate(
        conn,
        run_id=run_id,
        suffix="incumbent",
        theta_version="theta_baseline_v1",
        manifest={
            "experiment_id": "trained_local_generations",
            "role": "incumbent",
            "theta_strategy": "baseline_final_answer",
        },
    )
    child = _create_experiment_candidate(
        conn,
        run_id=run_id,
        suffix="child",
        parent_candidate_id=incumbent["candidate_id"],
        theta_version="theta_acceptance_checklist_v1",
        manifest={
            "experiment_id": "trained_local_generations",
            "role": "candidate",
            "mutation_target": "theta",
            "theta_strategy": "acceptance_checklist_first",
        },
    )

    return child


def _create_lineage_attempt_candidate(
    conn,
    *,
    run_id: str,
    generation: int,
    parent_candidate_id: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    suffix = f"gen{generation:02d}_attempt_source"
    theta_strategy_suffix = str(options.get("theta_strategy_suffix") or "initial")
    parent_row = _candidate_row(conn, parent_candidate_id)
    parent_manifest = json.loads(parent_row["manifest_json"] or "{}")
    inherited_adapter = _inherited_adapter_manifest(parent_manifest, parent_candidate_id=parent_candidate_id)
    replay_parent_candidate_id = parent_row["parent_candidate_id"] or parent_candidate_id
    candidate_mutation = _candidate_mutation(options, generation=generation, fallback_suffix=theta_strategy_suffix)
    mutation_target = candidate_mutation["mutation_target"]
    theta_strategy = candidate_mutation.get("theta_strategy") or parent_manifest.get("theta_strategy") or "acceptance_checklist_first"
    generation_options = _manifest_generation_options(options)
    version_suffix = _safe_manifest_suffix(candidate_mutation["mutation_seed"])
    versions = _mutated_versions(parent_row, mutation_target=mutation_target, version_suffix=version_suffix)
    manifest = {
        "experiment_id": "trained_local_generations",
        "role": "candidate_attempt",
        "lineage_parent_candidate_id": parent_candidate_id,
        "mutation_target": mutation_target,
        "mutation_seed": candidate_mutation["mutation_seed"],
        "diagnosis_category": candidate_mutation["diagnosis_category"],
        "delta_description": candidate_mutation["delta_description"],
        "candidate_mutation": candidate_mutation,
        "parent_derivation": "fresh_attempt_candidate_from_lineage_parent",
        "replay_parent_candidate_id": replay_parent_candidate_id,
        "theta_strategy": theta_strategy,
        "theta_strategy_suffix": theta_strategy_suffix,
        "theta_strategy_variant": f"{theta_strategy}::{theta_strategy_suffix}",
        **_mutation_policy_fields(candidate_mutation),
        **inherited_adapter,
        "attempt_artifact_suffix": options.get("attempt_artifact_suffix"),
        "candidate_variant": options.get("candidate_variant"),
        "evidence_selection_seed": options.get("evidence_selection_seed"),
        "generation": generation,
        "generation_options": generation_options,
        "trainer_config": deepcopy(options.get("trainer_config") or {}),
    }
    return create_candidate(
        conn,
        {
            "candidate_id": f"candidate_peds_first_local_{run_id}_{suffix}",
            "parent_candidate_id": replay_parent_candidate_id,
            "mutation_target": mutation_target,
            "delta_description": candidate_mutation["delta_description"],
            "manifest_json": manifest,
            **versions,
        },
    )


def _candidate_row(conn, candidate_id: str):
    row = conn.execute(
        """
        SELECT candidate_id, parent_candidate_id, manifest_json, phi_version, rho_version, theta_version, psi_version
        FROM candidates
        WHERE candidate_id = ?
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown parent candidate: {candidate_id}")
    return row


def _candidate_mutation(options: dict[str, Any], *, generation: int, fallback_suffix: str) -> dict[str, Any]:
    mutation = deepcopy(options.get("candidate_mutation") or {})
    if not mutation:
        mutation = {
            "mutation_target": "theta",
            "theta_strategy": "acceptance_checklist_fresh_retry",
            "retrieval_policy": None,
            "tool_policy": None,
            "update_policy": None,
            "delta_description": "Creates a fresh theta retry when no campaign mutation was supplied.",
            "mutation_seed": f"peds_mut_manual_gen_{generation:02d}_{_safe_manifest_suffix(fallback_suffix)}",
            "diagnosis_category": "unknown",
        }
    validation = validate_single_mutation(mutation)
    if not validation["valid"]:
        raise ValueError("Invalid candidate mutation: " + "; ".join(validation["failed_checks"]))
    return mutation


def _mutation_policy_fields(candidate_mutation: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for key in ("retrieval_policy", "tool_policy", "update_policy"):
        value = candidate_mutation.get(key)
        if value not in (None, {}, [], ""):
            fields[key] = deepcopy(value)
    return fields


def _mutated_versions(parent_row, *, mutation_target: str, version_suffix: str) -> dict[str, Any]:
    versions = {
        "phi_version": parent_row["phi_version"],
        "rho_version": parent_row["rho_version"],
        "theta_version": parent_row["theta_version"],
        "psi_version": parent_row["psi_version"],
    }
    if mutation_target == "theta":
        versions["theta_version"] = f"theta_acceptance_checklist_{version_suffix}"
    elif mutation_target == "phi":
        versions["phi_version"] = f"phi_retrieval_{version_suffix}"
    elif mutation_target == "rho":
        versions["rho_version"] = f"rho_policy_{version_suffix}"
    elif mutation_target == "psi":
        versions["psi_version"] = f"psi_update_{version_suffix}"
    else:
        raise ValueError(f"Unsupported mutation target: {mutation_target}")
    return versions


def _inherited_adapter_manifest(parent_manifest: dict[str, Any], *, parent_candidate_id: str) -> dict[str, Any]:
    keys = (
        "adapter_ref",
        "adapter_registry_id",
        "artifact_ref",
        "dataset_hash",
        "source_training_run_id",
        "trained",
        "trainer_mode",
        "trainer_name",
    )
    inherited = {
        key: deepcopy(parent_manifest[key])
        for key in keys
        if key in parent_manifest
    }
    if parent_manifest.get("adapter_registration"):
        inherited["adapter_registration"] = deepcopy(parent_manifest["adapter_registration"])
    inherited["inherited_adapter_parent_candidate_id"] = parent_candidate_id
    return inherited


def _manifest_generation_options(options: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "active_policy",
        "active_policy_candidate_id",
        "active_policy_start",
        "attempt_artifact_suffix",
        "candidate_variant",
        "candidate_mutation",
        "canary_selection_policy",
        "canary_variant",
        "correction_action",
        "evidence_batch_policy",
        "evidence_selection_seed",
        "evidence_suffix",
        "generation_evidence_scope",
        "parent_policy",
        "parent_selection_reason",
        "replay_selection_policy",
        "replay_variant",
        "shadow_sampling_policy",
        "shadow_variant",
        "theta_strategy_suffix",
        "trainer_config",
    }
    return {key: deepcopy(options[key]) for key in sorted(allowed) if key in options}


def _safe_manifest_suffix(value: Any) -> str:
    text = str(value or "initial").lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part)[:48] or "initial"


def _candidate_manifest(conn, candidate_id: str) -> dict[str, Any]:
    row = _candidate_row(conn, candidate_id)
    return json.loads(row["manifest_json"] or "{}")


def _record_generation_episodes(
    conn,
    *,
    run_id: str,
    generation: int,
    tag_suffix: str | None = None,
) -> dict[str, Any]:
    base_generation_tag = f"trained_generation_{generation:02d}"
    generation_tag = tag_suffix or base_generation_tag
    learned_signal = f"pedsgen{generation:02d}signal"
    learned_focus = f"{learned_signal}_focus"
    learned_acceptance = f"{learned_signal}_acceptance"
    learned_check = f"{learned_signal} {learned_focus} {learned_acceptance}"
    replay_ids: list[str] = []
    canary_ids: list[str] = []
    online_ids: list[str] = []

    for index in range(4):
        episode = _record_local_episode(
            conn,
            run_id=f"{run_id}_gen{generation:02d}",
            index=index,
            split="replay",
            prompt=f"{learned_check} replay {index}",
            acceptance_tests=[learned_check],
            policy_tags=["peds_experiment", "acceptance_checklist", base_generation_tag, generation_tag],
        )
        replay_ids.append(episode["episode_id"])
    for index in range(3):
        episode = _record_local_episode(
            conn,
            run_id=f"{run_id}_gen{generation:02d}",
            index=index,
            split="canary",
            prompt=f"{learned_check} canary {index}",
            acceptance_tests=[learned_check],
            policy_tags=["peds_experiment", "acceptance_checklist", "canary", base_generation_tag, generation_tag],
        )
        canary_ids.append(episode["episode_id"])
    for index in range(10):
        episode = _record_local_episode(
            conn,
            run_id=f"{run_id}_gen{generation:02d}",
            index=index,
            split="online",
            prompt=f"{learned_check} shadow {index}",
            acceptance_tests=[learned_check],
            policy_tags=["peds_experiment", "acceptance_checklist", "shadow_source", base_generation_tag, generation_tag],
        )
        online_ids.append(episode["episode_id"])

    return {
        "base_generation_tag": base_generation_tag,
        "canary": canary_ids,
        "generation_tag": generation_tag,
        "learned_signal": learned_signal,
        "online": online_ids,
        "replay": replay_ids,
        "shadow_policy_tags": [generation_tag],
    }


def _generated_child_from_cycle(cycle_result: dict[str, Any]) -> dict[str, Any]:
    final_step = cycle_result["step_log"][-1]
    changed = final_step["changed"]
    required = ("candidate", "replay_job", "training_result")
    missing = [key for key in required if key not in changed]
    if missing:
        raise AssertionError(f"Cycle did not create generated child payload: missing {missing!r}")
    return {
        "candidate": changed["candidate"],
        "replay_job": changed["replay_job"],
        "training_result": changed["training_result"],
    }


def _run_generated_replay_probe(
    conn,
    generated_candidate: dict[str, Any],
    *,
    replay_episode_ids: list[str],
    policy_tags: list[str],
) -> dict[str, Any]:
    artifact = load_candidate_local_adapter(conn, generated_candidate)
    replay_step = advance_peds_cycle(
        conn,
        generated_candidate["candidate_id"],
        operator="owner",
        auto_promote=False,
        replay_evaluator=acceptance_checklist_replay_evaluator,
        options={
            "policy_tags": policy_tags,
            "replay_episode_ids": replay_episode_ids,
        },
    )
    replay_run = replay_step["changed"]["replay_run"]
    return {
        "adapter_loaded": artifact is not None,
        "adapter_mode": artifact.get("mode") if artifact else None,
        "artifact": artifact,
        "replay_action": replay_step["action"],
        "replay_blocked": replay_step["blocked"],
        "replay_gate_decision": replay_run["gate"]["decision"],
        "replay_mean_delta": replay_run["aggregate"]["mean_delta"],
        "replay_results_written": replay_run["results_written"],
        "state_after_replay": get_candidate_current_state(conn, generated_candidate["candidate_id"]),
    }


def _evidence_report(
    conn,
    *,
    source_candidate_id: str,
    generated_replay_job_id: str | None,
    episode_ids: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    source_replay_job_id = _latest_job_id(conn, candidate_id=source_candidate_id, split="replay")
    source_canary_job_id = _latest_job_id(conn, candidate_id=source_candidate_id, split="canary")
    source_shadow_run_id = _latest_shadow_run_id(conn, candidate_id=source_candidate_id)
    return {
        "attempt_artifact_suffix": options.get("attempt_artifact_suffix"),
        "base_generation_tag": episode_ids.get("base_generation_tag"),
        "canary_episode_ids": list(episode_ids.get("canary") or []),
        "canary_job_episode_ids": _job_episode_ids(conn, source_canary_job_id) if source_canary_job_id else [],
        "canary_job_id": source_canary_job_id,
        "executable_policy_tags": list(episode_ids.get("shadow_policy_tags") or [episode_ids.get("generation_tag")]),
        "generation_tag": episode_ids.get("generation_tag"),
        "generated_replay_episode_ids": _job_episode_ids(conn, generated_replay_job_id) if generated_replay_job_id else [],
        "generated_replay_job_id": generated_replay_job_id,
        "online_episode_ids": list(episode_ids.get("online") or []),
        "replay_episode_ids": list(episode_ids.get("replay") or []),
        "replay_job_episode_ids": _job_episode_ids(conn, source_replay_job_id) if source_replay_job_id else [],
        "replay_job_id": source_replay_job_id,
        "shadow_policy_tags": list(episode_ids.get("shadow_policy_tags") or [episode_ids.get("generation_tag")]),
        "shadow_run_id": source_shadow_run_id,
        "shadow_source_episode_ids": _shadow_event_episode_ids(conn, source_shadow_run_id) if source_shadow_run_id else [],
    }


def _generation_report(
    conn,
    *,
    generation: int,
    source_candidate_id: str,
    cycle_result: dict[str, Any],
    generated_candidate: dict[str, Any],
    generated_replay_job: dict[str, Any],
    training_result: dict[str, Any],
    replay_probe: dict[str, Any],
    network_attempts: list[Any],
    episode_ids: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    adapter_row = conn.execute(
        "SELECT name, status, path FROM adapter_registry WHERE id = ?",
        (generated_candidate["manifest"]["adapter_registry_id"],),
    ).fetchone()
    replay_result_count = conn.execute(
        "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
        (generated_replay_job["job_id"],),
    ).fetchone()["count"]
    replay_episode_count = conn.execute(
        "SELECT COUNT(*) AS count FROM replay_job_episodes WHERE job_id = ?",
        (generated_replay_job["job_id"],),
    ).fetchone()["count"]
    source_replay_job_id = _latest_job_id(conn, candidate_id=source_candidate_id, split="replay")
    source_canary_job_id = _latest_job_id(conn, candidate_id=source_candidate_id, split="canary")
    source_shadow_run_id = _latest_shadow_run_id(conn, candidate_id=source_candidate_id)
    evidence = {
        "attempt_artifact_suffix": options.get("attempt_artifact_suffix"),
        "base_generation_tag": episode_ids.get("base_generation_tag"),
        "canary_episode_ids": list(episode_ids.get("canary") or []),
        "canary_job_episode_ids": _job_episode_ids(conn, source_canary_job_id) if source_canary_job_id else [],
        "canary_job_id": source_canary_job_id,
        "executable_policy_tags": list(episode_ids.get("shadow_policy_tags") or [episode_ids.get("generation_tag")]),
        "generation_tag": episode_ids.get("generation_tag"),
        "generated_replay_episode_ids": _job_episode_ids(conn, generated_replay_job["job_id"]),
        "generated_replay_job_id": generated_replay_job["job_id"],
        "online_episode_ids": list(episode_ids.get("online") or []),
        "replay_episode_ids": list(episode_ids.get("replay") or []),
        "replay_job_episode_ids": _job_episode_ids(conn, source_replay_job_id) if source_replay_job_id else [],
        "replay_job_id": source_replay_job_id,
        "shadow_policy_tags": list(episode_ids.get("shadow_policy_tags") or [episode_ids.get("generation_tag")]),
        "shadow_run_id": source_shadow_run_id,
        "shadow_source_episode_ids": _shadow_event_episode_ids(conn, source_shadow_run_id) if source_shadow_run_id else [],
    }
    return {
        "adapter_registry_status": adapter_row["status"],
        "adapter_registry_name": adapter_row["name"],
        "cycle_steps_run": cycle_result["steps_run"],
        "cycle_stopped_reason": cycle_result["stopped_reason"],
        "generated_candidate_id": generated_candidate["candidate_id"],
        "generated_candidate_state": get_candidate_current_state(conn, generated_candidate["candidate_id"]),
        "generated_replay_episode_count": replay_episode_count,
        "generated_replay_job_id": generated_replay_job["job_id"],
        "generated_replay_result_count": replay_result_count,
        "evidence": evidence,
        "generation": generation,
        "manifest_path": str(Path(training_result["training_run"]["artifact_ref"]).parent.parent / "distillation_manifest.json"),
        "network_attempt_count": len(network_attempts),
        "output_artifact": training_result["training_run"]["artifact_ref"],
        "replay_probe": {key: value for key, value in replay_probe.items() if key != "artifact"},
        "source_candidate_id": source_candidate_id,
        "training_run_id": training_result["training_run"]["training_run_id"],
        "training_trained": training_result["trained"],
    }


def _latest_job_id(conn, *, candidate_id: str, split: str) -> str | None:
    row = conn.execute(
        """
        SELECT job_id
        FROM replay_jobs
        WHERE candidate_id = ? AND split = ?
        ORDER BY created_at DESC, job_id DESC
        LIMIT 1
        """,
        (candidate_id, split),
    ).fetchone()
    return row["job_id"] if row else None


def _latest_shadow_run_id(conn, *, candidate_id: str) -> str | None:
    row = conn.execute(
        """
        SELECT shadow_run_id
        FROM shadow_runs
        WHERE candidate_id = ?
        ORDER BY started_at DESC, shadow_run_id DESC
        LIMIT 1
        """,
        (candidate_id,),
    ).fetchone()
    return row["shadow_run_id"] if row else None


def _job_episode_ids(conn, job_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT episode_id
        FROM replay_job_episodes
        WHERE job_id = ?
        ORDER BY episode_order ASC
        """,
        (job_id,),
    ).fetchall()
    return [row["episode_id"] for row in rows]


def _shadow_event_episode_ids(conn, shadow_run_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT episode_ref
        FROM shadow_events
        WHERE shadow_run_id = ?
        ORDER BY ordinal ASC
        """,
        (shadow_run_id,),
    ).fetchall()
    return [row["episode_ref"] for row in rows if row["episode_ref"]]


def _assert_generation_report(report: dict[str, Any]) -> None:
    probe = report["replay_probe"]
    assert report["adapter_registry_status"] == "trained", report
    assert report["training_trained"] is True, report
    assert report["generated_replay_episode_count"] > 0, report
    assert report["generated_replay_result_count"] == probe["replay_results_written"], report
    assert probe["adapter_loaded"] is True, report
    assert probe["adapter_mode"] == "local_json_trainer", report
    assert probe["replay_action"] == "run_replay_job", report
    assert probe["replay_blocked"] is False, report
    assert probe["replay_gate_decision"] == "replay_winner", report
    assert probe["replay_results_written"] > 0, report
    assert probe["state_after_replay"] == "replay_winner", report
    assert report["generated_candidate_state"] == "replay_winner", report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run trained local PEDS for multiple causal learning generations."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of trained local PEDS generations to run. Defaults to 3.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional local brain root. Defaults to proof_transformer/temp/peds_trained_local_generations/<run_id>.",
    )
    return parser.parse_args()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


if __name__ == "__main__":
    main()
