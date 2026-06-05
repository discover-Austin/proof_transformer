from __future__ import annotations

import argparse
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
from proof_transformer.peds.local_signals import (  # noqa: E402
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
    load_candidate_local_adapter,
    make_local_episode_shadow_event_provider,
)
from proof_transformer.peds.store import get_candidate_current_state  # noqa: E402


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


def run_generations(*, brain_root: Path, run_id: str, generations: int) -> dict[str, Any]:
    if not isinstance(generations, int) or isinstance(generations, bool) or generations <= 0:
        raise ValueError("generations must be a positive int")

    previous_root = os.environ.get("AUSTIN_AI_BRAIN_ROOT")
    os.environ["AUSTIN_AI_BRAIN_ROOT"] = str(brain_root)
    reset_cached_paths()
    conn = get_connection()
    try:
        paths = get_brain_paths(refresh=True)
        with _deny_network_calls() as network_attempts:
            current_candidate_id = _seed_initial_chain(conn, run_id=run_id)
            generation_reports: list[dict[str, Any]] = []
            generated_payloads: list[Any] = []

            for generation in range(1, generations + 1):
                episode_ids = _record_generation_episodes(conn, run_id=run_id, generation=generation)
                generation_dir = paths.root / "experiments" / f"trained_generation_{generation:02d}"
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
                        "promotion_reason": f"trained local PEDS generation {generation} promotion",
                        "distillation_reason": f"trained local PEDS generation {generation} selected for distillation",
                        "trainer_name": "local_json_trainer",
                        "trainer_config": {"epochs": 1, "generation": generation},
                        "run_training": True,
                        "trainer_mode": "local_json_trainer",
                        "adapter_name": f"acceptance_checklist_theta_gen_{generation:02d}",
                        "output_path": str(generation_dir / "distillation_manifest.json"),
                        "artifact_dir": str(generation_dir / "artifacts"),
                        "require_trained": True,
                        "replay_episode_ids": episode_ids["replay"],
                        "canary_episode_ids": episode_ids["canary"],
                        "policy_tags": [episode_ids["generation_tag"]],
                    },
                )
                generated = _generated_child_from_cycle(cycle_result)
                generated_candidate = generated["candidate"]
                generated_replay_job = generated["replay_job"]
                training_result = generated["training_result"]
                generated_payloads.extend([training_result, generated_candidate["manifest"]])

                replay_probe = _run_generated_replay_probe(conn, generated_candidate)
                generated_payloads.append(replay_probe["artifact"])
                generation_report = _generation_report(
                    conn,
                    generation=generation,
                    source_candidate_id=current_candidate_id,
                    cycle_result=cycle_result,
                    generated_candidate=generated_candidate,
                    generated_replay_job=generated_replay_job,
                    training_result=training_result,
                    replay_probe=replay_probe,
                    network_attempts=network_attempts,
                )
                _assert_generation_report(generation_report)
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


def _seed_initial_chain(conn, *, run_id: str) -> str:
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

    return child["candidate_id"]


def _record_generation_episodes(conn, *, run_id: str, generation: int) -> dict[str, Any]:
    generation_tag = f"trained_generation_{generation:02d}"
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
            policy_tags=["peds_experiment", "acceptance_checklist", generation_tag],
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
            policy_tags=["peds_experiment", "acceptance_checklist", "canary", generation_tag],
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
            policy_tags=["peds_experiment", "acceptance_checklist", "shadow_source", generation_tag],
        )
        online_ids.append(episode["episode_id"])

    return {
        "canary": canary_ids,
        "generation_tag": generation_tag,
        "learned_signal": learned_signal,
        "online": online_ids,
        "replay": replay_ids,
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


def _run_generated_replay_probe(conn, generated_candidate: dict[str, Any]) -> dict[str, Any]:
    artifact = load_candidate_local_adapter(conn, generated_candidate)
    replay_step = advance_peds_cycle(
        conn,
        generated_candidate["candidate_id"],
        operator="owner",
        auto_promote=False,
        replay_evaluator=acceptance_checklist_replay_evaluator,
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
        "generation": generation,
        "manifest_path": str(Path(training_result["training_run"]["artifact_ref"]).parent.parent / "distillation_manifest.json"),
        "network_attempt_count": len(network_attempts),
        "output_artifact": training_result["training_run"]["artifact_ref"],
        "replay_probe": {key: value for key, value in replay_probe.items() if key != "artifact"},
        "source_candidate_id": source_candidate_id,
        "training_run_id": training_result["training_run"]["training_run_id"],
        "training_trained": training_result["trained"],
    }


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
