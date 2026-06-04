from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.db import get_connection
from proof_transformer.paths import get_brain_paths, reset_cached_paths
from proof_transformer.peds.cycle import run_peds_cycle_until_blocked
from proof_transformer.peds.local_signals import (
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
    make_local_episode_shadow_event_provider,
)
from proof_transformer.peds.store import create_candidate, get_candidate_current_state, record_episode


def main() -> None:
    args = _parse_args()
    run_id = _run_id()
    brain_root = args.run_root or ROOT / "temp" / "peds_first_local_experiment" / run_id
    report = run_experiment(brain_root=brain_root, run_id=run_id, trainer_mode=args.trainer_mode)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def run_experiment(*, brain_root: Path, run_id: str, trainer_mode: str = "dry_run") -> dict[str, Any]:
    previous_root = os.environ.get("AUSTIN_AI_BRAIN_ROOT")
    os.environ["AUSTIN_AI_BRAIN_ROOT"] = str(brain_root)
    reset_cached_paths()
    conn = get_connection()
    try:
        paths = get_brain_paths(refresh=True)
        with _deny_network_calls() as network_attempts:
            incumbent = _create_experiment_candidate(
                conn,
                run_id=run_id,
                suffix="incumbent",
                theta_version="theta_baseline_v1",
                manifest={
                    "experiment_id": "acceptance_checklist_theta_v1",
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
                    "experiment_id": "acceptance_checklist_theta_v1",
                    "role": "candidate",
                    "mutation_target": "theta",
                    "theta_strategy": "acceptance_checklist_first",
                },
            )

            for index in range(4):
                _record_local_episode(
                    conn,
                    run_id=run_id,
                    index=index,
                    split="replay",
                    prompt=f"Implement local replay task {index} and map the answer to its acceptance checklist.",
                    acceptance_tests=[
                        f"python -m unittest tests.acceptance_replay_{index}",
                        "python -m compileall src",
                    ],
                    policy_tags=["peds_experiment", "acceptance_checklist"],
                )
            for index in range(3):
                _record_local_episode(
                    conn,
                    run_id=run_id,
                    index=index,
                    split="canary",
                    prompt=f"Handle local canary task {index} without dropping acceptance checklist coverage.",
                    acceptance_tests=[f"python -m unittest tests.acceptance_canary_{index}"],
                    policy_tags=["peds_experiment", "acceptance_checklist", "canary"],
                )
            for index in range(10):
                _record_local_episode(
                    conn,
                    run_id=run_id,
                    index=index,
                    split="online",
                    prompt=f"Observed accepted local task {index} with checklist-first shadow evidence.",
                    acceptance_tests=[f"python -m unittest tests.shadow_acceptance_{index}"],
                    policy_tags=["peds_experiment", "acceptance_checklist", "shadow_source"],
                )

            experiment_dir = paths.root / "experiments" / "acceptance_checklist_theta_v1"
            manifest_path = experiment_dir / "distillation_manifest.json"
            artifact_dir = experiment_dir / "artifacts"
            trainer_name = "local_json_trainer" if trainer_mode == "local_json_trainer" else "local_adapter_stub"
            result = run_peds_cycle_until_blocked(
                conn,
                child["candidate_id"],
                operator="owner",
                auto_promote=True,
                max_steps=10,
                replay_evaluator=acceptance_checklist_replay_evaluator,
                canary_evaluator=acceptance_checklist_canary_evaluator,
                shadow_event_provider=make_local_episode_shadow_event_provider(
                    split="online",
                    limit=10,
                    policy_tags=["shadow_source"],
                ),
                options={
                    "promotion_reason": "acceptance-checklist-first theta beat baseline on local PEDS evidence",
                    "distillation_reason": "accepted local theta experiment survivor selected for distillation",
                    "trainer_name": trainer_name,
                    "trainer_config": {"epochs": 1},
                    "run_training": True,
                    "trainer_mode": trainer_mode,
                    "adapter_name": "acceptance_checklist_theta_v1",
                    "output_path": str(manifest_path),
                    "artifact_dir": str(artifact_dir),
                },
            )

        final_step = result["step_log"][-1]
        generated_candidate = final_step["changed"]["candidate"]
        generated_replay_job = final_step["changed"]["replay_job"]
        training_result = final_step["changed"]["training_result"]
        training_run = training_result["training_run"]
        artifact_path = Path(training_run["artifact_ref"])
        artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        adapter_row = conn.execute(
            "SELECT name, status, path FROM adapter_registry WHERE id = ?",
            (generated_candidate["manifest"]["adapter_registry_id"],),
        ).fetchone()
        expected_trained = trainer_mode == "local_json_trainer"
        expected_adapter_status = "trained" if expected_trained else "metadata_only"

        child_state = get_candidate_current_state(conn, child["candidate_id"])
        generated_state = get_candidate_current_state(conn, generated_candidate["candidate_id"])
        generated_replay_episode_count = conn.execute(
            "SELECT COUNT(*) AS count FROM replay_job_episodes WHERE job_id = ?",
            (generated_replay_job["job_id"],),
        ).fetchone()["count"]
        generated_replay_result_count = conn.execute(
            "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
            (generated_replay_job["job_id"],),
        ).fetchone()["count"]
        model_modules_loaded = sorted(name for name in sys.modules if name.startswith("proof_transformer.model"))
        fake_weights_written = any(
            _contains_weight_payload(value)
            for value in (training_result, artifact_manifest, generated_candidate["manifest"])
        )

        _assert_experiment_result(
            child_state=child_state,
            child=child,
            generated_candidate=generated_candidate,
            generated_state=generated_state,
            generated_replay_job=generated_replay_job,
            generated_replay_episode_count=generated_replay_episode_count,
            generated_replay_result_count=generated_replay_result_count,
            training_result=training_result,
            adapter_row=dict(adapter_row),
            expected_trained=expected_trained,
            expected_adapter_status=expected_adapter_status,
            network_attempts=network_attempts,
            model_modules_loaded=model_modules_loaded,
            fake_weights_written=fake_weights_written,
        )

        decisions = [
            row["decision"]
            for row in conn.execute(
                """
                SELECT decision
                FROM promotion_decisions
                WHERE candidate_id = ?
                ORDER BY
                    CASE decision
                        WHEN 'replay_winner' THEN 1
                        WHEN 'canary_passed' THEN 2
                        WHEN 'shadow_started' THEN 3
                        WHEN 'shadow_passed' THEN 4
                        WHEN 'promoted' THEN 5
                        WHEN 'distillation_candidate' THEN 6
                        ELSE 99
                    END,
                    created_at,
                    decision_id
                """,
                (child["candidate_id"],),
            ).fetchall()
        ]
        return {
            "adapter_registry_status": adapter_row["status"],
            "brain_root": str(paths.root),
            "child_candidate_id": child["candidate_id"],
            "child_state": child_state,
            "decisions": decisions,
            "fake_weights_written": fake_weights_written,
            "generated_candidate_id": generated_candidate["candidate_id"],
            "generated_candidate_state": generated_state,
            "generated_replay_job_id": generated_replay_job["job_id"],
            "generated_replay_job_split": generated_replay_job["split"],
            "generated_replay_episode_count": generated_replay_episode_count,
            "generated_replay_result_count": generated_replay_result_count,
            "incumbent_candidate_id": incumbent["candidate_id"],
            "manifest_path": str(manifest_path),
            "model_modules_loaded": model_modules_loaded,
            "network_attempts": list(network_attempts),
            "output_artifact": str(artifact_path),
            "stopped_reason": result["stopped_reason"],
            "trainer_mode": trainer_mode,
            "training_trained": training_result["trained"],
            "steps_run": result["steps_run"],
        }
    finally:
        conn.close()
        if previous_root is None:
            os.environ.pop("AUSTIN_AI_BRAIN_ROOT", None)
        else:
            os.environ["AUSTIN_AI_BRAIN_ROOT"] = previous_root
        reset_cached_paths()


def _create_experiment_candidate(
    conn,
    *,
    run_id: str,
    suffix: str,
    theta_version: str,
    parent_candidate_id: str | None = None,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return create_candidate(
        conn,
        {
            "candidate_id": f"candidate_peds_first_local_{run_id}_{suffix}",
            "parent_candidate_id": parent_candidate_id,
            "mutation_target": "theta",
            "phi_version": "phi_local_experiment_v1",
            "rho_version": "rho_local_experiment_v1",
            "theta_version": theta_version,
            "psi_version": "psi_local_experiment_v1",
            "delta_description": f"First local PEDS theta experiment {suffix}",
            "manifest_json": dict(manifest),
        },
    )


def _record_local_episode(
    conn,
    *,
    run_id: str,
    index: int,
    split: str,
    prompt: str,
    acceptance_tests: list[str],
    policy_tags: list[str],
) -> dict[str, Any]:
    episode_id = f"episode_peds_first_local_{run_id}_{split}_{index}"
    return record_episode(
        conn,
        {
            "episode_id": episode_id,
            "schema_version": "peds.episode.v1",
            "split": split,
            "timestamp": f"2026-06-04T00:{index:02d}:00Z",
            "task": {
                "task_id": f"task_{episode_id}",
                "title": f"Acceptance checklist {split} task {index}",
                "prompt": prompt,
                "acceptance_tests": acceptance_tests,
                "policy_tags": policy_tags,
            },
            "snapshot": {
                "repo_commit": "local-peds-first-experiment",
                "repo_tree_hash": "local_tree_hash",
                "retrieval_index_hash": "local_retrieval_hash",
                "env_image_hash": "local_env_hash",
                "tool_profile": "local_safe_rw_v1",
                "model_lock": {
                    "provider": "local",
                    "model": "deterministic_fixture",
                    "revision": "acceptance_checklist_v1",
                    "temperature": 0,
                    "top_p": 1,
                    "seed": 101,
                    "timeout_s": 120,
                },
            },
            "run": {
                "context_selected": [{"uri": "file://local_acceptance_checklist.md", "kind": "file", "tokens": 80}],
                "actions": [
                    {
                        "step": 1,
                        "kind": "inspect",
                        "args": {"acceptance_tests": acceptance_tests},
                        "exit_code": 0,
                    }
                ],
                "patch": "diff --git a/example.py b/example.py",
                "final_message": "Completed the task and mapped the answer to the acceptance checklist.",
            },
            "outcome": {"judge_verdict": "pass", "user_verdict": "accepted"},
            "metrics": {
                "tests_passed": 1.0,
                "tokens_in": 120,
                "tokens_out": 40,
                "latency_ms": 240,
                "tool_calls": 2,
                "failed_tool_calls": 0,
                "diff_lines": 4,
                "files_edited": 1,
                "hallucinated_claims": 0,
                "unrelated_edit_ratio": 0.0,
            },
            "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
        },
    )


def _assert_experiment_result(
    *,
    child_state: str,
    child: dict[str, Any],
    generated_candidate: dict[str, Any],
    generated_state: str,
    generated_replay_job: dict[str, Any],
    generated_replay_episode_count: int,
    generated_replay_result_count: int,
    training_result: dict[str, Any],
    adapter_row: dict[str, Any],
    expected_trained: bool,
    expected_adapter_status: str,
    network_attempts: list[Any],
    model_modules_loaded: list[str],
    fake_weights_written: bool,
) -> None:
    assert child_state == "distillation_candidate", child_state
    assert generated_candidate["parent_candidate_id"] == child["candidate_id"]
    assert generated_candidate["mutation_target"] == "theta"
    assert generated_state == "draft_candidate", generated_state
    assert generated_replay_job["candidate_id"] == generated_candidate["candidate_id"]
    assert generated_replay_job["split"] == "replay"
    assert generated_replay_episode_count > 0
    assert generated_replay_result_count == 0
    assert training_result["trained"] is expected_trained
    assert training_result["training_run"]["metrics"]["trained"] is expected_trained
    assert adapter_row["status"] == expected_adapter_status
    assert not network_attempts
    assert not model_modules_loaded
    assert fake_weights_written is False


def _contains_weight_payload(value: Any) -> bool:
    weight_keys = {"weight", "weights", "fake_weight", "fake_weights", "model_weights"}
    if isinstance(value, dict):
        return any(str(key).lower() in weight_keys or _contains_weight_payload(nested) for key, nested in value.items())
    if isinstance(value, list):
        return any(_contains_weight_payload(nested) for nested in value)
    return False


@contextmanager
def _deny_network_calls():
    attempts: list[Any] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def blocked_connect(sock, address):
        attempts.append({"api": "socket.connect", "address": repr(address)})
        raise RuntimeError(f"Network calls are disabled for this local PEDS experiment: {address!r}")

    def blocked_connect_ex(sock, address):
        attempts.append({"api": "socket.connect_ex", "address": repr(address)})
        raise RuntimeError(f"Network calls are disabled for this local PEDS experiment: {address!r}")

    def blocked_create_connection(address, timeout=None, source_address=None):
        attempts.append({"api": "socket.create_connection", "address": repr(address)})
        raise RuntimeError(f"Network calls are disabled for this local PEDS experiment: {address!r}")

    socket.socket.connect = blocked_connect
    socket.socket.connect_ex = blocked_connect_ex
    socket.create_connection = blocked_create_connection
    try:
        yield attempts
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create_connection


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the first local acceptance-checklist-first theta PEDS experiment."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Optional local brain root. Defaults to proof_transformer/temp/peds_first_local_experiment/<run_id>.",
    )
    parser.add_argument(
        "--trainer-mode",
        choices=["dry_run", "metadata_only", "local_json_trainer"],
        default="dry_run",
        help="Trainer mode for the PEDS training step. Default keeps the first experiment metadata-only.",
    )
    return parser.parse_args()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


if __name__ == "__main__":
    main()
