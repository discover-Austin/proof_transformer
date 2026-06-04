from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import BrainTestCase
from proof_transformer.peds.cycle import run_peds_cycle_until_blocked
from proof_transformer.peds.local_signals import (
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
    make_local_episode_shadow_event_provider,
)
from proof_transformer.peds.store import create_candidate, get_candidate_current_state, record_episode


class PEDSExperimentTests(BrainTestCase):
    def _candidate(
        self,
        suffix: str,
        *,
        parent_candidate_id: str | None = None,
        theta_version: str,
        mutation_target: str = "theta",
        manifest: dict[str, Any] | None = None,
    ) -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_peds_experiment_{suffix}",
                "parent_candidate_id": parent_candidate_id,
                "mutation_target": mutation_target,
                "phi_version": "phi_experiment_v1",
                "rho_version": "rho_experiment_v1",
                "theta_version": theta_version,
                "psi_version": "psi_experiment_v1",
                "delta_description": f"PEDS experiment candidate {suffix}",
                "manifest_json": manifest or {"suffix": suffix},
            },
        )

    def _episode(self, episode_id: str, *, split: str, prompt: str, acceptance_tests: list[str]) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
                "timestamp": "2026-06-04T00:00:00Z",
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Acceptance checklist task {episode_id}",
                    "prompt": prompt,
                    "acceptance_tests": acceptance_tests,
                    "policy_tags": ["peds_experiment", "acceptance_checklist"],
                },
                "snapshot": {
                    "repo_commit": "deterministic-experiment",
                    "repo_tree_hash": "tree_hash",
                    "retrieval_index_hash": "retrieval_hash",
                    "env_image_hash": "env_hash",
                    "tool_profile": "safe_rw_v1",
                    "model_lock": {
                        "provider": "local",
                        "model": "deterministic_fixture",
                        "revision": "acceptance_checklist_v1",
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 55,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [{"uri": "file://README.md", "kind": "file", "tokens": 180}],
                    "actions": [{"step": 1, "kind": "inspect", "args": {"acceptance_tests": acceptance_tests}, "exit_code": 0}],
                    "patch": "diff --git a/example.py b/example.py",
                    "final_message": "Mapped the requested behavior to every acceptance check.",
                },
                "outcome": {"judge_verdict": "pass", "user_verdict": "accepted"},
                "metrics": {
                    "tests_passed": 1.0,
                    "tokens_in": 180,
                    "tokens_out": 60,
                    "latency_ms": 280,
                    "tool_calls": 2,
                    "failed_tool_calls": 0,
                    "diff_lines": 6,
                    "files_edited": 1,
                    "hallucinated_claims": 0,
                    "unrelated_edit_ratio": 0.0,
                },
                "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
            },
        )

    def test_acceptance_checklist_theta_experiment_runs_through_peds_loop(self) -> None:
        incumbent = self._candidate(
            "incumbent",
            theta_version="theta_baseline_v1",
            manifest={
                "experiment_id": "acceptance_checklist_theta_v1",
                "role": "incumbent",
                "theta_strategy": "baseline_final_answer",
            },
        )
        candidate = self._candidate(
            "child",
            parent_candidate_id=incumbent["candidate_id"],
            theta_version="theta_acceptance_checklist_v1",
            manifest={
                "experiment_id": "acceptance_checklist_theta_v1",
                "role": "candidate",
                "hypothesis": "explicit acceptance-check mapping improves local task reliability",
                "theta_strategy": "acceptance_checklist_first",
            },
        )

        self.assertEqual(candidate["parent_candidate_id"], incumbent["candidate_id"])
        self.assertEqual(candidate["mutation_target"], "theta")
        self.assertEqual(candidate["phi_version"], incumbent["phi_version"])
        self.assertEqual(candidate["rho_version"], incumbent["rho_version"])
        self.assertNotEqual(candidate["theta_version"], incumbent["theta_version"])
        self.assertEqual(candidate["psi_version"], incumbent["psi_version"])

        for index in range(4):
            self._episode(
                f"episode_peds_experiment_replay_{index}",
                split="replay",
                prompt=f"Implement a tiny behavior and explain how acceptance check {index} is satisfied.",
                acceptance_tests=[f"pytest tests/acceptance_replay_{index}.py", "python -m compileall src"],
            )
        for index in range(3):
            self._episode(
                f"episode_peds_experiment_canary_{index}",
                split="canary",
                prompt=f"Handle a canary task without losing acceptance check coverage {index}.",
                acceptance_tests=[f"pytest tests/acceptance_canary_{index}.py"],
            )
        for index in range(10):
            self._episode(
                f"episode_peds_experiment_online_{index}",
                split="online",
                prompt=f"Observed local accepted task with acceptance checklist coverage {index}.",
                acceptance_tests=[f"pytest tests/shadow_acceptance_{index}.py"],
            )

        shadow_event_provider = make_local_episode_shadow_event_provider(
            split="online",
            limit=10,
            policy_tags=["peds_experiment"],
        )

        manifest_path = self.paths.root / "experiments" / "acceptance-checklist-manifest.json"
        artifact_dir = self.paths.root / "experiments" / "acceptance-checklist-artifacts"
        result = run_peds_cycle_until_blocked(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            max_steps=10,
            options={
                "promotion_reason": "local acceptance-checklist theta experiment",
                "distillation_reason": "accepted local experiment survivor selected for distillation",
                "trainer_name": "local_adapter_stub",
                "trainer_config": {"epochs": 1, "experiment_id": "acceptance_checklist_theta_v1"},
                "run_training": True,
                "trainer_mode": "dry_run",
                "adapter_name": "acceptance_checklist_theta_adapter",
                "output_path": str(manifest_path),
                "artifact_dir": str(artifact_dir),
            },
            replay_evaluator=acceptance_checklist_replay_evaluator,
            canary_evaluator=acceptance_checklist_canary_evaluator,
            shadow_event_provider=shadow_event_provider,
        )

        final_step = result["step_log"][-1]
        self.assertIn("candidate", final_step["changed"], result)
        generated_candidate = final_step["changed"]["candidate"]
        generated_replay_job = final_step["changed"]["replay_job"]
        training_result = final_step["changed"]["training_result"]

        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "distillation_candidate")
        self.assertIn("post-training candidate created; start a new cycle for generated candidate", result["stopped_reason"])
        self.assertEqual(generated_candidate["parent_candidate_id"], candidate["candidate_id"])
        self.assertEqual(generated_candidate["mutation_target"], "theta")
        self.assertEqual(get_candidate_current_state(self.conn, generated_candidate["candidate_id"]), "draft_candidate")

        decisions = [
            row["decision"]
            for row in self.conn.execute(
                "SELECT decision FROM promotion_decisions WHERE candidate_id = ? ORDER BY created_at, decision_id",
                (candidate["candidate_id"],),
            ).fetchall()
        ]
        self.assertEqual(
            decisions,
            ["replay_winner", "canary_passed", "shadow_started", "shadow_passed", "promoted", "distillation_candidate"],
        )

        self.assertTrue(manifest_path.exists())
        distillation_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(distillation_manifest["candidate_id"], candidate["candidate_id"])
        self.assertGreater(distillation_manifest["n_examples"], 0)

        training_run = training_result["training_run"]
        self.assertEqual(training_run["status"], "completed")
        self.assertEqual(training_run["metrics"]["mode"], "dry_run")
        self.assertFalse(training_run["metrics"]["trained"])
        self.assertFalse(training_result["trained"])

        artifact_path = Path(training_run["artifact_ref"])
        self.assertTrue(artifact_path.exists())
        artifact_manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
        self._assert_no_trained_true(training_result)
        self._assert_no_trained_true(artifact_manifest)
        self._assert_no_trained_true(generated_candidate["manifest"])

        adapter_row = self.conn.execute(
            "SELECT name, status, dataset_hash, path FROM adapter_registry WHERE id = ?",
            (generated_candidate["manifest"]["adapter_registry_id"],),
        ).fetchone()
        self.assertEqual(adapter_row["name"], "acceptance_checklist_theta_adapter")
        self.assertEqual(adapter_row["status"], "metadata_only")
        self.assertEqual(adapter_row["dataset_hash"], training_run["dataset_hash"])
        self.assertEqual(adapter_row["path"], str(artifact_path))

        self.assertEqual(generated_replay_job["candidate_id"], generated_candidate["candidate_id"])
        self.assertEqual(generated_replay_job["split"], "replay")
        self.assertGreater(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM replay_job_episodes WHERE job_id = ?",
                (generated_replay_job["job_id"],),
            ).fetchone()["count"],
            0,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
                (generated_replay_job["job_id"],),
            ).fetchone()["count"],
            0,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
                (generated_candidate["candidate_id"],),
            ).fetchone()["count"],
            0,
        )

    def _assert_no_trained_true(self, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key == "trained":
                    self.assertFalse(nested)
                self._assert_no_trained_true(nested)
        elif isinstance(value, list):
            for nested in value:
                self._assert_no_trained_true(nested)
