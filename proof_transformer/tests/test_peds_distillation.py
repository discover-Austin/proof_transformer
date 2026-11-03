from __future__ import annotations

import json
from pathlib import Path

from common import BrainTestCase
from proof_transformer.learning.adapter_registry import AdapterRegistry
from proof_transformer.peds.canary import create_canary_job, evaluate_canary_job
from proof_transformer.peds.distillation import (
    build_distillation_examples,
    collect_distillation_sources,
    validate_distillation_manifest,
    write_distillation_manifest,
)
from proof_transformer.peds.promotion import mark_distillation_candidate, mark_shadow_deployment, promote_candidate
from proof_transformer.peds.scoring import evaluate_replay_job
from proof_transformer.peds.shadow import complete_shadow_run, record_shadow_event, start_shadow_run
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    record_episode,
    record_promotion_decision,
    record_replay_result,
)


class PEDSDistillationTests(BrainTestCase):
    def _candidate(self, suffix: str, mutation_target: str = "phi", *, manifest: dict | None = None) -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_{suffix}",
                "mutation_target": mutation_target,
                "phi_version": f"phi_{suffix}",
                "rho_version": "rho_v1",
                "theta_version": None,
                "psi_version": "psi_v1",
                "delta_description": f"candidate {suffix}",
                "manifest": manifest or {"note": suffix},
            },
        )

    def _episode(self, episode_id: str, *, split: str) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Task {episode_id}",
                    "prompt": f"Solve {episode_id}",
                    "acceptance_tests": [f"pytest tests/{episode_id}.py"],
                    "policy_tags": ["no_secret_read"],
                },
                "snapshot": {
                    "repo_commit": "abc123",
                    "repo_tree_hash": "tree_hash",
                    "retrieval_index_hash": "retrieval_hash",
                    "env_image_hash": "env_hash",
                    "tool_profile": "safe_rw_v1",
                    "model_lock": {
                        "provider": "local",
                        "model": "tiny",
                        "revision": "r1",
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 31,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [{"uri": "file://app.py", "kind": "file", "tokens": 120}],
                    "actions": [{"step": 1, "kind": "read", "args": {"path": "app.py"}, "exit_code": 0}],
                    "patch": f"diff --git a/{episode_id}.py b/{episode_id}.py",
                    "final_message": f"Solved {episode_id}.",
                },
                "outcome": {"judge_verdict": "pass", "user_verdict": "accepted"},
                "metrics": {"tests_passed": 1.0},
                "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
            },
        )

    def _candidate_pair(self, suffix: str) -> tuple[dict, dict]:
        return self._candidate(f"{suffix}_candidate"), self._candidate(f"{suffix}_incumbent", mutation_target="rho")

    def _write_replay_winner(self, candidate: dict, incumbent: dict, suffix: str) -> None:
        episodes = [self._episode(f"replay_{suffix}_{index}", split="replay") for index in range(3)]
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [episode["episode_id"] for episode in episodes],
        )
        for episode in episodes:
            record_replay_result(
                self.conn,
                {
                    "job_id": job["job_id"],
                    "episode_id": episode["episode_id"],
                    "candidate_score": 0.90,
                    "incumbent_score": 0.70,
                    "judge_verdict": "pass",
                    "canary_failures": [],
                },
            )
        result = evaluate_replay_job(self.conn, job["job_id"], write_decision=True)
        self.assertEqual(result["gate"]["decision"], "replay_winner")

    def _write_canary_passed(self, candidate: dict, incumbent: dict, suffix: str) -> None:
        episodes = [self._episode(f"canary_{suffix}_{index}", split="canary") for index in range(3)]
        job_id = create_canary_job(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            episode_ids=[episode["episode_id"] for episode in episodes],
        )
        for episode in episodes:
            record_replay_result(
                self.conn,
                {
                    "job_id": job_id,
                    "episode_id": episode["episode_id"],
                    "candidate_score": 0.84,
                    "incumbent_score": 0.80,
                    "judge_verdict": "pass",
                    "canary_failures": [],
                },
            )
        result = evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(result["gate"]["decision"], "canary_passed")

    def _write_shadow_passed(self, candidate: dict, incumbent: dict, suffix: str) -> None:
        run = start_shadow_run(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
        )
        for index in range(10):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"prompt": f"shadow task {index}"},
                incumbent_output={"answer": f"incumbent {index}"},
                candidate_output={"answer": f"candidate accepted {index}"},
                accepted_by_incumbent=True,
                task_ref=f"task-{index}",
                metadata={"quality": "accepted"},
            )
        for index in range(2):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"prompt": f"shadow rejected {index}"},
                incumbent_output={"answer": f"incumbent rejected {index}"},
                candidate_output={"answer": f"candidate rejected {index}"},
                accepted_by_incumbent=False,
                metadata={"quality": "rejected"},
            )
        result = complete_shadow_run(self.conn, run["shadow_run_id"], write_decision=True)
        self.assertEqual(result["gate"]["decision"], "shadow_passed")

    def _survivor(self, suffix: str, *, state: str = "distillation_candidate") -> tuple[dict, dict]:
        candidate, incumbent = self._candidate_pair(suffix)
        self._write_replay_winner(candidate, incumbent, suffix)
        self._write_canary_passed(candidate, incumbent, suffix)
        self._write_shadow_passed(candidate, incumbent, suffix)
        if state in {"promoted", "shadow_deployment", "distillation_candidate"}:
            promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="explicit promotion")
        if state in {"shadow_deployment", "distillation_candidate"}:
            mark_shadow_deployment(
                self.conn,
                candidate["candidate_id"],
                operator="owner",
                reason="deployment-observation marker",
            )
        if state == "distillation_candidate":
            mark_distillation_candidate(
                self.conn,
                candidate["candidate_id"],
                operator="owner",
                reason="eligible for dataset construction",
            )
        return candidate, incumbent

    def _trained_adapter_survivor(self, suffix: str) -> tuple[dict, dict]:
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        artifact_path = Path(self.paths.root) / "distillation-adapters" / f"{suffix}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_type": "peds_adapter_artifact_manifest",
            "training_run_id": f"peds_training_run_{suffix}",
            "candidate_id": f"candidate_{suffix}_candidate",
            "dataset_hash": f"sha256:{suffix}",
            "trainer_name": "local_json_trainer",
            "mode": "local_json_trainer",
            "trained": True,
            "operator": "owner",
            "n_examples": 1,
            "created_at": "2026-06-09T00:00:00Z",
            "learned_parameters": {
                "feature_weights": {
                    "acceptance_token_overlap": 1.0,
                    "prompt_token_overlap": 1.0,
                    "shadow_acceptance_signal": 1.0,
                    "replay_win_signal": 1.0,
                },
                "source_type_weights": {"shadow_event": 1.0},
                "tag_weights": {"accepted_by_incumbent": 1.0},
                "top_input_tokens": [
                    {"token": "solve", "count": 3},
                    {"token": "pytest", "count": 3},
                ],
                "top_chosen_tokens": [{"token": "solved", "count": 3}],
            },
            "metrics": {
                "mode": "local_json_trainer",
                "trained": True,
                "n_examples": 1,
                "dataset_hash": f"sha256:{suffix}",
            },
            "provenance": {"trained": True, "metadata_only": False},
        }
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        adapter = AdapterRegistry(self.conn).register_adapter(
            name=f"distillation_adapter_{suffix}",
            version="v1",
            path=artifact_path,
            dataset_hash=artifact["dataset_hash"],
            status="trained",
        )
        candidate = self._candidate(
            f"{suffix}_candidate",
            manifest={
                "source": "peds_training_run",
                "proof_mode": True,
                "parent_candidate_id": incumbent["candidate_id"],
                "source_training_run_id": artifact["training_run_id"],
                "trainer_mode": "local_json_trainer",
                "trained": True,
                "training_trained": True,
                "artifact_ref": str(artifact_path),
                "adapter_registry_id": adapter["id"],
                "adapter_ref": {
                    "name": adapter["name"],
                    "version": adapter["version"],
                    "path": adapter["path"],
                    "status": adapter["status"],
                },
            },
        )
        self._write_replay_winner(candidate, incumbent, suffix)
        self._write_canary_passed(candidate, incumbent, suffix)
        self._write_shadow_passed(candidate, incumbent, suffix)
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="trained survivor promotion")
        mark_shadow_deployment(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="trained survivor deployment marker",
        )
        mark_distillation_candidate(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="trained survivor selected for distillation",
        )
        return candidate, incumbent

    def _manual_source_packet(self) -> dict:
        return {
            "candidate_id": "candidate_manual",
            "current_state": "distillation_candidate",
            "decision_timeline": [],
            "replay_jobs": [
                {
                    "job_id": "job_manual_replay",
                    "split": "replay",
                    "created_at": "2026-06-02T00:00:00Z",
                    "evaluator_version": "eval_v1",
                    "results": [
                        self._manual_replay_result("result_win", 0, 0.90, 0.70, "pass", []),
                        self._manual_replay_result("result_loss", 1, 0.60, 0.70, "pass", []),
                        self._manual_replay_result("result_canary_failure", 2, 0.90, 0.70, "pass", ["secret_read"]),
                        self._manual_replay_result("result_unsafe", 3, 0.90, 0.70, "unsafe", []),
                    ],
                }
            ],
            "canary_jobs": [
                {
                    "job_id": "job_manual_canary",
                    "split": "canary",
                    "created_at": "2026-06-02T00:01:00Z",
                    "results": [self._manual_replay_result("result_canary_unsafe", 0, 0.9, 0.8, "unsafe", [])],
                }
            ],
            "shadow_runs": [],
            "warnings": [],
        }

    def _manual_replay_result(
        self,
        result_id: str,
        episode_order: int,
        candidate_score: float,
        incumbent_score: float,
        judge_verdict: str,
        canary_failures: list[str],
    ) -> dict:
        return {
            "result_id": result_id,
            "episode_id": f"episode_{result_id}",
            "episode_order": episode_order,
            "candidate_score": candidate_score,
            "incumbent_score": incumbent_score,
            "delta": candidate_score - incumbent_score,
            "judge_verdict": judge_verdict,
            "canary_failures": canary_failures,
            "audit_log_ref": f"audit://{result_id}",
            "created_at": f"2026-06-02T00:00:0{episode_order}Z",
            "task_prompt": f"Solve {result_id}",
            "task_title": f"Task {result_id}",
            "acceptance_tests": ["pytest"],
            "policy_tags": ["no_secret_read"],
            "raw_episode": {
                "task": {"task_id": f"task_{result_id}", "prompt": f"Solve {result_id}"},
                "run": {"final_message": f"Solved {result_id}.", "patch": f"diff {result_id}"},
            },
        }

    def test_collect_distillation_sources_rejects_missing_candidate(self) -> None:
        with self.assertRaises(ValueError):
            collect_distillation_sources(self.conn, "missing_candidate")

    def test_collect_distillation_sources_rejects_candidate_without_ready_state(self) -> None:
        candidate = self._candidate("not_ready")
        with self.assertRaises(ValueError):
            collect_distillation_sources(self.conn, candidate["candidate_id"])

    def test_collect_distillation_sources_collects_replay_canary_and_shadow_evidence(self) -> None:
        candidate, _ = self._survivor("collect")
        packet = collect_distillation_sources(self.conn, candidate["candidate_id"])

        self.assertEqual(packet["current_state"], "distillation_candidate")
        self.assertEqual(len(packet["replay_jobs"]), 1)
        self.assertEqual(len(packet["canary_jobs"]), 1)
        self.assertEqual(len(packet["shadow_runs"]), 1)
        self.assertEqual(len(packet["shadow_runs"][0]["events"]), 12)
        self.assertIsNotNone(packet["distillation_candidate_decision"])

    def test_collect_distillation_sources_rejects_shadow_failed_after_shadow_pass(self) -> None:
        candidate, incumbent = self._survivor("invalidating_shadow_failed")
        shadow_run = self.conn.execute(
            "SELECT shadow_run_id FROM shadow_runs WHERE candidate_id = ? ORDER BY started_at DESC LIMIT 1",
            (candidate["candidate_id"],),
        ).fetchone()
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "shadow_run_id": shadow_run["shadow_run_id"],
                "decision": "shadow_failed",
                "reason": "later shadow failure invalidates survivor evidence",
            },
        )
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "shadow_run_id": shadow_run["shadow_run_id"],
                "decision": "distillation_candidate",
                "reason": "corrupt fixture attempted to re-mark invalidated evidence",
            },
        )

        with self.assertRaises(ValueError) as raised:
            collect_distillation_sources(self.conn, candidate["candidate_id"])

        self.assertIn("invalidating post-shadow evidence", str(raised.exception))

    def test_trained_adapter_survivor_manifest_includes_adapter_replay_proof(self) -> None:
        candidate, _ = self._trained_adapter_survivor("trained_survivor")

        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"])
        adapter_summary = manifest["source_summary"]["adapter_survivor"]
        validation = validate_distillation_manifest(manifest)

        self.assertTrue(validation["valid"], validation["reason"])
        self.assertEqual(manifest["source_summary"]["candidate_source"], "peds_training_run")
        self.assertEqual(adapter_summary["source"], "peds_training_run")
        self.assertTrue(adapter_summary["proof_mode"])
        self.assertTrue(adapter_summary["training_trained"])
        self.assertEqual(adapter_summary["trainer_mode"], "local_json_trainer")
        self.assertEqual(adapter_summary["adapter_registry_status"], "trained")
        self.assertGreater(adapter_summary["adapter_loaded_replay_jobs"], 0)
        self.assertGreater(adapter_summary["behavior_delta_count"], 0)

    def test_validate_distillation_manifest_rejects_trained_adapter_summary_without_loaded_replay_proof(self) -> None:
        candidate, _ = self._trained_adapter_survivor("trained_survivor_missing_proof")
        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"])
        adapter_summary = manifest["source_summary"]["adapter_survivor"]
        adapter_summary["adapter_loaded_replay_jobs"] = 0
        adapter_summary["behavior_delta_count"] = 0

        result = validate_distillation_manifest(manifest)

        self.assertFalse(result["valid"])
        self.assertIn("adapter_loaded replay proof", result["reason"])
        self.assertIn("behavior-dependent replay delta proof", result["reason"])

    def test_build_distillation_examples_builds_shadow_examples_from_accepted_events(self) -> None:
        candidate, _ = self._survivor("shadow_examples")
        packet = collect_distillation_sources(self.conn, candidate["candidate_id"])
        examples = build_distillation_examples(packet)
        shadow_examples = [example for example in examples if example["source_type"] == "shadow_event"]

        self.assertEqual(len(shadow_examples), 10)
        self.assertEqual(shadow_examples[0]["chosen"], {"answer": "candidate accepted 0"})
        self.assertEqual(shadow_examples[0]["tags"], ["peds", "shadow_event", "accepted_by_incumbent"])

    def test_build_distillation_examples_excludes_unaccepted_shadow_events(self) -> None:
        candidate, _ = self._survivor("shadow_exclude")
        packet = collect_distillation_sources(self.conn, candidate["candidate_id"])
        examples = build_distillation_examples(packet)

        chosen_answers = [example["chosen"].get("answer") for example in examples if example["source_type"] == "shadow_event"]
        self.assertNotIn("candidate rejected 0", chosen_answers)
        self.assertNotIn("candidate rejected 1", chosen_answers)

    def test_build_distillation_examples_excludes_unsafe_replay_and_canary_evidence(self) -> None:
        examples = build_distillation_examples(self._manual_source_packet())

        self.assertEqual([example["source_ref"] for example in examples], ["result_win"])
        self.assertEqual({example["source_type"] for example in examples}, {"replay_result"})

    def test_build_distillation_examples_includes_only_replay_wins_without_canary_failures(self) -> None:
        examples = build_distillation_examples(self._manual_source_packet())

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["source_ref"], "result_win")
        self.assertEqual(examples[0]["chosen"]["final_message"], "Solved result_win.")

    def test_build_distillation_examples_uses_deterministic_ordering(self) -> None:
        candidate, _ = self._survivor("ordering")
        packet = collect_distillation_sources(self.conn, candidate["candidate_id"])

        first = build_distillation_examples(packet)
        second = build_distillation_examples(packet)

        self.assertEqual(first, second)
        self.assertEqual([example["source_type"] for example in first[:10]], ["shadow_event"] * 10)
        self.assertEqual([example["source_type"] for example in first[10:]], ["replay_result"] * 3)

    def test_write_distillation_manifest_hash_is_stable_across_repeated_builds(self) -> None:
        candidate, _ = self._survivor("stable_hash")

        first = write_distillation_manifest(self.conn, candidate["candidate_id"])
        second = write_distillation_manifest(self.conn, candidate["candidate_id"])

        self.assertEqual(first["dataset_hash"], second["dataset_hash"])
        self.assertEqual(first["n_examples"], second["n_examples"])

    def test_write_distillation_manifest_with_output_path_none_writes_nothing(self) -> None:
        candidate, _ = self._survivor("no_write")
        target = Path(self.tempdir.name) / "manifest.json"

        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"], output_path=None)

        self.assertGreater(manifest["n_examples"], 0)
        self.assertFalse(target.exists())

    def test_write_distillation_manifest_writes_json_manifest(self) -> None:
        candidate, _ = self._survivor("write_file")
        target = Path(self.tempdir.name) / "distillation" / "manifest.json"

        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"], output_path=str(target))

        self.assertTrue(target.exists())
        stored = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(stored["dataset_hash"], manifest["dataset_hash"])
        self.assertEqual(stored["candidate_id"], candidate["candidate_id"])

    def test_validate_distillation_manifest_rejects_empty_dataset(self) -> None:
        manifest = {
            "candidate_id": "candidate_empty",
            "created_at": "2026-06-02T00:00:00Z",
            "dataset_hash": "sha256:" + "0" * 64,
            "n_examples": 0,
            "examples": [],
            "source_summary": {},
            "warnings": [],
        }

        result = validate_distillation_manifest(manifest)

        self.assertFalse(result["valid"])
        self.assertIn("n_examples must be > 0", result["reason"])

    def test_validate_distillation_manifest_rejects_missing_provenance(self) -> None:
        candidate, _ = self._survivor("missing_provenance")
        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"], max_examples=1)
        manifest["examples"][0].pop("provenance")

        result = validate_distillation_manifest(manifest)

        self.assertFalse(result["valid"])
        self.assertIn("missing keys: provenance", result["reason"])

    def test_validate_distillation_manifest_passes_clean_manifest(self) -> None:
        candidate, _ = self._survivor("valid_manifest")
        manifest = write_distillation_manifest(self.conn, candidate["candidate_id"])

        result = validate_distillation_manifest(manifest)

        self.assertTrue(result["valid"])
        self.assertEqual(result["failed_checks"], [])
