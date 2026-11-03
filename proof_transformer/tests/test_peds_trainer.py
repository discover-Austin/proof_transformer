from __future__ import annotations

import hashlib
import json
import socket
import sys
from pathlib import Path

from common import BrainTestCase
from proof_transformer.peds.store import create_candidate
from proof_transformer.peds.trainer import (
    build_adapter_artifact_manifest,
    run_prepared_training_run,
    validate_trainer_mode,
)
from proof_transformer.peds.training import get_training_run, prepare_training_run


class PEDSTrainerTests(BrainTestCase):
    def _candidate(self, suffix: str = "trainer") -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_{suffix}",
                "mutation_target": "theta",
                "phi_version": "phi_v1",
                "rho_version": "rho_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_v1",
                "delta_description": f"candidate {suffix}",
                "manifest": {"note": suffix},
            },
        )

    def _example(
        self,
        candidate_id: str,
        *,
        suffix: str = "trainer",
        source_type: str = "shadow_event",
        prompt: str = "Explain the patch.",
        chosen: str = "The patch preserves the audit boundary.",
        acceptance_tests: list[str] | None = None,
    ) -> dict:
        tags = ["peds", source_type]
        provenance = {"source_table": "shadow_events"}
        if source_type == "shadow_event":
            tags.append("accepted_by_incumbent")
            provenance.update(
                {
                    "shadow_run_id": f"shadow_run_{suffix}",
                    "shadow_event_id": f"shadow_event_{suffix}",
                    "ordinal": 0,
                    "accepted_by_incumbent": True,
                }
            )
        elif source_type == "replay_result":
            tags.append("replay_win")
            provenance.update(
                {
                    "job_id": f"replay_job_{suffix}",
                    "result_id": f"replay_result_{suffix}",
                    "episode_id": f"episode_{suffix}",
                    "episode_order": 0,
                    "split": "replay",
                    "candidate_score": 0.91,
                    "incumbent_score": 0.72,
                    "delta": 0.19,
                }
            )
        return {
            "example_id": f"distill_example_000000_{suffix}",
            "candidate_id": candidate_id,
            "source_type": source_type,
            "source_ref": f"{source_type}_{suffix}",
            "input": {
                "prompt": prompt,
                "acceptance_tests": acceptance_tests or ["python -m unittest tests.test_acceptance"],
            },
            "chosen": {"answer": chosen},
            "tags": tags,
            "provenance": provenance,
            "safety_exclusions": [],
            "metadata": {"judge_verdict": "pass"},
        }

    def _manifest(self, candidate_id: str, *, examples: list[dict] | None = None) -> dict:
        examples = examples if examples is not None else [self._example(candidate_id)]
        return {
            "candidate_id": candidate_id,
            "created_at": "2026-06-02T00:00:00Z",
            "dataset_hash": self._dataset_hash(examples),
            "n_examples": len(examples),
            "examples": examples,
            "source_summary": {"candidate_id": candidate_id, "current_state": "distillation_candidate"},
            "warnings": [],
        }

    def _dataset_hash(self, examples: list[dict]) -> str:
        payload = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _prepared_run(self, suffix: str = "prepared") -> dict:
        candidate = self._candidate(suffix)
        return prepare_training_run(
            self.conn,
            self._manifest(candidate["candidate_id"]),
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 1},
        )

    def _prepared_run_with_examples(self, suffix: str, examples: list[dict]) -> dict:
        candidate = self._candidate(suffix)
        return prepare_training_run(
            self.conn,
            self._manifest(candidate["candidate_id"], examples=examples),
            trainer_name="local_json_trainer",
            trainer_config={"epochs": 1},
        )

    def _replace_training_manifest(self, training_run_id: str, manifest: dict) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE peds_training_runs SET manifest_json = ? WHERE training_run_id = ?",
                (
                    json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    training_run_id,
                ),
            )

    def test_run_prepared_training_run_rejects_missing_training_run_id(self) -> None:
        with self.assertRaises(ValueError):
            run_prepared_training_run(self.conn, "missing_training_run", operator="owner")

    def test_run_prepared_training_run_rejects_empty_operator(self) -> None:
        run = self._prepared_run("empty_operator")
        with self.assertRaises(ValueError):
            run_prepared_training_run(self.conn, run["training_run_id"], operator=" ")

    def test_validate_trainer_mode_rejects_invalid_mode(self) -> None:
        result = validate_trainer_mode("not_a_mode")
        self.assertFalse(result["valid"])
        self.assertIn("unsupported trainer mode", result["reason"])

    def test_validate_trainer_mode_rejects_local_trainer_without_abstraction(self) -> None:
        result = validate_trainer_mode("local_trainer")
        self.assertFalse(result["valid"])
        self.assertIn("no local adapter trainer abstraction", result["reason"])

    def test_validate_trainer_mode_accepts_local_json_trainer(self) -> None:
        result = validate_trainer_mode("local_json_trainer")
        self.assertTrue(result["valid"])
        self.assertIn("deterministic local trainer", result["reason"])

    def test_run_prepared_training_run_rejects_already_completed_training_run(self) -> None:
        run = self._prepared_run("already_completed")
        run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="dry_run")

        with self.assertRaises(ValueError):
            run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="dry_run")

    def test_dry_run_completes_prepared_run(self) -> None:
        run = self._prepared_run("dry_run")
        result = run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="dry_run")

        self.assertFalse(result["trained"])
        self.assertEqual(result["mode"], "dry_run")
        self.assertEqual(result["training_run"]["status"], "completed")
        self.assertEqual(result["training_run"]["metrics"]["trained"], False)

    def test_metadata_only_completes_prepared_run(self) -> None:
        run = self._prepared_run("metadata_only")
        result = run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="metadata_only")

        self.assertFalse(result["trained"])
        self.assertEqual(result["mode"], "metadata_only")
        self.assertEqual(result["training_run"]["status"], "completed")
        self.assertEqual(result["training_run"]["metrics"]["mode"], "metadata_only")

    def test_local_json_trainer_rejects_invalid_manifest(self) -> None:
        run = self._prepared_run("local_invalid_manifest")
        bad_manifest = dict(run["manifest"])
        bad_manifest["dataset_hash"] = "sha256:not-the-example-hash"
        self._replace_training_manifest(run["training_run_id"], bad_manifest)

        with self.assertRaises(ValueError) as raised:
            run_prepared_training_run(
                self.conn,
                run["training_run_id"],
                operator="owner",
                mode="local_json_trainer",
                artifact_dir=str(self.paths.root / "peds" / "invalid-artifact"),
            )

        self.assertIn("dataset_hash does not match", str(raised.exception))

    def test_local_json_trainer_rejects_empty_examples(self) -> None:
        run = self._prepared_run("local_empty_examples")
        empty_manifest = dict(run["manifest"])
        empty_manifest["examples"] = []
        empty_manifest["n_examples"] = 0
        empty_manifest["dataset_hash"] = self._dataset_hash([])
        self._replace_training_manifest(run["training_run_id"], empty_manifest)

        with self.assertRaises(ValueError) as raised:
            run_prepared_training_run(
                self.conn,
                run["training_run_id"],
                operator="owner",
                mode="local_json_trainer",
                artifact_dir=str(self.paths.root / "peds" / "empty-artifact"),
            )

        self.assertIn("n_examples must be > 0", str(raised.exception))

    def test_local_json_trainer_requires_artifact_dir(self) -> None:
        run = self._prepared_run("local_requires_artifact")

        with self.assertRaises(ValueError) as raised:
            run_prepared_training_run(
                self.conn,
                run["training_run_id"],
                operator="owner",
                mode="local_json_trainer",
            )

        self.assertIn("artifact_dir is required", str(raised.exception))

    def test_local_json_trainer_writes_trained_artifact_and_metrics(self) -> None:
        run = self._prepared_run("local_trained_artifact")
        artifact_dir = Path(self.paths.root) / "peds" / "local-json-artifacts"

        result = run_prepared_training_run(
            self.conn,
            run["training_run_id"],
            operator="owner",
            mode="local_json_trainer",
            artifact_dir=str(artifact_dir),
        )

        artifact_path = Path(result["artifact_ref"])
        stored = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertTrue(artifact_path.exists())
        self.assertTrue(result["trained"])
        self.assertTrue(result["training_run"]["metrics"]["trained"])
        self.assertTrue(stored["trained"])
        self.assertTrue(stored["metrics"]["trained"])
        self.assertEqual(stored["trainer_name"], "local_adapter_stub")
        self.assertEqual(stored["dataset_hash"], run["dataset_hash"])
        self.assertIn("learned_parameters", stored)
        self.assertGreater(stored["metrics"]["learned_parameter_count"], 0)
        self.assertEqual(stored["metrics"]["mode"], "local_json_trainer")

    def test_local_json_trainer_learned_parameters_depend_on_examples(self) -> None:
        candidate_one = self._candidate("local_data_one")
        run_one = prepare_training_run(
            self.conn,
            self._manifest(
                candidate_one["candidate_id"],
                examples=[
                    self._example(
                        candidate_one["candidate_id"],
                        suffix="data_one",
                        chosen="The patch preserves audit evidence and acceptance mapping.",
                        acceptance_tests=["python -m unittest tests.test_audit"],
                    )
                ],
            ),
            trainer_name="local_json_trainer",
            trainer_config={"epochs": 1},
        )
        candidate_two = self._candidate("local_data_two")
        run_two = prepare_training_run(
            self.conn,
            self._manifest(
                candidate_two["candidate_id"],
                examples=[
                    self._example(
                        candidate_two["candidate_id"],
                        suffix="data_two",
                        source_type="replay_result",
                        prompt="Keep compile checks visible.",
                        chosen="Compile checks and replay wins should dominate this candidate.",
                        acceptance_tests=["python -m compileall src", "python -m unittest tests.test_replay"],
                    )
                ],
            ),
            trainer_name="local_json_trainer",
            trainer_config={"epochs": 1},
        )

        result_one = run_prepared_training_run(
            self.conn,
            run_one["training_run_id"],
            operator="owner",
            mode="local_json_trainer",
            artifact_dir=str(self.paths.root / "peds" / "local-data-one"),
        )
        result_two = run_prepared_training_run(
            self.conn,
            run_two["training_run_id"],
            operator="owner",
            mode="local_json_trainer",
            artifact_dir=str(self.paths.root / "peds" / "local-data-two"),
        )

        self.assertNotEqual(
            result_one["artifact_manifest"]["learned_parameters"],
            result_two["artifact_manifest"]["learned_parameters"],
        )

    def test_local_json_trainer_ignores_chosen_metadata_when_learning_tokens(self) -> None:
        candidate = self._candidate("local_metadata_noise")
        example = self._example(
            candidate["candidate_id"],
            suffix="metadata_noise",
            chosen="Current signal should survive metadata noise.",
            acceptance_tests=["current_signal acceptance"],
        )
        example["chosen"] = {
            "acceptance_checklist": ["current_signal current_acceptance"],
            "final_message": "Current signal should survive metadata noise.",
            "metadata": {
                "adapter_behavior_profile": {
                    "learned_parameters": {
                        "top_chosen_tokens": [
                            {"token": "previous_signal", "count": 999},
                            {"token": "285714", "count": 999},
                        ],
                        "top_input_tokens": [
                            {"token": "stale_parent_token", "count": 999},
                        ],
                    }
                }
            },
        }
        run = prepare_training_run(
            self.conn,
            self._manifest(candidate["candidate_id"], examples=[example]),
            trainer_name="local_json_trainer",
            trainer_config={"epochs": 1},
        )

        result = run_prepared_training_run(
            self.conn,
            run["training_run_id"],
            operator="owner",
            mode="local_json_trainer",
            artifact_dir=str(self.paths.root / "peds" / "local-metadata-noise"),
        )

        chosen_tokens = {
            item["token"]
            for item in result["artifact_manifest"]["learned_parameters"]["top_chosen_tokens"]
        }
        self.assertIn("current_signal", chosen_tokens)
        self.assertNotIn("previous_signal", chosen_tokens)
        self.assertNotIn("stale_parent_token", chosen_tokens)
        self.assertNotIn("285714", chosen_tokens)

    def test_local_json_trainer_does_not_import_model_modules_or_call_network(self) -> None:
        run = self._prepared_run("local_no_network")
        before_model_modules = {name for name in sys.modules if name.startswith("proof_transformer.model")}
        attempts = []
        original_connect = socket.socket.connect
        original_create_connection = socket.create_connection

        def blocked_connect(sock, address):
            attempts.append(("connect", repr(address)))
            raise AssertionError("local_json_trainer attempted a network socket call")

        def blocked_create_connection(address, timeout=None, source_address=None):
            attempts.append(("create_connection", repr(address)))
            raise AssertionError("local_json_trainer attempted a network socket call")

        socket.socket.connect = blocked_connect
        socket.create_connection = blocked_create_connection
        try:
            run_prepared_training_run(
                self.conn,
                run["training_run_id"],
                operator="owner",
                mode="local_json_trainer",
                artifact_dir=str(self.paths.root / "peds" / "local-no-network"),
            )
        finally:
            socket.socket.connect = original_connect
            socket.create_connection = original_create_connection

        after_model_modules = {name for name in sys.modules if name.startswith("proof_transformer.model")}
        self.assertEqual(attempts, [])
        self.assertEqual(after_model_modules - before_model_modules, set())

    def test_dry_run_writes_artifact_manifest_when_artifact_dir_provided(self) -> None:
        run = self._prepared_run("artifact_write")
        artifact_dir = Path(self.paths.root) / "peds" / "trainer-artifacts"

        result = run_prepared_training_run(
            self.conn,
            run["training_run_id"],
            operator="owner",
            mode="dry_run",
            artifact_dir=str(artifact_dir),
        )

        artifact_path = Path(result["artifact_ref"])
        self.assertTrue(artifact_path.exists())
        stored = json.loads(artifact_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["artifact_type"], "peds_adapter_artifact_manifest")
        self.assertEqual(stored["training_run_id"], run["training_run_id"])

    def test_artifact_manifest_clearly_has_trained_false(self) -> None:
        run = self._prepared_run("manifest_false")
        manifest = build_adapter_artifact_manifest(run, operator="owner", mode="dry_run")

        self.assertFalse(manifest["trained"])
        self.assertFalse(manifest["metrics"]["trained"])
        self.assertFalse(manifest["provenance"]["trained"])
        self.assertTrue(manifest["provenance"]["metadata_only"])

    def test_training_runner_does_not_write_promotion_decisions(self) -> None:
        run = self._prepared_run("no_promotions")
        before = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="dry_run")

        after = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]
        self.assertEqual(after, before)

    def test_get_training_run_sees_completed_status_and_metrics(self) -> None:
        run = self._prepared_run("completed_metrics")
        result = run_prepared_training_run(self.conn, run["training_run_id"], operator="owner", mode="dry_run")

        loaded = get_training_run(self.conn, run["training_run_id"])

        self.assertEqual(loaded["status"], "completed")
        self.assertEqual(loaded["metrics"]["mode"], "dry_run")
        self.assertEqual(loaded["metrics"]["dataset_hash"], result["artifact_manifest"]["dataset_hash"])
        self.assertIsNone(loaded["error"])
