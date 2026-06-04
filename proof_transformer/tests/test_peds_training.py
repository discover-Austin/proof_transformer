from __future__ import annotations

import hashlib
import json
from pathlib import Path

from common import BrainTestCase
from proof_transformer.peds.store import create_candidate
from proof_transformer.peds.training import (
    get_training_run,
    prepare_training_run,
    record_training_result,
    validate_training_readiness,
)


class PEDSTrainingTests(BrainTestCase):
    def _candidate(self, suffix: str = "training") -> dict:
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

    def _example(self, candidate_id: str, *, unsafe: bool = False) -> dict:
        tags = ["peds", "shadow_event", "accepted_by_incumbent"]
        if unsafe:
            tags.append("unsafe")
        return {
            "example_id": "distill_example_000000_test",
            "candidate_id": candidate_id,
            "source_type": "shadow_event",
            "source_ref": "shadow_event_test",
            "input": {"prompt": "Explain the patch."},
            "chosen": {"answer": "The patch fixes the deterministic replay boundary."},
            "tags": tags,
            "provenance": {
                "source_table": "shadow_events",
                "shadow_run_id": "shadow_run_test",
                "shadow_event_id": "shadow_event_test",
                "ordinal": 0,
            },
            "safety_exclusions": [],
            "metadata": {"judge_verdict": "pass"},
        }

    def _manifest(self, candidate_id: str, *, unsafe: bool = False, empty: bool = False) -> dict:
        examples = [] if empty else [self._example(candidate_id, unsafe=unsafe)]
        return {
            "candidate_id": candidate_id,
            "created_at": "2026-06-02T00:00:00Z",
            "dataset_hash": self._dataset_hash(examples),
            "n_examples": len(examples),
            "examples": examples,
            "source_summary": {
                "candidate_id": candidate_id,
                "current_state": "distillation_candidate",
                "n_shadow_events": len(examples),
            },
            "warnings": [],
        }

    def _dataset_hash(self, examples: list[dict]) -> str:
        payload = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_prepare_training_run_rejects_invalid_manifest(self) -> None:
        candidate = self._candidate("invalid_manifest")
        manifest = self._manifest(candidate["candidate_id"], empty=True)

        with self.assertRaises(ValueError):
            prepare_training_run(self.conn, manifest, trainer_name="local_adapter_stub")

    def test_prepare_training_run_rejects_empty_trainer_name(self) -> None:
        candidate = self._candidate("empty_trainer")
        manifest = self._manifest(candidate["candidate_id"])

        with self.assertRaises(ValueError):
            prepare_training_run(self.conn, manifest, trainer_name=" ")

    def test_validate_training_readiness_accepts_clean_manifest(self) -> None:
        candidate = self._candidate("ready")
        manifest = self._manifest(candidate["candidate_id"])

        readiness = validate_training_readiness(self.conn, manifest, {"epochs": 1, "output_dir": "adapters/peds"})

        self.assertTrue(readiness["ready"])
        self.assertEqual(readiness["failed_checks"], [])
        self.assertEqual(readiness["manifest_summary"]["dataset_hash"], manifest["dataset_hash"])

    def test_validate_training_readiness_rejects_manifest_with_unsafe_example(self) -> None:
        candidate = self._candidate("unsafe")
        manifest = self._manifest(candidate["candidate_id"], unsafe=True)

        readiness = validate_training_readiness(self.conn, manifest, {"epochs": 1})

        self.assertFalse(readiness["ready"])
        self.assertIn("unsafe", readiness["reason"])

    def test_prepare_training_run_records_prepared_run(self) -> None:
        candidate = self._candidate("records")
        manifest = self._manifest(candidate["candidate_id"])

        run = prepare_training_run(
            self.conn,
            manifest,
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 1},
            output_dir="adapters/peds/run-records",
        )

        self.assertEqual(run["status"], "prepared")
        self.assertEqual(run["candidate_id"], candidate["candidate_id"])
        self.assertEqual(run["dataset_hash"], manifest["dataset_hash"])
        self.assertEqual(run["trainer_config"]["epochs"], 1)
        self.assertEqual(run["trainer_config"]["output_dir"], "adapters/peds/run-records")

    def test_prepare_training_run_does_not_train_or_write_artifacts(self) -> None:
        candidate = self._candidate("no_artifacts")
        manifest = self._manifest(candidate["candidate_id"])
        output_dir = self.paths.root / "peds-training-output"

        run = prepare_training_run(
            self.conn,
            manifest,
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 1},
            output_dir=str(output_dir),
        )

        self.assertEqual(run["status"], "prepared")
        self.assertIsNone(run["artifact_ref"])
        self.assertFalse(output_dir.exists())
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS count FROM adapter_registry").fetchone()["count"], 0)

    def test_get_training_run_decodes_json_fields(self) -> None:
        candidate = self._candidate("decode")
        manifest = self._manifest(candidate["candidate_id"])
        prepared = prepare_training_run(
            self.conn,
            manifest,
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 2, "lr": 0.001},
        )

        loaded = get_training_run(self.conn, prepared["training_run_id"])

        self.assertEqual(loaded["manifest"]["dataset_hash"], manifest["dataset_hash"])
        self.assertEqual(loaded["manifest_summary"]["n_examples"], 1)
        self.assertEqual(loaded["trainer_config"]["epochs"], 2)
        self.assertEqual(loaded["metrics"], {})

    def test_record_training_result_completed_stores_metrics_and_artifact_ref(self) -> None:
        candidate = self._candidate("completed")
        manifest = self._manifest(candidate["candidate_id"])
        run = prepare_training_run(self.conn, manifest, trainer_name="local_adapter_stub")

        updated = record_training_result(
            self.conn,
            run["training_run_id"],
            status="completed",
            metrics={"loss": 0.12, "examples": 1},
            artifact_ref="artifact://local/adapter-test",
        )

        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["metrics"]["loss"], 0.12)
        self.assertEqual(updated["artifact_ref"], "artifact://local/adapter-test")
        self.assertIsNone(updated["error"])

    def test_record_training_result_failed_stores_error(self) -> None:
        candidate = self._candidate("failed")
        manifest = self._manifest(candidate["candidate_id"])
        run = prepare_training_run(self.conn, manifest, trainer_name="local_adapter_stub")

        updated = record_training_result(
            self.conn,
            run["training_run_id"],
            status="failed",
            metrics={"attempted": 1},
            error="trainer not implemented",
        )

        self.assertEqual(updated["status"], "failed")
        self.assertEqual(updated["metrics"], {"attempted": 1})
        self.assertEqual(updated["error"], "trainer not implemented")

    def test_record_training_result_rejects_unknown_status(self) -> None:
        candidate = self._candidate("bad_status")
        manifest = self._manifest(candidate["candidate_id"])
        run = prepare_training_run(self.conn, manifest, trainer_name="local_adapter_stub")

        with self.assertRaises(ValueError):
            record_training_result(self.conn, run["training_run_id"], status="running")

    def test_training_layer_does_not_write_promotion_decisions(self) -> None:
        candidate = self._candidate("no_lifecycle")
        manifest = self._manifest(candidate["candidate_id"])
        before = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        run = prepare_training_run(self.conn, manifest, trainer_name="local_adapter_stub")
        record_training_result(self.conn, run["training_run_id"], status="aborted", error="operator stopped")

        after = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]
        self.assertEqual(after, before)

    def test_validate_training_readiness_rejects_unsafe_absolute_output_dir(self) -> None:
        candidate = self._candidate("unsafe_output")
        manifest = self._manifest(candidate["candidate_id"])
        unsafe_path = Path(self.tempdir.name).parent / "outside-brain-output"

        readiness = validate_training_readiness(self.conn, manifest, {"output_dir": str(unsafe_path)})

        self.assertFalse(readiness["ready"])
        self.assertIn("output_dir must be relative", readiness["reason"])
