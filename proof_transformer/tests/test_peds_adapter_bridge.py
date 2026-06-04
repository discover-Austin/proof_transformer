from __future__ import annotations

import hashlib
import json

from common import BrainTestCase
from proof_transformer.peds.adapter_bridge import (
    create_candidate_from_training_run,
    register_training_artifact,
    validate_training_artifact_for_candidate,
)
from proof_transformer.peds.store import create_candidate, get_candidate_current_state
from proof_transformer.peds.trainer import run_prepared_training_run
from proof_transformer.peds.training import prepare_training_run


class PEDSAdapterBridgeTests(BrainTestCase):
    def _candidate(self, suffix: str) -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_bridge_{suffix}",
                "mutation_target": "theta",
                "phi_version": "phi_v1",
                "rho_version": "rho_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_v1",
                "delta_description": f"bridge parent {suffix}",
                "manifest": {"suffix": suffix},
            },
        )

    def _example(self, candidate_id: str, suffix: str) -> dict:
        return {
            "example_id": f"distill_example_bridge_{suffix}",
            "candidate_id": candidate_id,
            "source_type": "shadow_event",
            "source_ref": f"shadow_event_bridge_{suffix}",
            "input": {"prompt": f"bridge input {suffix}"},
            "chosen": {"answer": f"bridge chosen {suffix}"},
            "tags": ["peds", "shadow_event", "accepted_by_incumbent"],
            "provenance": {
                "source_table": "shadow_events",
                "shadow_run_id": f"shadow_run_bridge_{suffix}",
                "shadow_event_id": f"shadow_event_bridge_{suffix}",
                "ordinal": 0,
            },
            "safety_exclusions": [],
            "metadata": {"judge_verdict": "pass"},
        }

    def _manifest(self, candidate_id: str, suffix: str) -> dict:
        examples = [self._example(candidate_id, suffix)]
        return {
            "candidate_id": candidate_id,
            "created_at": "2026-06-02T00:00:00Z",
            "dataset_hash": self._dataset_hash(examples),
            "n_examples": len(examples),
            "examples": examples,
            "source_summary": {"candidate_id": candidate_id, "n_shadow_events": 1},
            "warnings": [],
        }

    def _prepared_run(self, suffix: str) -> tuple[dict, dict]:
        candidate = self._candidate(suffix)
        run = prepare_training_run(
            self.conn,
            self._manifest(candidate["candidate_id"], suffix),
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 1},
            output_dir=f"adapters/peds/{suffix}",
        )
        return candidate, run

    def _completed_run(self, suffix: str, *, mode: str = "dry_run") -> tuple[dict, dict]:
        candidate, run = self._prepared_run(suffix)
        result = run_prepared_training_run(
            self.conn,
            run["training_run_id"],
            operator="owner",
            mode=mode,
            artifact_dir=str(self.paths.root / "peds_adapter_artifacts" / suffix),
        )
        return candidate, result["training_run"]

    def _dataset_hash(self, examples: list[dict]) -> str:
        payload = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def test_validate_training_artifact_rejects_missing_training_run(self) -> None:
        validation = validate_training_artifact_for_candidate(self.conn, "missing_training_run")

        self.assertFalse(validation["valid"])
        self.assertIn("Unknown PEDS training run", validation["reason"])

    def test_validate_training_artifact_rejects_non_completed_training_run(self) -> None:
        _, run = self._prepared_run("prepared")

        validation = validate_training_artifact_for_candidate(self.conn, run["training_run_id"])

        self.assertFalse(validation["valid"])
        self.assertIn("status must be completed", validation["reason"])

    def test_register_training_artifact_rejects_empty_adapter_name(self) -> None:
        _, run = self._completed_run("empty_adapter_name")

        with self.assertRaises(ValueError):
            register_training_artifact(
                self.conn,
                run["training_run_id"],
                adapter_name=" ",
                operator="owner",
            )

    def test_register_training_artifact_rejects_empty_operator(self) -> None:
        _, run = self._completed_run("empty_operator")

        with self.assertRaises(ValueError):
            register_training_artifact(
                self.conn,
                run["training_run_id"],
                adapter_name="peds_adapter",
                operator="",
            )

    def test_register_training_artifact_require_trained_rejects_dry_run(self) -> None:
        _, run = self._completed_run("require_trained")

        with self.assertRaises(ValueError):
            register_training_artifact(
                self.conn,
                run["training_run_id"],
                adapter_name="peds_adapter",
                operator="owner",
                require_trained=True,
            )

    def test_metadata_only_registration_succeeds_when_require_trained_false(self) -> None:
        _, run = self._completed_run("metadata_registration", mode="metadata_only")

        registration = register_training_artifact(
            self.conn,
            run["training_run_id"],
            adapter_name="peds_adapter",
            adapter_version="metadata-v1",
            operator="owner",
            require_trained=False,
        )

        self.assertFalse(registration["trained"])
        self.assertEqual(registration["status"], "metadata_only")
        self.assertEqual(registration["adapter"]["name"], "peds_adapter")
        self.assertEqual(registration["adapter"]["version"], "metadata-v1")
        self.assertEqual(registration["adapter"]["dataset_hash"], run["dataset_hash"])

    def test_trained_registration_succeeds_for_local_json_trainer(self) -> None:
        _, run = self._completed_run("trained_registration", mode="local_json_trainer")

        registration = register_training_artifact(
            self.conn,
            run["training_run_id"],
            adapter_name="peds_adapter",
            adapter_version="trained-v1",
            operator="owner",
            require_trained=True,
        )

        self.assertTrue(registration["trained"])
        self.assertEqual(registration["status"], "trained")
        self.assertEqual(registration["adapter"]["dataset_hash"], run["dataset_hash"])

    def test_create_candidate_from_training_run_creates_theta_candidate(self) -> None:
        parent, run = self._completed_run("create_candidate")

        candidate = create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
        )

        self.assertEqual(candidate["parent_candidate_id"], parent["candidate_id"])
        self.assertEqual(candidate["mutation_target"], "theta")
        self.assertTrue(candidate["theta_version"].startswith("adapter:peds_theta_adapter:"))
        self.assertIn("adapter_registration", candidate)

    def test_new_candidate_current_state_is_draft_candidate(self) -> None:
        parent, run = self._completed_run("draft_state")

        candidate = create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
        )

        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "draft_candidate")
        self.assertEqual(candidate["current_state"], "draft_candidate")

    def test_create_candidate_from_training_run_writes_no_promotion_decisions(self) -> None:
        parent, run = self._completed_run("no_decisions")
        before = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
        )

        after = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]
        self.assertEqual(after, before)

    def test_candidate_manifest_includes_training_and_adapter_metadata(self) -> None:
        parent, run = self._completed_run("manifest_metadata")

        candidate = create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
            delta_description="bridge candidate",
        )
        manifest = candidate["manifest"]

        self.assertEqual(manifest["source_training_run_id"], run["training_run_id"])
        self.assertEqual(manifest["dataset_hash"], run["dataset_hash"])
        self.assertEqual(manifest["trainer_name"], run["trainer_name"])
        self.assertFalse(manifest["trained"])
        self.assertEqual(manifest["adapter_ref"]["status"], "metadata_only")
        self.assertEqual(manifest["operator"], "owner")

    def test_trained_candidate_manifest_reflects_local_json_artifact(self) -> None:
        parent, run = self._completed_run("manifest_trained", mode="local_json_trainer")

        candidate = create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
            require_trained=True,
        )
        manifest = candidate["manifest"]

        self.assertTrue(manifest["trained"])
        self.assertEqual(manifest["trainer_mode"], "local_json_trainer")
        self.assertEqual(manifest["adapter_ref"]["status"], "trained")

    def test_parent_candidate_remains_unchanged(self) -> None:
        parent, run = self._completed_run("parent_unchanged")
        before = dict(
            self.conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (parent["candidate_id"],),
            ).fetchone()
        )

        create_candidate_from_training_run(
            self.conn,
            run["training_run_id"],
            parent_candidate_id=parent["candidate_id"],
            adapter_name="peds_theta_adapter",
            operator="owner",
        )

        after = dict(
            self.conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?",
                (parent["candidate_id"],),
            ).fetchone()
        )
        self.assertEqual(after, before)

    def test_create_candidate_rejects_mismatched_parent_candidate(self) -> None:
        _, run = self._completed_run("mismatched_parent")
        other_parent = self._candidate("other_parent")
        before_adapters = self.conn.execute("SELECT COUNT(*) AS count FROM adapter_registry").fetchone()["count"]

        with self.assertRaises(ValueError):
            create_candidate_from_training_run(
                self.conn,
                run["training_run_id"],
                parent_candidate_id=other_parent["candidate_id"],
                adapter_name="peds_theta_adapter",
                operator="owner",
            )

        after_adapters = self.conn.execute("SELECT COUNT(*) AS count FROM adapter_registry").fetchone()["count"]
        self.assertEqual(after_adapters, before_adapters)
