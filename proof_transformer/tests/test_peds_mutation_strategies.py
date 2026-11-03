from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import peds_training_campaign  # noqa: E402
from proof_transformer.peds.mutation_strategies import (  # noqa: E402
    propose_candidate_mutation,
    validate_single_mutation,
)


class PEDSMutationStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.parent = {
            "candidate_id": "candidate_parent",
            "manifest": {"theta_strategy": "acceptance_checklist_first"},
        }

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_each_diagnosis_category_produces_deterministic_valid_mutation(self) -> None:
        expected_targets = {
            "adapter_load_error": "theta",
            "canary_rejection": "rho",
            "insufficient_delta": "theta",
            "no_generated_child": "psi",
            "replay_underperformance": "theta",
            "shadow_rejection": "rho",
            "training_error": "psi",
            "unknown": "theta",
        }
        for category, expected_target in expected_targets.items():
            first = propose_candidate_mutation(
                self.parent,
                diagnosis={"category": category},
                attempt_index=1,
                generation_index=2,
            )
            second = propose_candidate_mutation(
                self.parent,
                diagnosis={"category": category},
                attempt_index=1,
                generation_index=2,
            )

            self.assertEqual(first, second)
            self.assertEqual(first["mutation_target"], expected_target)
            self.assertTrue(validate_single_mutation(first)["valid"])

    def test_same_input_produces_same_seed_and_attempt_index_changes_seed(self) -> None:
        first = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "canary_rejection"},
            attempt_index=1,
            generation_index=3,
        )
        repeat = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "canary_rejection"},
            attempt_index=1,
            generation_index=3,
        )
        next_attempt = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "canary_rejection"},
            attempt_index=2,
            generation_index=3,
        )

        self.assertEqual(first["mutation_seed"], repeat["mutation_seed"])
        self.assertNotEqual(first["mutation_seed"], next_attempt["mutation_seed"])
        self.assertEqual(next_attempt["mutation_target"], "psi")

    def test_validate_single_mutation_rejects_multiple_mutation_targets(self) -> None:
        payload = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "replay_underperformance"},
            attempt_index=1,
            generation_index=1,
        )
        payload["tool_policy"] = {"policy": "extra_policy"}

        validation = validate_single_mutation(payload)

        self.assertFalse(validation["valid"])
        self.assertIn("exactly one target-specific mutation field must be populated", validation["failed_checks"])

    def test_validate_single_mutation_accepts_exactly_one_mutation_target(self) -> None:
        payload = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "training_error"},
            attempt_index=1,
            generation_index=1,
        )

        validation = validate_single_mutation(payload)

        self.assertTrue(validation["valid"])
        self.assertEqual(validation["mutation_target"], "psi")

    def test_campaign_retry_includes_mutation_payload_in_attempt_source_manifest(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "campaign_retry_manifest",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._reject_once_runner,
        )

        rejected, retry = report["attempts"][:2]
        retry_manifest = retry["attempt_candidate_manifest"]

        self.assertEqual(rejected["outcome"], "rejected")
        self.assertEqual(retry["outcome"], "winner")
        self.assertIn("candidate_mutation", retry_manifest)
        self.assertEqual(retry_manifest["candidate_mutation"], retry["mutation"])
        self.assertEqual(retry_manifest["mutation_target"], retry["mutation_target"])
        self.assertEqual(retry_manifest["mutation_seed"], retry["mutation_seed"])
        self.assertEqual(retry_manifest["diagnosis_category"], retry["mutation"]["diagnosis_category"])

    def test_corrected_retry_mutation_differs_from_rejected_attempt_mutation(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "campaign_retry_differs",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._reject_once_runner,
        )

        rejected, retry = report["attempts"][:2]

        self.assertNotEqual(rejected["mutation_seed"], retry["mutation_seed"])
        self.assertNotEqual(rejected["mutation"]["diagnosis_category"], retry["mutation"]["diagnosis_category"])
        self.assertNotEqual(rejected["attempt_candidate_manifest"]["candidate_mutation"], retry["attempt_candidate_manifest"]["candidate_mutation"])

    def test_canary_rejection_does_not_weaken_canary_thresholds(self) -> None:
        options = {
            "candidate_variant": "acceptance_checklist_default",
            "canary_thresholds": {"max_canary_failures": 0},
            "trainer_config": {"epochs": 1},
        }
        mutation = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "canary_rejection"},
            attempt_index=2,
            generation_index=1,
        )

        updated = peds_training_campaign.apply_correction_to_generation_options(
            options,
            {"correction": "increase_canary_diversity", "mutation": mutation},
            attempt_index=2,
        )

        self.assertEqual(updated["canary_thresholds"], {"max_canary_failures": 0})
        self.assertEqual(updated["candidate_mutation"], mutation)

    def test_replay_underperformance_does_not_weaken_replay_thresholds(self) -> None:
        options = {
            "candidate_variant": "acceptance_checklist_default",
            "replay_thresholds": {"epsilon": 0.02},
            "trainer_config": {"epochs": 1},
        }
        mutation = propose_candidate_mutation(
            self.parent,
            diagnosis={"category": "replay_underperformance"},
            attempt_index=2,
            generation_index=1,
        )

        updated = peds_training_campaign.apply_correction_to_generation_options(
            options,
            {"correction": "increase_replay_diversity", "mutation": mutation},
            attempt_index=2,
        )

        self.assertEqual(updated["replay_thresholds"], {"epsilon": 0.02})
        self.assertEqual(updated["candidate_mutation"], mutation)

    def _reject_once_runner(self, **kwargs) -> dict:
        attempt = 2 if "attempt02" in kwargs["run_id"] else 1
        decision = "replay_winner" if attempt == 2 else "rejected"
        delta = 0.08 if attempt == 2 else -0.05
        options = dict(kwargs.get("generation_options") or {})
        source_candidate_id = f"source_attempt_{attempt}"
        return {
            "brain_root": str(kwargs["brain_root"]),
            "fake_weights_written": False,
            "final_candidate_id": f"candidate_final_{attempt}",
            "generation_reports": [
                {
                    "adapter_registry_name": "acceptance_checklist_theta_gen_01",
                    "adapter_registry_status": "trained",
                    "generated_candidate_id": f"candidate_generated_{attempt}",
                    "generation": 1,
                    "replay_probe": {
                        "adapter_loaded": True,
                        "replay_gate_decision": decision,
                        "replay_mean_delta": delta,
                        "replay_results_written": 4,
                        "state_after_replay": decision,
                    },
                    "source_candidate_id": source_candidate_id,
                    "training_trained": True,
                }
            ],
            "generations_completed": 1,
            "input_parent_candidate_id": kwargs.get("parent_candidate_id"),
            "input_source_candidate_id": source_candidate_id,
            "input_source_manifest": self._attempt_manifest(source_candidate_id, options),
            "model_modules_loaded": [],
            "network_attempts": [],
            "run_id": kwargs["run_id"],
        }

    def _attempt_manifest(self, source_candidate_id: str, options: dict) -> dict:
        mutation = options.get("candidate_mutation")
        return {
            "candidate_id": source_candidate_id,
            "candidate_mutation": mutation,
            "diagnosis_category": (mutation or {}).get("diagnosis_category"),
            "delta_description": (mutation or {}).get("delta_description"),
            "generation_options": {"candidate_mutation": mutation},
            "mutation_seed": (mutation or {}).get("mutation_seed"),
            "mutation_target": (mutation or {}).get("mutation_target"),
        }


if __name__ == "__main__":
    unittest.main()
