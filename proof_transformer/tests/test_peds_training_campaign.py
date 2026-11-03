from __future__ import annotations

import importlib
import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

peds_dashboard = importlib.import_module("peds_dashboard")
peds_latest_campaign_report = importlib.import_module("peds_latest_campaign_report")
peds_training_campaign = importlib.import_module("peds_training_campaign")
peds_trained_local_generations = importlib.import_module("peds_trained_local_generations")


class PEDSTrainingCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.runner_calls: list[dict] = []

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_analyze_campaign_detects_collapse_on_non_replay_winner(self) -> None:
        runs = [self._run_summary([self._metric(1, decision="rejected")])]

        aggregate = peds_training_campaign.analyze_campaign(runs, generations_requested=1)

        self.assertTrue(aggregate["collapse_detected"])
        self.assertEqual(aggregate["first_failure"]["replay_decision"], "rejected")

    def test_analyze_campaign_detects_collapse_on_non_positive_delta(self) -> None:
        runs = [self._run_summary([self._metric(1, delta=0.0)])]

        aggregate = peds_training_campaign.analyze_campaign(runs, generations_requested=1)

        self.assertTrue(aggregate["collapse_detected"])
        self.assertEqual(aggregate["first_failure"]["replay_mean_delta"], 0.0)

    def test_analyze_campaign_detects_plateau_across_plateau_window(self) -> None:
        runs = [self._run_summary([self._metric(1, delta=0.1), self._metric(2, delta=0.102), self._metric(3, delta=0.101)])]

        aggregate = peds_training_campaign.analyze_campaign(
            runs,
            generations_requested=3,
            plateau_window=3,
            plateau_epsilon=0.005,
        )

        self.assertTrue(aggregate["plateau_detected"])
        self.assertEqual(aggregate["delta_trend"], "flat")

    def test_analyze_campaign_identifies_best_and_worst_generation(self) -> None:
        runs = [self._run_summary([self._metric(1, delta=0.03), self._metric(2, delta=0.12), self._metric(3, delta=0.05)])]

        aggregate = peds_training_campaign.analyze_campaign(runs, generations_requested=3)

        self.assertEqual(aggregate["best_generation"]["generation"], 2)
        self.assertEqual(aggregate["worst_generation"]["generation"], 1)

    def test_aggregate_computes_rates(self) -> None:
        runs = [
            self._run_summary(
                [
                    self._metric(1),
                    self._metric(2, trained=False),
                    self._metric(3, adapter_loaded=False),
                ]
            )
        ]

        aggregate = peds_training_campaign.analyze_campaign(runs, generations_requested=3)

        self.assertAlmostEqual(aggregate["replay_winner_rate"], 1.0)
        self.assertAlmostEqual(aggregate["trained_rate"], 2 / 3)
        self.assertAlmostEqual(aggregate["adapter_loaded_rate"], 2 / 3)
        self.assertTrue(aggregate["collapse_detected"])

    def test_diagnose_attempt_outcome_classifies_core_rejections(self) -> None:
        replay = peds_training_campaign.diagnose_attempt_outcome(
            {"metric": self._metric(1, decision="rejected")},
            {"min_delta": 0.02},
        )
        canary = peds_training_campaign.diagnose_attempt_outcome(
            {"source_report": {"blocker": "canary rejected candidate"}},
            {"min_delta": 0.02},
        )
        canary_missing_child = peds_training_campaign.diagnose_attempt_outcome(
            {
                "source_report": {
                    "blocker": "AssertionError: cycle did not create generated child payload",
                    "cycle_actions": ["run_canary_job", "unsupported_candidate_state"],
                }
            },
            {"min_delta": 0.02},
        )
        adapter = peds_training_campaign.diagnose_attempt_outcome(
            {"metric": self._metric(1, adapter_loaded=False)},
            {"min_delta": 0.02},
        )
        insufficient = peds_training_campaign.diagnose_attempt_outcome(
            {"metric": self._metric(1, delta=0.01)},
            {"min_delta": 0.02},
        )

        self.assertEqual(replay["category"], "replay_underperformance")
        self.assertIn("confidence", replay)
        self.assertIn("diagnosis", replay)
        self.assertIn("evidence", replay)
        self.assertIn("recommended_corrections", replay)
        self.assertEqual(canary["category"], "canary_rejection")
        self.assertEqual(canary_missing_child["category"], "canary_rejection")
        self.assertEqual(adapter["category"], "adapter_load_error")
        self.assertEqual(insufficient["category"], "insufficient_delta")

    def test_choose_correction_uses_only_allowed_actions(self) -> None:
        for category in peds_training_campaign.DIAGNOSIS_CATEGORIES:
            correction = peds_training_campaign.choose_correction(
                {"category": category},
                {"attempt_index": 1},
            )

            self.assertIn(correction["correction"], peds_training_campaign.CORRECTION_ACTIONS)
            self.assertEqual(correction["action"], correction["correction"])
            self.assertIn("reason", correction)
            self.assertIsInstance(correction["parameters"], dict)

    def test_apply_correction_changes_campaign_inputs_without_lowering_gates(self) -> None:
        options = {
            "candidate_variant": "acceptance_checklist_default",
            "canary_threshold": 0.9,
            "min_delta": 0.02,
            "promotion_decision": "evidence_only",
            "replay_threshold": 0.8,
            "replay_thresholds": {"epsilon": 0.02},
            "shadow_thresholds": {"max_divergence_rate": 0.3},
            "trainer_config": {"epochs": 1},
        }

        updated = peds_training_campaign.apply_correction_to_generation_options(
            options,
            {"action": "reduce_trainer_noise"},
            attempt_index=2,
        )

        self.assertNotEqual(updated["candidate_variant"], options["candidate_variant"])
        self.assertEqual(updated["min_delta"], 0.02)
        self.assertEqual(updated["replay_threshold"], 0.8)
        self.assertEqual(updated["replay_thresholds"], {"epsilon": 0.02})
        self.assertEqual(updated["canary_threshold"], 0.9)
        self.assertEqual(updated["shadow_thresholds"], {"max_divergence_rate": 0.3})
        self.assertEqual(updated["promotion_decision"], "evidence_only")
        self.assertEqual(options["trainer_config"], {"epochs": 1})
        self.assertEqual(updated["trainer_config"]["noise_policy"], "reduced")

        canary_updated = peds_training_campaign.apply_correction_to_generation_options(
            options,
            {"correction": "increase_canary_diversity"},
            attempt_index=2,
        )
        self.assertTrue(canary_updated["force_new_canary_job"])
        self.assertEqual(canary_updated["canary_selection_policy"], "diverse_attempt_isolated")

    def test_adaptive_campaign_retries_rejected_candidate_until_winner(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "adaptive",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._reject_once_generation_runner,
        )

        self.assertEqual(report["aggregate"]["total_candidate_attempts"], 2)
        self.assertEqual(report["aggregate"]["total_rejected_attempts"], 1)
        self.assertEqual(report["aggregate"]["exhausted_generation_count"], 0)
        self.assertEqual(report["aggregate"]["correction_success_rate"], 1.0)
        self.assertFalse(report["aggregate"]["collapse_detected"])
        self.assertEqual(report["winning_attempts"][0]["winning_attempt_number"], 2)
        self.assertEqual(report["rejected_attempts"][0]["diagnosis"]["category"], "replay_underperformance")

    def test_lineage_advances_winner_child_across_generations(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=3,
            runs=1,
            run_root=self.root / "lineage",
            min_delta=0.02,
            generation_runner=self._lineage_generation_runner,
        )

        gen1_winner = report["winning_attempts"][0]["winning_candidate_id"]
        gen2_winner = report["winning_attempts"][1]["winning_candidate_id"]

        self.assertIsNone(self.runner_calls[0]["parent_candidate_id"])
        self.assertEqual(self.runner_calls[1]["parent_candidate_id"], gen1_winner)
        self.assertEqual(self.runner_calls[2]["parent_candidate_id"], gen2_winner)
        self.assertEqual(report["lineage"][0]["selected_parent_for_next_generation"], gen1_winner)
        self.assertEqual(report["lineage"][1]["selected_parent_for_next_generation"], gen2_winner)
        self.assertEqual(report["winning_attempts"][1]["parent_candidate_id"], gen1_winner)

    def test_rejected_attempt_does_not_advance_lineage_but_winning_retry_does(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=3,
            runs=1,
            run_root=self.root / "lineage_retry",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._reject_generation_two_once_runner,
        )

        gen1_winner = report["winning_attempts"][0]["winning_candidate_id"]
        gen2_rejected = report["rejected_attempts"][0]["generated_child_candidate_id"]
        gen2_winner = report["winning_attempts"][1]["winning_candidate_id"]

        gen2_calls = [call for call in self.runner_calls if call["generation"] == 2]
        gen3_call = [call for call in self.runner_calls if call["generation"] == 3][0]
        self.assertEqual(gen2_calls[0]["parent_candidate_id"], gen1_winner)
        self.assertEqual(gen2_calls[1]["parent_candidate_id"], gen1_winner)
        self.assertNotEqual(gen3_call["parent_candidate_id"], gen2_rejected)
        self.assertEqual(gen3_call["parent_candidate_id"], gen2_winner)

    def test_lineage_report_proves_four_generation_chain_when_winners_available(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=4,
            runs=1,
            run_root=self.root / "lineage_four",
            min_delta=0.02,
            generation_runner=self._lineage_generation_runner,
        )

        winners = [item["winning_candidate_id"] for item in report["winning_attempts"]]
        self.assertEqual(self.runner_calls[1]["parent_candidate_id"], winners[0])
        self.assertEqual(self.runner_calls[2]["parent_candidate_id"], winners[1])
        self.assertEqual(self.runner_calls[3]["parent_candidate_id"], winners[2])
        self.assertEqual(report["lineage"][2]["selected_parent_for_next_generation"], winners[2])
        self.assertEqual(report["lineage"][3]["parent_candidate_id"], winners[2])

    def test_selected_parent_is_protected_by_fresh_attempt_source_on_rejection_and_retry(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=3,
            runs=1,
            run_root=self.root / "protected_parent",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        gen1_winner = report["winning_attempts"][0]["winning_candidate_id"]
        gen2_rejected = [attempt for attempt in report["attempts"] if attempt["generation"] == 2][0]
        gen2_winner = report["winning_attempts"][1]
        gen3_first_call = [call for call in self.runner_calls if call["generation"] == 3][0]

        self.assertEqual(gen2_rejected["parent_candidate_id"], gen1_winner)
        self.assertNotEqual(gen2_rejected["attempt_candidate_id"], gen1_winner)
        self.assertNotEqual(gen2_rejected["source_candidate_id"], gen1_winner)
        self.assertIsNone(gen2_rejected["winning_candidate_id"])
        self.assertIsNone(gen2_rejected["selected_parent_for_next_generation"])
        self.assertEqual(gen2_rejected["parent_state_before_attempt"], "replay_winner")
        self.assertEqual(gen2_rejected["parent_state_after_attempt"], "replay_winner")
        self.assertEqual(gen2_winner["parent_candidate_id"], gen1_winner)
        self.assertNotEqual(gen2_winner["attempt_candidate_id"], gen1_winner)
        self.assertEqual(gen2_winner["selected_parent_for_next_generation"], gen2_winner["winning_candidate_id"])
        self.assertEqual(gen3_first_call["parent_candidate_id"], gen2_winner["winning_candidate_id"])
        self.assertNotEqual(gen3_first_call["parent_candidate_id"], gen2_rejected["attempt_candidate_id"])

    def test_corrected_retry_changes_real_manifest_options_and_preserves_gates(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "correction_reality",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        rejected, retry = report["attempts"][:2]
        rejected_manifest = rejected["attempt_candidate_manifest"]
        retry_manifest = retry["attempt_candidate_manifest"]

        self.assertEqual(rejected["outcome"], "rejected")
        self.assertEqual(retry["outcome"], "winner")
        self.assertIsNotNone(rejected["mutation"])
        self.assertIsNotNone(retry["mutation"])
        self.assertNotEqual(rejected["mutation_seed"], retry["mutation_seed"])
        self.assertEqual(retry_manifest["candidate_mutation"], retry["mutation"])
        self.assertNotEqual(retry_manifest, rejected_manifest)
        self.assertNotEqual(retry["options"]["candidate_variant"], rejected["options"]["candidate_variant"])
        self.assertNotEqual(retry["options"]["evidence_selection_seed"], rejected["options"]["evidence_selection_seed"])
        self.assertIn("deterministic_variant", retry["options"]["trainer_config"])
        self.assertNotEqual(
            retry_manifest["generation_options"],
            rejected_manifest["generation_options"],
        )
        self.assertNotIn("promotion_decision", retry["options"])
        self.assertEqual(retry["options"]["run_parent_as_source"], False)

    def test_rejected_attempt_followed_by_corrected_retry_includes_mutation_payload(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "mutation_retry",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        rejected, retry = report["attempts"][:2]

        self.assertEqual(rejected["outcome"], "rejected")
        self.assertEqual(retry["outcome"], "winner")
        self.assertIn("mutation_target", retry)
        self.assertIn("mutation_seed", retry)
        self.assertIn("delta_description", retry["mutation"])
        self.assertEqual(retry["attempt_candidate_manifest"]["mutation_target"], retry["mutation_target"])
        self.assertEqual(retry["attempt_candidate_manifest"]["mutation_seed"], retry["mutation_seed"])
        self.assertEqual(retry["attempt_candidate_manifest"]["diagnosis_category"], retry["mutation"]["diagnosis_category"])

    def test_report_exposes_mutation_target_seed_category_and_delta_description(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "mutation_report",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        mutation = report["mutations"][0]

        self.assertIn("mutation_target", mutation)
        self.assertIn("mutation_seed", mutation)
        self.assertIn("diagnosis_category", mutation)
        self.assertIn("delta_description", mutation)
        self.assertEqual(report["attempts"][0]["mutation_target"], mutation["mutation_target"])

    def test_winning_corrected_retry_advances_lineage_with_mutation_metadata_preserved(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=2,
            runs=1,
            run_root=self.root / "mutation_lineage",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        corrected_winner = report["winning_attempts"][0]
        gen2_first_call = [call for call in self.runner_calls if call["generation"] == 2][0]

        self.assertEqual(corrected_winner["winning_attempt_number"], 2)
        self.assertEqual(corrected_winner["selected_parent_for_next_generation"], corrected_winner["winning_candidate_id"])
        self.assertEqual(gen2_first_call["parent_candidate_id"], corrected_winner["winning_candidate_id"])
        self.assertIsNotNone(corrected_winner["mutation"])
        self.assertEqual(corrected_winner["mutation_seed"], corrected_winner["mutation"]["mutation_seed"])
        self.assertEqual(report["lineage"][0]["mutation_seed"], corrected_winner["mutation_seed"])

    def test_attempt_evidence_isolation_uses_attempt_specific_episode_sets(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=2,
            runs=1,
            run_root=self.root / "evidence_isolation",
            min_delta=0.02,
            max_attempts_per_generation=3,
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        gen2_attempts = [attempt for attempt in report["attempts"] if attempt["generation"] == 2]
        rejected_evidence = gen2_attempts[0]["evidence"]
        retry_evidence = gen2_attempts[1]["evidence"]

        self.assertEqual(rejected_evidence["replay_episode_ids"], rejected_evidence["replay_job_episode_ids"])
        self.assertEqual(retry_evidence["replay_episode_ids"], retry_evidence["replay_job_episode_ids"])
        self.assertEqual(rejected_evidence["canary_episode_ids"], rejected_evidence["canary_job_episode_ids"])
        self.assertEqual(retry_evidence["canary_episode_ids"], retry_evidence["canary_job_episode_ids"])
        self.assertTrue(set(rejected_evidence["replay_episode_ids"]).isdisjoint(retry_evidence["replay_episode_ids"]))
        self.assertTrue(set(rejected_evidence["canary_episode_ids"]).isdisjoint(retry_evidence["canary_episode_ids"]))
        self.assertIn("attempt_01", rejected_evidence["generation_tag"])
        self.assertIn("attempt_02", retry_evidence["generation_tag"])
        self.assertEqual(retry_evidence["shadow_policy_tags"], [retry_evidence["generation_tag"]])

    def test_fixed_baseline_policy_does_not_advance_lineage(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=3,
            runs=1,
            run_root=self.root / "fixed_baseline",
            min_delta=0.02,
            continuation_policy="fixed-baseline",
            generation_runner=self._lineage_generation_runner,
        )

        baseline_source = report["winning_attempts"][0]["source_candidate_id"]
        gen1_winner = report["winning_attempts"][0]["winning_candidate_id"]

        self.assertEqual(self.runner_calls[1]["parent_candidate_id"], baseline_source)
        self.assertEqual(self.runner_calls[2]["parent_candidate_id"], baseline_source)
        self.assertNotEqual(self.runner_calls[1]["parent_candidate_id"], gen1_winner)
        self.assertEqual(report["lineage"][0]["selected_parent_for_next_generation"], baseline_source)

    def test_last_valid_parent_policy_continues_from_latest_winner(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=2,
            runs=1,
            run_root=self.root / "last_valid",
            min_delta=0.02,
            continuation_policy="last-valid-parent",
            generation_runner=self._lineage_generation_runner,
        )

        self.assertEqual(self.runner_calls[1]["parent_candidate_id"], report["winning_attempts"][0]["winning_candidate_id"])

    def test_report_includes_lineage_parent_source_winner_and_next_parent_ids(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=2,
            runs=1,
            run_root=self.root / "lineage_report",
            min_delta=0.02,
            generation_runner=self._lineage_generation_runner,
        )

        attempt = report["attempts"][1]
        generation = report["runs"][0]["generation_summaries"][1]
        lineage = report["lineage"][1]

        self.assertIn("parent_candidate_id", attempt)
        self.assertIn("source_candidate_id", attempt)
        self.assertIn("winning_candidate_id", generation)
        self.assertIn("generated_child_candidate_id", generation)
        self.assertIn("selected_parent_for_next_generation", lineage)
        self.assertIn("parent_selection_reason", lineage)

    def test_adaptive_campaign_records_exhausted_generation_without_inventing_winner(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "exhausted",
            min_delta=0.02,
            max_attempts_per_generation=2,
            generation_runner=self._always_reject_generation_runner,
        )

        self.assertEqual(report["aggregate"]["total_candidate_attempts"], 2)
        self.assertEqual(report["aggregate"]["total_winning_attempts"], 0)
        self.assertEqual(report["aggregate"]["exhausted_generation_count"], 1)
        self.assertTrue(report["aggregate"]["collapse_detected"])
        self.assertEqual(report["exhausted_generations"][0]["winning_attempt_number"], None)
        self.assertEqual(peds_training_campaign._campaign_exit_code(report, strict=False), 0)
        self.assertEqual(peds_training_campaign._campaign_exit_code(report, strict=True), 1)

    def test_campaign_report_is_json_serializable(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=2,
            runs=1,
            run_root=self.root / "campaign",
            generation_runner=self._fake_generation_runner,
        )

        encoded = json.dumps(report, ensure_ascii=False, sort_keys=True)

        self.assertIn("campaign_report.json", encoded)
        self.assertTrue((self.root / "campaign" / "campaign_report.json").exists())

    def test_generation_runner_signature_supports_campaign_lineage_arguments(self) -> None:
        parameters = inspect.signature(peds_trained_local_generations.run_generations).parameters

        self.assertIn("parent_candidate_id", parameters)
        self.assertIn("generation", parameters)
        self.assertIn("generation_options", parameters)

    def test_command_builder_does_not_call_network_or_model_apis(self) -> None:
        command = peds_training_campaign.build_campaign_command(generations=5, runs=2, repo_root=ROOT)
        command_text = " ".join(command).lower()

        self.assertIn("peds_training_campaign.py", command[1])
        self.assertNotIn("http", command_text)
        self.assertNotIn("ollama", command_text)
        self.assertNotIn("openai", command_text)
        self.assertNotIn("model", command_text)

    def test_dashboard_imports_without_launching_mainloop_after_campaign_button_addition(self) -> None:
        self.assertTrue(hasattr(peds_dashboard, "build_campaign_command"))
        self.assertTrue(hasattr(peds_dashboard, "PEDSDashboard"))

    def test_tiny_campaign_smoke_uses_mocked_generation_runner(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "tiny",
            generation_runner=self._fake_generation_runner,
        )

        self.assertEqual(report["aggregate"]["total_generations_completed"], 1)
        self.assertFalse(report["aggregate"]["collapse_detected"])
        self.assertTrue(report["runs"][0]["passed"])

    def test_active_policy_seeds_mutation_context_without_db_parent(self) -> None:
        report = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "active_policy_campaign",
            max_attempts_per_generation=2,
            active_policy={
                "active_policy_id": "policy_active",
                "candidate_id": "candidate_active",
                "objective": "Improve Codex patch reliability",
            },
            generation_runner=self._fresh_source_retry_generation_runner,
        )

        first_call = self.runner_calls[0]
        first_options = first_call["generation_options"]

        self.assertEqual(first_options["active_policy_candidate_id"], "candidate_active")
        self.assertEqual(first_options["active_policy"]["active_policy_id"], "policy_active")
        self.assertTrue(first_options["active_policy_start"]["used_as_mutation_context"])
        self.assertFalse(first_options["active_policy_start"]["used_as_db_parent"])
        self.assertIsNone(first_call["parent_candidate_id"])
        self.assertTrue(report["active_policy_start"]["used_as_mutation_context"])
        self.assertFalse(report["active_policy_start"]["used_as_db_parent"])
        self.assertEqual(report["runs"][0]["attempts"][0]["options"]["active_policy_candidate_id"], "candidate_active")

    def test_latest_campaign_report_reads_latest_report(self) -> None:
        first = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "campaign_a",
            generation_runner=self._fake_generation_runner,
        )
        second = peds_training_campaign.run_campaign(
            generations=1,
            runs=1,
            run_root=self.root / "campaign_b",
            generation_runner=self._fake_generation_runner,
        )
        Path(first["report_path"]).touch()
        Path(second["report_path"]).touch()

        report = peds_latest_campaign_report.build_latest_campaign_report(self.root)

        self.assertTrue(report["found"])
        self.assertIn(report["campaign_id"], {first["campaign_id"], second["campaign_id"]})
        self.assertEqual(report["aggregate"]["total_generations_completed"], 1)
        self.assertIn("attempts", report)
        self.assertIn("rejected_attempts", report)
        self.assertIn("corrections", report)
        self.assertIn("mutations", report)

    def _run_summary(self, metrics: list[dict]) -> dict:
        return {
            "run_index": 1,
            "run_root": str(self.root / "run"),
            "generations_completed": len(metrics),
            "passed": not any(peds_training_campaign._metric_collapsed(metric) for metric in metrics),
            "blocker": None,
            "generation_metrics": metrics,
        }

    def _metric(
        self,
        generation: int,
        *,
        decision: str = "replay_winner",
        delta: float = 0.1,
        trained: bool = True,
        adapter_loaded: bool = True,
    ) -> dict:
        return {
            "generation": generation,
            "candidate_id": f"candidate_{generation}",
            "parent_candidate_id": f"parent_{generation}",
            "adapter_status": "trained",
            "training_trained": trained,
            "adapter_loaded": adapter_loaded,
            "replay_decision": decision,
            "replay_mean_delta": delta,
            "replay_results_written": 4,
            "state_after_replay": "replay_winner",
        }

    def _fake_generation_runner(self, *, brain_root: Path, run_id: str, generations: int, strict: bool = False) -> dict:
        generation_reports = []
        for generation in range(1, generations + 1):
            generation_reports.append(
                {
                    "adapter_registry_name": f"acceptance_checklist_theta_gen_{generation:02d}",
                    "adapter_registry_status": "trained",
                    "generated_candidate_id": f"candidate_{run_id}_{generation}",
                    "generation": generation,
                    "replay_probe": {
                        "adapter_loaded": True,
                        "replay_gate_decision": "replay_winner",
                        "replay_mean_delta": 0.1 / generation,
                        "replay_results_written": 4,
                        "state_after_replay": "replay_winner",
                    },
                    "source_candidate_id": f"parent_{run_id}_{generation}",
                    "training_trained": True,
                }
            )
        return {
            "brain_root": str(brain_root),
            "fake_weights_written": False,
            "final_candidate_id": f"candidate_{run_id}_final",
            "generation_reports": generation_reports,
            "generations_completed": generations,
            "model_modules_loaded": [],
            "network_attempts": [],
            "run_id": run_id,
        }

    def _reject_once_generation_runner(self, **kwargs) -> dict:
        attempt = 2 if "attempt02" in kwargs["run_id"] else 1
        delta = 0.08 if attempt == 2 else -0.05
        decision = "replay_winner" if attempt == 2 else "rejected"
        return self._runner_report(kwargs["brain_root"], kwargs["run_id"], delta=delta, decision=decision)

    def _always_reject_generation_runner(self, **kwargs) -> dict:
        return self._runner_report(kwargs["brain_root"], kwargs["run_id"], delta=-0.05, decision="rejected")

    def _lineage_generation_runner(self, **kwargs) -> dict:
        generation = int(kwargs.get("generation") or 1)
        parent_candidate_id = kwargs.get("parent_candidate_id")
        source_candidate_id = parent_candidate_id or f"seed_source_{kwargs['run_id']}"
        generated_candidate_id = f"winner_generation_{generation}_call_{len(self.runner_calls) + 1}"
        self.runner_calls.append(
            {
                "generation": generation,
                "generated_candidate_id": generated_candidate_id,
                "parent_candidate_id": parent_candidate_id,
                "source_candidate_id": source_candidate_id,
            }
        )
        return self._runner_report(
            kwargs["brain_root"],
            kwargs["run_id"],
            delta=0.08,
            decision="replay_winner",
            generated_candidate_id=generated_candidate_id,
            source_candidate_id=source_candidate_id,
        )

    def _reject_generation_two_once_runner(self, **kwargs) -> dict:
        generation = int(kwargs.get("generation") or 1)
        parent_candidate_id = kwargs.get("parent_candidate_id")
        source_candidate_id = parent_candidate_id or f"seed_source_{kwargs['run_id']}"
        reject = generation == 2 and "attempt01" in kwargs["run_id"]
        generated_candidate_id = f"{'rejected' if reject else 'winner'}_generation_{generation}_call_{len(self.runner_calls) + 1}"
        self.runner_calls.append(
            {
                "generation": generation,
                "generated_candidate_id": generated_candidate_id,
                "parent_candidate_id": parent_candidate_id,
                "source_candidate_id": source_candidate_id,
            }
        )
        return self._runner_report(
            kwargs["brain_root"],
            kwargs["run_id"],
            delta=-0.05 if reject else 0.08,
            decision="rejected" if reject else "replay_winner",
            generated_candidate_id=generated_candidate_id,
            source_candidate_id=source_candidate_id,
        )

    def _fresh_source_retry_generation_runner(self, **kwargs) -> dict:
        generation = int(kwargs.get("generation") or 1)
        attempt = 2 if "attempt02" in kwargs["run_id"] else 1
        parent_candidate_id = kwargs.get("parent_candidate_id")
        options = dict(kwargs.get("generation_options") or {})
        source_candidate_id = f"attempt_source_generation_{generation}_attempt_{attempt}_call_{len(self.runner_calls) + 1}"
        generated_candidate_id = f"winner_generation_{generation}_attempt_{attempt}_call_{len(self.runner_calls) + 1}"
        reject = generation == 2 and attempt == 1
        if generation == 1:
            reject = attempt == 1
        self.runner_calls.append(
            {
                "attempt": attempt,
                "generation": generation,
                "generated_candidate_id": generated_candidate_id,
                "generation_options": options,
                "parent_candidate_id": parent_candidate_id,
                "source_candidate_id": source_candidate_id,
            }
        )
        return self._runner_report(
            kwargs["brain_root"],
            kwargs["run_id"],
            delta=-0.05 if reject else 0.08,
            decision="rejected" if reject else "replay_winner",
            generated_candidate_id=generated_candidate_id,
            source_candidate_id=source_candidate_id,
            input_parent_candidate_id=parent_candidate_id,
            input_source_manifest=self._attempt_manifest(parent_candidate_id, source_candidate_id, options),
            evidence=self._attempt_evidence(options, generation, attempt),
        )

    def _attempt_manifest(self, parent_candidate_id: str | None, source_candidate_id: str, options: dict) -> dict:
        mutation = options.get("candidate_mutation")
        return {
            "candidate_id": source_candidate_id,
            "candidate_variant": options.get("candidate_variant"),
            "candidate_mutation": mutation,
            "diagnosis_category": (mutation or {}).get("diagnosis_category"),
            "delta_description": (mutation or {}).get("delta_description"),
            "generation_options": {
                "candidate_mutation": mutation,
                "candidate_variant": options.get("candidate_variant"),
                "correction_action": options.get("correction_action"),
                "evidence_selection_seed": options.get("evidence_selection_seed"),
                "theta_strategy_suffix": options.get("theta_strategy_suffix"),
                "trainer_config": options.get("trainer_config"),
            },
            "lineage_parent_candidate_id": parent_candidate_id,
            "mutation_seed": (mutation or {}).get("mutation_seed"),
            "mutation_target": (mutation or {}).get("mutation_target"),
            "parent_derivation": "fresh_attempt_candidate_from_lineage_parent",
            "theta_strategy": "acceptance_checklist_first",
            "theta_strategy_suffix": options.get("theta_strategy_suffix") or "initial",
            "trainer_config": options.get("trainer_config"),
        }

    def _attempt_evidence(self, options: dict, generation: int, attempt: int) -> dict:
        tag = options.get("attempt_artifact_suffix") or f"trained_generation_{generation:02d}_attempt_{attempt:02d}"
        replay = [f"{tag}_replay_{index}" for index in range(4)]
        canary = [f"{tag}_canary_{index}" for index in range(3)]
        shadow = [f"{tag}_online_{index}" for index in range(10)]
        return {
            "base_generation_tag": f"trained_generation_{generation:02d}",
            "canary_episode_ids": canary,
            "canary_job_episode_ids": list(canary),
            "generation_tag": tag,
            "online_episode_ids": shadow,
            "replay_episode_ids": replay,
            "replay_job_episode_ids": list(replay),
            "shadow_policy_tags": [tag],
            "shadow_source_episode_ids": shadow,
        }

    def _runner_report(
        self,
        brain_root: Path,
        run_id: str,
        *,
        delta: float,
        decision: str,
        generated_candidate_id: str | None = None,
        source_candidate_id: str | None = None,
        input_parent_candidate_id: str | None = None,
        input_source_manifest: dict | None = None,
        evidence: dict | None = None,
    ) -> dict:
        generated_candidate_id = generated_candidate_id or f"candidate_{run_id}_1"
        source_candidate_id = source_candidate_id or f"parent_{run_id}_1"
        return {
            "brain_root": str(brain_root),
            "fake_weights_written": False,
            "final_candidate_id": f"candidate_{run_id}_final",
            "generation_reports": [
                {
                    "adapter_registry_name": "acceptance_checklist_theta_gen_01",
                    "adapter_registry_status": "trained",
                    "generated_candidate_id": generated_candidate_id,
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
                    **({"evidence": evidence} if evidence is not None else {}),
                }
            ],
            "generations_completed": 1,
            "input_parent_candidate_id": input_parent_candidate_id,
            "input_parent_state_after_attempt": "replay_winner" if input_parent_candidate_id else None,
            "input_parent_state_before_attempt": "replay_winner" if input_parent_candidate_id else None,
            "input_source_candidate_id": source_candidate_id,
            "input_source_manifest": input_source_manifest,
            "model_modules_loaded": [],
            "network_attempts": [],
            "run_id": run_id,
        }


if __name__ == "__main__":
    unittest.main()
