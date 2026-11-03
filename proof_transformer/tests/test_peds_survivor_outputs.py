from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.peds.survivor_outputs import (  # noqa: E402
    append_active_policy,
    build_codex_prompt_from_survivor,
    build_codex_skill_from_survivor,
    build_repo_improvement_plan,
    build_survivor_profile,
    load_latest_active_policy,
    select_current_survivor,
    write_autopilot_artifacts,
)


class PEDSSurvivorOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_select_current_survivor_picks_latest_winning_attempt(self) -> None:
        survivor = select_current_survivor(self._report())

        self.assertIsNotNone(survivor)
        self.assertEqual(survivor["candidate_id"], "candidate_winner_2")
        self.assertEqual(survivor["generation"], 2)
        self.assertEqual(survivor["attempt_index"], 2)

    def test_select_current_survivor_never_picks_rejected_attempt(self) -> None:
        report = self._report()
        report["attempts"] = [attempt for attempt in report["attempts"] if attempt["outcome"] == "rejected"]
        report["winning_attempts"] = []
        report["lineage"] = []

        survivor = select_current_survivor(report)

        self.assertIsNone(survivor)

    def test_select_current_survivor_fallback_to_lineage_selected_parent(self) -> None:
        report = self._report()
        report["attempts"] = []
        report["winning_attempts"] = []

        survivor = select_current_survivor(report)

        self.assertIsNotNone(survivor)
        self.assertEqual(survivor["candidate_id"], "candidate_lineage_latest")
        self.assertEqual(survivor["selection_reason"], "latest lineage selected parent with clean safety")

    def test_select_current_survivor_reads_nested_run_attempts_when_flat_attempts_are_missing(self) -> None:
        report = self._report()
        report["attempts"] = []
        report["winning_attempts"] = []
        report["lineage"] = []
        report["aggregate"] = {}
        report["runs"] = [
            {
                "run_index": 1,
                "generation_summaries": [
                    {
                        "generation": 3,
                        "attempts": [
                            {
                                "generation": 3,
                                "attempt_index": 1,
                                "outcome": "rejected",
                                "winning_candidate_id": None,
                                "mutation": {"mutation_target": "theta", "diagnosis_category": "replay_underperformance"},
                            },
                            {
                                "generation": 3,
                                "attempt_index": 2,
                                "outcome": "winner",
                                "winning_candidate_id": "candidate_nested_winner",
                                "parent_candidate_id": "candidate_winner_2",
                                "mutation": {"mutation_target": "psi", "mutation_seed": "seed_nested"},
                                "metric": {"replay_mean_delta": 0.12, "adapter_loaded": True, "training_trained": True},
                            },
                        ],
                    }
                ],
            }
        ]

        survivor = select_current_survivor(report)
        profile = build_survivor_profile(report, "Improve Codex patch reliability", survivor)

        self.assertIsNotNone(survivor)
        self.assertEqual(survivor["candidate_id"], "candidate_nested_winner")
        self.assertEqual(profile["evidence_summary"]["total_candidate_attempts"], 2)

    def test_build_survivor_profile_includes_objective_evidence_safety_and_uses(self) -> None:
        survivor = select_current_survivor(self._report())
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability", survivor)

        self.assertEqual(profile["profile_type"], "peds_survivor_profile")
        self.assertEqual(profile["objective"], "Improve Codex patch reliability")
        self.assertEqual(profile["candidate_id"], "candidate_winner_2")
        self.assertTrue(profile["what_improved"])
        self.assertIn("evidence_summary", profile)
        self.assertIn("agentic_capability_coverage", profile)
        self.assertIn("agentic_capability_curriculum", profile)
        self.assertEqual(profile["agentic_capability_coverage"]["status"], "broadening_needed")
        self.assertIn("ui_or_browser_operation", profile["agentic_capability_coverage"]["untested_capabilities"])
        self.assertTrue(profile["agentic_capability_coverage"]["next_eval_tasks"])
        self.assertIn("safety", profile)
        self.assertTrue(profile["recommended_uses"])
        self.assertTrue(profile["next_tasks"])
        self.assertTrue(profile["validation_commands"])

    def test_build_codex_prompt_from_survivor_includes_key_evidence_and_rules(self) -> None:
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability")

        prompt = build_codex_prompt_from_survivor(profile)

        self.assertIn("Improve Codex patch reliability", prompt)
        self.assertIn("candidate_winner_2", prompt)
        self.assertIn("Winning strategy", prompt)
        self.assertIn("Mutation/correction evidence", prompt)
        self.assertIn("Validation/evidence", prompt)
        self.assertIn("Agentic capability coverage", prompt)
        self.assertIn("Next agentic eval tasks", prompt)
        self.assertIn("broad agentic progress", prompt)
        self.assertIn("--claim-codex-task", prompt)
        self.assertIn("--run-next", prompt)
        self.assertIn("--record-task-result", prompt)
        self.assertIn("Final report contract", prompt)
        self.assertIn("Do not", prompt)

    def test_build_repo_improvement_plan_includes_next_tasks_and_validation_ladder(self) -> None:
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability")

        plan = build_repo_improvement_plan(profile)

        self.assertIn("Next 3 concrete repo tasks", plan)
        self.assertIn("Agentic capability curriculum", plan)
        self.assertIn("Next agentic eval tasks", plan)
        self.assertIn("PEDS-to-Codex actuator handoff", plan)
        self.assertIn("--claim-codex-task", plan)
        self.assertIn("Validation ladder", plan)
        self.assertIn("Expected useful output", plan)

    def test_build_codex_skill_from_survivor_includes_when_to_use_and_rules(self) -> None:
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability")

        skill = build_codex_skill_from_survivor(profile)

        self.assertIn("When to use", skill)
        self.assertIn("Workflow", skill)
        self.assertIn("Agentic capability curriculum", skill)
        self.assertIn("Validation ladder", skill)
        self.assertIn("Operating rules", skill)
        self.assertIn("Output contract", skill)
        self.assertIn("Do-not-do rules", skill)

    def test_append_active_policy_writes_append_only_jsonl_without_mutating_profile(self) -> None:
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability")
        original = json.loads(json.dumps(profile, sort_keys=True))

        first = append_active_policy(profile, registry_root=self.root)
        second = append_active_policy(profile, registry_root=self.root)
        latest = load_latest_active_policy(self.root)

        lines = (self.root / "active_policy.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(first["active_policy_id"], second["active_policy_id"])
        self.assertEqual(latest["active_policy_id"], second["active_policy_id"])
        self.assertEqual(latest["agentic_capability_coverage"]["status"], "broadening_needed")
        self.assertEqual(profile, original)

    def test_write_autopilot_artifacts_writes_expected_files(self) -> None:
        profile = build_survivor_profile(self._report(), "Improve Codex patch reliability")

        paths = write_autopilot_artifacts(profile, self.root / "artifacts")

        self.assertTrue(Path(paths["survivor_profile"]).exists())
        self.assertTrue(Path(paths["codex_prompt"]).exists())
        self.assertTrue(Path(paths["repo_improvement_plan"]).exists())
        self.assertTrue(Path(paths["skill"]).exists())

    def _report(self) -> dict:
        mutation = {
            "mutation_target": "psi",
            "mutation_seed": "seed_psi_retry",
            "diagnosis_category": "no_generated_child",
            "delta_description": "Retry from last valid parent.",
        }
        return {
            "campaign_id": "campaign_survivor",
            "generations_requested": 2,
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            "aggregate": {
                "total_generations_completed": 2,
                "total_candidate_attempts": 3,
                "total_winning_attempts": 2,
                "total_rejected_attempts": 1,
                "corrections_attempted": 1,
                "corrections_followed_by_winner": 1,
                "correction_success_rate": 1.0,
                "collapse_detected": False,
                "exhausted_generation_count": 0,
            },
            "attempts": [
                {
                    "generation": 1,
                    "attempt_index": 1,
                    "outcome": "winner",
                    "winning_candidate_id": "candidate_winner_1",
                    "generated_child_candidate_id": "candidate_winner_1",
                    "parent_candidate_id": None,
                    "source_candidate_id": "source_1",
                    "mutation": {"mutation_target": "theta", "mutation_seed": "seed_theta", "diagnosis_category": "unknown", "delta_description": "Initial theta."},
                    "metric": {"replay_mean_delta": 0.08, "adapter_loaded": True, "training_trained": True},
                },
                {
                    "generation": 2,
                    "attempt_index": 1,
                    "outcome": "rejected",
                    "winning_candidate_id": None,
                    "generated_child_candidate_id": None,
                    "parent_candidate_id": "candidate_winner_1",
                    "source_candidate_id": "source_rejected",
                    "diagnosis": {"category": "no_generated_child"},
                    "mutation": {"mutation_target": "psi", "mutation_seed": "seed_rejected", "diagnosis_category": "no_generated_child", "delta_description": "Rejected."},
                    "metric": {"replay_mean_delta": None, "adapter_loaded": None, "training_trained": None},
                },
                {
                    "generation": 2,
                    "attempt_index": 2,
                    "outcome": "winner",
                    "winning_candidate_id": "candidate_winner_2",
                    "generated_child_candidate_id": "candidate_winner_2",
                    "parent_candidate_id": "candidate_winner_1",
                    "source_candidate_id": "source_2",
                    "mutation": mutation,
                    "metric": {"replay_mean_delta": 0.1, "adapter_loaded": True, "training_trained": True},
                },
            ],
            "winning_attempts": [
                {"generation": 1, "winning_candidate_id": "candidate_winner_1", "selected_parent_for_next_generation": "candidate_winner_1"},
                {"generation": 2, "winning_candidate_id": "candidate_winner_2", "selected_parent_for_next_generation": "candidate_winner_2", "mutation": mutation},
            ],
            "lineage": [
                {"generation": 1, "winning_candidate_id": "candidate_winner_1", "selected_parent_for_next_generation": "candidate_winner_1"},
                {"generation": 2, "winning_candidate_id": "candidate_lineage_latest", "selected_parent_for_next_generation": "candidate_lineage_latest"},
            ],
            "mutations": [
                {"generation": 2, "attempt_index": 2, "mutation": mutation, "mutation_target": "psi", "mutation_seed": "seed_psi_retry"}
            ],
        }


if __name__ == "__main__":
    unittest.main()
