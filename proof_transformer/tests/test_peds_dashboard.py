from __future__ import annotations

import importlib
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


class PEDSDashboardTests(unittest.TestCase):
    def test_imports_without_launching_mainloop(self) -> None:
        self.assertTrue(hasattr(peds_dashboard, "PEDSDashboard"))
        self.assertTrue(hasattr(peds_dashboard, "create_app"))
        self.assertTrue(hasattr(peds_dashboard, "build_campaign_command"))

    def test_build_campaign_command_builds_default_5x2_command(self) -> None:
        command = peds_dashboard.build_campaign_command(repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_training_campaign.py")
        self.assertEqual(command[command.index("--generations") + 1], "5")
        self.assertEqual(command[command.index("--runs") + 1], "2")
        self.assertIn("--json", command)

    def test_build_campaign_command_includes_max_attempts_per_generation(self) -> None:
        command = peds_dashboard.build_campaign_command(max_attempts_per_generation=7, repo_root=ROOT)

        self.assertIn("--max-attempts-per-generation", command)
        self.assertEqual(command[command.index("--max-attempts-per-generation") + 1], "7")

    def test_build_campaign_command_includes_strict_only_when_requested(self) -> None:
        default = peds_dashboard.build_campaign_command(repo_root=ROOT)
        strict = peds_dashboard.build_campaign_command(strict=True, repo_root=ROOT)

        self.assertNotIn("--strict", default)
        self.assertIn("--strict", strict)

    def test_build_autopilot_command_builds_peds_autopilot_command(self) -> None:
        command = peds_dashboard.build_autopilot_command(objective="Improve Codex patch reliability", repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_autopilot.py")
        self.assertIn("--objective", command)
        self.assertIn("--campaign", command)
        self.assertIn("--write", command)
        self.assertIn("--json", command)

    def test_build_autopilot_command_preserves_objective_as_one_argument(self) -> None:
        objective = "Improve Codex patch reliability in this repo"
        command = peds_dashboard.build_autopilot_command(objective=objective, use_latest=True, repo_root=ROOT)

        self.assertEqual(command[command.index("--objective") + 1], objective)
        self.assertEqual(command.count(objective), 1)
        self.assertIn("--use-latest", command)

    def test_build_watch_autopilot_command_wraps_autopilot_through_watcher(self) -> None:
        objective = "Improve Codex patch reliability in this repo"
        command = peds_dashboard.build_watch_autopilot_command(
            objective=objective,
            use_latest=True,
            status_dir="status",
            repo_root=ROOT,
        )

        self.assertEqual(Path(command[1]).name, "watch_command.py")
        inner = command[command.index("--") + 1 :]
        self.assertEqual(Path(inner[1]).name, "peds_autopilot.py")
        self.assertEqual(inner[inner.index("--objective") + 1], objective)
        self.assertIn("--use-latest", inner)

    def test_build_autopilot_command_rejects_blank_objective(self) -> None:
        with self.assertRaises(ValueError):
            peds_dashboard.build_autopilot_command(objective="   ", repo_root=ROOT)

    def test_build_brain_command_builds_peds_brain_command(self) -> None:
        command = peds_dashboard.build_brain_command("init", objective="Improve Codex patch reliability", max_iterations=5, repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_brain.py")
        self.assertIn("--init", command)
        self.assertEqual(command[command.index("--objective") + 1], "Improve Codex patch reliability")
        self.assertEqual(command[command.index("--max-iterations") + 1], "5")

    def test_build_brain_command_supports_codex_queue_actions(self) -> None:
        claim = peds_dashboard.build_brain_command("claim-codex-task", repo_root=ROOT)
        queue = peds_dashboard.build_brain_command("list-codex-queue", repo_root=ROOT)

        self.assertIn("--claim-codex-task", claim)
        self.assertIn("--list-codex-queue", queue)
        self.assertIn("--json", claim)
        self.assertIn("--json", queue)

    def test_build_brain_command_supports_execute_actuator_flag(self) -> None:
        command = peds_dashboard.build_brain_command("step", execute_actuator=True, repo_root=ROOT)

        self.assertIn("--step", command)
        self.assertIn("--execute-actuator", command)

    def test_build_actuator_command_builds_run_next(self) -> None:
        command = peds_dashboard.build_actuator_command("run-next", provider="manual", repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_actuator.py")
        self.assertIn("--run-next", command)
        self.assertIn("--provider", command)
        self.assertIn("manual", command)
        self.assertIn("--json", command)

    def test_build_actuator_command_builds_providers_refresh(self) -> None:
        command = peds_dashboard.build_actuator_command("providers", repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_actuator.py")
        self.assertIn("--providers", command)
        self.assertIn("--json", command)

    def test_dashboard_source_exposes_refresh_actuator_providers_button(self) -> None:
        source = (ROOT / "scripts" / "peds_dashboard.py").read_text(encoding="utf-8")

        self.assertIn("Refresh Actuator Providers", source)
        self.assertIn("refresh_actuator_providers", source)

    def test_summarize_actuator_provider_status_reports_selected_and_noninteractive(self) -> None:
        summary = peds_dashboard.summarize_actuator_provider_status(
            {
                "selected_provider": "manual",
                "providers": [
                    {"name": "codex", "available": False, "supports_noninteractive": False},
                    {"name": "manual", "available": True, "supports_noninteractive": False},
                ],
            }
        )

        self.assertEqual(summary["selected_provider"], "manual")
        self.assertFalse(summary["noninteractive_available"])
        self.assertIn("manual", summary["available_provider_names"])

    def test_build_brain_record_task_result_command_preserves_selected_path(self) -> None:
        result_path = ROOT / "temp" / "peds_brain" / "task_results" / "result file.md"
        command = peds_dashboard.build_brain_record_task_result_command(result_path, repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "peds_brain.py")
        self.assertIn("--record-task-result", command)
        self.assertEqual(command[command.index("--record-task-result") + 1], str(result_path))
        self.assertEqual(command.count(str(result_path)), 1)

    def test_build_watch_brain_command_wraps_brain_through_watcher(self) -> None:
        command = peds_dashboard.build_watch_brain_command("think", status_dir="status", repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "watch_command.py")
        inner = command[command.index("--") + 1 :]
        self.assertEqual(Path(inner[1]).name, "peds_brain.py")
        self.assertIn("--think", inner)

    def test_load_brain_status_reads_local_state_and_task_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_dir = root / "tasks"
            task_dir.mkdir()
            older = task_dir / "20260606T000000Z_inspect_artifact_quality_codex_task.md"
            newer = task_dir / "20260606T010000Z_improve_trainer_signal_codex_task.md"
            older.write_text("older", encoding="utf-8")
            newer.write_text("newer", encoding="utf-8")
            (root / "brain_state.json").write_text(
                '{"schema_version":"1.0","system":"PEDS","current_objective":"Improve","status":"idle"}',
                encoding="utf-8",
            )
            queue_path = root / "codex_actuator_queue.jsonl"
            queue_rows = [
                {"timestamp": "2026-06-06T00:00:00Z", "queue_id": "codexq_1", "status": "pending", "action": "inspect_artifact_quality"},
                {"timestamp": "2026-06-06T00:01:00Z", "queue_id": "codexq_1", "status": "claimed", "action": "inspect_artifact_quality"},
            ]
            queue_path.write_text("\n".join(json.dumps(row) for row in queue_rows) + "\n", encoding="utf-8")

            status = peds_dashboard.load_brain_status(root)
            task_path = peds_dashboard.latest_brain_task_path(root)

        self.assertTrue(status["found"])
        self.assertEqual(status["system"], "PEDS")
        self.assertEqual(task_path.name, newer.name)
        self.assertEqual(status["codex_queue"][0]["status"], "claimed")

    def test_load_codex_queue_status_returns_latest_event_per_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [
                {"timestamp": "2026-06-06T00:00:00Z", "task_id": "task_a", "queue_id": "task_a", "status": "pending"},
                {"timestamp": "2026-06-06T00:02:00Z", "task_id": "task_b", "queue_id": "task_b", "status": "pending"},
                {"timestamp": "2026-06-06T00:03:00Z", "task_id": "task_a", "queue_id": "task_a", "status": "completed"},
            ]
            (root / "task_queue.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            queue = peds_dashboard.load_codex_queue_status(root)
            groups = peds_dashboard.split_task_queue_by_status(queue)

        by_id = {row["queue_id"]: row for row in queue}
        self.assertEqual(by_id["task_a"]["status"], "completed")
        self.assertEqual(by_id["task_b"]["status"], "pending")
        self.assertEqual(len(groups["pending"]), 1)
        self.assertEqual(len(groups["completed"]), 1)

    def test_use_peds_helper_parses_autopilot_summary_and_prompt_preview(self) -> None:
        summary = peds_dashboard.parse_autopilot_summary(
            {
                "ok": True,
                "objective": "Improve",
                "candidate_id": "candidate_best",
                "useful_output_ready": True,
                "artifacts": {},
                "codex_prompt_preview": "Use this survivor.",
                "recommended_uses": ["Use this survivor on the next repo patch."],
                "agentic_capability_coverage": {
                    "status": "broadening_needed",
                    "pressured_capabilities": ["repo_navigation"],
                    "untested_capabilities": ["ui_or_browser_operation"],
                    "next_eval_tasks": ["Add a local ui/browser eval."],
                    "coverage_warning": "Do not overclaim.",
                },
                "agentic_next_eval_tasks": ["Add a local ui/browser eval."],
                "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            }
        )

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["candidate_id"], "candidate_best")
        self.assertEqual(summary["recommended_uses"][0], "Use this survivor on the next repo patch.")
        self.assertEqual(summary["agentic_capability_coverage"]["status"], "broadening_needed")
        self.assertEqual(summary["agentic_next_eval_tasks"], ["Add a local ui/browser eval."])
        self.assertEqual(peds_dashboard.prompt_preview_from_summary(summary), "Use this survivor.")

    def test_ui_browser_operation_capability_eval_mirrors_brain_without_owning_decisions(self) -> None:
        brain_status = {
            "found": True,
            "system": "PEDS",
            "status": "planning",
            "current_objective": "Improve agentic capability",
            "next_action": {"action": "expand_agentic_capability_curriculum"},
            "last_codex_task_result": {"status": "completed", "ok": True},
            "codex_queue": [{"task_id": "task_1", "status": "completed"}],
        }
        autopilot_summary = {
            "ok": True,
            "agentic_capability_coverage": {
                "status": "broadening_needed",
                "pressured_capabilities": ["repo_navigation", "code_editing"],
                "untested_capabilities": ["ui_or_browser_operation"],
                "next_eval_tasks": ["Verify dashboard mirrors PEDS state."],
            },
        }

        result = peds_dashboard.evaluate_ui_browser_operation_capability(brain_status, autopilot_summary)

        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["decision_owner"], "PEDS brain")
        self.assertTrue(result["mirrors_brain_state"])
        self.assertIn("ui_or_browser_operation", result["untested_capabilities"])

    def test_prompt_preview_helper_handles_missing_prompt(self) -> None:
        preview = peds_dashboard.prompt_preview_from_summary({})

        self.assertIn("No Codex prompt", preview)

    def test_build_validation_commands_target_expected_test_files(self) -> None:
        cases = {
            "mutation": "test_peds_mutation_strategies.py",
            "campaign": "test_peds_training_campaign.py",
            "operator": "test_peds_operator_tools.py",
            "broad_peds": "test_peds_*.py",
        }
        for kind, pattern in cases.items():
            with self.subTest(kind=kind):
                command = peds_dashboard.build_validation_command(kind, repo_root=ROOT)
                self.assertIn("unittest", command)
                self.assertEqual(command[-1], pattern)

    def test_latest_report_command_calls_campaign_report_json(self) -> None:
        command = peds_dashboard.build_latest_campaign_report_command(ROOT)

        self.assertEqual(Path(command[1]).name, "peds_latest_campaign_report.py")
        self.assertEqual(command[-1], "--json")

    def test_build_watch_command_wraps_existing_watch_command(self) -> None:
        watched = [sys.executable, "-m", "unittest"]
        command = peds_dashboard.build_watch_command(name="unit", command=watched, status_dir="status", repo_root=ROOT)

        self.assertEqual(Path(command[1]).name, "watch_command.py")
        self.assertIn("--name", command)
        self.assertIn("--status-dir", command)
        self.assertEqual(command[command.index("--") + 1 :], watched)

    def test_summarize_campaign_report_handles_missing_empty_report(self) -> None:
        summary = peds_dashboard.summarize_campaign_report({})

        self.assertFalse(summary["found"])
        self.assertEqual(summary["total_candidate_attempts"], 0)
        self.assertEqual(summary["total_winning_attempts"], 0)
        self.assertTrue(summary["safety_clean"])

    def test_summarize_campaign_report_includes_attempts_winners_rejections_corrections_and_safety(self) -> None:
        report = self._sample_report()

        summary = peds_dashboard.summarize_campaign_report(report)

        self.assertEqual(summary["campaign_id"], "campaign_demo")
        self.assertEqual(summary["total_candidate_attempts"], 2)
        self.assertEqual(summary["total_winning_attempts"], 1)
        self.assertEqual(summary["total_rejected_attempts"], 1)
        self.assertEqual(summary["corrections_attempted"], 1)
        self.assertEqual(summary["corrections_followed_by_winner"], 1)
        self.assertTrue(summary["safety_clean"])
        self.assertEqual(summary["network_attempts"], [])

    def test_extract_lineage_rows_returns_identity_chain_fields(self) -> None:
        row = peds_dashboard.extract_lineage_rows(self._sample_report())[0]

        self.assertEqual(row["generation"], 1)
        self.assertEqual(row["parent_candidate_id"], "parent_1")
        self.assertEqual(row["source_candidate_id"], "source_1")
        self.assertEqual(row["attempt_candidate_id"], "attempt_1")
        self.assertEqual(row["winning_candidate_id"], "winner_1")
        self.assertEqual(row["selected_parent_for_next_generation"], "winner_1")

    def test_extract_attempt_rows_returns_diagnosis_correction_mutation_and_replay_fields(self) -> None:
        rows = peds_dashboard.extract_attempt_rows(self._sample_report())

        rejected = rows[0]
        winner = rows[1]
        self.assertEqual(rejected["diagnosis_category"], "replay_underperformance")
        self.assertEqual(rejected["correction"], "retry_with_last_valid_parent")
        self.assertEqual(rejected["mutation_target"], "theta")
        self.assertEqual(rejected["mutation_seed"], "seed_rejected")
        self.assertEqual(rejected["replay_mean_delta"], -0.01)
        self.assertFalse(rejected["adapter_loaded"])
        self.assertTrue(winner["training_trained"])

    def test_extract_mutation_rows_returns_target_seed_and_description(self) -> None:
        row = peds_dashboard.extract_mutation_rows(self._sample_report())[0]

        self.assertEqual(row["generation"], 1)
        self.assertEqual(row["attempt_index"], 2)
        self.assertEqual(row["diagnosis_category"], "replay_underperformance")
        self.assertEqual(row["mutation_target"], "theta")
        self.assertEqual(row["mutation_seed"], "seed_retry")
        self.assertEqual(row["delta_description"], "Retry theta with isolated evidence.")

    def test_build_evolution_story_handles_successful_winner_chain(self) -> None:
        report = self._story_report()

        story = "\n".join(peds_dashboard.build_evolution_story(report))

        self.assertIn("Run 1", story)
        self.assertIn("G1 A1", story)
        self.assertIn("winner", story)
        self.assertIn("G2 A1", story)
        self.assertIn("next parent", story)

    def test_build_evolution_story_shows_rejected_attempt_followed_by_corrected_winner(self) -> None:
        story = "\n".join(peds_dashboard.build_evolution_story(self._sample_report()))

        self.assertIn("rejected", story)
        self.assertIn("diagnosis replay_underperformance", story)
        self.assertIn("corrected winner", story)
        self.assertIn("mutation theta", story)

    def test_build_evolution_story_shortens_candidate_ids(self) -> None:
        report = self._story_report()
        full_id = report["attempts"][0]["selected_parent_for_next_generation"]

        story = "\n".join(peds_dashboard.build_evolution_story(report))

        self.assertNotIn(full_id, story)
        self.assertIn(full_id[:12], story)

    def test_summarize_peds_learning_reports_campaign_outcome_and_patterns(self) -> None:
        learning = peds_dashboard.summarize_peds_learning(self._sample_report())

        self.assertIn("completed 1 generations", learning)
        self.assertIn("2 attempts", learning)
        self.assertIn("1 attempts were rejected", learning)
        self.assertIn("1 of 1 corrections", learning)
        self.assertIn("dominant diagnosis was replay_underperformance", learning)
        self.assertIn("dominant mutation was theta", learning)
        self.assertIn("Safety remained clean", learning)

    def test_summarize_peds_learning_identifies_narrow_mutation_diversity(self) -> None:
        learning = peds_dashboard.summarize_peds_learning(self._sample_report())

        self.assertIn("Mutation diversity is still narrow", learning)

    def test_recommend_next_action_with_no_report_recommends_5x2_campaign(self) -> None:
        recommendation = peds_dashboard.recommend_next_action({})
        command = recommendation["command"]

        self.assertEqual(recommendation["command_kind"], "campaign_5x2")
        self.assertIn("5x2", recommendation["label"])
        self.assertEqual(command[command.index("--generations") + 1], "5")
        self.assertEqual(command[command.index("--runs") + 1], "2")

    def test_recommend_next_action_with_safety_issue_does_not_recommend_larger_campaign(self) -> None:
        report = self._sample_report()
        report["safety"] = {"network_attempts": [{"api": "socket.connect"}], "model_modules_loaded": [], "fake_weights_written": False}

        recommendation = peds_dashboard.recommend_next_action(report)

        self.assertEqual(recommendation["command_kind"], "inspect_safety")
        self.assertNotIn("larger", recommendation["label"].lower())
        self.assertNotIn("10x3", recommendation["label"])

    def test_recommend_next_action_with_high_correction_success_recommends_scale_or_mutation_stress(self) -> None:
        recommendation = peds_dashboard.recommend_next_action(self._sample_report())
        command = recommendation["command"]

        self.assertIn(recommendation["command_kind"], {"campaign_10x3", "mutation_stress_campaign"})
        self.assertEqual(command[command.index("--generations") + 1], "10")
        self.assertEqual(command[command.index("--runs") + 1], "3")

    def test_recommend_next_action_with_collapse_recommends_inspection(self) -> None:
        report = self._sample_report()
        report["aggregate"] = dict(report["aggregate"], collapse_detected=True)

        recommendation = peds_dashboard.recommend_next_action(report)

        self.assertEqual(recommendation["command_kind"], "inspect_campaign")
        self.assertIn("Inspect", recommendation["label"])

    def test_recommended_command_can_be_run_through_watch_command(self) -> None:
        recommendation = peds_dashboard.recommend_next_action(self._sample_report())
        watched = peds_dashboard.build_watch_command(name="recommended", command=recommendation["command"], status_dir="status", repo_root=ROOT)

        self.assertEqual(watched[watched.index("--") + 1 :], recommendation["command"])

    def test_overview_parsing_tolerates_missing_and_old_report_shapes(self) -> None:
        status = peds_dashboard.build_brain_status({})
        old_report = {"lineage": [{"generation": 1, "winning_candidate_id": "candidate_old_winner"}]}
        story = peds_dashboard.build_evolution_story(old_report)

        self.assertEqual(status["Current Mode"], "Idle")
        self.assertIn("Recommended Next Action", status)
        self.assertTrue(story)

    def test_opening_missing_report_paths_does_not_crash(self) -> None:
        self.assertFalse(peds_dashboard.open_campaign_folder_from_report({"found": False}))
        self.assertFalse(peds_dashboard.open_campaign_json_from_report({"found": False}))
        self.assertFalse(peds_dashboard.open_report_path(None))

    def test_gui_class_can_be_constructed_when_tk_display_is_available(self) -> None:
        if peds_dashboard.tk is None:
            self.skipTest("tkinter is not available")
        try:
            root = peds_dashboard.tk.Tk()
        except Exception as exc:  # tkinter.TclError when no display is available.
            self.skipTest(f"Tk display is not available: {exc}")
        try:
            root.withdraw()
            dashboard = peds_dashboard.PEDSDashboard(root, repo_root=ROOT, auto_refresh=False)
            self.assertEqual(dashboard.repo_root, ROOT)
        finally:
            root.destroy()

    def _story_report(self) -> dict:
        first = "candidate_abcdefghijklmnop_first_winner"
        second = "candidate_qrstuvwxyzabcdef_second_winner"
        return {
            "found": True,
            "campaign_id": "story_campaign",
            "aggregate": {
                "total_generations_completed": 2,
                "total_candidate_attempts": 2,
                "total_winning_attempts": 2,
                "total_rejected_attempts": 0,
                "corrections_attempted": 0,
                "corrections_followed_by_winner": 0,
                "correction_success_rate": 0.0,
                "collapse_detected": False,
            },
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            "attempts": [
                {
                    "run_index": 1,
                    "generation": 1,
                    "attempt_index": 1,
                    "outcome": "winner",
                    "winning_candidate_id": first,
                    "selected_parent_for_next_generation": first,
                    "mutation": {"mutation_target": "theta", "mutation_seed": "seed_story_1"},
                    "metric": {"replay_mean_delta": 0.1, "adapter_loaded": True, "training_trained": True},
                },
                {
                    "run_index": 1,
                    "generation": 2,
                    "attempt_index": 1,
                    "outcome": "winner",
                    "parent_candidate_id": first,
                    "winning_candidate_id": second,
                    "selected_parent_for_next_generation": second,
                    "mutation": {"mutation_target": "psi", "mutation_seed": "seed_story_2"},
                    "metric": {"replay_mean_delta": 0.08, "adapter_loaded": True, "training_trained": True},
                },
            ],
            "winning_attempts": [
                {"run_index": 1, "generation": 1, "winning_candidate_id": first, "selected_parent_for_next_generation": first},
                {"run_index": 1, "generation": 2, "parent_candidate_id": first, "winning_candidate_id": second, "selected_parent_for_next_generation": second},
            ],
        }

    def _sample_report(self) -> dict:
        mutation_retry = {
            "diagnosis_category": "replay_underperformance",
            "delta_description": "Retry theta with isolated evidence.",
            "mutation_seed": "seed_retry",
            "mutation_target": "theta",
        }
        return {
            "found": True,
            "campaign_id": "campaign_demo",
            "campaign_root": str(ROOT / "temp" / "campaign_demo"),
            "report_path": str(ROOT / "temp" / "campaign_demo" / "campaign_report.json"),
            "generations_requested": 1,
            "aggregate": {
                "adapter_loaded_rate": 0.5,
                "collapse_detected": False,
                "correction_success_rate": 1.0,
                "corrections_attempted": 1,
                "corrections_followed_by_winner": 1,
                "exhausted_generation_count": 0,
                "total_candidate_attempts": 2,
                "total_generations_completed": 1,
                "total_rejected_attempts": 1,
                "total_winning_attempts": 1,
                "trained_rate": 1.0,
            },
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            "corrections": [{"generation": 1, "attempt_index": 1, "action": "retry_with_last_valid_parent"}],
            "winning_attempts": [{"generation": 1, "attempt_index": 2}],
            "rejected_attempts": [{"generation": 1, "attempt_index": 1}],
            "lineage": [
                {
                    "generation": 1,
                    "parent_candidate_id": "parent_1",
                    "source_candidate_id": "source_1",
                    "attempt_candidate_id": "attempt_1",
                    "winning_candidate_id": "winner_1",
                    "generated_child_candidate_id": "winner_1",
                    "selected_parent_for_next_generation": "winner_1",
                    "parent_selection_reason": "winner advances lineage",
                }
            ],
            "attempts": [
                {
                    "generation": 1,
                    "attempt_index": 1,
                    "outcome": "rejected",
                    "diagnosis": {"category": "replay_underperformance"},
                    "correction": {"correction": "retry_with_last_valid_parent"},
                    "mutation": {
                        "mutation_target": "theta",
                        "mutation_seed": "seed_rejected",
                        "delta_description": "Rejected mutation.",
                    },
                    "metric": {
                        "replay_mean_delta": -0.01,
                        "adapter_loaded": False,
                        "training_trained": True,
                    },
                },
                {
                    "generation": 1,
                    "attempt_index": 2,
                    "outcome": "winner",
                    "mutation": mutation_retry,
                    "metric": {
                        "replay_mean_delta": 0.08,
                        "adapter_loaded": True,
                        "training_trained": True,
                    },
                },
            ],
            "mutations": [
                {
                    "generation": 1,
                    "attempt_index": 2,
                    "mutation": mutation_retry,
                    "mutation_target": "theta",
                    "mutation_seed": "seed_retry",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
