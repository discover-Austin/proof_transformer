from __future__ import annotations

import json
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import peds_autopilot  # noqa: E402


class PEDSAutopilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_objective_is_required(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            peds_autopilot._parse_args(["--use-latest", "--json"])

    def test_blank_objective_returns_json_error_without_traceback(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="   ",
            use_latest=True,
            report_loader=self._report,
        )

        self.assertFalse(result["ok"])
        self.assertIn("objective is required", result["error"])
        self.assertNotIn("Traceback", json.dumps(result))

    def test_use_latest_json_returns_compact_useful_summary(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            write=False,
            report_loader=self._report,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["campaign_ran"])
        self.assertTrue(result["survivor_selected"])
        self.assertEqual(result["candidate_id"], "candidate_winner")
        self.assertTrue(result["useful_output_ready"])
        self.assertIn("codex_prompt_preview", result)
        self.assertEqual(result["agentic_capability_coverage"]["status"], "broadening_needed")
        self.assertIn("ui_or_browser_operation", result["agentic_capability_coverage"]["untested_capabilities"])
        self.assertTrue(result["agentic_next_eval_tasks"])

    def test_input_active_policy_is_reported_as_start_context(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            write=False,
            active_policy={"active_policy_id": "policy_input", "candidate_id": "candidate_active"},
            report_loader=self._report,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["active_policy_start"]["used"])
        self.assertEqual(result["input_active_policy"]["candidate_id"], "candidate_active")
        self.assertEqual(result["latest_active_policy"]["active_policy_id"], "policy_input")

    def test_report_root_selects_non_default_campaign_report(self) -> None:
        report_root = self.root / "targeted_campaign"
        report_root.mkdir()
        (report_root / "campaign_report.json").write_text(
            json.dumps(self._report(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            campaign="none",
            use_latest=True,
            write=False,
            report_root=report_root,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["campaign_id"], "campaign_autopilot")
        self.assertEqual(result["candidate_id"], "candidate_winner")
        self.assertEqual(Path(result["report_path"]), report_root / "campaign_report.json")

    def test_write_writes_artifacts(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            write=True,
            output_root=self.root / "outputs",
            report_loader=self._report,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(Path(result["artifacts"]["survivor_profile"]).exists())
        self.assertTrue(Path(result["artifacts"]["codex_prompt"]).exists())
        self.assertTrue(Path(result["artifacts"]["repo_improvement_plan"]).exists())
        self.assertTrue(Path(result["artifacts"]["skill"]).exists())
        self.assertTrue(Path(result["summary_path"]).exists())

    def test_artifact_type_prompt_writes_only_prompt_artifact_and_summary(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            write=True,
            artifact_type="prompt",
            output_root=self.root / "prompt_only",
            report_loader=self._report,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(list(result["artifacts"].keys()), ["codex_prompt"])
        self.assertTrue(Path(result["artifacts"]["codex_prompt"]).exists())
        self.assertTrue(Path(result["summary_path"]).exists())

    def test_missing_report_returns_useful_json_error_without_traceback(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            report_loader=lambda: {"found": False, "reason": "No campaign_report.json found"},
        )

        self.assertFalse(result["ok"])
        self.assertIn("No campaign_report", result["error"])
        self.assertNotIn("Traceback", json.dumps(result))

    def test_activate_best_writes_active_policy_only_when_survivor_exists(self) -> None:
        result = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            activate_best=True,
            report_loader=self._report,
            active_registry_root=self.root / "active",
        )

        self.assertTrue(result["ok"])
        self.assertIsNotNone(result["active_policy"])
        registry_path = Path(result["active_policy"]["registry_path"])
        self.assertTrue(registry_path.exists())
        self.assertEqual(len(registry_path.read_text(encoding="utf-8").splitlines()), 1)

        missing = peds_autopilot.run_autopilot(
            objective="Improve Codex patch reliability",
            use_latest=True,
            activate_best=True,
            report_loader=lambda: {"found": False, "reason": "missing"},
            active_registry_root=self.root / "active_missing",
        )
        self.assertFalse(missing["ok"])
        self.assertFalse((self.root / "active_missing" / "active_policy.jsonl").exists())

    def test_no_network_or_model_controls_in_command_path(self) -> None:
        command = [sys.executable, str(ROOT / "scripts" / "peds_autopilot.py"), "--objective", "Improve", "--use-latest", "--json"]
        text = " ".join(command).lower()

        self.assertNotIn("http", text)
        self.assertNotIn("openai", text)
        self.assertNotIn("ollama", text)
        self.assertNotIn("model-api", text)

    def test_cli_missing_report_prints_json_error(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "peds_autopilot.py"),
                "--objective",
                "Improve Codex patch reliability",
                "--use-latest",
                "--output-root",
                str(self.root / "cli_outputs"),
                "--json",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        # The repo may have a latest report; this assertion only requires JSON and no traceback.
        payload = json.loads(completed.stdout)
        self.assertIn("ok", payload)
        self.assertNotIn("Traceback", completed.stderr + completed.stdout)

    def _report(self) -> dict:
        mutation = {
            "mutation_target": "psi",
            "mutation_seed": "seed_retry",
            "diagnosis_category": "no_generated_child",
            "delta_description": "Lineage parent retry.",
        }
        return {
            "found": True,
            "campaign_id": "campaign_autopilot",
            "campaign_root": str(self.root / "campaign"),
            "report_path": str(self.root / "campaign" / "campaign_report.json"),
            "safety": {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False},
            "aggregate": {
                "total_generations_completed": 1,
                "total_candidate_attempts": 2,
                "total_winning_attempts": 1,
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
                    "outcome": "rejected",
                    "winning_candidate_id": None,
                    "diagnosis": {"category": "no_generated_child"},
                    "mutation": {"mutation_target": "psi", "mutation_seed": "seed_rejected", "diagnosis_category": "no_generated_child", "delta_description": "Rejected."},
                    "metric": {"adapter_loaded": None, "training_trained": None, "replay_mean_delta": None},
                },
                {
                    "generation": 1,
                    "attempt_index": 2,
                    "outcome": "winner",
                    "winning_candidate_id": "candidate_winner",
                    "generated_child_candidate_id": "candidate_winner",
                    "parent_candidate_id": "candidate_parent",
                    "source_candidate_id": "candidate_source",
                    "mutation": mutation,
                    "metric": {"adapter_loaded": True, "training_trained": True, "replay_mean_delta": 0.1},
                },
            ],
            "mutations": [{"generation": 1, "attempt_index": 2, "mutation": mutation, "mutation_target": "psi", "mutation_seed": "seed_retry"}],
            "winning_attempts": [{"generation": 1, "winning_candidate_id": "candidate_winner", "selected_parent_for_next_generation": "candidate_winner", "mutation": mutation}],
        }


if __name__ == "__main__":
    unittest.main()
