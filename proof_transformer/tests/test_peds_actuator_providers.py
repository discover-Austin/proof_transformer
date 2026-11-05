from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import peds_actuator  # noqa: E402
from proof_transformer.peds import actuator_providers  # noqa: E402
from proof_transformer.peds.brain import derive_current_task_queue, generate_codex_task, initialize_brain, observe_latest_evidence  # noqa: E402


class PEDSActuatorProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        initialize_brain(objective="Improve provider handling", root=self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_manual_provider_always_available(self) -> None:
        detected = actuator_providers.detect_actuator_providers()

        manual = next(item for item in detected["providers"] if item["name"] == "manual")
        self.assertTrue(manual["available"])
        self.assertFalse(manual["supports_noninteractive"])

    def test_unavailable_codex_and_pi_report_without_exception(self) -> None:
        with mock.patch.object(actuator_providers.subprocess, "run", side_effect=FileNotFoundError):
            detected = actuator_providers.detect_actuator_providers()

        by_name = {item["name"]: item for item in detected["providers"]}
        self.assertFalse(by_name["codex"]["available"])
        self.assertFalse(by_name["pi"]["available"])
        self.assertEqual(detected["selected_provider"], "manual")

    def test_auto_falls_back_to_manual_when_no_noninteractive_provider_exists(self) -> None:
        task_path = self.root / "task.md"
        task_path.write_text("# task", encoding="utf-8")
        with mock.patch.object(actuator_providers.subprocess, "run", side_effect=FileNotFoundError):
            result = actuator_providers.run_actuator_task(task_path=task_path, provider="auto", output_dir=self.root / "out")

        self.assertEqual(result["provider"], "manual")
        self.assertEqual(result["status"], "manual_actuator_required")
        self.assertFalse(result["ok"])


    def test_codex_command_uses_node_codex_js_on_windows(self) -> None:
        npm_dir = self.root / "npm"
        codex_js = npm_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        codex_js.parent.mkdir(parents=True)
        codex_js.write_text("// codex", encoding="utf-8")
        shim = npm_dir / "codex.cmd"

        def fake_which(name: str) -> str | None:
            if name == "codex":
                return str(shim)
            if name == "node":
                return "node.exe"
            return None

        with (
            mock.patch.object(actuator_providers.sys, "platform", "win32"),
            mock.patch.object(actuator_providers.shutil, "which", side_effect=fake_which),
        ):
            command = actuator_providers._codex_command()

        self.assertEqual(command, ["node.exe", str(codex_js)])

    def test_codex_probe_runs_node_codex_js_not_the_cmd_shim(self) -> None:
        npm_dir = self.root / "npm"
        codex_js = npm_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        codex_js.parent.mkdir(parents=True)
        codex_js.write_text("// codex", encoding="utf-8")
        shim = npm_dir / "codex.cmd"

        def fake_which(name: str) -> str | None:
            if name == "codex":
                return str(shim)
            if name == "node":
                return "node.exe"
            return None

        probe = mock.Mock(returncode=0, stdout="", stderr="")
        with (
            mock.patch.object(actuator_providers.sys, "platform", "win32"),
            mock.patch.object(actuator_providers.shutil, "which", side_effect=fake_which),
            mock.patch.object(actuator_providers.subprocess, "run", return_value=probe),
        ):
            detected = actuator_providers.detect_actuator_providers()

        by_name = {item["name"]: item for item in detected["providers"]}
        self.assertTrue(by_name["codex"]["available"])
        self.assertTrue(by_name["codex"]["supports_noninteractive"])
        self.assertEqual(by_name["codex"]["probe_commands"][0], ["node.exe", str(codex_js), "--help"])
        self.assertEqual(detected["selected_provider"], "codex")

    def test_build_command_returns_list_and_source_has_no_shell_true(self) -> None:
        task_path = self.root / "task.md"
        task_path.write_text("# task", encoding="utf-8")

        command = actuator_providers.build_actuator_command(provider="codex", task_path=task_path)
        source = (ROOT / "src" / "proof_transformer" / "peds" / "actuator_providers.py").read_text(encoding="utf-8")

        self.assertIsInstance(command, list)
        self.assertNotIn("shell=True", source)


    def _codex_detection(self) -> dict:
        return {
            "providers": [
                {
                    "name": "codex",
                    "available": True,
                    "supports_noninteractive": True,
                    "probe_commands": [],
                    "reason": None,
                },
                {
                    "name": "manual",
                    "available": True,
                    "supports_noninteractive": False,
                    "probe_commands": [],
                    "reason": None,
                },
            ],
            "selected_provider": "codex",
        }

    def test_codex_build_command_uses_stdin_marker_not_task_text(self) -> None:
        task_text = "# big task " + ("x" * 60000)
        task_path = self.root / "task.md"
        task_path.write_text(task_text, encoding="utf-8")

        command = actuator_providers.build_actuator_command(provider="codex", task_path=task_path)

        self.assertIsInstance(command, list)
        self.assertEqual(command[-2:], ["exec", "-"])
        self.assertNotIn(task_text, command)
        for element in command:
            self.assertNotIn("big task", element)

    def test_codex_run_passes_task_text_via_stdin_input(self) -> None:
        task_text = "# task via stdin"
        task_path = self.root / "task.md"
        task_path.write_text(task_text, encoding="utf-8")
        completed = mock.Mock(returncode=0, stdout="done", stderr="")

        with (
            mock.patch.object(actuator_providers, "detect_actuator_providers", return_value=self._codex_detection()),
            mock.patch.object(actuator_providers.subprocess, "run", return_value=completed) as mock_run,
        ):
            result = actuator_providers.run_actuator_task(task_path=task_path, provider="codex", output_dir=self.root / "out")

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["ok"])
        self.assertEqual(result["command"][-2:], ["exec", "-"])
        self.assertNotIn(task_text, result["command"])
        self.assertEqual(mock_run.call_count, 1)
        self.assertEqual(mock_run.call_args.kwargs.get("input"), task_text)
        self.assertNotIn(task_text, mock_run.call_args.args[0])

    def test_run_actuator_task_records_failed_result_on_oserror(self) -> None:
        task_path = self.root / "task.md"
        task_path.write_text("# task", encoding="utf-8")

        with (
            mock.patch.object(actuator_providers, "detect_actuator_providers", return_value=self._codex_detection()),
            mock.patch.object(actuator_providers.subprocess, "run", side_effect=OSError("[WinError 206] The filename or extension is too long")),
        ):
            result = actuator_providers.run_actuator_task(task_path=task_path, provider="codex", output_dir=self.root / "out")

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["ok"])
        self.assertIn("WinError 206", result["summary"])
        self.assertIn("codex_spawn_failed", result["issues_found"])
        recorded = json.loads(Path(result["result_path"]).read_text(encoding="utf-8"))
        self.assertEqual(recorded["status"], "failed")

    def test_manual_provider_returns_manual_required_and_probe_summary(self) -> None:
        task_path = self.root / "task.md"
        task_path.write_text("# task", encoding="utf-8")

        result = actuator_providers.run_actuator_task(task_path=task_path, provider="manual", output_dir=self.root / "out")

        self.assertEqual(result["status"], "manual_actuator_required")
        self.assertFalse(result["ok"])
        self.assertIn("manual_command", result)
        self.assertIn("probe_summary", result)

    def test_cli_providers_returns_json(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "peds_actuator.py"), "--providers", "--json"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(completed.returncode, 0)
        self.assertTrue(payload["ok"])
        self.assertIn("selected_provider", payload)
        self.assertIn("providers", payload)

    def test_run_next_provider_manual_records_manual_required_result(self) -> None:
        generated = self._generate_task()
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "peds_actuator.py"), "--run-next", "--provider", "manual", "--root", str(self.root), "--json"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )

        payload = json.loads(completed.stdout)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["actuator_result"]["status"], "manual_actuator_required")
        self.assertEqual(payload["actuator_result"]["provider"], "manual")
        self.assertFalse(payload["actuator_result"]["ok"])
        self.assertIn("probe_summary", payload["actuator_result"])
        self.assertIn("manual_command", payload["actuator_result"])
        queue = derive_current_task_queue(root=self.root)
        self.assertEqual(queue[0]["task_id"], generated["task_id"])
        self.assertEqual(queue[0]["status"], "manual_actuator_required")

    def _generate_task(self) -> dict:
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(campaign_report={"found": False}, autopilot_summary={"ok": False}, root=self.root)
        return generate_codex_task(decision=decision, evidence=evidence, root=self.root)


if __name__ == "__main__":
    unittest.main()
