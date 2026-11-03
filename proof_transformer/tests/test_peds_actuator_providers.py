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

    def test_build_command_returns_list_and_source_has_no_shell_true(self) -> None:
        task_path = self.root / "task.md"
        task_path.write_text("# task", encoding="utf-8")

        command = actuator_providers.build_actuator_command(provider="codex", task_path=task_path)
        source = (ROOT / "src" / "proof_transformer" / "peds" / "actuator_providers.py").read_text(encoding="utf-8")

        self.assertIsInstance(command, list)
        self.assertNotIn("shell=True", source)

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
