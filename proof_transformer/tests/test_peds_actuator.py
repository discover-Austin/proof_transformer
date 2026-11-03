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
from proof_transformer.peds.brain import (  # noqa: E402
    derive_current_task_queue,
    generate_codex_task,
    initialize_brain,
    observe_latest_evidence,
)


class PEDSActuatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        initialize_brain(objective="Improve Codex patch reliability", root=self.root)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_status_json(self) -> None:
        payload = self._run_cli("--status")

        self.assertTrue(payload["ok"])
        self.assertIn("queue_by_state", payload)
        self.assertIn("codex_support", payload)
        self.assertIn("actuator_providers", payload)

    def test_claim_next_json(self) -> None:
        generated = self._generate_task()

        payload = self._run_cli("--claim-next")

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["task"]["task_path"], generated["task_path"])
        self.assertEqual(derive_current_task_queue(root=self.root)[0]["status"], "claimed")

    def test_run_next_graceful_fallback_when_codex_unsupported(self) -> None:
        self._generate_task()
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "peds_actuator.py"), "--run-next", "--root", str(self.root), "--json"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC), "PATH": ""},
        )

        payload = json.loads(completed.stdout)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["actuator_result"]["status"], "manual_actuator_required")
        self.assertEqual(payload["actuator_result"]["actuator"], "manual")
        self.assertEqual(payload["actuator_result"]["provider"], "manual")
        self.assertFalse(payload["actuator_result"]["ok"])
        self.assertIn("manual_command", payload["actuator_result"])
        self.assertEqual(derive_current_task_queue(root=self.root)[0]["status"], "manual_actuator_required")

    def test_record_result_json_updates_queue(self) -> None:
        generated = self._generate_task()
        result_path = self.root / "result.json"
        result_path.write_text(json.dumps({"task_id": generated["task_id"], "task_path": generated["task_path"], "status": "completed", "ok": True, "summary": "done"}), encoding="utf-8")

        payload = self._run_cli("--record-result", str(result_path))

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["task_result"]["status"], "completed")
        self.assertEqual(derive_current_task_queue(root=self.root)[0]["status"], "completed")

    def test_stdout_stderr_paths_recorded_when_provider_runs(self) -> None:
        self._generate_task()

        def fake_run_actuator_task(*, task_path, provider, timeout_seconds, output_dir):  # type: ignore[no-untyped-def]
            stdout_path = Path(output_dir) / "stdout.txt"
            stderr_path = Path(output_dir) / "stderr.txt"
            result_path = Path(output_dir) / "actuator_result.json"
            stdout_path.write_text("provider stdout", encoding="utf-8")
            stderr_path.write_text("provider stderr", encoding="utf-8")
            result = {
                "task_path": str(task_path),
                "status": "completed",
                "ok": True,
                "provider": "codex",
                "provider_available": True,
                "provider_supports_noninteractive": True,
                "actuator": "codex",
                "command": ["codex", "exec", "task"],
                "returncode": 0,
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "summary": "done",
                "files_changed": [],
                "validation": [],
                "issues_found": [],
                "issues_fixed": [],
                "manual_command": None,
                "probe_summary": {"selected_provider": "codex", "providers": []},
                "result_path": str(result_path),
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            return result

        with mock.patch.object(peds_actuator, "run_actuator_task", side_effect=fake_run_actuator_task):
            payload = peds_actuator.run_next_task(root=self.root, project_root=ROOT)

        result = payload["actuator_result"]
        self.assertTrue(payload["ok"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["provider"], "codex")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(Path(result["stdout_path"]).exists())
        self.assertTrue(Path(result["stderr_path"]).exists())
        self.assertIn("provider stdout", Path(result["stdout_path"]).read_text(encoding="utf-8"))

    def test_no_shell_true_in_actuator_source(self) -> None:
        source = (ROOT / "scripts" / "peds_actuator.py").read_text(encoding="utf-8")

        self.assertNotIn("shell=True", source)

    def _run_cli(self, *args: str) -> dict:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "peds_actuator.py"), *args, "--root", str(self.root), "--json"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONPATH": str(SRC)},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
        return json.loads(completed.stdout)

    def _generate_task(self) -> dict:
        decision = {"action": "inspect_artifact_quality", "reason": "Artifacts are unknown.", "hypothesis_kind": "artifact_quality"}
        evidence = observe_latest_evidence(
            campaign_report={"found": False, "reason": "unit"},
            autopilot_summary={"ok": False, "error": "unit"},
            root=self.root,
        )
        return generate_codex_task(decision=decision, evidence=evidence, root=self.root)


if __name__ == "__main__":
    unittest.main()
