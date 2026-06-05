from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

peds_dashboard = importlib.import_module("peds_dashboard")
peds_latest_report = importlib.import_module("peds_latest_report")
watch_command = importlib.import_module("watch_command")


class PEDSOperatorToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_peds_latest_report_handles_missing_run_folder(self) -> None:
        report = peds_latest_report.build_latest_report(self.root / "missing")

        self.assertFalse(report["found"])
        self.assertIn("No PEDS trained-local-generations runs found", report["reason"])

    def test_peds_latest_report_returns_latest_run_when_multiple_run_folders_exist(self) -> None:
        old_run = self._write_sample_report("20260101T000000", generations=1)
        new_run = self._write_sample_report("20260102T000000", generations=2)
        os.utime(old_run, (1, 1))
        os.utime(new_run, (2, 2))

        report = peds_latest_report.build_latest_report(self.root)

        self.assertTrue(report["found"])
        self.assertEqual(report["run_id"], "20260102T000000")
        self.assertEqual(report["generations_completed"], 2)

    def test_peds_latest_report_includes_db_counts_when_brain_sqlite_exists(self) -> None:
        run_root = self._write_sample_report("20260103T000000", generations=1)
        db_path = run_root / "memory" / "brain.sqlite3"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            for table in peds_latest_report.COUNT_TABLES:
                conn.execute(f"CREATE TABLE {table}(id TEXT)")
                conn.execute(f"INSERT INTO {table}(id) VALUES ('one')")
            conn.commit()
        finally:
            conn.close()

        report = peds_latest_report.build_latest_report(self.root)

        self.assertEqual(report["db_counts"]["candidates"], 1)
        self.assertEqual(report["db_counts"]["adapter_registry"], 1)

    def test_peds_latest_report_extracts_generation_replay_deltas_and_statuses(self) -> None:
        self._write_sample_report("20260104T000000", generations=2)

        report = peds_latest_report.build_latest_report(self.root)

        self.assertEqual(
            report["replay_deltas"],
            [
                {"generation": 1, "mean_delta": 0.1},
                {"generation": 2, "mean_delta": 0.2},
            ],
        )
        self.assertEqual(report["replay_decisions"][0]["decision"], "replay_winner")
        self.assertEqual(report["adapter_statuses"][1]["status"], "trained")
        self.assertEqual(report["training_trained"][1]["trained"], True)
        self.assertEqual(report["adapter_loaded"][1]["loaded"], True)
        self.assertTrue(report["passed"])

    def test_watch_command_writes_passed_status(self) -> None:
        status_dir = self.root / "status"

        returncode = watch_command.run_watched_command(
            name="pass",
            command=[sys.executable, "-c", "print('ok')"],
            cwd=self.root,
            status_dir=status_dir,
        )

        status = self._read_status(status_dir / "pass.status.json")
        self.assertEqual(returncode, 0)
        self.assertEqual(status["status"], "passed")
        self.assertEqual(status["returncode"], 0)
        self.assertIn("ok", status["last_stdout_lines"])

    def test_watch_command_writes_failed_status(self) -> None:
        status_dir = self.root / "status"

        returncode = watch_command.run_watched_command(
            name="fail",
            command=[sys.executable, "-c", "import sys; sys.exit(7)"],
            cwd=self.root,
            status_dir=status_dir,
        )

        status = self._read_status(status_dir / "fail.status.json")
        self.assertEqual(returncode, 7)
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["returncode"], 7)

    def test_watch_command_captures_last_stdout_and_stderr_lines(self) -> None:
        status_dir = self.root / "status"
        code = "import sys; print('last-out'); print('last-err', file=sys.stderr)"

        watch_command.run_watched_command(
            name="lines",
            command=[sys.executable, "-c", code],
            cwd=self.root,
            status_dir=status_dir,
        )

        status = self._read_status(status_dir / "lines.status.json")
        self.assertIn("last-out", status["last_stdout_lines"])
        self.assertIn("last-err", status["last_stderr_lines"])

    def test_peds_dashboard_imports_without_launching_mainloop(self) -> None:
        self.assertTrue(hasattr(peds_dashboard, "build_generation_command"))
        self.assertTrue(hasattr(peds_dashboard, "create_app"))

    def test_peds_dashboard_exposes_command_builders(self) -> None:
        one = peds_dashboard.build_generation_command(1, ROOT)
        three = peds_dashboard.build_generation_command(3, ROOT)
        validation = peds_dashboard.build_broad_validation_command(ROOT)

        self.assertIn("peds_trained_local_generations.py", one[1])
        self.assertEqual(one[-1], "1")
        self.assertEqual(three[-1], "3")
        self.assertEqual(validation[-3:], ["-s", "tests", "-p", "test_peds*.py"][-3:])
        self.assertIn("unittest", validation)

    def _write_sample_report(self, run_id: str, *, generations: int) -> Path:
        run_root = self.root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        generation_reports = []
        for generation in range(1, generations + 1):
            generation_reports.append(
                {
                    "generation": generation,
                    "adapter_registry_name": f"acceptance_checklist_theta_gen_{generation:02d}",
                    "adapter_registry_status": "trained",
                    "training_trained": True,
                    "manifest_path": str(run_root / f"gen{generation}" / "distillation_manifest.json"),
                    "output_artifact": str(run_root / f"gen{generation}" / "adapter.json"),
                    "replay_probe": {
                        "adapter_loaded": True,
                        "replay_gate_decision": "replay_winner",
                        "replay_mean_delta": round(generation / 10, 3),
                    },
                }
            )
        payload = {
            "brain_root": str(run_root),
            "fake_weights_written": False,
            "final_candidate_id": f"candidate_{run_id}",
            "generation_reports": generation_reports,
            "generations_completed": generations,
            "model_modules_loaded": [],
            "network_attempts": [],
            "run_id": run_id,
        }
        (run_root / "peds_generation_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return run_root

    def _read_status(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
