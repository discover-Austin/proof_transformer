from __future__ import annotations

import sys

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


class RepairPlanningTests(BrainTestCase):
    def test_plan_repair_detects_suspected_workspace_file(self) -> None:
        workspace = self.paths.temp / "repair_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        broken = workspace / "broken.py"
        broken.write_text(
            "def run():\n    return missing_name\n\nrun()\n",
            encoding="utf-8",
        )
        orchestrator = ProofTransformerOrchestrator(self.conn)
        plan = orchestrator.plan_repair(workspace, [sys.executable, "broken.py"])
        self.assertEqual(plan["status"], "planned")
        self.assertIn("broken.py", plan["plan"]["suspected_files"])
        self.assertEqual(plan["plan"]["edits"], [])
        self.assertIn("repair_workspace", plan["workspace_root"])

    def test_autonomous_repair_returns_review_needed_when_no_safe_patch(self) -> None:
        workspace = self.paths.temp / "auto_repair_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        broken = workspace / "broken.py"
        broken.write_text(
            "def run():\n    return missing_name\n\nrun()\n",
            encoding="utf-8",
        )
        orchestrator = ProofTransformerOrchestrator(self.conn)
        result = orchestrator.autonomous_repair(workspace, [sys.executable, "broken.py"], max_iterations=1)
        self.assertEqual(result["status"], "needs_stronger_model_or_manual_review")
        self.assertTrue(any(item["stage"] == "planned" for item in result["history"]))
