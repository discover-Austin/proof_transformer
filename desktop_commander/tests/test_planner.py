from __future__ import annotations

from common import DesktopCommanderTestCase
from desktop_commander.planner import TaskPlanner


class PlannerTests(DesktopCommanderTestCase):
    def test_planner_builds_multi_step_plan(self) -> None:
        planner = TaskPlanner()
        payload = planner.draft_plan(
            "open powershell and type Get-ChildItem and search the web for tkinter treeview examples and open folder ."
        )
        actions = [step["action"] for step in payload["steps"]]
        self.assertEqual(actions, ["launch_process", "type_text", "search_web", "open_path"])
        self.assertIn("4 step(s)", payload["summary"])

    def test_planner_falls_back_to_web_search_for_unknown_text(self) -> None:
        planner = TaskPlanner()
        payload = planner.draft_plan("best workflow for desktop execution")
        self.assertEqual(payload["steps"][0]["action"], "search_web")
        self.assertTrue(payload["notes"])
