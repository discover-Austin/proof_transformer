from __future__ import annotations

import json
from urllib.request import Request, urlopen

from common import DesktopCommanderTestCase
from desktop_commander.api import DesktopCommanderAPIServer
from desktop_commander.executor import DesktopExecutor
from desktop_commander.history import HistoryStore
from desktop_commander.planner import TaskPlanner


class HistoryAndAPITests(DesktopCommanderTestCase):
    def test_history_store_saves_runs_and_presets(self) -> None:
        history = HistoryStore()
        record = history.append_run(task_text="search", plan={"steps": [{"action": "search_web", "query": "x"}]}, result={"status": "completed"})
        self.assertTrue(record["id"].startswith("run_"))
        self.assertEqual(len(history.list_runs(limit=5)), 1)
        preset = history.save_preset(name="Search", task_text="search", plan={"steps": [{"action": "search_web", "query": "x"}]})
        self.assertEqual(preset["name"], "Search")
        presets = history.list_presets()
        self.assertTrue(any(item["name"] == "Search" for item in presets))
        draft = history.save_draft(task_text="search", plan={"steps": [{"action": "search_web", "query": "x"}]})
        self.assertEqual(draft["task_text"], "search")
        self.assertEqual(history.load_draft()["plan"]["steps"][0]["action"], "search_web")

    def test_api_server_exposes_draft_execute_and_presets(self) -> None:
        executor = DesktopExecutor(
            input_controller=None,
            browser_opener=lambda _url: True,
            path_opener=lambda _path: True,
            process_launcher=lambda *args, **kwargs: None,
            sleep_fn=lambda _seconds: None,
            url_fetcher=lambda url, timeout: {"status_code": 200, "content_type": "text/plain", "text": url},
        )
        planner = TaskPlanner()
        history = HistoryStore()
        server = DesktopCommanderAPIServer(host="127.0.0.1", port=0, executor=executor, planner=planner, history=history)
        server.start_in_thread()
        host, port = server.address
        try:
            with urlopen(f"http://{host}:{port}/capabilities", timeout=10) as response:
                capabilities = json.loads(response.read().decode("utf-8"))
            self.assertIn("search_web", capabilities["actions"])

            draft_body = json.dumps({"task_text": "open powershell and search the web for python"}).encode("utf-8")
            draft_request = Request(
                f"http://{host}:{port}/draft-plan",
                data=draft_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(draft_request, timeout=10) as response:
                draft = json.loads(response.read().decode("utf-8"))
            self.assertTrue(draft["steps"])

            execute_body = json.dumps({"task_text": "search", "steps": [{"action": "search_web", "query": "python"}]}).encode("utf-8")
            execute_request = Request(
                f"http://{host}:{port}/execute",
                data=execute_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(execute_request, timeout=10) as response:
                execution = json.loads(response.read().decode("utf-8"))
            self.assertEqual(execution["status"], "completed")
            self.assertEqual(len(history.list_runs(limit=5)), 1)

            preset_body = json.dumps(
                {"name": "Search preset", "task_text": "search", "steps": [{"action": "search_web", "query": "python"}]}
            ).encode("utf-8")
            preset_request = Request(
                f"http://{host}:{port}/save-preset",
                data=preset_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(preset_request, timeout=10) as response:
                preset = json.loads(response.read().decode("utf-8"))
            self.assertEqual(preset["name"], "Search preset")
        finally:
            server.shutdown()
