from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from common import DesktopCommanderTestCase
from desktop_commander.executor import DesktopExecutor


class StubInputController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def focus_window(self, title: str) -> dict:
        self.calls.append(("focus_window", title))
        return {"matched_title": title}

    def type_text(self, text: str, *, interval_ms: int = 0, press_enter: bool = False) -> dict:
        self.calls.append(("type_text", {"text": text, "interval_ms": interval_ms, "press_enter": press_enter}))
        return {"characters": len(text)}

    def send_hotkey(self, keys: list[str]) -> dict:
        self.calls.append(("send_hotkey", keys))
        return {"keys": keys}

    def click(self, *, x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
        self.calls.append(("click", {"x": x, "y": y, "button": button, "clicks": clicks}))
        return {"x": x, "y": y}


class ExecutorTests(DesktopCommanderTestCase):
    def test_execute_plan_runs_file_command_web_and_desktop_steps(self) -> None:
        input_controller = StubInputController()
        opened_urls: list[str] = []
        opened_paths: list[str] = []
        sleep_calls: list[float] = []
        launched: list[list[str]] = []

        executor = DesktopExecutor(
            input_controller=input_controller,
            browser_opener=lambda url: opened_urls.append(url) or True,
            path_opener=lambda path: opened_paths.append(str(path)) or True,
            process_launcher=lambda command, cwd=None, shell=False, creationflags=0: launched.append(command) or SimpleNamespace(pid=321),
            sleep_fn=sleep_calls.append,
            url_fetcher=lambda url, timeout: {"status_code": 200, "content_type": "text/plain", "text": f"fetched:{url}:{timeout}"},
        )

        target = "note.txt"
        result = executor.execute_plan(
            {
                "steps": [
                    {"action": "write_file", "path": target, "text": "hello"},
                    {"action": "read_file", "path": target},
                    {"action": "list_files", "path": ".", "pattern": "*.txt", "recursive": False},
                    {"action": "run_command", "command": [sys.executable, "-c", "print('ok-run')"]},
                    {"action": "launch_process", "command": ["powershell.exe", "-NoExit"]},
                    {"action": "search_web", "query": "desktop commander"},
                    {"action": "fetch_url", "url": "https://example.com"},
                    {"action": "open_path", "path": "."},
                    {"action": "focus_window", "title": "PowerShell"},
                    {"action": "type_text", "text": "dir", "press_enter": True},
                    {"action": "send_hotkey", "keys": ["ctrl", "l"]},
                    {"action": "click", "x": 4, "y": 8},
                    {"action": "wait", "seconds": 0.5},
                ]
            }
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_executed"], 13)
        self.assertEqual(result["results"][1]["result"]["text"], "hello")
        self.assertEqual(result["results"][3]["result"]["status"], "completed")
        self.assertTrue(launched[0][0].lower().endswith("powershell.exe"))
        self.assertEqual(launched[0][1], "-NoExit")
        self.assertEqual(len(opened_urls), 1)
        self.assertIn("desktop+commander", opened_urls[0])
        self.assertIn("fetched:https://example.com:20", result["results"][6]["result"]["text"])
        self.assertEqual(len(opened_paths), 1)
        self.assertEqual(sleep_calls, [0.5])
        self.assertEqual(
            input_controller.calls,
            [
                ("focus_window", "PowerShell"),
                ("type_text", {"text": "dir", "interval_ms": 0, "press_enter": True}),
                ("send_hotkey", ["ctrl", "l"]),
                ("click", {"x": 4, "y": 8, "button": "left", "clicks": 1}),
            ],
        )

    def test_execute_plan_continues_after_error_when_requested(self) -> None:
        executor = DesktopExecutor(
            input_controller=StubInputController(),
            browser_opener=lambda _url: True,
            path_opener=lambda _path: True,
            process_launcher=lambda *args, **kwargs: SimpleNamespace(pid=999),
            sleep_fn=lambda _seconds: None,
            url_fetcher=lambda url, timeout: {"status_code": 200, "content_type": "text/plain", "text": url},
        )
        result = executor.execute_plan(
            {
                "steps": [
                    {"action": "read_file", "path": "missing.txt"},
                    {"action": "write_file", "path": "recovered.txt", "text": "after"},
                ]
            },
            continue_on_error=True,
        )
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps_executed"], 2)
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(result["results"][1]["status"], "completed")

    def test_run_command_failure_is_reported_as_step_failure(self) -> None:
        executor = DesktopExecutor(
            input_controller=StubInputController(),
            browser_opener=lambda _url: True,
            path_opener=lambda _path: True,
            process_launcher=lambda *args, **kwargs: SimpleNamespace(pid=999),
            sleep_fn=lambda _seconds: None,
            url_fetcher=lambda url, timeout: {"status_code": 200, "content_type": "text/plain", "text": url},
        )
        result = executor.execute_plan({"steps": [{"action": "run_command", "command": ["__definitely_missing_executable__"]}]})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(result["results"][0]["result"]["status"], "error")

    def test_resolve_executable_uses_windows_fallback_when_path_lookup_is_missing(self) -> None:
        executor = DesktopExecutor(
            input_controller=StubInputController(),
            browser_opener=lambda _url: True,
            path_opener=lambda _path: True,
            process_launcher=lambda *args, **kwargs: SimpleNamespace(pid=999),
            sleep_fn=lambda _seconds: None,
            url_fetcher=lambda url, timeout: {"status_code": 200, "content_type": "text/plain", "text": url},
        )
        with patch("desktop_commander.executor.shutil.which", return_value=None), patch("desktop_commander.executor.os.name", "nt"):
            fake_env = {
                "SystemRoot": "C:\\Windows",
                "ProgramFiles": "C:\\Program Files",
                "ProgramFiles(x86)": "C:\\Program Files (x86)",
            }
            with patch("desktop_commander.executor.os.environ", fake_env), patch.object(Path, "exists", return_value=True):
                resolved = executor._resolve_executable("powershell.exe")
        self.assertEqual(resolved, "C:\\Windows\\System32\\powershell.exe")
