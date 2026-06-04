from __future__ import annotations

import sys
from types import SimpleNamespace

from common import BrainTestCase
from proof_transformer.agent.desktop_executor import DesktopTaskExecutor
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


class StubInputController:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def focus_window(self, title: str) -> dict:
        self.calls.append(("focus_window", title))
        return {"matched_title": title, "window_handle": 101}

    def type_text(self, text: str, *, interval_ms: int = 0, press_enter: bool = False) -> dict:
        self.calls.append(
            (
                "type_text",
                {
                    "text": text,
                    "interval_ms": interval_ms,
                    "press_enter": press_enter,
                },
            )
        )
        return {"characters": len(text), "press_enter": press_enter}

    def send_hotkey(self, keys: list[str]) -> dict:
        self.calls.append(("send_hotkey", keys))
        return {"keys": keys}

    def click(self, *, x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
        self.calls.append(("click", {"x": x, "y": y, "button": button, "clicks": clicks}))
        return {"x": x, "y": y, "button": button, "clicks": clicks}


class DesktopTaskExecutorTests(BrainTestCase):
    def test_execute_plan_handles_desktop_terminal_browser_and_file_steps(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        input_controller = StubInputController()
        opened_urls: list[str] = []
        opened_paths: list[str] = []
        launched_processes: list[dict] = []
        sleep_calls: list[float] = []

        def browser_opener(url: str) -> bool:
            opened_urls.append(url)
            return True

        def path_opener(path) -> bool:
            opened_paths.append(str(path))
            return True

        def process_launcher(command, cwd=None, shell=False, creationflags=0):
            launched_processes.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "shell": shell,
                    "creationflags": creationflags,
                }
            )
            return SimpleNamespace(pid=4321)

        executor = DesktopTaskExecutor(
            orchestrator.command_runner,
            input_controller=input_controller,
            browser_opener=browser_opener,
            path_opener=path_opener,
            process_launcher=process_launcher,
            sleep_fn=sleep_calls.append,
        )
        target_path = self.paths.temp / "desktop_note.txt"
        result = executor.execute_plan(
            [
                {"action": "write_file", "path": str(target_path), "text": "brain desktop"},
                {"action": "read_file", "path": str(target_path)},
                {"action": "list_files", "path": str(self.paths.temp), "pattern": "*.txt", "recursive": False},
                {"action": "run_command", "command": [sys.executable, "-c", "print('desktop-ok')"]},
                {"action": "launch_process", "command": ["powershell.exe", "-NoExit"]},
                {"action": "search_web", "query": "proof transformer desktop actions"},
                {"action": "open_path", "path": str(target_path)},
                {"action": "focus_window", "title": "PowerShell"},
                {"action": "type_text", "text": "Get-Location", "press_enter": True},
                {"action": "send_hotkey", "keys": ["ctrl", "l"]},
                {"action": "click", "x": 10, "y": 20},
                {"action": "wait", "seconds": 0.25},
            ]
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps_executed"], 12)
        self.assertEqual(target_path.read_text(encoding="utf-8"), "brain desktop")
        self.assertEqual(result["results"][1]["result"]["text"], "brain desktop")
        self.assertEqual(result["results"][3]["result"]["exit_code"], 0)
        command_stdout_path = result["results"][3]["result"]["stdout_path"]
        self.assertIn("desktop-ok", open(command_stdout_path, encoding="utf-8").read())
        self.assertEqual(launched_processes[0]["command"], ["powershell.exe", "-NoExit"])
        self.assertEqual(len(opened_urls), 1)
        self.assertIn("proof+transformer+desktop+actions", opened_urls[0])
        self.assertEqual(opened_paths, [str(target_path.resolve())])
        self.assertEqual(
            input_controller.calls,
            [
                ("focus_window", "PowerShell"),
                ("type_text", {"text": "Get-Location", "interval_ms": 0, "press_enter": True}),
                ("send_hotkey", ["ctrl", "l"]),
                ("click", {"x": 10, "y": 20, "button": "left", "clicks": 1}),
            ],
        )
        self.assertEqual(sleep_calls, [0.25])

    def test_execute_plan_can_continue_after_errors(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        executor = DesktopTaskExecutor(
            orchestrator.command_runner,
            input_controller=StubInputController(),
            browser_opener=lambda _url: True,
            path_opener=lambda _path: True,
            process_launcher=lambda *args, **kwargs: SimpleNamespace(pid=999),
            sleep_fn=lambda _seconds: None,
        )
        target_path = self.paths.temp / "recovered.txt"
        result = executor.execute_plan(
            [
                {"action": "read_file", "path": str(self.paths.temp / "missing.txt")},
                {"action": "write_file", "path": str(target_path), "text": "after error"},
            ],
            continue_on_error=True,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["steps_executed"], 2)
        self.assertEqual(result["results"][0]["status"], "error")
        self.assertEqual(result["results"][1]["status"], "completed")
        self.assertEqual(target_path.read_text(encoding="utf-8"), "after error")
