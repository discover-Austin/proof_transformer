from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Callable
from urllib.parse import quote_plus

from .command_runner import CommandRunner

DEFAULT_SEARCH_URL_TEMPLATE = "https://duckduckgo.com/?q={query}"


class DesktopTaskExecutor:
    def __init__(
        self,
        command_runner: CommandRunner,
        *,
        input_controller: WindowsDesktopInputController | None = None,
        browser_opener: Callable[[str], bool] | None = None,
        path_opener: Callable[[Path], bool] | None = None,
        process_launcher: Callable[..., object] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.command_runner = command_runner
        self.input_controller = input_controller or self._build_default_input_controller()
        self.browser_opener = browser_opener or webbrowser.open
        self.path_opener = path_opener or self._default_open_path
        self.process_launcher = process_launcher or subprocess.Popen
        self.sleep_fn = sleep_fn or time.sleep

    def capabilities(self) -> dict:
        return {
            "platform": sys.platform,
            "search_url_template": DEFAULT_SEARCH_URL_TEMPLATE,
            "actions": [
                "run_command",
                "launch_process",
                "open_url",
                "search_web",
                "open_path",
                "read_file",
                "write_file",
                "list_files",
                "focus_window",
                "type_text",
                "send_hotkey",
                "click",
                "wait",
            ],
            "notes": [
                "Window focus, typing, hotkeys, and clicking require Windows.",
                "run_command captures logs and persists execution metadata.",
                "launch_process is intended for visible apps like PowerShell or a browser.",
            ],
        }

    def execute_plan(self, steps: list[dict], *, continue_on_error: bool = False) -> dict:
        if not isinstance(steps, list) or not steps:
            raise ValueError("Desktop plan must be a non-empty list of step objects.")
        results: list[dict] = []
        status = "completed"
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                raise ValueError(f"Step {index} must be an object.")
            action = str(step.get("action") or "").strip()
            if not action:
                raise ValueError(f"Step {index} is missing an action.")
            try:
                result = self.execute_step(step)
                step_result = {
                    "step_index": index,
                    "step_id": step.get("id"),
                    "action": action,
                    "status": "completed",
                    "result": result,
                }
            except Exception as exc:
                status = "failed"
                step_result = {
                    "step_index": index,
                    "step_id": step.get("id"),
                    "action": action,
                    "status": "error",
                    "error": str(exc),
                }
                if not continue_on_error:
                    results.append(step_result)
                    return {
                        "status": status,
                        "steps_executed": len(results),
                        "continue_on_error": continue_on_error,
                        "results": results,
                    }
            results.append(step_result)
        return {
            "status": status,
            "steps_executed": len(results),
            "continue_on_error": continue_on_error,
            "results": results,
        }

    def execute_step(self, step: dict) -> dict:
        action = str(step.get("action") or "").strip()
        if action == "run_command":
            command = step.get("command")
            if not command:
                raise ValueError("run_command requires a command.")
            return self.command_runner.run(
                command,
                cwd=step.get("cwd"),
                timeout=step.get("timeout"),
                ingest_output=bool(step.get("ingest_output", True)),
            )
        if action == "launch_process":
            command = step.get("command")
            if not command:
                raise ValueError("launch_process requires a command.")
            return self._launch_process(
                command,
                cwd=step.get("cwd"),
                new_console=bool(step.get("new_console", True)),
            )
        if action == "open_url":
            url = str(step.get("url") or "").strip()
            if not url:
                raise ValueError("open_url requires a url.")
            return {"url": url, "opened": bool(self.browser_opener(url))}
        if action == "search_web":
            query = str(step.get("query") or "").strip()
            if not query:
                raise ValueError("search_web requires a query.")
            template = str(step.get("search_url_template") or DEFAULT_SEARCH_URL_TEMPLATE)
            url = template.format(query=quote_plus(query))
            return {"query": query, "url": url, "opened": bool(self.browser_opener(url))}
        if action == "open_path":
            raw_path = step.get("path")
            if not raw_path:
                raise ValueError("open_path requires a path.")
            path = Path(raw_path).expanduser().resolve()
            return {"path": str(path), "opened": bool(self.path_opener(path))}
        if action == "read_file":
            return self._read_file(step)
        if action == "write_file":
            return self._write_file(step)
        if action == "list_files":
            return self._list_files(step)
        if action == "focus_window":
            title = str(step.get("title") or "").strip()
            if not title:
                raise ValueError("focus_window requires a title.")
            return self.input_controller.focus_window(title)
        if action == "type_text":
            text = str(step.get("text") or "")
            return self.input_controller.type_text(
                text,
                interval_ms=int(step.get("interval_ms", 0)),
                press_enter=bool(step.get("press_enter", False)),
            )
        if action == "send_hotkey":
            keys = step.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError("send_hotkey requires a non-empty keys list.")
            return self.input_controller.send_hotkey([str(item) for item in keys])
        if action == "click":
            return self.input_controller.click(
                x=int(step["x"]),
                y=int(step["y"]),
                button=str(step.get("button", "left")),
                clicks=int(step.get("clicks", 1)),
            )
        if action == "wait":
            seconds = float(step.get("seconds", 0))
            if seconds < 0:
                raise ValueError("wait seconds must be >= 0.")
            self.sleep_fn(seconds)
            return {"seconds": seconds}
        raise ValueError(f"Unsupported desktop action: {action}")

    def _launch_process(self, command: list[str] | str, *, cwd: str | None = None, new_console: bool = True) -> dict:
        command_list = command if isinstance(command, list) else [command]
        if not command_list:
            raise ValueError("launch_process command cannot be empty.")
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" and new_console else 0
        process = self.process_launcher(
            command_list,
            cwd=cwd,
            shell=False,
            creationflags=creationflags,
        )
        return {
            "pid": int(process.pid),
            "command": command_list,
            "cwd": cwd,
            "new_console": new_console,
        }

    def _read_file(self, step: dict) -> dict:
        raw_path = step.get("path")
        if not raw_path:
            raise ValueError("read_file requires a path.")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        max_bytes = int(step.get("max_bytes", 200_000))
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0.")
        data = path.read_bytes()
        sample = data[:max_bytes]
        encoding = str(step.get("encoding", "utf-8"))
        return {
            "path": str(path),
            "size_bytes": len(data),
            "returned_bytes": len(sample),
            "truncated": len(data) > len(sample),
            "text": sample.decode(encoding, errors="replace"),
        }

    def _write_file(self, step: dict) -> dict:
        raw_path = step.get("path")
        if not raw_path:
            raise ValueError("write_file requires a path.")
        path = Path(raw_path).expanduser().resolve()
        create_parent_dirs = bool(step.get("create_parent_dirs", True))
        if create_parent_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        text = str(step.get("text") or "")
        encoding = str(step.get("encoding", "utf-8"))
        mode = str(step.get("mode", "overwrite"))
        if mode not in {"overwrite", "append"}:
            raise ValueError("write_file mode must be overwrite or append.")
        if mode == "append":
            with path.open("a", encoding=encoding) as handle:
                handle.write(text)
        else:
            path.write_text(text, encoding=encoding)
        return {
            "path": str(path),
            "mode": mode,
            "bytes_written": len(text.encode(encoding)),
        }

    def _list_files(self, step: dict) -> dict:
        root_value = step.get("path") or step.get("root") or "."
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        pattern = str(step.get("pattern", "*"))
        recursive = bool(step.get("recursive", True))
        include_directories = bool(step.get("include_directories", False))
        limit = int(step.get("limit", 200))
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        matches: list[str] = []
        total = 0
        for candidate in iterator:
            if not include_directories and candidate.is_dir():
                continue
            total += 1
            if len(matches) < limit:
                matches.append(str(candidate.resolve()))
        return {
            "root": str(root),
            "pattern": pattern,
            "recursive": recursive,
            "count": total,
            "truncated": total > len(matches),
            "matches": matches,
        }

    def _build_default_input_controller(self) -> WindowsDesktopInputController:
        return WindowsDesktopInputController()

    def _default_open_path(self, path: Path) -> bool:
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(path))
            return True
        return bool(webbrowser.open(path.as_uri()))


class WindowsDesktopInputController:
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_UNICODE = 0x0004
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_MIDDLEDOWN = 0x0020
    MOUSEEVENTF_MIDDLEUP = 0x0040
    SW_RESTORE = 9
    VK_MAP = {
        "alt": 0x12,
        "backspace": 0x08,
        "ctrl": 0x11,
        "delete": 0x2E,
        "down": 0x28,
        "end": 0x23,
        "enter": 0x0D,
        "esc": 0x1B,
        "escape": 0x1B,
        "home": 0x24,
        "left": 0x25,
        "pagedown": 0x22,
        "pageup": 0x21,
        "right": 0x27,
        "shift": 0x10,
        "space": 0x20,
        "tab": 0x09,
        "up": 0x26,
        "win": 0x5B,
    }

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("Windows desktop input actions require Windows.")
        self.user32 = ctypes.windll.user32

    def focus_window(self, title_query: str) -> dict:
        query = title_query.strip().lower()
        if not query:
            raise ValueError("Window title query cannot be empty.")
        matches: list[tuple[int, str]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_windows(hwnd, _lparam):
            if not self.user32.IsWindowVisible(hwnd):
                return True
            length = self.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            self.user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = buffer.value.strip()
            if title and query in title.lower():
                matches.append((int(hwnd), title))
            return True

        self.user32.EnumWindows(enum_windows, 0)
        if not matches:
            raise ValueError(f"No visible window title matched '{title_query}'.")
        matches.sort(key=lambda item: (query != item[1].lower(), len(item[1])))
        hwnd, matched_title = matches[0]
        self.user32.ShowWindow(hwnd, self.SW_RESTORE)
        if not self.user32.SetForegroundWindow(hwnd):
            raise OSError(f"Unable to focus window '{matched_title}'.")
        return {"title_query": title_query, "matched_title": matched_title, "window_handle": hwnd}

    def type_text(self, text: str, *, interval_ms: int = 0, press_enter: bool = False) -> dict:
        for character in text:
            self._send_unicode_character(character)
            if interval_ms > 0:
                time.sleep(interval_ms / 1000.0)
        if press_enter:
            self.send_hotkey(["enter"])
        return {
            "text": text,
            "characters": len(text),
            "interval_ms": interval_ms,
            "press_enter": press_enter,
        }

    def send_hotkey(self, keys: list[str]) -> dict:
        virtual_keys = [self._virtual_key_for(name) for name in keys]
        for vk in virtual_keys:
            self._send_virtual_key(vk, key_up=False)
        for vk in reversed(virtual_keys):
            self._send_virtual_key(vk, key_up=True)
        return {"keys": keys}

    def click(self, *, x: int, y: int, button: str = "left", clicks: int = 1) -> dict:
        if clicks <= 0:
            raise ValueError("clicks must be > 0.")
        flags = self._mouse_flags(button)
        if not self.user32.SetCursorPos(int(x), int(y)):
            raise OSError(f"Unable to move mouse to ({x}, {y}).")
        for _ in range(clicks):
            self.user32.mouse_event(flags[0], 0, 0, 0, 0)
            self.user32.mouse_event(flags[1], 0, 0, 0, 0)
        return {"x": x, "y": y, "button": button, "clicks": clicks}

    def _virtual_key_for(self, name: str) -> int:
        normalized = name.strip().lower()
        if normalized in self.VK_MAP:
            return self.VK_MAP[normalized]
        if len(normalized) == 1 and normalized.isalpha():
            return ord(normalized.upper())
        if len(normalized) == 1 and normalized.isdigit():
            return ord(normalized)
        if normalized.startswith("f") and normalized[1:].isdigit():
            number = int(normalized[1:])
            if 1 <= number <= 24:
                return 0x70 + number - 1
        raise ValueError(f"Unsupported hotkey name: {name}")

    def _mouse_flags(self, button: str) -> tuple[int, int]:
        normalized = button.strip().lower()
        if normalized == "left":
            return (self.MOUSEEVENTF_LEFTDOWN, self.MOUSEEVENTF_LEFTUP)
        if normalized == "right":
            return (self.MOUSEEVENTF_RIGHTDOWN, self.MOUSEEVENTF_RIGHTUP)
        if normalized == "middle":
            return (self.MOUSEEVENTF_MIDDLEDOWN, self.MOUSEEVENTF_MIDDLEUP)
        raise ValueError(f"Unsupported mouse button: {button}")

    def _send_unicode_character(self, character: str) -> None:
        codepoint = ord(character)
        self._send_keyboard_input(0, codepoint, self.KEYEVENTF_UNICODE)
        self._send_keyboard_input(0, codepoint, self.KEYEVENTF_UNICODE | self.KEYEVENTF_KEYUP)

    def _send_virtual_key(self, virtual_key: int, *, key_up: bool) -> None:
        flags = self.KEYEVENTF_KEYUP if key_up else 0
        self._send_keyboard_input(virtual_key, 0, flags)

    def _send_keyboard_input(self, virtual_key: int, scan_code: int, flags: int) -> None:
        extra = ctypes.c_ulong(0)
        keyboard = _KEYBDINPUT(
            wVk=virtual_key,
            wScan=scan_code,
            dwFlags=flags,
            time=0,
            dwExtraInfo=ctypes.pointer(extra),
        )
        input_record = _INPUT(type=self.INPUT_KEYBOARD, union=_INPUTUNION(ki=keyboard))
        sent = self.user32.SendInput(1, ctypes.byref(input_record), ctypes.sizeof(_INPUT))
        if sent != 1:
            raise OSError("SendInput failed.")


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("ki", _KEYBDINPUT),
        ("mi", _MOUSEINPUT),
        ("hi", _HARDWAREINPUT),
    ]


class _INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("union", _INPUTUNION),
    ]
