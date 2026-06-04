from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Callable
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from uuid import uuid4

from .paths import command_runs_dir

DEFAULT_SEARCH_URL_TEMPLATE = "https://duckduckgo.com/?q={query}"


class DesktopExecutor:
    def __init__(
        self,
        *,
        input_controller: WindowsDesktopInputController | None = None,
        browser_opener: Callable[[str], bool] | None = None,
        path_opener: Callable[[Path], bool] | None = None,
        process_launcher: Callable[..., object] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        url_fetcher: Callable[[str, int], dict] | None = None,
    ) -> None:
        self.input_controller = input_controller if input_controller is not None else self._default_input_controller()
        self.browser_opener = browser_opener or webbrowser.open
        self.path_opener = path_opener or self._default_open_path
        self.process_launcher = process_launcher or subprocess.Popen
        self.sleep_fn = sleep_fn or time.sleep
        self.url_fetcher = url_fetcher or self._fetch_url

    def capabilities(self) -> dict:
        return {
            "platform": sys.platform,
            "search_url_template": DEFAULT_SEARCH_URL_TEMPLATE,
            "actions": [
                "run_command",
                "launch_process",
                "open_url",
                "search_web",
                "fetch_url",
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
                "This app is standalone and does not depend on proof_transformer.",
                "Desktop input actions require a visible Windows desktop session.",
                "Command runs are logged under runtime_data/command_runs.",
            ],
        }

    def execute_plan(self, plan: dict | list[dict], *, continue_on_error: bool = False) -> dict:
        steps = plan["steps"] if isinstance(plan, dict) else plan
        if not isinstance(steps, list) or not steps:
            raise ValueError("Plan must contain a non-empty steps list.")
        results: list[dict] = []
        status = "completed"
        for index, step in enumerate(steps):
            action = str(step.get("action") or "").strip()
            if not action:
                raise ValueError(f"Step {index} is missing an action.")
            try:
                result = self.execute_step(step)
                step_status = self._step_status_from_result(result)
                if step_status != "completed":
                    status = "failed"
                step_result = {
                    "step_index": index,
                    "step_id": step.get("id"),
                    "action": action,
                    "status": step_status,
                    "result": result,
                }
                if step_status != "completed" and not continue_on_error:
                    results.append(step_result)
                    return {
                        "status": status,
                        "steps_executed": len(results),
                        "continue_on_error": continue_on_error,
                        "results": results,
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

    def _step_status_from_result(self, result: dict) -> str:
        nested_status = str(result.get("status") or "").lower() if isinstance(result, dict) else ""
        if nested_status in {"failed", "error", "timeout"}:
            return nested_status
        return "completed"

    def execute_step(self, step: dict) -> dict:
        action = str(step.get("action") or "").strip()
        if action == "run_command":
            command = step.get("command")
            if not command:
                raise ValueError("run_command requires a command.")
            return self._run_command(
                command=command,
                cwd=step.get("cwd"),
                timeout=step.get("timeout"),
            )
        if action == "launch_process":
            command = step.get("command")
            if not command:
                raise ValueError("launch_process requires a command.")
            return self._launch_process(command, cwd=step.get("cwd"), new_console=bool(step.get("new_console", True)))
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
        if action == "fetch_url":
            url = str(step.get("url") or "").strip()
            if not url:
                raise ValueError("fetch_url requires a url.")
            timeout = int(step.get("timeout", 20))
            max_chars = int(step.get("max_chars", 20000))
            payload = self.url_fetcher(url, timeout)
            text = payload["text"]
            return {
                "url": url,
                "status_code": payload["status_code"],
                "content_type": payload["content_type"],
                "text": text[:max_chars],
                "truncated": len(text) > max_chars,
            }
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
            return self._require_input_controller().focus_window(str(step.get("title") or ""))
        if action == "type_text":
            return self._require_input_controller().type_text(
                str(step.get("text") or ""),
                interval_ms=int(step.get("interval_ms", 0)),
                press_enter=bool(step.get("press_enter", False)),
            )
        if action == "send_hotkey":
            keys = step.get("keys")
            if not isinstance(keys, list) or not keys:
                raise ValueError("send_hotkey requires a non-empty keys list.")
            return self._require_input_controller().send_hotkey([str(item) for item in keys])
        if action == "click":
            return self._require_input_controller().click(
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
        raise ValueError(f"Unsupported action: {action}")

    def _default_input_controller(self) -> WindowsDesktopInputController | None:
        if os.name != "nt":
            return None
        try:
            return WindowsDesktopInputController()
        except Exception:
            return None

    def _require_input_controller(self) -> WindowsDesktopInputController:
        if self.input_controller is None:
            raise OSError("Desktop input actions require Windows and a visible desktop session.")
        return self.input_controller

    def _launch_process(self, command: list[str] | str, *, cwd: str | None = None, new_console: bool = True) -> dict:
        command_list = self._normalize_command(command)
        creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if os.name == "nt" and new_console else 0
        process = self.process_launcher(command_list, cwd=cwd, shell=False, creationflags=creationflags)
        return {
            "pid": int(process.pid),
            "command": command_list,
            "cwd": cwd,
            "new_console": new_console,
        }

    def _run_command(self, *, command: list[str] | str, cwd: str | None = None, timeout: int | None = None) -> dict:
        command_list = self._normalize_command(command)
        if not command_list:
            raise ValueError("Command cannot be empty.")
        timeout_seconds = int(timeout or 120)
        run_id = f"cmd_{uuid4().hex[:12]}"
        run_dir = command_runs_dir() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        meta_path = run_dir / "metadata.json"
        status = "completed"
        exit_code: int | None = None
        stdout_text = ""
        stderr_text = ""
        try:
            completed = subprocess.run(
                command_list,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
            exit_code = int(completed.returncode)
            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            if exit_code != 0:
                status = "failed"
        except subprocess.TimeoutExpired as exc:
            status = "timeout"
            stdout_text = exc.stdout or ""
            stderr_text = (exc.stderr or "") + f"\nCommand timed out after {timeout_seconds} seconds."
        except Exception as exc:
            status = "error"
            stderr_text = str(exc)
        stdout_path.write_text(stdout_text, encoding="utf-8")
        stderr_path.write_text(stderr_text, encoding="utf-8")
        meta = {
            "id": run_id,
            "command": command_list,
            "cwd": cwd,
            "timeout_seconds": timeout_seconds,
            "status": status,
            "exit_code": exit_code,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta

    def _normalize_command(self, command: list[str] | str) -> list[str]:
        command_list = command if isinstance(command, list) else [command]
        if not command_list:
            return []
        normalized = [str(item) for item in command_list]
        normalized[0] = self._resolve_executable(normalized[0])
        return normalized

    def _resolve_executable(self, executable: str) -> str:
        if not executable:
            raise ValueError("Executable cannot be empty.")
        if Path(executable).anchor or any(sep in executable for sep in ("/", "\\")):
            return executable
        resolved = shutil.which(executable)
        if resolved:
            return resolved
        if os.name == "nt":
            system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
            program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            program_files_x86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
            search_dirs = [
                system_root / "System32",
                system_root / "System32" / "WindowsPowerShell" / "v1.0",
                system_root,
                program_files / "PowerShell" / "7",
                program_files_x86 / "PowerShell" / "7",
            ]
            names = [executable]
            if not executable.lower().endswith(".exe"):
                names.append(f"{executable}.exe")
            for search_dir in search_dirs:
                for name in names:
                    candidate = search_dir / name
                    if candidate.exists():
                        return str(candidate)
        return executable

    def _read_file(self, step: dict) -> dict:
        raw_path = step.get("path")
        if not raw_path:
            raise ValueError("read_file requires a path.")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        max_bytes = int(step.get("max_bytes", 200000))
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
        if bool(step.get("create_parent_dirs", True)):
            path.parent.mkdir(parents=True, exist_ok=True)
        text = str(step.get("text") or "")
        encoding = str(step.get("encoding", "utf-8"))
        mode = str(step.get("mode", "overwrite"))
        if mode == "append":
            with path.open("a", encoding=encoding) as handle:
                handle.write(text)
        elif mode == "overwrite":
            path.write_text(text, encoding=encoding)
        else:
            raise ValueError("write_file mode must be overwrite or append.")
        return {"path": str(path), "mode": mode, "bytes_written": len(text.encode(encoding))}

    def _list_files(self, step: dict) -> dict:
        root = Path(step.get("path") or step.get("root") or ".").expanduser().resolve()
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

    def _default_open_path(self, path: Path) -> bool:
        if os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(path))
            return True
        return bool(webbrowser.open(path.as_uri()))

    def _fetch_url(self, url: str, timeout: int) -> dict:
        request = Request(url, headers={"User-Agent": "DesktopCommander/0.1"})
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            content_type = response.headers.get("Content-Type", "application/octet-stream")
            text = body.decode("utf-8", errors="replace")
            return {
                "status_code": int(getattr(response, "status", 200)),
                "content_type": content_type,
                "text": text,
            }


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
        "ctrl": 0x11,
        "delete": 0x2E,
        "down": 0x28,
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
            raise OSError("Desktop input requires Windows.")
        self.user32 = ctypes.windll.user32

    def focus_window(self, title_query: str) -> dict:
        query = title_query.strip().lower()
        if not query:
            raise ValueError("focus_window requires a title.")
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
            raise ValueError(f"No visible window matched '{title_query}'.")
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
        return {"text": text, "characters": len(text), "interval_ms": interval_ms, "press_enter": press_enter}

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
        if not self.user32.SetCursorPos(int(x), int(y)):
            raise OSError(f"Unable to move cursor to ({x}, {y}).")
        down_flag, up_flag = self._mouse_flags(button)
        for _ in range(clicks):
            self.user32.mouse_event(down_flag, 0, 0, 0, 0)
            self.user32.mouse_event(up_flag, 0, 0, 0, 0)
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
        raise ValueError(f"Unsupported hotkey key: {name}")

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
