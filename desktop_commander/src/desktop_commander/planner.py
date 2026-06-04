from __future__ import annotations

import re
from pathlib import Path


class TaskPlanner:
    def draft_plan(self, task_text: str) -> dict:
        text = task_text.strip()
        if not text:
            raise ValueError("Task text cannot be empty.")
        segments = self._split_segments(text)
        steps: list[dict] = []
        notes: list[str] = []
        for segment in segments:
            generated = self._steps_for_segment(segment)
            if generated:
                steps.extend(generated)
            else:
                notes.append(f"No exact parser rule matched: {segment}")
        if not steps:
            steps.append({"action": "search_web", "query": text})
            notes.append("Fell back to a web search because no stronger action match was found.")
        return {
            "task_text": text,
            "summary": self._summary_for_steps(steps),
            "continue_on_error": False,
            "notes": notes,
            "steps": steps,
        }

    def _split_segments(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        segments = re.split(r"\b(?:and then|then|also|and)\b", normalized, flags=re.IGNORECASE)
        cleaned = [segment.strip(" ,;") for segment in segments]
        return [segment for segment in cleaned if segment]

    def _steps_for_segment(self, segment: str) -> list[dict]:
        lowered = segment.lower()

        search_match = re.search(r"(?:search (?:the web )?for|look up)\s+(.+)", segment, flags=re.IGNORECASE)
        if search_match:
            return [{"action": "search_web", "query": search_match.group(1).strip()}]

        url_match = re.search(r"(https?://\S+)", segment, flags=re.IGNORECASE)
        if url_match and ("open" in lowered or lowered.startswith("go to")):
            return [{"action": "open_url", "url": url_match.group(1)}]
        if url_match and ("fetch" in lowered or "read url" in lowered):
            return [{"action": "fetch_url", "url": url_match.group(1)}]

        if any(phrase in lowered for phrase in ("open powershell", "open terminal", "launch powershell")):
            return [{"action": "launch_process", "command": ["powershell.exe", "-NoExit"]}]

        wait_match = re.search(r"wait\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?)", lowered)
        if wait_match:
            return [{"action": "wait", "seconds": float(wait_match.group(1))}]

        type_match = re.search(r"type\s+(.+)", segment, flags=re.IGNORECASE)
        if type_match:
            text = type_match.group(1).strip().strip("\"'")
            press_enter = any(token in lowered for token in ("press enter", "hit enter", "and enter", "submit"))
            return [{"action": "type_text", "text": text, "press_enter": press_enter}]

        hotkey_match = re.search(r"press\s+([a-z0-9+ ]+)$", lowered)
        if hotkey_match:
            chord = hotkey_match.group(1).replace(" plus ", "+").replace(" ", "")
            keys = [item for item in chord.split("+") if item]
            if keys:
                return [{"action": "send_hotkey", "keys": keys}]

        write_match = re.search(
            r"write\s+[\"'](?P<text>.+?)[\"']\s+to\s+(?:file\s+)?(?P<path>.+)",
            segment,
            flags=re.IGNORECASE,
        )
        if write_match:
            return [
                {
                    "action": "write_file",
                    "path": write_match.group("path").strip(),
                    "text": write_match.group("text"),
                }
            ]

        read_match = re.search(r"read\s+(?:file\s+)?(.+)", segment, flags=re.IGNORECASE)
        if read_match:
            path = read_match.group(1).strip()
            if self._looks_like_path(path):
                return [{"action": "read_file", "path": path}]

        list_match = re.search(r"list\s+files\s+in\s+(.+)", segment, flags=re.IGNORECASE)
        if list_match:
            return [{"action": "list_files", "path": list_match.group(1).strip()}]

        open_path_match = re.search(r"open\s+(?:folder|file|path)\s+(.+)", segment, flags=re.IGNORECASE)
        if open_path_match:
            return [{"action": "open_path", "path": open_path_match.group(1).strip()}]

        focus_match = re.search(r"focus\s+(.+)", segment, flags=re.IGNORECASE)
        if focus_match:
            return [{"action": "focus_window", "title": focus_match.group(1).strip()}]

        click_match = re.search(r"click\s+(\d+)\s*,\s*(\d+)", lowered)
        if click_match:
            return [{"action": "click", "x": int(click_match.group(1)), "y": int(click_match.group(2))}]

        command_match = re.search(r"run\s+command\s+(.+)", segment, flags=re.IGNORECASE)
        if command_match:
            command_text = command_match.group(1).strip()
            return [{"action": "run_command", "command": ["powershell.exe", "-Command", command_text]}]

        return []

    def _looks_like_path(self, value: str) -> bool:
        text = value.strip().strip("\"'")
        if re.match(r"^[A-Za-z]:[\\/]", text):
            return True
        if text.startswith(".") or text.startswith("\\") or text.startswith("/"):
            return True
        return Path(text).suffix != ""

    def _summary_for_steps(self, steps: list[dict]) -> str:
        action_names = ", ".join(step["action"] for step in steps[:6])
        if len(steps) > 6:
            action_names += ", ..."
        return f"{len(steps)} step(s): {action_names}"
