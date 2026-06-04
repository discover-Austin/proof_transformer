from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .paths import draft_path, history_path, presets_path

STARTER_PRESETS = [
    {
        "id": "starter_search_web",
        "name": "Search The Web",
        "task_text": "search the web for Windows automation tips",
        "plan": {"steps": [{"action": "search_web", "query": "Windows automation tips"}]},
        "updated_at": "starter",
    },
    {
        "id": "starter_open_terminal",
        "name": "Open PowerShell",
        "task_text": "open powershell",
        "plan": {"steps": [{"action": "launch_process", "command": ["powershell.exe", "-NoExit"]}]},
        "updated_at": "starter",
    },
    {
        "id": "starter_runtime_folder",
        "name": "Open Runtime Folder",
        "task_text": "open folder runtime_data",
        "plan": {"steps": [{"action": "open_path", "path": "runtime_data"}]},
        "updated_at": "starter",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryStore:
    def __init__(self, history_file: Path | None = None, presets_file: Path | None = None, draft_file: Path | None = None) -> None:
        self.history_file = history_file or history_path()
        self.presets_file = presets_file or presets_path()
        self.draft_file = draft_file or draft_path()
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.presets_file.parent.mkdir(parents=True, exist_ok=True)
        self.draft_file.parent.mkdir(parents=True, exist_ok=True)

    def append_run(self, *, task_text: str | None, plan: dict, result: dict) -> dict:
        record = {
            "id": f"run_{uuid4().hex[:12]}",
            "timestamp": utc_now(),
            "task_text": task_text,
            "plan": plan,
            "result": result,
        }
        with self.history_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def list_runs(self, limit: int = 50) -> list[dict]:
        if not self.history_file.exists():
            return []
        with self.history_file.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return list(reversed(records[-max(int(limit), 1) :]))

    def save_preset(self, *, name: str, task_text: str | None, plan: dict) -> dict:
        presets = self.list_presets()
        record = {
            "id": f"preset_{uuid4().hex[:12]}",
            "name": name.strip(),
            "task_text": task_text,
            "plan": plan,
            "updated_at": utc_now(),
        }
        if not record["name"]:
            raise ValueError("Preset name cannot be empty.")
        updated = [item for item in presets if item["name"].lower() != record["name"].lower()]
        updated.append(record)
        updated.sort(key=lambda item: item["name"].lower())
        self.presets_file.write_text(json.dumps(updated, indent=2), encoding="utf-8")
        return record

    def list_presets(self) -> list[dict]:
        if not self.presets_file.exists():
            return list(STARTER_PRESETS)
        payload = json.loads(self.presets_file.read_text(encoding="utf-8"))
        presets = payload if isinstance(payload, list) else []
        if not presets:
            return list(STARTER_PRESETS)
        return presets

    def save_draft(self, *, task_text: str | None, plan: dict) -> dict:
        payload = {
            "task_text": task_text,
            "plan": plan,
            "updated_at": utc_now(),
        }
        self.draft_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def load_draft(self) -> dict | None:
        if not self.draft_file.exists():
            return None
        payload = json.loads(self.draft_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
