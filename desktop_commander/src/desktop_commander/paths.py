from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_app_home() -> Path:
    raw = os.getenv("DESKTOP_COMMANDER_HOME")
    if raw:
        path = Path(raw).expanduser()
        return path if path.is_absolute() else path.absolute()
    return PROJECT_ROOT / "runtime_data"


def ensure_app_home() -> Path:
    home = get_app_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "command_runs").mkdir(parents=True, exist_ok=True)
    return home


def history_path() -> Path:
    return ensure_app_home() / "run_history.jsonl"


def presets_path() -> Path:
    return ensure_app_home() / "presets.json"


def draft_path() -> Path:
    return ensure_app_home() / "last_draft.json"


def command_runs_dir() -> Path:
    return ensure_app_home() / "command_runs"
