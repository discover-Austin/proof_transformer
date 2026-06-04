from __future__ import annotations

import json
from pathlib import Path

from ..paths import get_brain_paths
from ..timeutils import utc_now


class SessionLog:
    def __init__(self, path: str | Path | None = None) -> None:
        brain = get_brain_paths()
        brain.ensure()
        self.path = Path(path) if path else brain.sessions / f"session_{utc_now().replace(':', '-')}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict) -> None:
        record = {"timestamp": utc_now(), "event_type": event_type, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
