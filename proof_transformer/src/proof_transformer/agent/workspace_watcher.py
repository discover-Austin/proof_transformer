from __future__ import annotations

import time
from pathlib import Path

from .orchestrator import ProofTransformerOrchestrator


class WorkspaceWatcher:
    def __init__(
        self,
        orchestrator: ProofTransformerOrchestrator,
        workspace_path: str | Path,
        *,
        poll_interval: float | None = None,
        max_file_bytes: int = 1_000_000,
        extensions: set[str] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.workspace_path = Path(workspace_path)
        self.poll_interval = (
            orchestrator.settings.workspace_watch_poll_interval if poll_interval is None else max(float(poll_interval), 0.0)
        )
        self.max_file_bytes = int(max_file_bytes)
        self.extensions = extensions

    def run_once(self) -> dict:
        return self.orchestrator.index_workspace(
            self.workspace_path,
            max_file_bytes=self.max_file_bytes,
            extensions=self.extensions,
        )

    def serve(self, max_cycles: int | None = None) -> dict:
        cycles = 0
        history: list[dict] = []
        while max_cycles is None or cycles < max_cycles:
            history.append(self.run_once())
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            time.sleep(self.poll_interval)
        return {
            "workspace_path": str(self.workspace_path),
            "cycles": cycles,
            "poll_interval": self.poll_interval,
            "history": history,
        }
