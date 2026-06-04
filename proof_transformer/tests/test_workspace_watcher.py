from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.agent.workspace_watcher import WorkspaceWatcher


class WorkspaceWatcherTests(BrainTestCase):
    def test_workspace_watcher_runs_cycles(self) -> None:
        workspace = self.paths.temp / "watch_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")

        orchestrator = ProofTransformerOrchestrator(self.conn)
        watcher = WorkspaceWatcher(orchestrator, workspace, poll_interval=0.0)
        result = watcher.serve(max_cycles=2)
        self.assertEqual(result["cycles"], 2)
        self.assertEqual(result["history"][0]["indexed"], 1)
