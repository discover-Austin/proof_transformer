from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.memory.graph import GraphMemory


class WorkspaceSyncTests(BrainTestCase):
    def test_incremental_sync_updates_and_removes_workspace_files(self) -> None:
        workspace = self.paths.temp / "sync_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        target = workspace / "worker.py"
        target.write_text("class QueueWorker:\n    pass\n", encoding="utf-8")

        orchestrator = ProofTransformerOrchestrator(self.conn)
        first = orchestrator.index_workspace(workspace)
        self.assertEqual(first["new_files"], 1)
        self.assertEqual(first["updated_files"], 0)
        self.assertEqual(first["unchanged_files"], 0)

        second = orchestrator.index_workspace(workspace)
        self.assertEqual(second["unchanged_files"], 1)

        target.write_text("class QueueWorker:\n    def run(self):\n        return 1\n", encoding="utf-8")
        third = orchestrator.index_workspace(workspace)
        self.assertEqual(third["updated_files"], 1)

        graph = GraphMemory(self.conn)
        hits = graph.search_nodes_keyword("run", source_uri_prefix=str(workspace))
        self.assertTrue(any("run @ worker.py" in node["name"] for node in hits))

        target.unlink()
        fourth = orchestrator.index_workspace(workspace)
        self.assertEqual(fourth["removed_files"], 1)
        removed_hits = graph.search_nodes_keyword("QueueWorker", source_uri_prefix=str(workspace))
        self.assertFalse(any(node["source_uri"] and "worker.py" in node["source_uri"] for node in removed_hits))
