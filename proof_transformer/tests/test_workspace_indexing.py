from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.memory.graph import GraphMemory


class WorkspaceIndexingTests(BrainTestCase):
    def test_index_workspace_ingests_code_and_supports_assist(self) -> None:
        workspace = self.paths.temp / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "app.py").write_text(
            "class TaskQueue:\n    pass\n\ndef add(a, b):\n    return a + b\n\n# Retrieval ranker helper\n",
            encoding="utf-8",
        )
        (workspace / "README.md").write_text(
            "The workspace contains a small add function and local notes.\n",
            encoding="utf-8",
        )
        (workspace / "ignore.bin").write_bytes(b"\x00\x01\x02")

        orchestrator = ProofTransformerOrchestrator(self.conn)
        indexed = orchestrator.index_workspace(workspace)
        self.assertEqual(indexed["indexed"], 2)
        self.assertEqual(indexed["skipped_unsupported"], 1)
        self.assertGreaterEqual(indexed["symbols_indexed"], 2)

        graph_hits = GraphMemory(self.conn).search_nodes_keyword("TaskQueue", source_uri_prefix=str(workspace))
        self.assertTrue(any("TaskQueue" in node["name"] for node in graph_hits))

        result = orchestrator.assist("What does add return?", workspace_path=workspace)
        self.assertTrue(result["included_chunks"])
        self.assertIn("add", result["context_text"].lower())
