from __future__ import annotations

import sys

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


class TaskQueueTests(BrainTestCase):
    def test_enqueue_and_run_queue(self) -> None:
        workspace = self.paths.temp / "queue_workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "brain.md").write_text(
            "The retrieval ranker is the first mutable learning target.\n",
            encoding="utf-8",
        )

        orchestrator = ProofTransformerOrchestrator(self.conn)
        task_a = orchestrator.enqueue_task("index_workspace", {"path": str(workspace)})
        task_b = orchestrator.enqueue_task("ask", {"query": "What is the first mutable learning target?"})
        result = orchestrator.run_queue()

        self.assertEqual(result["ran"], 2)
        completed = orchestrator.list_tasks(status="completed")
        self.assertEqual({task["id"] for task in completed}, {task_a["id"], task_b["id"]})
        ask_task = next(task for task in completed if task["id"] == task_b["id"])
        self.assertIn("retrieval_event_id", ask_task["result"])
        self.assertEqual(orchestrator.status()["task_status_counts"].get("completed"), 2)

    def test_queue_can_run_local_command(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        queued = orchestrator.enqueue_task(
            "run_command",
            {"command": [sys.executable, "-c", "print('queued-brain-ok')"]},
        )
        result = orchestrator.run_queue()
        self.assertEqual(result["ran"], 1)
        completed = orchestrator.list_tasks(status="completed")
        task = next(item for item in completed if item["id"] == queued["id"])
        self.assertEqual(task["result"]["status"], "completed")
        self.assertEqual(task["result"]["exit_code"], 0)
