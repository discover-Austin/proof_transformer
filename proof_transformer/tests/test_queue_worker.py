from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.agent.worker import QueueWorker


class QueueWorkerTests(BrainTestCase):
    def test_worker_serves_pending_queue(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        orchestrator.enqueue_task("status", {})
        worker = QueueWorker(orchestrator, poll_interval=0.0)
        result = worker.serve(max_cycles=1)
        self.assertEqual(result["cycles"], 1)
        self.assertEqual(result["total_ran"], 1)
        self.assertEqual(orchestrator.list_tasks(status="completed")[0]["task_type"], "status")
