from __future__ import annotations

import sys

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


class CommandRunnerTests(BrainTestCase):
    def test_run_command_captures_output_and_persists_execution(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        result = orchestrator.run_command([sys.executable, "-c", "print('brain-ok')"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["stdout_path"])
        self.assertTrue(result["output_document_id"])
        listed = orchestrator.list_execution_runs(limit=5)
        self.assertTrue(any(run["id"] == result["id"] for run in listed))

    def test_review_execution_queries_ingested_output(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        run = orchestrator.run_command([sys.executable, "-c", "print('brain-review-ok')"])
        review = orchestrator.review_execution(run["id"], "What did the command print?")
        self.assertEqual(review["execution_id"], run["id"])
        self.assertTrue(review["included_chunks"])
        self.assertIn("brain-review-ok", review["context_text"])
