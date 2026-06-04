from __future__ import annotations

import os

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.db import get_connection
from proof_transformer.paths import get_brain_paths, reset_cached_paths


class RankerLearningTests(BrainTestCase):
    def test_feedback_accepts_and_updates_ranker(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        orchestrator.document_store.ingest_text(
            "ranker",
            "The retrieval ranker is the first mutable learning target in the local architecture.",
        )
        ask = orchestrator.ask("What is the first mutable learning target?")
        useful = [ask["included_chunks"][0]["chunk_id"]]
        result = orchestrator.feedback(ask["retrieval_event_id"], useful_chunk_ids=useful, irrelevant_chunk_ids=[])
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(orchestrator.status()["ranker_version"], 2)

    def test_feedback_rejection_rolls_back_ranker_version(self) -> None:
        self.conn.close()
        os.environ["PROOF_TRANSFORMER_RANKER_LR"] = "3.0"
        reset_cached_paths()
        self.paths = get_brain_paths(refresh=True)
        self.conn = get_connection()
        orchestrator = ProofTransformerOrchestrator(self.conn)
        orchestrator.document_store.ingest_text(
            "ranker",
            "The retrieval ranker is the first mutable learning target in the local architecture.",
        )
        ask = orchestrator.ask("What is the first mutable learning target?")
        useful = [ask["included_chunks"][0]["chunk_id"]]
        result = orchestrator.feedback(ask["retrieval_event_id"], useful_chunk_ids=useful, irrelevant_chunk_ids=[])
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(orchestrator.status()["ranker_version"], 1)
