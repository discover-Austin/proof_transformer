from __future__ import annotations

from common import BrainTestCase
from proof_transformer.memory.documents import DocumentStore
from proof_transformer.memory.graph import GraphMemory
from proof_transformer.memory.retrieval import Retriever


class RetrievalTests(BrainTestCase):
    def test_retrieval_combines_vector_and_graph_signal(self) -> None:
        store = DocumentStore(self.conn)
        store.ingest_text("ranker", "The retrieval ranker is the first mutable learning target in v0.3.")
        store.ingest_text("ledger", "The ledger stores validated learning transactions and proof of improvement.")
        graph = GraphMemory(self.conn)
        graph.upsert_node_by_name_kind(
            "project_fact",
            "Retrieval ranker",
            "The first mutable learning target is the retrieval ranker.",
            confidence=0.95,
            importance=0.95,
        )
        results = Retriever(self.conn).retrieve("What is the first mutable learning target?", top_k=2)
        self.assertTrue(results)
        self.assertIn("ranker", results[0].text.lower())
        self.assertGreater(results[0].score_parts["vector_similarity"], 0.0)
