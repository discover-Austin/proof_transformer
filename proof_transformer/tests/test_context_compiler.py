from __future__ import annotations

from common import BrainTestCase
from proof_transformer.memory.context_compiler import ContextCompiler
from proof_transformer.memory.documents import DocumentStore
from proof_transformer.memory.graph import GraphMemory


class ContextCompilerTests(BrainTestCase):
    def test_compile_creates_context_packet_and_retrieval_event(self) -> None:
        store = DocumentStore(self.conn)
        store.ingest_text("brain", "Persistent graph-KV memory acts like externalized attention.")
        GraphMemory(self.conn).upsert_node_by_name_kind(
            "project_fact",
            "Externalized attention",
            "Persistent graph-KV memory acts like externalized attention.",
            confidence=0.9,
            importance=0.9,
        )
        packet = ContextCompiler(self.conn).compile("How does graph memory act like attention?")
        self.assertTrue(packet.context_text)
        self.assertTrue(packet.retrieval_event_id)
        self.assertGreater(packet.token_estimate, 0)
        self.assertTrue(packet.included_chunks or packet.included_nodes)
