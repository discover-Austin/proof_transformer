from __future__ import annotations

import math

from common import BrainTestCase
from proof_transformer.memory.documents import DocumentStore
from proof_transformer.memory.embeddings import HashEmbeddingModel
from proof_transformer.memory.vector_store import VectorStore


class VectorStoreTests(BrainTestCase):
    def test_embedding_normalization_and_document_ingestion(self) -> None:
        model = HashEmbeddingModel(dimension=384)
        embedding = model.embed_text("persistent transformer memory")
        self.assertEqual(len(embedding.vector), 384)
        norm = math.sqrt(sum(value * value for value in embedding.vector))
        self.assertAlmostEqual(norm, 1.0, places=6)

        store = DocumentStore(self.conn, embedding_model=model)
        doc = store.ingest_text("brain_note", "Persistent memory lives on the SSD.\n\nVectors are deterministic and local.")
        duplicate = store.ingest_text("brain_note_copy", "Persistent memory lives on the SSD.\n\nVectors are deterministic and local.")
        self.assertFalse(doc["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        self.assertGreaterEqual(len(doc["chunks"]), 1)

    def test_vector_search_returns_relevant_chunk(self) -> None:
        model = HashEmbeddingModel(dimension=384)
        store = DocumentStore(self.conn, embedding_model=model)
        store.ingest_text("ranker", "The retrieval ranker is the first mutable learning target.")
        store.ingest_text("adapter", "Adapters are a future path for neural checkpoint learning.")
        vector_store = VectorStore(self.conn)
        query = model.embed_text("What is the first mutable learning target?")
        hits = vector_store.search(query.vector, top_k=2)
        self.assertIn("first mutable learning target", hits[0].text.lower())
