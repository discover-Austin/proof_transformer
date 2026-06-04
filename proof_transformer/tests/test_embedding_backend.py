from __future__ import annotations

import os

from common import BrainTestCase
from proof_transformer.config import load_settings
from proof_transformer.memory.embeddings import HashEmbeddingModel, SentenceTransformerEmbeddingModel, build_embedding_model


class EmbeddingBackendTests(BrainTestCase):
    def test_unknown_backend_uses_hash_embedding(self) -> None:
        os.environ["PROOF_TRANSFORMER_EMBED_BACKEND"] = "not-a-real-backend"
        settings = load_settings()
        model = build_embedding_model(settings)
        self.assertIsInstance(model, HashEmbeddingModel)
        self.assertTrue(model.health()["available"])

    def test_sentence_transformers_request_never_breaks_local_embedding(self) -> None:
        os.environ["PROOF_TRANSFORMER_EMBED_BACKEND"] = "sentence-transformers"
        os.environ["PROOF_TRANSFORMER_EMBED_MODEL"] = "sentence-transformers/all-MiniLM-L6-v2"
        settings = load_settings()
        model = build_embedding_model(settings)
        self.assertTrue(isinstance(model, (HashEmbeddingModel, SentenceTransformerEmbeddingModel)))
        self.assertTrue(model.health()["available"])
