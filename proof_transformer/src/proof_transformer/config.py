from __future__ import annotations

import os
from dataclasses import dataclass, field

from .paths import BrainPaths, get_brain_paths


@dataclass(frozen=True)
class Settings:
    brain: BrainPaths
    embedding_dimension: int = 384
    embedding_model_name: str = "hash-embedding-v1"
    embedding_backend: str = "hash"
    chunk_target_tokens: int = 180
    context_token_budget: int = 1200
    retrieval_top_k: int = 6
    graph_max_hops: int = 2
    workspace_watch_poll_interval: float = 2.0
    command_timeout: int = 120
    ollama_base_url: str | None = None
    ollama_model: str = "llama3.2:3b"
    llama_cpp_base_url: str | None = None
    llama_cpp_n_predict: int = 384
    llm_system_prompt: str = (
        "You are Austin Grandstaff's local Proof-of-Improvement Persistent Transformer. "
        "Use retrieved evidence, prefer local paths, avoid placeholders, and state when a local fallback is active."
    )
    ranker_learning_rate: float = 0.12
    validator_weights: dict[str, float] = field(
        default_factory=lambda: {
            "PreferenceValidator": 0.20,
            "ContradictionValidator": 0.20,
            "ScopeValidator": 0.20,
            "ReplayValidator": 0.15,
            "EvidenceValidator": 0.15,
            "OutcomeValidator": 0.10,
        }
    )


def load_settings(refresh_paths: bool = False) -> Settings:
    brain = get_brain_paths(refresh=refresh_paths)
    return Settings(
        brain=brain,
        embedding_dimension=int(os.getenv("PROOF_TRANSFORMER_EMBED_DIM", "384")),
        embedding_model_name=os.getenv("PROOF_TRANSFORMER_EMBED_MODEL", "hash-embedding-v1"),
        embedding_backend=os.getenv("PROOF_TRANSFORMER_EMBED_BACKEND", "hash"),
        chunk_target_tokens=int(os.getenv("PROOF_TRANSFORMER_CHUNK_TOKENS", "180")),
        context_token_budget=int(os.getenv("PROOF_TRANSFORMER_CONTEXT_TOKENS", "1200")),
        retrieval_top_k=int(os.getenv("PROOF_TRANSFORMER_TOP_K", "6")),
        graph_max_hops=int(os.getenv("PROOF_TRANSFORMER_GRAPH_HOPS", "2")),
        workspace_watch_poll_interval=float(os.getenv("PROOF_TRANSFORMER_WATCH_POLL", "2.0")),
        command_timeout=int(os.getenv("PROOF_TRANSFORMER_COMMAND_TIMEOUT", "120")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL") or None,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        llama_cpp_base_url=os.getenv("LLAMA_CPP_BASE_URL") or None,
        llama_cpp_n_predict=int(os.getenv("LLAMA_CPP_N_PREDICT", "384")),
        ranker_learning_rate=float(os.getenv("PROOF_TRANSFORMER_RANKER_LR", "0.12")),
    )
