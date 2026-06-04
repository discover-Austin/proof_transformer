from __future__ import annotations

from ..config import load_settings
from .fallback_generator import FallbackGenerator
from .llama_cpp_client import LlamaCppClient
from .ollama_client import OllamaClient


class LocalLLM:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.backend = self._select_backend()

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "backend_name", self.backend.__class__.__name__.lower())

    def health(self) -> dict:
        checker = getattr(self.backend, "health", None)
        if callable(checker):
            payload = checker()
        else:
            payload = {"backend": self.backend_name, "available": True}
        payload.setdefault("backend", self.backend_name)
        return payload

    def _select_backend(self):
        if self.settings.ollama_base_url:
            return OllamaClient(self.settings.ollama_base_url, self.settings.ollama_model)
        if self.settings.llama_cpp_base_url:
            return LlamaCppClient(self.settings.llama_cpp_base_url, self.settings.llama_cpp_n_predict)
        return FallbackGenerator()

    def generate(self, prompt: str, system: str | None = None) -> str:
        try:
            return self.backend.generate(prompt, system)
        except Exception as exc:
            fallback = FallbackGenerator()
            return (
                f"Primary local backend failed ({self.backend_name}: {exc}).\n"
                + fallback.generate(prompt, system)
            )
