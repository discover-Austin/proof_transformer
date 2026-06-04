from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class OllamaClient:
    backend_name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.model = model

    def health(self) -> dict:
        request = Request(urljoin(self.base_url, "api/tags"), method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            return {
                "backend": self.backend_name,
                "available": False,
                "base_url": self.base_url,
                "model": self.model,
                "error": str(exc),
            }
        models = body.get("models") or []
        names = [item.get("name") for item in models if isinstance(item, dict)]
        return {
            "backend": self.backend_name,
            "available": True,
            "base_url": self.base_url,
            "model": self.model,
            "model_present": self.model in names,
            "available_models": names,
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
        }
        request = Request(
            urljoin(self.base_url, "api/generate"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"Ollama unavailable at {self.base_url}: {exc}") from exc
        return str(body.get("response", "")).strip()
