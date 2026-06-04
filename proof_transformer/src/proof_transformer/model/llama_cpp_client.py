from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class LlamaCppClient:
    backend_name = "llama.cpp"

    def __init__(self, base_url: str, n_predict: int = 384) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.n_predict = n_predict

    def health(self) -> dict:
        request = Request(self.base_url, method="GET")
        try:
            with urlopen(request, timeout=10) as response:
                status = getattr(response, "status", 200)
        except URLError as exc:
            return {
                "backend": self.backend_name,
                "available": False,
                "base_url": self.base_url,
                "error": str(exc),
            }
        return {
            "backend": self.backend_name,
            "available": 200 <= int(status) < 500,
            "base_url": self.base_url,
            "n_predict": self.n_predict,
            "http_status": int(status),
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        full_prompt = f"{system or ''}\n\n{prompt}".strip()
        payload = {
            "prompt": full_prompt,
            "n_predict": self.n_predict,
            "temperature": 0.1,
            "stop": ["</s>"],
        }
        request = Request(
            urljoin(self.base_url, "completion"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"llama.cpp unavailable at {self.base_url}: {exc}") from exc
        if "content" in body:
            return str(body["content"]).strip()
        if "choices" in body and body["choices"]:
            choice = body["choices"][0]
            return str(choice.get("text") or choice.get("content") or "").strip()
        return ""
