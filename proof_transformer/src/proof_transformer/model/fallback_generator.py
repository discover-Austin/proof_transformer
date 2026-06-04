from __future__ import annotations

import re

from ..memory.embeddings import tokenize


class FallbackGenerator:
    backend_name = "fallback"

    def health(self) -> dict:
        return {
            "backend": self.backend_name,
            "available": True,
            "mode": "deterministic-local-fallback",
        }

    def generate(self, prompt: str, system: str | None = None) -> str:
        _ = system
        question = self._extract_section(prompt, "QUESTION:") or prompt.strip()
        context = self._extract_section(prompt, "CONTEXT:") or prompt
        evidence = self._best_evidence(question, context)
        if evidence:
            bullets = "\n".join(f"- {line}" for line in evidence)
            answer_core = evidence[0]
            return (
                "Using local fallback generator.\n"
                "Grounded evidence:\n"
                f"{bullets}\n"
                f"Answer: Based on retrieved local memory, {answer_core}"
            )
        return "Using local fallback generator. No strong local evidence was retrieved for this question."

    def _extract_section(self, prompt: str, marker: str) -> str:
        if marker not in prompt:
            return ""
        after = prompt.split(marker, 1)[1]
        stop_markers = ["\nQUESTION:", "\nCONTEXT:", "\nINSTRUCTIONS:"]
        stops = [after.find(stop) for stop in stop_markers if after.find(stop) >= 0]
        end = min(stops) if stops else len(after)
        return after[:end].strip()

    def _best_evidence(self, question: str, context: str) -> list[str]:
        q_tokens = set(tokenize(question))
        identifier_matches = re.findall(r"\b[A-Z][A-Za-z0-9_]+\b", question)
        identifiers = [match for match in identifier_matches if len(match) > 2]
        candidates = [line.strip() for line in re.split(r"\n+", context) if line.strip()]
        scored: list[tuple[float, str]] = []
        for line in candidates:
            lower = line.lower()
            if lower.startswith(("query:", "retrieved chunks:", "relevant nodes:", "relevant edges:")):
                continue
            if re.match(r"^\[\d+\]\s+source=", lower):
                continue
            line_tokens = set(tokenize(line))
            overlap = len(q_tokens & line_tokens) / max(len(q_tokens), 1) if q_tokens else 0.0
            bonus = 0.05 if line.startswith("-") else 0.0
            for identifier in identifiers:
                if identifier.lower() in lower:
                    bonus += 0.35
                    if any(pattern in lower for pattern in (f"class {identifier.lower()}", f"def {identifier.lower()}", f"import {identifier.lower()}")):
                        bonus += 0.25
            scored.append((overlap + bonus, line))
        scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        return [line for score, line in scored if score > 0][:3]
