from __future__ import annotations


def classify_event(text: str, explicit_type: str | None = None) -> str:
    if explicit_type:
        return explicit_type
    lower = text.lower()
    if any(word in lower for word in ("prefer", "always", "never")):
        return "preference"
    if any(word in lower for word in ("fixed", "bug", "regression")):
        return "bug_fix"
    if any(word in lower for word in ("wrong", "correction", "actually")):
        return "correction"
    if any(word in lower for word in ("fact", "remember", "project")):
        return "project_fact"
    return "general_observation"
