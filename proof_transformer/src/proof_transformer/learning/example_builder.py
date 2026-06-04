from __future__ import annotations


def build_example(event_type: str, source_text: str, context: dict | None = None) -> tuple[str, str, str | None]:
    context = context or {}
    if event_type == "correction":
        return (
            "Prefer the corrected answer.",
            source_text,
            context.get("prior_output"),
        )
    if event_type == "preference":
        return (
            "Apply this durable preference.",
            source_text,
            context.get("prior_behavior"),
        )
    if event_type == "retrieval_feedback":
        return (
            "Update the retrieval ranker using this feedback.",
            source_text,
            context.get("rejected"),
        )
    return (
        "Store this validated observation.",
        source_text,
        context.get("rejected"),
    )
