from __future__ import annotations


def parse_csv_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def summarize_answer(result: dict) -> dict:
    return {
        "answer": result.get("answer"),
        "retrieval_event_id": result.get("retrieval_event_id"),
        "backend": result.get("backend"),
    }
