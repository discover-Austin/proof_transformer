from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def age_days(value: str) -> float:
    delta = datetime.now(timezone.utc) - parse_utc(value)
    return delta.total_seconds() / 86400.0


def recency_score(value: str, half_life_days: float = 30.0) -> float:
    days = max(age_days(value), 0.0)
    return 0.5 ** (days / max(half_life_days, 0.001))
