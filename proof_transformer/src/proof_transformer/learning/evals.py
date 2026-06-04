from __future__ import annotations


def precision_at_k(ranked_ids: list[str], useful_ids: list[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    window = ranked_ids[:k]
    if not window:
        return 0.0
    useful = set(useful_ids)
    return len([item for item in window if item in useful]) / len(window)


def weight_distance(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    keys = set(old_weights) | set(new_weights)
    return sum(abs(float(old_weights.get(key, 0.0)) - float(new_weights.get(key, 0.0))) for key in keys)
