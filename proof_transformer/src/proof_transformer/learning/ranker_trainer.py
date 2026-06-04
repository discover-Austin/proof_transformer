from __future__ import annotations

import json
import sqlite3

from ..config import load_settings
from .evals import weight_distance
from .ranker import FEATURES, RankerStore, RankerWeights


class RankerTrainer:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.settings = load_settings()
        self.store = RankerStore(conn)

    def build_feedback_change(
        self,
        retrieval_event_id: str,
        useful_chunk_ids: list[str],
        irrelevant_chunk_ids: list[str],
    ) -> tuple[str, dict]:
        row = self.conn.execute("SELECT * FROM retrieval_events WHERE id = ?", (retrieval_event_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown retrieval event: {retrieval_event_id}")
        scores_map = json.loads(row["scores_json"])
        old_version, old_weights = self.store.get_current()
        new_weights, metrics = self.apply_feedback(
            current_weights=old_weights,
            scores_map=scores_map,
            useful_chunk_ids=useful_chunk_ids,
            irrelevant_chunk_ids=irrelevant_chunk_ids,
            learning_rate=self.settings.ranker_learning_rate,
        )
        source_text = (
            f"Retrieval feedback for {retrieval_event_id}: useful={','.join(useful_chunk_ids) or 'none'}; "
            f"irrelevant={','.join(irrelevant_chunk_ids) or 'none'}."
        )
        proposed_change = {
            "target": "ranker",
            "event_type": "retrieval_feedback",
            "retrieval_event_id": retrieval_event_id,
            "old_version": old_version,
            "old_weights": old_weights.as_dict(),
            "new_weights": new_weights.as_dict(),
            "metrics": metrics,
            "high_impact": metrics["delta_l1"] > 0.35,
            "evidence": source_text,
        }
        return source_text, proposed_change

    def apply_feedback(
        self,
        *,
        current_weights: RankerWeights,
        scores_map: dict,
        useful_chunk_ids: list[str],
        irrelevant_chunk_ids: list[str],
        learning_rate: float,
    ) -> tuple[RankerWeights, dict]:
        useful = [scores_map[chunk_id]["score_parts"] for chunk_id in useful_chunk_ids if chunk_id in scores_map]
        irrelevant = [scores_map[chunk_id]["score_parts"] for chunk_id in irrelevant_chunk_ids if chunk_id in scores_map]
        deltas = {feature: 0.0 for feature in FEATURES}
        for feature in FEATURES:
            useful_signal = sum(float(parts.get(feature, 0.0)) for parts in useful) / len(useful) if useful else 0.0
            irrelevant_signal = (
                sum(float(parts.get(feature, 0.0)) for parts in irrelevant) / len(irrelevant) if irrelevant else 0.0
            )
            deltas[feature] = useful_signal - irrelevant_signal

        updated = current_weights.as_dict()
        for feature, delta in deltas.items():
            updated[feature] = max(0.0, updated[feature] + float(learning_rate) * delta)
        normalized = RankerWeights.from_dict(updated)
        metrics = {
            "learning_rate": learning_rate,
            "useful_count": len(useful),
            "irrelevant_count": len(irrelevant),
            "delta_by_feature": deltas,
            "delta_l1": weight_distance(current_weights.as_dict(), normalized.as_dict()),
        }
        return normalized, metrics
