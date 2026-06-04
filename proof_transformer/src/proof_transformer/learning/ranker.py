from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass

from ..ids import new_id
from ..timeutils import utc_now


FEATURES = (
    "vector_similarity",
    "keyword_overlap",
    "graph_proximity",
    "importance",
    "confidence",
    "recency",
    "prior_feedback",
)


@dataclass(frozen=True)
class RankerWeights:
    vector_similarity: float = 0.40
    keyword_overlap: float = 0.15
    graph_proximity: float = 0.15
    importance: float = 0.10
    confidence: float = 0.10
    recency: float = 0.05
    prior_feedback: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def normalized(self) -> "RankerWeights":
        return RankerWeights.from_dict(self.as_dict())

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "RankerWeights":
        cleaned = {feature: max(0.0, float(data.get(feature, 0.0))) for feature in FEATURES}
        total = sum(cleaned.values())
        if total <= 0.0:
            cleaned = cls().as_dict()
            total = sum(cleaned.values())
        normalized = {feature: cleaned[feature] / total for feature in FEATURES}
        return cls(**normalized)

    def score(self, parts: dict[str, float]) -> float:
        weights = self.as_dict()
        return sum(float(weights[feature]) * float(parts.get(feature, 0.0)) for feature in FEATURES)


class RankerStore:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def ensure_default(self) -> tuple[int, RankerWeights]:
        row = self.conn.execute("SELECT version, weights_json FROM ranker_weights ORDER BY version DESC LIMIT 1").fetchone()
        if row:
            return int(row["version"]), RankerWeights.from_dict(json.loads(row["weights_json"]))
        weights = RankerWeights().normalized()
        with self.conn:
            self.conn.execute(
                "INSERT INTO ranker_weights(id, version, weights_json, metrics_json, created_at) VALUES(?, 1, ?, ?, ?)",
                (new_id("ranker"), json.dumps(weights.as_dict()), json.dumps({"type": "default"}), utc_now()),
            )
        return 1, weights

    def get_current(self) -> tuple[int, RankerWeights]:
        return self.ensure_default()

    def save(self, weights: RankerWeights, metrics: dict | None = None) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(version), 0) AS version FROM ranker_weights").fetchone()
        version = int(row["version"]) + 1
        normalized = weights.normalized()
        with self.conn:
            self.conn.execute(
                "INSERT INTO ranker_weights(id, version, weights_json, metrics_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (new_id("ranker"), version, json.dumps(normalized.as_dict()), json.dumps(metrics or {}), utc_now()),
            )
        return version

    def list_versions(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM ranker_weights ORDER BY version ASC").fetchall()
        return [dict(row) for row in rows]
