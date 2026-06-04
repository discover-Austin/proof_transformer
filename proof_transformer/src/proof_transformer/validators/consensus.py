from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from ..config import load_settings
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..learning.ranker import RankerStore
from ..ledger.policy import PolicyStore
from .base import ValidationContext
from .contradiction import ContradictionValidator
from .evidence import EvidenceValidator
from .outcome import OutcomeValidator
from .preference import PreferenceValidator
from .replay import ReplayValidator
from .scope import ScopeValidator


@dataclass
class ConsensusDecision:
    status: str
    consensus_score: float
    results: list[ValidatorResult]

    def summary(self) -> dict:
        return {
            "status": self.status,
            "consensus_score": self.consensus_score,
            "results": [result.as_record() for result in self.results],
        }


class ConsensusEngine:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.settings = load_settings()
        self.policy_store = PolicyStore(conn)
        self.ranker_store = RankerStore(conn)
        self.validators = [
            PreferenceValidator(),
            ContradictionValidator(),
            ScopeValidator(),
            ReplayValidator(),
            EvidenceValidator(),
            OutcomeValidator(),
        ]

    def evaluate(self, candidate: TrainingCandidate, event_context: dict | None = None) -> ConsensusDecision:
        context = self._build_context(event_context or {})
        results = [validator.validate(candidate, context) for validator in self.validators]
        score = self._weighted_score(results)
        required = [result for result in results if self._validator_required(result.validator_name)]
        all_required_pass = all(result.passed for result in required)
        min_required = min((result.score for result in required), default=1.0)
        if all_required_pass and min_required >= 0.65 and score >= 0.75:
            status = "accepted"
        elif (not all_required_pass) or min_required < 0.65 or score < 0.60:
            status = "rejected"
        else:
            status = "quarantined"
        return ConsensusDecision(status=status, consensus_score=score, results=results)

    def persist_results(self, results: list[ValidatorResult]) -> None:
        with self.conn:
            for result in results:
                self.conn.execute(
                    """
                    INSERT INTO validator_results(
                        id, candidate_id, validator_name, passed, score,
                        reason, metadata_json, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.id,
                        result.candidate_id,
                        result.validator_name,
                        1 if result.passed else 0,
                        result.score,
                        result.reason,
                        json.dumps(result.metadata),
                        result.created_at,
                    ),
                )

    def _weighted_score(self, results: list[ValidatorResult]) -> float:
        weights = self.settings.validator_weights
        total_weight = 0.0
        total_score = 0.0
        for result in results:
            weight = float(weights.get(result.validator_name, 0.0))
            total_weight += weight
            total_score += weight * float(result.score)
        return total_score / total_weight if total_weight else 0.0

    def _validator_required(self, validator_name: str) -> bool:
        return validator_name in {"PreferenceValidator", "ContradictionValidator", "ScopeValidator"}

    def _build_context(self, event_context: dict) -> ValidationContext:
        policy_version, policy = self.policy_store.get_current()
        ranker_version, ranker_weights = self.ranker_store.get_current()
        return ValidationContext(
            conn=self.conn,
            policy_version=policy_version,
            policy=policy,
            ranker_version=ranker_version,
            ranker_weights=ranker_weights,
            event_context=event_context,
        )
