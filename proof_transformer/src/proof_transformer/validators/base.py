from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..ledger.models import DecoderPolicy, TrainingCandidate, ValidatorResult
from ..learning.ranker import RankerWeights


@dataclass
class ValidationContext:
    conn: sqlite3.Connection
    policy_version: int
    policy: DecoderPolicy
    ranker_version: int
    ranker_weights: RankerWeights
    event_context: dict = field(default_factory=dict)


class BaseValidator(ABC):
    name: str
    required: bool = False

    @abstractmethod
    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        raise NotImplementedError
