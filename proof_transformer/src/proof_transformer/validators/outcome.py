from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class OutcomeValidator(BaseValidator):
    name = "OutcomeValidator"
    required = False

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        event_type = candidate.event_type
        score = 0.7
        reason = "Neutral outcome evidence."
        if event_type == "accepted_output" or "user_approved" in candidate.tags:
            score = 0.95
            reason = "Explicit user approval boosts confidence."
        elif event_type == "bug_fix" and candidate.proposed_change.get("tests_passed"):
            score = 0.93
            reason = "Test-passing bug fix boosts confidence."
        elif event_type == "failed_output":
            score = 0.85
            reason = "Observed failure supports tightening policy."
        elif event_type == "general_observation":
            score = 0.6
            reason = "General observation remains weak until reinforced."
        return ValidatorResult(
            id=new_id("val"),
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=score >= 0.55,
            score=score,
            reason=reason,
            metadata={"event_type": event_type},
            created_at=utc_now(),
        )
