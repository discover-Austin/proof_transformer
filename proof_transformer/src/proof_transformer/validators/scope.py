from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class ScopeValidator(BaseValidator):
    name = "ScopeValidator"
    required = True

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        proposed = candidate.proposed_change
        word_count = len(candidate.chosen.split())
        passed = True
        score = 0.88
        reason = "Change is acceptably scoped."
        metadata: dict = {"word_count": word_count}

        if candidate.target == "policy":
            change_count = len(proposed.get("updates", {}))
            metadata["change_count"] = change_count
            if change_count > 3 and word_count < 20:
                passed = False
                score = 0.3
                reason = "Policy change is too broad for the amount of evidence."
        elif candidate.target == "ranker":
            metrics = proposed.get("metrics", {})
            evidence_count = int(metrics.get("useful_count", 0)) + int(metrics.get("irrelevant_count", 0))
            delta_l1 = float(metrics.get("delta_l1", 0.0))
            metadata.update({"evidence_count": evidence_count, "delta_l1": delta_l1})
            if delta_l1 > 0.35 and evidence_count < 2:
                passed = False
                score = 0.25
                reason = "Ranker update is overbroad relative to the narrow feedback evidence."
            elif evidence_count == 0:
                passed = False
                score = 0.2
                reason = "No retrieval evidence was supplied."
        else:
            if word_count < 4 and candidate.event_type != "retrieval_feedback":
                passed = False
                score = 0.45
                reason = "Observation is too narrow to justify a stored update."

        return ValidatorResult(
            id=new_id("val"),
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metadata=metadata,
            created_at=utc_now(),
        )
