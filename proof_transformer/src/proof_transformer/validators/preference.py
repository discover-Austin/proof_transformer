from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class PreferenceValidator(BaseValidator):
    name = "PreferenceValidator"
    required = True

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        text = " ".join(
            [
                candidate.instruction,
                candidate.chosen,
                candidate.rejected or "",
                str(candidate.proposed_change),
            ]
        ).lower()
        banned = [word for word in ("todo", "placeholder", "stub", "fake logic", "mock logic") if word in text]
        passed = True
        score = 0.88
        reason = "Aligned with durable preferences and anti-placeholder policy."
        if banned:
            passed = False
            score = 0.15
            reason = f"Contains banned placeholder/fake-logic signals: {', '.join(banned)}"
        if candidate.target == "policy":
            updates = candidate.proposed_change.get("updates", {})
            if float(updates.get("avoid_placeholders", context.policy.avoid_placeholders)) < 0.7:
                passed = False
                score = min(score, 0.2)
                reason = "Policy change weakens avoid_placeholders below safe threshold."
            if float(updates.get("no_fake_autonomy", context.policy.no_fake_autonomy)) < 0.7:
                passed = False
                score = min(score, 0.2)
                reason = "Policy change weakens no_fake_autonomy below safe threshold."
            if updates:
                score = max(score, 0.9 if passed else score)
        return ValidatorResult(
            id=new_id("val"),
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metadata={"banned_terms": banned},
            created_at=utc_now(),
        )
