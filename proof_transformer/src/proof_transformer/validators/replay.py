from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..learning.evals import weight_distance
from ..learning.ranker import RankerWeights
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class ReplayValidator(BaseValidator):
    name = "ReplayValidator"
    required = False

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        proposed = candidate.proposed_change
        score = 0.82
        passed = True
        reason = "Built-in replay checks passed."
        metadata: dict = {}

        if candidate.target == "ranker":
            weights = RankerWeights.from_dict(proposed.get("new_weights", {})).as_dict()
            total = sum(weights.values())
            delta_l1 = weight_distance(context.ranker_weights.as_dict(), weights)
            metadata = {"sum": total, "delta_l1": delta_l1}
            if abs(total - 1.0) > 1e-6:
                passed = False
                score = 0.2
                reason = "Ranker weights are not normalized."
            elif weights["vector_similarity"] < 0.05:
                passed = False
                score = 0.25
                reason = "Ranker replay would underweight semantic retrieval too aggressively."
            elif delta_l1 > 0.55:
                passed = False
                score = 0.4
                reason = "Ranker change is too large for a single replay-safe update."
            else:
                score = 0.88

        if candidate.target == "policy":
            updates = candidate.proposed_change.get("updates", {})
            future = context.policy.apply_updates(updates)
            if future.avoid_placeholders < 0.7 or future.no_fake_autonomy < 0.7:
                passed = False
                score = 0.2
                reason = "Policy replay would weaken core safety constraints."
            else:
                metadata = {"future_policy": future.as_dict()}
                score = 0.86

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
