from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class EvidenceValidator(BaseValidator):
    name = "EvidenceValidator"
    required = False

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        proposed = candidate.proposed_change
        evidence = str(proposed.get("evidence") or candidate.chosen).strip()
        sources = context.event_context.get("sources") or []
        high_impact = candidate.target in {"policy", "ranker", "adapter"} or "high_impact" in candidate.tags
        enough = len(evidence.split()) >= 5 or bool(sources)
        passed = True
        score = 0.9 if enough else 0.55
        reason = "Candidate has usable source grounding." if enough else "Candidate lacks strong source grounding."
        if high_impact and not enough:
            passed = False
            score = 0.35
            reason = "High-impact change lacks evidence."
        return ValidatorResult(
            id=new_id("val"),
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=passed,
            score=score,
            reason=reason,
            metadata={"high_impact": high_impact, "sources": len(sources)},
            created_at=utc_now(),
        )
