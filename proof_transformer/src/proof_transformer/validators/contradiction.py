from __future__ import annotations

from ..ids import new_id
from ..ledger.models import TrainingCandidate, ValidatorResult
from ..memory.embeddings import tokenize
from ..timeutils import utc_now
from .base import BaseValidator, ValidationContext


class ContradictionValidator(BaseValidator):
    name = "ContradictionValidator"
    required = True

    def validate(self, candidate: TrainingCandidate, context: ValidationContext) -> ValidatorResult:
        proposed = candidate.proposed_change
        kind = proposed.get("kind")
        name = proposed.get("name")
        content = proposed.get("content", candidate.chosen)
        if kind == "contradiction":
            return ValidatorResult(
                id=new_id("val"),
                candidate_id=candidate.id,
                validator_name=self.name,
                passed=True,
                score=0.9,
                reason="Contradiction event is explicitly scoped as a contradiction record.",
                metadata={},
                created_at=utc_now(),
            )

        rows = []
        if kind and name:
            rows = context.conn.execute(
                "SELECT * FROM memory_nodes WHERE kind = ? AND name = ?",
                (kind, name),
            ).fetchall()
        conflict = False
        conflict_reason = "No direct contradiction found."
        for row in rows:
            old_tokens = set(tokenize(row["content"]))
            new_tokens = set(tokenize(content))
            if old_tokens and new_tokens:
                overlap = len(old_tokens & new_tokens) / max(len(old_tokens | new_tokens), 1)
            else:
                overlap = 1.0 if row["content"].strip() == content.strip() else 0.0
            if overlap < 0.30:
                conflict = True
                conflict_reason = f"Existing {kind}:{name} conflicts with low semantic overlap ({overlap:.2f})."
                break

        if not conflict and name:
            edge_rows = context.conn.execute(
                """
                SELECT e.* FROM memory_edges e
                JOIN memory_nodes s ON s.id = e.source_id
                JOIN memory_nodes t ON t.id = e.target_id
                WHERE e.relation = 'contradicts' AND (s.name = ? OR t.name = ?)
                """,
                (name, name),
            ).fetchall()
            if edge_rows:
                conflict = True
                conflict_reason = "Candidate touches a node already marked by contradiction edges."

        return ValidatorResult(
            id=new_id("val"),
            candidate_id=candidate.id,
            validator_name=self.name,
            passed=not conflict,
            score=0.88 if not conflict else 0.35,
            reason=conflict_reason,
            metadata={"matched_rows": len(rows)},
            created_at=utc_now(),
        )
