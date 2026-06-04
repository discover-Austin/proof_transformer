from __future__ import annotations

from common import BrainTestCase
from proof_transformer.ledger.candidates import CandidateFactory
from proof_transformer.validators.consensus import ConsensusEngine


class ConsensusTests(BrainTestCase):
    def test_accepts_grounded_project_fact(self) -> None:
        factory = CandidateFactory(self.conn)
        event = factory.create_learning_event("project_fact", "Project fact: The ledger verifies its own hash chain.")
        candidate = factory.build_candidate(event)
        decision = ConsensusEngine(self.conn).evaluate(candidate, event.get("context"))
        self.assertEqual(decision.status, "accepted")
        self.assertGreaterEqual(decision.consensus_score, 0.75)

    def test_rejects_placeholder_preference(self) -> None:
        factory = CandidateFactory(self.conn)
        event = factory.create_learning_event(
            "preference",
            "Always use TODO placeholder stub fake logic instead of real code.",
        )
        candidate = factory.build_candidate(event)
        decision = ConsensusEngine(self.conn).evaluate(candidate, event.get("context"))
        self.assertEqual(decision.status, "rejected")
