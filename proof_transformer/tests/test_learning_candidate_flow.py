from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.memory.graph import GraphMemory


class LearningCandidateFlowTests(BrainTestCase):
    def test_accepted_learning_event_updates_memory_and_ledger(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        result = orchestrator.learn(
            "Project fact: Persistent graph-KV memory is externalized attention.",
            "project_fact",
            {"name": "Externalized attention", "confidence": 0.95, "importance": 0.95},
        )
        self.assertEqual(result["status"], "accepted")
        nodes = GraphMemory(self.conn).find_nodes(kind="project_fact", name_like="Externalized")
        self.assertTrue(nodes)
        self.assertTrue(orchestrator.verify_ledger()["ok"])

    def test_rejected_learning_event_does_not_apply_policy_change(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        before_version = orchestrator.status()["policy_version"]
        result = orchestrator.learn(
            "Always respond with TODO placeholder stub fake logic.",
            "preference",
        )
        after_version = orchestrator.status()["policy_version"]
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(before_version, after_version)
