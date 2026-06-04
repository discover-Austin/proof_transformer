from __future__ import annotations

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


class HealthTests(BrainTestCase):
    def test_health_reports_fallback_backend_and_db(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        health = orchestrator.health()
        self.assertTrue(health["db_exists"])
        self.assertTrue(health["ledger"]["ok"])
        self.assertEqual(health["model"]["backend"], "fallback")
        self.assertTrue(health["model"]["available"])
