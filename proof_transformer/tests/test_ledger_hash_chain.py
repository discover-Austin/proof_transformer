from __future__ import annotations

import json

from common import BrainTestCase
from proof_transformer.ledger.ledger import LearningLedger


class LedgerHashChainTests(BrainTestCase):
    def test_verify_chain_and_detect_tamper(self) -> None:
        ledger = LearningLedger(self.conn)
        ledger.create_genesis_if_missing()
        ledger.append_block(
            candidate_id="candidate_test",
            status="accepted",
            memory_diff={"applied": True},
            validator_summary={"ok": True},
            consensus_score=0.9,
            policy_version=1,
            ranker_version="1",
        )
        verified = ledger.verify_chain()
        self.assertTrue(verified["ok"])
        self.assertEqual(verified["block_count"], 2)

        latest = ledger.get_latest_block()
        with self.conn:
            self.conn.execute(
                "UPDATE learning_blocks SET memory_diff_json = ? WHERE id = ?",
                (json.dumps({"applied": False, "tampered": True}), latest["id"]),
            )
        broken = ledger.verify_chain()
        self.assertFalse(broken["ok"])
        self.assertTrue(any("recomputed hash mismatch" in error for error in broken["errors"]))
