from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.agent.actions import parse_csv_ids
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python examples/train_ranker_from_feedback.py EVENT_ID [useful_csv] [irrelevant_csv]')
    useful = parse_csv_ids(sys.argv[2] if len(sys.argv) > 2 else "")
    irrelevant = parse_csv_ids(sys.argv[3] if len(sys.argv) > 3 else "")
    print(
        json.dumps(
            ProofTransformerOrchestrator().feedback(
                retrieval_event_id=sys.argv[1],
                useful_chunk_ids=useful,
                irrelevant_chunk_ids=irrelevant,
            ),
            indent=2,
        )
    )
