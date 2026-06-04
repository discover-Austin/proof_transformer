from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.agent.worker import QueueWorker


if __name__ == "__main__":
    max_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    worker = QueueWorker(ProofTransformerOrchestrator(), poll_interval=0.0)
    print(json.dumps(worker.serve(max_cycles=max_cycles), indent=2))
