from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python examples/plan_repair.py WORKSPACE COMMAND [ARGS...]")
    print(json.dumps(ProofTransformerOrchestrator().plan_repair(sys.argv[1], sys.argv[2:]), indent=2))
