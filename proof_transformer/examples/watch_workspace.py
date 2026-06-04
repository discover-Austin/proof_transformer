from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.agent.workspace_watcher import WorkspaceWatcher


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python examples/watch_workspace.py PATH [cycles]")
    cycles = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    watcher = WorkspaceWatcher(ProofTransformerOrchestrator(), sys.argv[1], poll_interval=0.0)
    print(json.dumps(watcher.serve(max_cycles=cycles), indent=2))
