#!/usr/bin/env python3
"""Register AGP-1 with the live PEDS brain through the brain's own library
API (proof_transformer.peds.brain): a hypothesis record in hypotheses.jsonl
plus a journal entry in the brain journal. No brain files are hand-edited;
all writes go through brain.record_hypotheses / brain.append_brain_journal,
the same functions the brain's own step uses.

Investigation note (2026-11-05): PEDS has no external campaign registry —
peds_autopilot.py --campaign is a fixed enum of PEDS-internal training
profiles (10x3, 3gen, 5x2, mutation-stress, none), and the brain's action
-> command map is hardwired to peds_training_campaign.py / autopilot.
The hypothesis + journal records are the supported ledger surface through
which the brain owns this campaign; the campaign runner additionally
journals seed completions and the verdict through the same API.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAMPAIGN_DIR.parents[1]
PROOF_ROOT = REPO_ROOT / "proof_transformer"
BRAIN_ROOT = REPO_ROOT / "peds_brain_live"
CONFIG = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))


def main() -> int:
    sys.path.insert(0, str(PROOF_ROOT / "src"))
    from proof_transformer.peds import brain as peds_brain

    state = peds_brain.load_brain_state(BRAIN_ROOT)
    hypothesis = peds_brain._hypothesis(
        state,
        "agp_campaign",
        "Bred-only genomes (mutation/crossover from random founders, lineage-verified) "
        "can solve the file_not_found_summary diagnosis family under hidden-final discipline.",
        "Every prior BTA diagnosis champion was produced by the typed-synthesis portfolio "
        "(used_evolutionary_search: false). Whether breeding itself works in this domain is "
        "the founding open question of Austin's AGP program; AGP-1 answers it under a frozen gate.",
        {
            "prereg_repo_copy": str(CAMPAIGN_DIR / "PREREG-AGP1.md"),
            "campaign_dir": str(CAMPAIGN_DIR),
            "runner": str(CAMPAIGN_DIR / "agp1_runner.py"),
            "engine_sha256": CONFIG["engine_sha256_frozen"],
            "approved_by": "Austin, 2026-11-05 03:50 EST",
        },
        {
            "arm": "primary_bred_only",
            "task": CONFIG["task"],
            "seeds": CONFIG["primary"]["seeds"],
            "pop_size": CONFIG["primary"]["pop_size"],
            "generations": CONFIG["primary"]["generations"],
            "gate": CONFIG["gate"],
        },
        "A verdict (PASS/FAIL/INCONCLUSIVE) with per-seed gate numbers, fitness curves and "
        "lineage records in campaigns/agp1/runs/, journaled back to this brain at each seed "
        "completion and at the verdict.",
    )
    records = peds_brain.record_hypotheses([hypothesis], root=BRAIN_ROOT)
    entry = peds_brain.append_brain_journal(
        "campaign_registered",
        "AGP-1 registered: bred-only file_not_found_summary campaign (PREREG-AGP1, frozen) "
        "is now a campaign of this brain; runner journals seed results and the verdict here.",
        evidence={
            "campaign_id": CONFIG["campaign_id"],
            "hypothesis_id": records[0].get("hypothesis_id"),
            "campaign_dir": str(CAMPAIGN_DIR),
            "prereg": str(CAMPAIGN_DIR / "PREREG-AGP1.md"),
        },
        root=BRAIN_ROOT,
    )
    print(json.dumps({"hypothesis": records[0], "journal_entry": entry}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
