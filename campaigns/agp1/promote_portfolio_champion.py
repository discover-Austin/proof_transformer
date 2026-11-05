#!/usr/bin/env python3
"""AGP-1 FAIL path (PREREG-AGP1, Austin's decision 2026-11-05 04:55 EST):
register the portfolio champion for file_not_found_summary in BTA's promoted
pool, and journal the FAIL-path execution into the live PEDS brain.

The champion package itself was produced by BTA's own normal path during the
campaign runner's secondary (portfolio) arm: run_file_task_evolution ->
evolve_file_task -> save_file_agent_package, with the engine's promotion
decision (scorecard "promoted": true) made by the engine. That stock run is
also where the champion's single hidden_final evaluation happened (engine
semantics: computed once for the distilled program at the end of evolve;
scorecard records hidden_final_used_in_search: false).

This script performs NO search and NO evaluation of any kind. It registers
the already-saved, already-evaluated package into the real pool root
(C:\\breeding-tool-agent\\file_agents) under the next pool id, using the same
numbering rule as the engine's next_agent_dir (breeding_tool_agents.py lines
762-775). Only genome.json's agent_id field is rewritten (to the pool id the
engine itself would have assigned had the run's root been the pool); every
other record file is copied byte-for-byte.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAMPAIGN_DIR.parents[1]
PROOF_ROOT = REPO_ROOT / "proof_transformer"
BRAIN_ROOT = REPO_ROOT / "peds_brain_live"
POOL_ROOT = Path(r"C:\breeding-tool-agent\file_agents")
TASK = "file_not_found_summary"
CHAMPION_SEED = 7  # first seed in the frozen order whose package the engine promoted


def brain_journal(event_type: str, summary: str, evidence: dict) -> None:
    try:
        sys.path.insert(0, str(PROOF_ROOT / "src"))
        from proof_transformer.peds import brain as peds_brain

        peds_brain.append_brain_journal(event_type, summary, evidence=evidence, root=BRAIN_ROOT)
        print(f"[promote] brain journal: {event_type}")
    except Exception as exc:  # noqa: BLE001
        print(f"[promote] brain journal write failed ({event_type}): {exc}")


def next_pool_dir(root: Path) -> tuple[int, Path]:
    """Mirror of breeding_tool_agents.next_agent_dir (engine lines 762-775)."""
    root.mkdir(parents=True, exist_ok=True)
    max_id = 0
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("agent_"):
            try:
                max_id = max(max_id, int(child.name.split("_", 1)[1]))
            except ValueError:
                pass
    agent_id = max_id + 1
    path = root / f"agent_{agent_id:06d}"
    path.mkdir()
    (path / "traces").mkdir()
    return agent_id, path


def main() -> int:
    run_dirs = sorted((CAMPAIGN_DIR / "runs").glob("agp1-portfolio-*"))
    if not run_dirs:
        raise RuntimeError("no portfolio run dir found")
    run_dir = run_dirs[-1]
    src = run_dir / "portfolio" / f"seed_{CHAMPION_SEED}" / "agents" / "agent_000001"
    if not src.exists():
        raise RuntimeError(f"champion package missing: {src}")

    scorecard = json.loads((src / "scorecard.json").read_text(encoding="utf-8"))
    checks = {
        "task": scorecard.get("task") == TASK,
        "promoted": scorecard.get("promoted") is True,
        "train": scorecard.get("train_passed") == scorecard.get("train_total") == 20,
        "validation": scorecard.get("validation_passed") == scorecard.get("validation_total") == 20,
        "audit": scorecard.get("audit_passed") == scorecard.get("audit_total") == 40,
        "stress": scorecard.get("stress_passed") == scorecard.get("stress_total") == 20,
        "hidden_final": scorecard.get("hidden_final_passed") == scorecard.get("hidden_final_total") == 20,
        "hidden_not_in_search": scorecard.get("hidden_final_used_in_search") is False,
        "no_destructive_writes": scorecard.get("after_cost", {}).get("destructive_write_attempts") == 0,
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"champion package fails promotion preconditions: {failed}")

    # Refuse double promotion: the pool must not already hold this task.
    for pkg in POOL_ROOT.glob("agent_*"):
        sc_path = pkg / "scorecard.json"
        if sc_path.exists():
            other = json.loads(sc_path.read_text(encoding="utf-8"))
            if other.get("task") == TASK:
                raise RuntimeError(f"pool already holds {TASK}: {pkg.name} - refusing to double-promote")

    agent_id, dest = next_pool_dir(POOL_ROOT)
    genome = json.loads((src / "genome.json").read_text(encoding="utf-8"))
    genome["agent_id"] = f"agent_{agent_id:06d}"
    (dest / "genome.json").write_text(json.dumps(genome, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in ("lineage.json", "program.txt", "scorecard.json"):
        shutil.copyfile(src / name, dest / name)
    trace_src = src / "traces" / "audit_trace.json"
    if trace_src.exists():
        shutil.copyfile(trace_src, dest / "traces" / "audit_trace.json")

    registration = {
        "campaign_id": "AGP-1",
        "action": "fail_path_portfolio_champion_pool_registration",
        "task": TASK,
        "champion_seed": CHAMPION_SEED,
        "source_package": str(src),
        "pool_package": str(dest),
        "pool_agent_id": f"agent_{agent_id:06d}",
        "program": (dest / "program.txt").read_text(encoding="utf-8").strip().splitlines(),
        "scores": {
            "train": [scorecard["train_passed"], scorecard["train_total"]],
            "validation": [scorecard["validation_passed"], scorecard["validation_total"]],
            "audit": [scorecard["audit_passed"], scorecard["audit_total"]],
            "stress": [scorecard["stress_passed"], scorecard["stress_total"]],
            "hidden_final": [scorecard["hidden_final_passed"], scorecard["hidden_final_total"]],
        },
        "winning_search_strategy": scorecard.get("winning_search_strategy"),
        "used_evolutionary_search": scorecard.get("used_evolutionary_search"),
        "destructive_write_attempts": scorecard.get("after_cost", {}).get("destructive_write_attempts"),
        "hidden_final_note": (
            "Evaluated exactly once for this program, by the BTA engine inside the stock "
            "evolve (run_file_task_evolution) during the portfolio arm; used_in_search=false. "
            "No re-audit or second evaluation was performed at registration."
        ),
    }
    (run_dir / "pool_registration.json").write_text(
        json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(registration, indent=2, sort_keys=True))

    brain_journal(
        "agp_failpath_executed",
        (
            f"AGP-1 FAIL path executed (Austin, 2026-11-05 04:55 EST): portfolio champion for "
            f"{TASK} promoted into BTA's pool as agent_{agent_id:06d} (typed_skeleton, "
            f"20/20 20/20 40/40 20/20, hidden_final 20/20 from the engine's single stock-run "
            f"evaluation). Bred-only search is STOPPED for diagnosis families; breeding budget "
            f"redirected to the arena genome line (breeding-tool-agent-full-breeder). "
            f"Scoping: ~/workspace/projects/agp/AGP2-ARENA-SCOPING.md."
        ),
        {
            "campaign_id": "AGP-1",
            "decision_by": "Austin, 2026-11-05 04:55 EST",
            "pool_package": str(dest),
            "registration": str(run_dir / "pool_registration.json"),
            "portfolio_report": str(run_dir / "report.md"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
