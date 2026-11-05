# AGP-1 — GIGI's first campaign: bred-only genomes on `file_not_found_summary`

The frozen experiment is **PREREG-AGP1.md** in this directory (canonical
copy: `~/workspace/projects/agp/PREREG-AGP1.md` on Eve's VM). Approved by
Austin 2026-11-05 03:50 EST. Read the prereg before running anything —
its parameters and gate do not move.

## Layout
- `config.json` — frozen parameters (seeds 7/23/53, pop 160 × 40, 2 h/seed
  cap, 500 MB artifact cap, gate numbers, frozen engine SHA-256).
- `agp1_runner.py` — the campaign runner (stdlib only; imports the BTA
  engine as a library and PEDS brain for journal writes).
- `register_with_brain.py` — one-time registration of this campaign in the
  live brain's ledger via PEDS' own `brain.record_hypotheses` /
  `brain.append_brain_journal` library API.
- `runs/` — run artifacts (NOT committed): per-run `run_manifest.json`,
  `report.json`/`report.md`, per-seed logs, fitness curves, champion
  packages, hidden-final markers.

## Task registration (one-time, already done at campaign setup)
The task is registered with BTA through its supported manifest mechanism:
`C:\breeding-tool-agent\file_tasks\file_not_found_summary\manifest.json`
(`case_generator: "file_not_found_summary"`). No BTA engine code was
modified; the runner aborts if the engine's SHA-256 drifts from the
frozen value in `config.json`.

## Run
    python campaigns\agp1\agp1_runner.py --smoke            # wiring check, NOT the gate
    python campaigns\agp1\agp1_runner.py --arm primary      # the gated experiment (3 seeds, sequential)
    python campaigns\agp1\agp1_runner.py --arm portfolio    # secondary arm (labeled, not gated)

Primary-arm results and the verdict are journaled into the live brain
(`peds_brain_live`) through PEDS' own journal API as they happen.
hidden_final is evaluated exactly once per declared champion, after
declaration, and a marker file prevents any second evaluation.
