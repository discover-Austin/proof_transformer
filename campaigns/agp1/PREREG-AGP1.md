# PREREG-AGP1 — Bred-only genomes on `file_not_found_summary` (BTA engine, PEDS campaign AGP-1)

**Status: FROZEN / REGISTERED.**
**Approved by Austin 2026-11-05 03:50 EST** ("Wire the AGP experiment in as her first real campaign").
Registered by Eve's wiring session 2026-11-05 (Acer `gigi`), before any search ran. The frozen
experiment text below is §6 of `~/workspace/projects/agp/AGP-STACK-MAP.md` verbatim (its
internal "DRAFT" wording is superseded by this registration block). Parameters and gates do
not move after the verdict evaluation is first touched; amendments are appended at the
bottom with timestamps and reasons, per §6's own rules.

---

## Registration record — frozen splits and engine identity (recorded BEFORE any search)

§6 requires: "split seeds recorded in the run manifest before any search starts."
Verified on the Acer 2026-11-05: the task's generator
(`generated_file_not_found_summary_cases`, breeding_tool_agents.py line 1729) is **fully
deterministic and takes no seed argument** — there are no RNG split seeds to draw. The
splits are fixed by the generator source itself, so freezing = pinning the generator's
identity and case-index ranges:

- Engine file: `C:\breeding-tool-agent\breeding_tool_agents.py`
- Engine size: 370,283 bytes
- **Engine SHA-256 at freeze: `397a5fda96986e34a358b506fbcfe07908d553be3b94bb7c49bbd5f5d0912ac1`**
- Split construction (case index ranges in `missing_file_case`): train = idx 0–19 (20),
  validation = idx 20–39 (20), audit = idx 40–79 (40), stress = idx 80–99 (20),
  hidden_final = idx 100–119 (20).
- Split sizes verified live from the generator: train 20 / validation 20 / audit 40 /
  stress 20 / hidden_final 20.
- Task registration: `file_not_found_summary` had no manifest and was not in `FILE_TASKS`;
  it is registered through BTA's supported manifest mechanism
  (`file_tasks/file_not_found_summary/manifest.json`, `case_generator:
  "file_not_found_summary"`), modeled field-for-field on the proven sibling diagnosis
  manifests (module_not_found_summary / assertion_failure_summary).
  Manifest SHA-256 at registration: `c81208501b631abbfa03790f23f2eebbc1870cbd29da8182f91af1b50e1374d1`
  (also recorded in each run manifest at launch).
- Search seeds (unchanged from §6): **7, 23, 53**.

## Wiring findings recorded at registration (how §6 maps onto BTA's real mechanics)

1. All five §6 flags exist in BTA's argparse and set engine globals in `main()`
   (breeding_tool_agents.py lines 8261–8265): `--blind-file-fitness`,
   `--strict-file-grammar`, `--audit-hidden`, `--hidden-final`, plus
   `--no-file-seed-programs`. `FILE_MAX_LEN = 22` (line 1084). `evolve_file_task`
   defaults are exactly pop_size=160, generations=40 (line 4916), and the CLI path
   (`run_file_task_evolution`, line 5070) passes those defaults through.
2. **The stock CLI evolve cannot run the primary arm.** `evolve_file_task` always
   searches through `cegis_file_search` → `file_search_portfolio_candidates`, which
   unconditionally tries seed program → LLM proposals → subagent backend → macro
   library → skeletons → pipelines → grammar searches before evolution, with no
   disable switch. A CLI run would therefore not be breeding-only, whatever the
   flags say. The primary arm is consequently driven through the engine's own
   `evolutionary_file_search` (the function §6 itself names as the variation
   reference: "Tournament + elitism per `evolutionary_file_search`") by the
   campaign runner in the GIGI repo. No engine code is modified; the runner sets
   the same module globals the CLI flags set (`BLIND_FILE_FITNESS`,
   `STRICT_FILE_GRAMMAR`, `AUDIT_HIDDEN`, `HIDDEN_FINAL`) and calls engine
   functions as implemented.
3. Portfolio contamination inside `evolutionary_file_search` is mechanically
   absent for this task: `LLM_SEED_PROGRAMS` is empty (len 0, verified live), so
   `file_llm_proposal_programs(task)` returns []; the new task has no
   `seed_program` (and `use_seed_programs=False` besides); subagent/macro/skeleton
   code paths exist only in the portfolio function, which the primary arm never
   calls.
4. **hidden_final discipline.** `evolve_file_task` evaluates the distilled
   champion on hidden_final unconditionally at the end of every CLI evolve
   (line ~4978) — incompatible with §6's "exactly once, after a champion is
   declared." The campaign runner therefore never calls `evolve_file_task` for
   the primary arm: search, pruning (train+validation only — the `AUDIT_HIDDEN`
   semantics of `evolve_file_task` line ~4941), feature/ablation checks, and
   split scoring use engine functions that never touch hidden_final. hidden_final
   is evaluated exactly once per declared champion, by the runner's re-audit
   step, after the champion is declared on the visible splits.
5. Lineage: the EA does not log per-event mutation/crossover records. The runner
   wraps the engine's own `mutate_file_program` / `crossover_file_program`
   (call-through counters, behavior unchanged) so every variation event in the
   winning champion's ancestry is counted; a champion counts only if its
   recorded lineage shows ≥1 mutation or crossover event and a recorded
   per-generation fitness curve (the runner logs best-fitness per generation).
6. `destructive_write_attempts`: BTA's `file_agent_tool_costs` hardcodes this
   field to 0 (line 2905) — the interpreter exposes no destructive file op to
   file genomes, and WRITE_JSON targets only the case's virtual output files.
   Gate item 4 is verified via the runner's tool-cost record plus the
   diagnosis-only contract checks (program ops restricted to the manifest's
   allowed ops; no source-modification op exists in the file genome).
7. Fitness wording: §6's "− 0.02 × errors" does not appear literally in
   `file_fitness` (line 2401): fitness = mean over train cases of
   (`file_case_score` + `file_progress_score` shaping) + blind-mode diversity
   bonus (0.08 when outputs differ). The code as implemented governs; fitness
   appears in no gate, so the discrepancy cannot affect the verdict.
8. Secondary arm: `--portfolio` mode of the same runner reproduces the exact CLI
   semantics by driving the engine's `run_file_task_evolution` in-process with
   the CLI flag globals set (identical code path to the command line, minus the
   CLI's end-of-run hidden_final print semantics — its scorecard is read from
   the engine's own records and labeled PORTFOLIO / NOT GATED). Launchable, not
   launched at registration.

## Exact primary-arm invocation (per seed s ∈ {7, 23, 53})

Engine-level (what the campaign runner executes, per seed, sequentially):

    evolutionary_file_search(task="file_not_found_summary", cases=train(20),
        rng=random.Random(s), pop_size=160, generations=40, log=<run log>,
        use_seed_programs=False)
    with engine globals BLIND_FILE_FITNESS=True, STRICT_FILE_GRAMMAR=True,
    AUDIT_HIDDEN=True, HIDDEN_FINAL=True

For reference, the closest stock command line (verified to parse and to map to
the same globals/defaults — but NOT used for the primary arm, per finding 2):

    python breeding_tool_agents.py --file-task-evolve file_not_found_summary --seeds <s> --no-file-seed-programs --strict-file-grammar --blind-file-fitness --audit-hidden --hidden-final --file-agent-dir <experiment dir>

---

## 6. Pre-registered first experiment — DRAFT (Austin approves before anything runs)

Written in the frozen-gate style of
`~/workspace/projects/btc15-analysis/PREREG-MAKER.md`. Parameters and
gates below do not move after the verdict evaluation is first touched.
Amendments are allowed only before that moment, and each one is appended
at the bottom with a timestamp and a reason.

### The question
BTA's diagnosis families were solved by its search *portfolio* — the
recorded champions show `used_evolutionary_search: false`. The breeder
repo's founding claim is that this doesn't count as breeding. Can
**bred** genomes — produced only by mutation/crossover from random or
parent material, with lineage-verified genetic origin — solve a fresh
Repo Doctor diagnosis family under hidden-final discipline?

### Domain and task (fixed)
- Task: **`file_not_found_summary`** (Repo Doctor diagnosis: read
  `pytest_output.txt` for a FileNotFoundError traceback, write
  `failure_report.json`, diagnosis only, no source modification).
- Why this slice: it is the encoded next step in *both* BTA's
  `BTA_RUNNER_ROADMAP` and the full-breeder ledger; its case generator
  already exists in the BTA engine
  (`generated_file_not_found_summary_cases`), and **no promoted agent
  and no manifest exist for it yet** — it is the one diagnosis family
  no search has touched. Stated limitation, up front: the generator
  code was written 2026-06-27, so split integrity rests on freezing
  the split seeds at registration time (below), not on the generator
  being new.
- Success also completes the fifth diagnosis family, unblocking
  `repo_doctor/all_diagnosis` per BTA's own continuation path.

### Genome, operators, search config (fixed)
- Genome: BTA file genome, `FILE_MAX_LEN = 22`, typed-slot grammar
  constraints ON (`--strict-file-grammar`), blind fitness ON
  (`--blind-file-fitness`), audit hidden ON (`--audit-hidden`),
  hidden-final ON (`--hidden-final`).
- Variation: `mutate_file_program` and `crossover_file_program`
  exactly as implemented (modify 0.45 / insert 0.30 / delete else;
  single-point splice). Tournament + elitism per
  `evolutionary_file_search`.
- **Primary arm — breeding only:** `--no-file-seed-programs`; macro
  library, typed skeletons, pipelines, subagent backend, and LLM
  proposals all DISABLED. A champion counts only if its lineage shows
  ≥1 mutation or crossover event (the breeder repo's bred-survivor
  rule).
- **Secondary arm — labeled, not gated:** the standard portfolio
  (skeletons/macros allowed) on identical splits and budgets. It
  exists to measure the breeding-vs-portfolio gap, and its results
  cannot turn a primary-arm FAIL into a PASS.

### Data / splits (frozen at registration)
- Generated by the task's deterministic generator with split seeds
  recorded in the run manifest before any search starts:
  train 20 / validation 20 / audit 40 / stress 20 — matching the four
  proven diagnosis families.
- **hidden_final: 20 cases** generated from a separate held seed,
  written to the manifest store at registration, and evaluated
  **exactly once**, after a champion is declared, through the re-audit
  path. No search, pruning, cost optimization, or CEGIS counterexample
  selection may read it.

### Budget (sized to the Acer, fixed)
- 3 seeds: **7, 23, 53** (the project's standing TRIAL_SEEDS).
- Per seed: population 160 × 40 generations (the `evolve_file_task`
  defaults), single process, sequential across seeds, wall-clock cap
  **2 hours per seed** on the Acer (4-core i7-4770K per tonight's CIM
  reading; if the machine proves to be the 12400F, the cap stays —
  budgets do not grow mid-experiment).
- Runs detached, logged, never concurrent with PEDS test suites or
  brain steps. Evaluations/second and total candidate count recorded
  in the run report.
- New artifacts capped at 500 MB; failed candidates preserved per
  breeder rules but slim-checkpoint discipline (no release zips inside
  the experiment tree).

### Fitness (fixed)
`file_fitness` as implemented: mean over train cases of
(`file_case_score` exact/whitespace score + `file_progress_score`
shaping − 0.02 × errors) + blind-mode diversity bonus. Fitness guides
search only; it appears in no gate below.

### Gate (fixed)
PASS requires, on the primary arm:
1. At least **2 of 3 seeds** produce a champion.
2. Each champion scores exactly: train 20/20, validation 20/20,
   audit 40/40, stress 20/20 (file_case_score == 1.0 per case).
3. Each champion's single hidden_final evaluation ≥ **19/20**.
4. Feature/ablation requirements from the task manifest pass;
   `destructive_write_attempts == 0`; no source files modified
   (diagnosis-only contract).
5. Each champion's lineage verifies ≥1 mutation or crossover event
   and a recorded fitness curve (bred-survivor rule).
Anything else is FAIL or INCONCLUSIVE (wall-clock cap hit in ≥2 seeds
= INCONCLUSIVE), reported as-is. A negative result is a deliverable.

### Stop / scale (fixed)
- **STOP** the bred-only approach for diagnosis families if FAIL: the
  conclusion stands that typed synthesis is the productive mechanism
  for this domain, breeding budget is redirected to the arena genome
  line (where bred survivors already exist), and the portfolio
  champion for this task (secondary arm) is promoted through the
  normal BTA path so the roadmap still advances.
- **SCALE** if PASS: register each champion as a PEDS candidate
  (adapter in §3.2) for replay/shadow against the incumbent diagnosis
  agents, and clone this frozen template unchanged for
  `no_traceback_failure_summary`.
- Either way: results, fitness curves, and all gate numbers are
  written to the shared lineage store (Stage 1) or, if Stage 1 is not
  yet built, to a run report in `~/workspace/projects/agp/` — never
  only to a terminal.

### Amendments
(none yet — DRAFT, awaiting Austin's approval)

---

---

## Amendments (appended per §6's rules; all before the verdict evaluation is touched)

### Amendment 1 — 2026-11-05 (registration, pre-verdict)
Concurrency clause relaxed. §6's budget section reads "never concurrent with PEDS test
suites or brain steps." It is amended to: **"not concurrent with PEDS test suites; the
live brain's steps may interleave (they are brief and mostly idle)."** Reason: the brain
now runs continuously (ops\gigi_runner.py, ~2-minute cycles) and this campaign is hers —
forbidding interleave with brain steps would forbid the campaign from ever running.
Approved as part of Austin's 2026-11-05 03:50 EST campaign approval.

---

## FAIL path executed — 2026-11-05

The primary arm's verdict was **FAIL** (run report:
`~/workspace/projects/agp/runs/agp1-primary-20261105T090355Z/report.md`): on all three
seeds a generation-0 random founder already scored a perfect 20/20, 20/20, 40/40, 20/20,
so no lineage ever accumulated a mutation/crossover event and gate item 5 failed on
every seed. hidden_final was never evaluated by the primary arm.

**Austin's decision, 2026-11-05 04:55 EST:** "Follow the prereg: promote the portfolio
champion and redirect breeding to the arena line." The §6 Stop prescription was executed
the same night, as follows.

### Secondary (portfolio) arm — run 2026-11-05, labeled NOT GATED

Run through the campaign runner's `--arm portfolio` mode (the engine's stock
`run_file_task_evolution` code path, engine SHA-256 verified unchanged at the frozen
`397a5fda…`), same frozen splits, same seeds 7/23/53. Run report:
`campaigns/agp1/runs/agp1-portfolio-20261105T095925Z/` (mirrored to
`~/workspace/projects/agp/runs/agp1-portfolio-20261105T095925Z/`). Whole arm completed
in ~3 seconds — the typed-skeleton strategy solves this family immediately, which is
itself the measurement the secondary arm existed to make: there is no
breeding-vs-portfolio gap to measure on this task, because random founders and the
skeleton both trivialize it; the typed grammar is the productive mechanism.

Per seed (identical outcome on all three):

| seed | train | validation | audit | stress | hidden_final | winning strategy | promoted by engine |
|---|---|---|---|---|---|---|---|
| 7 | 20/20 | 20/20 | 40/40 | 20/20 | 20/20 | typed_skeleton | yes |
| 23 | 20/20 | 20/20 | 40/40 | 20/20 | 20/20 | typed_skeleton | yes |
| 53 | 20/20 | 20/20 | 40/40 | 20/20 | 20/20 | typed_skeleton | yes |

Distilled champion program (byte-identical across seeds, 4 ops):
`READ_TEXT(pytest_output.txt) → EXTRACT_TRACEBACK_BLOCK → CLASSIFY_PYTHON_ERROR →
WRITE_JSON(failure_report.json)`. `used_evolutionary_search: false`;
`destructive_write_attempts: 0`.

**hidden_final accounting (precise):** the stock engine computes hidden_final exactly
once per evolve, for that run's distilled program, at the end of the search
(breeding_tool_agents.py line 4971), with `hidden_final_used_in_search: false`,
`hidden_final_used_in_pruning: false`, `hidden_final_used_in_optimizer: false`. The
portfolio arm therefore carries three such engine-internal computations — one per seed
run, each on the (identical) distilled champion program — exactly as the stock CLI has
always behaved for the proven families. The primary arm's reserved once-only
re-audit evaluation was never consumed by the primary arm, and **no additional
evaluation was performed at promotion**: the champion was promoted by registering the
engine's own already-evaluated, already-saved package (below), not by re-running
anything.

### Promotion — the champion is in BTA's pool

The champion (seed 7, first in the frozen order) was registered into BTA's promoted
file-agent pool through the engine's own record format: the package BTA's
`save_file_agent_package` produced during the stock run was placed under the next pool
id by `campaigns/agp1/promote_portfolio_champion.py` (numbering mirrors the engine's
`next_agent_dir`; only `genome.json`'s `agent_id` was set to the pool id; all other
records byte-identical; the script performs no search and no evaluation, and refuses
to run if the pool already holds the task).

- **Pool record: `C:\breeding-tool-agent\file_agents\agent_000139`** (pool 134 → 135),
  scorecard `promoted: true`, scores as tabled above, hidden_final 20/20.
- Registration record: `pool_registration.json` in the portfolio run dir.
- With this promotion, **every Repo Doctor diagnosis family in `BTA_RUNNER_ROADMAP`
  now holds a promoted agent** (pytest_failure_summary agent_000130,
  syntax_error_summary agent_000135, module_not_found_summary agent_000136,
  import_error_summary agent_000137, assertion_failure_summary agent_000138,
  file_not_found_summary agent_000139). The roadmap's recorded next action for this
  step is: "Extend BTA_RUNNER_ROADMAP with no_traceback_failure_summary, then run the
  next diagnosis template" — the `repo_doctor/all_diagnosis` continuation §6 pointed
  at is unblocked.

### Standing conclusion and redirect (now in force)

- **Bred-only search is STOPPED for diagnosis families.** The conclusion stands as
  §6 wrote it: typed synthesis (the strict grammar + skeleton portfolio) is the
  productive mechanism for this domain.
- **The breeding budget is redirected to the arena genome line**
  (`C:\breeding-tool-agent-full-breeder`), where bred survivors already exist.
  Baseline recorded at redirect time (verified from its BREEDING_LEDGER.md tail,
  2026-07-19): 105 arena runs, 9 promoted arena survivors, last run NO_PROMOTION,
  `stage_6_rate` 0.000, transfer 4/12. Scoping for the first arena campaign:
  `~/workspace/projects/agp/AGP2-ARENA-SCOPING.md` (scoping only — nothing was
  launched in the arena, and no full-breeder code was touched).
- This execution was journaled into the live PEDS brain via its supported library
  API (`agp_failpath_executed` journal entry, 2026-11-05), alongside the runner's
  own `agp_campaign_started` / `agp_campaign_verdict` entries for the portfolio arm.
