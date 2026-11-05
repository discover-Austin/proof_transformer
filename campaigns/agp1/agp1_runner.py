#!/usr/bin/env python3
"""AGP-1 campaign runner — bred-only genomes on `file_not_found_summary`.

Implements the frozen prereg in this package (PREREG-AGP1.md; canonical copy
at ~/workspace/projects/agp/PREREG-AGP1.md on Eve's VM). Approved by Austin
2026-11-05 03:50 EST.

WHY THIS DRIVES THE ENGINE'S LIBRARY, NOT THE BTA CLI
The stock CLI evolve (`--file-task-evolve`) always searches through
`cegis_file_search` -> `file_search_portfolio_candidates`, which
unconditionally tries seed program / LLM proposals / subagent backend /
macro library / skeletons / pipelines / grammar searches before evolution,
with no disable switch. A CLI run therefore cannot be breeding-only.
The primary arm instead calls the engine's own `evolutionary_file_search`
semantics directly: identical population init (`rand_file_program`),
identical fitness (`file_fitness`), identical tournament (size 4) +
elitism (8), identical operators (`crossover_file_program`,
`mutate_file_program`, mutation probability 0.7), identical pop/gens/seeds.
The only additions are bookkeeping (per-candidate variation-event lineage,
fitness curve, wall-clock cap) that the gate requires and the engine's EA
does not record. No BTA engine code is modified; this runner sets the same
module globals the CLI flags set (BLIND_FILE_FITNESS, STRICT_FILE_GRAMMAR,
AUDIT_HIDDEN, HIDDEN_FINAL) and calls engine functions as implemented.

HIDDEN_FINAL DISCIPLINE
`evolve_file_task` evaluates hidden_final unconditionally at the end of
every run, so this runner never calls it on the primary arm. Search,
pruning (train+validation only — the AUDIT_HIDDEN semantics), feature /
ablation checks and visible-split scoring never touch hidden_final.
hidden_final is evaluated exactly once per declared champion, in
`hidden_reaudit`, after the champion is declared on the visible splits,
and a marker file makes a second evaluation impossible.

Arms:
  --arm primary   (default) bred-only, gated, seeds from config.
  --arm portfolio secondary arm: engine's run_file_task_evolution (the
                  exact CLI code path) with seed programs allowed.
                  Labeled PORTFOLIO / NOT GATED. Its scorecards use the
                  engine's standard semantics (hidden_final evaluated by
                  the engine at end of evolve, never used in search).
  --smoke         throwaway wiring check: seed 999, pop 16 x 3 gens,
                  visible splits only, hidden_final never referenced,
                  no champion can be declared. NOT part of the gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO_ROOT = CAMPAIGN_DIR.parents[1]
PROOF_ROOT = REPO_ROOT / "proof_transformer"
BRAIN_ROOT = REPO_ROOT / "peds_brain_live"
RUNS_DIR = CAMPAIGN_DIR / "runs"
CONFIG = json.loads((CAMPAIGN_DIR / "config.json").read_text(encoding="utf-8"))
TASK = CONFIG["task"]
BTA_DIR = Path(CONFIG["bta_dir"])
ENGINE_PATH = BTA_DIR / CONFIG["engine_file"]
VISIBLE_SPLITS = ("train", "validation", "audit", "stress")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_id_now() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def load_engine():
    """Import the BTA engine and set the prereg's global toggles (the same
    globals the CLI flags set in main()). Verifies the frozen engine hash."""
    actual = sha256_file(ENGINE_PATH)
    if actual != CONFIG["engine_sha256_frozen"]:
        raise RuntimeError(
            f"engine hash drift: {actual} != frozen {CONFIG['engine_sha256_frozen']}; "
            "the prereg froze the engine — aborting rather than running a moved experiment"
        )
    sys.path.insert(0, str(BTA_DIR))
    import breeding_tool_agents as bta

    bta.BLIND_FILE_FITNESS = True
    bta.STRICT_FILE_GRAMMAR = True
    bta.AUDIT_HIDDEN = True
    bta.HIDDEN_FINAL = True
    if TASK not in bta.FILE_TASKS:
        raise RuntimeError(f"task {TASK} not registered in FILE_TASKS (manifest missing?)")
    contract_errors = bta.file_tool_contract_errors_for_task(TASK)
    if contract_errors:
        raise RuntimeError("file tool contract gate failed: " + " | ".join(contract_errors[:5]))
    if bta.file_llm_proposal_programs(TASK):
        raise RuntimeError("LLM proposals exist for this task — primary arm would be contaminated")
    return bta


def brain_journal(event_type: str, summary: str, evidence: dict) -> None:
    """Record campaign progress in the live brain's journal through PEDS'
    own library API (append-only; never edits brain files directly).
    Best-effort: a brain-write failure must never kill a campaign run."""
    try:
        sys.path.insert(0, str(PROOF_ROOT / "src"))
        from proof_transformer.peds import brain as peds_brain

        peds_brain.append_brain_journal(event_type, summary, evidence=evidence, root=BRAIN_ROOT)
    except Exception as exc:  # noqa: BLE001 - journaling is auxiliary to the run
        print(f"[agp1] brain journal write failed ({event_type}): {exc}", flush=True)


# --------------------------------------------------------------------------
# Primary arm: evolutionary search with lineage bookkeeping.
# Mirrors breeding_tool_agents.evolutionary_file_search step for step,
# using the engine's own init / fitness / selection / operator functions.
# --------------------------------------------------------------------------
def run_ea(bta, cases, seed, pop_size, generations, deadline, log):
    rng = random.Random(seed)
    pop = [bta.rand_file_program(TASK, rng) for _ in range(pop_size)]
    lineage = [{"crossover_events": 0, "mutation_events": 0, "origin": "random_founder"} for _ in range(pop_size)]
    best = pop[0]
    best_score = bta.file_fitness(best, TASK, cases)
    best_lineage = dict(lineage[0])
    fitness_evals = 1
    curve = []
    cap_hit = False
    gens_run = 0
    for gen in range(generations):
        if time.time() > deadline:
            cap_hit = True
            break
        scores = [bta.file_fitness(program, TASK, cases) for program in pop]
        fitness_evals += len(pop)
        gens_run = gen + 1
        order = sorted(range(len(pop)), key=lambda idx: scores[idx], reverse=True)
        if scores[order[0]] > best_score:
            best, best_score = pop[order[0]], scores[order[0]]
            best_lineage = dict(lineage[order[0]])
        passed = bta.file_passes(best, TASK, cases)
        curve.append(
            {
                "generation": gen,
                "best_fitness": scores[order[0]],
                "mean_fitness": sum(scores) / len(scores),
                "best_train_passed": passed,
            }
        )
        log(f"file gen {gen:02d} | train {passed}/{len(cases)} | best fitness {scores[order[0]]:.4f} | tools {len(best)}")
        if passed == len(cases):
            break
        elite_idx = order[:8]
        nxt = [pop[idx] for idx in elite_idx]
        nxt_lineage = [lineage[idx] for idx in elite_idx]

        def tour():
            choices = [rng.randrange(len(pop)) for _ in range(4)]
            return max(choices, key=lambda idx: scores[idx])

        while len(nxt) < pop_size:
            p1, p2 = tour(), tour()
            child = bta.crossover_file_program(pop[p1], pop[p2], TASK, rng)
            events = {
                "crossover_events": lineage[p1]["crossover_events"] + lineage[p2]["crossover_events"] + 1,
                "mutation_events": lineage[p1]["mutation_events"] + lineage[p2]["mutation_events"],
                "origin": "bred",
            }
            if rng.random() < 0.7:
                child = bta.mutate_file_program(child, TASK, rng)
                events["mutation_events"] += 1
            nxt.append(child)
            nxt_lineage.append(events)
        pop, lineage = nxt, nxt_lineage
    return {
        "best": best,
        "best_score": best_score,
        "best_lineage": best_lineage,
        "fitness_curve": curve,
        "fitness_evaluations": fitness_evals,
        "generations_run": gens_run,
        "cap_hit": cap_hit,
    }


def score_visible_splits(bta, prog):
    spec = bta.FILE_TASKS[TASK]
    return {split: (bta.file_passes(prog, TASK, spec[split]), len(spec[split])) for split in VISIBLE_SPLITS}


def evaluate_candidate(bta, prog):
    """Post-search pipeline mirroring evolve_file_task's post-search steps,
    except hidden_final is never touched here."""
    spec = bta.FILE_TASKS[TASK]
    prune_cases = spec["train"] + spec.get("validation", [])
    distilled = bta.prune_file_program(prog, TASK, prune_cases)
    scores = score_visible_splits(bta, distilled)
    features = bta.file_feature_requirements_with_ablation(distilled, TASK, spec["audit"])
    cost_cases = spec["train"] + spec["validation"] + spec["audit"] + spec["stress"]
    costs = bta.file_agent_tool_costs(distilled, TASK, cases=cost_cases, search_meta={})
    return distilled, scores, features, costs


def write_champion_package(bta, champ_dir: Path, distilled, scores, features, costs, lineage, seed):
    champ_dir.mkdir(parents=True, exist_ok=True)
    spec = bta.FILE_TASKS[TASK]
    (champ_dir / "program.txt").write_text("\n".join(bta.render_file_program(distilled, TASK)) + "\n", encoding="utf-8")
    genome = {
        "campaign": CONFIG["campaign_id"],
        "task": TASK,
        "kind": "file_agent_champion",
        "seed": seed,
        "ops": spec["allowed_ops"],
        "constants": spec["constants"],
        "genome": [{"op": op, "dst": dst, "a": a, "b": b} for op, dst, a, b in distilled],
    }
    (champ_dir / "genome.json").write_text(json.dumps(genome, indent=2, sort_keys=True), encoding="utf-8")
    scorecard = {
        "task": TASK,
        "seed": seed,
        "train_passed": scores["train"][0],
        "train_total": scores["train"][1],
        "validation_passed": scores["validation"][0],
        "validation_total": scores["validation"][1],
        "audit_passed": scores["audit"][0],
        "audit_total": scores["audit"][1],
        "stress_passed": scores["stress"][0],
        "stress_total": scores["stress"][1],
        "feature_requirements": features,
        "tool_costs": costs,
        "destructive_write_attempts": costs.get("destructive_write_attempts"),
        "used_seed_programs": False,
        "used_evolutionary_search": True,
        "used_skeleton_search": False,
        "used_macro_search": False,
        "used_subagent_search": False,
        "used_llm_proposals": False,
        "winning_search_strategy": "evolutionary_bred_only",
        "hidden_final_evaluated": False,
    }
    (champ_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")
    lineage_doc = {
        "kind": "file_agent",
        "task": TASK,
        "seed": seed,
        "verifier": "exact output-file match over train/validation/audit/stress; hidden_final once after declaration",
        "best_lineage": lineage,
        "fitness_curve_file": "../fitness_curve.json",
    }
    (champ_dir / "lineage.json").write_text(json.dumps(lineage_doc, indent=2, sort_keys=True), encoding="utf-8")
    return scorecard


def hidden_reaudit(bta, champ_dir: Path, distilled) -> int:
    """The single permitted hidden_final evaluation for a declared champion.
    A marker file makes a second evaluation impossible."""
    marker = champ_dir / "hidden_final_evaluated.json"
    if marker.exists():
        raise RuntimeError("hidden_final already evaluated for this champion — refusing a second evaluation")
    spec = bta.FILE_TASKS[TASK]
    passed = bta.file_passes(distilled, TASK, spec["hidden_final"])
    marker.write_text(
        json.dumps(
            {"passed": passed, "total": len(spec["hidden_final"]), "evaluated_at": utc_now(), "evaluations": 1},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return passed


def gate_check(scores, features, costs, lineage):
    gate = CONFIG["gate"]
    reasons = []
    for split in VISIBLE_SPLITS:
        if scores[split][0] != gate[split] or scores[split][1] != gate[split]:
            reasons.append(f"{split} {scores[split][0]}/{scores[split][1]} != {gate[split]}/{gate[split]}")
    if not features.get("passes"):
        reasons.append("feature/ablation requirements failed")
    if costs.get("destructive_write_attempts") != 0:
        reasons.append(f"destructive_write_attempts={costs.get('destructive_write_attempts')} != 0")
    events = lineage.get("crossover_events", 0) + lineage.get("mutation_events", 0)
    if events < 1:
        reasons.append("lineage shows no mutation/crossover event (not bred)")
    return reasons


def run_primary_seed(bta, seed, run_dir: Path):
    cfg = CONFIG["primary"]
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg):
        line = f"[{utc_now()}] {msg}"
        log_lines.append(line)
        print(line, flush=True)

    spec = bta.FILE_TASKS[TASK]
    started = time.time()
    deadline = started + cfg["cap_seconds_per_seed"]
    log(f"AGP-1 primary seed {seed}: pop {cfg['pop_size']} x {cfg['generations']} gens, bred-only")
    ea = run_ea(bta, spec["train"], seed, cfg["pop_size"], cfg["generations"], deadline, log)
    distilled, scores, features, costs = evaluate_candidate(bta, ea["best"])
    elapsed = time.time() - started
    result = {
        "arm": "primary_bred_only",
        "seed": seed,
        "cap_hit": ea["cap_hit"],
        "generations_run": ea["generations_run"],
        "fitness_evaluations": ea["fitness_evaluations"],
        "evaluations_per_second": ea["fitness_evaluations"] / max(elapsed, 0.001),
        "wall_seconds": elapsed,
        "best_fitness": ea["best_score"],
        "best_lineage": ea["best_lineage"],
        "scores": {k: {"passed": v[0], "total": v[1]} for k, v in scores.items()},
        "feature_requirements_pass": bool(features.get("passes")),
        "destructive_write_attempts": costs.get("destructive_write_attempts"),
        "champion_declared": False,
        "hidden_final_passed": None,
    }
    reasons = gate_check(scores, features, costs, ea["best_lineage"])
    result["gate_reasons"] = reasons
    if not reasons:
        result["champion_declared"] = True
        champ_dir = seed_dir / "champion"
        write_champion_package(bta, champ_dir, distilled, scores, features, costs, ea["best_lineage"], seed)
        result["hidden_final_passed"] = hidden_reaudit(bta, champ_dir, distilled)
        result["hidden_final_total"] = 20
        log(f"champion declared; hidden_final re-audit: {result['hidden_final_passed']}/20")
    else:
        log(f"no champion for seed {seed}: {'; '.join(reasons)}")
    (seed_dir / "fitness_curve.json").write_text(json.dumps(ea["fitness_curve"], indent=2), encoding="utf-8")
    (seed_dir / "ea_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (seed_dir / "seed_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    brain_journal(
        "agp_campaign_seed_completed",
        f"AGP-1 primary seed {seed} finished: champion={result['champion_declared']}, "
        f"scores train {scores['train'][0]}/{scores['train'][1]} val {scores['validation'][0]}/{scores['validation'][1]} "
        f"audit {scores['audit'][0]}/{scores['audit'][1]} stress {scores['stress'][0]}/{scores['stress'][1]}",
        {"campaign_id": CONFIG["campaign_id"], "seed": seed, "seed_result": str(seed_dir / "seed_result.json")},
    )
    return result


def run_portfolio_seed(bta, seed, run_dir: Path):
    """Secondary arm — the engine's own CLI code path (run_file_task_evolution),
    seed programs allowed, portfolio intact. Labeled, not gated."""
    seed_dir = run_dir / "portfolio" / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(msg):
        lines.append(str(msg))
        print(str(msg), flush=True)

    summaries = bta.run_file_task_evolution(
        TASK, seed, str(seed_dir / "agents"), log=log, use_seed_programs=True, optimize_file_cost=False
    )
    slim_keys = (
        "task", "train_passed", "train_total", "validation_passed", "validation_total",
        "audit_passed", "audit_total", "stress_passed", "stress_total",
        "hidden_final_passed", "hidden_final_total", "hidden_final_used_in_search",
        "winning_search_strategy", "used_evolutionary_search", "used_skeleton_search",
        "used_macro_search", "used_subagent_search", "used_llm_proposals", "saved_path", "promoted",
    )
    slim = [{k: s.get(k) for k in slim_keys} for s in summaries]
    (seed_dir / "portfolio_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (seed_dir / "seed_result.json").write_text(
        json.dumps({"arm": "portfolio_not_gated", "seed": seed, "summaries": slim}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return slim


def write_reports(run_dir: Path, manifest: dict, seed_results: list, verdict: str | None):
    report = {"manifest": manifest, "seeds": seed_results, "verdict": verdict, "finished_at": utc_now()}
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md = [
        f"# AGP-1 run report — {run_dir.name}",
        "",
        f"- Campaign: {CONFIG['campaign_id']} ({manifest.get('arm')})",
        f"- Task: {TASK}",
        f"- Engine SHA-256: {manifest.get('engine_sha256')}",
        f"- Started: {manifest.get('started_at')}  Finished: {report['finished_at']}",
        f"- Verdict: **{verdict}**",
        "",
        "| seed | gens | fitness evals | evals/s | wall (s) | train | val | audit | stress | champion | hidden_final |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in seed_results:
        s = r.get("scores", {})
        def cell(name):
            return f"{s[name]['passed']}/{s[name]['total']}" if name in s else "-"
        md.append(
            f"| {r.get('seed')} | {r.get('generations_run')} | {r.get('fitness_evaluations')} | "
            f"{(r.get('evaluations_per_second') or 0):.2f} | {(r.get('wall_seconds') or 0):.0f} | "
            f"{cell('train')} | {cell('validation')} | {cell('audit')} | {cell('stress')} | "
            f"{r.get('champion_declared')} | {r.get('hidden_final_passed')} |"
        )
    (run_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="AGP-1 campaign runner (see PREREG-AGP1.md)")
    parser.add_argument("--arm", choices=("primary", "portfolio"), default="primary")
    parser.add_argument("--smoke", action="store_true", help="throwaway wiring check; not part of the gate")
    parser.add_argument("--seeds", type=int, nargs="*", help="override seed list (default: config)")
    args = parser.parse_args()

    bta = load_engine()
    manifest_path = BTA_DIR / "file_tasks" / TASK / "manifest.json"

    if args.smoke:
        cfg = CONFIG["smoke"]
        run_dir = RUNS_DIR / f"smoke-{run_id_now()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        spec = bta.FILE_TASKS[TASK]
        lines = []

        def log(msg):
            line = f"[{utc_now()}] {msg}"
            lines.append(line)
            print(line, flush=True)

        started = time.time()
        ea = run_ea(bta, spec["train"], cfg["seed"], cfg["pop_size"], cfg["generations"], started + cfg["cap_seconds"], log)
        scores = score_visible_splits(bta, ea["best"])
        result = {
            "mode": "smoke",
            "arm": "primary_bred_only",
            "seed": cfg["seed"],
            "pop_size": cfg["pop_size"],
            "generations": cfg["generations"],
            "generations_run": ea["generations_run"],
            "fitness_evaluations": ea["fitness_evaluations"],
            "wall_seconds": time.time() - started,
            "best_fitness": ea["best_score"],
            "best_lineage": ea["best_lineage"],
            "scores_visible_splits": {k: {"passed": v[0], "total": v[1]} for k, v in scores.items()},
            "hidden_final_evaluated": False,
            "hidden_final_referenced": False,
        }
        (run_dir / "smoke_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        (run_dir / "ea_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (run_dir / "fitness_curve.json").write_text(json.dumps(ea["fitness_curve"], indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    run_dir = RUNS_DIR / f"agp1-{args.arm}-{run_id_now()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "campaign_id": CONFIG["campaign_id"],
        "arm": args.arm,
        "task": TASK,
        "engine_sha256": sha256_file(ENGINE_PATH),
        "task_manifest_sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        "config": CONFIG,
        "started_at": utc_now(),
        "run_dir": str(run_dir),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    brain_journal(
        "agp_campaign_started",
        f"AGP-1 {args.arm} arm launched: bred-only genomes on {TASK} (prereg PREREG-AGP1).",
        {"campaign_id": CONFIG["campaign_id"], "run_dir": str(run_dir), "arm": args.arm},
    )

    seed_results = []
    if args.arm == "primary":
        seeds = args.seeds if args.seeds else CONFIG["primary"]["seeds"]
        for seed in seeds:
            seed_results.append(run_primary_seed(bta, seed, run_dir))
            size = dir_size_bytes(run_dir)
            if size > CONFIG["artifact_cap_bytes"]:
                raise RuntimeError(f"artifact cap exceeded: {size} > {CONFIG['artifact_cap_bytes']}")
        gate = CONFIG["gate"]
        champions = [
            r for r in seed_results
            if r["champion_declared"] and (r["hidden_final_passed"] or 0) >= gate["hidden_final_min"]
        ]
        cap_hits = [r for r in seed_results if r["cap_hit"]]
        if len(champions) >= gate["champions_needed"]:
            verdict = "PASS"
        elif len(cap_hits) >= 2:
            verdict = "INCONCLUSIVE"
        else:
            verdict = "FAIL"
    else:
        seeds = args.seeds if args.seeds else CONFIG["primary"]["seeds"]
        for seed in seeds:
            seed_results.append({"seed": seed, "summaries": run_portfolio_seed(bta, seed, run_dir)})
        verdict = "PORTFOLIO_COMPLETE_NOT_GATED"

    write_reports(run_dir, manifest, seed_results, verdict)
    brain_journal(
        "agp_campaign_verdict",
        f"AGP-1 {args.arm} arm verdict: {verdict}. Report: {run_dir / 'report.md'}",
        {"campaign_id": CONFIG["campaign_id"], "verdict": verdict, "report": str(run_dir / "report.json")},
    )
    print(f"AGP-1 {args.arm} verdict: {verdict} — report at {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
