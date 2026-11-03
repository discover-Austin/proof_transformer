from __future__ import annotations

import argparse
import copy
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
EXAMPLES = PROJECT_ROOT / "examples"
for path in (SRC, EXAMPLES):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peds_trained_local_generations import run_generations  # noqa: E402
from proof_transformer.peds.mutation_strategies import (  # noqa: E402
    propose_candidate_mutation,
    validate_single_mutation,
)

DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "temp" / "peds_training_campaigns"
DEFAULT_GENERATIONS = 25
DEFAULT_RUNS = 3
DEFAULT_MAX_ATTEMPTS_PER_GENERATION = 3
CONTINUATION_POLICIES = {
    "continue",
    "stop",
    "last-valid-parent",
    "fixed-baseline",
    "fresh-baseline",
}
SUPPORTED_VARIANTS = {"acceptance_checklist_default"}
KNOWN_UNSUPPORTED_VARIANTS = {
    "acceptance_checklist_harder_replay",
    "acceptance_checklist_noisy_shadow",
}
GenerationRunner = Callable[..., dict[str, Any]]
DIAGNOSIS_CATEGORIES = {
    "replay_underperformance",
    "canary_rejection",
    "shadow_rejection",
    "training_error",
    "adapter_load_error",
    "insufficient_delta",
    "no_generated_child",
    "infrastructure_error",
    "unknown",
}
CORRECTION_ACTIONS = {
    "increase_replay_diversity",
    "increase_canary_diversity",
    "isolate_generation_evidence",
    "lower_noise_shadow_sampling",
    "retry_with_fresh_candidate",
    "retry_with_fixed_baseline",
    "retry_with_last_valid_parent",
    "skip_bad_evidence_batch",
    "reduce_trainer_noise",
    "no_safe_correction_available",
}
IMMUTABLE_GATE_OPTION_KEYS = {
    "auto_promote",
    "canary_gate",
    "canary_threshold",
    "gate_decision",
    "lifecycle_state",
    "min_delta",
    "pass_threshold",
    "promotion_decision",
    "replay_gate",
    "replay_threshold",
    "replay_thresholds",
    "shadow_gate",
    "shadow_thresholds",
    "canary_thresholds",
}


def _load_active_policy(raw_json: str | None, file_path: Path | None) -> dict[str, Any] | None:
    if raw_json:
        return _normalize_active_policy(json.loads(raw_json), source="cli_json")
    if file_path is None:
        return None
    path = Path(file_path)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    if path.suffix == ".jsonl":
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        payload = json.loads(lines[-1])
    else:
        payload = json.loads(text)
    return _normalize_active_policy(payload, source=str(path))


def _normalize_active_policy(policy: dict[str, Any] | None, *, source: str | None = None) -> dict[str, Any] | None:
    if not isinstance(policy, dict):
        return None
    candidate_id = str(policy.get("candidate_id") or "").strip()
    if not candidate_id:
        return None
    keys = (
        "active_policy_id",
        "candidate_id",
        "objective",
        "profile_path",
        "registry_path",
        "selection_reason",
        "timestamp",
    )
    normalized = {key: copy.deepcopy(policy[key]) for key in keys if policy.get(key) not in (None, "")}
    safety = policy.get("safety")
    if isinstance(safety, dict):
        normalized["safety"] = copy.deepcopy(safety)
    if source is not None:
        normalized["source"] = source
    elif policy.get("source"):
        normalized["source"] = str(policy["source"])
    return normalized


def _public_active_policy(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = _normalize_active_policy(policy)
    return copy.deepcopy(normalized) if normalized else None


def _active_policy_start_summary(policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_active_policy(policy)
    return {
        "supported": True,
        "provided": bool(normalized),
        "used_as_mutation_context": bool(normalized),
        "used_as_db_parent": False,
        "candidate_id": normalized.get("candidate_id") if normalized else None,
        "active_policy_id": normalized.get("active_policy_id") if normalized else None,
        "source": normalized.get("source") if normalized else None,
        "reason": (
            "Active policy seeds mutation context only; it is not treated as a campaign DB parent "
            "unless the run owns that candidate."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    active_policy = _load_active_policy(args.active_policy_json, args.active_policy_file)
    report = run_campaign(
        generations=args.generations,
        runs=args.runs,
        run_root=args.run_root,
        variants=_parse_variants(args.variant),
        baseline_mode=args.baseline_mode,
        min_delta=args.min_delta,
        max_attempts_per_generation=args.max_attempts_per_generation,
        continuation_policy=args.continuation_policy,
        plateau_window=args.plateau_window,
        plateau_epsilon=args.plateau_epsilon,
        stop_on_failure=args.stop_on_failure,
        active_policy=active_policy,
    )
    indent = None if args.json else 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return _campaign_exit_code(report, strict=args.strict)


def run_campaign(
    *,
    generations: int = DEFAULT_GENERATIONS,
    runs: int = DEFAULT_RUNS,
    run_root: Path | str | None = None,
    variants: list[str] | None = None,
    baseline_mode: str = "parent",
    min_delta: float = 0.02,
    max_attempts_per_generation: int = DEFAULT_MAX_ATTEMPTS_PER_GENERATION,
    continuation_policy: str = "last-valid-parent",
    plateau_window: int = 5,
    plateau_epsilon: float = 0.005,
    stop_on_failure: bool = False,
    generation_runner: GenerationRunner = run_generations,
    active_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if generations <= 0:
        raise ValueError("generations must be positive")
    if runs <= 0:
        raise ValueError("runs must be positive")
    if max_attempts_per_generation <= 0:
        raise ValueError("max_attempts_per_generation must be positive")
    if continuation_policy not in CONTINUATION_POLICIES:
        raise ValueError(f"continuation_policy must be one of {sorted(CONTINUATION_POLICIES)!r}")
    campaign_id = _campaign_id()
    campaign_root = Path(run_root) if run_root is not None else DEFAULT_CAMPAIGNS_ROOT / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)
    requested_variants = variants or ["acceptance_checklist_default"]
    executable_variants, blockers = _select_executable_variants(requested_variants, baseline_mode)
    created_at = _utc_now()
    run_summaries: list[dict[str, Any]] = []
    safety = {"network_attempts": [], "model_modules_loaded": [], "fake_weights_written": False}
    active_policy_context = _normalize_active_policy(active_policy)

    for variant in executable_variants:
        for run_index in range(1, runs + 1):
            run_dir = campaign_root / variant / f"run_{run_index:02d}"
            run_id = f"{campaign_id}_{variant}_run{run_index:02d}"
            summary = _run_one_campaign_slot(
                generation_runner=generation_runner,
                generations=generations,
                run_root=run_dir,
                run_id=run_id,
                run_index=run_index,
                variant=variant,
                baseline_mode=baseline_mode,
                min_delta=min_delta,
                max_attempts_per_generation=max_attempts_per_generation,
                continuation_policy=continuation_policy,
                active_policy=active_policy_context,
            )
            run_summaries.append(summary)
            _merge_safety(safety, summary.get("source_report") or {})
            _merge_attempt_safety(safety, summary)
            if summary.get("blocker"):
                blockers.append(
                    {
                        "type": "run_failure",
                        "variant": variant,
                        "run_index": run_index,
                        "blocker": summary["blocker"],
                    }
                )
            if stop_on_failure and not summary["passed"]:
                break
        if stop_on_failure and run_summaries and not run_summaries[-1]["passed"]:
            break

    aggregate = analyze_campaign(
        run_summaries,
        generations_requested=generations,
        min_delta=min_delta,
        plateau_window=plateau_window,
        plateau_epsilon=plateau_epsilon,
    )
    report = {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root),
        "created_at": created_at,
        "runs_requested": runs,
        "generations_requested": generations,
        "variants_requested": requested_variants,
        "variants_executed": executable_variants,
        "baseline_mode": baseline_mode,
        "min_delta": min_delta,
        "max_attempts_per_generation": max_attempts_per_generation,
        "continuation_policy": continuation_policy,
        "plateau_window": plateau_window,
        "plateau_epsilon": plateau_epsilon,
        "active_policy": _public_active_policy(active_policy_context),
        "active_policy_start": _active_policy_start_summary(active_policy_context),
        "runs": [_public_run_summary(item) for item in run_summaries],
        "attempts": _public_attempts(run_summaries),
        "rejected_attempts": _public_rejected_attempts(run_summaries),
        "diagnoses": _public_diagnoses(run_summaries),
        "corrections": _public_corrections(run_summaries),
        "mutations": _public_mutations(run_summaries),
        "winning_attempts": _public_winning_attempts(run_summaries),
        "exhausted_generations": _public_exhausted_generations(run_summaries),
        "lineage": _public_lineage(run_summaries),
        "aggregate": aggregate,
        "safety": safety,
        "blockers": blockers,
    }
    report_path = campaign_root / "campaign_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def analyze_campaign(
    runs: list[dict[str, Any]],
    *,
    generations_requested: int,
    min_delta: float = 0.02,
    plateau_window: int = 5,
    plateau_epsilon: float = 0.005,
) -> dict[str, Any]:
    adaptive = any("attempts" in run for run in runs)
    attempts = [attempt for run in runs for attempt in run.get("attempts", [])]
    rejected_attempts = [attempt for attempt in attempts if attempt.get("outcome") == "rejected"]
    process_failure_attempts = [attempt for attempt in attempts if attempt.get("process_failure")]
    exhausted_generations = [
        generation
        for run in runs
        for generation in run.get("generation_summaries", [])
        if generation.get("exhausted")
    ]
    if adaptive:
        metrics = [
            generation["winning_metric"]
            for run in runs
            for generation in run.get("generation_summaries", [])
            if generation.get("winning_metric")
        ]
        attempt_metrics = [attempt["metric"] for attempt in attempts if attempt.get("metric")]
    else:
        metrics = [metric for run in runs for metric in run.get("generation_metrics", [])]
        attempt_metrics = metrics
    completed_metrics = [metric for metric in metrics if metric.get("generation") is not None]
    first_failure = _first_failure(runs, completed_metrics)
    delta_metrics = [metric for metric in attempt_metrics if _is_number(metric.get("replay_mean_delta"))]
    best_generation = max(delta_metrics, key=lambda item: item["replay_mean_delta"], default=None)
    worst_generation = min(delta_metrics, key=lambda item: item["replay_mean_delta"], default=None)
    mean_by_generation = _delta_stat_by_generation(completed_metrics, generations_requested, "mean")
    min_by_generation = _delta_stat_by_generation(completed_metrics, generations_requested, "min")
    max_by_generation = _delta_stat_by_generation(completed_metrics, generations_requested, "max")
    if adaptive:
        collapse_detected = bool(exhausted_generations or process_failure_attempts)
    else:
        collapse_detected = any(_metric_collapsed(metric) for metric in completed_metrics) or any(
            not run.get("passed") for run in runs
        )
    correction_stats = _correction_stats(runs)
    return {
        "total_generations_attempted": generations_requested * len(runs),
        "total_generations_completed": len(completed_metrics),
        "total_candidate_attempts": len(attempts) if adaptive else len(metrics),
        "total_rejected_attempts": len(rejected_attempts) if adaptive else len([m for m in metrics if _metric_collapsed(m)]),
        "total_winning_attempts": len([attempt for attempt in attempts if attempt.get("outcome") == "winner"])
        if adaptive
        else len([m for m in metrics if not _metric_collapsed(m)]),
        "exhausted_generation_count": len(exhausted_generations),
        "exhausted_generations": exhausted_generations,
        "infrastructure_failure_count": len(process_failure_attempts),
        "corrections_attempted": correction_stats["corrections_attempted"],
        "corrections_followed_by_winner": correction_stats["corrections_followed_by_winner"],
        "correction_success_rate": correction_stats["correction_success_rate"],
        "replay_winner_rate": _rate(
            attempt_metrics if adaptive else completed_metrics,
            lambda metric: metric.get("replay_decision") == "replay_winner",
        ),
        "generation_winner_rate": _rate(
            [{"won": True} for _ in completed_metrics]
            + [{"won": False} for _ in range(max(0, generations_requested * len(runs) - len(completed_metrics)))],
            lambda item: item.get("won") is True,
        ),
        "trained_rate": _rate(attempt_metrics if adaptive else completed_metrics, lambda metric: metric.get("training_trained") is True),
        "adapter_loaded_rate": _rate(attempt_metrics if adaptive else completed_metrics, lambda metric: metric.get("adapter_loaded") is True),
        "mean_delta_by_generation": mean_by_generation,
        "min_delta_by_generation": min_by_generation,
        "max_delta_by_generation": max_by_generation,
        "attempt_replay_deltas": [
            {
                "generation": attempt.get("generation"),
                "attempt_index": attempt.get("attempt_index"),
                "mean_delta": (attempt.get("metric") or {}).get("replay_mean_delta"),
                "outcome": attempt.get("outcome"),
            }
            for attempt in attempts
        ],
        "delta_trend": _delta_trend(mean_by_generation, plateau_epsilon),
        "plateau_detected": _plateau_detected(mean_by_generation, plateau_window, plateau_epsilon),
        "collapse_detected": collapse_detected,
        "first_failure": first_failure,
        "best_generation": best_generation,
        "worst_generation": worst_generation,
        "min_delta_threshold": min_delta,
        "below_min_delta_generations": [
            metric
            for metric in completed_metrics
            if _is_number(metric.get("replay_mean_delta")) and metric["replay_mean_delta"] < min_delta
        ],
    }


def build_campaign_command(
    *,
    generations: int = DEFAULT_GENERATIONS,
    runs: int = DEFAULT_RUNS,
    max_attempts_per_generation: int = DEFAULT_MAX_ATTEMPTS_PER_GENERATION,
    repo_root: Path | None = None,
) -> list[str]:
    root = repo_root or PROJECT_ROOT
    return [
        sys.executable,
        str(root / "scripts" / "peds_training_campaign.py"),
        "--generations",
        str(generations),
        "--runs",
        str(runs),
        "--max-attempts-per-generation",
        str(max_attempts_per_generation),
        "--json",
    ]


def _run_one_campaign_slot(
    *,
    generation_runner: GenerationRunner,
    generations: int,
    run_root: Path,
    run_id: str,
    run_index: int,
    variant: str,
    baseline_mode: str = "parent",
    min_delta: float = 0.02,
    max_attempts_per_generation: int = DEFAULT_MAX_ATTEMPTS_PER_GENERATION,
    continuation_policy: str = "last-valid-parent",
    active_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    generation_summaries: list[dict[str, Any]] = []
    generation_metrics: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    lineage: list[dict[str, Any]] = []
    last_winning_candidate_id: str | None = None
    fixed_baseline_candidate_id: str | None = None
    parent_policy = _normalize_continuation_policy(continuation_policy)
    active_policy_context = _normalize_active_policy(active_policy)
    active_policy_candidate_id = active_policy_context.get("candidate_id") if active_policy_context else None

    for generation in range(1, generations + 1):
        selected_parent_candidate_id, parent_selection_reason = _select_parent_for_generation(
            parent_policy=parent_policy,
            generation=generation,
            last_winning_candidate_id=last_winning_candidate_id,
            fixed_baseline_candidate_id=fixed_baseline_candidate_id,
        )
        base_options = {
            "baseline_mode": baseline_mode,
            "candidate_variant": variant,
            "evidence_selection_seed": f"{run_id}:generation:{generation:02d}",
            "generation": generation,
            "parent_candidate_id": selected_parent_candidate_id,
            "parent_policy": parent_policy,
            "parent_selection_reason": parent_selection_reason,
            "trainer_config": {"epochs": 1, "generation": generation},
        }
        if active_policy_context:
            base_options["active_policy"] = _public_active_policy(active_policy_context)
            base_options["active_policy_candidate_id"] = active_policy_candidate_id
            base_options["active_policy_start"] = {
                "used_as_mutation_context": True,
                "used_as_db_parent": False,
            }
        base_options["candidate_mutation"] = propose_candidate_mutation(
            _mutation_parent_payload(
                candidate_id=selected_parent_candidate_id or active_policy_candidate_id,
                attempt=None,
                variant=variant,
                active_policy=active_policy_context,
            ),
            diagnosis={"category": "unknown"},
            attempt_index=1,
            generation_index=generation,
        )
        pending_correction: dict[str, Any] | None = None
        generation_attempts: list[dict[str, Any]] = []
        winner: dict[str, Any] | None = None

        for attempt_index in range(1, max_attempts_per_generation + 1):
            attempt_options = copy.deepcopy(base_options)
            if pending_correction is not None:
                attempt_options = apply_correction_to_generation_options(
                    attempt_options,
                    pending_correction,
                    attempt_index,
                )
            attempt = _run_generation_attempt(
                generation_runner=generation_runner,
                generation=generation,
                attempt_index=attempt_index,
                run_root=run_root,
                run_id=run_id,
                variant=variant,
                min_delta=min_delta,
                parent_candidate_id=selected_parent_candidate_id,
                parent_selection_reason=parent_selection_reason,
                options=attempt_options,
            )
            generation_attempts.append(attempt)
            source_report = attempt.get("source_report") or source_report
            if fixed_baseline_candidate_id is None and attempt.get("source_candidate_id"):
                fixed_baseline_candidate_id = attempt["source_candidate_id"]
            if attempt["outcome"] == "winner":
                winner = attempt
                break

            diagnosis = diagnose_attempt_outcome(
                attempt,
                {
                    "attempt_index": attempt_index,
                    "generation": generation,
                    "max_attempts_per_generation": max_attempts_per_generation,
                    "min_delta": min_delta,
                    "previous_correction": pending_correction,
                    "variant": variant,
                },
            )
            correction = choose_correction(
                diagnosis,
                {
                    "attempt_index": attempt_index,
                    "generation": generation,
                    "max_attempts_per_generation": max_attempts_per_generation,
                    "variant": variant,
                },
            )
            correction["mutation"] = propose_candidate_mutation(
                _mutation_parent_payload(
                    candidate_id=selected_parent_candidate_id or active_policy_candidate_id,
                    attempt=attempt,
                    variant=variant,
                    active_policy=active_policy_context,
                ),
                diagnosis=diagnosis,
                attempt_index=attempt_index + 1,
                generation_index=generation,
            )
            attempt["diagnosis"] = diagnosis
            attempt["correction"] = correction
            pending_correction = correction
            if _correction_name(correction) == "no_safe_correction_available":
                break

        attempts.extend(generation_attempts)
        if winner is not None:
            metric = copy.deepcopy(winner.get("metric") or {})
            winning_candidate_id = _attempt_winning_candidate_id(winner)
            last_winning_candidate_id = winning_candidate_id or last_winning_candidate_id
            selected_parent_for_next, next_parent_reason = _select_next_parent_after_generation(
                parent_policy=parent_policy,
                last_winning_candidate_id=last_winning_candidate_id,
                fixed_baseline_candidate_id=fixed_baseline_candidate_id,
                generation_won=True,
            )
            winner["selected_parent_for_next_generation"] = selected_parent_for_next
            winner["next_parent_selection_reason"] = next_parent_reason
            generation_metrics.append(metric)
            generation_summary = _generation_summary(
                generation=generation,
                generation_attempts=generation_attempts,
                exhausted=False,
                winner=winner,
                winning_metric=metric,
                selected_parent_for_next_generation=selected_parent_for_next,
                parent_selection_reason=next_parent_reason,
            )
            generation_summaries.append(generation_summary)
            lineage.append(_lineage_event_from_summary(generation_summary))
            continue

        exhausted = {
            "generation": generation,
            "attempts_used": len(generation_attempts),
            "max_attempts_per_generation": max_attempts_per_generation,
            "last_diagnosis": generation_attempts[-1].get("diagnosis") if generation_attempts else None,
            "last_correction": generation_attempts[-1].get("correction") if generation_attempts else None,
        }
        blockers.append({"type": "generation_exhausted", **exhausted})
        selected_parent_for_next, next_parent_reason = _select_next_parent_after_generation(
            parent_policy=parent_policy,
            last_winning_candidate_id=last_winning_candidate_id,
            fixed_baseline_candidate_id=fixed_baseline_candidate_id,
            generation_won=False,
        )
        generation_summary = _generation_summary(
            generation=generation,
            generation_attempts=generation_attempts,
            exhausted=True,
            winner=None,
            winning_metric=None,
            selected_parent_for_next_generation=selected_parent_for_next,
            parent_selection_reason=next_parent_reason,
        )
        generation_summaries.append(generation_summary)
        lineage.append(_lineage_event_from_summary(generation_summary))
        if continuation_policy == "stop":
            break

    infrastructure_failures = [attempt for attempt in attempts if attempt.get("process_failure")]
    passed = not blockers and not infrastructure_failures and len(generation_metrics) == generations
    blocker = None
    if infrastructure_failures:
        blocker = "infrastructure_failure"
    elif blockers:
        blocker = "generation_exhausted"

    return {
        "run_index": run_index,
        "variant": variant,
        "run_root": str(run_root),
        "generations_completed": len(generation_metrics),
        "generations_requested": generations,
        "passed": passed,
        "blocker": blocker,
        "blockers": blockers,
        "attempts": attempts,
        "generation_summaries": generation_summaries,
        "generation_metrics": generation_metrics,
        "lineage": lineage,
        "source_report": source_report,
        "active_policy_start": _active_policy_start_summary(active_policy_context),
    }


def _normalize_continuation_policy(policy: str) -> str:
    if policy == "continue":
        return "last-valid-parent"
    return policy


def _select_parent_for_generation(
    *,
    parent_policy: str,
    generation: int,
    last_winning_candidate_id: str | None,
    fixed_baseline_candidate_id: str | None,
) -> tuple[str | None, str]:
    if parent_policy == "fresh-baseline":
        return None, "fresh-baseline policy starts this generation from a newly seeded baseline"
    if parent_policy == "fixed-baseline":
        if fixed_baseline_candidate_id:
            return fixed_baseline_candidate_id, "fixed-baseline policy reuses the first observed source candidate"
        return None, "fixed-baseline policy has no observed baseline yet, so generation seeds the baseline"
    if last_winning_candidate_id:
        return last_winning_candidate_id, "last-valid-parent policy uses the latest winning child"
    return None, "no prior winner exists, so generation seeds the initial chain"


def _select_next_parent_after_generation(
    *,
    parent_policy: str,
    last_winning_candidate_id: str | None,
    fixed_baseline_candidate_id: str | None,
    generation_won: bool,
) -> tuple[str | None, str]:
    if parent_policy == "fresh-baseline":
        return None, "fresh-baseline policy does not carry lineage forward"
    if parent_policy == "fixed-baseline":
        if fixed_baseline_candidate_id:
            return fixed_baseline_candidate_id, "fixed-baseline policy keeps the original baseline as next parent"
        return None, "fixed-baseline policy has no source candidate to reuse yet"
    if last_winning_candidate_id:
        reason = "winner advances lineage to the next generation" if generation_won else "exhausted generation keeps last valid winner"
        return last_winning_candidate_id, reason
    return None, "no valid winner is available for the next generation"


def _generation_summary(
    *,
    generation: int,
    generation_attempts: list[dict[str, Any]],
    exhausted: bool,
    winner: dict[str, Any] | None,
    winning_metric: dict[str, Any] | None,
    selected_parent_for_next_generation: str | None,
    parent_selection_reason: str,
) -> dict[str, Any]:
    first_attempt = generation_attempts[0] if generation_attempts else {}
    winning_candidate_id = _attempt_winning_candidate_id(winner) if winner else None
    source_candidate_id = winner.get("source_candidate_id") if winner else None
    attempt_candidate_id = winner.get("attempt_candidate_id") if winner else None
    parent_candidate_id = winner.get("parent_candidate_id") if winner else first_attempt.get("parent_candidate_id")
    generated_child_candidate_id = winner.get("generated_child_candidate_id") if winner else None
    mutation = winner.get("mutation") if winner else None
    return {
        "generation": generation,
        "attempts": [_public_attempt(attempt) for attempt in generation_attempts],
        "diagnoses": [
            attempt.get("diagnosis")
            for attempt in generation_attempts
            if attempt.get("diagnosis")
        ],
        "corrections": [
            attempt.get("correction")
            for attempt in generation_attempts
            if attempt.get("correction")
        ],
        "exhausted": exhausted,
        "attempt_candidate_id": attempt_candidate_id,
        "generated_child_candidate_id": generated_child_candidate_id,
        "mutation": _public_mutation(mutation),
        "mutation_target": (mutation or {}).get("mutation_target") if isinstance(mutation, dict) else None,
        "mutation_seed": (mutation or {}).get("mutation_seed") if isinstance(mutation, dict) else None,
        "parent_candidate_id": parent_candidate_id,
        "parent_selection_reason": parent_selection_reason,
        "selected_parent_for_next_generation": selected_parent_for_next_generation,
        "source_candidate_id": source_candidate_id,
        "winning_attempt_id": winner.get("attempt_id") if winner else None,
        "winning_attempt_number": winner.get("attempt_index") if winner else None,
        "winning_candidate_id": winning_candidate_id,
        "winning_metric": winning_metric,
    }


def _lineage_event_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation": summary.get("generation"),
        "exhausted": summary.get("exhausted"),
        "parent_candidate_id": summary.get("parent_candidate_id"),
        "source_candidate_id": summary.get("source_candidate_id"),
        "attempt_candidate_id": summary.get("attempt_candidate_id"),
        "winning_candidate_id": summary.get("winning_candidate_id"),
        "generated_child_candidate_id": summary.get("generated_child_candidate_id"),
        "mutation_target": summary.get("mutation_target"),
        "mutation_seed": summary.get("mutation_seed"),
        "selected_parent_for_next_generation": summary.get("selected_parent_for_next_generation"),
        "parent_selection_reason": summary.get("parent_selection_reason"),
    }


def _attempt_winning_candidate_id(attempt: dict[str, Any] | None) -> str | None:
    if not attempt:
        return None
    return attempt.get("winning_candidate_id") or attempt.get("generated_child_candidate_id") or (
        attempt.get("metric") or {}
    ).get("candidate_id")


def _mutation_parent_payload(
    *,
    candidate_id: str | None,
    attempt: dict[str, Any] | None,
    variant: str,
    active_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {}
    if attempt and isinstance(attempt.get("attempt_candidate_manifest"), dict):
        manifest = copy.deepcopy(attempt["attempt_candidate_manifest"])
    if not manifest and active_policy:
        manifest = {
            "active_policy": _public_active_policy(active_policy),
            "active_policy_start": {
                "used_as_mutation_context": True,
                "used_as_db_parent": False,
            },
        }
    return {
        "candidate_id": candidate_id or (attempt or {}).get("source_candidate_id") or f"seed_parent:{variant}",
        "manifest": manifest,
    }


def diagnose_attempt_outcome(event: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    metric = event.get("metric") if isinstance(event.get("metric"), dict) else event
    source_report = event.get("source_report") if isinstance(event.get("source_report"), dict) else {}
    probe = event.get("replay_probe") or metric.get("replay_probe") or {}
    blocker = _joined_text(
        event.get("blocker"),
        source_report.get("blocker"),
        source_report.get("cycle_stopped_reason"),
        source_report.get("cycle_actions"),
        source_report.get("cycle_blockers"),
        source_report.get("cycle_state_transitions"),
        source_report.get("last_cycle_action"),
        source_report.get("last_cycle_blockers"),
        event.get("error"),
    )
    replay_decision = metric.get("replay_decision")
    if replay_decision is None:
        replay_decision = probe.get("replay_gate_decision")
    replay_state = metric.get("state_after_replay")
    if replay_state is None:
        replay_state = probe.get("state_after_replay")
    replay_delta = metric.get("replay_mean_delta")
    if replay_delta is None:
        replay_delta = probe.get("replay_mean_delta")
    min_delta = context.get("min_delta")
    adapter_status = metric.get("adapter_status")
    training_trained = metric.get("training_trained")
    adapter_loaded = metric.get("adapter_loaded")
    process_failure = bool(
        event.get("process_failure")
        or event.get("exception")
        or source_report.get("network_attempts")
        or source_report.get("model_modules_loaded")
        or source_report.get("fake_weights_written")
    )

    category = "unknown"
    reason = "No decisive failure signal was present."
    if "canary" in blocker or replay_decision == "canary_rejected":
        category = "canary_rejection"
        reason = "The attempt was blocked or rejected by canary evidence."
    elif "shadow" in blocker or replay_decision == "shadow_rejected":
        category = "shadow_rejection"
        reason = "The attempt was blocked or rejected by shadow evidence."
    elif "generated child" in blocker or ("missing" in blocker and "candidate" in blocker):
        category = "no_generated_child"
        reason = "The cycle did not produce the generated child payload required for replay."
    elif process_failure:
        category = "infrastructure_error"
        reason = "The attempt reported an infrastructure or safety process failure."
    elif training_trained is False or (adapter_status is not None and adapter_status != "trained"):
        category = "training_error"
        reason = "The local training artifact was not honestly trained."
    elif adapter_loaded is False:
        category = "adapter_load_error"
        reason = "The generated candidate did not load the trained local adapter."
    elif (
        replay_decision == "replay_winner"
        and _is_number(replay_delta)
        and _is_number(min_delta)
        and float(replay_delta) < float(min_delta)
    ):
        category = "insufficient_delta"
        reason = "Replay won, but the measured delta was below the conservative campaign floor."
    elif replay_decision is not None and replay_decision != "replay_winner":
        category = "replay_underperformance"
        reason = "Replay did not produce a replay_winner decision."
    elif replay_state is not None and replay_state != "replay_winner":
        category = "replay_underperformance"
        reason = "Replay did not leave the candidate in replay_winner state."
    elif _is_number(replay_delta) and float(replay_delta) <= 0.0:
        category = "replay_underperformance"
        reason = "Replay delta was non-positive."

    evidence = {
        "adapter_loaded": adapter_loaded,
        "adapter_status": adapter_status,
        "blocker": blocker or None,
        "replay_decision": replay_decision,
        "replay_mean_delta": replay_delta,
        "replay_state": replay_state,
        "training_trained": training_trained,
    }
    confidence = _diagnosis_confidence(category, evidence)
    recommended_corrections = _recommended_corrections(category)
    return {
        "category": category,
        "confidence": confidence,
        "diagnosis": reason,
        "evidence": evidence,
        "recommended_corrections": recommended_corrections,
        "reason": reason,
        "safe_to_retry": category != "infrastructure_error",
        "signals": evidence,
    }


def choose_correction(diagnosis: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or {}
    category = str(diagnosis.get("category") or "unknown")
    attempt_index = int(context.get("attempt_index") or 1)
    if category not in DIAGNOSIS_CATEGORIES:
        category = "unknown"

    action_by_category = {
        "adapter_load_error": "retry_with_fixed_baseline",
        "canary_rejection": "increase_canary_diversity",
        "insufficient_delta": "retry_with_fresh_candidate",
        "no_generated_child": "retry_with_last_valid_parent",
        "replay_underperformance": "increase_replay_diversity",
        "shadow_rejection": "lower_noise_shadow_sampling",
        "training_error": "reduce_trainer_noise",
        "unknown": "retry_with_fresh_candidate",
    }
    if category == "infrastructure_error":
        action = "no_safe_correction_available"
    elif category == "replay_underperformance" and attempt_index > 1:
        action = "isolate_generation_evidence"
    elif category == "training_error" and attempt_index > 1:
        action = "skip_bad_evidence_batch"
    else:
        action = action_by_category[category]

    return {
        "action": action,
        "correction": action,
        "diagnosis_category": category,
        "parameters": _correction_parameters(action, context),
        "reason": _correction_reason(action, category),
        "safe": action != "no_safe_correction_available",
    }


def apply_correction_to_generation_options(
    options: dict[str, Any],
    correction: dict[str, Any],
    attempt_index: int,
) -> dict[str, Any]:
    updated = copy.deepcopy(options)
    action = _correction_name(correction)
    if action not in CORRECTION_ACTIONS:
        action = "no_safe_correction_available"
    suffix = f"attempt{attempt_index:02d}_{action}"
    updated["correction_action"] = action
    updated["candidate_variant"] = f"{updated.get('candidate_variant', 'candidate')}__{suffix}"
    updated["theta_strategy_suffix"] = suffix
    updated["evidence_selection_seed"] = f"{updated.get('evidence_selection_seed', 'evidence')}__{suffix}"
    updated["evidence_suffix"] = suffix
    mutation = correction.get("mutation") if isinstance(correction.get("mutation"), dict) else None
    if mutation is not None:
        updated["candidate_mutation"] = _validated_mutation(mutation)

    trainer_config = copy.deepcopy(updated.get("trainer_config") or {})
    trainer_config["deterministic_variant"] = suffix
    updated["trainer_config"] = trainer_config

    if action == "increase_replay_diversity":
        updated["replay_variant"] = suffix
        updated["replay_selection_policy"] = "diverse_attempt_isolated"
    elif action == "increase_canary_diversity":
        updated["canary_variant"] = suffix
        updated["canary_selection_policy"] = "diverse_attempt_isolated"
        updated["force_new_canary_job"] = True
    elif action == "isolate_generation_evidence":
        updated["generation_evidence_scope"] = "attempt_isolated"
    elif action == "lower_noise_shadow_sampling":
        updated["shadow_variant"] = suffix
        updated["shadow_sampling_policy"] = "lower_noise"
    elif action == "retry_with_fresh_candidate":
        updated["parent_policy"] = "fresh_candidate"
    elif action == "retry_with_fixed_baseline":
        updated["parent_policy"] = "fixed_baseline"
    elif action == "retry_with_last_valid_parent":
        updated["parent_policy"] = "last_valid_parent"
    elif action == "skip_bad_evidence_batch":
        updated["evidence_batch_policy"] = "skip_previous_batch"
    elif action == "reduce_trainer_noise":
        trainer_config["noise_policy"] = "reduced"
        trainer_config["metadata_policy"] = "semantic_fields_only"
    elif action == "no_safe_correction_available":
        updated["safe_to_continue"] = False

    for key in IMMUTABLE_GATE_OPTION_KEYS:
        if key in options:
            updated[key] = copy.deepcopy(options[key])
    return updated


def _run_generation_attempt(
    *,
    generation_runner: GenerationRunner,
    generation: int,
    attempt_index: int,
    run_root: Path,
    run_id: str,
    variant: str,
    min_delta: float,
    parent_candidate_id: str | None,
    parent_selection_reason: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    attempt_id = f"generation_{generation:02d}_attempt_{attempt_index:02d}"
    suffix = _safe_suffix(options.get("correction_action") or "initial")
    attempt_run_id = f"{run_id}_gen{generation:02d}_attempt{attempt_index:02d}_{suffix}"
    attempt_root = run_root / f"generation_{generation:02d}" / f"attempt_{attempt_index:02d}_{suffix}"
    runner_options = copy.deepcopy(options)
    runner_options.setdefault("attempt_artifact_suffix", f"trained_generation_{generation:02d}_attempt_{attempt_index:02d}_{suffix}")
    runner_options["run_parent_as_source"] = False
    mutation = _public_mutation(runner_options.get("candidate_mutation"))
    try:
        source_report = _call_generation_runner(
            generation_runner,
            brain_root=run_root,
            run_id=attempt_run_id,
            generations=1,
            generation=generation,
            options=runner_options,
            parent_candidate_id=parent_candidate_id,
        )
        source_report["report_path"] = str(attempt_root / "peds_generation_report.json")
        attempt_root.mkdir(parents=True, exist_ok=True)
        (attempt_root / "peds_generation_report.json").write_text(
            json.dumps(source_report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metrics = _generation_metrics_from_report(source_report, generation_override=generation)
        metric = metrics[0] if metrics else _metric_from_report_blocker(source_report, generation)
        process_failure = _source_report_has_process_failure(source_report)
        outcome = "winner" if not process_failure and _metric_won(metric, min_delta=min_delta) else "rejected"
        if process_failure:
            outcome = "rejected"
        source_candidate_id = (
            metric.get("source_candidate_id")
            or source_report.get("input_source_candidate_id")
            or metric.get("parent_candidate_id")
            or parent_candidate_id
        )
        attempt_candidate_id = source_report.get("attempt_candidate_id") or source_candidate_id
        generated_child_candidate_id = metric.get("generated_child_candidate_id")
        if generated_child_candidate_id is None and metric.get("replay_decision") is not None:
            generated_child_candidate_id = metric.get("candidate_id")
        winning_candidate_id = generated_child_candidate_id if outcome == "winner" else None
        return {
            "attempt_candidate_id": attempt_candidate_id,
            "attempt_candidate_manifest": source_report.get("input_source_manifest"),
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "attempt_run_id": attempt_run_id,
            "attempt_root": str(attempt_root),
            "brain_root": str(run_root),
            "evidence": _attempt_evidence(source_report),
            "generated_child_candidate_id": generated_child_candidate_id,
            "generation": generation,
            "metric": metric,
            "min_delta": min_delta,
            "mutation": mutation,
            "mutation_target": (mutation or {}).get("mutation_target") if isinstance(mutation, dict) else None,
            "mutation_seed": (mutation or {}).get("mutation_seed") if isinstance(mutation, dict) else None,
            "options": _public_generation_options(runner_options),
            "outcome": outcome,
            "parent_candidate_id": parent_candidate_id,
            "parent_selection_reason": parent_selection_reason,
            "parent_state_after_attempt": source_report.get("input_parent_state_after_attempt"),
            "parent_state_before_attempt": source_report.get("input_parent_state_before_attempt"),
            "process_failure": process_failure,
            "report_path": source_report["report_path"],
            "run_root": str(run_root),
            "selected_parent_for_next_generation": winning_candidate_id if outcome == "winner" else None,
            "source_candidate_id": source_candidate_id,
            "source_report": source_report,
            "variant": variant,
            "winning_candidate_id": winning_candidate_id,
        }
    except Exception as exc:  # noqa: BLE001 - attempts are evidence, infrastructure failures are explicit.
        metric = {
            "generation": generation,
            "source_candidate_id": parent_candidate_id,
            "replay_decision": None,
            "replay_mean_delta": None,
            "adapter_loaded": None,
            "training_trained": None,
        }
        return {
            "attempt_candidate_id": None,
            "attempt_candidate_manifest": None,
            "attempt_id": attempt_id,
            "attempt_index": attempt_index,
            "attempt_run_id": attempt_run_id,
            "attempt_root": str(attempt_root),
            "blocker": f"{type(exc).__name__}: {exc}",
            "brain_root": str(run_root),
            "error": f"{type(exc).__name__}: {exc}",
            "evidence": {},
            "generated_child_candidate_id": None,
            "generation": generation,
            "metric": metric,
            "min_delta": min_delta,
            "mutation": mutation,
            "mutation_target": (mutation or {}).get("mutation_target") if isinstance(mutation, dict) else None,
            "mutation_seed": (mutation or {}).get("mutation_seed") if isinstance(mutation, dict) else None,
            "options": _public_generation_options(options),
            "outcome": "rejected",
            "parent_candidate_id": parent_candidate_id,
            "parent_selection_reason": parent_selection_reason,
            "parent_state_after_attempt": None,
            "parent_state_before_attempt": None,
            "process_failure": True,
            "report_path": None,
            "run_root": str(run_root),
            "selected_parent_for_next_generation": None,
            "source_candidate_id": None,
            "source_report": {},
            "variant": variant,
            "winning_candidate_id": None,
        }


def _attempt_evidence(source_report: dict[str, Any]) -> dict[str, Any]:
    generation_reports = source_report.get("generation_reports") or []
    if generation_reports and isinstance(generation_reports[0], dict):
        evidence = generation_reports[0].get("evidence")
        return copy.deepcopy(evidence) if isinstance(evidence, dict) else {}
    evidence = source_report.get("evidence")
    return copy.deepcopy(evidence) if isinstance(evidence, dict) else {}


def _generation_metrics_from_report(
    report: dict[str, Any],
    *,
    generation_override: int | None = None,
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for generation_report in report.get("generation_reports", []):
        probe = generation_report.get("replay_probe") or {}
        source_candidate_id = generation_report.get("source_candidate_id") or report.get("input_source_candidate_id")
        metrics.append(
            {
                "generation": generation_override or generation_report.get("generation"),
                "candidate_id": generation_report.get("generated_candidate_id"),
                "attempt_candidate_id": source_candidate_id,
                "generated_child_candidate_id": generation_report.get("generated_candidate_id"),
                "source_candidate_id": source_candidate_id,
                "parent_candidate_id": report.get("input_parent_candidate_id"),
                "adapter_status": generation_report.get("adapter_registry_status"),
                "training_trained": generation_report.get("training_trained"),
                "adapter_loaded": probe.get("adapter_loaded"),
                "replay_decision": probe.get("replay_gate_decision"),
                "replay_mean_delta": probe.get("replay_mean_delta"),
                "replay_results_written": probe.get("replay_results_written"),
                "state_after_replay": probe.get("state_after_replay"),
            }
        )
    return metrics


def _metric_won(metric: dict[str, Any], *, min_delta: float) -> bool:
    if _metric_collapsed(metric):
        return False
    delta = metric.get("replay_mean_delta")
    return _is_number(delta) and float(delta) >= float(min_delta)


def _metric_from_report_blocker(report: dict[str, Any], generation: int) -> dict[str, Any]:
    return {
        "generation": generation,
        "candidate_id": report.get("final_candidate_id"),
        "attempt_candidate_id": report.get("input_source_candidate_id") or report.get("final_candidate_id"),
        "generated_child_candidate_id": None,
        "source_candidate_id": report.get("input_source_candidate_id") or report.get("final_candidate_id"),
        "parent_candidate_id": report.get("input_parent_candidate_id"),
        "adapter_status": None,
        "training_trained": None,
        "adapter_loaded": None,
        "replay_decision": None,
        "replay_mean_delta": None,
        "replay_results_written": None,
        "state_after_replay": report.get("final_candidate_state"),
    }


def _source_report_has_process_failure(report: dict[str, Any]) -> bool:
    return bool(
        report.get("network_attempts")
        or report.get("model_modules_loaded")
        or report.get("fake_weights_written")
    )


def _call_generation_runner(
    generation_runner: GenerationRunner,
    *,
    brain_root: Path,
    run_id: str,
    generations: int,
    generation: int,
    options: dict[str, Any],
    parent_candidate_id: str | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "brain_root": brain_root,
        "run_id": run_id,
        "generations": generations,
        "strict": False,
    }
    try:
        signature = inspect.signature(generation_runner)
        parameters = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        if accepts_kwargs or "generation_options" in parameters:
            kwargs["generation_options"] = options
        if accepts_kwargs or "generation" in parameters:
            kwargs["generation"] = generation
        if accepts_kwargs or "parent_candidate_id" in parameters:
            kwargs["parent_candidate_id"] = parent_candidate_id
    except (TypeError, ValueError):
        pass
    return generation_runner(**kwargs)


def _public_generation_options(options: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "active_policy",
        "active_policy_candidate_id",
        "active_policy_start",
        "attempt_artifact_suffix",
        "baseline_mode",
        "candidate_mutation",
        "candidate_variant",
        "canary_selection_policy",
        "canary_thresholds",
        "canary_variant",
        "correction_action",
        "evidence_batch_policy",
        "evidence_selection_seed",
        "evidence_suffix",
        "force_new_canary_job",
        "generation",
        "generation_evidence_scope",
        "parent_candidate_id",
        "parent_policy",
        "parent_selection_reason",
        "replay_selection_policy",
        "replay_thresholds",
        "replay_variant",
        "run_parent_as_source",
        "safe_to_continue",
        "shadow_sampling_policy",
        "shadow_thresholds",
        "shadow_variant",
        "theta_strategy_suffix",
        "trainer_config",
    )
    return {key: copy.deepcopy(options[key]) for key in keys if key in options}


def _public_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_candidate_id": attempt.get("attempt_candidate_id"),
        "attempt_candidate_manifest": attempt.get("attempt_candidate_manifest"),
        "attempt_id": attempt.get("attempt_id"),
        "attempt_index": attempt.get("attempt_index"),
        "attempt_run_id": attempt.get("attempt_run_id"),
        "attempt_root": attempt.get("attempt_root"),
        "brain_root": attempt.get("brain_root"),
        "correction": attempt.get("correction"),
        "diagnosis": attempt.get("diagnosis"),
        "evidence": attempt.get("evidence", {}),
        "generated_child_candidate_id": attempt.get("generated_child_candidate_id"),
        "generation": attempt.get("generation"),
        "metric": attempt.get("metric"),
        "min_delta": attempt.get("min_delta"),
        "mutation": _public_mutation(attempt.get("mutation")),
        "mutation_target": attempt.get("mutation_target"),
        "mutation_seed": attempt.get("mutation_seed"),
        "options": attempt.get("options", {}),
        "outcome": attempt.get("outcome"),
        "parent_candidate_id": attempt.get("parent_candidate_id"),
        "parent_selection_reason": attempt.get("parent_selection_reason"),
        "parent_state_after_attempt": attempt.get("parent_state_after_attempt"),
        "parent_state_before_attempt": attempt.get("parent_state_before_attempt"),
        "process_failure": attempt.get("process_failure", False),
        "report_path": attempt.get("report_path"),
        "run_root": attempt.get("run_root"),
        "selected_parent_for_next_generation": attempt.get("selected_parent_for_next_generation"),
        "source_candidate_id": attempt.get("source_candidate_id"),
        "variant": attempt.get("variant"),
        "winning_candidate_id": attempt.get("winning_candidate_id"),
    }


def _public_attempts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_public_attempt(attempt) for run in runs for attempt in run.get("attempts", [])]


def _public_rejected_attempts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _public_attempt(attempt)
        for run in runs
        for attempt in run.get("attempts", [])
        if attempt.get("outcome") == "rejected"
    ]


def _public_diagnoses(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for attempt in run.get("attempts", []):
            if not attempt.get("diagnosis"):
                continue
            rows.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "attempt_index": attempt.get("attempt_index"),
                    "category": attempt["diagnosis"].get("category"),
                    "diagnosis": attempt["diagnosis"],
                    "generation": attempt.get("generation"),
                    "variant": attempt.get("variant"),
                }
            )
    return rows


def _public_corrections(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for attempt in run.get("attempts", []):
            if not attempt.get("correction"):
                continue
            rows.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "attempt_index": attempt.get("attempt_index"),
                    "action": _correction_name(attempt["correction"]),
                    "correction": attempt["correction"],
                    "generation": attempt.get("generation"),
                    "variant": attempt.get("variant"),
                }
            )
    return rows


def _public_mutations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for attempt in run.get("attempts", []):
            mutation = _public_mutation(attempt.get("mutation"))
            if mutation is None:
                continue
            rows.append(
                {
                    "attempt_id": attempt.get("attempt_id"),
                    "attempt_index": attempt.get("attempt_index"),
                    "generation": attempt.get("generation"),
                    "variant": attempt.get("variant"),
                    "mutation": mutation,
                    "mutation_target": mutation.get("mutation_target"),
                    "mutation_seed": mutation.get("mutation_seed"),
                    "diagnosis_category": mutation.get("diagnosis_category"),
                    "delta_description": mutation.get("delta_description"),
                }
            )
    return rows


def _public_winning_attempts(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for generation in run.get("generation_summaries", []):
            if generation.get("winning_attempt_number") is None:
                continue
            rows.append(
                {
                    "generation": generation.get("generation"),
                    "attempt_candidate_id": generation.get("attempt_candidate_id"),
                    "generated_child_candidate_id": generation.get("generated_child_candidate_id"),
                    "mutation": generation.get("mutation"),
                    "mutation_target": generation.get("mutation_target"),
                    "mutation_seed": generation.get("mutation_seed"),
                    "parent_candidate_id": generation.get("parent_candidate_id"),
                    "parent_selection_reason": generation.get("parent_selection_reason"),
                    "run_index": run.get("run_index"),
                    "selected_parent_for_next_generation": generation.get("selected_parent_for_next_generation"),
                    "source_candidate_id": generation.get("source_candidate_id"),
                    "variant": run.get("variant"),
                    "winning_attempt_id": generation.get("winning_attempt_id"),
                    "winning_attempt_number": generation.get("winning_attempt_number"),
                    "winning_candidate_id": generation.get("winning_candidate_id"),
                }
            )
    return rows


def _public_exhausted_generations(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for generation in run.get("generation_summaries", []):
            if not generation.get("exhausted"):
                continue
            rows.append(
                {
                    "generation": generation.get("generation"),
                    "attempt_candidate_id": generation.get("attempt_candidate_id"),
                    "mutation": generation.get("mutation"),
                    "mutation_target": generation.get("mutation_target"),
                    "mutation_seed": generation.get("mutation_seed"),
                    "parent_candidate_id": generation.get("parent_candidate_id"),
                    "parent_selection_reason": generation.get("parent_selection_reason"),
                    "run_index": run.get("run_index"),
                    "selected_parent_for_next_generation": generation.get("selected_parent_for_next_generation"),
                    "variant": run.get("variant"),
                    "winning_attempt_number": None,
                }
            )
    return rows


def _public_lineage(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for item in run.get("lineage", []):
            row = copy.deepcopy(item)
            row["run_index"] = run.get("run_index")
            row["variant"] = run.get("variant")
            rows.append(row)
    return rows


def _correction_stats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    corrections_attempted = 0
    corrections_followed_by_winner = 0
    for run in runs:
        for generation in run.get("generation_summaries", []):
            winning_attempt = generation.get("winning_attempt_number")
            for attempt in generation.get("attempts", []):
                correction = attempt.get("correction") or {}
                if _correction_name(correction) in (None, "no_safe_correction_available"):
                    continue
                corrections_attempted += 1
                if winning_attempt is not None and attempt.get("attempt_index", 0) < winning_attempt:
                    corrections_followed_by_winner += 1
    return {
        "corrections_attempted": corrections_attempted,
        "corrections_followed_by_winner": corrections_followed_by_winner,
        "correction_success_rate": (
            corrections_followed_by_winner / corrections_attempted if corrections_attempted else 0.0
        ),
    }


def _correction_reason(action: str, category: str) -> str:
    reasons = {
        "increase_replay_diversity": "Try a more diverse replay selection without lowering replay gates.",
        "increase_canary_diversity": "Try a more diverse canary selection without weakening canary rejection.",
        "isolate_generation_evidence": "Isolate this generation's evidence to reduce stale or contaminated replay input.",
        "lower_noise_shadow_sampling": "Reduce noisy shadow sampling while keeping shadow rejection intact.",
        "retry_with_fresh_candidate": "Create a fresh candidate variant for the next attempt.",
        "retry_with_fixed_baseline": "Retry against a fixed baseline to separate adapter loading from lineage drift.",
        "retry_with_last_valid_parent": "Retry from the last valid parent after no generated child was produced.",
        "skip_bad_evidence_batch": "Skip the suspect evidence batch and keep trainer gates unchanged.",
        "reduce_trainer_noise": "Reduce trainer input noise while preserving the trained-artifact requirement.",
        "no_safe_correction_available": "No safe campaign-level correction is available for this diagnosis.",
    }
    return f"{reasons[action]} Diagnosis: {category}."


def _diagnosis_confidence(category: str, evidence: dict[str, Any]) -> float:
    if category == "unknown":
        return 0.2
    strong_signals = {
        "adapter_load_error": evidence.get("adapter_loaded") is False,
        "canary_rejection": bool(evidence.get("blocker")) and "canary" in str(evidence.get("blocker")),
        "infrastructure_error": bool(evidence.get("blocker")),
        "insufficient_delta": _is_number(evidence.get("replay_mean_delta")),
        "replay_underperformance": evidence.get("replay_decision") not in (None, "replay_winner"),
        "shadow_rejection": bool(evidence.get("blocker")) and "shadow" in str(evidence.get("blocker")),
        "training_error": evidence.get("training_trained") is False or evidence.get("adapter_status") not in (None, "trained"),
    }
    return 0.9 if strong_signals.get(category) else 0.65


def _recommended_corrections(category: str) -> list[str]:
    mapping = {
        "adapter_load_error": ["retry_with_fixed_baseline", "retry_with_last_valid_parent"],
        "canary_rejection": ["increase_canary_diversity", "retry_with_fresh_candidate"],
        "insufficient_delta": ["retry_with_fresh_candidate", "increase_replay_diversity"],
        "no_generated_child": ["retry_with_last_valid_parent", "retry_with_fresh_candidate"],
        "replay_underperformance": ["increase_replay_diversity", "isolate_generation_evidence"],
        "shadow_rejection": ["lower_noise_shadow_sampling", "isolate_generation_evidence"],
        "training_error": ["reduce_trainer_noise", "skip_bad_evidence_batch"],
        "infrastructure_error": ["no_safe_correction_available"],
        "unknown": ["retry_with_fresh_candidate"],
    }
    return mapping.get(category, ["no_safe_correction_available"])


def _correction_parameters(action: str, context: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_index": context.get("attempt_index"),
        "generation": context.get("generation"),
        "max_attempts_per_generation": context.get("max_attempts_per_generation"),
    }


def _correction_name(correction: dict[str, Any]) -> str:
    return str(correction.get("correction") or correction.get("action") or "no_safe_correction_available")


def _validated_mutation(mutation: dict[str, Any]) -> dict[str, Any]:
    validation = validate_single_mutation(mutation)
    if not validation["valid"]:
        raise ValueError("Invalid candidate mutation: " + "; ".join(validation["failed_checks"]))
    return copy.deepcopy(mutation)


def _public_mutation(mutation: Any) -> dict[str, Any] | None:
    if not isinstance(mutation, dict):
        return None
    return {
        "mutation_target": mutation.get("mutation_target"),
        "theta_strategy": copy.deepcopy(mutation.get("theta_strategy")),
        "retrieval_policy": copy.deepcopy(mutation.get("retrieval_policy")),
        "tool_policy": copy.deepcopy(mutation.get("tool_policy")),
        "update_policy": copy.deepcopy(mutation.get("update_policy")),
        "delta_description": mutation.get("delta_description"),
        "mutation_seed": mutation.get("mutation_seed"),
        "diagnosis_category": mutation.get("diagnosis_category"),
    }


def _joined_text(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if value in (None, "", [], {}):
            continue
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


def _safe_suffix(value: Any) -> str:
    text = str(value or "attempt").lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part)[:80] or "attempt"


def _run_passed(report: dict[str, Any], metrics: list[dict[str, Any]], *, generations_requested: int) -> bool:
    if report.get("network_attempts") or report.get("model_modules_loaded") or report.get("fake_weights_written"):
        return False
    if report.get("blocker"):
        return False
    return len(metrics) == generations_requested and not any(_metric_collapsed(metric) for metric in metrics)


def _metric_collapsed(metric: dict[str, Any]) -> bool:
    delta = metric.get("replay_mean_delta")
    return (
        metric.get("replay_decision") != "replay_winner"
        or not _is_number(delta)
        or float(delta) <= 0.0
        or metric.get("adapter_loaded") is not True
        or metric.get("training_trained") is not True
    )


def _first_failure(runs: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    for metric in metrics:
        if _metric_collapsed(metric):
            return metric
    for run in runs:
        if run.get("blocker"):
            return {
                "run_index": run.get("run_index"),
                "variant": run.get("variant"),
                "blocker": run.get("blocker"),
            }
    return None


def _infer_run_blocker(report: dict[str, Any], metrics: list[dict[str, Any]]) -> str | None:
    if report.get("blocker"):
        reason = str(report["blocker"])
        stopped = report.get("cycle_stopped_reason")
        blocked_generation = report.get("blocked_generation")
        parts = [reason]
        if blocked_generation is not None:
            parts.append(f"blocked_generation={blocked_generation}")
        if stopped:
            parts.append(f"cycle_stopped_reason={stopped}")
        return "; ".join(parts)
    if report.get("network_attempts"):
        return "network_attempts_present"
    if report.get("model_modules_loaded"):
        return "model_modules_loaded"
    if report.get("fake_weights_written"):
        return "fake_weights_written"
    first = _first_failure([], metrics)
    if first:
        return f"generation_{first.get('generation')}_collapsed"
    return "no_generation_metrics"


def _delta_stat_by_generation(
    metrics: list[dict[str, Any]],
    generations_requested: int,
    stat: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for generation in range(1, generations_requested + 1):
        values = [
            float(metric["replay_mean_delta"])
            for metric in metrics
            if metric.get("generation") == generation and _is_number(metric.get("replay_mean_delta"))
        ]
        if not values:
            value = None
        elif stat == "mean":
            value = sum(values) / len(values)
        elif stat == "min":
            value = min(values)
        elif stat == "max":
            value = max(values)
        else:
            raise ValueError(f"Unknown delta stat: {stat}")
        rows.append({"generation": generation, f"{stat}_delta": value})
    return rows


def _delta_trend(mean_by_generation: list[dict[str, Any]], epsilon: float) -> str:
    values = [row.get("mean_delta") for row in mean_by_generation if _is_number(row.get("mean_delta"))]
    if len(values) < 2:
        return "unknown"
    diffs = [float(later) - float(earlier) for earlier, later in zip(values, values[1:])]
    if all(abs(diff) <= epsilon for diff in diffs):
        return "flat"
    improving = [diff for diff in diffs if diff > epsilon]
    decaying = [diff for diff in diffs if diff < -epsilon]
    threshold = max(1, int(len(diffs) * 0.7))
    if len(improving) >= threshold and float(values[-1]) > float(values[0]) + epsilon:
        return "improving"
    if len(decaying) >= threshold and float(values[-1]) < float(values[0]) - epsilon:
        return "decaying"
    return "mixed"


def _plateau_detected(mean_by_generation: list[dict[str, Any]], window: int, epsilon: float) -> bool:
    if window <= 1:
        return False
    values = [row.get("mean_delta") for row in mean_by_generation if _is_number(row.get("mean_delta"))]
    if len(values) < window:
        return False
    for start in range(0, len(values) - window + 1):
        subset = [float(value) for value in values[start : start + window]]
        if max(subset) - min(subset) <= epsilon:
            return True
    return False


def _rate(metrics: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    if not metrics:
        return 0.0
    return sum(1 for metric in metrics if predicate(metric)) / len(metrics)


def _merge_safety(safety: dict[str, Any], report: dict[str, Any]) -> None:
    for key in ("network_attempts", "model_modules_loaded"):
        for item in report.get(key, []):
            if item not in safety[key]:
                safety[key].append(item)
    safety["fake_weights_written"] = bool(safety["fake_weights_written"] or report.get("fake_weights_written"))


def _merge_attempt_safety(safety: dict[str, Any], summary: dict[str, Any]) -> None:
    for attempt in summary.get("attempts", []):
        report = attempt.get("source_report") or {}
        _merge_safety(safety, report)


def _public_run_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_index": summary.get("run_index"),
        "variant": summary.get("variant"),
        "run_root": summary.get("run_root"),
        "generations_completed": summary.get("generations_completed"),
        "generations_requested": summary.get("generations_requested"),
        "passed": summary.get("passed"),
        "blocker": summary.get("blocker"),
        "blockers": summary.get("blockers", []),
        "attempts": [_public_attempt(attempt) for attempt in summary.get("attempts", [])],
        "active_policy_start": summary.get("active_policy_start"),
        "generation_summaries": summary.get("generation_summaries", []),
        "generation_metrics": summary.get("generation_metrics", []),
        "lineage": summary.get("lineage", []),
    }


def _select_executable_variants(variants: list[str], baseline_mode: str) -> tuple[list[str], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    executable: list[str] = []
    if baseline_mode != "parent":
        blockers.append(
            {
                "type": "unsupported_baseline_mode",
                "baseline_mode": baseline_mode,
                "reason": "Existing generation runner cleanly supports parent-chained replay only.",
            }
        )
        return [], blockers
    for variant in variants:
        if variant in SUPPORTED_VARIANTS and variant not in executable:
            executable.append(variant)
            continue
        reason = "Variant support would require changing the current generation evidence path."
        if variant not in KNOWN_UNSUPPORTED_VARIANTS:
            reason = "Unknown campaign variant."
        blockers.append({"type": "unsupported_variant", "variant": variant, "reason": reason})
    return executable, blockers


def _campaign_exit_code(report: dict[str, Any], *, strict: bool = False) -> int:
    aggregate = report.get("aggregate") or {}
    safety = report.get("safety") or {}
    infrastructure_failed = bool(
        aggregate.get("infrastructure_failure_count")
        or safety.get("network_attempts")
        or safety.get("model_modules_loaded")
        or safety.get("fake_weights_written")
    )
    no_executable_work = bool(report.get("blockers")) and not report.get("attempts")
    if infrastructure_failed or no_executable_work:
        return 1
    if strict and (
        aggregate.get("collapse_detected")
        or aggregate.get("exhausted_generation_count")
        or report.get("blockers")
    ):
        return 1
    return 0


def _parse_variants(values: list[str] | None) -> list[str]:
    if not values:
        return ["acceptance_checklist_default"]
    variants: list[str] = []
    for value in values:
        variants.extend(item.strip() for item in value.split(",") if item.strip())
    return variants or ["acceptance_checklist_default"]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _campaign_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and analyze a measured PEDS training campaign.")
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--variant", action="append", default=None)
    parser.add_argument("--baseline-mode", choices=["parent", "fixed-baseline"], default="parent")
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument(
        "--max-attempts-per-generation",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS_PER_GENERATION,
    )
    parser.add_argument(
        "--continuation-policy",
        choices=sorted(CONTINUATION_POLICIES),
        default="last-valid-parent",
    )
    parser.add_argument("--plateau-window", type=int, default=5)
    parser.add_argument("--plateau-epsilon", type=float, default=0.005)
    parser.add_argument("--active-policy-file", type=Path, default=None)
    parser.add_argument("--active-policy-json", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
