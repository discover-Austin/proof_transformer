from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
SCRIPTS = PROJECT_ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from peds_latest_campaign_report import build_latest_campaign_report  # noqa: E402
from proof_transformer.peds.survivor_outputs import (  # noqa: E402
    append_active_policy,
    build_codex_prompt_from_survivor,
    build_repo_improvement_plan,
    build_survivor_profile,
    load_latest_active_policy,
    select_current_survivor,
    write_autopilot_artifacts,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp" / "peds_autopilot_outputs"
CAMPAIGN_PRESETS = {
    "3gen": {"generations": 3, "runs": 1, "max_attempts_per_generation": 3},
    "5x2": {"generations": 5, "runs": 2, "max_attempts_per_generation": 5},
    "10x3": {"generations": 10, "runs": 3, "max_attempts_per_generation": 5},
    "mutation-stress": {"generations": 10, "runs": 3, "max_attempts_per_generation": 5},
    "none": None,
}

ReportLoader = Callable[[], dict[str, Any]]
CampaignRunner = Callable[..., dict[str, Any]]


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
    normalized = {key: deepcopy(policy[key]) for key in keys if policy.get(key) not in (None, "")}
    safety = policy.get("safety")
    if isinstance(safety, dict):
        normalized["safety"] = deepcopy(safety)
    if source is not None:
        normalized["source"] = source
    elif policy.get("source"):
        normalized["source"] = str(policy["source"])
    return normalized


def _public_active_policy(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    normalized = _normalize_active_policy(policy)
    return deepcopy(normalized) if normalized else None


def _active_policy_start_summary(policy: dict[str, Any] | None) -> dict[str, Any]:
    normalized = _normalize_active_policy(policy)
    return {
        "supported": True,
        "provided": bool(normalized),
        "used": bool(normalized),
        "candidate_id": normalized.get("candidate_id") if normalized else None,
        "active_policy_id": normalized.get("active_policy_id") if normalized else None,
        "source": normalized.get("source") if normalized else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    active_policy = _load_active_policy(args.active_policy_json, args.active_policy_file)
    result = run_autopilot(
        objective=args.objective,
        campaign=args.campaign,
        use_latest=args.use_latest,
        activate_best=args.activate_best,
        write=args.write,
        include_details=args.include_details,
        output_root=args.output_root,
        report_root=args.report_root,
        max_attempts_per_generation=args.max_attempts_per_generation,
        strict=args.strict,
        artifact_type=args.artifact_type,
        active_policy=active_policy,
    )
    indent = None if args.json else 2
    print(json.dumps(result, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0 if result.get("ok") else 1


def run_autopilot(
    *,
    objective: str,
    campaign: str = "5x2",
    use_latest: bool = False,
    activate_best: bool = False,
    write: bool = False,
    include_details: bool = False,
    output_root: Path | str | None = None,
    report_root: Path | str | None = None,
    max_attempts_per_generation: int | None = None,
    strict: bool = False,
    artifact_type: str = "all",
    report_loader: ReportLoader | None = None,
    campaign_runner: CampaignRunner | None = None,
    active_registry_root: Path | None = None,
    active_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        objective = _require_objective(objective)
    except ValueError as exc:
        return _error(str(exc), campaign=campaign)
    if campaign not in CAMPAIGN_PRESETS:
        return _error(f"Unsupported campaign preset: {campaign!r}", objective=objective, campaign=campaign)
    if artifact_type not in {"all", "profile", "prompt", "plan", "skill"}:
        return _error(f"Unsupported artifact type: {artifact_type!r}", objective=objective, campaign=campaign)

    input_active_policy = _normalize_active_policy(active_policy)
    campaign_ran = False
    campaign_result: dict[str, Any] | None = None
    if not use_latest and campaign != "none":
        runner = campaign_runner or _run_campaign_subprocess
        campaign_result = _call_campaign_runner(
            runner,
            campaign,
            max_attempts_per_generation,
            strict,
            active_policy=input_active_policy,
        )
        campaign_ran = bool(campaign_result.get("ok"))
        if not campaign_result.get("ok"):
            return {
                "ok": False,
                "objective": objective,
                "campaign": campaign,
                "campaign_ran": campaign_ran,
                "error": campaign_result.get("error") or "campaign failed",
                "campaign_result": campaign_result,
                "input_active_policy": _public_active_policy(input_active_policy),
                "active_policy_start": _active_policy_start_summary(input_active_policy),
            }

    loader = report_loader or (lambda: build_latest_campaign_report(report_root))
    report = loader()
    if not isinstance(report, dict) or not report.get("found", bool(report.get("campaign_id"))):
        return {
            "ok": False,
            "objective": objective,
            "campaign": campaign,
            "campaign_ran": campaign_ran,
            "error": (report or {}).get("reason") or "No campaign report is available for survivor selection.",
            "report_found": False,
            "input_active_policy": _public_active_policy(input_active_policy),
            "active_policy_start": _active_policy_start_summary(input_active_policy),
        }

    survivor = select_current_survivor(report)
    if survivor is None:
        return {
            "ok": False,
            "objective": objective,
            "campaign": campaign,
            "campaign_ran": campaign_ran,
            "error": "No usable clean survivor is available from the latest campaign evidence.",
            "report_found": True,
            "safety": report.get("safety", {}),
            "input_active_policy": _public_active_policy(input_active_policy),
            "active_policy_start": _active_policy_start_summary(input_active_policy),
        }

    try:
        profile = build_survivor_profile(report, objective, survivor)
    except ValueError as exc:
        return _error(str(exc), objective=objective, campaign=campaign, campaign_ran=campaign_ran)

    output_dir: Path | None = None
    artifacts: dict[str, str] = {}
    if write:
        root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
        output_dir = root / _run_id()
        artifacts = write_autopilot_artifacts(profile, output_dir, artifact_type=artifact_type)
        if artifacts.get("survivor_profile"):
            profile = dict(profile, profile_path=artifacts["survivor_profile"])
        summary_path = output_dir / "autopilot_summary.json"
    else:
        summary_path = None

    activated_policy = None
    if activate_best:
        activated_policy = append_active_policy(profile, registry_root=active_registry_root)

    prompt = build_codex_prompt_from_survivor(profile)
    plan = build_repo_improvement_plan(profile)
    result: dict[str, Any] = {
        "ok": True,
        "objective": objective,
        "campaign": campaign,
        "campaign_ran": campaign_ran,
        "campaign_result": campaign_result if include_details else _compact_campaign_result(campaign_result),
        "report_found": True,
        "campaign_id": report.get("campaign_id"),
        "campaign_root": report.get("campaign_root"),
        "report_path": report.get("report_path"),
        "survivor_selected": True,
        "candidate_id": survivor.get("candidate_id"),
        "selection_reason": survivor.get("selection_reason"),
        "generation": survivor.get("generation"),
        "attempt_index": survivor.get("attempt_index"),
        "mutation_target": survivor.get("mutation_target"),
        "mutation_seed": survivor.get("mutation_seed"),
        "diagnosis_category": survivor.get("diagnosis_category"),
        "useful_output_ready": bool(prompt and plan),
        "artifacts_written": bool(artifacts),
        "artifacts": artifacts,
        "output_dir": str(output_dir) if output_dir else None,
        "input_active_policy": _public_active_policy(input_active_policy),
        "active_policy_start": _active_policy_start_summary(input_active_policy),
        "active_policy": activated_policy,
        "latest_active_policy": activated_policy or _public_active_policy(input_active_policy) or load_latest_active_policy(),
        "safety": report.get("safety", {}),
        "evidence_summary": profile.get("evidence_summary"),
        "agentic_capability_coverage": profile.get("agentic_capability_coverage"),
        "agentic_next_eval_tasks": (
            (profile.get("agentic_capability_coverage") or {}).get("next_eval_tasks")
            if isinstance(profile.get("agentic_capability_coverage"), dict)
            else []
        ),
        "what_improved": profile.get("what_improved"),
        "recommended_uses": profile.get("recommended_uses"),
        "codex_prompt_preview": prompt[:1200],
    }
    if include_details:
        result["survivor"] = survivor
        result["profile"] = profile
    if write and summary_path is not None:
        result["summary_path"] = str(summary_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _call_campaign_runner(
    runner: CampaignRunner,
    campaign: str,
    max_attempts_per_generation: int | None,
    strict: bool,
    *,
    active_policy: dict[str, Any] | None,
) -> dict[str, Any]:
    if active_policy and _runner_accepts_active_policy(runner):
        return runner(campaign, max_attempts_per_generation, strict, active_policy)
    return runner(campaign, max_attempts_per_generation, strict)


def _runner_accepts_active_policy(runner: CampaignRunner) -> bool:
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return True
    parameters = list(signature.parameters.values())
    if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in parameters):
        return True
    positional = [
        param
        for param in parameters
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 4


def _run_campaign_subprocess(
    campaign: str,
    max_attempts_per_generation: int | None,
    strict: bool,
    active_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset = CAMPAIGN_PRESETS[campaign]
    if preset is None:
        return {"ok": True, "campaign": campaign, "ran": False}
    max_attempts = max_attempts_per_generation or int(preset["max_attempts_per_generation"])
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "peds_training_campaign.py"),
        "--generations",
        str(preset["generations"]),
        "--runs",
        str(preset["runs"]),
        "--max-attempts-per-generation",
        str(max_attempts),
        "--json",
    ]
    if active_policy:
        command.extend(
            [
                "--active-policy-json",
                json.dumps(_public_active_policy(active_policy), ensure_ascii=True, sort_keys=True),
            ]
        )
    if strict:
        command.append("--strict")
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
    payload: dict[str, Any] | None = None
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError:
            payload = None
    if completed.returncode != 0:
        return {
            "ok": False,
            "campaign": campaign,
            "command": command,
            "returncode": completed.returncode,
            "error": completed.stderr.strip() or "campaign command failed",
            "stdout_preview": completed.stdout[-2000:],
        }
    return {
        "ok": True,
        "campaign": campaign,
        "command": command,
        "returncode": completed.returncode,
        "report_path": (payload or {}).get("report_path"),
        "campaign_id": (payload or {}).get("campaign_id"),
    }


def _compact_campaign_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {key: result.get(key) for key in ("ok", "campaign", "returncode", "report_path", "campaign_id", "error") if key in result}


def _require_objective(objective: str) -> str:
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective is required")
    return objective.strip()


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, **extra}


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PEDS Autopilot and produce useful survivor artifacts.")
    parser.add_argument("--objective", required=True)
    parser.add_argument("--campaign", choices=sorted(CAMPAIGN_PRESETS), default="5x2")
    parser.add_argument("--use-latest", action="store_true")
    parser.add_argument("--activate-best", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--include-details", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=None)
    parser.add_argument("--max-attempts-per-generation", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--artifact-type", choices=["all", "profile", "prompt", "plan", "skill"], default="all")
    parser.add_argument("--active-policy-file", type=Path, default=None)
    parser.add_argument("--active-policy-json", default=None)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
