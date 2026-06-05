from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "temp" / "peds_trained_local_generations"
REPORT_FILENAMES = (
    "peds_generation_report.json",
    "generation_report.json",
    "run_report.json",
    "report.json",
)
COUNT_TABLES = (
    "candidates",
    "episodes",
    "replay_jobs",
    "replay_results",
    "promotion_decisions",
    "peds_training_runs",
    "adapter_registry",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_latest_report(args.run_root)
    indent = None if args.json else 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


def build_latest_report(run_root: Path | str | None = None) -> dict[str, Any]:
    requested_root = Path(run_root) if run_root is not None else DEFAULT_RUNS_ROOT
    latest_run = find_latest_run(requested_root)
    if latest_run is None:
        return {
            "found": False,
            "reason": f"No PEDS trained-local-generations runs found under {requested_root}",
            "run_root": str(requested_root),
        }

    saved_report = _load_saved_report(latest_run)
    db_counts = _read_db_counts(latest_run / "memory" / "brain.sqlite3")
    report = _compact_from_saved_report(saved_report, latest_run)
    if saved_report is None:
        report.update(_compact_from_run_artifacts(latest_run))
    report["db_counts"] = db_counts
    report["passed"] = _report_passed(report)
    if report["passed"]:
        report["blocker"] = None
    elif not report.get("blocker"):
        report["blocker"] = _infer_blocker(report)
    return report


def find_latest_run(run_root: Path | str) -> Path | None:
    root = Path(run_root)
    if _looks_like_run_root(root):
        return root
    if not root.exists() or not root.is_dir():
        return None
    candidates = [path for path in root.iterdir() if path.is_dir() and _looks_like_run_root(path)]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _looks_like_run_root(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if any((path / name).exists() for name in REPORT_FILENAMES):
        return True
    return (path / "memory" / "brain.sqlite3").exists() or (path / "experiments").is_dir()


def _load_saved_report(run_root: Path) -> dict[str, Any] | None:
    for filename in REPORT_FILENAMES:
        path = run_root / filename
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"blocker": f"Could not parse saved report JSON: {path}", "report_path": str(path)}
    return None


def _compact_from_saved_report(saved_report: dict[str, Any] | None, run_root: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "found": True,
        "run_id": run_root.name,
        "run_root": str(run_root),
        "generations_completed": None,
        "final_candidate_id": None,
        "replay_deltas": [],
        "replay_decisions": [],
        "adapter_statuses": [],
        "training_trained": [],
        "adapter_loaded": [],
        "network_attempts": [],
        "model_modules_loaded": [],
        "fake_weights_written": None,
        "latest_artifact_paths": [],
        "blocker": None,
    }
    if saved_report is None:
        base["blocker"] = "saved_generation_report_missing; reconstructed from artifacts/database"
        return base
    if "blocker" in saved_report and "generation_reports" not in saved_report:
        base["blocker"] = saved_report["blocker"]
        base["latest_artifact_paths"] = [saved_report.get("report_path")] if saved_report.get("report_path") else []
        return base

    generation_reports = saved_report.get("generation_reports") or []
    base.update(
        {
            "run_id": str(saved_report.get("run_id") or run_root.name),
            "run_root": str(saved_report.get("brain_root") or run_root),
            "generations_completed": saved_report.get("generations_completed"),
            "final_candidate_id": saved_report.get("final_candidate_id"),
            "network_attempts": saved_report.get("network_attempts", []),
            "model_modules_loaded": saved_report.get("model_modules_loaded", []),
            "fake_weights_written": saved_report.get("fake_weights_written"),
            "report_path": saved_report.get("report_path"),
        }
    )
    artifact_paths: list[str] = []
    for item in generation_reports:
        generation = item.get("generation")
        probe = item.get("replay_probe") or {}
        base["replay_deltas"].append({"generation": generation, "mean_delta": probe.get("replay_mean_delta")})
        base["replay_decisions"].append({"generation": generation, "decision": probe.get("replay_gate_decision")})
        base["adapter_statuses"].append(
            {
                "generation": generation,
                "name": item.get("adapter_registry_name"),
                "status": item.get("adapter_registry_status"),
            }
        )
        base["training_trained"].append({"generation": generation, "trained": item.get("training_trained")})
        if "adapter_loaded" in probe:
            base["adapter_loaded"].append({"generation": generation, "loaded": probe.get("adapter_loaded")})
        for key in ("manifest_path", "output_artifact"):
            if item.get(key):
                artifact_paths.append(str(item[key]))
    if saved_report.get("report_path"):
        artifact_paths.append(str(saved_report["report_path"]))
    base["latest_artifact_paths"] = artifact_paths
    return base


def _compact_from_run_artifacts(run_root: Path) -> dict[str, Any]:
    experiments_root = run_root / "experiments"
    generation_dirs = sorted(experiments_root.glob("trained_generation_*")) if experiments_root.exists() else []
    adapter_statuses: list[dict[str, Any]] = []
    training_trained: list[dict[str, Any]] = []
    artifact_paths: list[str] = []
    for index, generation_dir in enumerate(generation_dirs, start=1):
        for manifest_path in generation_dir.glob("artifacts/*_adapter_artifact_manifest.json"):
            artifact_paths.append(str(manifest_path))
            artifact = _read_json_file(manifest_path)
            adapter_statuses.append(
                {
                    "generation": index,
                    "name": artifact.get("adapter_name") or artifact.get("name"),
                    "status": "trained" if artifact.get("trained") else artifact.get("status"),
                }
            )
            training_trained.append({"generation": index, "trained": artifact.get("trained")})
        distillation_manifest = generation_dir / "distillation_manifest.json"
        if distillation_manifest.exists():
            artifact_paths.append(str(distillation_manifest))
    return {
        "generations_completed": len(generation_dirs) or None,
        "adapter_statuses": adapter_statuses,
        "training_trained": training_trained,
        "latest_artifact_paths": artifact_paths,
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _read_db_counts(db_path: Path) -> dict[str, int]:
    if not db_path.exists():
        return {}
    counts: dict[str, int] = {}
    conn = sqlite3.connect(str(db_path))
    try:
        for table in COUNT_TABLES:
            try:
                row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            except sqlite3.Error:
                continue
            counts[table] = int(row[0]) if row else 0
    finally:
        conn.close()
    return counts


def _report_passed(report: dict[str, Any]) -> bool:
    if not report.get("found"):
        return False
    if report.get("blocker") and report.get("blocker") != "saved_generation_report_missing; reconstructed from artifacts/database":
        return False
    if report.get("fake_weights_written") is True:
        return False
    if report.get("network_attempts"):
        return False
    if report.get("model_modules_loaded"):
        return False
    if report.get("replay_decisions"):
        if any(item.get("decision") != "replay_winner" for item in report["replay_decisions"]):
            return False
    if report.get("adapter_statuses"):
        if any(item.get("status") != "trained" for item in report["adapter_statuses"]):
            return False
    if report.get("training_trained"):
        if any(item.get("trained") is not True for item in report["training_trained"]):
            return False
    if report.get("adapter_loaded"):
        if any(item.get("loaded") is not True for item in report["adapter_loaded"]):
            return False
    return bool(report.get("generations_completed") or report.get("db_counts"))


def _infer_blocker(report: dict[str, Any]) -> str | None:
    if not report.get("found"):
        return report.get("reason")
    if report.get("fake_weights_written") is True:
        return "fake_weights_written"
    if report.get("network_attempts"):
        return "network_attempts_present"
    if report.get("model_modules_loaded"):
        return "model_modules_loaded"
    for item in report.get("replay_decisions", []):
        if item.get("decision") != "replay_winner":
            return f"generation_{item.get('generation')}_not_replay_winner"
    for item in report.get("adapter_statuses", []):
        if item.get("status") != "trained":
            return f"generation_{item.get('generation')}_adapter_not_trained"
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the latest PEDS trained-local-generations report.")
    parser.add_argument("--run-root", type=Path, default=None, help="Run root or parent directory to inspect.")
    parser.add_argument("--json", action="store_true", help="Print compact JSON.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
