from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMPAIGNS_ROOT = PROJECT_ROOT / "temp" / "peds_training_campaigns"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = build_latest_campaign_report(args.run_root)
    indent = None if args.json else 2
    print(json.dumps(report, ensure_ascii=False, indent=indent, sort_keys=True))
    return 0


def build_latest_campaign_report(run_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(run_root) if run_root is not None else DEFAULT_CAMPAIGNS_ROOT
    report_path = find_latest_campaign_report(root)
    if report_path is None:
        return {
            "found": False,
            "reason": f"No campaign_report.json found under {root}",
            "run_root": str(root),
        }
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "found": False,
            "reason": f"Could not parse campaign report JSON: {report_path}",
            "run_root": str(root),
            "report_path": str(report_path),
        }
    compact = {
        "found": True,
        "campaign_id": report.get("campaign_id") or report_path.parent.name,
        "campaign_root": report.get("campaign_root") or str(report_path.parent),
        "report_path": str(report_path),
        "runs_requested": report.get("runs_requested"),
        "generations_requested": report.get("generations_requested"),
        "max_attempts_per_generation": report.get("max_attempts_per_generation"),
        "continuation_policy": report.get("continuation_policy"),
        "variants_executed": report.get("variants_executed", []),
        "attempts": report.get("attempts", []),
        "rejected_attempts": report.get("rejected_attempts", []),
        "diagnoses": report.get("diagnoses", []),
        "corrections": report.get("corrections", []),
        "mutations": report.get("mutations", []),
        "winning_attempts": report.get("winning_attempts", []),
        "exhausted_generations": report.get("exhausted_generations", []),
        "lineage": report.get("lineage", []),
        "aggregate": report.get("aggregate", {}),
        "safety": report.get("safety", {}),
        "blockers": report.get("blockers", []),
    }
    return compact


def find_latest_campaign_report(root: Path | str) -> Path | None:
    path = Path(root)
    if path.is_file() and path.name == "campaign_report.json":
        return path
    if (path / "campaign_report.json").exists():
        return path / "campaign_report.json"
    if not path.exists() or not path.is_dir():
        return None
    reports = list(path.glob("*/campaign_report.json"))
    if not reports:
        return None
    return max(reports, key=lambda item: (item.stat().st_mtime, item.parent.name))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the latest PEDS training campaign report.")
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
