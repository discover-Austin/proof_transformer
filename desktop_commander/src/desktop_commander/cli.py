from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import DesktopCommanderAPIServer
from .executor import DesktopExecutor
from .gui import launch_gui
from .history import HistoryStore
from .planner import TaskPlanner


def _load_plan(plan_json: str | None, plan_file: str | None) -> dict:
    raw = plan_json
    if plan_file:
        raw = Path(plan_file).read_text(encoding="utf-8")
    if raw is None:
        raise ValueError("Provide --plan-json or --plan-file.")
    payload = json.loads(raw)
    if isinstance(payload, list):
        return {"steps": payload}
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        return payload
    raise ValueError("Plan JSON must be a list of steps or an object with a steps list.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="desktop-commander")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ui")
    subparsers.add_parser("capabilities")

    draft = subparsers.add_parser("draft-plan")
    draft.add_argument("task_text")

    execute = subparsers.add_parser("execute")
    group = execute.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-json")
    group.add_argument("--plan-file")
    execute.add_argument("--continue-on-error", action="store_true")
    execute.add_argument("--task-text")

    history = subparsers.add_parser("history")
    history.add_argument("--limit", type=int, default=20)

    subparsers.add_parser("presets")

    serve_api = subparsers.add_parser("serve-api")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8877)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    executor = DesktopExecutor()
    planner = TaskPlanner()
    history = HistoryStore()

    if args.command == "ui":
        launch_gui(executor=executor, planner=planner, history=history)
        return 0
    if args.command == "capabilities":
        print(json.dumps(executor.capabilities(), indent=2))
        return 0
    if args.command == "draft-plan":
        print(json.dumps(planner.draft_plan(args.task_text), indent=2))
        return 0
    if args.command == "execute":
        plan = _load_plan(args.plan_json, args.plan_file)
        result = executor.execute_plan(plan, continue_on_error=bool(args.continue_on_error))
        history.append_run(task_text=args.task_text or plan.get("task_text"), plan=plan, result=result)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "history":
        print(json.dumps(history.list_runs(limit=args.limit), indent=2))
        return 0
    if args.command == "presets":
        print(json.dumps(history.list_presets(), indent=2))
        return 0
    if args.command == "serve-api":
        server = DesktopCommanderAPIServer(
            host=args.host,
            port=args.port,
            executor=executor,
            planner=planner,
            history=history,
        )
        host, port = server.address
        print(json.dumps({"status": "serving", "host": host, "port": port}, indent=2))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
