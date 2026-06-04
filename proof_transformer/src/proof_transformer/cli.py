from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent.actions import parse_csv_ids
from .agent.orchestrator import ProofTransformerOrchestrator
from .agent.worker import QueueWorker
from .agent.workspace_watcher import WorkspaceWatcher
from .db import get_connection
from .peds.cycle import advance_peds_cycle, run_peds_cycle_until_blocked
from .server import AssistantAPIServer


def _load_desktop_plan(plan_json: str | None, plan_file: str | None) -> dict:
    raw = plan_json
    if plan_file:
        raw = Path(plan_file).read_text(encoding="utf-8")
    if raw is None:
        raise ValueError("Desktop plan requires --plan-json or --plan-file.")
    payload = json.loads(raw)
    if isinstance(payload, list):
        return {"steps": payload}
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        return payload
    raise ValueError("Desktop plan JSON must be a list of steps or an object with a 'steps' list.")


def _load_json_object(raw: str | None, file_path: str | None, *, label: str) -> dict:
    if raw and file_path:
        raise ValueError(f"Use only one of --{label}-json or --{label}-file.")
    if file_path:
        raw = Path(file_path).read_text(encoding="utf-8")
    if raw is None or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid --{label}-json: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON must be an object.")
    return payload


def _run_peds_cycle_cli(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        options = _load_json_object(args.options_json, args.options_file, label="options")
    except ValueError as exc:
        parser.error(str(exc))

    if not args.dry_run and not args.operator:
        parser.error("--operator is required unless --dry-run is supplied.")

    conn = get_connection()
    try:
        if args.dry_run:
            result = advance_peds_cycle(
                conn,
                args.candidate_id,
                operator=args.operator or "dry-run",
                mode="cli",
                auto_promote=bool(args.auto_promote),
                dry_run=True,
                options=options,
            )
        else:
            result = run_peds_cycle_until_blocked(
                conn,
                args.candidate_id,
                operator=args.operator,
                auto_promote=bool(args.auto_promote),
                max_steps=args.max_steps,
                options=options,
            )
        print(json.dumps(result, indent=2))
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proof-transformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init")

    ingest = subparsers.add_parser("ingest")
    ingest.add_argument("path")

    ask = subparsers.add_parser("ask")
    ask.add_argument("question")

    assist = subparsers.add_parser("assist")
    assist.add_argument("question")
    assist.add_argument("--workspace")
    assist.add_argument("--index-workspace", action="store_true")

    chat = subparsers.add_parser("chat")
    chat.add_argument("--workspace")
    chat.add_argument("--index-workspace", action="store_true")

    index_workspace = subparsers.add_parser("index-workspace")
    index_workspace.add_argument("path")
    index_workspace.add_argument("--max-file-bytes", type=int, default=1000000)

    watch_workspace = subparsers.add_parser("watch-workspace")
    watch_workspace.add_argument("path")
    watch_workspace.add_argument("--poll-interval", type=float, default=2.0)
    watch_workspace.add_argument("--max-cycles", type=int)
    watch_workspace.add_argument("--max-file-bytes", type=int, default=1000000)

    learn = subparsers.add_parser("learn")
    learn.add_argument("--type", required=True, dest="event_type")
    learn.add_argument("--text", required=True)

    feedback = subparsers.add_parser("feedback")
    feedback.add_argument("--event-id", required=True)
    feedback.add_argument("--useful", default="")
    feedback.add_argument("--irrelevant", default="")

    plan_repair = subparsers.add_parser("plan-repair")
    plan_repair.add_argument("workspace")
    plan_repair.add_argument("--timeout", type=int)
    plan_repair.add_argument("command_args", nargs="+")

    apply_repair = subparsers.add_parser("apply-repair-plan")
    apply_repair.add_argument("plan_id")
    apply_repair.add_argument("--no-post-commands", action="store_true")

    autonomous_repair = subparsers.add_parser("autonomous-repair")
    autonomous_repair.add_argument("workspace")
    autonomous_repair.add_argument("--timeout", type=int)
    autonomous_repair.add_argument("--max-iterations", type=int, default=1)
    autonomous_repair.add_argument("command_args", nargs="+")

    list_repair_plans = subparsers.add_parser("list-repair-plans")
    list_repair_plans.add_argument("--status")
    list_repair_plans.add_argument("--limit", type=int, default=20)

    run_command = subparsers.add_parser("run-command")
    run_command.add_argument("--cwd")
    run_command.add_argument("--timeout", type=int)
    run_command.add_argument("--no-ingest-output", action="store_true")
    run_command.add_argument("command_args", nargs="+")

    peds_cycle = subparsers.add_parser("peds-cycle")
    peds_cycle.add_argument("--candidate-id", required=True)
    peds_cycle.add_argument("--operator")
    peds_cycle.add_argument("--max-steps", type=int, default=10)
    peds_cycle.add_argument("--auto-promote", action="store_true")
    peds_cycle.add_argument("--dry-run", action="store_true")
    peds_cycle.add_argument("--options-json")
    peds_cycle.add_argument("--options-file")

    subparsers.add_parser("desktop-capabilities")

    desktop_execute = subparsers.add_parser("desktop-execute")
    desktop_group = desktop_execute.add_mutually_exclusive_group(required=True)
    desktop_group.add_argument("--plan-json")
    desktop_group.add_argument("--plan-file")
    desktop_execute.add_argument("--continue-on-error", action="store_true")

    list_executions = subparsers.add_parser("list-executions")
    list_executions.add_argument("--limit", type=int, default=20)

    review_execution = subparsers.add_parser("review-execution")
    review_execution.add_argument("execution_id")
    review_execution.add_argument("question")

    enqueue = subparsers.add_parser("enqueue")
    enqueue.add_argument("task_type")
    enqueue.add_argument("--payload", default="{}")
    enqueue.add_argument("--priority", type=int, default=100)

    run_queue = subparsers.add_parser("run-queue")
    run_queue.add_argument("--limit", type=int)

    list_tasks = subparsers.add_parser("list-tasks")
    list_tasks.add_argument("--status")
    list_tasks.add_argument("--limit", type=int, default=50)

    serve_queue = subparsers.add_parser("serve-queue")
    serve_queue.add_argument("--poll-interval", type=float, default=2.0)
    serve_queue.add_argument("--max-cycles", type=int)
    serve_queue.add_argument("--limit-per-cycle", type=int)

    serve_api = subparsers.add_parser("serve-api")
    serve_api.add_argument("--host", default="127.0.0.1")
    serve_api.add_argument("--port", type=int, default=8765)

    subparsers.add_parser("verify-ledger")
    subparsers.add_parser("status")
    subparsers.add_parser("health")

    export = subparsers.add_parser("export-audit")
    export.add_argument("path")

    subparsers.add_parser("demo")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "peds-cycle":
        return _run_peds_cycle_cli(args, parser)

    orchestrator = ProofTransformerOrchestrator()

    if args.command == "init":
        print(json.dumps(orchestrator.init(), indent=2))
        return 0
    if args.command == "ingest":
        print(json.dumps(orchestrator.ingest(args.path), indent=2))
        return 0
    if args.command == "ask":
        print(json.dumps(orchestrator.ask(args.question), indent=2))
        return 0
    if args.command == "assist":
        print(
            json.dumps(
                orchestrator.assist(
                    args.question,
                    workspace_path=args.workspace,
                    index_workspace=bool(args.index_workspace),
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "chat":
        if args.workspace and args.index_workspace:
            print(json.dumps(orchestrator.index_workspace(args.workspace), indent=2))
        print("Proof Transformer chat. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                query = input("you> ").strip()
            except EOFError:
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit"}:
                break
            result = orchestrator.assist(query, workspace_path=args.workspace, index_workspace=False)
            print(result["answer"])
            print(f"retrieval_event_id={result['retrieval_event_id']}")
        return 0
    if args.command == "index-workspace":
        print(
            json.dumps(
                orchestrator.index_workspace(args.path, max_file_bytes=args.max_file_bytes),
                indent=2,
            )
        )
        return 0
    if args.command == "watch-workspace":
        watcher = WorkspaceWatcher(
            orchestrator,
            args.path,
            poll_interval=args.poll_interval,
            max_file_bytes=args.max_file_bytes,
        )
        print(json.dumps(watcher.serve(max_cycles=args.max_cycles), indent=2))
        return 0
    if args.command == "learn":
        print(json.dumps(orchestrator.learn(args.text, args.event_type), indent=2))
        return 0
    if args.command == "feedback":
        print(
            json.dumps(
                orchestrator.feedback(
                    retrieval_event_id=args.event_id,
                    useful_chunk_ids=parse_csv_ids(args.useful),
                    irrelevant_chunk_ids=parse_csv_ids(args.irrelevant),
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "plan-repair":
        command = [part for part in args.command_args if part]
        print(
            json.dumps(
                orchestrator.plan_repair(args.workspace, command, timeout=args.timeout),
                indent=2,
            )
        )
        return 0
    if args.command == "apply-repair-plan":
        print(
            json.dumps(
                orchestrator.apply_repair_plan(args.plan_id, run_post_commands=not bool(args.no_post_commands)),
                indent=2,
            )
        )
        return 0
    if args.command == "autonomous-repair":
        command = [part for part in args.command_args if part]
        print(
            json.dumps(
                orchestrator.autonomous_repair(
                    args.workspace,
                    command,
                    max_iterations=args.max_iterations,
                    timeout=args.timeout,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "list-repair-plans":
        print(json.dumps(orchestrator.list_repair_plans(status=args.status, limit=args.limit), indent=2))
        return 0
    if args.command == "run-command":
        command = [part for part in args.command_args if part]
        print(
            json.dumps(
                orchestrator.run_command(
                    command=command,
                    cwd=args.cwd,
                    timeout=args.timeout,
                    ingest_output=not bool(args.no_ingest_output),
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "desktop-capabilities":
        print(json.dumps(orchestrator.desktop_capabilities(), indent=2))
        return 0
    if args.command == "desktop-execute":
        payload = _load_desktop_plan(args.plan_json, args.plan_file)
        print(
            json.dumps(
                orchestrator.execute_desktop_plan(
                    payload["steps"],
                    continue_on_error=bool(args.continue_on_error or payload.get("continue_on_error", False)),
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "list-executions":
        print(json.dumps(orchestrator.list_execution_runs(limit=args.limit), indent=2))
        return 0
    if args.command == "review-execution":
        print(json.dumps(orchestrator.review_execution(args.execution_id, args.question), indent=2))
        return 0
    if args.command == "enqueue":
        print(
            json.dumps(
                orchestrator.enqueue_task(
                    args.task_type,
                    payload=json.loads(args.payload),
                    priority=args.priority,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "run-queue":
        print(json.dumps(orchestrator.run_queue(limit=args.limit), indent=2))
        return 0
    if args.command == "list-tasks":
        print(json.dumps(orchestrator.list_tasks(status=args.status, limit=args.limit), indent=2))
        return 0
    if args.command == "serve-queue":
        worker = QueueWorker(orchestrator, poll_interval=args.poll_interval)
        print(json.dumps(worker.serve(max_cycles=args.max_cycles, limit_per_cycle=args.limit_per_cycle), indent=2))
        return 0
    if args.command == "serve-api":
        server = AssistantAPIServer(host=args.host, port=args.port, orchestrator=orchestrator)
        host, port = server.address
        print(json.dumps({"host": host, "port": port, "status": "serving"}, indent=2))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return 0
    if args.command == "verify-ledger":
        print(json.dumps(orchestrator.verify_ledger(), indent=2))
        return 0
    if args.command == "status":
        print(json.dumps(orchestrator.status(), indent=2))
        return 0
    if args.command == "health":
        print(json.dumps(orchestrator.health(), indent=2))
        return 0
    if args.command == "export-audit":
        output = orchestrator.export_audit(Path(args.path))
        print(json.dumps({"path": output}, indent=2))
        return 0
    if args.command == "demo":
        print(json.dumps(orchestrator.demo(), indent=2))
        return 0
    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
