from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .agent.orchestrator import ProofTransformerOrchestrator


class AssistantAPIServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, orchestrator: ProofTransformerOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or ProofTransformerOrchestrator()
        self._server = _ThreadedAssistantHTTPServer((host, port), AssistantRequestHandler, self.orchestrator)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def start_in_thread(self) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()
        return self._thread

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class _ThreadedAssistantHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, orchestrator: ProofTransformerOrchestrator):
        super().__init__(server_address, RequestHandlerClass)
        self.orchestrator = orchestrator


class AssistantRequestHandler(BaseHTTPRequestHandler):
    server: _ThreadedAssistantHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(self.server.orchestrator.health())
                return
            if parsed.path == "/status":
                self._send_json(self.server.orchestrator.status())
                return
            if parsed.path == "/tasks":
                params = parse_qs(parsed.query)
                status = params.get("status", [None])[0]
                limit = int(params.get("limit", [50])[0])
                self._send_json(self.server.orchestrator.list_tasks(status=status, limit=limit))
                return
            if parsed.path == "/executions":
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", [20])[0])
                self._send_json(self.server.orchestrator.list_execution_runs(limit=limit))
                return
            if parsed.path == "/repair-plans":
                params = parse_qs(parsed.query)
                status = params.get("status", [None])[0]
                limit = int(params.get("limit", [20])[0])
                self._send_json(self.server.orchestrator.list_repair_plans(status=status, limit=limit))
                return
            if parsed.path == "/desktop/capabilities":
                self._send_json(self.server.orchestrator.desktop_capabilities())
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/ask":
                self._send_json(self.server.orchestrator.ask(payload["query"]))
                return
            if parsed.path == "/assist":
                self._send_json(
                    self.server.orchestrator.assist(
                        payload["query"],
                        workspace_path=payload.get("workspace_path"),
                        index_workspace=bool(payload.get("index_workspace", False)),
                    )
                )
                return
            if parsed.path == "/run-command":
                self._send_json(
                    self.server.orchestrator.run_command(
                        payload["command"],
                        cwd=payload.get("cwd"),
                        timeout=payload.get("timeout"),
                        ingest_output=bool(payload.get("ingest_output", True)),
                    )
                )
                return
            if parsed.path == "/desktop/execute":
                self._send_json(
                    self.server.orchestrator.execute_desktop_plan(
                        payload["steps"],
                        continue_on_error=bool(payload.get("continue_on_error", False)),
                    )
                )
                return
            if parsed.path == "/enqueue":
                self._send_json(
                    self.server.orchestrator.enqueue_task(
                        payload["task_type"],
                        payload=payload.get("payload"),
                        priority=int(payload.get("priority", 100)),
                    )
                )
                return
            if parsed.path == "/run-queue":
                self._send_json(self.server.orchestrator.run_queue(limit=payload.get("limit")))
                return
            if parsed.path == "/plan-repair":
                self._send_json(
                    self.server.orchestrator.plan_repair(
                        payload["workspace_path"],
                        payload["command"],
                        execution_id=payload.get("execution_id"),
                        timeout=payload.get("timeout"),
                    )
                )
                return
            if parsed.path == "/apply-repair-plan":
                self._send_json(
                    self.server.orchestrator.apply_repair_plan(
                        payload["plan_id"],
                        run_post_commands=bool(payload.get("run_post_commands", True)),
                    )
                )
                return
            if parsed.path == "/autonomous-repair":
                self._send_json(
                    self.server.orchestrator.autonomous_repair(
                        payload["workspace_path"],
                        payload["command"],
                        max_iterations=int(payload.get("max_iterations", 1)),
                        timeout=payload.get("timeout"),
                    )
                )
                return
            if parsed.path == "/review-execution":
                self._send_json(
                    self.server.orchestrator.review_execution(
                        payload["execution_id"],
                        payload["question"],
                    )
                )
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except KeyError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Missing required field: {exc}")
        except json.JSONDecodeError as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise json.JSONDecodeError("Body must decode to a JSON object", raw.decode("utf-8"), 0)
        return payload

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message, "status": int(status)}, status=status)
