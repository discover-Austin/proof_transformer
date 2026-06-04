from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from .executor import DesktopExecutor
from .history import HistoryStore
from .planner import TaskPlanner


class DesktopCommanderAPIServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8877,
        executor: DesktopExecutor | None = None,
        planner: TaskPlanner | None = None,
        history: HistoryStore | None = None,
    ) -> None:
        self.executor = executor or DesktopExecutor()
        self.planner = planner or TaskPlanner()
        self.history = history or HistoryStore()
        self._server = _ThreadedHTTPServer((host, port), DesktopCommanderRequestHandler, self.executor, self.planner, self.history)
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


class _ThreadedHTTPServer(HTTPServer):
    def __init__(self, server_address, handler_class, executor: DesktopExecutor, planner: TaskPlanner, history: HistoryStore):
        super().__init__(server_address, handler_class)
        self.executor = executor
        self.planner = planner
        self.history = history


class DesktopCommanderRequestHandler(BaseHTTPRequestHandler):
    server: _ThreadedHTTPServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "capabilities": self.server.executor.capabilities(),
                        "history_count": len(self.server.history.list_runs(limit=200)),
                        "preset_count": len(self.server.history.list_presets()),
                    }
                )
                return
            if parsed.path == "/capabilities":
                self._send_json(self.server.executor.capabilities())
                return
            if parsed.path == "/history":
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", [50])[0])
                self._send_json(self.server.history.list_runs(limit=limit))
                return
            if parsed.path == "/presets":
                self._send_json(self.server.history.list_presets())
                return
            self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {parsed.path}")
        except Exception as exc:  # pragma: no cover
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/draft-plan":
                self._send_json(self.server.planner.draft_plan(payload["task_text"]))
                return
            if parsed.path == "/execute":
                plan = payload if "steps" in payload else {"steps": payload["plan"]}
                result = self.server.executor.execute_plan(
                    plan,
                    continue_on_error=bool(payload.get("continue_on_error", False)),
                )
                self.server.history.append_run(
                    task_text=payload.get("task_text"),
                    plan=plan,
                    result=result,
                )
                self._send_json(result)
                return
            if parsed.path == "/save-preset":
                self._send_json(
                    self.server.history.save_preset(
                        name=payload["name"],
                        task_text=payload.get("task_text"),
                        plan={"steps": payload["steps"]},
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
