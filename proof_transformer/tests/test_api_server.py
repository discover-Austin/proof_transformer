from __future__ import annotations

import json
from urllib.request import Request, urlopen

from common import BrainTestCase
from proof_transformer.agent.orchestrator import ProofTransformerOrchestrator
from proof_transformer.server import AssistantAPIServer


class APIServerTests(BrainTestCase):
    def test_api_server_exposes_health_and_ask_endpoints(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)
        server = AssistantAPIServer(host="127.0.0.1", port=0, orchestrator=orchestrator)
        server.start_in_thread()
        host, port = server.address
        try:
            with urlopen(f"http://{host}:{port}/health", timeout=10) as response:
                health = json.loads(response.read().decode("utf-8"))
            self.assertTrue(health["db_exists"])

            payload = json.dumps({"query": "What is the first mutable learning target?"}).encode("utf-8")
            request = Request(
                f"http://{host}:{port}/ask",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                answer = json.loads(response.read().decode("utf-8"))
            self.assertIn("retrieval_event_id", answer)
            self.assertIn("answer", answer)
        finally:
            server.shutdown()

    def test_api_server_exposes_desktop_capabilities_and_execution(self) -> None:
        orchestrator = ProofTransformerOrchestrator(self.conn)

        class StubDesktopExecutor:
            def capabilities(self) -> dict:
                return {"actions": ["search_web", "type_text"], "platform": "test"}

            def execute_plan(self, steps: list[dict], *, continue_on_error: bool = False) -> dict:
                return {
                    "status": "completed",
                    "steps_executed": len(steps),
                    "continue_on_error": continue_on_error,
                    "results": [{"step_index": 0, "action": steps[0]["action"], "status": "completed", "result": {"ok": True}}],
                }

        orchestrator.desktop_executor = StubDesktopExecutor()
        server = AssistantAPIServer(host="127.0.0.1", port=0, orchestrator=orchestrator)
        server.start_in_thread()
        host, port = server.address
        try:
            with urlopen(f"http://{host}:{port}/desktop/capabilities", timeout=10) as response:
                capabilities = json.loads(response.read().decode("utf-8"))
            self.assertIn("search_web", capabilities["actions"])

            payload = json.dumps({"steps": [{"action": "search_web", "query": "desktop automation"}]}).encode("utf-8")
            request = Request(
                f"http://{host}:{port}/desktop/execute",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=10) as response:
                execution = json.loads(response.read().decode("utf-8"))
            self.assertEqual(execution["status"], "completed")
            self.assertEqual(execution["steps_executed"], 1)
        finally:
            server.shutdown()
