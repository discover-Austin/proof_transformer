from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from ..config import load_settings
from ..db import get_connection
from ..ids import new_id
from ..ledger.blocks import summarize_validator_results
from ..ledger.candidates import CandidateFactory
from ..ledger.ledger import LearningLedger
from ..ledger.models import TrainingCandidate
from ..ledger.policy import PolicyStore
from ..learning.adapter_registry import AdapterRegistry
from ..learning.ranker import RankerStore, RankerWeights
from ..learning.ranker_trainer import RankerTrainer
from ..memory.context_compiler import ContextCompiler
from ..memory.documents import DocumentStore
from ..memory.graph import GraphMemory
from ..memory.workspace import WorkspaceIndexer
from ..model.local_llm import LocalLLM
from ..paths import get_brain_paths
from ..timeutils import utc_now
from ..validators.consensus import ConsensusEngine
from .command_runner import CommandRunner
from .desktop_executor import DesktopTaskExecutor
from .patching import PatchApplier
from .repair_planner import RepairPlanner
from .session import SessionLog
from .task_queue import TaskQueue


class ProofTransformerOrchestrator:
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.settings = load_settings()
        self.paths = get_brain_paths()
        self.conn = conn or get_connection()
        self.document_store = DocumentStore(self.conn)
        self.graph = GraphMemory(self.conn)
        self.context_compiler = ContextCompiler(self.conn)
        self.workspace_indexer = WorkspaceIndexer(self.document_store, self.graph)
        self.ledger = LearningLedger(self.conn)
        self.policy_store = PolicyStore(self.conn)
        self.candidate_factory = CandidateFactory(self.conn)
        self.consensus = ConsensusEngine(self.conn)
        self.ranker_store = RankerStore(self.conn)
        self.ranker_trainer = RankerTrainer(self.conn)
        self.adapter_registry = AdapterRegistry(self.conn)
        self.task_queue = TaskQueue(self.conn)
        self.command_runner = CommandRunner(self.conn)
        self.desktop_executor = DesktopTaskExecutor(self.command_runner)
        self.patch_applier = PatchApplier()
        self.repair_planner = RepairPlanner()
        self.llm = LocalLLM()
        self.session_log = SessionLog()

    def init(self) -> dict:
        self.paths.ensure()
        policy_version, policy = self.policy_store.ensure_default()
        ranker_version, ranker = self.ranker_store.ensure_default()
        genesis = self.ledger.create_genesis_if_missing()
        result = {
            "brain_root": str(self.paths.root),
            "db": str(self.paths.db),
            "policy_version": policy_version,
            "policy": policy.as_dict(),
            "ranker_version": ranker_version,
            "ranker": ranker.as_dict(),
            "genesis_block": genesis,
        }
        self.session_log.append("init", result)
        return result

    def ingest(self, path: str | Path) -> dict:
        result = self.document_store.ingest_file(path)
        self.session_log.append("ingest", {"path": str(path), "document_id": result["id"], "duplicate": result["duplicate"]})
        return result

    def ask(
        self,
        query: str,
        *,
        source_uri_prefix: str | None = None,
        include_graph: bool = True,
        extra_context_text: str | None = None,
    ) -> dict:
        packet = self.context_compiler.compile(
            query,
            source_uri_prefix=source_uri_prefix,
            include_graph=include_graph,
        )
        full_context_text = packet.context_text if not extra_context_text else f"{extra_context_text}\n\n{packet.context_text}"
        prompt = (
            f"CONTEXT:\n{full_context_text}\n\n"
            f"QUESTION: {query}\n\n"
            "INSTRUCTIONS:\nAnswer with grounded local evidence. State if local fallback is active."
        )
        answer = self.llm.generate(prompt, system=self.settings.llm_system_prompt)
        self.context_compiler.mark_used_in_answer(packet.retrieval_event_id, used=True)
        result = {
            "answer": answer,
            "context_text": full_context_text,
            "retrieval_event_id": packet.retrieval_event_id,
            "included_chunks": packet.included_chunks,
            "included_nodes": packet.included_nodes,
            "included_edges": packet.included_edges,
            "token_estimate": packet.token_estimate,
            "backend": self.llm.backend_name,
        }
        self.session_log.append("ask", {"query": query, **result})
        return result

    def assist(self, query: str, workspace_path: str | Path | None = None, index_workspace: bool = False) -> dict:
        workspace_result = None
        if workspace_path and index_workspace:
            workspace_result = self.index_workspace(workspace_path)
        workspace_prefix = str(Path(workspace_path)) if workspace_path else None
        extra_context_text = None
        symbol_hits: list[dict] = []
        if workspace_prefix:
            symbol_hits = self.graph.search_nodes_keyword(query, limit=6, source_uri_prefix=workspace_prefix)
            if symbol_hits:
                lines = ["Workspace symbol hits:"]
                for node in symbol_hits:
                    lines.append(f"- {node['name']} ({node['kind']}): {node.get('summary') or node['content']}")
                extra_context_text = "\n".join(lines)
        result = self.ask(
            query,
            source_uri_prefix=workspace_prefix,
            include_graph=True,
            extra_context_text=extra_context_text,
        )
        direct_answer = self._direct_workspace_answer(query, symbol_hits)
        if direct_answer and result["backend"] == "fallback":
            result["answer"] = direct_answer
        if workspace_result is not None:
            result["workspace_index"] = workspace_result
        self.session_log.append(
            "assist",
            {"query": query, "workspace_path": str(workspace_path) if workspace_path else None, **result},
        )
        return result

    def index_workspace(
        self,
        path: str | Path,
        *,
        max_file_bytes: int = 1_000_000,
        extensions: set[str] | None = None,
    ) -> dict:
        result = self.workspace_indexer.index_path(path, max_file_bytes=max_file_bytes, extensions=extensions)
        self.session_log.append("index_workspace", result)
        return result

    def learn(self, source_text: str, event_type: str, context: dict | None = None) -> dict:
        self.init()
        event_context = context or {}
        learning_event = self.candidate_factory.create_learning_event(event_type, source_text, event_context)
        candidate = self.candidate_factory.build_candidate(learning_event)
        if candidate is None:
            result = {
                "status": "ignored",
                "learning_event_id": learning_event["id"],
                "reason": "Event did not meet candidate-worthiness thresholds.",
            }
            self.session_log.append("learn", result)
            return result
        self.candidate_factory.save_candidate(candidate)
        decision = self.consensus.evaluate(candidate, event_context)
        self.consensus.persist_results(decision.results)

        apply_result: dict = {"memory_diff": {"applied": False, "proposed_change": candidate.proposed_change}}
        if decision.status == "accepted":
            apply_result = self._apply_candidate(candidate)
        self.candidate_factory.update_candidate_status(candidate.id, decision.status)

        block = self.ledger.append_block(
            candidate_id=candidate.id,
            status=decision.status,
            memory_diff=apply_result["memory_diff"],
            validator_summary=summarize_validator_results(decision.results),
            consensus_score=decision.consensus_score,
            policy_version=apply_result.get("policy_version"),
            adapter_version=apply_result.get("adapter_version"),
            ranker_version=apply_result.get("ranker_version"),
        )
        result = {
            "status": decision.status,
            "consensus_score": decision.consensus_score,
            "learning_event_id": learning_event["id"],
            "candidate_id": candidate.id,
            "block_id": block["id"],
            "memory_diff": apply_result["memory_diff"],
            "validators": [item.as_record() for item in decision.results],
        }
        self.session_log.append("learn", result)
        return result

    def feedback(
        self,
        retrieval_event_id: str,
        useful_chunk_ids: list[str],
        irrelevant_chunk_ids: list[str],
    ) -> dict:
        source_text, proposed_change = self.ranker_trainer.build_feedback_change(
            retrieval_event_id,
            useful_chunk_ids,
            irrelevant_chunk_ids,
        )
        feedback_count = max(len(useful_chunk_ids) + len(irrelevant_chunk_ids), 1)
        feedback_score = (len(useful_chunk_ids) - len(irrelevant_chunk_ids)) / feedback_count
        with self.conn:
            self.conn.execute(
                "UPDATE retrieval_events SET feedback_score = ? WHERE id = ?",
                (feedback_score, retrieval_event_id),
            )
        result = self.learn(
            source_text=source_text,
            event_type="retrieval_feedback",
            context={
                "proposed_change": proposed_change,
                "useful_chunk_ids": useful_chunk_ids,
                "irrelevant_chunk_ids": irrelevant_chunk_ids,
                "sources": [retrieval_event_id],
            },
        )
        result["retrieval_event_id"] = retrieval_event_id
        result["feedback_score"] = feedback_score
        self.session_log.append("feedback", result)
        return result

    def enqueue_task(self, task_type: str, payload: dict | None = None, priority: int = 100) -> dict:
        result = self.task_queue.enqueue(task_type, payload=payload, priority=priority)
        self.session_log.append("enqueue_task", result)
        return result

    def list_tasks(self, status: str | None = None, limit: int = 50) -> list[dict]:
        return self.task_queue.list_tasks(status=status, limit=limit)

    def run_queue(self, limit: int | None = None) -> dict:
        result = self.task_queue.run_pending(self, limit=limit)
        self.session_log.append("run_queue", result)
        return result

    def run_command(
        self,
        command: list[str] | str,
        *,
        cwd: str | Path | None = None,
        timeout: int | None = None,
        ingest_output: bool = True,
    ) -> dict:
        result = self.command_runner.run(
            command,
            cwd=cwd,
            timeout=timeout,
            ingest_output=ingest_output,
        )
        self.session_log.append("run_command", result)
        return result

    def list_execution_runs(self, limit: int = 50) -> list[dict]:
        return self.command_runner.list_runs(limit=limit)

    def desktop_capabilities(self) -> dict:
        result = self.desktop_executor.capabilities()
        self.session_log.append("desktop_capabilities", result)
        return result

    def execute_desktop_plan(self, steps: list[dict], *, continue_on_error: bool = False) -> dict:
        result = self.desktop_executor.execute_plan(steps, continue_on_error=continue_on_error)
        self.session_log.append(
            "desktop_execute",
            {
                "continue_on_error": continue_on_error,
                "steps": steps,
                "result": result,
            },
        )
        return result

    def plan_repair(
        self,
        workspace_path: str | Path,
        command: list[str],
        *,
        execution_id: str | None = None,
        timeout: int | None = None,
    ) -> dict:
        workspace_root = Path(workspace_path).resolve()
        self.index_workspace(workspace_root)
        execution = self.command_runner.get_run(execution_id) if execution_id else None
        if execution is None:
            execution = self.run_command(command, cwd=workspace_root, timeout=timeout)
        analysis_text, plan = self.repair_planner.build_plan(
            workspace_root=workspace_root,
            command=command,
            execution=execution,
            llm=self.llm,
        )
        now = utc_now()
        plan_id = new_id("repair")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO repair_plans(
                    id, workspace_root, command_json, execution_id, analysis_text,
                    plan_json, status, created_at, updated_at, applied_at, result_execution_id
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                (
                    plan_id,
                    str(workspace_root),
                    json.dumps(command),
                    execution["id"],
                    analysis_text,
                    json.dumps(plan),
                    "planned",
                    now,
                    now,
                ),
            )
        result = self.get_repair_plan(plan_id)
        self.session_log.append("plan_repair", result)
        return result

    def get_repair_plan(self, plan_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM repair_plans WHERE id = ?", (plan_id,)).fetchone()
        return self._decode_repair_plan(row) if row else None

    def list_repair_plans(self, status: str | None = None, limit: int = 20) -> list[dict]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM repair_plans WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, int(limit)),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM repair_plans ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [self._decode_repair_plan(row) for row in rows]

    def apply_repair_plan(self, plan_id: str, *, run_post_commands: bool = True) -> dict:
        plan_record = self.get_repair_plan(plan_id)
        if not plan_record:
            raise ValueError(f"Unknown repair plan: {plan_id}")
        plan = plan_record["plan"]
        workspace_root = Path(plan_record["workspace_root"]).resolve()
        patch_result = self.patch_applier.apply_plan(workspace_root, plan)
        result_execution = None
        if patch_result["applied"]:
            self.index_workspace(workspace_root)
            if run_post_commands and plan.get("post_commands"):
                first = list(plan["post_commands"][0])
                result_execution = self.run_command(first, cwd=workspace_root)
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "UPDATE repair_plans SET status = ?, updated_at = ?, applied_at = ?, result_execution_id = ? WHERE id = ?",
                (
                    "applied" if patch_result["applied"] else "review_only",
                    now,
                    now if patch_result["applied"] else None,
                    result_execution["id"] if result_execution else None,
                    plan_id,
                ),
            )
        result = self.get_repair_plan(plan_id)
        result["patch_result"] = patch_result
        result["result_execution"] = result_execution
        self.session_log.append("apply_repair_plan", result)
        return result

    def autonomous_repair(
        self,
        workspace_path: str | Path,
        command: list[str],
        *,
        max_iterations: int = 1,
        timeout: int | None = None,
    ) -> dict:
        workspace_root = Path(workspace_path).resolve()
        history: list[dict] = []
        last_execution = self.run_command(command, cwd=workspace_root, timeout=timeout)
        history.append({"stage": "initial_execution", "execution": last_execution})
        if last_execution["status"] == "completed" and int(last_execution.get("exit_code") or 0) == 0:
            return {
                "workspace_root": str(workspace_root),
                "command": command,
                "status": "already_passing",
                "history": history,
            }
        for _ in range(max(int(max_iterations), 1)):
            plan = self.plan_repair(workspace_root, command, execution_id=last_execution["id"], timeout=timeout)
            history.append({"stage": "planned", "plan": plan})
            if not plan["plan"].get("edits"):
                return {
                    "workspace_root": str(workspace_root),
                    "command": command,
                    "status": "needs_stronger_model_or_manual_review",
                    "history": history,
                }
            applied = self.apply_repair_plan(plan["id"], run_post_commands=True)
            history.append({"stage": "applied", "result": applied})
            last_execution = applied.get("result_execution") or last_execution
            if last_execution and last_execution["status"] == "completed" and int(last_execution.get("exit_code") or 0) == 0:
                return {
                    "workspace_root": str(workspace_root),
                    "command": command,
                    "status": "repaired",
                    "history": history,
                }
        return {
            "workspace_root": str(workspace_root),
            "command": command,
            "status": "max_iterations_reached",
            "history": history,
        }

    def review_execution(self, execution_id: str, question: str) -> dict:
        execution = self.command_runner.get_run(execution_id)
        if not execution:
            raise ValueError(f"Unknown execution run: {execution_id}")
        run_dir = str(Path(execution["stdout_path"]).parent)
        extra_context = (
            f"Execution review target:\n"
            f"- command: {' '.join(execution['command'])}\n"
            f"- status: {execution['status']}\n"
            f"- exit_code: {execution['exit_code']}"
        )
        result = self.ask(
            question,
            source_uri_prefix=run_dir,
            include_graph=False,
            extra_context_text=extra_context,
        )
        direct_answer = self._direct_execution_answer(question, execution)
        if direct_answer and result["backend"] == "fallback":
            result["answer"] = direct_answer
        result["execution_id"] = execution_id
        result["execution"] = execution
        self.session_log.append("review_execution", {"execution_id": execution_id, "question": question, **result})
        return result

    def health(self) -> dict:
        result = {
            "brain_root": str(self.paths.root),
            "db_exists": self.paths.db.exists(),
            "ledger": self.ledger.verify_chain(),
            "model": self.llm.health(),
            "embedding": self.document_store.embedding_model.health(),
            "queue": {
                "counts": self.task_queue.status_counts(),
                "pending": len(self.task_queue.list_tasks(status="pending", limit=5)),
            },
            "executions": {
                "count": self._count("execution_runs"),
                "recent": self.command_runner.list_runs(limit=3),
            },
            "repair_plans": {
                "count": self._count("repair_plans"),
                "recent": self.list_repair_plans(limit=3),
            },
        }
        self.session_log.append("health", result)
        return result

    def verify_ledger(self) -> dict:
        result = self.ledger.verify_chain()
        self.session_log.append("verify_ledger", result)
        return result

    def export_audit(self, path: str | Path) -> str:
        export_path = self.ledger.export_audit_report(path)
        self.session_log.append("export_audit", {"path": str(export_path)})
        return str(export_path)

    def status(self) -> dict:
        counts = {
            "documents": self._count("documents"),
            "chunks": self._count("chunks"),
            "memory_nodes": self._count("memory_nodes"),
            "memory_edges": self._count("memory_edges"),
            "learning_events": self._count("learning_events"),
            "training_candidates": self._count("training_candidates"),
            "ledger_blocks": self._count("learning_blocks"),
            "queued_tasks": self._count("task_queue"),
            "task_runs": self._count("task_runs"),
            "workspace_roots": self._count("workspace_roots"),
            "workspace_files": self._count("workspace_files"),
            "execution_runs": self._count("execution_runs"),
            "repair_plans": self._count("repair_plans"),
        }
        policy_version, policy = self.policy_store.get_current()
        ranker_version, ranker = self.ranker_store.get_current()
        active_adapter = self.adapter_registry.get_active_adapter()
        return {
            "brain_root": str(self.paths.root),
            "db": str(self.paths.db),
            "counts": counts,
            "policy_version": policy_version,
            "policy": policy.as_dict(),
            "ranker_version": ranker_version,
            "ranker": ranker.as_dict(),
            "active_adapter": active_adapter,
            "task_status_counts": self.task_queue.status_counts(),
            "model": self.llm.health(),
            "embedding": self.document_store.embedding_model.health(),
            "recent_executions": self.command_runner.list_runs(limit=3),
            "recent_repair_plans": self.list_repair_plans(limit=3),
            "ledger_ok": self.ledger.verify_chain()["ok"],
        }

    def demo(self) -> dict:
        self.init()
        demo_path = self.paths.temp / "demo_notes.md"
        demo_path.write_text(
            "# Proof-of-Improvement Persistent Transformer\n\n"
            "Austin Grandstaff's architecture treats persistent SSD storage as externalized transformer memory.\n\n"
            "Learning updates are committed only after validator consensus produces proof of improvement.\n\n"
            "The first mutable learning target in v0.3 is the retrieval ranker, not direct neural weight updates.\n",
            encoding="utf-8",
        )
        ingest_result = self.ingest(demo_path)
        fact_result = self.learn(
            "Project fact: The ledger is hash-linked and detects tampering.",
            "project_fact",
            {"name": "Ledger tamper detection", "confidence": 0.96, "importance": 0.95},
        )
        ask_result = self.ask("What is the first mutable learning target?")
        chunk_ids = [chunk["chunk_id"] for chunk in ask_result["included_chunks"][:1]]
        feedback_result = self.feedback(ask_result["retrieval_event_id"], useful_chunk_ids=chunk_ids, irrelevant_chunk_ids=[])
        demo = {
            "ingest": {"document_id": ingest_result["id"], "chunks": len(ingest_result["chunks"]), "duplicate": ingest_result["duplicate"]},
            "learn": {"status": fact_result["status"], "candidate_id": fact_result.get("candidate_id")},
            "ask": {"answer": ask_result["answer"], "retrieval_event_id": ask_result["retrieval_event_id"], "backend": ask_result["backend"]},
            "feedback": {"status": feedback_result["status"], "consensus_score": feedback_result.get("consensus_score")},
        }
        self.session_log.append("demo", demo)
        return demo

    def _direct_execution_answer(self, question: str, execution: dict) -> str | None:
        lowered = question.lower()
        stdout_text = Path(execution["stdout_path"]).read_text(encoding="utf-8") if execution.get("stdout_path") else ""
        stderr_text = Path(execution["stderr_path"]).read_text(encoding="utf-8") if execution.get("stderr_path") else ""
        if any(phrase in lowered for phrase in ("what did the command print", "stdout", "what was printed")):
            snippet = stdout_text.strip() or "<no stdout>"
            return f"Using local fallback generator. Command stdout:\n{snippet}"
        if any(phrase in lowered for phrase in ("stderr", "error output", "what error")):
            snippet = stderr_text.strip() or "<no stderr>"
            return f"Using local fallback generator. Command stderr:\n{snippet}"
        if any(phrase in lowered for phrase in ("exit code", "did it fail", "status")):
            return (
                "Using local fallback generator. "
                f"Execution status={execution['status']}, exit_code={execution['exit_code']}."
            )
        return None

    def _direct_workspace_answer(self, query: str, symbol_hits: list[dict]) -> str | None:
        if not symbol_hits:
            return None
        lowered = query.lower()
        if not any(phrase in lowered for phrase in ("where is", "where does", "what file", "defined", "definition")):
            return None
        preferred = next((node for node in symbol_hits if node["kind"] == "code_symbol"), symbol_hits[0])
        summary = preferred.get("summary") or preferred.get("content") or preferred["name"]
        source_uri = preferred.get("source_uri") or "unknown source"
        match = re.search(r" in ([^:]+):(\d+)", summary)
        if match:
            relative_path = match.group(1)
            line_number = match.group(2)
            return f"Using local fallback generator. {preferred['name']} is defined in {relative_path} at line {line_number}. Source: {source_uri}"
        return f"Using local fallback generator. Best local symbol match: {preferred['name']} from {source_uri}."

    def _apply_candidate(self, candidate: TrainingCandidate) -> dict:
        proposed = candidate.proposed_change
        target = candidate.target
        if target == "memory":
            node = self.graph.upsert_node_by_name_kind(
                kind=proposed["kind"],
                name=proposed["name"],
                content=proposed["content"],
                summary=proposed.get("summary"),
                source_uri=proposed.get("source_uri"),
                confidence=float(proposed.get("confidence", 0.8)),
                importance=float(proposed.get("importance", 0.7)),
            )
            edge_info = None
            link_to = proposed.get("link_to")
            if link_to:
                edge_info = self.graph.create_edge(
                    source_id=link_to.get("source_id", node["id"]),
                    target_id=node["id"],
                    relation=link_to.get("relation", "supports"),
                    weight=float(link_to.get("weight", 0.8)),
                    evidence=link_to.get("evidence"),
                    confidence=float(link_to.get("confidence", 0.8)),
                )
            return {
                "memory_diff": {
                    "applied": True,
                    "target": "memory",
                    "node_id": node["id"],
                    "edge_id": edge_info["id"] if edge_info else None,
                    "operation": proposed.get("operation", "upsert_node"),
                }
            }
        if target == "policy":
            version, policy = self.policy_store.apply_delta(proposed.get("updates", {}))
            return {
                "policy_version": version,
                "memory_diff": {
                    "applied": True,
                    "target": "policy",
                    "policy_version": version,
                    "updates": proposed.get("updates", {}),
                    "policy": policy.as_dict(),
                },
            }
        if target == "ranker":
            weights = RankerWeights.from_dict(proposed.get("new_weights", {}))
            version = self.ranker_store.save(weights, metrics=proposed.get("metrics", {}))
            return {
                "ranker_version": str(version),
                "memory_diff": {
                    "applied": True,
                    "target": "ranker",
                    "ranker_version": version,
                    "old_version": proposed.get("old_version"),
                    "new_weights": weights.as_dict(),
                    "metrics": proposed.get("metrics", {}),
                },
            }
        if target == "adapter":
            action = proposed.get("action", "accept")
            name = proposed["name"]
            version = proposed["version"]
            adapter = self.adapter_registry.accept_adapter(name=name, version=version) if action == "accept" else self.adapter_registry.reject_adapter(name=name, version=version)
            return {
                "adapter_version": adapter["version"],
                "memory_diff": {
                    "applied": True,
                    "target": "adapter",
                    "name": adapter["name"],
                    "version": adapter["version"],
                    "status": adapter["status"],
                },
            }
        raise ValueError(f"Unsupported candidate target: {target}")

    def _decode_repair_plan(self, row) -> dict:
        payload = dict(row)
        payload["command"] = json.loads(payload.pop("command_json"))
        payload["plan"] = json.loads(payload.pop("plan_json"))
        return payload

    def _count(self, table: str) -> int:
        return int(self.conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"])
