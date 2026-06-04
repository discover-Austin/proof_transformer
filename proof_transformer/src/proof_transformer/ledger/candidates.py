from __future__ import annotations

import json
import sqlite3

from ..ids import new_id
from ..timeutils import utc_now
from .models import TrainingCandidate


CANDIDATE_EVENT_TYPES = {
    "correction",
    "accepted_output",
    "failed_output",
    "preference",
    "project_fact",
    "bug_fix",
    "retrieval_feedback",
    "contradiction",
    "general_observation",
}


class CandidateFactory:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create_learning_event(self, event_type: str, source_text: str, context: dict | None = None) -> dict:
        if event_type not in CANDIDATE_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type}")
        event_id = new_id("event")
        created_at = utc_now()
        payload = context or {}
        with self.conn:
            self.conn.execute(
                "INSERT INTO learning_events(id, event_type, source_text, context_json, created_at) VALUES(?, ?, ?, ?, ?)",
                (event_id, event_type, source_text, json.dumps(payload), created_at),
            )
        return {
            "id": event_id,
            "event_type": event_type,
            "source_text": source_text,
            "context": payload,
            "created_at": created_at,
        }

    def build_candidate(self, learning_event: dict) -> TrainingCandidate | None:
        event_type = learning_event["event_type"]
        source_text = learning_event["source_text"].strip()
        context = learning_event.get("context") or {}
        if event_type == "general_observation" and not self._looks_candidate_worthy(source_text):
            return None

        candidate_id = new_id("candidate")
        created_at = utc_now()
        instruction, chosen, rejected, tags, proposed_change = self._build_payload(event_type, source_text, context)
        return TrainingCandidate(
            id=candidate_id,
            event_id=learning_event["id"],
            instruction=instruction,
            chosen=chosen,
            rejected=rejected,
            tags=tags,
            proposed_change=proposed_change,
            status="proposed",
            created_at=created_at,
        )

    def save_candidate(self, candidate: TrainingCandidate) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO training_candidates(
                    id, event_id, instruction, chosen, rejected,
                    tags_json, proposed_change_json, status, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.event_id,
                    candidate.instruction,
                    candidate.chosen,
                    candidate.rejected,
                    json.dumps(candidate.tags),
                    json.dumps(candidate.proposed_change),
                    candidate.status,
                    candidate.created_at,
                ),
            )

    def update_candidate_status(self, candidate_id: str, status: str) -> None:
        with self.conn:
            self.conn.execute("UPDATE training_candidates SET status = ? WHERE id = ?", (status, candidate_id))

    def get_candidate(self, candidate_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM training_candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["tags"] = json.loads(result.pop("tags_json"))
        result["proposed_change"] = json.loads(result.pop("proposed_change_json"))
        return result

    def _build_payload(
        self,
        event_type: str,
        source_text: str,
        context: dict,
    ) -> tuple[str, str, str | None, list[str], dict]:
        if event_type == "preference":
            updates = self._infer_policy_updates(source_text)
            if updates:
                return (
                    "Apply this durable user preference to future decoding decisions.",
                    source_text,
                    context.get("prior_behavior"),
                    ["preference", "policy"],
                    {
                        "target": "policy",
                        "event_type": event_type,
                        "updates": updates,
                        "evidence": source_text,
                    },
                )
            return (
                "Remember this durable user preference and surface it during future retrieval.",
                source_text,
                context.get("prior_behavior"),
                ["preference", "memory"],
                {
                    "target": "memory",
                    "event_type": event_type,
                    "operation": "upsert_node",
                    "kind": "preference",
                    "name": self._name_from_text(source_text, fallback="Durable preference"),
                    "content": source_text,
                    "summary": source_text[:200],
                    "confidence": 0.95,
                    "importance": 0.90,
                },
            )

        if event_type == "project_fact":
            name = context.get("name") or self._name_from_text(source_text, fallback="Project fact")
            return (
                "Store this project fact for future retrieval and contradiction checks.",
                source_text,
                context.get("rejected"),
                ["project_fact", "memory"],
                {
                    "target": "memory",
                    "event_type": event_type,
                    "operation": "upsert_node",
                    "kind": "project_fact",
                    "name": name,
                    "content": source_text,
                    "summary": source_text[:200],
                    "confidence": float(context.get("confidence", 0.95)),
                    "importance": float(context.get("importance", 0.95)),
                },
            )

        if event_type == "bug_fix":
            return (
                "Remember this bug fix and prefer similar verified fixes later.",
                source_text,
                context.get("prior_output"),
                ["bug_fix", "memory"],
                {
                    "target": "memory",
                    "event_type": event_type,
                    "operation": "upsert_node",
                    "kind": "bug_fix",
                    "name": self._name_from_text(source_text, fallback="Bug fix"),
                    "content": source_text,
                    "summary": source_text[:200],
                    "confidence": float(context.get("confidence", 0.92)),
                    "importance": float(context.get("importance", 0.90)),
                    "tests_passed": bool(context.get("tests_passed", False)),
                },
            )

        if event_type == "correction":
            return (
                "Prefer this corrected answer over the rejected prior behavior.",
                source_text,
                context.get("prior_output"),
                ["correction", "memory"],
                {
                    "target": "memory",
                    "event_type": event_type,
                    "operation": "upsert_node",
                    "kind": "correction",
                    "name": self._name_from_text(source_text, fallback="Correction"),
                    "content": source_text,
                    "summary": source_text[:200],
                    "confidence": 0.9,
                    "importance": 0.88,
                },
            )

        if event_type == "accepted_output":
            return (
                "Remember this accepted output pattern as positive evidence.",
                source_text,
                context.get("rejected"),
                ["accepted_output", "memory", "user_approved"],
                {
                    "target": "memory",
                    "event_type": event_type,
                    "operation": "upsert_node",
                    "kind": "accepted_output",
                    "name": self._name_from_text(source_text, fallback="Accepted output"),
                    "content": source_text,
                    "summary": source_text[:200],
                    "confidence": 0.9,
                    "importance": 0.8,
                },
            )

        if event_type == "failed_output":
            updates = self._infer_policy_updates(source_text) or {"evidence_required": 0.95, "scope_control": 0.95}
            return (
                "Use this failure signal to tighten future decoding policy.",
                source_text,
                context.get("prior_output"),
                ["failed_output", "policy"],
                {
                    "target": "policy",
                    "event_type": event_type,
                    "updates": updates,
                    "evidence": source_text,
                },
            )

        if event_type == "retrieval_feedback":
            proposed_change = dict(context.get("proposed_change") or {})
            proposed_change.setdefault("target", "ranker")
            proposed_change.setdefault("event_type", event_type)
            tags = ["retrieval_feedback", "ranker"]
            if proposed_change.get("high_impact"):
                tags.append("high_impact")
            return (
                "Apply this validated retrieval feedback to update ranker weights.",
                source_text,
                context.get("rejected"),
                tags,
                proposed_change,
            )

        if event_type == "contradiction":
            name = context.get("name") or self._name_from_text(source_text, fallback="Contradiction")
            proposed = {
                "target": "memory",
                "event_type": event_type,
                "operation": "upsert_node",
                "kind": "contradiction",
                "name": name,
                "content": source_text,
                "summary": source_text[:200],
                "confidence": float(context.get("confidence", 0.85)),
                "importance": float(context.get("importance", 0.85)),
            }
            if context.get("existing_node_id"):
                proposed["link_to"] = {
                    "source_id": context["existing_node_id"],
                    "relation": "contradicts",
                    "weight": 0.9,
                    "evidence": source_text,
                    "confidence": 0.9,
                }
            return (
                "Record this contradiction and surface it during future validation.",
                source_text,
                context.get("rejected"),
                ["contradiction", "memory"],
                proposed,
            )

        return (
            "Store this observation with cautious scope for later retrieval.",
            source_text,
            context.get("rejected"),
            ["general_observation", "memory"],
            {
                "target": "memory",
                "event_type": event_type,
                "operation": "upsert_node",
                "kind": "observation",
                "name": self._name_from_text(source_text, fallback="Observation"),
                "content": source_text,
                "summary": source_text[:200],
                "confidence": float(context.get("confidence", 0.65)),
                "importance": float(context.get("importance", 0.55)),
            },
        )

    def _looks_candidate_worthy(self, text: str) -> bool:
        lower = text.lower()
        triggers = ("always", "never", "prefer", "must", "important", "fact", "because")
        return len(text.split()) >= 8 and any(trigger in lower for trigger in triggers)

    def _infer_policy_updates(self, text: str) -> dict[str, float]:
        lower = text.lower()
        updates: dict[str, float] = {}
        if any(word in lower for word in ("root cause", "root-cause", "diagnose first")):
            updates["root_cause_first"] = 1.0
        if any(word in lower for word in ("full replacement", "full code", "complete code", "entire file")):
            updates["full_replacement_code"] = 1.0
        if any(word in lower for word in ("placeholder", "todo", "stub", "fake logic", "mock logic")):
            updates["avoid_placeholders"] = 1.0
        if "preserve context" in lower:
            updates["preserve_context"] = 1.0
        if any(word in lower for word in ("evidence", "cite", "source")):
            updates["evidence_required"] = 1.0
            updates["cite_sources_when_available"] = 1.0
        if any(word in lower for word in ("scope", "narrow", "scoped")):
            updates["scope_control"] = 1.0
        if any(word in lower for word in ("local path", "local file", "local only")):
            updates["prefer_local_paths"] = 1.0
        if any(word in lower for word in ("no fake autonomy", "don't pretend", "do not pretend", "say when fallback")):
            updates["no_fake_autonomy"] = 1.0
        if any(word in lower for word in ("be concise", "concise")):
            updates["concise_when_simple"] = 1.0
        if any(word in lower for word in ("go deep", "deep when", "project-level", "serious project")):
            updates["deep_when_project"] = 1.0
        return updates

    def _name_from_text(self, text: str, fallback: str) -> str:
        first = text.strip().splitlines()[0].strip()
        if ":" in first and first.index(":") <= 60:
            return first.split(":", 1)[0].strip()[:80] or fallback
        words = first.split()
        return (" ".join(words[:8])[:80]).strip() or fallback
