from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class DecoderPolicy:
    root_cause_first: float = 0.90
    full_replacement_code: float = 0.95
    avoid_placeholders: float = 1.00
    preserve_context: float = 0.90
    evidence_required: float = 0.85
    scope_control: float = 0.85
    cite_sources_when_available: float = 0.75
    prefer_local_paths: float = 1.00
    no_fake_autonomy: float = 1.00
    concise_when_simple: float = 0.70
    deep_when_project: float = 0.90

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def apply_updates(self, updates: dict[str, float]) -> "DecoderPolicy":
        payload = self.as_dict()
        for key, value in updates.items():
            if key in payload:
                payload[key] = max(0.0, min(1.0, float(value)))
        return DecoderPolicy(**payload)


@dataclass
class TrainingCandidate:
    id: str
    event_id: str
    instruction: str
    chosen: str
    rejected: str | None
    tags: list[str] = field(default_factory=list)
    proposed_change: dict = field(default_factory=dict)
    status: str = "proposed"
    created_at: str = ""

    @property
    def target(self) -> str:
        return str(self.proposed_change.get("target", "memory"))

    @property
    def event_type(self) -> str:
        return str(self.proposed_change.get("event_type", "general_observation"))

    def as_record(self) -> dict:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "instruction": self.instruction,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "tags": self.tags,
            "proposed_change": self.proposed_change,
            "status": self.status,
            "created_at": self.created_at,
        }


@dataclass
class ValidatorResult:
    id: str
    candidate_id: str
    validator_name: str
    passed: bool
    score: float
    reason: str
    metadata: dict = field(default_factory=dict)
    created_at: str = ""

    def as_record(self) -> dict:
        return {
            "id": self.id,
            "candidate_id": self.candidate_id,
            "validator_name": self.validator_name,
            "passed": self.passed,
            "score": self.score,
            "reason": self.reason,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }
