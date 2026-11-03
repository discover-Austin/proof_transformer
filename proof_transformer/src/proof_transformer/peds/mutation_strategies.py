from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

MUTATION_TARGETS = {"theta", "phi", "rho", "psi"}
_TARGET_FIELDS = {
    "theta": "theta_strategy",
    "phi": "retrieval_policy",
    "rho": "tool_policy",
    "psi": "update_policy",
}


def propose_candidate_mutation(
    parent_candidate: dict[str, Any],
    *,
    diagnosis: dict[str, Any],
    attempt_index: int,
    generation_index: int,
    allowed_targets: list[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(parent_candidate, dict):
        raise ValueError("parent_candidate must be a dict")
    if not isinstance(diagnosis, dict):
        raise ValueError("diagnosis must be a dict")
    if not isinstance(attempt_index, int) or isinstance(attempt_index, bool) or attempt_index <= 0:
        raise ValueError("attempt_index must be a positive int")
    if not isinstance(generation_index, int) or isinstance(generation_index, bool) or generation_index <= 0:
        raise ValueError("generation_index must be a positive int")

    allowed = _normalize_allowed_targets(allowed_targets)
    category = str(diagnosis.get("category") or "unknown")
    spec = _select_spec(category, allowed_targets=allowed, attempt_index=attempt_index)
    seed = _mutation_seed(
        parent_candidate=parent_candidate,
        category=category,
        attempt_index=attempt_index,
        generation_index=generation_index,
        spec=spec,
    )
    return _build_payload(spec, category=category, mutation_seed=seed)


def validate_single_mutation(candidate_payload: dict[str, Any]) -> dict[str, Any]:
    payload = candidate_payload.get("candidate_mutation") if isinstance(candidate_payload, dict) else None
    if not isinstance(payload, dict):
        payload = candidate_payload
    if not isinstance(payload, dict):
        return {"valid": False, "failed_checks": ["candidate payload must be a dict"], "mutation_target": None}

    failed_checks: list[str] = []
    mutation_target = payload.get("mutation_target")
    if mutation_target not in MUTATION_TARGETS:
        failed_checks.append("mutation_target must be one of theta, phi, rho, psi")

    active_fields = [
        target
        for target, field in _TARGET_FIELDS.items()
        if payload.get(field) not in (None, {}, [], "")
    ]
    if len(active_fields) != 1:
        failed_checks.append("exactly one target-specific mutation field must be populated")
    elif mutation_target in MUTATION_TARGETS and active_fields[0] != mutation_target:
        failed_checks.append(
            f"mutation_target {mutation_target!r} does not match populated field for {active_fields[0]!r}"
        )

    for key in ("delta_description", "mutation_seed", "diagnosis_category"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            failed_checks.append(f"{key} must be a non-empty string")

    return {
        "valid": not failed_checks,
        "failed_checks": failed_checks,
        "mutation_target": mutation_target if mutation_target in MUTATION_TARGETS else None,
    }


def _normalize_allowed_targets(allowed_targets: list[str] | None) -> set[str]:
    if allowed_targets is None:
        return set(MUTATION_TARGETS)
    allowed = {str(item) for item in allowed_targets}
    unknown = sorted(allowed - MUTATION_TARGETS)
    if unknown:
        raise ValueError(f"allowed_targets contains unknown mutation targets: {', '.join(unknown)}")
    if not allowed:
        raise ValueError("allowed_targets must not be empty")
    return allowed


def _select_spec(category: str, *, allowed_targets: set[str], attempt_index: int) -> dict[str, Any]:
    specs = _specs_by_category().get(category) or _specs_by_category()["unknown"]
    offset = (attempt_index - 1) % len(specs)
    ordered = specs[offset:] + specs[:offset]
    for spec in ordered:
        if spec["mutation_target"] in allowed_targets:
            return spec

    fallback_target = sorted(allowed_targets)[0]
    return {
        "mutation_target": fallback_target,
        "name": f"{fallback_target}_allowed_target_retry",
        "delta_description": f"Fallback {fallback_target} retry because requested diagnosis targets were disallowed.",
        **_fallback_field(fallback_target),
    }


def _specs_by_category() -> dict[str, list[dict[str, Any]]]:
    return {
        "replay_underperformance": [
            {
                "mutation_target": "theta",
                "theta_strategy": "acceptance_checklist_ordered_replay_focus",
                "name": "stronger_acceptance_ordering",
                "delta_description": "Reorders answer construction around acceptance checks before final response.",
            },
            {
                "mutation_target": "phi",
                "retrieval_policy": {
                    "policy": "diverse_replay_retrieval",
                    "batch_scope": "attempt_isolated",
                    "prefer_acceptance_checklist_hits": True,
                },
                "name": "diverse_replay_retrieval",
                "delta_description": "Retrieves a more diverse replay evidence slice before judging the candidate.",
            },
        ],
        "insufficient_delta": [
            {
                "mutation_target": "theta",
                "theta_strategy": "acceptance_checklist_feature_diversity",
                "name": "feature_diversity_theta",
                "delta_description": "Adds feature-diverse acceptance cues while keeping replay thresholds unchanged.",
            }
        ],
        "canary_rejection": [
            {
                "mutation_target": "rho",
                "tool_policy": {
                    "policy": "canary_safety_first",
                    "tool_risk_budget": "low",
                    "require_explicit_safety_check": True,
                },
                "name": "canary_safety_tool_policy",
                "delta_description": "Tightens safety/tool behavior for canary-sensitive attempts without weakening canary gates.",
            },
            {
                "mutation_target": "psi",
                "update_policy": {
                    "policy": "canary_evidence_isolation",
                    "batch_scope": "attempt_isolated",
                    "skip_previous_canary_batch": True,
                },
                "name": "canary_evidence_isolation",
                "delta_description": "Isolates canary evidence for the next attempt while preserving rejection semantics.",
            },
        ],
        "shadow_rejection": [
            {
                "mutation_target": "rho",
                "tool_policy": {
                    "policy": "shadow_low_noise_actions",
                    "shadow_sampling": "lower_noise",
                    "require_action_consistency": True,
                },
                "name": "shadow_low_noise_tool_policy",
                "delta_description": "Reduces noisy shadow behavior without lowering shadow gates.",
            }
        ],
        "training_error": [
            {
                "mutation_target": "psi",
                "update_policy": {
                    "policy": "trainer_noise_reduction",
                    "metadata_policy": "semantic_fields_only",
                    "deterministic_training": True,
                },
                "name": "trainer_noise_reduction",
                "delta_description": "Narrows trainer inputs to semantic fields while preserving trained-artifact requirements.",
            }
        ],
        "adapter_load_error": [
            {
                "mutation_target": "theta",
                "theta_strategy": "acceptance_checklist_artifact_reference_retry",
                "name": "artifact_reference_retry",
                "delta_description": "Retries with an explicit trained-adapter reference in the candidate manifest.",
            }
        ],
        "no_generated_child": [
            {
                "mutation_target": "psi",
                "update_policy": {
                    "policy": "lineage_parent_retry",
                    "parent_policy": "last_valid_parent",
                    "preserve_parent_state": True,
                },
                "name": "lineage_parent_retry",
                "delta_description": "Retries from the last valid lineage parent without promoting the rejected attempt.",
            }
        ],
        "unknown": [
            {
                "mutation_target": "theta",
                "theta_strategy": "acceptance_checklist_fresh_retry",
                "name": "fresh_theta_retry",
                "delta_description": "Creates a fresh theta retry when no stronger diagnosis signal is available.",
            }
        ],
    }


def _fallback_field(target: str) -> dict[str, Any]:
    if target == "theta":
        return {"theta_strategy": "acceptance_checklist_allowed_target_retry"}
    if target == "phi":
        return {"retrieval_policy": {"policy": "allowed_target_retrieval_retry"}}
    if target == "rho":
        return {"tool_policy": {"policy": "allowed_target_tool_retry"}}
    return {"update_policy": {"policy": "allowed_target_update_retry"}}


def _build_payload(spec: dict[str, Any], *, category: str, mutation_seed: str) -> dict[str, Any]:
    target = spec["mutation_target"]
    payload = {
        "mutation_target": target,
        "theta_strategy": None,
        "retrieval_policy": None,
        "tool_policy": None,
        "update_policy": None,
        "delta_description": spec["delta_description"],
        "mutation_seed": mutation_seed,
        "diagnosis_category": category,
    }
    field = _TARGET_FIELDS[target]
    payload[field] = deepcopy(spec[field])
    return payload


def _mutation_seed(
    *,
    parent_candidate: dict[str, Any],
    category: str,
    attempt_index: int,
    generation_index: int,
    spec: dict[str, Any],
) -> str:
    parent_manifest = parent_candidate.get("manifest")
    if not isinstance(parent_manifest, dict):
        parent_manifest = {}
    seed_payload = {
        "attempt_index": attempt_index,
        "diagnosis_category": category,
        "generation_index": generation_index,
        "parent_candidate_id": parent_candidate.get("candidate_id"),
        "parent_manifest": parent_manifest,
        "spec_name": spec.get("name"),
        "mutation_target": spec.get("mutation_target"),
    }
    encoded = json.dumps(seed_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "peds_mut_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
