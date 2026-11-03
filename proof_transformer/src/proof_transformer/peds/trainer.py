from __future__ import annotations

from collections import Counter
import json
import sqlite3
from pathlib import Path
import re
from typing import Any

from ..timeutils import utc_now
from .distillation import validate_distillation_manifest
from .training import get_training_run, record_training_result

_SUPPORTED_METADATA_MODES = {"dry_run", "metadata_only"}
_SUPPORTED_REAL_MODES = {"local_json_trainer"}
_PREPARED_STATUSES = {"prepared", "queued"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def validate_trainer_mode(mode: str) -> dict[str, Any]:
    failed_checks: list[str] = []
    if mode in _SUPPORTED_METADATA_MODES:
        return {
            "valid": True,
            "mode": mode,
            "reason": f"{mode} is supported as an explicit non-training metadata mode",
            "failed_checks": failed_checks,
        }
    if mode in _SUPPORTED_REAL_MODES:
        return {
            "valid": True,
            "mode": mode,
            "reason": f"{mode} is supported as a deterministic local trainer",
            "failed_checks": failed_checks,
        }
    if mode == "local_trainer":
        failed_checks.append("no local adapter trainer abstraction is available in this repository")
    else:
        failed_checks.append(f"unsupported trainer mode: {mode!r}")
    return {
        "valid": False,
        "mode": mode,
        "reason": "; ".join(failed_checks),
        "failed_checks": failed_checks,
    }


def build_adapter_artifact_manifest(training_run: dict[str, Any], *, operator: str, mode: str) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    mode_check = validate_trainer_mode(mode)
    if not mode_check["valid"]:
        raise ValueError(mode_check["reason"])

    manifest = training_run["manifest"]
    n_examples = int(manifest.get("n_examples", 0))
    dataset_hash = training_run["dataset_hash"]
    metrics = {
        "mode": mode,
        "trained": False,
        "n_examples": n_examples,
        "dataset_hash": dataset_hash,
    }
    return {
        "artifact_type": "peds_adapter_artifact_manifest",
        "training_run_id": training_run["training_run_id"],
        "candidate_id": training_run["candidate_id"],
        "dataset_hash": dataset_hash,
        "trainer_name": training_run["trainer_name"],
        "mode": mode,
        "trained": False,
        "operator": operator_text,
        "n_examples": n_examples,
        "created_at": utc_now(),
        "metrics": metrics,
        "provenance": {
            "training_run_id": training_run["training_run_id"],
            "training_run_status": training_run["status"],
            "manifest_created_at": manifest.get("created_at"),
            "manifest_dataset_hash": manifest.get("dataset_hash"),
            "trainer_config": training_run.get("trainer_config") or {},
            "metadata_only": True,
            "trained": False,
        },
    }


def run_prepared_training_run(
    conn: sqlite3.Connection,
    training_run_id: str,
    *,
    operator: str,
    mode: str = "dry_run",
    artifact_dir: str | None = None,
) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    mode_check = validate_trainer_mode(mode)
    if not mode_check["valid"]:
        raise ValueError(mode_check["reason"])

    training_run = get_training_run(conn, training_run_id)
    if training_run["status"] not in _PREPARED_STATUSES:
        raise ValueError(
            f"Training run {training_run_id} must be prepared or queued, got {training_run['status']!r}"
        )

    manifest_validation = validate_distillation_manifest(training_run["manifest"])
    if not manifest_validation["valid"]:
        raise ValueError("Training run manifest is invalid: " + manifest_validation["reason"])

    if mode == "local_json_trainer":
        if not artifact_dir:
            raise ValueError("artifact_dir is required for local_json_trainer")
        artifact_manifest = build_local_json_adapter_artifact(training_run, operator=operator_text)
        artifact_ref = _write_artifact_manifest(artifact_dir, training_run_id, artifact_manifest)
        metrics = dict(artifact_manifest["metrics"])
        metrics["operator"] = operator_text
        metrics["artifact_manifest_written"] = True
        completed_run = record_training_result(
            conn,
            training_run_id,
            status="completed",
            metrics=metrics,
            artifact_ref=artifact_ref,
            error=None,
        )
        return {
            "training_run_id": training_run_id,
            "candidate_id": completed_run["candidate_id"],
            "mode": mode,
            "operator": operator_text,
            "trained": True,
            "artifact_ref": artifact_ref,
            "artifact_manifest": artifact_manifest,
            "training_run": completed_run,
            "mode_validation": mode_check,
        }

    artifact_manifest = build_adapter_artifact_manifest(training_run, operator=operator_text, mode=mode)
    artifact_ref = None
    if artifact_dir is not None:
        artifact_ref = _write_artifact_manifest(artifact_dir, training_run_id, artifact_manifest)

    metrics = {
        "mode": mode,
        "trained": False,
        "n_examples": artifact_manifest["n_examples"],
        "dataset_hash": artifact_manifest["dataset_hash"],
        "operator": operator_text,
        "artifact_manifest_written": artifact_ref is not None,
    }
    completed_run = record_training_result(
        conn,
        training_run_id,
        status="completed",
        metrics=metrics,
        artifact_ref=artifact_ref,
        error=None,
    )

    return {
        "training_run_id": training_run_id,
        "candidate_id": completed_run["candidate_id"],
        "mode": mode,
        "operator": operator_text,
        "trained": False,
        "artifact_ref": artifact_ref,
        "artifact_manifest": artifact_manifest,
        "training_run": completed_run,
        "mode_validation": mode_check,
    }


def build_local_json_adapter_artifact(training_run: dict[str, Any], *, operator: str) -> dict[str, Any]:
    operator_text = _require_non_empty_text(operator, "operator")
    manifest = training_run["manifest"]
    validation = validate_distillation_manifest(manifest)
    if not validation["valid"]:
        raise ValueError("Training run manifest is invalid: " + validation["reason"])

    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("local_json_trainer requires at least one manifest example")

    learned_parameters, trainer_metrics = _learn_local_json_parameters(examples)
    learned_parameter_count = _count_learned_parameters(learned_parameters)
    n_examples = len(examples)
    dataset_hash = training_run["dataset_hash"]
    metrics = {
        "mode": "local_json_trainer",
        "trained": True,
        "n_examples": n_examples,
        "dataset_hash": dataset_hash,
        "learned_parameter_count": learned_parameter_count,
        **trainer_metrics,
    }
    return {
        "artifact_type": "peds_adapter_artifact_manifest",
        "training_run_id": training_run["training_run_id"],
        "candidate_id": training_run["candidate_id"],
        "dataset_hash": dataset_hash,
        "trainer_name": training_run["trainer_name"],
        "mode": "local_json_trainer",
        "trained": True,
        "operator": operator_text,
        "n_examples": n_examples,
        "created_at": utc_now(),
        "learned_parameters": learned_parameters,
        "metrics": metrics,
        "provenance": {
            "training_run_id": training_run["training_run_id"],
            "training_run_status": training_run["status"],
            "manifest_created_at": manifest.get("created_at"),
            "manifest_dataset_hash": manifest.get("dataset_hash"),
            "trainer_config": training_run.get("trainer_config") or {},
            "metadata_only": False,
            "trained": True,
            "trainer_inputs": {
                "example_ids": [str(example.get("example_id")) for example in examples],
                "source_summary": manifest.get("source_summary") or {},
            },
        },
    }


def _learn_local_json_parameters(examples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_type_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    chosen_token_counts: list[int] = []
    input_token_counts: list[int] = []
    acceptance_overlaps: list[float] = []
    prompt_overlaps: list[float] = []
    replay_deltas: list[float] = []
    chosen_vocab: Counter[str] = Counter()
    input_vocab: Counter[str] = Counter()
    accepted_shadow_count = 0
    replay_win_count = 0
    safety_exclusion_count = 0

    for example in examples:
        source_type = str(example.get("source_type") or "unknown")
        source_type_counts[source_type] += 1
        tags = [str(tag) for tag in example.get("tags") or []]
        tag_counts.update(tags)

        chosen_tokens = _chosen_signal_tokens(example.get("chosen"))
        input_tokens = _tokens(example.get("input"))
        acceptance_tokens = _tokens(_acceptance_tests(example))
        prompt_tokens = _tokens(_prompt_text(example))

        chosen_token_counts.append(len(chosen_tokens))
        input_token_counts.append(len(input_tokens))
        chosen_vocab.update(chosen_tokens)
        input_vocab.update(input_tokens)
        acceptance_overlaps.append(_overlap_ratio(acceptance_tokens, chosen_tokens))
        prompt_overlaps.append(_overlap_ratio(prompt_tokens, chosen_tokens))

        provenance = example.get("provenance") or {}
        if provenance.get("accepted_by_incumbent") is True or "accepted_by_incumbent" in tags:
            accepted_shadow_count += 1
        if source_type == "replay_result" or "replay_win" in tags:
            replay_win_count += 1
        if "delta" in provenance:
            replay_deltas.append(_float_or_zero(provenance.get("delta")))
        safety_exclusion_count += len(example.get("safety_exclusions") or [])

    n_examples = len(examples)
    learned_parameters = {
        "feature_weights": {
            "acceptance_token_overlap": _round(_mean(acceptance_overlaps)),
            "prompt_token_overlap": _round(_mean(prompt_overlaps)),
            "shadow_acceptance_signal": _round(accepted_shadow_count / n_examples),
            "replay_win_signal": _round(replay_win_count / n_examples),
            "mean_replay_delta": _round(_mean(replay_deltas)),
            "chosen_length_signal": _round(min(_mean(chosen_token_counts) / 200.0, 1.0)),
            "safety_exclusion_penalty": _round(-(safety_exclusion_count / n_examples)),
        },
        "source_type_weights": _normalized_counter(source_type_counts, n_examples),
        "tag_weights": _normalized_counter(tag_counts, n_examples),
        "top_chosen_tokens": _top_counter(chosen_vocab),
        "top_input_tokens": _top_counter(input_vocab),
    }
    metrics = {
        "n_shadow_examples": source_type_counts.get("shadow_event", 0),
        "n_replay_examples": source_type_counts.get("replay_result", 0),
        "mean_acceptance_token_overlap": _round(_mean(acceptance_overlaps)),
        "mean_prompt_token_overlap": _round(_mean(prompt_overlaps)),
        "mean_chosen_token_count": _round(_mean(chosen_token_counts)),
        "mean_input_token_count": _round(_mean(input_token_counts)),
        "safety_exclusion_count": safety_exclusion_count,
    }
    return learned_parameters, metrics


def _acceptance_tests(example: dict[str, Any]) -> Any:
    input_value = example.get("input") if isinstance(example.get("input"), dict) else {}
    return input_value.get("acceptance_tests")


def _prompt_text(example: dict[str, Any]) -> Any:
    input_value = example.get("input") if isinstance(example.get("input"), dict) else {}
    return input_value.get("prompt")


def _chosen_signal_tokens(value: Any) -> list[str]:
    signal = _chosen_signal_value(value)
    if signal not in (None, "", [], {}):
        return _tokens(signal)
    return _tokens(value)


def _chosen_signal_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value

    signal_keys = ("acceptance_checklist", "answer", "final_message", "patch", "actions")
    signal_values = [value[key] for key in signal_keys if key in value and value[key] not in (None, "", [], {})]
    if signal_values:
        return signal_values
    return None


def _tokens(value: Any) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(_stringify(value))]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_stringify(nested) for _, nested in sorted(value.items(), key=lambda item: str(item[0])))
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify(nested) for nested in value)
    return str(value)


def _overlap_ratio(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def _normalized_counter(counter: Counter[str], denominator: int) -> dict[str, float]:
    return {key: _round(counter[key] / denominator) for key in sorted(counter)}


def _top_counter(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"token": token, "count": count}
        for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _count_learned_parameters(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_learned_parameters(nested) for nested in value.values())
    if isinstance(value, list):
        return sum(_count_learned_parameters(nested) for nested in value)
    return 1


def _mean(values: list[float] | list[int]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _write_artifact_manifest(artifact_dir: str, training_run_id: str, manifest: dict[str, Any]) -> str:
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{training_run_id}_adapter_artifact_manifest.json"
    target.write_text(_canonical_json_pretty(manifest), encoding="utf-8")
    return str(target)


def _require_non_empty_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _canonical_json_pretty(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
