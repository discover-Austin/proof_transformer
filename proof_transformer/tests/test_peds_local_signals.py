from __future__ import annotations

import json
from typing import Any

from common import BrainTestCase
from proof_transformer.learning.adapter_registry import AdapterRegistry
from proof_transformer.peds.local_signals import (
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
    load_candidate_local_adapter,
    local_adapter_behavior_profile,
    make_local_episode_shadow_event_provider,
)
from proof_transformer.peds.store import create_candidate, record_episode


class PEDSLocalSignalTests(BrainTestCase):
    def _candidate(
        self,
        suffix: str,
        *,
        parent_candidate_id: str | None = None,
        theta_strategy: str,
        manifest: dict[str, Any] | None = None,
    ) -> dict:
        manifest_body = {
            "suffix": suffix,
            "theta_strategy": theta_strategy,
        }
        if manifest:
            manifest_body.update(manifest)
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_local_signal_{suffix}",
                "parent_candidate_id": parent_candidate_id,
                "mutation_target": "theta",
                "phi_version": "phi_local_signal_v1",
                "rho_version": "rho_local_signal_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_local_signal_v1",
                "delta_description": f"local signal candidate {suffix}",
                "manifest_json": manifest_body,
            },
        )

    def _episode(
        self,
        episode_id: str,
        *,
        split: str = "replay",
        policy_tags: list[str] | None = None,
        judge_verdict: str = "pass",
        user_verdict: str = "accepted",
        prompt: str | None = None,
        acceptance_tests: list[str] | None = None,
    ) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
                "timestamp": "2026-06-04T00:00:00Z",
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Local signal task {episode_id}",
                    "prompt": prompt or f"Handle local signal task {episode_id}",
                    "acceptance_tests": acceptance_tests or [f"pytest tests/{episode_id}.py"],
                    "policy_tags": policy_tags or ["local_signal"],
                },
                "snapshot": {
                    "repo_commit": "local-signal",
                    "repo_tree_hash": "tree_hash",
                    "retrieval_index_hash": "retrieval_hash",
                    "env_image_hash": "env_hash",
                    "tool_profile": "safe_rw_v1",
                    "model_lock": {
                        "provider": "local",
                        "model": "deterministic_signal",
                        "revision": "acceptance_checklist_v1",
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 77,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [{"uri": "file://README.md", "kind": "file", "tokens": 80}],
                    "actions": [{"step": 1, "kind": "inspect", "args": {"episode_id": episode_id}, "exit_code": 0}],
                    "patch": "diff --git a/example.py b/example.py",
                    "final_message": f"Accepted local answer for {episode_id}.",
                },
                "outcome": {"judge_verdict": judge_verdict, "user_verdict": user_verdict},
                "metrics": {
                    "tests_passed": 1.0,
                    "tokens_in": 100,
                    "tokens_out": 25,
                    "latency_ms": 200,
                    "tool_calls": 2,
                    "failed_tool_calls": 0,
                    "diff_lines": 3,
                    "files_edited": 1,
                    "hallucinated_claims": 0,
                    "unrelated_edit_ratio": 0.0,
                },
                "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
            },
        )

    def _adapter_artifact(
        self,
        suffix: str,
        *,
        trained: bool,
        mode: str = "local_json_trainer",
        top_tokens: list[str] | None = None,
    ) -> dict[str, Any]:
        tokens = top_tokens or [f"adapter_token_{suffix}"]
        learned_parameters = {
            "feature_weights": {
                "acceptance_token_overlap": 1.0,
                "prompt_token_overlap": 0.5,
                "shadow_acceptance_signal": 1.0,
                "replay_win_signal": 0.0,
                "mean_replay_delta": 0.0,
                "chosen_length_signal": 0.1,
                "safety_exclusion_penalty": 0.0,
            },
            "source_type_weights": {"shadow_event": 1.0},
            "tag_weights": {"accepted_by_incumbent": 1.0},
            "top_input_tokens": [{"token": token, "count": 3} for token in tokens],
            "top_chosen_tokens": [{"token": token, "count": 2} for token in tokens],
        }
        return {
            "artifact_type": "peds_adapter_artifact_manifest",
            "training_run_id": f"training_run_local_signal_{suffix}",
            "candidate_id": f"candidate_local_signal_{suffix}",
            "dataset_hash": f"sha256:{suffix}",
            "trainer_name": "local_json_trainer",
            "mode": mode,
            "trained": trained,
            "operator": "owner",
            "n_examples": 1,
            "created_at": "2026-06-04T00:00:00Z",
            "learned_parameters": learned_parameters,
            "metrics": {
                "mode": mode,
                "trained": trained,
                "n_examples": 1,
                "dataset_hash": f"sha256:{suffix}",
            },
            "provenance": {"trained": trained, "metadata_only": not trained},
        }

    def _candidate_with_adapter(
        self,
        suffix: str,
        *,
        trained: bool = True,
        mode: str = "local_json_trainer",
        top_tokens: list[str] | None = None,
        theta_strategy: str = "baseline_final_answer",
    ) -> dict:
        artifact = self._adapter_artifact(
            suffix,
            trained=trained,
            mode=mode,
            top_tokens=top_tokens,
        )
        artifact_path = self.paths.root / "local_signal_adapters" / f"{suffix}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(artifact, sort_keys=True), encoding="utf-8")
        adapter = AdapterRegistry(self.conn).register_adapter(
            name=f"local_signal_adapter_{suffix}",
            version="v1",
            path=artifact_path,
            dataset_hash=artifact["dataset_hash"],
            status="trained" if trained else "metadata_only",
        )
        return self._candidate(
            f"{suffix}_adapter_candidate",
            theta_strategy=theta_strategy,
            manifest={
                "trained": trained,
                "trainer_mode": mode,
                "adapter_registry_id": adapter["id"],
                "adapter_ref": {
                    "name": adapter["name"],
                    "version": adapter["version"],
                    "path": adapter["path"],
                    "status": adapter["status"],
                },
                "artifact_ref": str(artifact_path),
            },
        )

    def test_acceptance_checklist_evaluator_scores_manifest_strategy_from_episode_signal(self) -> None:
        incumbent = self._candidate("incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate(
            "candidate",
            parent_candidate_id=incumbent["candidate_id"],
            theta_strategy="acceptance_checklist_first",
        )
        episode = self._episode("episode_local_signal_replay")

        replay_result = acceptance_checklist_replay_evaluator(
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )
        canary_result = acceptance_checklist_canary_evaluator(
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )

        self.assertGreater(replay_result["candidate_score"], replay_result["incumbent_score"])
        self.assertEqual(replay_result["judge_verdict"], "pass")
        self.assertEqual(canary_result["canary_failures"], [])
        self.assertTrue(replay_result["audit_log_ref"].startswith("local:acceptance_checklist:replay"))

    def test_load_candidate_local_adapter_returns_none_for_metadata_only_and_dry_run_candidate(self) -> None:
        dry_run_candidate = self._candidate("adapter_dry_run", theta_strategy="baseline_final_answer")
        metadata_candidate = self._candidate_with_adapter(
            "adapter_metadata_only",
            trained=False,
            mode="metadata_only",
        )

        self.assertIsNone(load_candidate_local_adapter(self.conn, dry_run_candidate))
        self.assertIsNone(load_candidate_local_adapter(self.conn, metadata_candidate))

    def test_load_candidate_local_adapter_loads_trained_local_json_artifact(self) -> None:
        candidate = self._candidate_with_adapter(
            "adapter_trained",
            top_tokens=["adapter_trained_signal"],
        )

        artifact = load_candidate_local_adapter(self.conn, candidate)
        profile = local_adapter_behavior_profile(self.conn, candidate)

        self.assertIsNotNone(artifact)
        self.assertTrue(artifact["trained"])
        self.assertEqual(artifact["mode"], "local_json_trainer")
        self.assertTrue(profile["adapter"]["loaded"])
        self.assertIn("top_input_tokens", profile["learned_parameters"])

    def test_evaluator_score_changes_when_trained_adapter_matches_episode_signal(self) -> None:
        incumbent = self._candidate("adapter_match_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate_with_adapter(
            "adapter_match",
            top_tokens=["adapter_match_signal"],
        )
        episode = self._episode(
            "episode_local_signal_adapter_match",
            prompt="Handle a task with adapter_match_signal coverage.",
            acceptance_tests=["pytest tests/adapter_match_signal.py"],
        )

        result = acceptance_checklist_replay_evaluator(
            conn=self.conn,
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )

        self.assertGreater(result["candidate_score"], result["incumbent_score"])
        self.assertTrue(result["candidate_adapter_behavior_profile"]["adapter"]["loaded"])

    def test_evaluator_score_does_not_change_for_unrelated_learned_parameters(self) -> None:
        incumbent = self._candidate("adapter_unrelated_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate_with_adapter(
            "adapter_unrelated",
            top_tokens=["totally_unrelated_adapter_signal"],
        )
        episode = self._episode(
            "episode_local_signal_adapter_unrelated",
            prompt="Handle a normal local signal task.",
            acceptance_tests=["pytest tests/ordinary_local_signal.py"],
        )

        result = acceptance_checklist_replay_evaluator(
            conn=self.conn,
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )

        self.assertEqual(result["candidate_score"], result["incumbent_score"])
        self.assertTrue(result["candidate_adapter_behavior_profile"]["adapter"]["loaded"])

    def test_local_episode_shadow_provider_reads_filtered_online_episode_rows(self) -> None:
        incumbent = self._candidate("shadow_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate(
            "shadow_candidate",
            parent_candidate_id=incumbent["candidate_id"],
            theta_strategy="acceptance_checklist_first",
        )
        kept_ids = []
        for index in range(3):
            episode = self._episode(
                f"episode_local_signal_online_{index}",
                split="online",
                policy_tags=["local_signal", "shadow_source"],
            )
            kept_ids.append(episode["episode_id"])
        self._episode(
            "episode_local_signal_online_filtered",
            split="online",
            policy_tags=["other_signal"],
        )

        provider = make_local_episode_shadow_event_provider(
            split="online",
            limit=10,
            policy_tags=["shadow_source"],
        )
        events = provider(
            conn=self.conn,
            candidate=candidate,
            incumbent=incumbent,
            shadow_run={"shadow_run_id": "shadow_run_local_signal"},
        )

        self.assertEqual([event["episode_ref"] for event in events], kept_ids)
        self.assertTrue(all(event["accepted_by_incumbent"] for event in events))
        self.assertTrue(all(event["divergence"] is None for event in events))
        self.assertTrue(all("acceptance_checklist" in event["candidate_output"] for event in events))
        self.assertEqual(events[0]["metadata"]["provider"], "local_episode_shadow_events")

    def test_local_episode_shadow_provider_includes_adapter_behavior_profile(self) -> None:
        incumbent = self._candidate("shadow_profile_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate_with_adapter(
            "shadow_profile_candidate",
            top_tokens=["shadow_profile_signal"],
            theta_strategy="acceptance_checklist_first",
        )
        self._episode(
            "episode_local_signal_shadow_profile",
            split="online",
            policy_tags=["local_signal", "shadow_profile"],
        )

        provider = make_local_episode_shadow_event_provider(
            split="online",
            limit=1,
            policy_tags=["shadow_profile"],
        )
        events = provider(
            conn=self.conn,
            candidate=candidate,
            incumbent=incumbent,
            shadow_run={"shadow_run_id": "shadow_run_local_signal_profile"},
        )

        candidate_profile = events[0]["candidate_output"]["metadata"]["adapter_behavior_profile"]
        incumbent_profile = events[0]["incumbent_output"]["metadata"]["adapter_behavior_profile"]
        self.assertTrue(candidate_profile["adapter"]["loaded"])
        self.assertFalse(incumbent_profile["adapter"]["loaded"])

    def test_canary_evaluator_marks_local_safety_failures(self) -> None:
        incumbent = self._candidate("unsafe_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate(
            "unsafe_candidate",
            parent_candidate_id=incumbent["candidate_id"],
            theta_strategy="acceptance_checklist_first",
        )
        episode = self._episode(
            "episode_local_signal_unsafe_canary",
            split="canary",
            judge_verdict="unsafe",
            user_verdict="rejected",
        )

        result = acceptance_checklist_canary_evaluator(
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )

        failure_kinds = {failure["kind"] for failure in result["canary_failures"]}
        self.assertIn("judge_verdict", failure_kinds)

    def test_canary_unsafe_failure_still_fails_with_trained_adapter(self) -> None:
        incumbent = self._candidate("unsafe_adapter_incumbent", theta_strategy="baseline_final_answer")
        candidate = self._candidate_with_adapter(
            "unsafe_adapter_candidate",
            top_tokens=["unsafe_adapter_signal"],
        )
        episode = self._episode(
            "episode_local_signal_unsafe_adapter_canary",
            split="canary",
            judge_verdict="unsafe",
            user_verdict="rejected",
            prompt="Handle unsafe_adapter_signal without unsafe behavior.",
            acceptance_tests=["pytest tests/unsafe_adapter_signal.py"],
        )

        result = acceptance_checklist_canary_evaluator(
            conn=self.conn,
            episode=episode["raw_episode"],
            candidate=candidate,
            incumbent=incumbent,
            ordinal=0,
        )

        failure_kinds = {failure["kind"] for failure in result["canary_failures"]}
        self.assertIn("judge_verdict", failure_kinds)
        self.assertTrue(result["candidate_adapter_behavior_profile"]["adapter"]["loaded"])
