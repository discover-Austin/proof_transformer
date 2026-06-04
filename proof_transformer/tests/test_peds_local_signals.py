from __future__ import annotations

from typing import Any

from common import BrainTestCase
from proof_transformer.peds.local_signals import (
    acceptance_checklist_canary_evaluator,
    acceptance_checklist_replay_evaluator,
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
    ) -> dict:
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
                "manifest_json": {
                    "suffix": suffix,
                    "theta_strategy": theta_strategy,
                },
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
                    "prompt": f"Handle local signal task {episode_id}",
                    "acceptance_tests": [f"pytest tests/{episode_id}.py"],
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
