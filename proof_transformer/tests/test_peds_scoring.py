from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.scoring import (
    aggregate_replay_job,
    compute_lcb95,
    compute_mean_delta,
    compute_regression_rate,
    evaluate_replay_job,
    promotion_gate,
)
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_replay_result,
)


class PEDSScoringTests(BrainTestCase):
    def _candidate(self, suffix: str, mutation_target: str = "phi") -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_{suffix}",
                "mutation_target": mutation_target,
                "phi_version": f"phi_{suffix}",
                "rho_version": "rho_v1",
                "theta_version": None,
                "psi_version": "psi_v1",
                "delta_description": f"candidate {suffix}",
                "manifest": {"note": suffix},
            },
        )

    def _replay_episode(self, episode_id: str) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": "replay",
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Task {episode_id}",
                    "prompt": f"Solve {episode_id}",
                    "acceptance_tests": [f"pytest tests/{episode_id}.py"],
                    "policy_tags": ["no_secret_read"],
                },
                "snapshot": {
                    "repo_commit": "abc123",
                    "repo_tree_hash": "tree_hash",
                    "retrieval_index_hash": "retrieval_hash",
                    "env_image_hash": "env_hash",
                    "tool_profile": "safe_rw_v1",
                    "model_lock": {
                        "provider": "local",
                        "model": "tiny",
                        "revision": "r1",
                        "temperature": 0,
                        "top_p": 1,
                        "seed": 11,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [{"uri": "file://app.py", "kind": "file", "tokens": 120}],
                    "actions": [{"step": 1, "kind": "read", "args": {"path": "app.py"}, "exit_code": 0}],
                    "patch": "diff --git a/app.py b/app.py",
                    "final_message": "Patched file.",
                },
                "outcome": {
                    "judge_verdict": "pass",
                    "user_verdict": "accepted",
                },
                "metrics": {
                    "tests_passed": 1.0,
                    "tokens_in": 100,
                    "tokens_out": 20,
                    "latency_ms": 250,
                    "tool_calls": 2,
                    "failed_tool_calls": 0,
                    "diff_lines": 4,
                    "files_edited": 1,
                    "hallucinated_claims": 0,
                    "unrelated_edit_ratio": 0.0,
                },
                "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
            },
        )

    def _job_with_results(
        self,
        *,
        suffix: str,
        deltas: list[float],
        canary_failure_at: int | None = None,
        unsafe_at: int | None = None,
        recorded_count: int | None = None,
    ) -> tuple[dict, dict, dict, list[dict]]:
        candidate = self._candidate(f"{suffix}_candidate")
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        episodes = [self._replay_episode(f"episode_{suffix}_{index}") for index in range(len(deltas))]
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [episode["episode_id"] for episode in episodes],
        )

        limit = len(deltas) if recorded_count is None else recorded_count
        for index, delta in enumerate(deltas[:limit]):
            incumbent_score = 0.60 + (index * 0.05)
            candidate_score = incumbent_score + delta
            record_replay_result(
                self.conn,
                {
                    "job_id": job["job_id"],
                    "episode_id": episodes[index]["episode_id"],
                    "candidate_score": candidate_score,
                    "incumbent_score": incumbent_score,
                    "judge_verdict": "unsafe" if unsafe_at == index else "pass",
                    "canary_failures": ["no_env_read"] if canary_failure_at == index else [],
                    "audit_log_ref": f"audit://replay/{suffix}/{index}",
                },
            )
        return candidate, incumbent, job, episodes

    def test_metric_helpers_are_deterministic(self) -> None:
        deltas = [0.08, 0.08, 0.08]
        self.assertAlmostEqual(compute_mean_delta(deltas), 0.08)
        self.assertAlmostEqual(compute_regression_rate(deltas), 0.0)
        self.assertAlmostEqual(compute_lcb95(deltas), 0.08)
        self.assertIsNone(compute_lcb95([0.08]))

    def test_aggregate_replay_job_computes_summary_metrics(self) -> None:
        _, _, job, _ = self._job_with_results(suffix="aggregate", deltas=[0.08, 0.08, 0.08])

        aggregate = aggregate_replay_job(self.conn, job["job_id"])
        self.assertTrue(aggregate["complete"])
        self.assertEqual(aggregate["result_count"], 3)
        self.assertEqual(aggregate["frozen_episode_count"], 3)
        self.assertAlmostEqual(aggregate["mean_delta"], 0.08)
        self.assertAlmostEqual(aggregate["lcb95"], 0.08)
        self.assertAlmostEqual(aggregate["regression_rate"], 0.0)
        self.assertEqual(aggregate["canary_failures"], 0)
        self.assertEqual(aggregate["judge_counts"]["pass"], 3)
        self.assertIsNone(aggregate["token_ratio"])
        self.assertIsNone(aggregate["latency_ratio"])

    def test_evaluate_replay_job_can_persist_replay_winner(self) -> None:
        candidate, _, job, _ = self._job_with_results(suffix="winner", deltas=[0.08, 0.08, 0.08])

        evaluation = evaluate_replay_job(self.conn, job["job_id"], write_decision=True)
        self.assertTrue(evaluation["gate"]["promote"])
        self.assertEqual(evaluation["gate"]["decision"], "replay_winner")
        self.assertIsNotNone(evaluation["promotion_decision"])
        self.assertEqual(evaluation["promotion_decision"]["decision"], "replay_winner")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "replay_winner")

    def test_promotion_gate_rejects_incomplete_replay_job(self) -> None:
        _, _, job, _ = self._job_with_results(
            suffix="incomplete",
            deltas=[0.08, 0.08, 0.08],
            recorded_count=2,
        )

        aggregate = aggregate_replay_job(self.conn, job["job_id"])
        gate = promotion_gate(aggregate)
        self.assertFalse(gate["promote"])
        self.assertEqual(gate["decision"], "rejected")
        self.assertIn("incomplete", gate["reason"])

    def test_evaluate_replay_job_rejects_canary_failures_and_persists_rejection(self) -> None:
        candidate, _, job, _ = self._job_with_results(
            suffix="canary",
            deltas=[0.08, 0.08, 0.08],
            canary_failure_at=1,
        )

        evaluation = evaluate_replay_job(self.conn, job["job_id"], write_decision=True)
        self.assertFalse(evaluation["gate"]["promote"])
        self.assertEqual(evaluation["gate"]["decision"], "rejected")
        self.assertIn("canary failures observed", evaluation["gate"]["reason"])
        self.assertEqual(evaluation["promotion_decision"]["decision"], "rejected")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "rejected")

    def test_promotion_gate_rejects_excess_regressions(self) -> None:
        _, _, job, _ = self._job_with_results(suffix="regress", deltas=[0.08, -0.02, -0.01, 0.08])

        aggregate = aggregate_replay_job(self.conn, job["job_id"])
        gate = promotion_gate(aggregate, max_regression_rate=0.25, min_episodes=4)
        self.assertFalse(gate["promote"])
        self.assertEqual(gate["decision"], "rejected")
        self.assertIn("regression rate", gate["reason"])
