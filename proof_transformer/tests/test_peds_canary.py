from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.canary import (
    canary_gate,
    create_canary_job,
    evaluate_canary_job,
    run_canary_job,
)
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_replay_result,
)


class PEDSCanaryTests(BrainTestCase):
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

    def _episode(self, episode_id: str, *, split: str) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
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
                        "seed": 17,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [{"uri": "file://app.py", "kind": "file", "tokens": 120}],
                    "actions": [{"step": 1, "kind": "read", "args": {"path": "app.py"}, "exit_code": 0}],
                    "patch": "diff --git a/app.py b/app.py",
                    "final_message": "Patched file.",
                },
                "outcome": {"judge_verdict": "pass", "user_verdict": "accepted"},
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

    def _canary_job(self, suffix: str, episode_ids: list[str]) -> tuple[dict, dict, str]:
        candidate = self._candidate(f"{suffix}_candidate")
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        job_id = create_canary_job(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            episode_ids=episode_ids,
        )
        return candidate, incumbent, job_id

    def _record_job_results(self, job_id: str, episode_ids: list[str], deltas: list[float], *, canary_failure_at: int | None = None) -> None:
        for index, episode_id in enumerate(episode_ids):
            incumbent_score = 0.70 + (index * 0.01)
            record_replay_result(
                self.conn,
                {
                    "job_id": job_id,
                    "episode_id": episode_id,
                    "candidate_score": incumbent_score + deltas[index],
                    "incumbent_score": incumbent_score,
                    "judge_verdict": "pass",
                    "canary_failures": ["secret_read"] if canary_failure_at == index else [],
                },
            )

    def _aggregate(self, **overrides: object) -> dict:
        aggregate = {
            "split": "canary",
            "complete": True,
            "result_count": 3,
            "frozen_episode_count": 3,
            "mean_delta": 0.02,
            "regression_rate": 0.0,
            "canary_failures": 0,
            "hard_safety_violations": 0,
            "token_ratio": None,
            "latency_ratio": None,
        }
        aggregate.update(overrides)
        return aggregate

    def test_create_canary_job_freezes_ordered_canary_episode_set(self) -> None:
        first = self._episode("canary_create_1", split="canary")
        second = self._episode("canary_create_2", split="canary")
        third = self._episode("canary_create_3", split="canary")
        _, _, job_id = self._canary_job("create", [third["episode_id"], first["episode_id"], second["episode_id"]])

        job_row = self.conn.execute("SELECT split FROM replay_jobs WHERE job_id = ?", (job_id,)).fetchone()
        frozen = self.conn.execute(
            "SELECT episode_id FROM replay_job_episodes WHERE job_id = ? ORDER BY episode_order",
            (job_id,),
        ).fetchall()
        self.assertEqual(job_row["split"], "canary")
        self.assertEqual([row["episode_id"] for row in frozen], [third["episode_id"], first["episode_id"], second["episode_id"]])

    def test_run_canary_job_executes_all_frozen_canary_episodes(self) -> None:
        first = self._episode("canary_run_1", split="canary")
        second = self._episode("canary_run_2", split="canary")
        third = self._episode("canary_run_3", split="canary")
        _, _, job_id = self._canary_job("run", [first["episode_id"], second["episode_id"], third["episode_id"]])

        summary = run_canary_job(
            self.conn,
            job_id,
            lambda **kwargs: {
                "candidate_score": 0.82 + (kwargs["ordinal"] * 0.01),
                "incumbent_score": 0.79 + (kwargs["ordinal"] * 0.01),
            },
        )
        self.assertEqual(summary["episodes_run"], 3)
        self.assertEqual(summary["results_written"], 3)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["gate"]["decision"], "canary_passed")

    def test_run_canary_job_refuses_missing_job_id(self) -> None:
        with self.assertRaises(ValueError):
            run_canary_job(self.conn, "missing_canary_job", lambda **_: {"candidate_score": 1.0, "incumbent_score": 0.0})

    def test_run_canary_job_refuses_non_canary_job(self) -> None:
        candidate = self._candidate("replay_candidate")
        incumbent = self._candidate("replay_incumbent", mutation_target="rho")
        first = self._episode("replay_episode_1", split="replay")
        second = self._episode("replay_episode_2", split="replay")
        third = self._episode("replay_episode_3", split="replay")
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [first["episode_id"], second["episode_id"], third["episode_id"]],
        )

        with self.assertRaises(ValueError):
            run_canary_job(self.conn, job["job_id"], lambda **_: {"candidate_score": 1.0, "incumbent_score": 0.0})

    def test_canary_gate_passes_clean_candidate(self) -> None:
        gate = canary_gate(self._aggregate())
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["decision"], "canary_passed")

    def test_canary_gate_rejects_any_canary_failure_under_default_thresholds(self) -> None:
        gate = canary_gate(self._aggregate(canary_failures=1, mean_delta=0.20))
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["decision"], "rejected")
        self.assertIn("canary_failures > max_canary_failures", gate["reason"])

    def test_canary_gate_rejects_high_regression_rate(self) -> None:
        gate = canary_gate(self._aggregate(regression_rate=0.10))
        self.assertFalse(gate["passed"])
        self.assertIn("regression_rate > max_regression_rate", gate["reason"])

    def test_canary_gate_tolerates_missing_token_and_latency_ratios(self) -> None:
        gate = canary_gate(self._aggregate(token_ratio=None, latency_ratio=None))
        self.assertTrue(gate["passed"])

    def test_evaluate_canary_job_without_write_decision_does_not_append_promotion_decisions(self) -> None:
        first = self._episode("canary_eval_none_1", split="canary")
        second = self._episode("canary_eval_none_2", split="canary")
        third = self._episode("canary_eval_none_3", split="canary")
        candidate, _, job_id = self._canary_job("eval_none", [first["episode_id"], second["episode_id"], third["episode_id"]])
        self._record_job_results(job_id, [first["episode_id"], second["episode_id"], third["episode_id"]], [0.03, 0.03, 0.03])

        result = evaluate_canary_job(self.conn, job_id, write_decision=False)
        self.assertEqual(result["gate"]["decision"], "canary_passed")
        self.assertIsNone(result["promotion_decision_id"])
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "draft_candidate")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions WHERE job_id = ?", (job_id,)).fetchone()["count"],
            0,
        )

    def test_evaluate_canary_job_with_write_decision_appends_canary_passed(self) -> None:
        first = self._episode("canary_eval_pass_1", split="canary")
        second = self._episode("canary_eval_pass_2", split="canary")
        third = self._episode("canary_eval_pass_3", split="canary")
        candidate, _, job_id = self._canary_job("eval_pass", [first["episode_id"], second["episode_id"], third["episode_id"]])
        self._record_job_results(job_id, [first["episode_id"], second["episode_id"], third["episode_id"]], [0.03, 0.03, 0.03])

        result = evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(result["gate"]["decision"], "canary_passed")
        self.assertIsNotNone(result["promotion_decision_id"])
        decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (result["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(decision["decision"], "canary_passed")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "canary_passed")

    def test_evaluate_canary_job_with_write_decision_appends_rejected_when_gate_fails(self) -> None:
        first = self._episode("canary_eval_fail_1", split="canary")
        second = self._episode("canary_eval_fail_2", split="canary")
        third = self._episode("canary_eval_fail_3", split="canary")
        candidate, _, job_id = self._canary_job("eval_fail", [first["episode_id"], second["episode_id"], third["episode_id"]])
        self._record_job_results(
            job_id,
            [first["episode_id"], second["episode_id"], third["episode_id"]],
            [0.03, 0.03, 0.03],
            canary_failure_at=1,
        )

        result = evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(result["gate"]["decision"], "rejected")
        self.assertIsNotNone(result["promotion_decision_id"])
        decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (result["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(decision["decision"], "rejected")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "rejected")

    def test_candidate_current_state_updates_only_through_appended_promotion_decisions(self) -> None:
        first = self._episode("canary_state_1", split="canary")
        second = self._episode("canary_state_2", split="canary")
        third = self._episode("canary_state_3", split="canary")
        candidate, _, job_id = self._canary_job("state", [first["episode_id"], second["episode_id"], third["episode_id"]])
        self._record_job_results(job_id, [first["episode_id"], second["episode_id"], third["episode_id"]], [0.02, 0.02, 0.02])

        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "draft_candidate")
        evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "canary_passed")

    def test_canary_failure_overrides_positive_mean_delta(self) -> None:
        gate = canary_gate(self._aggregate(canary_failures=1, mean_delta=0.50, regression_rate=0.0))
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["decision"], "rejected")
        self.assertIn("canary_failures > max_canary_failures", gate["reason"])
