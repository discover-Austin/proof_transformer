from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.replay import load_replay_job_bundle, run_replay_job
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
)


class PEDSReplayTests(BrainTestCase):
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
                        "seed": 13,
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

    def _replay_job(self, suffix: str, episode_ids: list[str]) -> tuple[dict, dict, dict]:
        candidate = self._candidate(f"{suffix}_candidate")
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            episode_ids,
        )
        return candidate, incumbent, job

    def test_run_replay_job_executes_all_frozen_episodes_and_writes_replay_results(self) -> None:
        first = self._replay_episode("episode_run_1")
        second = self._replay_episode("episode_run_2")
        third = self._replay_episode("episode_run_3")
        _, _, job = self._replay_job("run_all", [first["episode_id"], second["episode_id"], third["episode_id"]])

        summary = run_replay_job(
            self.conn,
            job["job_id"],
            lambda **kwargs: {
                "candidate_score": 0.82 + (kwargs["ordinal"] * 0.01),
                "incumbent_score": 0.70 + (kwargs["ordinal"] * 0.01),
            },
        )

        self.assertEqual(summary["job_id"], job["job_id"])
        self.assertEqual(summary["episodes_run"], 3)
        self.assertEqual(summary["results_written"], 3)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["aggregate"]["result_count"], 3)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?", (job["job_id"],)).fetchone()["count"],
            3,
        )
        job_row = self.conn.execute("SELECT status, started_at, finished_at FROM replay_jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
        self.assertEqual(job_row["status"], "completed")
        self.assertIsNotNone(job_row["started_at"])
        self.assertIsNotNone(job_row["finished_at"])

    def test_run_replay_job_preserves_frozen_episode_order(self) -> None:
        first = self._replay_episode("episode_order_1")
        second = self._replay_episode("episode_order_2")
        third = self._replay_episode("episode_order_3")
        expected_order = [third["episode_id"], first["episode_id"], second["episode_id"]]
        _, _, job = self._replay_job("order", expected_order)

        bundle = load_replay_job_bundle(self.conn, job["job_id"])
        self.assertEqual([entry["episode_id"] for entry in bundle["episodes"]], expected_order)

        seen: list[tuple[int, str]] = []

        def evaluator(*, episode: dict, ordinal: int, **_: dict) -> dict:
            seen.append((ordinal, episode["episode_id"]))
            return {"candidate_score": 0.9, "incumbent_score": 0.8}

        run_replay_job(self.conn, job["job_id"], evaluator)
        self.assertEqual(seen, list(enumerate(expected_order)))

    def test_run_replay_job_refuses_missing_job_id(self) -> None:
        with self.assertRaises(ValueError):
            run_replay_job(self.conn, "missing_job", lambda **_: {"candidate_score": 1.0, "incumbent_score": 0.0})

    def test_run_replay_job_refuses_job_with_no_frozen_episodes(self) -> None:
        candidate = self._candidate("orphan_candidate")
        incumbent = self._candidate("orphan_incumbent", mutation_target="rho")

        # Direct SQL is required here because the store helper correctly rejects empty replay sets.
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO replay_jobs(
                    job_id, created_at, started_at, finished_at, split,
                    candidate_id, incumbent_id, evaluator_version, runtime_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "job_empty",
                    "2026-06-02T00:00:00Z",
                    None,
                    None,
                    "replay",
                    candidate["candidate_id"],
                    incumbent["candidate_id"],
                    "eval_v1",
                    "{}",
                    "queued",
                ),
            )

        with self.assertRaises(ValueError):
            run_replay_job(self.conn, "job_empty", lambda **_: {"candidate_score": 1.0, "incumbent_score": 0.0})

    def test_run_replay_job_refuses_to_rerun_job_with_existing_results(self) -> None:
        first = self._replay_episode("episode_rerun_1")
        second = self._replay_episode("episode_rerun_2")
        third = self._replay_episode("episode_rerun_3")
        _, _, job = self._replay_job("rerun", [first["episode_id"], second["episode_id"], third["episode_id"]])

        run_replay_job(self.conn, job["job_id"], lambda **_: {"candidate_score": 0.8, "incumbent_score": 0.7})
        with self.assertRaises(ValueError):
            run_replay_job(self.conn, job["job_id"], lambda **_: {"candidate_score": 0.8, "incumbent_score": 0.7})

    def test_run_replay_job_rejects_missing_candidate_score(self) -> None:
        first = self._replay_episode("episode_missing_candidate_1")
        second = self._replay_episode("episode_missing_candidate_2")
        third = self._replay_episode("episode_missing_candidate_3")
        _, _, job = self._replay_job(
            "missing_candidate_score",
            [first["episode_id"], second["episode_id"], third["episode_id"]],
        )

        with self.assertRaises(ValueError):
            run_replay_job(self.conn, job["job_id"], lambda **_: {"incumbent_score": 0.7})
        job_row = self.conn.execute("SELECT status FROM replay_jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
        self.assertEqual(job_row["status"], "failed")

    def test_run_replay_job_rejects_missing_incumbent_score(self) -> None:
        first = self._replay_episode("episode_missing_incumbent_1")
        second = self._replay_episode("episode_missing_incumbent_2")
        third = self._replay_episode("episode_missing_incumbent_3")
        _, _, job = self._replay_job(
            "missing_incumbent_score",
            [first["episode_id"], second["episode_id"], third["episode_id"]],
        )

        with self.assertRaises(ValueError):
            run_replay_job(self.conn, job["job_id"], lambda **_: {"candidate_score": 0.8})
        job_row = self.conn.execute("SELECT status FROM replay_jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
        self.assertEqual(job_row["status"], "failed")

    def test_run_replay_job_rejects_non_numeric_scores(self) -> None:
        first = self._replay_episode("episode_bad_score_1")
        second = self._replay_episode("episode_bad_score_2")
        third = self._replay_episode("episode_bad_score_3")
        _, _, job = self._replay_job("bad_score", [first["episode_id"], second["episode_id"], third["episode_id"]])

        with self.assertRaises(ValueError):
            run_replay_job(self.conn, job["job_id"], lambda **_: {"candidate_score": "bad", "incumbent_score": 0.7})
        job_row = self.conn.execute("SELECT status FROM replay_jobs WHERE job_id = ?", (job["job_id"],)).fetchone()
        self.assertEqual(job_row["status"], "failed")

    def test_run_replay_job_without_write_decision_does_not_append_promotion_decisions(self) -> None:
        first = self._replay_episode("episode_no_decision_1")
        second = self._replay_episode("episode_no_decision_2")
        third = self._replay_episode("episode_no_decision_3")
        _, _, job = self._replay_job(
            "no_decision",
            [first["episode_id"], second["episode_id"], third["episode_id"]],
        )

        summary = run_replay_job(self.conn, job["job_id"], lambda **_: {"candidate_score": 0.86, "incumbent_score": 0.72})
        self.assertIsNone(summary["promotion_decision_id"])
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions WHERE job_id = ?", (job["job_id"],)).fetchone()["count"],
            0,
        )

    def test_run_replay_job_with_write_decision_appends_promotion_decisions(self) -> None:
        first = self._replay_episode("episode_with_decision_1")
        second = self._replay_episode("episode_with_decision_2")
        third = self._replay_episode("episode_with_decision_3")
        _, _, job = self._replay_job(
            "with_decision",
            [first["episode_id"], second["episode_id"], third["episode_id"]],
        )

        summary = run_replay_job(
            self.conn,
            job["job_id"],
            lambda **_: {"candidate_score": 0.86, "incumbent_score": 0.72},
            write_decision=True,
        )
        self.assertIsNotNone(summary["promotion_decision_id"])
        decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (summary["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(decision["decision"], "replay_winner")

    def test_run_replay_job_with_strong_candidate_updates_derived_candidate_state(self) -> None:
        first = self._replay_episode("episode_strong_1")
        second = self._replay_episode("episode_strong_2")
        third = self._replay_episode("episode_strong_3")
        candidate, _, job = self._replay_job("strong", [first["episode_id"], second["episode_id"], third["episode_id"]])

        summary = run_replay_job(
            self.conn,
            job["job_id"],
            lambda **_: {"candidate_score": 0.91, "incumbent_score": 0.74},
            write_decision=True,
        )
        self.assertEqual(summary["gate"]["decision"], "replay_winner")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "replay_winner")

    def test_run_replay_job_with_weak_candidate_writes_rejected_decision(self) -> None:
        first = self._replay_episode("episode_weak_1")
        second = self._replay_episode("episode_weak_2")
        third = self._replay_episode("episode_weak_3")
        candidate, _, job = self._replay_job("weak", [first["episode_id"], second["episode_id"], third["episode_id"]])

        summary = run_replay_job(
            self.conn,
            job["job_id"],
            lambda **_: {"candidate_score": 0.65, "incumbent_score": 0.72},
            write_decision=True,
        )
        self.assertEqual(summary["gate"]["decision"], "rejected")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "rejected")

    def test_run_replay_job_reflects_canary_failures_in_gate_behavior(self) -> None:
        first = self._replay_episode("episode_canary_1")
        second = self._replay_episode("episode_canary_2")
        third = self._replay_episode("episode_canary_3")
        _, _, job = self._replay_job("canary", [first["episode_id"], second["episode_id"], third["episode_id"]])

        summary = run_replay_job(
            self.conn,
            job["job_id"],
            lambda **kwargs: {
                "candidate_score": 0.91,
                "incumbent_score": 0.74,
                "canary_failures": 2 if kwargs["ordinal"] == 1 else None,
            },
            write_decision=True,
        )
        self.assertEqual(summary["gate"]["decision"], "rejected")
        self.assertIn("canary failures observed", summary["gate"]["reason"])
        self.assertEqual(summary["aggregate"]["canary_failures"], 2)
