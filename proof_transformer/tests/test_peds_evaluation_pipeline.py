from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.evaluation_pipeline import (
    create_post_training_replay_job,
    infer_incumbent_for_candidate,
    plan_post_training_evaluation,
    select_replay_episodes_for_candidate,
)
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_promotion_decision,
)


class PEDSEvaluationPipelineTests(BrainTestCase):
    def _candidate(
        self,
        suffix: str,
        *,
        parent_candidate_id: str | None = None,
        mutation_target: str = "theta",
    ) -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_eval_{suffix}",
                "parent_candidate_id": parent_candidate_id,
                "mutation_target": mutation_target,
                "phi_version": "phi_v1",
                "rho_version": "rho_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_v1",
                "delta_description": f"evaluation pipeline candidate {suffix}",
                "manifest": {"suffix": suffix},
            },
        )

    def _episode(
        self,
        episode_id: str,
        *,
        split: str = "replay",
        timestamp: str = "2026-06-02T00:00:00Z",
        policy_tags: list[str] | None = None,
    ) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
                "timestamp": timestamp,
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Task {episode_id}",
                    "prompt": f"Solve {episode_id}",
                    "acceptance_tests": [f"pytest tests/{episode_id}.py"],
                    "policy_tags": policy_tags or ["general"],
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
                        "seed": 23,
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

    def _parent_and_child(self, suffix: str) -> tuple[dict, dict]:
        parent = self._candidate(f"{suffix}_parent", mutation_target="rho")
        child = self._candidate(f"{suffix}_child", parent_candidate_id=parent["candidate_id"])
        return parent, child

    def _mark_promoted(self, suffix: str) -> dict:
        candidate = self._candidate(f"{suffix}_promoted")
        incumbent = self._candidate(f"{suffix}_baseline", mutation_target="rho")
        episode = self._episode(f"episode_eval_{suffix}_promoted", timestamp="2026-06-02T00:00:00Z")
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [episode["episode_id"]],
        )
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "job_id": job["job_id"],
                "decision": "promoted",
                "reason": "Explicitly promoted incumbent for fallback inference.",
            },
        )
        return candidate

    def test_select_replay_episodes_deterministic_order(self) -> None:
        candidate = self._candidate("select_order")
        self._episode("episode_eval_order_b", timestamp="2026-06-02T00:00:02Z")
        self._episode("episode_eval_order_a", timestamp="2026-06-02T00:00:01Z")
        self._episode("episode_eval_order_c", timestamp="2026-06-02T00:00:02Z")

        selection = select_replay_episodes_for_candidate(self.conn, candidate["candidate_id"])

        self.assertEqual(
            selection["episode_ids"],
            ["episode_eval_order_a", "episode_eval_order_b", "episode_eval_order_c"],
        )
        self.assertEqual(selection["selection"]["count"], 3)

    def test_select_replay_episodes_applies_limit(self) -> None:
        candidate = self._candidate("select_limit")
        self._episode("episode_eval_limit_1", timestamp="2026-06-02T00:00:01Z")
        self._episode("episode_eval_limit_2", timestamp="2026-06-02T00:00:02Z")
        self._episode("episode_eval_limit_3", timestamp="2026-06-02T00:00:03Z")

        selection = select_replay_episodes_for_candidate(self.conn, candidate["candidate_id"], limit=2)

        self.assertEqual(selection["episode_ids"], ["episode_eval_limit_1", "episode_eval_limit_2"])
        self.assertEqual(selection["selection"]["limit"], 2)

    def test_select_replay_episodes_filters_policy_tags(self) -> None:
        candidate = self._candidate("select_tags")
        self._episode("episode_eval_tags_safe", timestamp="2026-06-02T00:00:01Z", policy_tags=["safe", "python"])
        self._episode("episode_eval_tags_ui", timestamp="2026-06-02T00:00:02Z", policy_tags=["ui"])
        self._episode("episode_eval_tags_canary", timestamp="2026-06-02T00:00:03Z", policy_tags=["canary", "safe"])

        selection = select_replay_episodes_for_candidate(
            self.conn,
            candidate["candidate_id"],
            policy_tags=["safe"],
        )

        self.assertEqual(selection["episode_ids"], ["episode_eval_tags_safe", "episode_eval_tags_canary"])
        self.assertEqual(selection["selection"]["policy_tags"], ["safe"])

    def test_infer_incumbent_prefers_parent_candidate(self) -> None:
        parent, child = self._parent_and_child("infer_parent")
        promoted = self._mark_promoted("infer_parent_fallback")

        inference = infer_incumbent_for_candidate(self.conn, child["candidate_id"])

        self.assertEqual(inference["incumbent_id"], parent["candidate_id"])
        self.assertNotEqual(inference["incumbent_id"], promoted["candidate_id"])
        self.assertEqual(inference["strategy"], "parent_candidate")

    def test_infer_incumbent_falls_back_to_latest_promoted_candidate(self) -> None:
        promoted = self._mark_promoted("infer_latest")
        candidate = self._candidate("infer_latest_childless")

        inference = infer_incumbent_for_candidate(self.conn, candidate["candidate_id"])

        self.assertEqual(inference["incumbent_id"], promoted["candidate_id"])
        self.assertEqual(inference["strategy"], "latest_promoted")

    def test_infer_incumbent_warns_when_none_available(self) -> None:
        candidate = self._candidate("infer_none")

        inference = infer_incumbent_for_candidate(self.conn, candidate["candidate_id"])

        self.assertIsNone(inference["incumbent_id"])
        self.assertEqual(inference["strategy"], "none")
        self.assertTrue(inference["warnings"])

    def test_plan_post_training_evaluation_writes_nothing(self) -> None:
        _, child = self._parent_and_child("plan_read_only")
        self._episode("episode_eval_plan", timestamp="2026-06-02T00:00:01Z")
        before_jobs = self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"]
        before_job_episodes = self.conn.execute("SELECT COUNT(*) AS count FROM replay_job_episodes").fetchone()["count"]
        before_decisions = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        plan = plan_post_training_evaluation(self.conn, child["candidate_id"])

        self.assertTrue(plan["would_create_job"])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"], before_jobs)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM replay_job_episodes").fetchone()["count"],
            before_job_episodes,
        )
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"],
            before_decisions,
        )

    def test_create_post_training_replay_job_rejects_non_draft_candidate(self) -> None:
        parent, child = self._parent_and_child("reject_non_draft")
        episode = self._episode("episode_eval_non_draft", timestamp="2026-06-02T00:00:01Z")
        job = create_replay_job(self.conn, child["candidate_id"], parent["candidate_id"], [episode["episode_id"]])
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": child["candidate_id"],
                "incumbent_id": parent["candidate_id"],
                "job_id": job["job_id"],
                "decision": "rejected",
                "reason": "Existing replay evidence already changed lifecycle.",
            },
        )

        with self.assertRaises(ValueError):
            create_post_training_replay_job(self.conn, child["candidate_id"], episode_ids=[episode["episode_id"]])

    def test_create_post_training_replay_job_rejects_missing_incumbent(self) -> None:
        candidate = self._candidate("missing_incumbent")
        self._episode("episode_eval_missing_incumbent", timestamp="2026-06-02T00:00:01Z")

        with self.assertRaises(ValueError):
            create_post_training_replay_job(self.conn, candidate["candidate_id"])

    def test_create_post_training_replay_job_rejects_no_episodes(self) -> None:
        _, child = self._parent_and_child("no_episodes")

        with self.assertRaises(ValueError):
            create_post_training_replay_job(self.conn, child["candidate_id"])

    def test_create_post_training_replay_job_creates_frozen_replay_job(self) -> None:
        parent, child = self._parent_and_child("create_job")
        first = self._episode("episode_eval_create_1", timestamp="2026-06-02T00:00:01Z")
        second = self._episode("episode_eval_create_2", timestamp="2026-06-02T00:00:02Z")

        job = create_post_training_replay_job(self.conn, child["candidate_id"], limit=2)

        self.assertEqual(job["candidate_id"], child["candidate_id"])
        self.assertEqual(job["incumbent_id"], parent["candidate_id"])
        self.assertEqual(job["episode_ids"], [first["episode_id"], second["episode_id"]])
        frozen = self.conn.execute(
            "SELECT episode_id FROM replay_job_episodes WHERE job_id = ? ORDER BY episode_order",
            (job["job_id"],),
        ).fetchall()
        self.assertEqual([row["episode_id"] for row in frozen], [first["episode_id"], second["episode_id"]])

    def test_created_job_split_is_replay(self) -> None:
        _, child = self._parent_and_child("split_replay")
        self._episode("episode_eval_split_replay", timestamp="2026-06-02T00:00:01Z")

        job = create_post_training_replay_job(self.conn, child["candidate_id"])

        self.assertEqual(job["split"], "replay")

    def test_created_job_has_no_replay_results(self) -> None:
        _, child = self._parent_and_child("no_results")
        self._episode("episode_eval_no_results", timestamp="2026-06-02T00:00:01Z")

        job = create_post_training_replay_job(self.conn, child["candidate_id"])

        count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
            (job["job_id"],),
        ).fetchone()["count"]
        self.assertEqual(count, 0)

    def test_created_job_runtime_includes_post_training_mode(self) -> None:
        parent, child = self._parent_and_child("runtime")
        self._episode("episode_eval_runtime", timestamp="2026-06-02T00:00:01Z")

        job = create_post_training_replay_job(
            self.conn,
            child["candidate_id"],
            runtime={"requested_by": "test"},
        )

        self.assertEqual(job["runtime"]["mode"], "post_training_candidate_evaluation")
        self.assertEqual(job["runtime"]["candidate_id"], child["candidate_id"])
        self.assertEqual(job["runtime"]["incumbent_id"], parent["candidate_id"])
        self.assertEqual(job["runtime"]["requested_by"], "test")
        self.assertEqual(job["runtime"]["episode_selection"]["split"], "replay")

    def test_create_post_training_replay_job_writes_no_promotion_decisions(self) -> None:
        _, child = self._parent_and_child("no_decisions")
        self._episode("episode_eval_no_decisions", timestamp="2026-06-02T00:00:01Z")
        before = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        create_post_training_replay_job(self.conn, child["candidate_id"])

        after = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]
        self.assertEqual(after, before)
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "draft_candidate")
