from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_promotion_decision,
    record_replay_result,
)


class PEDSStoreTests(BrainTestCase):
    def _candidate(self, *, suffix: str, mutation_target: str = "phi") -> dict:
        return create_candidate(
            self.conn,
            {
                "candidate_id": f"candidate_{suffix}",
                "parent_candidate_id": None,
                "mutation_target": mutation_target,
                "phi_version": f"phi_{suffix}",
                "rho_version": "rho_v1",
                "theta_version": None,
                "psi_version": "psi_v1",
                "delta_description": f"candidate {suffix}",
                "manifest": {"note": suffix},
            },
        )

    def _episode(self, *, episode_id: str, split: str, candidate_id: str | None = None) -> dict:
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
                        "seed": 7,
                        "timeout_s": 120,
                    },
                },
                "candidate": {"candidate_id": candidate_id} if candidate_id else {},
                "run": {
                    "context_selected": [{"uri": "file://app.py", "kind": "file", "tokens": 120}],
                    "actions": [{"step": 1, "kind": "read", "args": {"path": "app.py"}, "exit_code": 0}],
                    "patch": "diff --git a/app.py b/app.py",
                    "final_message": "Patched file.",
                },
                "outcome": {
                    "judge_verdict": "pass",
                    "user_verdict": "accepted",
                    "failure_reason": None,
                    "repair_trace": None,
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

    def test_create_candidate_returns_manifest_and_draft_state(self) -> None:
        candidate = self._candidate(suffix="alpha")
        self.assertEqual(candidate["candidate_id"], "candidate_alpha")
        self.assertEqual(candidate["current_state"], "draft_candidate")
        self.assertEqual(candidate["manifest"]["candidate_id"], "candidate_alpha")
        self.assertEqual(get_candidate_current_state(self.conn, "candidate_alpha"), "draft_candidate")

    def test_record_episode_normalizes_nested_protocol_shape(self) -> None:
        candidate = self._candidate(suffix="beta")
        episode = self._episode(episode_id="episode_beta", split="online", candidate_id=candidate["candidate_id"])
        self.assertEqual(episode["episode_id"], "episode_beta")
        self.assertEqual(episode["candidate_id"], candidate["candidate_id"])
        self.assertEqual(episode["task_title"], "Task episode_beta")
        self.assertEqual(episode["model_lock"]["model"], "tiny")
        self.assertEqual(episode["context"][0]["uri"], "file://app.py")
        self.assertEqual(episode["actions"][0]["kind"], "read")
        self.assertEqual(episode["metrics"]["tool_calls"], 2)
        self.assertFalse(episode["privacy"]["contains_secret"])

    def test_create_replay_job_freezes_order_and_record_replay_result(self) -> None:
        candidate = self._candidate(suffix="candidate")
        incumbent = self._candidate(suffix="incumbent", mutation_target="rho")
        first = self._episode(episode_id="episode_r1", split="replay")
        second = self._episode(episode_id="episode_r2", split="replay")

        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [second["episode_id"], first["episode_id"]],
            runtime={"network": "off", "max_parallel": 2},
        )
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["episode_ids"], ["episode_r2", "episode_r1"])

        frozen = self.conn.execute(
            "SELECT episode_id FROM replay_job_episodes WHERE job_id = ? ORDER BY episode_order",
            (job["job_id"],),
        ).fetchall()
        self.assertEqual([row["episode_id"] for row in frozen], ["episode_r2", "episode_r1"])

        result = record_replay_result(
            self.conn,
            {
                "job_id": job["job_id"],
                "episode_id": second["episode_id"],
                "candidate_score": 0.84,
                "incumbent_score": 0.71,
                "judge_verdict": "pass",
                "canary_failures": [],
                "audit_log_ref": "audit://replay/1",
            },
        )
        self.assertAlmostEqual(result["delta"], 0.13)
        self.assertEqual(result["canary_failures"], [])

    def test_record_promotion_decision_appends_state_transitions(self) -> None:
        candidate = self._candidate(suffix="winner")
        incumbent = self._candidate(suffix="baseline")
        episode = self._episode(episode_id="episode_promote", split="replay")
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [episode["episode_id"]],
        )

        decision = record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "job_id": job["job_id"],
                "decision": "replay_winner",
                "mean_delta": 0.07,
                "lcb95": 0.03,
                "regression_rate": 0.10,
                "canary_failures": 0,
                "token_ratio": 0.92,
                "latency_ratio": 0.88,
                "reason": "Replay confidence exceeded epsilon.",
            },
        )
        self.assertEqual(decision["current_state"], "replay_winner")

        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "job_id": job["job_id"],
                "decision": "promoted",
                "mean_delta": 0.07,
                "lcb95": 0.03,
                "regression_rate": 0.10,
                "canary_failures": 0,
                "token_ratio": 0.92,
                "latency_ratio": 0.88,
                "reason": "Shadow deployment remained clean.",
            },
        )
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "promoted")

    def test_validation_rejects_duplicate_or_mismatched_replay_inputs(self) -> None:
        candidate = self._candidate(suffix="dup_a")
        incumbent = self._candidate(suffix="dup_b")
        replay_episode = self._episode(episode_id="episode_valid_replay", split="replay")
        online_episode = self._episode(episode_id="episode_online", split="online")

        with self.assertRaises(ValueError):
            create_replay_job(
                self.conn,
                candidate["candidate_id"],
                incumbent["candidate_id"],
                [replay_episode["episode_id"], replay_episode["episode_id"]],
            )

        with self.assertRaises(ValueError):
            create_replay_job(
                self.conn,
                candidate["candidate_id"],
                incumbent["candidate_id"],
                [online_episode["episode_id"]],
            )

        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [replay_episode["episode_id"]],
        )
        with self.assertRaises(ValueError):
            record_replay_result(
                self.conn,
                {
                    "job_id": job["job_id"],
                    "episode_id": online_episode["episode_id"],
                    "candidate_score": 0.6,
                    "incumbent_score": 0.5,
                    "judge_verdict": "pass",
                },
            )
