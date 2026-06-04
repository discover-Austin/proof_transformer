from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.canary import create_canary_job, evaluate_canary_job
from proof_transformer.peds.shadow import (
    complete_shadow_run,
    record_shadow_event,
    shadow_gate,
    start_shadow_run,
    summarize_shadow_run,
)
from proof_transformer.peds.store import create_candidate, get_candidate_current_state, record_episode, record_replay_result


class PEDSShadowTests(BrainTestCase):
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

    def _canary_episode(self, episode_id: str) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": "canary",
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Task {episode_id}",
                    "prompt": f"Solve {episode_id}",
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
                        "seed": 19,
                        "timeout_s": 120,
                    },
                },
                "run": {
                    "context_selected": [],
                    "actions": [],
                    "final_message": "Observed.",
                },
                "outcome": {"judge_verdict": "pass", "user_verdict": "accepted"},
                "metrics": {"tests_passed": 1.0},
                "privacy": {"contains_secret": False, "redaction_profile": "default_v1"},
            },
        )

    def _shadow_ready_pair(self, suffix: str) -> tuple[dict, dict]:
        candidate = self._candidate(f"{suffix}_candidate")
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        canary_episode = self._canary_episode(f"canary_ready_{suffix}")
        job_id = create_canary_job(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            episode_ids=[canary_episode["episode_id"]],
        )
        record_replay_result(
            self.conn,
            {
                "job_id": job_id,
                "episode_id": canary_episode["episode_id"],
                "candidate_score": 0.82,
                "incumbent_score": 0.78,
                "judge_verdict": "pass",
                "canary_failures": [],
            },
        )
        evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "canary_passed")
        return candidate, incumbent

    def _active_shadow_run(self, suffix: str, *, write_decision: bool = False) -> tuple[dict, dict, dict]:
        candidate, incumbent = self._shadow_ready_pair(suffix)
        run = start_shadow_run(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            thresholds={"min_events": 3, "max_divergence_rate": 0.5, "min_accepted_by_incumbent_rate": 0.5},
            write_decision=write_decision,
        )
        return candidate, incumbent, run

    def test_start_shadow_run_creates_active_shadow_run(self) -> None:
        candidate, incumbent, run = self._active_shadow_run("start_active")
        self.assertEqual(run["status"], "active")
        self.assertEqual(run["candidate_id"], candidate["candidate_id"])
        self.assertEqual(run["incumbent_id"], incumbent["candidate_id"])

    def test_start_shadow_run_optionally_appends_shadow_started(self) -> None:
        candidate, _, run = self._active_shadow_run("start_decision", write_decision=True)
        self.assertIsNotNone(run["promotion_decision_id"])
        decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (run["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(decision["decision"], "shadow_started")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_started")

    def test_start_shadow_run_does_not_mutate_candidate_current_state_directly(self) -> None:
        candidate, _, run = self._active_shadow_run("start_no_decision", write_decision=False)
        self.assertIsNone(run["promotion_decision_id"])
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "canary_passed")

    def test_record_shadow_event_appends_event_to_active_shadow_run(self) -> None:
        _, _, run = self._active_shadow_run("event_append")
        event = record_shadow_event(
            self.conn,
            shadow_run_id=run["shadow_run_id"],
            input={"task": "review"},
            incumbent_output={"answer": "incumbent"},
            candidate_output={"answer": "candidate"},
            divergence={"kind": "text"},
            accepted_by_incumbent=True,
        )
        self.assertEqual(event["ordinal"], 0)
        self.assertEqual(event["input"]["task"], "review")

    def test_record_shadow_event_refuses_missing_shadow_run_id(self) -> None:
        with self.assertRaises(ValueError):
            record_shadow_event(
                self.conn,
                shadow_run_id="missing_run",
                input={},
                incumbent_output={},
                candidate_output={},
            )

    def test_record_shadow_event_refuses_completed_shadow_run(self) -> None:
        _, _, run = self._active_shadow_run("completed_run")
        complete_shadow_run(self.conn, run["shadow_run_id"], thresholds={"min_events": 0}, write_decision=False)
        with self.assertRaises(ValueError):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={},
                incumbent_output={},
                candidate_output={},
            )

    def test_candidate_would_act_is_stored_as_proposal_only(self) -> None:
        _, _, run = self._active_shadow_run("proposal_only")
        event = record_shadow_event(
            self.conn,
            shadow_run_id=run["shadow_run_id"],
            input={"task": "patch"},
            incumbent_output={"answer": "no-op"},
            candidate_output={"answer": "propose"},
            candidate_would_act={"edit": "app.py"},
            incumbent_did_act=None,
        )
        self.assertEqual(event["candidate_would_act"], {"edit": "app.py"})
        self.assertIsNone(event["incumbent_did_act"])

    def test_summarize_shadow_run_computes_n_events(self) -> None:
        _, _, run = self._active_shadow_run("summary_events")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
            )
        summary = summarize_shadow_run(self.conn, run["shadow_run_id"])
        self.assertEqual(summary["n_events"], 3)

    def test_summarize_shadow_run_computes_divergence_rate(self) -> None:
        _, _, run = self._active_shadow_run("summary_divergence")
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={}, divergence={"kind": "a"})
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={})
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={}, divergence={"kind": "b"})
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={})
        summary = summarize_shadow_run(self.conn, run["shadow_run_id"])
        self.assertAlmostEqual(summary["divergence_rate"], 0.5)

    def test_summarize_shadow_run_computes_accepted_by_incumbent_rate(self) -> None:
        _, _, run = self._active_shadow_run("summary_acceptance")
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={}, accepted_by_incumbent=True)
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={}, accepted_by_incumbent=False)
        record_shadow_event(self.conn, shadow_run_id=run["shadow_run_id"], input={}, incumbent_output={}, candidate_output={}, accepted_by_incumbent=None)
        summary = summarize_shadow_run(self.conn, run["shadow_run_id"])
        self.assertAlmostEqual(summary["accepted_by_incumbent_rate"], 0.5)

    def test_shadow_gate_rejects_insufficient_events(self) -> None:
        gate = shadow_gate({"n_events": 2, "divergence_rate": 0.0, "accepted_by_incumbent_rate": 1.0})
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["decision"], "shadow_failed")

    def test_shadow_gate_rejects_excessive_divergence_rate(self) -> None:
        gate = shadow_gate({"n_events": 10, "divergence_rate": 0.4, "accepted_by_incumbent_rate": 1.0})
        self.assertFalse(gate["passed"])
        self.assertIn("divergence_rate > max_divergence_rate", gate["reason"])

    def test_shadow_gate_passes_sufficient_clean_shadow_evidence(self) -> None:
        gate = shadow_gate({"n_events": 12, "divergence_rate": 0.1, "accepted_by_incumbent_rate": 0.9})
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["decision"], "shadow_passed")

    def test_complete_shadow_run_marks_run_completed(self) -> None:
        _, _, run = self._active_shadow_run("complete_status")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                accepted_by_incumbent=True,
            )
        result = complete_shadow_run(self.conn, run["shadow_run_id"], write_decision=False)
        self.assertEqual(result["shadow_run"]["status"], "completed")
        self.assertIsNotNone(result["shadow_run"]["completed_at"])

    def test_complete_shadow_run_with_write_decision_appends_shadow_passed_or_shadow_failed(self) -> None:
        _, _, pass_run = self._active_shadow_run("complete_pass")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=pass_run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                accepted_by_incumbent=True,
            )
        pass_result = complete_shadow_run(self.conn, pass_run["shadow_run_id"], write_decision=True)
        pass_decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (pass_result["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(pass_decision["decision"], "shadow_passed")

        _, _, fail_run = self._active_shadow_run("complete_fail")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=fail_run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                divergence={"kind": "different"},
                accepted_by_incumbent=False,
            )
        fail_result = complete_shadow_run(self.conn, fail_run["shadow_run_id"], write_decision=True)
        fail_decision = self.conn.execute(
            "SELECT decision FROM promotion_decisions WHERE decision_id = ?",
            (fail_result["promotion_decision_id"],),
        ).fetchone()
        self.assertEqual(fail_decision["decision"], "shadow_failed")

    def test_candidate_current_state_updates_only_through_appended_promotion_decisions(self) -> None:
        candidate, _, run = self._active_shadow_run("state_update")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                accepted_by_incumbent=True,
            )
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "canary_passed")
        complete_shadow_run(self.conn, run["shadow_run_id"], write_decision=True)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_passed")

    def test_candidate_would_act_is_never_counted_as_incumbent_did_act(self) -> None:
        _, _, run = self._active_shadow_run("proposal_counting")
        for index in range(3):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                candidate_would_act={"patch": index},
                incumbent_did_act=None,
            )
        summary = summarize_shadow_run(self.conn, run["shadow_run_id"])
        self.assertEqual(summary["n_candidate_would_act"], 3)
        self.assertEqual(summary["n_incumbent_did_act"], 0)
