from __future__ import annotations

from common import BrainTestCase
from proof_transformer.peds.canary import create_canary_job, evaluate_canary_job
from proof_transformer.peds.promotion import (
    assert_promotion_eligible,
    evaluate_promotion_eligibility,
    mark_distillation_candidate,
    mark_shadow_deployment,
    promote_candidate,
    promotion_eligibility_gate,
    summarize_promotion_evidence,
)
from proof_transformer.peds.scoring import evaluate_replay_job
from proof_transformer.peds.shadow import complete_shadow_run, record_shadow_event, start_shadow_run
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_promotion_decision,
    record_replay_result,
)


class PEDSPromotionTests(BrainTestCase):
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
                        "seed": 23,
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

    def _candidate_pair(self, suffix: str) -> tuple[dict, dict]:
        candidate = self._candidate(f"{suffix}_candidate")
        incumbent = self._candidate(f"{suffix}_incumbent", mutation_target="rho")
        return candidate, incumbent

    def _write_replay_winner(self, candidate: dict, incumbent: dict, suffix: str) -> dict:
        episodes = [self._episode(f"replay_{suffix}_{index}", split="replay") for index in range(3)]
        job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [episode["episode_id"] for episode in episodes],
        )
        for episode in episodes:
            record_replay_result(
                self.conn,
                {
                    "job_id": job["job_id"],
                    "episode_id": episode["episode_id"],
                    "candidate_score": 0.90,
                    "incumbent_score": 0.70,
                    "judge_verdict": "pass",
                    "canary_failures": [],
                },
            )
        result = evaluate_replay_job(self.conn, job["job_id"], write_decision=True)
        self.assertEqual(result["gate"]["decision"], "replay_winner")
        return result

    def _write_canary_passed(self, candidate: dict, incumbent: dict, suffix: str) -> dict:
        episodes = [self._episode(f"canary_{suffix}_{index}", split="canary") for index in range(3)]
        job_id = create_canary_job(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
            episode_ids=[episode["episode_id"] for episode in episodes],
        )
        for episode in episodes:
            record_replay_result(
                self.conn,
                {
                    "job_id": job_id,
                    "episode_id": episode["episode_id"],
                    "candidate_score": 0.84,
                    "incumbent_score": 0.80,
                    "judge_verdict": "pass",
                    "canary_failures": [],
                },
            )
        result = evaluate_canary_job(self.conn, job_id, write_decision=True)
        self.assertEqual(result["gate"]["decision"], "canary_passed")
        return result

    def _write_shadow_passed(self, candidate: dict, incumbent: dict, suffix: str, *, n_events: int = 10) -> dict:
        run = start_shadow_run(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
        )
        for index in range(n_events):
            record_shadow_event(
                self.conn,
                shadow_run_id=run["shadow_run_id"],
                input={"task": index},
                incumbent_output={"answer": "incumbent"},
                candidate_output={"answer": "candidate"},
                accepted_by_incumbent=True,
            )
        result = complete_shadow_run(self.conn, run["shadow_run_id"], write_decision=True)
        self.assertEqual(result["gate"]["decision"], "shadow_passed")
        return result

    def _promotion_ready_candidate(self, suffix: str) -> tuple[dict, dict]:
        candidate, incumbent = self._candidate_pair(suffix)
        self._write_replay_winner(candidate, incumbent, suffix)
        self._write_canary_passed(candidate, incumbent, suffix)
        self._write_shadow_passed(candidate, incumbent, suffix)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_passed")
        return candidate, incumbent

    def _evidence(self, **overrides: object) -> dict:
        replay = {
            "decision": "replay_winner",
            "decision_id": "decision_replay",
            "created_at": "2026-06-02T00:00:00Z",
            "incumbent_id": "candidate_incumbent",
        }
        canary = {
            "decision": "canary_passed",
            "decision_id": "decision_canary",
            "created_at": "2026-06-02T00:01:00Z",
            "incumbent_id": "candidate_incumbent",
        }
        shadow = {
            "decision": "shadow_passed",
            "decision_id": "decision_shadow",
            "created_at": "2026-06-02T00:02:00Z",
            "incumbent_id": "candidate_incumbent",
            "shadow_run_id": "shadow_run_gate",
        }
        summary = {
            "shadow_run_id": "shadow_run_gate",
            "candidate_id": "candidate_gate",
            "incumbent_id": "candidate_incumbent",
            "status": "completed",
            "n_events": 10,
            "n_divergences": 1,
            "divergence_rate": 0.10,
            "accepted_by_incumbent_rate": 0.80,
            "n_candidate_would_act": 0,
            "n_incumbent_did_act": 10,
        }
        evidence = {
            "candidate_id": "candidate_gate",
            "current_state": "shadow_passed",
            "decision_timeline": [shadow, canary, replay],
            "latest_replay_winner": replay,
            "latest_replay_winner_decision": replay,
            "latest_canary_passed": canary,
            "latest_canary_passed_decision": canary,
            "latest_shadow_decision": shadow,
            "latest_shadow_passed": shadow,
            "latest_shadow_passed_decision": shadow,
            "latest_shadow_summary": summary,
            "latest_completed_shadow_summary": summary,
            "invalidating_decisions_after_shadow_pass": [],
            "incumbent_id": "candidate_incumbent",
            "warnings": [],
            "rejected_after_shadow_evidence": False,
            "rejected_after_shadow_passed": False,
            "shadow_failed_after_shadow_passed": False,
        }
        evidence.update(overrides)
        return evidence

    def test_summarize_promotion_evidence_sees_replay_winner(self) -> None:
        candidate, incumbent = self._candidate_pair("summary_replay")
        self._write_replay_winner(candidate, incumbent, "summary_replay")

        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        self.assertEqual(evidence["latest_replay_winner_decision"]["decision"], "replay_winner")
        self.assertEqual(evidence["current_state"], "replay_winner")

    def test_summarize_promotion_evidence_sees_canary_passed(self) -> None:
        candidate, incumbent = self._candidate_pair("summary_canary")
        self._write_replay_winner(candidate, incumbent, "summary_canary")
        self._write_canary_passed(candidate, incumbent, "summary_canary")

        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        self.assertEqual(evidence["latest_canary_passed_decision"]["decision"], "canary_passed")
        self.assertEqual(evidence["current_state"], "canary_passed")

    def test_summarize_promotion_evidence_sees_shadow_passed(self) -> None:
        candidate, _ = self._promotion_ready_candidate("summary_shadow")

        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        self.assertEqual(evidence["latest_shadow_decision"]["decision"], "shadow_passed")
        self.assertEqual(evidence["latest_shadow_passed_decision"]["decision"], "shadow_passed")
        self.assertEqual(evidence["latest_completed_shadow_summary"]["n_events"], 10)
        self.assertEqual(evidence["decision_timeline"][0]["decision"], "shadow_passed")
        self.assertEqual(evidence["incumbent_id"], "candidate_summary_shadow_incumbent")
        self.assertEqual(evidence["warnings"], [])

    def test_summarize_promotion_evidence_rejects_missing_candidate(self) -> None:
        with self.assertRaises(ValueError):
            summarize_promotion_evidence(self.conn, "missing_candidate")

    def test_promotion_eligibility_gate_rejects_draft_candidate(self) -> None:
        gate = promotion_eligibility_gate(self._evidence(current_state="draft_candidate"))
        self.assertFalse(gate["eligible"])
        self.assertEqual(gate["decision"], "promotion_ineligible")
        self.assertIn("current_state", gate["reason"])

    def test_promotion_eligibility_gate_rejects_missing_replay_evidence(self) -> None:
        gate = promotion_eligibility_gate(
            self._evidence(latest_replay_winner=None, latest_replay_winner_decision=None)
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("missing replay_winner evidence", gate["reason"])

    def test_promotion_eligibility_gate_rejects_missing_canary_evidence(self) -> None:
        gate = promotion_eligibility_gate(
            self._evidence(latest_canary_passed=None, latest_canary_passed_decision=None)
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("missing canary_passed evidence", gate["reason"])

    def test_promotion_eligibility_gate_rejects_missing_shadow_evidence(self) -> None:
        gate = promotion_eligibility_gate(
            self._evidence(latest_shadow_decision=None, latest_shadow_passed=None, latest_shadow_passed_decision=None)
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("missing shadow_passed evidence", gate["reason"])

    def test_promotion_eligibility_gate_rejects_insufficient_shadow_events(self) -> None:
        evidence = self._evidence()
        summary = dict(evidence["latest_shadow_summary"], n_events=9)
        evidence["latest_shadow_summary"] = summary
        evidence["latest_completed_shadow_summary"] = summary
        gate = promotion_eligibility_gate(evidence)
        self.assertFalse(gate["eligible"])
        self.assertIn("shadow n_events < min_shadow_events", gate["reason"])

    def test_promotion_eligibility_gate_rejects_excessive_divergence(self) -> None:
        evidence = self._evidence()
        summary = dict(evidence["latest_shadow_summary"], divergence_rate=0.31)
        evidence["latest_shadow_summary"] = summary
        evidence["latest_completed_shadow_summary"] = summary
        gate = promotion_eligibility_gate(evidence)
        self.assertFalse(gate["eligible"])
        self.assertIn("shadow divergence_rate > max_shadow_divergence_rate", gate["reason"])

    def test_promotion_eligibility_gate_rejects_low_acceptance_rate(self) -> None:
        evidence = self._evidence()
        summary = dict(evidence["latest_shadow_summary"], accepted_by_incumbent_rate=0.69)
        evidence["latest_shadow_summary"] = summary
        evidence["latest_completed_shadow_summary"] = summary
        gate = promotion_eligibility_gate(evidence)
        self.assertFalse(gate["eligible"])
        self.assertIn("shadow accepted_by_incumbent_rate < min_shadow_acceptance_rate", gate["reason"])

    def test_promotion_eligibility_gate_passes_complete_evidence(self) -> None:
        gate = promotion_eligibility_gate(self._evidence())
        self.assertTrue(gate["eligible"])
        self.assertEqual(gate["decision"], "promotion_eligible")

    def test_assert_promotion_eligible_passes_for_complete_evidence(self) -> None:
        candidate, _ = self._promotion_ready_candidate("assert_pass")
        gate = assert_promotion_eligible(self.conn, candidate["candidate_id"])
        self.assertTrue(gate["eligible"])

    def test_assert_promotion_eligible_raises_for_ineligible_candidate(self) -> None:
        candidate = self._candidate("assert_fail")
        with self.assertRaises(ValueError):
            assert_promotion_eligible(self.conn, candidate["candidate_id"])

    def test_promotion_eligibility_gate_rejects_later_rejection_after_shadow(self) -> None:
        gate = promotion_eligibility_gate(
            self._evidence(invalidating_decisions_after_shadow_pass=[{"decision": "rejected"}])
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("invalidating decision after latest shadow_passed", gate["reason"])

    def test_promotion_eligibility_gate_rejects_later_shadow_failed_after_shadow(self) -> None:
        gate = promotion_eligibility_gate(
            self._evidence(invalidating_decisions_after_shadow_pass=[{"decision": "shadow_failed"}])
        )
        self.assertFalse(gate["eligible"])
        self.assertIn("invalidating decision after latest shadow_passed", gate["reason"])

    def test_evaluate_promotion_eligibility_does_not_append_promotion_decisions(self) -> None:
        candidate, _ = self._promotion_ready_candidate("eval_no_append")
        before_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
            (candidate["candidate_id"],),
        ).fetchone()["count"]

        gate = evaluate_promotion_eligibility(self.conn, candidate["candidate_id"])

        after_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
            (candidate["candidate_id"],),
        ).fetchone()["count"]
        self.assertTrue(gate["eligible"])
        self.assertEqual(after_count, before_count)

    def test_evaluate_promotion_eligibility_does_not_write_promoted(self) -> None:
        candidate, _ = self._promotion_ready_candidate("eval_no_promoted")
        evaluate_promotion_eligibility(self.conn, candidate["candidate_id"])

        promoted_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ? AND decision = 'promoted'",
            (candidate["candidate_id"],),
        ).fetchone()["count"]
        self.assertEqual(promoted_count, 0)

    def test_candidate_current_state_remains_derived(self) -> None:
        candidate, _ = self._promotion_ready_candidate("derived_state")
        latest_decision = self.conn.execute(
            """
            SELECT decision
            FROM promotion_decisions
            WHERE candidate_id = ?
            ORDER BY created_at DESC, decision_id DESC
            LIMIT 1
            """,
            (candidate["candidate_id"],),
        ).fetchone()["decision"]
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), latest_decision)

        evaluate_promotion_eligibility(self.conn, candidate["candidate_id"])

        latest_decision_after = self.conn.execute(
            """
            SELECT decision
            FROM promotion_decisions
            WHERE candidate_id = ?
            ORDER BY created_at DESC, decision_id DESC
            LIMIT 1
            """,
            (candidate["candidate_id"],),
        ).fetchone()["decision"]
        self.assertEqual(latest_decision_after, "shadow_passed")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), latest_decision_after)

    def test_summarize_promotion_evidence_detects_later_rejected_invalidator(self) -> None:
        candidate, incumbent = self._promotion_ready_candidate("later_rejected")
        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        shadow_run_id = evidence["latest_shadow_passed"]["shadow_run_id"]
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "shadow_run_id": shadow_run_id,
                "decision": "rejected",
                "reason": "Operator rejected after shadow evidence.",
            },
        )

        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        gate = promotion_eligibility_gate(evidence)
        self.assertEqual(evidence["invalidating_decisions_after_shadow_pass"][0]["decision"], "rejected")
        self.assertFalse(gate["eligible"])

    def test_summarize_promotion_evidence_detects_later_shadow_failed_invalidator(self) -> None:
        candidate, incumbent = self._promotion_ready_candidate("later_shadow_failed")
        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        shadow_run_id = evidence["latest_shadow_passed"]["shadow_run_id"]
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "shadow_run_id": shadow_run_id,
                "decision": "shadow_failed",
                "reason": "Late shadow regression detected.",
            },
        )

        evidence = summarize_promotion_evidence(self.conn, candidate["candidate_id"])
        gate = promotion_eligibility_gate(evidence)
        self.assertEqual(evidence["invalidating_decisions_after_shadow_pass"][0]["decision"], "shadow_failed")
        self.assertFalse(gate["eligible"])

    def test_promote_candidate_rejects_ineligible_candidate(self) -> None:
        candidate = self._candidate("promote_ineligible")
        with self.assertRaises(ValueError):
            promote_candidate(
                self.conn,
                candidate["candidate_id"],
                operator="owner",
                reason="not enough evidence",
            )

    def test_promote_candidate_requires_operator(self) -> None:
        candidate, _ = self._promotion_ready_candidate("promote_operator")
        with self.assertRaises(ValueError):
            promote_candidate(self.conn, candidate["candidate_id"], operator="", reason="ready")

    def test_promote_candidate_requires_reason(self) -> None:
        candidate, _ = self._promotion_ready_candidate("promote_reason")
        with self.assertRaises(ValueError):
            promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason=" ")

    def test_promote_candidate_appends_promoted_only_after_eligibility(self) -> None:
        candidate, incumbent = self._promotion_ready_candidate("promote_success")
        result = promote_candidate(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="shadow evidence is clean",
        )

        self.assertEqual(result["candidate_id"], candidate["candidate_id"])
        self.assertEqual(result["incumbent_id"], incumbent["candidate_id"])
        self.assertEqual(result["promotion_decision"]["decision"], "promoted")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "promoted")

    def test_promote_candidate_preserves_prior_decisions(self) -> None:
        candidate, _ = self._promotion_ready_candidate("promote_append_only")
        before = [
            dict(row)
            for row in self.conn.execute(
                "SELECT decision_id, decision, reason FROM promotion_decisions WHERE candidate_id = ? ORDER BY created_at, decision_id",
                (candidate["candidate_id"],),
            ).fetchall()
        ]

        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="append only")

        after = [
            dict(row)
            for row in self.conn.execute(
                "SELECT decision_id, decision, reason FROM promotion_decisions WHERE candidate_id = ? ORDER BY created_at, decision_id",
                (candidate["candidate_id"],),
            ).fetchall()
        ]
        self.assertEqual(after[: len(before)], before)
        self.assertEqual(after[-1]["decision"], "promoted")

    def test_evaluate_promotion_eligibility_still_writes_nothing_after_promotion_api_exists(self) -> None:
        candidate, _ = self._promotion_ready_candidate("eval_still_read_only")
        before_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
            (candidate["candidate_id"],),
        ).fetchone()["count"]
        evaluate_promotion_eligibility(self.conn, candidate["candidate_id"])
        after_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
            (candidate["candidate_id"],),
        ).fetchone()["count"]
        self.assertEqual(after_count, before_count)

    def test_mark_shadow_deployment_rejects_non_promoted_candidate(self) -> None:
        candidate, _ = self._promotion_ready_candidate("deploy_not_promoted")
        with self.assertRaises(ValueError):
            mark_shadow_deployment(
                self.conn,
                candidate["candidate_id"],
                operator="owner",
                reason="not explicitly promoted yet",
            )

    def test_mark_shadow_deployment_appends_for_promoted_candidate(self) -> None:
        candidate, _ = self._promotion_ready_candidate("deploy_success")
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="promote first")

        decision = mark_shadow_deployment(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="watch deployment-like behavior",
            metadata={"window": "manual"},
        )

        self.assertEqual(decision["decision"], "shadow_deployment")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_deployment")

    def test_mark_shadow_deployment_does_not_create_shadow_events(self) -> None:
        candidate, _ = self._promotion_ready_candidate("deploy_no_events")
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="promote first")
        before_events = self.conn.execute("SELECT COUNT(*) AS count FROM shadow_events").fetchone()["count"]

        mark_shadow_deployment(self.conn, candidate["candidate_id"], operator="owner", reason="marker only")

        after_events = self.conn.execute("SELECT COUNT(*) AS count FROM shadow_events").fetchone()["count"]
        self.assertEqual(after_events, before_events)

    def test_mark_distillation_candidate_rejects_draft_replay_and_canary_only_candidates(self) -> None:
        draft = self._candidate("distill_draft")
        with self.assertRaises(ValueError):
            mark_distillation_candidate(self.conn, draft["candidate_id"], operator="owner", reason="too early")

        replay_candidate, replay_incumbent = self._candidate_pair("distill_replay_only")
        self._write_replay_winner(replay_candidate, replay_incumbent, "distill_replay_only")
        with self.assertRaises(ValueError):
            mark_distillation_candidate(self.conn, replay_candidate["candidate_id"], operator="owner", reason="too early")

        canary_candidate, canary_incumbent = self._candidate_pair("distill_canary_only")
        self._write_replay_winner(canary_candidate, canary_incumbent, "distill_canary_only")
        self._write_canary_passed(canary_candidate, canary_incumbent, "distill_canary_only")
        with self.assertRaises(ValueError):
            mark_distillation_candidate(self.conn, canary_candidate["candidate_id"], operator="owner", reason="too early")

    def test_mark_distillation_candidate_accepts_promoted_candidate_with_full_evidence(self) -> None:
        candidate, _ = self._promotion_ready_candidate("distill_promoted")
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="promoted")

        decision = mark_distillation_candidate(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="ready for dataset construction",
        )

        self.assertEqual(decision["decision"], "distillation_candidate")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "distillation_candidate")

    def test_mark_distillation_candidate_accepts_shadow_deployment_candidate_with_full_evidence(self) -> None:
        candidate, _ = self._promotion_ready_candidate("distill_deployed")
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="promoted")
        mark_shadow_deployment(self.conn, candidate["candidate_id"], operator="owner", reason="deployment marker")

        decision = mark_distillation_candidate(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="deployment shadow stayed clean enough to consider distillation",
            metadata={"source": "manual"},
        )

        self.assertEqual(decision["decision"], "distillation_candidate")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "distillation_candidate")

    def test_mark_distillation_candidate_does_not_train_or_write_artifacts(self) -> None:
        candidate, _ = self._promotion_ready_candidate("distill_no_artifacts")
        promote_candidate(self.conn, candidate["candidate_id"], operator="owner", reason="promoted")
        before_adapters = self.conn.execute("SELECT COUNT(*) AS count FROM adapter_registry").fetchone()["count"]

        decision = mark_distillation_candidate(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            reason="marker only",
        )

        after_adapters = self.conn.execute("SELECT COUNT(*) AS count FROM adapter_registry").fetchone()["count"]
        self.assertEqual(decision["decision"], "distillation_candidate")
        self.assertEqual(after_adapters, before_adapters)
