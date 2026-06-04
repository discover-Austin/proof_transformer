from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from common import BrainTestCase
from proof_transformer.peds import cycle
from proof_transformer.peds.cycle import advance_peds_cycle, plan_peds_cycle, run_peds_cycle_until_blocked
from proof_transformer.peds.shadow import complete_shadow_run, record_shadow_event, start_shadow_run
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_promotion_decision,
)
from proof_transformer.peds.trainer import run_prepared_training_run
from proof_transformer.peds.training import prepare_training_run


class PEDSCycleTests(BrainTestCase):
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
                "candidate_id": f"candidate_cycle_{suffix}",
                "parent_candidate_id": parent_candidate_id,
                "mutation_target": mutation_target,
                "phi_version": "phi_v1",
                "rho_version": "rho_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_v1",
                "delta_description": f"cycle candidate {suffix}",
                "manifest": {"suffix": suffix},
            },
        )

    def _episode(
        self,
        episode_id: str,
        *,
        split: str = "replay",
        timestamp: str = "2026-06-02T00:00:00Z",
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
                    "policy_tags": ["cycle"],
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
                        "seed": 31,
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

    def _distillation_manifest(self, candidate_id: str, suffix: str) -> dict:
        example = {
            "example_id": f"distill_example_cycle_{suffix}",
            "candidate_id": candidate_id,
            "source_type": "shadow_event",
            "source_ref": f"shadow_event_cycle_{suffix}",
            "input": {"prompt": "Explain the cycle."},
            "chosen": {"answer": "The cycle freezes replay before authority."},
            "tags": ["peds", "shadow_event", "accepted_by_incumbent"],
            "provenance": {
                "source_table": "shadow_events",
                "shadow_run_id": f"shadow_run_cycle_{suffix}",
                "shadow_event_id": f"shadow_event_cycle_{suffix}",
                "ordinal": 0,
            },
            "safety_exclusions": [],
            "metadata": {"judge_verdict": "pass"},
        }
        examples = [example]
        payload = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return {
            "candidate_id": candidate_id,
            "created_at": "2026-06-02T00:00:00Z",
            "dataset_hash": "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "n_examples": 1,
            "examples": examples,
            "source_summary": {"candidate_id": candidate_id, "current_state": "distillation_candidate"},
            "warnings": [],
        }

    def _completed_training_run_for(self, candidate_id: str, suffix: str) -> dict:
        prepared = prepare_training_run(
            self.conn,
            self._distillation_manifest(candidate_id, suffix),
            trainer_name="local_adapter_stub",
            trainer_config={"epochs": 1},
        )
        result = run_prepared_training_run(
            self.conn,
            prepared["training_run_id"],
            operator="owner",
            mode="dry_run",
            artifact_dir=str(self.paths.root / "cycle-artifacts" / suffix),
        )
        return result["training_run"]

    def _shadow_passed_candidate(self, suffix: str) -> tuple[dict, dict]:
        candidate, incumbent = self._parent_and_child(suffix)
        replay_episode = self._episode(f"episode_cycle_{suffix}_replay", split="replay")
        replay_job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [replay_episode["episode_id"]],
            split="replay",
        )
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "job_id": replay_job["job_id"],
                "decision": "replay_winner",
                "reason": "Replay winner fixture.",
            },
        )

        canary_episode = self._episode(f"episode_cycle_{suffix}_canary", split="canary")
        canary_job = create_replay_job(
            self.conn,
            candidate["candidate_id"],
            incumbent["candidate_id"],
            [canary_episode["episode_id"]],
            split="canary",
        )
        record_promotion_decision(
            self.conn,
            {
                "candidate_id": candidate["candidate_id"],
                "incumbent_id": incumbent["candidate_id"],
                "job_id": canary_job["job_id"],
                "decision": "canary_passed",
                "reason": "Canary passed fixture.",
            },
        )

        shadow_run = start_shadow_run(
            self.conn,
            candidate_id=candidate["candidate_id"],
            incumbent_id=incumbent["candidate_id"],
        )
        for index in range(10):
            record_shadow_event(
                self.conn,
                shadow_run_id=shadow_run["shadow_run_id"],
                input={"prompt": f"shadow task {index}"},
                incumbent_output={"answer": f"incumbent {index}"},
                candidate_output={"answer": f"candidate {index}"},
                accepted_by_incumbent=True,
            )
        complete_shadow_run(self.conn, shadow_run["shadow_run_id"], write_decision=True)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_passed")
        return candidate, incumbent

    def _passing_replay_evaluator(self, **kwargs: dict) -> dict:
        return {
            "candidate_score": 0.91 + (kwargs["ordinal"] * 0.01),
            "incumbent_score": 0.74 + (kwargs["ordinal"] * 0.01),
            "judge_verdict": "pass",
        }

    def _passing_canary_evaluator(self, **kwargs: dict) -> dict:
        return {
            "candidate_score": 0.84 + (kwargs["ordinal"] * 0.01),
            "incumbent_score": 0.80 + (kwargs["ordinal"] * 0.01),
            "judge_verdict": "pass",
            "canary_failures": [],
        }

    def _passing_shadow_events(self, *, count: int = 10):
        def provider(**_: dict) -> list[dict]:
            return [
                {
                    "input": {"prompt": f"shadow input {index}"},
                    "incumbent_output": {"answer": f"incumbent {index}"},
                    "candidate_output": {"answer": f"candidate {index}"},
                    "accepted_by_incumbent": True,
                    "metadata": {"fixture": "peds_cycle"},
                }
                for index in range(count)
            ]

        return provider

    def _draft_candidate_with_replay_episodes(self, suffix: str) -> dict:
        _, child = self._parent_and_child(suffix)
        for index in range(3):
            self._episode(f"episode_cycle_{suffix}_replay_{index}", split="replay")
        return child

    def _draft_candidate_with_replay_and_canary_episodes(self, suffix: str) -> dict:
        child = self._draft_candidate_with_replay_episodes(suffix)
        for index in range(3):
            self._episode(f"episode_cycle_{suffix}_canary_{index}", split="canary")
        return child

    def test_plan_peds_cycle_returns_create_replay_job_for_draft_candidate(self) -> None:
        _, child = self._parent_and_child("plan_draft")

        plan = plan_peds_cycle(self.conn, child["candidate_id"])

        self.assertEqual(plan["current_state"], "draft_candidate")
        self.assertEqual(plan["next_step"], "create_replay_job")

    def test_plan_peds_cycle_returns_promotion_step_for_shadow_passed(self) -> None:
        candidate, _ = self._shadow_passed_candidate("plan_shadow")

        plan = plan_peds_cycle(self.conn, candidate["candidate_id"])

        self.assertEqual(plan["current_state"], "shadow_passed")
        self.assertEqual(plan["next_step"], "promote_or_distill")

    def test_advance_peds_cycle_dry_run_writes_nothing(self) -> None:
        _, child = self._parent_and_child("dry_run")
        self._episode("episode_cycle_dry_run", split="replay")
        before_jobs = self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"]
        before_decisions = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        result = advance_peds_cycle(self.conn, child["candidate_id"], operator="owner", dry_run=True)

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["changed"], {})
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"], before_jobs)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"],
            before_decisions,
        )

    def test_advance_shadow_passed_without_auto_promote_does_not_promote(self) -> None:
        candidate, _ = self._shadow_passed_candidate("no_auto_promote")
        before = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]

        result = advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=False,
        )

        after = self.conn.execute("SELECT COUNT(*) AS count FROM promotion_decisions").fetchone()["count"]
        self.assertEqual(result["action"], "ready_for_promotion")
        self.assertTrue(result["blocked"])
        self.assertEqual(after, before)
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "shadow_passed")

    def test_advance_shadow_passed_with_auto_promote_promotes(self) -> None:
        candidate, _ = self._shadow_passed_candidate("auto_promote")

        result = advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            options={"promotion_reason": "owner explicitly allowed promotion"},
        )

        self.assertEqual(result["action"], "promote_candidate")
        self.assertFalse(result["blocked"])
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "promoted")
        self.assertEqual(result["changed"]["promotion"]["promotion_decision"]["decision"], "promoted")

    def test_run_peds_cycle_until_blocked_stops_cleanly_on_missing_replay_evidence(self) -> None:
        _, child = self._parent_and_child("missing_replay")
        before_jobs = self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"]

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=3,
        )

        self.assertEqual(result["steps_run"], 1)
        self.assertTrue(result["step_log"][0]["blocked"])
        self.assertIn("at least one replay episode is required", result["stopped_reason"])
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"], before_jobs)

    def test_run_peds_cycle_until_blocked_stops_at_draft_when_replay_evaluator_missing(self) -> None:
        child = self._draft_candidate_with_replay_episodes("missing_replay_evaluator")

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=3,
        )

        self.assertEqual(result["steps_run"], 1)
        self.assertEqual(result["step_log"][0]["action"], "create_replay_job")
        self.assertTrue(result["step_log"][0]["blocked"])
        self.assertIn("replay_evaluator required", result["stopped_reason"])
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "draft_candidate")
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM replay_jobs WHERE candidate_id = ? AND split = 'replay'",
                (child["candidate_id"],),
            ).fetchone()["count"],
            1,
        )

    def test_run_peds_cycle_until_blocked_runs_replay_and_reaches_replay_winner(self) -> None:
        child = self._draft_candidate_with_replay_episodes("runs_replay")

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=3,
            replay_evaluator=self._passing_replay_evaluator,
        )

        self.assertEqual(result["step_log"][0]["action"], "run_replay_job")
        self.assertEqual(result["step_log"][0]["state_after"], "replay_winner")
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "replay_winner")
        self.assertIn("no canary episodes available", result["stopped_reason"])

    def test_run_peds_cycle_until_blocked_runs_canary_and_reaches_canary_passed(self) -> None:
        child = self._draft_candidate_with_replay_and_canary_episodes("runs_canary")

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=4,
            replay_evaluator=self._passing_replay_evaluator,
            canary_evaluator=self._passing_canary_evaluator,
        )

        self.assertEqual(result["step_log"][0]["state_after"], "replay_winner")
        self.assertEqual(result["step_log"][1]["action"], "run_canary_job")
        self.assertEqual(result["step_log"][1]["state_after"], "canary_passed")
        self.assertEqual(result["step_log"][2]["action"], "start_shadow_run")
        self.assertIn("shadow_event_provider required", result["stopped_reason"])

    def test_run_peds_cycle_until_blocked_records_shadow_events_and_reaches_shadow_passed(self) -> None:
        child = self._draft_candidate_with_replay_and_canary_episodes("runs_shadow")

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=3,
            replay_evaluator=self._passing_replay_evaluator,
            canary_evaluator=self._passing_canary_evaluator,
            shadow_event_provider=self._passing_shadow_events(),
        )

        self.assertEqual(result["steps_run"], 3)
        self.assertEqual(result["step_log"][2]["action"], "complete_shadow_run")
        self.assertEqual(result["step_log"][2]["state_after"], "shadow_passed")
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "shadow_passed")

    def test_run_peds_cycle_until_blocked_stops_ready_for_promotion_without_auto_promote(self) -> None:
        child = self._draft_candidate_with_replay_and_canary_episodes("ready_for_promotion")

        result = run_peds_cycle_until_blocked(
            self.conn,
            child["candidate_id"],
            operator="owner",
            max_steps=4,
            replay_evaluator=self._passing_replay_evaluator,
            canary_evaluator=self._passing_canary_evaluator,
            shadow_event_provider=self._passing_shadow_events(),
        )

        self.assertEqual(result["step_log"][-1]["action"], "ready_for_promotion")
        self.assertTrue(result["step_log"][-1]["blocked"])
        self.assertEqual(result["step_log"][-1]["state_before"], "shadow_passed")
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "shadow_passed")

    def test_run_peds_cycle_until_blocked_auto_promotes_shadow_passed_candidate(self) -> None:
        candidate, _ = self._shadow_passed_candidate("run_auto_promote")

        result = run_peds_cycle_until_blocked(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            max_steps=1,
            options={"promotion_reason": "owner allowed bounded cycle promotion"},
        )

        self.assertEqual(result["step_log"][0]["action"], "promote_candidate")
        self.assertFalse(result["step_log"][0]["blocked"])
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "promoted")

    def test_promoted_candidate_becomes_distillation_candidate(self) -> None:
        candidate, _ = self._shadow_passed_candidate("mark_distill")
        advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            options={"promotion_reason": "explicit promotion for distillation marker"},
        )

        result = advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            options={"distillation_reason": "ready for distillation"},
        )

        self.assertEqual(result["action"], "mark_distillation_candidate")
        self.assertEqual(result["changed"]["promotion_decision"]["decision"], "distillation_candidate")
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "distillation_candidate")

    def test_distillation_candidate_builds_manifest_and_prepares_training_run(self) -> None:
        candidate, _ = self._shadow_passed_candidate("prepare_training")
        advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            options={"promotion_reason": "explicit promotion for training preparation"},
        )
        advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            options={"distillation_reason": "ready for training preparation"},
        )
        output_path = self.paths.root / "cycle-manifests" / "distillation.json"

        result = advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            options={
                "output_path": str(output_path),
                "trainer_name": "local_adapter_stub",
                "trainer_config": {"epochs": 1},
            },
        )

        self.assertEqual(result["action"], "prepare_training_run")
        self.assertTrue(result["blocked"])
        self.assertIn("run_training=True required", result["blockers"])
        self.assertEqual(result["changed"]["distillation_manifest"]["candidate_id"], candidate["candidate_id"])
        self.assertEqual(result["changed"]["training_run"]["status"], "prepared")
        self.assertTrue(output_path.exists())

    def test_peds_first_real_bounded_cycle_reaches_child_replay_job(self) -> None:
        incumbent = self._candidate("first_real_incumbent", mutation_target="theta")
        candidate = self._candidate(
            "first_real_child",
            parent_candidate_id=incumbent["candidate_id"],
            mutation_target="theta",
        )
        self.assertEqual(candidate["parent_candidate_id"], incumbent["candidate_id"])
        self.assertEqual(candidate["mutation_target"], "theta")
        self.assertEqual(candidate["phi_version"], incumbent["phi_version"])
        self.assertEqual(candidate["rho_version"], incumbent["rho_version"])
        self.assertNotEqual(candidate["theta_version"], incumbent["theta_version"])
        self.assertEqual(candidate["psi_version"], incumbent["psi_version"])

        for index in range(3):
            self._episode(f"episode_cycle_first_real_replay_{index}", split="replay")
            self._episode(f"episode_cycle_first_real_canary_{index}", split="canary")

        def replay_evaluator(**kwargs: dict) -> dict:
            return {
                "candidate_score": 0.93 + (kwargs["ordinal"] * 0.01),
                "incumbent_score": 0.76 + (kwargs["ordinal"] * 0.01),
                "judge_verdict": "pass",
                "audit_log_ref": "deterministic:first-real-replay",
            }

        def canary_evaluator(**kwargs: dict) -> dict:
            return {
                "candidate_score": 0.84 + (kwargs["ordinal"] * 0.01),
                "incumbent_score": 0.84 + (kwargs["ordinal"] * 0.01),
                "judge_verdict": "pass",
                "canary_failures": [],
                "audit_log_ref": "deterministic:first-real-canary",
            }

        def shadow_event_provider(**_: dict) -> list[dict]:
            return [
                {
                    "input": {"prompt": f"deterministic shadow input {index}"},
                    "incumbent_output": {"answer": f"incumbent baseline {index}"},
                    "candidate_output": {"answer": f"candidate accepted {index}"},
                    "accepted_by_incumbent": True,
                    "metadata": {
                        "fixture": "deterministic_first_bounded_peds_proof_cycle",
                        "ordinal": index,
                    },
                }
                for index in range(10)
            ]

        manifest_path = self.paths.root / "cycle-manifests" / "first-real-distillation.json"
        artifact_dir = self.paths.root / "cycle-artifacts" / "first-real"
        result = run_peds_cycle_until_blocked(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            max_steps=10,
            options={
                "promotion_reason": "deterministic first bounded PEDS proof cycle",
                "distillation_reason": "promoted candidate selected for deterministic distillation fixture",
                "trainer_name": "local_adapter_stub",
                "trainer_config": {"epochs": 1},
                "run_training": True,
                "trainer_mode": "dry_run",
                "adapter_name": "deterministic_cycle_adapter",
                "output_path": str(manifest_path),
                "artifact_dir": str(artifact_dir),
            },
            replay_evaluator=replay_evaluator,
            canary_evaluator=canary_evaluator,
            shadow_event_provider=shadow_event_provider,
        )

        final_step = result["step_log"][-1]
        generated_candidate = final_step["changed"]["candidate"]
        generated_replay_job = final_step["changed"]["replay_job"]
        training_run = final_step["changed"]["training_result"]["training_run"]
        artifact_manifest = final_step["changed"]["training_result"]["artifact_manifest"]
        distillation_manifest = final_step["changed"]["distillation_manifest"]

        self.assertIn(
            "post-training candidate created; start a new cycle for generated candidate",
            result["stopped_reason"],
        )
        self.assertEqual(get_candidate_current_state(self.conn, candidate["candidate_id"]), "distillation_candidate")
        self.assertEqual(result["final_candidate_id"], generated_candidate["candidate_id"])
        self.assertEqual(generated_candidate["parent_candidate_id"], candidate["candidate_id"])
        self.assertEqual(generated_candidate["mutation_target"], "theta")
        self.assertEqual(get_candidate_current_state(self.conn, generated_candidate["candidate_id"]), "draft_candidate")

        decision_rows = self.conn.execute(
            """
            SELECT decision
            FROM promotion_decisions
            WHERE candidate_id = ?
            ORDER BY created_at, decision_id
            """,
            (candidate["candidate_id"],),
        ).fetchall()
        decisions = [row["decision"] for row in decision_rows]
        self.assertIn("replay_winner", decisions)
        self.assertIn("canary_passed", decisions)
        self.assertIn("shadow_started", decisions)
        self.assertIn("shadow_passed", decisions)
        self.assertIn("promoted", decisions)
        self.assertIn("distillation_candidate", decisions)

        self.assertTrue(manifest_path.exists())
        self.assertEqual(distillation_manifest["candidate_id"], candidate["candidate_id"])
        self.assertGreater(distillation_manifest["n_examples"], 0)
        stored_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(stored_manifest["dataset_hash"], distillation_manifest["dataset_hash"])

        stored_training_run = self.conn.execute(
            "SELECT status, metrics_json, artifact_ref FROM peds_training_runs WHERE training_run_id = ?",
            (training_run["training_run_id"],),
        ).fetchone()
        self.assertEqual(stored_training_run["status"], "completed")
        stored_metrics = json.loads(stored_training_run["metrics_json"])
        self.assertEqual(stored_metrics["mode"], "dry_run")
        self.assertFalse(stored_metrics["trained"])
        self.assertEqual(training_run["metrics"]["trained"], False)
        self.assertEqual(final_step["changed"]["training_result"]["trained"], False)

        artifact_path = stored_training_run["artifact_ref"]
        self.assertIsNotNone(artifact_path)
        persisted_artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        self.assertFalse(artifact_manifest["trained"])
        self.assertFalse(artifact_manifest["metrics"]["trained"])
        self.assertFalse(artifact_manifest["provenance"]["trained"])
        self.assertFalse(persisted_artifact["trained"])
        self.assertFalse(persisted_artifact["metrics"]["trained"])
        self.assertFalse(persisted_artifact["provenance"]["trained"])

        adapter = generated_candidate["adapter_registration"]["adapter"]
        self.assertEqual(adapter["name"], "deterministic_cycle_adapter")
        self.assertEqual(adapter["status"], "metadata_only")
        adapter_row = self.conn.execute(
            "SELECT name, status, dataset_hash, path FROM adapter_registry WHERE id = ?",
            (generated_candidate["manifest"]["adapter_registry_id"],),
        ).fetchone()
        self.assertEqual(adapter_row["name"], "deterministic_cycle_adapter")
        self.assertEqual(adapter_row["status"], "metadata_only")
        self.assertEqual(adapter_row["dataset_hash"], training_run["dataset_hash"])
        self.assertEqual(adapter_row["path"], artifact_path)

        self.assertFalse(generated_candidate["manifest"]["trained"])
        self.assertEqual(generated_candidate["manifest"]["trainer_mode"], "dry_run")
        self.assertEqual(generated_candidate["manifest"]["adapter_ref"]["status"], "metadata_only")
        self.assertEqual(generated_candidate["manifest"]["source_training_run_id"], training_run["training_run_id"])

        replay_job_row = self.conn.execute(
            "SELECT job_id, candidate_id, split, status FROM replay_jobs WHERE job_id = ?",
            (generated_replay_job["job_id"],),
        ).fetchone()
        self.assertEqual(replay_job_row["candidate_id"], generated_candidate["candidate_id"])
        self.assertEqual(replay_job_row["split"], "replay")
        self.assertEqual(replay_job_row["status"], "queued")
        frozen_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM replay_job_episodes WHERE job_id = ?",
            (generated_replay_job["job_id"],),
        ).fetchone()["count"]
        self.assertGreater(frozen_count, 0)
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM replay_results WHERE job_id = ?",
                (generated_replay_job["job_id"],),
            ).fetchone()["count"],
            0,
        )
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
                (generated_candidate["candidate_id"],),
            ).fetchone()["count"],
            0,
        )

    def test_completed_training_run_creates_candidate_and_replay_job_without_authority(self) -> None:
        candidate, _ = self._shadow_passed_candidate("completed_training")
        advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            auto_promote=True,
            options={"promotion_reason": "explicit promotion for cycle bridge test"},
        )
        advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            options={"distillation_reason": "ready for completed training bridge test"},
        )
        training_run = self._completed_training_run_for(candidate["candidate_id"], "completed_training")

        result = advance_peds_cycle(
            self.conn,
            candidate["candidate_id"],
            operator="owner",
            options={"adapter_name": "cycle_adapter"},
        )

        new_candidate = result["changed"]["candidate"]
        replay_job = result["changed"]["replay_job"]
        self.assertEqual(result["action"], "create_candidate_from_training_run")
        self.assertEqual(new_candidate["parent_candidate_id"], candidate["candidate_id"])
        self.assertEqual(new_candidate["mutation_target"], "theta")
        self.assertEqual(new_candidate["manifest"]["source_training_run_id"], training_run["training_run_id"])
        self.assertEqual(get_candidate_current_state(self.conn, new_candidate["candidate_id"]), "draft_candidate")
        self.assertEqual(replay_job["candidate_id"], new_candidate["candidate_id"])
        self.assertEqual(replay_job["split"], "replay")
        self.assertEqual(
            self.conn.execute(
                "SELECT COUNT(*) AS count FROM promotion_decisions WHERE candidate_id = ?",
                (new_candidate["candidate_id"],),
            ).fetchone()["count"],
            0,
        )

    def test_cycle_does_not_duplicate_promotion_logic(self) -> None:
        source = inspect.getsource(cycle)

        self.assertIn("promote_candidate", source)
        self.assertNotIn("record_promotion_decision", source)
        self.assertNotIn("INSERT INTO promotion_decisions", source)

    def test_cycle_uses_existing_replay_runner_once(self) -> None:
        source = inspect.getsource(cycle)

        self.assertEqual(source.count("run_replay_job("), 1)
        self.assertNotIn("_execute_frozen_job", source)

    def test_cycle_has_no_network_or_model_api_calls(self) -> None:
        source = inspect.getsource(cycle)

        for forbidden in ("requests", "urllib", "openai", "anthropic", "socket"):
            self.assertNotIn(forbidden, source)
