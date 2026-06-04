from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from common import BrainTestCase
from proof_transformer.cli import main
from proof_transformer.peds.shadow import complete_shadow_run, record_shadow_event, start_shadow_run
from proof_transformer.peds.store import (
    create_candidate,
    create_replay_job,
    get_candidate_current_state,
    record_episode,
    record_promotion_decision,
)


class PEDSCLITests(BrainTestCase):
    def _run_cli_json(self, args: list[str]) -> dict:
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        self.assertEqual(code, 0)
        return json.loads(output.getvalue())

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
                "candidate_id": f"candidate_peds_cli_{suffix}",
                "parent_candidate_id": parent_candidate_id,
                "mutation_target": mutation_target,
                "phi_version": "phi_v1",
                "rho_version": "rho_v1",
                "theta_version": f"theta_{suffix}",
                "psi_version": "psi_v1",
                "delta_description": f"PEDS CLI candidate {suffix}",
                "manifest": {"suffix": suffix},
            },
        )

    def _episode(self, episode_id: str, *, split: str) -> dict:
        return record_episode(
            self.conn,
            {
                "episode_id": episode_id,
                "schema_version": "peds.episode.v1",
                "split": split,
                "timestamp": "2026-06-02T00:00:00Z",
                "task": {
                    "task_id": f"task_{episode_id}",
                    "title": f"Task {episode_id}",
                    "prompt": f"Solve {episode_id}",
                    "acceptance_tests": [f"pytest tests/{episode_id}.py"],
                    "policy_tags": ["peds_cli"],
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
                        "seed": 41,
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

    def _draft_candidate_with_replay_episode(self, suffix: str) -> dict:
        _, child = self._parent_and_child(suffix)
        self._episode(f"episode_peds_cli_{suffix}_replay", split="replay")
        return child

    def _shadow_passed_candidate(self, suffix: str) -> dict:
        candidate, incumbent = self._parent_and_child(suffix)
        replay_episode = self._episode(f"episode_peds_cli_{suffix}_replay", split="replay")
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

        canary_episode = self._episode(f"episode_peds_cli_{suffix}_canary", split="canary")
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
        return candidate

    def test_peds_cycle_cli_dry_run_prints_plan_and_writes_nothing(self) -> None:
        child = self._draft_candidate_with_replay_episode("dry_run")
        before_jobs = self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"]

        result = self._run_cli_json(
            [
                "peds-cycle",
                "--candidate-id",
                child["candidate_id"],
                "--dry-run",
            ]
        )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["action"], "dry_run")
        self.assertEqual(result["plan"]["current_state"], "draft_candidate")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) AS count FROM replay_jobs").fetchone()["count"], before_jobs)

    def test_peds_cycle_cli_without_evaluator_reports_replay_blocker(self) -> None:
        child = self._draft_candidate_with_replay_episode("missing_evaluator")

        result = self._run_cli_json(
            [
                "peds-cycle",
                "--candidate-id",
                child["candidate_id"],
                "--operator",
                "owner",
                "--max-steps",
                "2",
            ]
        )

        self.assertEqual(result["steps_run"], 1)
        self.assertIn("replay_evaluator required", result["stopped_reason"])
        self.assertTrue(result["step_log"][0]["blocked"])
        self.assertEqual(get_candidate_current_state(self.conn, child["candidate_id"]), "draft_candidate")

    def test_peds_cycle_cli_requires_candidate_id(self) -> None:
        with self.assertRaises(SystemExit) as context:
            main(["peds-cycle", "--operator", "owner"])
        self.assertEqual(context.exception.code, 2)

    def test_peds_cycle_cli_requires_operator_for_run_mode(self) -> None:
        child = self._draft_candidate_with_replay_episode("missing_operator")

        with self.assertRaises(SystemExit) as context:
            main(["peds-cycle", "--candidate-id", child["candidate_id"]])

        self.assertEqual(context.exception.code, 2)

    def test_peds_cycle_cli_rejects_invalid_options_json(self) -> None:
        child = self._draft_candidate_with_replay_episode("bad_options")

        with self.assertRaises(SystemExit) as context:
            main(
                [
                    "peds-cycle",
                    "--candidate-id",
                    child["candidate_id"],
                    "--dry-run",
                    "--options-json",
                    "{bad",
                ]
            )

        self.assertEqual(context.exception.code, 2)

    def test_peds_cycle_cli_respects_auto_promote_flag(self) -> None:
        without_flag = self._shadow_passed_candidate("no_auto_promote")
        result_without_flag = self._run_cli_json(
            [
                "peds-cycle",
                "--candidate-id",
                without_flag["candidate_id"],
                "--operator",
                "owner",
                "--max-steps",
                "1",
            ]
        )

        self.assertEqual(result_without_flag["step_log"][0]["action"], "ready_for_promotion")
        self.assertEqual(get_candidate_current_state(self.conn, without_flag["candidate_id"]), "shadow_passed")

        with_flag = self._shadow_passed_candidate("auto_promote")
        result_with_flag = self._run_cli_json(
            [
                "peds-cycle",
                "--candidate-id",
                with_flag["candidate_id"],
                "--operator",
                "owner",
                "--auto-promote",
                "--max-steps",
                "1",
                "--options-json",
                '{"promotion_reason":"operator allowed CLI promotion"}',
            ]
        )

        self.assertEqual(result_with_flag["step_log"][0]["action"], "promote_candidate")
        self.assertEqual(get_candidate_current_state(self.conn, with_flag["candidate_id"]), "promoted")
