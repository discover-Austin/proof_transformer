from __future__ import annotations

from common import BrainTestCase
from proof_transformer.db import table_names, view_names


class PEDSSchemaTests(BrainTestCase):
    def test_peds_tables_and_view_exist(self) -> None:
        required_tables = {
            "peds_schema_meta",
            "candidates",
            "episodes",
            "replay_jobs",
            "replay_job_episodes",
            "replay_results",
            "shadow_runs",
            "shadow_events",
            "promotion_decisions",
            "peds_training_runs",
        }
        self.assertTrue(required_tables.issubset(table_names(self.conn)))
        self.assertIn("candidate_current_state", view_names(self.conn))

        versions = {
            row["version"]
            for row in self.conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
        self.assertIn(6, versions)
        self.assertIn(7, versions)
        self.assertIn(8, versions)

    def test_candidate_current_state_is_derived_from_latest_decision(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO candidates(
                    candidate_id, parent_candidate_id, created_at, mutation_target,
                    phi_version, rho_version, theta_version, psi_version,
                    delta_description, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "candidate_a",
                    None,
                    "2026-06-02T00:00:00Z",
                    "phi",
                    "phi_v1",
                    "rho_v1",
                    None,
                    "psi_v1",
                    "baseline candidate",
                    "{}",
                ),
            )
            self.conn.execute(
                """
                INSERT INTO candidates(
                    candidate_id, parent_candidate_id, created_at, mutation_target,
                    phi_version, rho_version, theta_version, psi_version,
                    delta_description, manifest_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "candidate_b",
                    None,
                    "2026-06-02T00:00:00Z",
                    "phi",
                    "phi_v1",
                    "rho_v1",
                    None,
                    "psi_v1",
                    "incumbent candidate",
                    "{}",
                ),
            )

            state = self.conn.execute(
                "SELECT current_state FROM candidate_current_state WHERE candidate_id = ?",
                ("candidate_a",),
            ).fetchone()
            self.assertEqual(state["current_state"], "draft_candidate")

            self.conn.execute(
                """
                INSERT INTO replay_jobs(
                    job_id, created_at, started_at, finished_at, split,
                    candidate_id, incumbent_id, evaluator_version, runtime_json, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "job_1",
                    "2026-06-02T00:01:00Z",
                    "2026-06-02T00:01:01Z",
                    "2026-06-02T00:01:10Z",
                    "replay",
                    "candidate_a",
                    "candidate_b",
                    "eval_v1",
                    "{}",
                    "completed",
                ),
            )
            self.conn.execute(
                """
                INSERT INTO promotion_decisions(
                    decision_id, candidate_id, incumbent_id, job_id, decision,
                    mean_delta, lcb95, regression_rate, canary_failures,
                    token_ratio, latency_ratio, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "decision_1",
                    "candidate_a",
                    "candidate_b",
                    "job_1",
                    "replay_winner",
                    0.08,
                    0.03,
                    0.10,
                    0,
                    0.95,
                    0.90,
                    "Replay confidence exceeded threshold.",
                    "2026-06-02T00:02:00Z",
                ),
            )
            self.conn.execute(
                """
                INSERT INTO promotion_decisions(
                    decision_id, candidate_id, incumbent_id, job_id, decision,
                    mean_delta, lcb95, regression_rate, canary_failures,
                    token_ratio, latency_ratio, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "decision_2",
                    "candidate_a",
                    "candidate_b",
                    "job_1",
                    "promoted",
                    0.08,
                    0.03,
                    0.10,
                    0,
                    0.95,
                    0.90,
                    "Shadow deployment remained clean.",
                    "2026-06-02T00:03:00Z",
                ),
            )

        state = self.conn.execute(
            "SELECT current_state FROM candidate_current_state WHERE candidate_id = ?",
            ("candidate_a",),
        ).fetchone()
        self.assertEqual(state["current_state"], "promoted")
