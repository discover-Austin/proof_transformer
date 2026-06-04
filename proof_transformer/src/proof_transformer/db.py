from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from .paths import get_brain_paths
from .timeutils import utc_now


# Keep in sync with the repo-root peds_schema.sql export.
PEDS_MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS peds_schema_meta (
    schema_name TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    parent_candidate_id TEXT,
    created_at TEXT NOT NULL,
    mutation_target TEXT NOT NULL CHECK (mutation_target IN ('phi', 'rho', 'theta', 'psi')),
    phi_version TEXT NOT NULL,
    rho_version TEXT NOT NULL,
    theta_version TEXT,
    psi_version TEXT NOT NULL,
    delta_description TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    FOREIGN KEY(parent_candidate_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_candidates_parent ON candidates(parent_candidate_id);
CREATE INDEX IF NOT EXISTS idx_candidates_target_created ON candidates(mutation_target, created_at);

CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('online', 'train', 'replay', 'canary')),
    timestamp TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_title TEXT,
    task_prompt TEXT NOT NULL,
    acceptance_tests_json TEXT,
    policy_tags_json TEXT,
    repo_commit TEXT,
    repo_tree_hash TEXT,
    retrieval_index_hash TEXT,
    env_image_hash TEXT,
    tool_profile TEXT,
    model_lock_json TEXT NOT NULL,
    candidate_id TEXT,
    context_json TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    patch TEXT,
    final_message TEXT,
    judge_verdict TEXT CHECK (judge_verdict IN ('pass', 'fail', 'unsafe', 'timeout')),
    user_verdict TEXT CHECK (user_verdict IN ('accepted', 'rejected', 'revised')),
    failure_reason TEXT,
    repair_trace TEXT,
    metrics_json TEXT NOT NULL,
    privacy_json TEXT NOT NULL,
    raw_episode_json TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_episodes_split_timestamp ON episodes(split, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task_id, split, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_candidate ON episodes(candidate_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_episodes_repo_commit ON episodes(repo_commit);

CREATE TABLE IF NOT EXISTS replay_jobs (
    job_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    split TEXT NOT NULL CHECK (split IN ('replay', 'canary')),
    candidate_id TEXT NOT NULL,
    incumbent_id TEXT NOT NULL,
    evaluator_version TEXT NOT NULL,
    runtime_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'aborted')),
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(incumbent_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_replay_jobs_status_created ON replay_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_replay_jobs_candidate ON replay_jobs(candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_replay_jobs_incumbent ON replay_jobs(incumbent_id, created_at);

CREATE TABLE IF NOT EXISTS replay_job_episodes (
    job_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    episode_order INTEGER NOT NULL,
    PRIMARY KEY (job_id, episode_id),
    UNIQUE (job_id, episode_order),
    FOREIGN KEY(job_id) REFERENCES replay_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
);
CREATE INDEX IF NOT EXISTS idx_replay_job_episodes_episode ON replay_job_episodes(episode_id);

CREATE TABLE IF NOT EXISTS replay_results (
    result_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    candidate_score REAL NOT NULL,
    incumbent_score REAL NOT NULL,
    delta REAL NOT NULL,
    judge_verdict TEXT NOT NULL CHECK (judge_verdict IN ('pass', 'fail', 'unsafe', 'timeout')),
    canary_failures_json TEXT NOT NULL,
    audit_log_ref TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (job_id, episode_id),
    FOREIGN KEY(job_id) REFERENCES replay_jobs(job_id) ON DELETE CASCADE,
    FOREIGN KEY(episode_id) REFERENCES episodes(episode_id)
);
CREATE INDEX IF NOT EXISTS idx_replay_results_job ON replay_results(job_id, episode_id);
CREATE INDEX IF NOT EXISTS idx_replay_results_delta ON replay_results(job_id, delta);

CREATE TABLE IF NOT EXISTS shadow_runs (
    shadow_run_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    incumbent_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'failed', 'aborted')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    thresholds_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(incumbent_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_shadow_runs_candidate_status ON shadow_runs(candidate_id, status, started_at);
CREATE INDEX IF NOT EXISTS idx_shadow_runs_incumbent_status ON shadow_runs(incumbent_id, status, started_at);

CREATE TABLE IF NOT EXISTS shadow_events (
    shadow_event_id TEXT PRIMARY KEY,
    shadow_run_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    task_ref TEXT,
    episode_ref TEXT,
    input_json TEXT NOT NULL,
    incumbent_output_json TEXT NOT NULL,
    candidate_output_json TEXT NOT NULL,
    divergence_json TEXT,
    candidate_would_act_json TEXT,
    incumbent_did_act_json TEXT,
    accepted_by_incumbent INTEGER,
    created_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    UNIQUE (shadow_run_id, ordinal),
    FOREIGN KEY(shadow_run_id) REFERENCES shadow_runs(shadow_run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_shadow_events_run_ordinal ON shadow_events(shadow_run_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_shadow_events_created_at ON shadow_events(created_at);

CREATE TABLE IF NOT EXISTS promotion_decisions (
    decision_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    incumbent_id TEXT NOT NULL,
    job_id TEXT,
    shadow_run_id TEXT,
    decision TEXT NOT NULL CHECK (
        decision IN (
            'rejected',
            'replay_winner',
            'canary_passed',
            'shadow_started',
            'shadow_passed',
            'shadow_failed',
            'shadow_deployment',
            'promoted',
            'distillation_candidate'
        )
    ),
    mean_delta REAL,
    lcb95 REAL,
    regression_rate REAL,
    canary_failures INTEGER,
    token_ratio REAL,
    latency_ratio REAL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    CHECK (
        (job_id IS NOT NULL AND shadow_run_id IS NULL)
        OR (job_id IS NULL AND shadow_run_id IS NOT NULL)
    ),
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(incumbent_id) REFERENCES candidates(candidate_id),
    FOREIGN KEY(job_id) REFERENCES replay_jobs(job_id),
    FOREIGN KEY(shadow_run_id) REFERENCES shadow_runs(shadow_run_id)
);
CREATE INDEX IF NOT EXISTS idx_promotion_decisions_candidate_created ON promotion_decisions(candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_promotion_decisions_job ON promotion_decisions(job_id, created_at);
CREATE INDEX IF NOT EXISTS idx_promotion_decisions_shadow_run ON promotion_decisions(shadow_run_id, created_at);
CREATE INDEX IF NOT EXISTS idx_promotion_decisions_decision ON promotion_decisions(decision, created_at);

CREATE TABLE IF NOT EXISTS peds_training_runs (
    training_run_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('prepared', 'queued', 'completed', 'failed', 'aborted')),
    dataset_hash TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    manifest_summary_json TEXT NOT NULL,
    trainer_name TEXT NOT NULL,
    trainer_config_json TEXT NOT NULL,
    output_dir TEXT,
    metrics_json TEXT NOT NULL,
    artifact_ref TEXT,
    error TEXT,
    FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_peds_training_runs_candidate_created ON peds_training_runs(candidate_id, created_at);
CREATE INDEX IF NOT EXISTS idx_peds_training_runs_status_created ON peds_training_runs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_peds_training_runs_dataset_hash ON peds_training_runs(dataset_hash);

CREATE VIEW IF NOT EXISTS candidate_current_state AS
SELECT
    c.candidate_id,
    COALESCE(
        (
            SELECT pd.decision
            FROM promotion_decisions pd
            WHERE pd.candidate_id = c.candidate_id
            ORDER BY pd.created_at DESC, pd.decision_id DESC
            LIMIT 1
        ),
        'draft_candidate'
    ) AS current_state
FROM candidates c;
"""


def _create_candidate_current_state_view(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIEW IF NOT EXISTS candidate_current_state AS
        SELECT
            c.candidate_id,
            COALESCE(
                (
                    SELECT pd.decision
                    FROM promotion_decisions pd
                    WHERE pd.candidate_id = c.candidate_id
                    ORDER BY pd.created_at DESC, pd.decision_id DESC
                    LIMIT 1
                ),
                'draft_candidate'
            ) AS current_state
        FROM candidates c;
        """
    )


def _ensure_shadow_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shadow_runs (
            shadow_run_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            incumbent_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'completed', 'failed', 'aborted')),
            started_at TEXT NOT NULL,
            completed_at TEXT,
            thresholds_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY(incumbent_id) REFERENCES candidates(candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_runs_candidate_status ON shadow_runs(candidate_id, status, started_at);
        CREATE INDEX IF NOT EXISTS idx_shadow_runs_incumbent_status ON shadow_runs(incumbent_id, status, started_at);

        CREATE TABLE IF NOT EXISTS shadow_events (
            shadow_event_id TEXT PRIMARY KEY,
            shadow_run_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            task_ref TEXT,
            episode_ref TEXT,
            input_json TEXT NOT NULL,
            incumbent_output_json TEXT NOT NULL,
            candidate_output_json TEXT NOT NULL,
            divergence_json TEXT,
            candidate_would_act_json TEXT,
            incumbent_did_act_json TEXT,
            accepted_by_incumbent INTEGER,
            created_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            UNIQUE (shadow_run_id, ordinal),
            FOREIGN KEY(shadow_run_id) REFERENCES shadow_runs(shadow_run_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_events_run_ordinal ON shadow_events(shadow_run_id, ordinal);
        CREATE INDEX IF NOT EXISTS idx_shadow_events_created_at ON shadow_events(created_at);
        """
    )


def _table_has_column(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def _rebuild_promotion_decisions_for_shadow(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS candidate_current_state")
    conn.execute("ALTER TABLE promotion_decisions RENAME TO promotion_decisions_old")
    conn.executescript(
        """
        CREATE TABLE promotion_decisions (
            decision_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            incumbent_id TEXT NOT NULL,
            job_id TEXT,
            shadow_run_id TEXT,
            decision TEXT NOT NULL CHECK (
                decision IN (
                    'rejected',
                    'replay_winner',
                    'canary_passed',
                    'shadow_started',
                    'shadow_passed',
                    'shadow_failed',
                    'shadow_deployment',
                    'promoted',
                    'distillation_candidate'
                )
            ),
            mean_delta REAL,
            lcb95 REAL,
            regression_rate REAL,
            canary_failures INTEGER,
            token_ratio REAL,
            latency_ratio REAL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            CHECK (
                (job_id IS NOT NULL AND shadow_run_id IS NULL)
                OR (job_id IS NULL AND shadow_run_id IS NOT NULL)
            ),
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY(incumbent_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY(job_id) REFERENCES replay_jobs(job_id),
            FOREIGN KEY(shadow_run_id) REFERENCES shadow_runs(shadow_run_id)
        );
        """
    )

    if _table_has_column(conn, "promotion_decisions_old", "shadow_run_id"):
        conn.execute(
            """
            INSERT INTO promotion_decisions(
                decision_id, candidate_id, incumbent_id, job_id, shadow_run_id, decision,
                mean_delta, lcb95, regression_rate, canary_failures,
                token_ratio, latency_ratio, reason, created_at
            )
            SELECT
                decision_id, candidate_id, incumbent_id, job_id, shadow_run_id, decision,
                mean_delta, lcb95, regression_rate, canary_failures,
                token_ratio, latency_ratio, reason, created_at
            FROM promotion_decisions_old
            """
        )
    else:
        conn.execute(
            """
            INSERT INTO promotion_decisions(
                decision_id, candidate_id, incumbent_id, job_id, shadow_run_id, decision,
                mean_delta, lcb95, regression_rate, canary_failures,
                token_ratio, latency_ratio, reason, created_at
            )
            SELECT
                decision_id, candidate_id, incumbent_id, job_id, NULL, decision,
                mean_delta, lcb95, regression_rate, canary_failures,
                token_ratio, latency_ratio, reason, created_at
            FROM promotion_decisions_old
            """
        )

    conn.execute("DROP TABLE promotion_decisions_old")
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_promotion_decisions_candidate_created ON promotion_decisions(candidate_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_promotion_decisions_job ON promotion_decisions(job_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_promotion_decisions_shadow_run ON promotion_decisions(shadow_run_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_promotion_decisions_decision ON promotion_decisions(decision, created_at);
        """
    )
    _create_candidate_current_state_view(conn)


def _migrate_peds_shadow(conn: sqlite3.Connection) -> None:
    _ensure_shadow_tables(conn)
    if not _table_has_column(conn, "promotion_decisions", "shadow_run_id"):
        _rebuild_promotion_decisions_for_shadow(conn)
        return
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_promotion_decisions_shadow_run ON promotion_decisions(shadow_run_id, created_at)"
    )
    conn.execute("DROP VIEW IF EXISTS candidate_current_state")
    _create_candidate_current_state_view(conn)


def _ensure_peds_training_runs(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS peds_training_runs (
            training_run_id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL CHECK (status IN ('prepared', 'queued', 'completed', 'failed', 'aborted')),
            dataset_hash TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            manifest_summary_json TEXT NOT NULL,
            trainer_name TEXT NOT NULL,
            trainer_config_json TEXT NOT NULL,
            output_dir TEXT,
            metrics_json TEXT NOT NULL,
            artifact_ref TEXT,
            error TEXT,
            FOREIGN KEY(candidate_id) REFERENCES candidates(candidate_id)
        );
        CREATE INDEX IF NOT EXISTS idx_peds_training_runs_candidate_created ON peds_training_runs(candidate_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_peds_training_runs_status_created ON peds_training_runs(status, created_at);
        CREATE INDEX IF NOT EXISTS idx_peds_training_runs_dataset_hash ON peds_training_runs(dataset_hash);
        """
    )


MIGRATIONS: list[tuple[int, str | Callable[[sqlite3.Connection], None]]] = [
    (
        1,
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS memory_nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            content TEXT NOT NULL,
            summary TEXT,
            source_uri TEXT,
            source_hash TEXT,
            confidence REAL NOT NULL,
            importance REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memory_nodes_kind_name ON memory_nodes(kind, name);
        CREATE INDEX IF NOT EXISTS idx_memory_nodes_name ON memory_nodes(name);

        CREATE TABLE IF NOT EXISTS memory_edges (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            evidence TEXT,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES memory_nodes(id) ON DELETE CASCADE,
            FOREIGN KEY(target_id) REFERENCES memory_nodes(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_memory_edges_source ON memory_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_memory_edges_target ON memory_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_memory_edges_relation ON memory_edges(relation);

        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            path TEXT,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            source_uri TEXT,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_estimate INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);

        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            vector_json TEXT NOT NULL,
            model_name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS retrieval_events (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            selected_chunk_ids_json TEXT NOT NULL,
            selected_node_ids_json TEXT NOT NULL,
            scores_json TEXT NOT NULL,
            used_in_answer INTEGER NOT NULL DEFAULT 0,
            feedback_score REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_retrieval_events_created_at ON retrieval_events(created_at);

        CREATE TABLE IF NOT EXISTS learning_events (
            id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            source_text TEXT NOT NULL,
            context_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS training_candidates (
            id TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            instruction TEXT NOT NULL,
            chosen TEXT NOT NULL,
            rejected TEXT,
            tags_json TEXT NOT NULL,
            proposed_change_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES learning_events(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_training_candidates_status ON training_candidates(status);

        CREATE TABLE IF NOT EXISTS validator_results (
            id TEXT PRIMARY KEY,
            candidate_id TEXT NOT NULL,
            validator_name TEXT NOT NULL,
            passed INTEGER NOT NULL,
            score REAL NOT NULL,
            reason TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(candidate_id) REFERENCES training_candidates(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_validator_results_candidate ON validator_results(candidate_id);

        CREATE TABLE IF NOT EXISTS policy_snapshots (
            id TEXT PRIMARY KEY,
            version INTEGER NOT NULL UNIQUE,
            policy_json TEXT NOT NULL,
            policy_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS learning_blocks (
            id TEXT PRIMARY KEY,
            height INTEGER NOT NULL UNIQUE,
            parent_id TEXT,
            parent_hash TEXT,
            candidate_id TEXT NOT NULL,
            policy_version INTEGER,
            adapter_version TEXT,
            ranker_version TEXT,
            memory_diff_json TEXT NOT NULL,
            validator_summary_json TEXT NOT NULL,
            consensus_score REAL NOT NULL,
            status TEXT NOT NULL,
            block_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_learning_blocks_status ON learning_blocks(status);

        CREATE TABLE IF NOT EXISTS ranker_weights (
            id TEXT PRIMARY KEY,
            version INTEGER NOT NULL UNIQUE,
            weights_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS adapter_registry (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            path TEXT NOT NULL,
            parent_version TEXT,
            dataset_hash TEXT,
            eval_report_hash TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_adapter_registry_name_version ON adapter_registry(name, version);
        """,
    ),
    (
        2,
        """
        CREATE TABLE IF NOT EXISTS task_queue (
            id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            result_json TEXT,
            error_text TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_task_queue_status_priority ON task_queue(status, priority, created_at);

        CREATE TABLE IF NOT EXISTS task_runs (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            status TEXT NOT NULL,
            output_json TEXT,
            error_text TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY(task_id) REFERENCES task_queue(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_task_runs_task ON task_runs(task_id, created_at);
        """,
    ),
    (
        3,
        """
        CREATE TABLE IF NOT EXISTS workspace_roots (
            id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_scan_at TEXT
        );

        CREATE TABLE IF NOT EXISTS workspace_files (
            id TEXT PRIMARY KEY,
            workspace_root_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_at TEXT NOT NULL,
            document_id TEXT,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_indexed_at TEXT NOT NULL,
            FOREIGN KEY(workspace_root_id) REFERENCES workspace_roots(id) ON DELETE CASCADE,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL,
            UNIQUE(workspace_root_id, relative_path)
        );
        CREATE INDEX IF NOT EXISTS idx_workspace_files_root_active ON workspace_files(workspace_root_id, active, relative_path);
        CREATE INDEX IF NOT EXISTS idx_workspace_files_file_path ON workspace_files(file_path);
        """,
    ),
    (
        4,
        """
        CREATE TABLE IF NOT EXISTS execution_runs (
            id TEXT PRIMARY KEY,
            command_json TEXT NOT NULL,
            cwd TEXT,
            status TEXT NOT NULL,
            exit_code INTEGER,
            stdout_path TEXT,
            stderr_path TEXT,
            output_document_id TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            metadata_json TEXT NOT NULL,
            FOREIGN KEY(output_document_id) REFERENCES documents(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_execution_runs_started_at ON execution_runs(started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_execution_runs_status ON execution_runs(status, started_at DESC);
        """,
    ),
    (
        5,
        """
        CREATE TABLE IF NOT EXISTS repair_plans (
            id TEXT PRIMARY KEY,
            workspace_root TEXT NOT NULL,
            command_json TEXT NOT NULL,
            execution_id TEXT NOT NULL,
            analysis_text TEXT NOT NULL,
            plan_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            applied_at TEXT,
            result_execution_id TEXT,
            FOREIGN KEY(execution_id) REFERENCES execution_runs(id) ON DELETE CASCADE,
            FOREIGN KEY(result_execution_id) REFERENCES execution_runs(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_repair_plans_status_created ON repair_plans(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_repair_plans_workspace ON repair_plans(workspace_root, created_at DESC);
        """,
    ),
    (6, PEDS_MIGRATION_SQL),
    (7, _migrate_peds_shadow),
    (8, _ensure_peds_training_runs),
]


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    paths = get_brain_paths()
    paths.ensure()
    target = Path(db_path) if db_path else paths.db
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    initialize_database(conn)
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
    )
    for version, script in MIGRATIONS:
        applied = conn.execute(
            "SELECT 1 AS found FROM schema_migrations WHERE version = ? LIMIT 1",
            (version,),
        ).fetchone()
        if applied:
            continue
        with conn:
            if callable(script):
                script(conn)
            else:
                conn.executescript(script)
            conn.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                (version, utc_now()),
            )


def table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


def view_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}
