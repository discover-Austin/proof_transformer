-- Personal Evolutionary Distillation System (PEDS)
-- Protocol schema v0.1
--
-- Portable baseline for SQLite and Postgres.
-- JSON payloads are stored as TEXT so the recorder/replay service can remain
-- database-agnostic.
--
-- Append-only contract:
--   - candidates
--   - episodes
--   - replay_results
--   - promotion_decisions
--
-- Keep those tables immutable at the application layer or add vendor-specific
-- UPDATE/DELETE blocking triggers when you bind this schema to a concrete DB.

BEGIN;

CREATE TABLE IF NOT EXISTS peds_schema_meta (
  schema_name TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL,
  notes TEXT,
  created_at TEXT NOT NULL
);

-- Immutable candidate manifests. Lifecycle state is derived from the most
-- recent promotion_decisions row rather than mutating this table in place.
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

-- Recorded task trajectories.
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

-- Pairwise candidate-vs-incumbent evaluation jobs.
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

-- Frozen episode membership for a replay job.
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

-- Per-episode pairwise replay results.
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

-- Immutable promotion / rejection audit trail.
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

-- Operational training-run ledger. This records training intent/results only;
-- candidate lifecycle remains append-only in promotion_decisions.
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

-- Derived current lifecycle state without mutating candidates.
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

COMMIT;
