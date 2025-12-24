-- verus_dataset/schema.sql
-- SQLite schema for minimizer-driven Verus dataset mining

-- Metadata table for schema version and config
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Repository registry with pinned commits
CREATE TABLE IF NOT EXISTS repos (
    repo_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    url TEXT,
    local_path TEXT NOT NULL,
    pinned_rev TEXT,
    head_rev TEXT,
    license TEXT,
    fetched_at TEXT,
    notes TEXT
);

-- Candidate source files discovered by scanning
CREATE TABLE IF NOT EXISTS source_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    repo_id INTEGER NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    nlines INTEGER NOT NULL,
    verus_score INTEGER NOT NULL,
    dependency_free INTEGER NOT NULL CHECK (dependency_free IN (0, 1)),
    scanned_at TEXT NOT NULL,
    UNIQUE (repo_id, rel_path, sha256)
);

-- Reduction jobs tracking minimization attempts
CREATE TABLE IF NOT EXISTS reduction_jobs (
    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES source_files(file_id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('pass', 'fail')),
    reducer TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'timeout', 'error', 'skipped')),
    params_json TEXT NOT NULL,
    work_dir TEXT NOT NULL,
    input_relpath TEXT NOT NULL,
    output_relpath TEXT,
    expected_fingerprint TEXT,
    pre_exit_code INTEGER,
    post_exit_code INTEGER,
    pre_fingerprint TEXT,
    post_fingerprint TEXT,
    nbytes_before INTEGER,
    nbytes_after INTEGER,
    loc_before INTEGER,
    loc_after INTEGER,
    started_at TEXT,
    finished_at TEXT,
    log_relpath TEXT,
    notes TEXT
);

-- Final minimized samples ready for task generation
CREATE TABLE IF NOT EXISTS samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_uid TEXT NOT NULL UNIQUE,
    repo_id INTEGER NOT NULL REFERENCES repos(repo_id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES source_files(file_id) ON DELETE SET NULL,
    job_id INTEGER REFERENCES reduction_jobs(job_id) ON DELETE SET NULL,
    minimized INTEGER NOT NULL CHECK (minimized IN (0, 1)),
    verification_result TEXT NOT NULL CHECK (verification_result IN ('pass', 'fail')),
    fingerprint TEXT,
    code_sha256 TEXT NOT NULL,
    code_text TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    nlines INTEGER NOT NULL,
    segments_json TEXT,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Verification run logs for debugging and provenance
CREATE TABLE IF NOT EXISTS verify_runs (
    verify_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER REFERENCES source_files(file_id) ON DELETE CASCADE,
    job_id INTEGER REFERENCES reduction_jobs(job_id) ON DELETE CASCADE,
    sample_id INTEGER REFERENCES samples(sample_id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    command TEXT NOT NULL,
    cwd TEXT NOT NULL,
    exit_code INTEGER NOT NULL,
    timed_out INTEGER NOT NULL CHECK (timed_out IN (0, 1)),
    wall_time_sec REAL NOT NULL,
    stdout_trunc TEXT,
    stderr_trunc TEXT,
    fingerprint TEXT,
    created_at TEXT NOT NULL,
    -- Exactly one of file_id, job_id, sample_id must be set
    CHECK (
        (file_id IS NOT NULL) +
        (job_id IS NOT NULL) +
        (sample_id IS NOT NULL) = 1
    )
);

-- Generated training tasks derived from samples
CREATE TABLE IF NOT EXISTS tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_uid TEXT NOT NULL UNIQUE,
    sample_id INTEGER NOT NULL REFERENCES samples(sample_id) ON DELETE CASCADE,
    task_type TEXT NOT NULL CHECK (task_type IN ('spec_gen', 'code_synth', 'spec_and_code', 'repair')),
    split TEXT NOT NULL CHECK (split IN ('unsplit', 'train', 'val', 'test')),
    prompt_text TEXT NOT NULL,
    target_text TEXT,
    full_text TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_source_files_repo_score
    ON source_files(repo_id, verus_score DESC);

CREATE INDEX IF NOT EXISTS idx_source_files_dependency_free
    ON source_files(dependency_free, verus_score DESC);

CREATE INDEX IF NOT EXISTS idx_reduction_jobs_status
    ON reduction_jobs(status, file_id);

CREATE INDEX IF NOT EXISTS idx_samples_repo_result
    ON samples(repo_id, verification_result);

CREATE INDEX IF NOT EXISTS idx_samples_minimized
    ON samples(minimized, verification_result);

CREATE INDEX IF NOT EXISTS idx_tasks_type_split
    ON tasks(task_type, split);

CREATE INDEX IF NOT EXISTS idx_verify_runs_stage
    ON verify_runs(stage, created_at DESC);
