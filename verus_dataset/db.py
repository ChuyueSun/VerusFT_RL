"""
verus_dataset/db.py - SQLite database utilities for the dataset pipeline.

Provides a Database class that wraps sqlite3 with convenience methods
for all pipeline operations.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
SCHEMA_VERSION = "1.0.0"


def init_db(db_path: str | Path) -> "Database":
    """Initialize a new database with the schema, or connect to existing."""
    db = Database(db_path)
    db.ensure_schema()
    return db


class Database:
    """SQLite database wrapper for the Verus dataset pipeline."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for a transaction with automatic commit/rollback."""
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        schema_sql = SCHEMA_PATH.read_text()
        self.conn.executescript(schema_sql)
        # Set schema version
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value."""
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    # ─────────────────────────────────────────────────────────────────────────
    # Repos
    # ─────────────────────────────────────────────────────────────────────────

    def add_repo(
        self,
        name: str,
        local_path: str,
        url: Optional[str] = None,
        pinned_rev: Optional[str] = None,
        head_rev: Optional[str] = None,
        license: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Add or update a repository entry. Returns repo_id."""
        now = datetime.utcnow().isoformat()
        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO repos (name, url, local_path, pinned_rev, head_rev, license, fetched_at, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    url = excluded.url,
                    local_path = excluded.local_path,
                    pinned_rev = excluded.pinned_rev,
                    head_rev = excluded.head_rev,
                    license = excluded.license,
                    fetched_at = excluded.fetched_at,
                    notes = excluded.notes
                """,
                (name, url, local_path, pinned_rev, head_rev, license, now, notes),
            )
        # Fetch the repo_id
        row = self.conn.execute(
            "SELECT repo_id FROM repos WHERE name = ?", (name,)
        ).fetchone()
        return row["repo_id"]

    def get_repo(self, name: str) -> Optional[sqlite3.Row]:
        """Get a repo by name."""
        return self.conn.execute(
            "SELECT * FROM repos WHERE name = ?", (name,)
        ).fetchone()

    def get_repo_by_id(self, repo_id: int) -> Optional[sqlite3.Row]:
        """Get a repo by ID."""
        return self.conn.execute(
            "SELECT * FROM repos WHERE repo_id = ?", (repo_id,)
        ).fetchone()

    def list_repos(self) -> list[sqlite3.Row]:
        """List all repos."""
        return self.conn.execute("SELECT * FROM repos ORDER BY name").fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Source Files
    # ─────────────────────────────────────────────────────────────────────────

    def add_source_file(
        self,
        repo_id: int,
        rel_path: str,
        abs_path: str,
        sha256: str,
        nbytes: int,
        nlines: int,
        verus_score: int,
        dependency_free: bool,
    ) -> int:
        """Add a source file entry. Returns file_id."""
        now = datetime.utcnow().isoformat()
        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO source_files
                    (repo_id, rel_path, abs_path, sha256, nbytes, nlines, verus_score, dependency_free, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(repo_id, rel_path, sha256) DO UPDATE SET
                    abs_path = excluded.abs_path,
                    nbytes = excluded.nbytes,
                    nlines = excluded.nlines,
                    verus_score = excluded.verus_score,
                    dependency_free = excluded.dependency_free,
                    scanned_at = excluded.scanned_at
                """,
                (
                    repo_id,
                    rel_path,
                    abs_path,
                    sha256,
                    nbytes,
                    nlines,
                    verus_score,
                    1 if dependency_free else 0,
                    now,
                ),
            )
        # Fetch file_id
        row = self.conn.execute(
            "SELECT file_id FROM source_files WHERE repo_id = ? AND rel_path = ? AND sha256 = ?",
            (repo_id, rel_path, sha256),
        ).fetchone()
        return row["file_id"]

    def get_source_file(self, file_id: int) -> Optional[sqlite3.Row]:
        """Get a source file by ID."""
        return self.conn.execute(
            "SELECT * FROM source_files WHERE file_id = ?", (file_id,)
        ).fetchone()

    def list_source_files(
        self,
        repo_id: Optional[int] = None,
        min_score: int = 0,
        dependency_free_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        """List source files with optional filters."""
        query = "SELECT * FROM source_files WHERE verus_score >= ?"
        params: list[Any] = [min_score]

        if repo_id is not None:
            query += " AND repo_id = ?"
            params.append(repo_id)

        if dependency_free_only:
            query += " AND dependency_free = 1"

        query += " ORDER BY verus_score DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        return self.conn.execute(query, params).fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Reduction Jobs
    # ─────────────────────────────────────────────────────────────────────────

    def add_reduction_job(
        self,
        file_id: int,
        mode: str,
        reducer: str,
        params_json: str,
        work_dir: str,
        input_relpath: str,
    ) -> int:
        """Create a new reduction job. Returns job_id."""
        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO reduction_jobs
                    (file_id, mode, reducer, status, params_json, work_dir, input_relpath)
                VALUES (?, ?, ?, 'pending', ?, ?, ?)
                """,
                (file_id, mode, reducer, params_json, work_dir, input_relpath),
            )
            return cursor.lastrowid

    def update_job_status(
        self,
        job_id: int,
        status: str,
        **kwargs: Any,
    ) -> None:
        """Update a job's status and optional fields."""
        updates = ["status = ?"]
        params: list[Any] = [status]

        for key, value in kwargs.items():
            if key in (
                "output_relpath",
                "expected_fingerprint",
                "pre_exit_code",
                "post_exit_code",
                "pre_fingerprint",
                "post_fingerprint",
                "nbytes_before",
                "nbytes_after",
                "loc_before",
                "loc_after",
                "started_at",
                "finished_at",
                "log_relpath",
                "notes",
            ):
                updates.append(f"{key} = ?")
                params.append(value)

        params.append(job_id)
        query = f"UPDATE reduction_jobs SET {', '.join(updates)} WHERE job_id = ?"
        with self.transaction():
            self.conn.execute(query, params)

    def get_job(self, job_id: int) -> Optional[sqlite3.Row]:
        """Get a reduction job by ID."""
        return self.conn.execute(
            "SELECT * FROM reduction_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

    def list_pending_jobs(self, limit: Optional[int] = None) -> list[sqlite3.Row]:
        """List pending reduction jobs."""
        query = "SELECT * FROM reduction_jobs WHERE status = 'pending' ORDER BY job_id"
        if limit is not None:
            query += f" LIMIT {limit}"
        return self.conn.execute(query).fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Samples
    # ─────────────────────────────────────────────────────────────────────────

    def add_sample(
        self,
        sample_uid: str,
        repo_id: int,
        minimized: bool,
        verification_result: str,
        code_sha256: str,
        code_text: str,
        meta_json: str,
        file_id: Optional[int] = None,
        job_id: Optional[int] = None,
        fingerprint: Optional[str] = None,
        segments_json: Optional[str] = None,
    ) -> int:
        """Add a sample. Returns sample_id."""
        now = datetime.utcnow().isoformat()
        nlines = len(code_text.splitlines())
        nbytes = len(code_text.encode("utf-8"))

        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO samples
                    (sample_uid, repo_id, file_id, job_id, minimized, verification_result,
                     fingerprint, code_sha256, code_text, nbytes, nlines, segments_json, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample_uid,
                    repo_id,
                    file_id,
                    job_id,
                    1 if minimized else 0,
                    verification_result,
                    fingerprint,
                    code_sha256,
                    code_text,
                    nbytes,
                    nlines,
                    segments_json,
                    meta_json,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_sample(self, sample_id: int) -> Optional[sqlite3.Row]:
        """Get a sample by ID."""
        return self.conn.execute(
            "SELECT * FROM samples WHERE sample_id = ?", (sample_id,)
        ).fetchone()

    def get_sample_by_uid(self, sample_uid: str) -> Optional[sqlite3.Row]:
        """Get a sample by UID."""
        return self.conn.execute(
            "SELECT * FROM samples WHERE sample_uid = ?", (sample_uid,)
        ).fetchone()

    def list_samples(
        self,
        repo_id: Optional[int] = None,
        verification_result: Optional[str] = None,
        minimized_only: bool = False,
        limit: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        """List samples with optional filters."""
        query = "SELECT * FROM samples WHERE 1=1"
        params: list[Any] = []

        if repo_id is not None:
            query += " AND repo_id = ?"
            params.append(repo_id)

        if verification_result is not None:
            query += " AND verification_result = ?"
            params.append(verification_result)

        if minimized_only:
            query += " AND minimized = 1"

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        return self.conn.execute(query, params).fetchall()

    # ─────────────────────────────────────────────────────────────────────────
    # Verify Runs
    # ─────────────────────────────────────────────────────────────────────────

    def add_verify_run(
        self,
        stage: str,
        command: str,
        cwd: str,
        exit_code: int,
        timed_out: bool,
        wall_time_sec: float,
        file_id: Optional[int] = None,
        job_id: Optional[int] = None,
        sample_id: Optional[int] = None,
        stdout_trunc: Optional[str] = None,
        stderr_trunc: Optional[str] = None,
        fingerprint: Optional[str] = None,
    ) -> int:
        """Log a verification run. Returns verify_id."""
        now = datetime.utcnow().isoformat()

        # Exactly one of file_id, job_id, sample_id must be set
        assert sum(x is not None for x in (file_id, job_id, sample_id)) == 1

        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO verify_runs
                    (file_id, job_id, sample_id, stage, command, cwd, exit_code,
                     timed_out, wall_time_sec, stdout_trunc, stderr_trunc, fingerprint, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    job_id,
                    sample_id,
                    stage,
                    command,
                    cwd,
                    exit_code,
                    1 if timed_out else 0,
                    wall_time_sec,
                    stdout_trunc,
                    stderr_trunc,
                    fingerprint,
                    now,
                ),
            )
            return cursor.lastrowid

    # ─────────────────────────────────────────────────────────────────────────
    # Tasks
    # ─────────────────────────────────────────────────────────────────────────

    def add_task(
        self,
        task_uid: str,
        sample_id: int,
        task_type: str,
        prompt_text: str,
        full_text: str,
        meta_json: str,
        target_text: Optional[str] = None,
        split: str = "unsplit",
    ) -> int:
        """Add a training task. Returns task_id."""
        now = datetime.utcnow().isoformat()

        with self.transaction():
            cursor = self.conn.execute(
                """
                INSERT INTO tasks
                    (task_uid, sample_id, task_type, split, prompt_text, target_text, full_text, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_uid,
                    sample_id,
                    task_type,
                    split,
                    prompt_text,
                    target_text,
                    full_text,
                    meta_json,
                    now,
                ),
            )
            return cursor.lastrowid

    def get_task(self, task_id: int) -> Optional[sqlite3.Row]:
        """Get a task by ID."""
        return self.conn.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

    def get_task_by_uid(self, task_uid: str) -> Optional[sqlite3.Row]:
        """Get a task by UID."""
        return self.conn.execute(
            "SELECT * FROM tasks WHERE task_uid = ?", (task_uid,)
        ).fetchone()

    def list_tasks(
        self,
        task_type: Optional[str] = None,
        split: Optional[str] = None,
        sample_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[sqlite3.Row]:
        """List tasks with optional filters."""
        query = "SELECT * FROM tasks WHERE 1=1"
        params: list[Any] = []

        if task_type is not None:
            query += " AND task_type = ?"
            params.append(task_type)

        if split is not None:
            query += " AND split = ?"
            params.append(split)

        if sample_id is not None:
            query += " AND sample_id = ?"
            params.append(sample_id)

        query += " ORDER BY created_at DESC"

        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        return self.conn.execute(query, params).fetchall()

    def count_tasks(
        self,
        task_type: Optional[str] = None,
        split: Optional[str] = None,
    ) -> int:
        """Count tasks with optional filters."""
        query = "SELECT COUNT(*) as cnt FROM tasks WHERE 1=1"
        params: list[Any] = []

        if task_type is not None:
            query += " AND task_type = ?"
            params.append(task_type)

        if split is not None:
            query += " AND split = ?"
            params.append(split)

        row = self.conn.execute(query, params).fetchone()
        return row["cnt"]

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        stats = {}

        # Repo count
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM repos").fetchone()
        stats["repos"] = row["cnt"]

        # Source file counts
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM source_files").fetchone()
        stats["source_files_total"] = row["cnt"]

        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM source_files WHERE dependency_free = 1"
        ).fetchone()
        stats["source_files_dependency_free"] = row["cnt"]

        # Reduction job stats
        for status in ("pending", "running", "done", "timeout", "error", "skipped"):
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM reduction_jobs WHERE status = ?",
                (status,),
            ).fetchone()
            stats[f"jobs_{status}"] = row["cnt"]

        # Sample counts
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM samples").fetchone()
        stats["samples_total"] = row["cnt"]

        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM samples WHERE verification_result = 'pass'"
        ).fetchone()
        stats["samples_pass"] = row["cnt"]

        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM samples WHERE verification_result = 'fail'"
        ).fetchone()
        stats["samples_fail"] = row["cnt"]

        # Task counts
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM tasks").fetchone()
        stats["tasks_total"] = row["cnt"]

        for task_type in ("spec_gen", "code_synth", "spec_and_code", "repair"):
            row = self.conn.execute(
                "SELECT COUNT(*) as cnt FROM tasks WHERE task_type = ?",
                (task_type,),
            ).fetchone()
            stats[f"tasks_{task_type}"] = row["cnt"]

        return stats
