"""SQLite-backed server state.

The server is the only process that writes this database. Workers communicate
state changes through HTTP control-plane calls and never open the SQLite file.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .models import Job, JobEvent, JobState, PodRecord, PodRole, ProgressSnapshot, Worker, WorkerState


SCHEMA_VERSION = 1
DEFAULT_RUN_ID = "default"


def utc_now() -> float:
    return time.time()


def _to_json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, sort_keys=True)


def _from_json(value: str | None) -> Any:
    if not value:
        return {}
    return json.loads(value)


class SQLiteStore:
    """Durable scheduler state with atomic job claims."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def initialize(self, reset: bool = False) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            if reset:
                con.executescript(
                    """
                    DROP TABLE IF EXISTS schema_meta;
                    DROP TABLE IF EXISTS jobs;
                    DROP TABLE IF EXISTS workers;
                    DROP TABLE IF EXISTS pods;
                    DROP TABLE IF EXISTS config;
                    DROP TABLE IF EXISTS job_events;
                    """
                )
            con.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL,
                    priority INTEGER NOT NULL DEFAULT 0,
                    total_units REAL NOT NULL DEFAULT 1,
                    completed_units REAL NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    lease_expires_at REAL,
                    last_progress_at REAL,
                    result_json TEXT,
                    error TEXT,
                    output_path TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_state_priority
                    ON jobs(state, priority DESC, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_jobs_worker
                    ON jobs(worker_id, state);
                CREATE TABLE IF NOT EXISTS workers (
                    worker_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    state TEXT NOT NULL,
                    host TEXT,
                    labels_json TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    last_heartbeat_at REAL,
                    drain_requested_at REAL,
                    drained_at REAL,
                    current_job_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_workers_state
                    ON workers(state, last_heartbeat_at);
                CREATE TABLE IF NOT EXISTS pods (
                    pod_id TEXT PRIMARY KEY,
                    role TEXT NOT NULL,
                    provider_status TEXT NOT NULL,
                    run_id TEXT,
                    worker_id TEXT,
                    gpu_type TEXT,
                    cost_per_hr REAL NOT NULL DEFAULT 0,
                    adjusted_cost_per_hr REAL,
                    start_time REAL,
                    stop_time REAL,
                    data_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    worker_id TEXT,
                    event_type TEXT NOT NULL,
                    message TEXT,
                    created_at REAL NOT NULL,
                    data_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_job_events_job
                    ON job_events(job_id, id);
                CREATE INDEX IF NOT EXISTS idx_job_events_worker
                    ON job_events(worker_id, id);
                CREATE INDEX IF NOT EXISTS idx_job_events_type
                    ON job_events(event_type, id);
                """
            )
            con.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            con.execute("PRAGMA foreign_keys=ON")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    @contextmanager
    def transaction(self, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        with self.connect() as con:
            if immediate:
                con.execute("BEGIN IMMEDIATE")
            yield con

    def export_backup(self, target_path: str | Path) -> Path:
        """Write a consistent SQLite backup for Server Pod volume export."""

        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as src, sqlite3.connect(target) as dst:
            src.backup(dst)
        return target

    def set_config(self, key: str, value: Any, now: float | None = None) -> None:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO config(key, value_json, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, _to_json(value), ts),
            )

    def get_config(self, key: str, default: Any = None) -> Any:
        with self.connect() as con:
            row = con.execute("SELECT value_json FROM config WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return _from_json(row["value_json"])

    def add_event(
        self,
        *,
        event_type: str,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        job_id: str | None = None,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> None:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            self._add_event(con, job_id, worker_id, event_type, message, data or {}, ts)

    def set_cost_cap(self, cap_per_hour: float | None, hard_cap: bool = False) -> None:
        self.set_config("cost", {"cap_per_hour": cap_per_hour, "hard_cap": bool(hard_cap)})

    def get_cost_config(self) -> dict[str, Any]:
        value = self.get_config("cost", {})
        return {
            "cap_per_hour": value.get("cap_per_hour"),
            "hard_cap": bool(value.get("hard_cap", False)),
        }

    def set_run_paused(self, run_id: str, paused: bool, now: float | None = None) -> None:
        self.set_config(f"run:{run_id}:paused", {"paused": bool(paused)}, now=now)

    def is_run_paused(self, run_id: str | None) -> bool:
        if run_id is None:
            return False
        value = self.get_config(f"run:{run_id}:paused", {})
        return bool(value.get("paused", False))

    def _is_run_paused_in_connection(self, con: sqlite3.Connection, run_id: str | None) -> bool:
        if run_id is None:
            return False
        row = con.execute("SELECT value_json FROM config WHERE key=?", (f"run:{run_id}:paused",)).fetchone()
        if row is None:
            return False
        return bool(_from_json(row["value_json"]).get("paused", False))

    def enqueue_job(
        self,
        task_type: str,
        payload: dict[str, Any],
        *,
        run_id: str = DEFAULT_RUN_ID,
        total_units: float = 1.0,
        priority: int = 0,
        max_attempts: int = 1,
        output_path: str | None = None,
        job_id: str | None = None,
        now: float | None = None,
    ) -> str:
        ts = now if now is not None else utc_now()
        jid = job_id or uuid.uuid4().hex
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO jobs(
                    id, run_id, task_type, payload_json, state, priority,
                    total_units, completed_units, attempt, max_attempts,
                    created_at, updated_at, output_path
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?)
                """,
                (
                    jid,
                    run_id,
                    task_type,
                    _to_json(payload),
                    JobState.QUEUED.value,
                    priority,
                    float(total_units),
                    int(max_attempts),
                    ts,
                    ts,
                    output_path,
                ),
            )
            self._add_event(con, jid, None, "queued", None, payload, ts)
        return jid

    def register_worker(
        self,
        *,
        worker_id: str | None = None,
        run_id: str = DEFAULT_RUN_ID,
        role: str = "worker",
        host: str | None = None,
        labels: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> Worker:
        ts = now if now is not None else utc_now()
        wid = worker_id or uuid.uuid4().hex
        with self.connect() as con:
            existing = con.execute(
                "SELECT state FROM workers WHERE worker_id=?", (wid,)
            ).fetchone()
            state = existing["state"] if existing is not None else WorkerState.ACTIVE.value
            if state in {WorkerState.STALE.value, WorkerState.LOST.value, WorkerState.STOPPED.value}:
                state = WorkerState.ACTIVE.value
            con.execute(
                """
                INSERT INTO workers(
                    worker_id, run_id, role, state, host, labels_json,
                    registered_at, last_heartbeat_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    run_id=excluded.run_id,
                    role=excluded.role,
                    host=excluded.host,
                    labels_json=excluded.labels_json,
                    last_heartbeat_at=excluded.last_heartbeat_at,
                    state=CASE
                        WHEN workers.state IN ('draining', 'drained') THEN workers.state
                        ELSE excluded.state
                    END
                """,
                (wid, run_id, role, state, host, _to_json(labels or {}), ts, ts),
            )
            self._add_event(con, None, wid, "worker_registered", None, labels or {}, ts)
        return self.get_worker(wid)

    def heartbeat_worker(
        self,
        worker_id: str,
        *,
        current_job_id: str | None = None,
        now: float | None = None,
    ) -> Worker:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = con.execute("SELECT state FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown worker: {worker_id}")
            state = row["state"]
            if state in {WorkerState.STALE.value, WorkerState.LOST.value}:
                state = WorkerState.ACTIVE.value
            con.execute(
                """
                UPDATE workers
                SET last_heartbeat_at=?, current_job_id=?, state=?
                WHERE worker_id=?
                """,
                (ts, current_job_id, state, worker_id),
            )
        return self.get_worker(worker_id)

    def get_worker(self, worker_id: str) -> Worker:
        with self.connect() as con:
            row = con.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown worker: {worker_id}")
        return self._worker_from_row(row)

    def list_workers(self, states: Sequence[WorkerState | str] | None = None) -> list[Worker]:
        with self.connect() as con:
            if states:
                values = [s.value if isinstance(s, WorkerState) else s for s in states]
                placeholders = ",".join("?" for _ in values)
                rows = con.execute(f"SELECT * FROM workers WHERE state IN ({placeholders})", values).fetchall()
            else:
                rows = con.execute("SELECT * FROM workers ORDER BY registered_at").fetchall()
        return [self._worker_from_row(row) for row in rows]

    def request_worker_drain(self, worker_id: str, now: float | None = None) -> Worker:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = con.execute("SELECT state FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown worker: {worker_id}")
            if row["state"] != WorkerState.DRAINED.value:
                con.execute(
                    """
                    UPDATE workers
                    SET state=?, drain_requested_at=COALESCE(drain_requested_at, ?)
                    WHERE worker_id=?
                    """,
                    (WorkerState.DRAINING.value, ts, worker_id),
                )
            self._add_event(con, None, worker_id, "drain_requested", None, {}, ts)
        return self.get_worker(worker_id)

    def request_run_drain(
        self,
        *,
        run_id: str,
        role: str | None = None,
        now: float | None = None,
    ) -> list[Worker]:
        ts = now if now is not None else utc_now()
        values: list[Any] = [run_id, WorkerState.ACTIVE.value, WorkerState.DRAINING.value]
        query = """
            SELECT worker_id FROM workers
            WHERE run_id=? AND state IN (?, ?)
        """
        if role:
            query += " AND role=?"
            values.append(role)
        with self.connect() as con:
            rows = con.execute(query, values).fetchall()
            for row in rows:
                con.execute(
                    """
                    UPDATE workers
                    SET state=?, drain_requested_at=COALESCE(drain_requested_at, ?)
                    WHERE worker_id=? AND state != ?
                    """,
                    (WorkerState.DRAINING.value, ts, row["worker_id"], WorkerState.DRAINED.value),
                )
                self._add_event(con, None, row["worker_id"], "run_drain_requested", None, {"run_id": run_id}, ts)
        return [self.get_worker(row["worker_id"]) for row in rows]

    def ack_worker_drained(self, worker_id: str, now: float | None = None) -> Worker:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            active = con.execute(
                """
                SELECT COUNT(*) AS n FROM jobs
                WHERE worker_id=? AND state=?
                """,
                (worker_id, JobState.RUNNING.value),
            ).fetchone()["n"]
            if active:
                raise RuntimeError(f"worker {worker_id} still has {active} running job(s)")
            con.execute(
                """
                UPDATE workers
                SET state=?, drained_at=?, current_job_id=NULL
                WHERE worker_id=?
                """,
                (WorkerState.DRAINED.value, ts, worker_id),
            )
            self._add_event(con, None, worker_id, "drained", None, {}, ts)
        return self.get_worker(worker_id)

    def get_worker_command(self, worker_id: str) -> str | None:
        worker = self.get_worker(worker_id)
        if worker.state == WorkerState.DRAINING:
            return "drain"
        return None

    def claim_job(
        self,
        worker_id: str,
        *,
        lease_seconds: int = 300,
        run_id: str | None = None,
        now: float | None = None,
    ) -> Job | None:
        ts = now if now is not None else utc_now()
        lease_until = ts + lease_seconds
        with self.transaction(immediate=True) as con:
            worker = con.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            if worker is None:
                raise KeyError(f"unknown worker: {worker_id}")
            if worker["state"] != WorkerState.ACTIVE.value:
                return None
            effective_run_id = run_id or worker["run_id"]
            if self._is_run_paused_in_connection(con, effective_run_id):
                return None
            active = con.execute(
                """
                SELECT * FROM jobs
                WHERE worker_id=? AND state=?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (worker_id, JobState.RUNNING.value),
            ).fetchone()
            if active is not None:
                return self._job_from_row(active)
            values: list[Any] = [JobState.QUEUED.value, JobState.RETRYABLE.value]
            query = """
                SELECT * FROM jobs
                WHERE state IN (?, ?)
            """
            if run_id is not None:
                query += " AND run_id=?"
                values.append(run_id)
            query += " ORDER BY priority DESC, created_at ASC LIMIT 1"
            row = con.execute(query, values).fetchone()
            if row is None:
                return None
            next_attempt = int(row["attempt"]) + 1
            con.execute(
                """
                UPDATE jobs
                SET state=?, worker_id=?, attempt=?, lease_expires_at=?,
                    started_at=COALESCE(started_at, ?), updated_at=?,
                    last_progress_at=?
                WHERE id=?
                """,
                (
                    JobState.RUNNING.value,
                    worker_id,
                    next_attempt,
                    lease_until,
                    ts,
                    ts,
                    ts,
                    row["id"],
                ),
            )
            con.execute(
                "UPDATE workers SET current_job_id=?, last_heartbeat_at=? WHERE worker_id=?",
                (row["id"], ts, worker_id),
            )
            self._add_event(con, row["id"], worker_id, "claimed", None, {"attempt": next_attempt}, ts)
            claimed = con.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        return self._job_from_row(claimed)

    def claim_jobs(
        self,
        worker_id: str,
        *,
        max_jobs: int,
        lease_seconds: int = 300,
        run_id: str | None = None,
        task_type: str | None = None,
        now: float | None = None,
    ) -> list[Job]:
        """Claim a batch of queued jobs for a worker.

        Batch claims support worker-local prefetch pipelines. Each claimed job
        still completes or fails independently, so retries remain slide-scoped.
        """

        if max_jobs <= 0:
            return []
        ts = now if now is not None else utc_now()
        lease_until = ts + lease_seconds
        claimed_ids: list[str] = []
        with self.transaction(immediate=True) as con:
            worker = con.execute("SELECT * FROM workers WHERE worker_id=?", (worker_id,)).fetchone()
            if worker is None:
                raise KeyError(f"unknown worker: {worker_id}")
            if worker["state"] != WorkerState.ACTIVE.value:
                return []
            effective_run_id = run_id or worker["run_id"]
            if self._is_run_paused_in_connection(con, effective_run_id):
                return []
            values: list[Any] = [JobState.QUEUED.value, JobState.RETRYABLE.value]
            query = """
                SELECT * FROM jobs
                WHERE state IN (?, ?)
            """
            if run_id is not None:
                query += " AND run_id=?"
                values.append(run_id)
            if task_type is not None:
                query += " AND task_type=?"
                values.append(task_type)
            query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
            values.append(int(max_jobs))
            rows = con.execute(query, values).fetchall()
            for row in rows:
                next_attempt = int(row["attempt"]) + 1
                con.execute(
                    """
                    UPDATE jobs
                    SET state=?, worker_id=?, attempt=?, lease_expires_at=?,
                        started_at=COALESCE(started_at, ?), updated_at=?,
                        last_progress_at=?
                    WHERE id=?
                    """,
                    (
                        JobState.RUNNING.value,
                        worker_id,
                        next_attempt,
                        lease_until,
                        ts,
                        ts,
                        ts,
                        row["id"],
                    ),
                )
                claimed_ids.append(row["id"])
                self._add_event(con, row["id"], worker_id, "batch_claimed", None, {"attempt": next_attempt}, ts)
            current_job_id = claimed_ids[0] if claimed_ids else None
            con.execute(
                "UPDATE workers SET current_job_id=?, last_heartbeat_at=? WHERE worker_id=?",
                (current_job_id, ts, worker_id),
            )
            if claimed_ids:
                placeholders = ",".join("?" for _ in claimed_ids)
                claimed_rows = con.execute(f"SELECT * FROM jobs WHERE id IN ({placeholders})", claimed_ids).fetchall()
            else:
                claimed_rows = []
        jobs_by_id = {row["id"]: self._job_from_row(row) for row in claimed_rows}
        return [jobs_by_id[job_id] for job_id in claimed_ids]

    def report_progress(
        self,
        worker_id: str,
        job_id: str,
        *,
        completed_units: float,
        total_units: float | None = None,
        message: str | None = None,
        lease_seconds: int | None = None,
        now: float | None = None,
    ) -> Job:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = self._get_running_job_for_worker(con, worker_id, job_id)
            total = float(total_units) if total_units is not None else float(row["total_units"])
            completed = max(0.0, min(float(completed_units), total))
            lease_until = ts + lease_seconds if lease_seconds else row["lease_expires_at"]
            con.execute(
                """
                UPDATE jobs
                SET completed_units=?, total_units=?, last_progress_at=?,
                    lease_expires_at=?, updated_at=?
                WHERE id=?
                """,
                (completed, total, ts, lease_until, ts, job_id),
            )
            con.execute(
                "UPDATE workers SET last_heartbeat_at=?, current_job_id=? WHERE worker_id=?",
                (ts, job_id, worker_id),
            )
            self._add_event(
                con,
                job_id,
                worker_id,
                "progress",
                message,
                {"completed_units": completed, "total_units": total},
                ts,
            )
        return self.get_job(job_id)

    def complete_job(
        self,
        worker_id: str,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> Job:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = self._get_running_job_for_worker(con, worker_id, job_id)
            total = float(row["total_units"])
            output_path = (result or {}).get("output_path") or row["output_path"]
            con.execute(
                """
                UPDATE jobs
                SET state=?, completed_units=?, completed_at=?, updated_at=?,
                    lease_expires_at=NULL, result_json=?, error=NULL, output_path=?
                WHERE id=?
                """,
                (
                    JobState.COMPLETED.value,
                    total,
                    ts,
                    ts,
                    _to_json(result or {}),
                    output_path,
                    job_id,
                ),
            )
            con.execute(
                "UPDATE workers SET current_job_id=NULL, last_heartbeat_at=? WHERE worker_id=?",
                (ts, worker_id),
            )
            self._add_event(con, job_id, worker_id, "completed", None, result or {}, ts)
        return self.get_job(job_id)

    def fail_job(
        self,
        worker_id: str,
        job_id: str,
        *,
        error: str,
        retryable: bool = True,
        now: float | None = None,
    ) -> Job:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = self._get_running_job_for_worker(con, worker_id, job_id)
            can_retry = retryable and int(row["attempt"]) < int(row["max_attempts"])
            next_state = JobState.RETRYABLE.value if can_retry else JobState.FAILED.value
            con.execute(
                """
                UPDATE jobs
                SET state=?, worker_id=NULL, lease_expires_at=NULL,
                    updated_at=?, error=?
                WHERE id=?
                """,
                (next_state, ts, error, job_id),
            )
            con.execute(
                "UPDATE workers SET current_job_id=NULL, last_heartbeat_at=? WHERE worker_id=?",
                (ts, worker_id),
            )
            self._add_event(
                con,
                job_id,
                worker_id,
                "failed",
                error,
                {"retryable": can_retry, "next_state": next_state},
                ts,
            )
        return self.get_job(job_id)

    def release_claimed_job(
        self,
        worker_id: str,
        job_id: str,
        *,
        reason: str = "released by worker",
        now: float | None = None,
    ) -> Job:
        """Return a claimed-but-unstarted job to the queue.

        Batch workers claim multiple jobs to enable local prefetch. When a drain
        is requested, the worker can finish its current slide and release any
        unstarted batch jobs without consuming retry attempts.
        """

        ts = now if now is not None else utc_now()
        with self.connect() as con:
            row = self._get_running_job_for_worker(con, worker_id, job_id)
            next_attempt = max(0, int(row["attempt"]) - 1)
            con.execute(
                """
                UPDATE jobs
                SET state=?, worker_id=NULL, attempt=?, lease_expires_at=NULL,
                    completed_units=0, updated_at=?, error=NULL
                WHERE id=?
                """,
                (JobState.QUEUED.value, next_attempt, ts, job_id),
            )
            con.execute(
                """
                UPDATE workers
                SET current_job_id=NULL, last_heartbeat_at=?
                WHERE worker_id=? AND current_job_id=?
                """,
                (ts, worker_id, job_id),
            )
            self._add_event(con, job_id, worker_id, "released", reason, {"next_state": JobState.QUEUED.value}, ts)
        return self.get_job(job_id)

    def cancel_job(self, job_id: str, now: float | None = None) -> Job:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            con.execute(
                """
                UPDATE jobs
                SET state=?, updated_at=?, lease_expires_at=NULL
                WHERE id=? AND state IN (?, ?)
                """,
                (
                    JobState.CANCELLED.value,
                    ts,
                    job_id,
                    JobState.QUEUED.value,
                    JobState.RETRYABLE.value,
                ),
            )
            self._add_event(con, job_id, None, "cancelled", None, {}, ts)
        return self.get_job(job_id)

    def recover_stale_jobs(
        self,
        *,
        now: float | None = None,
        worker_timeout_seconds: int = 600,
    ) -> dict[str, int]:
        ts = now if now is not None else utc_now()
        recovered_jobs = 0
        failed_jobs = 0
        stale_workers = 0
        with self.transaction(immediate=True) as con:
            stale_worker_rows = con.execute(
                """
                SELECT worker_id FROM workers
                WHERE state IN (?, ?)
                  AND last_heartbeat_at IS NOT NULL
                  AND last_heartbeat_at < ?
                """,
                (
                    WorkerState.ACTIVE.value,
                    WorkerState.DRAINING.value,
                    ts - worker_timeout_seconds,
                ),
            ).fetchall()
            for row in stale_worker_rows:
                stale_workers += 1
                con.execute(
                    "UPDATE workers SET state=? WHERE worker_id=?",
                    (WorkerState.STALE.value, row["worker_id"]),
                )
                self._add_event(con, None, row["worker_id"], "worker_stale", None, {}, ts)
            expired = con.execute(
                """
                SELECT * FROM jobs
                WHERE state=? AND lease_expires_at IS NOT NULL AND lease_expires_at < ?
                """,
                (JobState.RUNNING.value, ts),
            ).fetchall()
            for row in expired:
                if int(row["attempt"]) < int(row["max_attempts"]):
                    state = JobState.RETRYABLE.value
                    recovered_jobs += 1
                else:
                    state = JobState.FAILED.value
                    failed_jobs += 1
                con.execute(
                    """
                    UPDATE jobs
                    SET state=?, worker_id=NULL, lease_expires_at=NULL,
                        updated_at=?, error=COALESCE(error, ?)
                    WHERE id=?
                    """,
                    (state, ts, "lease expired", row["id"]),
                )
                if row["worker_id"]:
                    con.execute(
                        """
                        UPDATE workers SET current_job_id=NULL
                        WHERE worker_id=? AND current_job_id=?
                        """,
                        (row["worker_id"], row["id"]),
                    )
                self._add_event(
                    con,
                    row["id"],
                    row["worker_id"],
                    "lease_expired",
                    None,
                    {"next_state": state},
                    ts,
                )
        return {"recovered_jobs": recovered_jobs, "failed_jobs": failed_jobs, "stale_workers": stale_workers}

    def get_job(self, job_id: str) -> Job:
        with self.connect() as con:
            row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown job: {job_id}")
        return self._job_from_row(row)

    def list_jobs(self, states: Sequence[JobState | str] | None = None) -> list[Job]:
        with self.connect() as con:
            if states:
                values = [s.value if isinstance(s, JobState) else s for s in states]
                placeholders = ",".join("?" for _ in values)
                rows = con.execute(
                    f"SELECT * FROM jobs WHERE state IN ({placeholders}) ORDER BY created_at",
                    values,
                ).fetchall()
            else:
                rows = con.execute("SELECT * FROM jobs ORDER BY created_at").fetchall()
        return [self._job_from_row(row) for row in rows]

    def list_events(
        self,
        *,
        job_id: str | None = None,
        worker_id: str | None = None,
        event_type: str | None = None,
        since_id: int | None = None,
        limit: int = 100,
        newest_first: bool = False,
    ) -> list[JobEvent]:
        clauses: list[str] = []
        values: list[Any] = []
        if job_id:
            clauses.append("job_id=?")
            values.append(job_id)
        if worker_id:
            clauses.append("worker_id=?")
            values.append(worker_id)
        if event_type:
            clauses.append("event_type=?")
            values.append(event_type)
        if since_id is not None:
            clauses.append("id>?")
            values.append(int(since_id))
        query = "SELECT * FROM job_events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC" if newest_first else " ORDER BY id ASC"
        query += " LIMIT ?"
        values.append(max(1, min(int(limit), 1000)))
        with self.connect() as con:
            rows = con.execute(query, values).fetchall()
        return [self._event_from_row(row) for row in rows]

    def latest_progress_by_job(self, job_ids: Sequence[str] | None = None) -> dict[str, JobEvent]:
        values: list[Any] = ["progress"]
        query = "SELECT * FROM job_events WHERE event_type=? AND job_id IS NOT NULL"
        if job_ids is not None:
            ids = [job_id for job_id in job_ids if job_id]
            if not ids:
                return {}
            placeholders = ",".join("?" for _ in ids)
            query += f" AND job_id IN ({placeholders})"
            values.extend(ids)
        query += " ORDER BY id ASC"
        latest: dict[str, JobEvent] = {}
        with self.connect() as con:
            rows = con.execute(query, values).fetchall()
        for row in rows:
            event = self._event_from_row(row)
            if event.job_id:
                latest[event.job_id] = event
        return latest

    def job_state_counts(self) -> dict[str, int]:
        with self.connect() as con:
            rows = con.execute("SELECT state, COUNT(*) AS n FROM jobs GROUP BY state").fetchall()
        return {row["state"]: int(row["n"]) for row in rows}

    def progress_snapshot(self, now: float | None = None) -> ProgressSnapshot:
        _ = now if now is not None else utc_now()
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT state, total_units, completed_units, created_at, started_at
                FROM jobs
                """
            ).fetchall()
        total = 0.0
        completed = 0.0
        running = 0.0
        earliest: float | None = None
        for row in rows:
            total_units = float(row["total_units"])
            completed_units = float(row["completed_units"])
            total += total_units
            if row["state"] == JobState.COMPLETED.value:
                completed += total_units
            else:
                completed += min(completed_units, total_units)
            if row["state"] == JobState.RUNNING.value:
                running += min(completed_units, total_units)
            activity = row["started_at"] or row["created_at"]
            if activity is not None:
                earliest = activity if earliest is None else min(earliest, float(activity))
        remaining = max(0.0, total - completed)
        return ProgressSnapshot(total, completed, running, remaining, earliest)

    def upsert_pod(
        self,
        *,
        pod_id: str,
        role: PodRole | str,
        provider_status: str,
        run_id: str | None = None,
        worker_id: str | None = None,
        gpu_type: str | None = None,
        cost_per_hr: float = 0.0,
        adjusted_cost_per_hr: float | None = None,
        start_time: float | None = None,
        stop_time: float | None = None,
        data: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> PodRecord:
        ts = now if now is not None else utc_now()
        role_value = role.value if isinstance(role, PodRole) else role
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO pods(
                    pod_id, role, provider_status, run_id, worker_id, gpu_type,
                    cost_per_hr, adjusted_cost_per_hr, start_time, stop_time,
                    data_json, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pod_id) DO UPDATE SET
                    role=excluded.role,
                    provider_status=excluded.provider_status,
                    run_id=excluded.run_id,
                    worker_id=excluded.worker_id,
                    gpu_type=excluded.gpu_type,
                    cost_per_hr=excluded.cost_per_hr,
                    adjusted_cost_per_hr=excluded.adjusted_cost_per_hr,
                    start_time=COALESCE(excluded.start_time, pods.start_time),
                    stop_time=excluded.stop_time,
                    data_json=excluded.data_json,
                    updated_at=excluded.updated_at
                """,
                (
                    pod_id,
                    role_value,
                    provider_status,
                    run_id,
                    worker_id,
                    gpu_type,
                    float(cost_per_hr),
                    adjusted_cost_per_hr,
                    start_time,
                    stop_time,
                    _to_json(data or {}),
                    ts,
                    ts,
                ),
            )
        return self.get_pod(pod_id)

    def stop_pod_record(self, pod_id: str, provider_status: str = "stopped", now: float | None = None) -> PodRecord:
        ts = now if now is not None else utc_now()
        with self.connect() as con:
            con.execute(
                """
                UPDATE pods
                SET provider_status=?, stop_time=COALESCE(stop_time, ?), updated_at=?
                WHERE pod_id=?
                """,
                (provider_status, ts, ts, pod_id),
            )
        return self.get_pod(pod_id)

    def get_pod(self, pod_id: str) -> PodRecord:
        with self.connect() as con:
            row = con.execute("SELECT * FROM pods WHERE pod_id=?", (pod_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown pod: {pod_id}")
        return self._pod_from_row(row)

    def list_pods(self) -> list[PodRecord]:
        with self.connect() as con:
            rows = con.execute("SELECT * FROM pods ORDER BY created_at").fetchall()
        return [self._pod_from_row(row) for row in rows]

    def running_job_count_for_worker(self, worker_id: str) -> int:
        with self.connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE worker_id=? AND state=?",
                (worker_id, JobState.RUNNING.value),
            ).fetchone()
        return int(row["n"])

    def _get_running_job_for_worker(self, con: sqlite3.Connection, worker_id: str, job_id: str) -> sqlite3.Row:
        row = con.execute(
            """
            SELECT * FROM jobs
            WHERE id=? AND worker_id=? AND state=?
            """,
            (job_id, worker_id, JobState.RUNNING.value),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"job {job_id} is not running on worker {worker_id}")
        return row

    def _add_event(
        self,
        con: sqlite3.Connection,
        job_id: str | None,
        worker_id: str | None,
        event_type: str,
        message: str | None,
        data: dict[str, Any],
        created_at: float,
    ) -> None:
        con.execute(
            """
            INSERT INTO job_events(job_id, worker_id, event_type, message, created_at, data_json)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (job_id, worker_id, event_type, message, created_at, _to_json(data)),
        )

    def _event_from_row(self, row: sqlite3.Row) -> JobEvent:
        return JobEvent(
            id=int(row["id"]),
            job_id=row["job_id"],
            worker_id=row["worker_id"],
            event_type=row["event_type"],
            message=row["message"],
            created_at=float(row["created_at"]),
            data=_from_json(row["data_json"]),
        )

    def _job_from_row(self, row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            run_id=row["run_id"],
            task_type=row["task_type"],
            payload=_from_json(row["payload_json"]),
            state=JobState(row["state"]),
            priority=int(row["priority"]),
            total_units=float(row["total_units"]),
            completed_units=float(row["completed_units"]),
            attempt=int(row["attempt"]),
            max_attempts=int(row["max_attempts"]),
            worker_id=row["worker_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            lease_expires_at=row["lease_expires_at"],
            last_progress_at=row["last_progress_at"],
            result=_from_json(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            output_path=row["output_path"],
        )

    def _worker_from_row(self, row: sqlite3.Row) -> Worker:
        return Worker(
            worker_id=row["worker_id"],
            run_id=row["run_id"],
            role=row["role"],
            state=WorkerState(row["state"]),
            host=row["host"],
            labels=_from_json(row["labels_json"]),
            registered_at=float(row["registered_at"]),
            last_heartbeat_at=row["last_heartbeat_at"],
            drain_requested_at=row["drain_requested_at"],
            drained_at=row["drained_at"],
            current_job_id=row["current_job_id"],
        )

    def _pod_from_row(self, row: sqlite3.Row) -> PodRecord:
        return PodRecord(
            pod_id=row["pod_id"],
            role=PodRole(row["role"]),
            provider_status=row["provider_status"],
            run_id=row["run_id"],
            worker_id=row["worker_id"],
            gpu_type=row["gpu_type"],
            cost_per_hr=float(row["cost_per_hr"]),
            adjusted_cost_per_hr=row["adjusted_cost_per_hr"],
            start_time=row["start_time"],
            stop_time=row["stop_time"],
            data=_from_json(row["data_json"]),
        )
