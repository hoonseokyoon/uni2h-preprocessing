"""Shared data models for the distributed execution prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYABLE = "retryable"
    CANCELLED = "cancelled"


class WorkerState(str, Enum):
    PROVISIONING = "provisioning"
    ACTIVE = "active"
    DRAINING = "draining"
    DRAINED = "drained"
    STALE = "stale"
    LOST = "lost"
    STOPPED = "stopped"


class PodRole(str, Enum):
    SERVER = "server"
    WORKER = "worker"


@dataclass(frozen=True)
class Job:
    id: str
    run_id: str
    task_type: str
    payload: dict[str, Any]
    state: JobState
    priority: int
    total_units: float
    completed_units: float
    attempt: int
    max_attempts: int
    worker_id: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    completed_at: float | None = None
    lease_expires_at: float | None = None
    last_progress_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    output_path: str | None = None


@dataclass(frozen=True)
class Worker:
    worker_id: str
    run_id: str
    role: str
    state: WorkerState
    host: str | None = None
    labels: dict[str, Any] = field(default_factory=dict)
    registered_at: float = 0.0
    last_heartbeat_at: float | None = None
    drain_requested_at: float | None = None
    drained_at: float | None = None
    current_job_id: str | None = None


@dataclass(frozen=True)
class PodRecord:
    pod_id: str
    role: PodRole
    provider_status: str
    run_id: str | None
    cost_per_hr: float
    adjusted_cost_per_hr: float | None = None
    worker_id: str | None = None
    gpu_type: str | None = None
    start_time: float | None = None
    stop_time: float | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProgressSnapshot:
    total_units: float
    completed_units: float
    running_units: float
    remaining_units: float
    earliest_activity_at: float | None


@dataclass(frozen=True)
class CostSummary:
    current_burn_rate_per_hr: float
    spent_so_far: float
    eta_seconds: float | None
    estimated_cost_to_completion: float | None
    cap_per_hour: float | None
    hard_cap_enabled: bool
    scale_up_allowed: bool
    scale_up_block_reason: str | None = None


@dataclass(frozen=True)
class JobEvent:
    id: int
    job_id: str | None
    worker_id: str | None
    event_type: str
    message: str | None
    created_at: float
    data: dict[str, Any] = field(default_factory=dict)
