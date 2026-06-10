"""Cost accounting and cost-cap decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import CostSummary, PodRecord, PodRole, WorkerState
from .store import SQLiteStore, utc_now


ACTIVE_PROVIDER_STATES = {
    "created",
    "creating",
    "initializing",
    "provisioning",
    "running",
    "active",
    "ready",
}


@dataclass(frozen=True)
class DrainCandidate:
    worker_id: str
    pod_id: str | None
    hourly_rate: float


class CostManager:
    """Computes burn rate, spend, ETA, and cap decisions from server state."""

    def __init__(self, store: SQLiteStore):
        self.store = store

    def summarize(
        self,
        *,
        now: float | None = None,
        additional_hourly_rate: float = 0.0,
    ) -> CostSummary:
        ts = now if now is not None else utc_now()
        cap_cfg = self.store.get_cost_config()
        cap = cap_cfg["cap_per_hour"]
        hard_cap = cap_cfg["hard_cap"]
        burn = self.current_burn_rate(now=ts)
        spent = self.spent_so_far(now=ts)
        eta = self.estimate_eta_seconds(now=ts)
        cost_to_completion = None if eta is None else burn * eta / 3600.0
        projected = burn + additional_hourly_rate
        allowed = cap is None or projected <= float(cap)
        reason = None if allowed else f"projected hourly burn ${projected:.2f}/hr exceeds cap ${float(cap):.2f}/hr"
        return CostSummary(
            current_burn_rate_per_hr=burn,
            spent_so_far=spent,
            eta_seconds=eta,
            estimated_cost_to_completion=cost_to_completion,
            cap_per_hour=cap,
            hard_cap_enabled=hard_cap,
            scale_up_allowed=allowed,
            scale_up_block_reason=reason,
        )

    def current_burn_rate(self, *, now: float | None = None) -> float:
        _ = now if now is not None else utc_now()
        return sum(self._rate(pod) for pod in self._active_pods())

    def spent_so_far(self, *, now: float | None = None) -> float:
        ts = now if now is not None else utc_now()
        spent = 0.0
        for pod in self.store.list_pods():
            if pod.start_time is None:
                continue
            end = pod.stop_time if pod.stop_time is not None else ts
            if end <= pod.start_time:
                continue
            spent += self._rate(pod) * ((end - pod.start_time) / 3600.0)
        return spent

    def estimate_eta_seconds(self, *, now: float | None = None) -> float | None:
        ts = now if now is not None else utc_now()
        snapshot = self.store.progress_snapshot(now=ts)
        if snapshot.remaining_units <= 0:
            return 0.0
        if snapshot.completed_units <= 0 or snapshot.earliest_activity_at is None:
            return None
        elapsed = max(1.0, ts - snapshot.earliest_activity_at)
        throughput = snapshot.completed_units / elapsed
        if throughput <= 0:
            return None
        return snapshot.remaining_units / throughput

    def can_add_worker(self, hourly_rate: float, *, now: float | None = None) -> tuple[bool, str | None]:
        summary = self.summarize(now=now, additional_hourly_rate=hourly_rate)
        return summary.scale_up_allowed, summary.scale_up_block_reason

    def hard_cap_drain_candidates(self, *, now: float | None = None) -> list[DrainCandidate]:
        """Return idle active workers that can be drained to bring burn under cap."""

        ts = now if now is not None else utc_now()
        cap = self.store.get_cost_config()["cap_per_hour"]
        if cap is None:
            return []
        excess = self.current_burn_rate(now=ts) - float(cap)
        if excess <= 0:
            return []
        pod_by_worker = {
            pod.worker_id: pod
            for pod in self._active_pods()
            if pod.role == PodRole.WORKER and pod.worker_id
        }
        candidates: list[DrainCandidate] = []
        workers = self.store.list_workers([WorkerState.ACTIVE])
        for worker in workers:
            if self.store.running_job_count_for_worker(worker.worker_id) > 0:
                continue
            pod = pod_by_worker.get(worker.worker_id)
            candidates.append(
                DrainCandidate(
                    worker_id=worker.worker_id,
                    pod_id=pod.pod_id if pod else None,
                    hourly_rate=self._rate(pod) if pod else 0.0,
                )
            )
        candidates.sort(key=lambda item: item.hourly_rate, reverse=True)
        selected: list[DrainCandidate] = []
        removed = 0.0
        for candidate in candidates:
            selected.append(candidate)
            removed += candidate.hourly_rate
            if removed >= excess:
                break
        return selected

    def request_hard_cap_drains(self, *, now: float | None = None) -> list[DrainCandidate]:
        candidates = self.hard_cap_drain_candidates(now=now)
        for candidate in candidates:
            self.store.request_worker_drain(candidate.worker_id, now=now)
        return candidates

    def _active_pods(self) -> Iterable[PodRecord]:
        for pod in self.store.list_pods():
            if pod.start_time is None:
                continue
            if pod.stop_time is not None:
                continue
            if pod.provider_status.lower() in ACTIVE_PROVIDER_STATES:
                yield pod

    @staticmethod
    def _rate(pod: PodRecord | None) -> float:
        if pod is None:
            return 0.0
        return float(pod.adjusted_cost_per_hr if pod.adjusted_cost_per_hr is not None else pod.cost_per_hr)
