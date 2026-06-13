"""Scheduler-facing operations composed from store, cost, and RunPod client."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cost import CostManager, DrainCandidate
from .models import PodRole
from .runpod_client import RunPodClient, WorkerPodSpec
from .store import DEFAULT_RUN_ID, SQLiteStore, utc_now


@dataclass(frozen=True)
class AddWorkerPlan:
    allowed: bool
    dry_run: bool
    payload: dict[str, Any]
    reason: str | None = None
    pod_response: dict[str, Any] | None = None


class Scheduler:
    def __init__(self, store: SQLiteStore, runpod_client: RunPodClient | None = None):
        self.store = store
        self.cost = CostManager(store)
        self.runpod_client = runpod_client

    def enqueue_demo_jobs(
        self,
        *,
        count: int,
        output_dir: str | Path,
        run_id: str = DEFAULT_RUN_ID,
        work_units: int = 5,
        delay_seconds: float = 0.0,
        max_attempts: int = 2,
    ) -> list[str]:
        output_root = Path(output_dir)
        job_ids: list[str] = []
        for index in range(count):
            payload = {
                "output_path": str(output_root / f"demo_job_{index + 1:04d}.txt"),
                "work_units": work_units,
                "delay_seconds": delay_seconds,
                "content": f"run={run_id};index={index + 1}",
            }
            job_id = self.store.enqueue_job(
                "demo_file",
                payload,
                run_id=run_id,
                total_units=work_units,
                max_attempts=max_attempts,
                output_path=payload["output_path"],
            )
            job_ids.append(job_id)
        return job_ids

    def add_worker(
        self,
        *,
        name: str,
        image_name: str,
        server_pod_id: str,
        server_port: int,
        run_id: str,
        workspace_root: str,
        gpu_type_ids: list[str],
        hourly_cost: float,
        worker_token: str | None = None,
        worker_role: str = "wsi-preprocess",
        gpu_count: int = 1,
        network_volume_id: str | None = None,
        data_center_ids: list[str] | None = None,
        adjusted_hourly_cost: float | None = None,
        dry_run: bool = True,
        env: dict[str, str] | None = None,
        docker_entrypoint: list[str] | None = None,
        docker_start_cmd: list[str] | None = None,
        worker_id: str | None = None,
        now: float | None = None,
    ) -> AddWorkerPlan:
        ts = now if now is not None else utc_now()
        token = worker_token or secrets.token_urlsafe(32)
        extra_env = dict(env or {})
        assigned_worker_id = worker_id or extra_env.get("WORKER_ID") or f"worker-{secrets.token_hex(6)}"
        extra_env["WORKER_ID"] = assigned_worker_id
        spec = WorkerPodSpec(
            name=name,
            image_name=image_name,
            server_pod_id=server_pod_id,
            server_port=server_port,
            run_id=run_id,
            worker_token=token,
            worker_role=worker_role,
            workspace_root=workspace_root,
            gpu_type_ids=gpu_type_ids,
            gpu_count=gpu_count,
            network_volume_id=network_volume_id,
            data_center_ids=data_center_ids,
            env=extra_env,
            docker_entrypoint=docker_entrypoint,
            docker_start_cmd=docker_start_cmd,
        )
        payload = RunPodClient.create_worker_payload(spec)
        projected_rate = adjusted_hourly_cost if adjusted_hourly_cost is not None else hourly_cost
        allowed, reason = self.cost.can_add_worker(projected_rate, now=ts)
        if not allowed:
            return AddWorkerPlan(False, dry_run, payload, reason=reason)
        if dry_run:
            return AddWorkerPlan(True, True, payload)
        if self.runpod_client is None:
            self.runpod_client = RunPodClient()
        response = self.runpod_client.create_pod(payload)
        pod_id = str(response.get("id") or response.get("podId") or response.get("pod_id") or name)
        self.store.upsert_pod(
            pod_id=pod_id,
            role=PodRole.WORKER,
            provider_status=str(response.get("status") or "provisioning"),
            run_id=run_id,
            worker_id=assigned_worker_id,
            gpu_type=",".join(gpu_type_ids),
            cost_per_hr=hourly_cost,
            adjusted_cost_per_hr=adjusted_hourly_cost,
            start_time=ts,
            data={"create_payload": payload, "create_response": response},
            now=ts,
        )
        return AddWorkerPlan(True, False, payload, pod_response=response)

    def drain_worker(self, worker_id: str) -> None:
        self.store.request_worker_drain(worker_id)

    def recover_stale(self, *, worker_timeout_seconds: int = 600) -> dict[str, int]:
        return self.store.recover_stale_jobs(worker_timeout_seconds=worker_timeout_seconds)

    def enforce_hard_cap(self) -> list[DrainCandidate]:
        cfg = self.store.get_cost_config()
        if not cfg["hard_cap"]:
            return []
        return self.cost.request_hard_cap_drains()
