"""Worker runtime and HTTP client for Server Pod communication."""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import TaskResult, run_task


class WorkerClientError(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, server_url: str, token: str, timeout_seconds: int = 30):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def register(
        self,
        *,
        worker_id: str,
        run_id: str,
        role: str,
        labels: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/workers/register",
            {"worker_id": worker_id, "run_id": run_id, "role": role, "host": socket.gethostname(), "labels": labels or {}},
        )

    def heartbeat(self, worker_id: str, current_job_id: str | None = None) -> dict[str, Any]:
        return self._request("POST", f"/workers/{worker_id}/heartbeat", {"current_job_id": current_job_id})

    def claim_job(self, worker_id: str, *, run_id: str, lease_seconds: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/jobs/claim",
            {"worker_id": worker_id, "run_id": run_id, "lease_seconds": lease_seconds},
        )

    def claim_jobs(
        self,
        worker_id: str,
        *,
        run_id: str,
        lease_seconds: int,
        max_jobs: int,
        task_type: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/jobs/claim-batch",
            {
                "worker_id": worker_id,
                "run_id": run_id,
                "lease_seconds": lease_seconds,
                "max_jobs": max_jobs,
                "task_type": task_type,
            },
        )

    def report_progress(
        self,
        *,
        worker_id: str,
        job_id: str,
        completed_units: float,
        total_units: float,
        message: str | None,
        lease_seconds: int,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/jobs/{job_id}/progress",
            {
                "worker_id": worker_id,
                "completed_units": completed_units,
                "total_units": total_units,
                "message": message,
                "lease_seconds": lease_seconds,
            },
        )

    def complete_job(self, *, worker_id: str, job_id: str, result: TaskResult) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/jobs/{job_id}/complete",
            {
                "worker_id": worker_id,
                "result": {"output_path": result.output_path, **result.metadata},
            },
        )

    def fail_job(self, *, worker_id: str, job_id: str, error: str, retryable: bool = True) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/jobs/{job_id}/fail",
            {"worker_id": worker_id, "error": error, "retryable": retryable},
        )

    def release_job(self, *, worker_id: str, job_id: str, reason: str = "released by worker") -> dict[str, Any]:
        return self._request(
            "POST",
            f"/jobs/{job_id}/release",
            {"worker_id": worker_id, "reason": reason},
        )

    def drain_ack(self, worker_id: str) -> dict[str, Any]:
        return self._request("POST", f"/workers/{worker_id}/drain-ack", {})

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8")
                return json.loads(text) if text else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise WorkerClientError(f"{method} {path} failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise WorkerClientError(f"{method} {path} failed: {exc}") from exc


@dataclass
class WorkerRunner:
    client: WorkerClient
    worker_id: str
    run_id: str
    workspace_root: Path
    role: str = "wsi-preprocess"
    lease_seconds: int = 300
    idle_sleep_seconds: float = 3.0
    max_jobs: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        server_url: str,
        token: str,
        run_id: str,
        workspace_root: str,
        worker_role: str = "wsi-preprocess",
        worker_id: str | None = None,
    ) -> "WorkerRunner":
        return cls(
            client=WorkerClient(server_url, token),
            worker_id=worker_id or f"worker-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            workspace_root=Path(workspace_root),
            role=worker_role,
        )

    def run_forever(self) -> int:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        response = self.client.register(
            worker_id=self.worker_id,
            run_id=self.run_id,
            role=self.role,
            labels={"workspace_root": str(self.workspace_root)},
        )
        if response.get("command") == "drain":
            self.client.drain_ack(self.worker_id)
            return 0
        completed_jobs = 0
        while self.max_jobs is None or completed_jobs < self.max_jobs:
            heartbeat = self.client.heartbeat(self.worker_id)
            if heartbeat.get("command") == "drain":
                self.client.drain_ack(self.worker_id)
                return 0
            claim = self.client.claim_job(self.worker_id, run_id=self.run_id, lease_seconds=self.lease_seconds)
            if claim.get("command") == "drain":
                self.client.drain_ack(self.worker_id)
                return 0
            job = claim.get("job")
            if not job:
                time.sleep(self.idle_sleep_seconds)
                continue
            self._run_one_job(job)
            completed_jobs += 1
        return 0

    def _run_one_job(self, job: dict[str, Any]) -> None:
        job_id = job["id"]

        def progress(completed_units: float, total_units: float, message: str | None) -> None:
            self.client.report_progress(
                worker_id=self.worker_id,
                job_id=job_id,
                completed_units=completed_units,
                total_units=total_units,
                message=message,
                lease_seconds=self.lease_seconds,
            )

        try:
            result = run_task(
                job["task_type"],
                job.get("payload") or {},
                job_id=job_id,
                workspace_root=self.workspace_root,
                progress=progress,
            )
        except Exception as exc:
            self.client.fail_job(worker_id=self.worker_id, job_id=job_id, error=repr(exc), retryable=True)
            return
        self.client.complete_job(worker_id=self.worker_id, job_id=job_id, result=result)
