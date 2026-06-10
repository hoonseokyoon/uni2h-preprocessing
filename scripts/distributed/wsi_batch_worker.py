"""Batch worker runtime for WSI UNI2-h preprocessing."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tasks import TaskResult
from .worker import WorkerClient, WorkerClientError
from .wsi_preprocess import (
    TASK_TOTAL_UNITS,
    StagedWSI,
    cleanup_staged_wsi,
    execute_staged_wsi_task,
    parse_optional_int,
    planned_artifact_paths,
    should_skip_existing,
    stage_wsi,
)


@dataclass
class WSIBatchWorkerRunner:
    client: WorkerClient
    worker_id: str
    run_id: str
    workspace_root: Path
    local_cache_dir: Path
    role: str = "wsi-preprocess"
    lease_seconds: int = 1800
    idle_sleep_seconds: float = 5.0
    batch_jobs: int = 4
    prefetch_jobs: int = 2
    prefetch_max_bytes: int = 250 * 1024**3
    unknown_prefetch_size_bytes: int = 20 * 1024**3
    max_batches: int | None = None

    @classmethod
    def from_env(
        cls,
        *,
        server_url: str,
        token: str,
        run_id: str,
        workspace_root: str,
        local_cache_dir: str,
        worker_role: str = "wsi-preprocess",
        worker_id: str | None = None,
    ) -> "WSIBatchWorkerRunner":
        return cls(
            client=WorkerClient(server_url, token),
            worker_id=worker_id or f"wsi-worker-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            workspace_root=Path(workspace_root),
            local_cache_dir=Path(local_cache_dir),
            role=worker_role,
        )

    def run_forever(self) -> int:
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        response = self.client.register(
            worker_id=self.worker_id,
            run_id=self.run_id,
            role=self.role,
            labels={
                "workspace_root": str(self.workspace_root),
                "local_cache_dir": str(self.local_cache_dir),
                "batch_jobs": self.batch_jobs,
                "prefetch_jobs": self.prefetch_jobs,
                "prefetch_max_bytes": self.prefetch_max_bytes,
                "task_type": "wsi_uni2h",
            },
        )
        if response.get("command") == "drain":
            self.client.drain_ack(self.worker_id)
            return 0

        completed_batches = 0
        while self.max_batches is None or completed_batches < self.max_batches:
            heartbeat = self.client.heartbeat(self.worker_id)
            if heartbeat.get("command") == "drain":
                self.client.drain_ack(self.worker_id)
                return 0

            claim = self.client.claim_jobs(
                self.worker_id,
                run_id=self.run_id,
                lease_seconds=self.lease_seconds,
                max_jobs=max(1, self.batch_jobs),
                task_type="wsi_uni2h",
            )
            if claim.get("command") == "drain":
                self.client.drain_ack(self.worker_id)
                return 0
            jobs = list(claim.get("jobs") or [])
            if not jobs:
                time.sleep(self.idle_sleep_seconds)
                continue

            drain_requested = self._process_batch(jobs)
            completed_batches += 1
            if drain_requested:
                self.client.drain_ack(self.worker_id)
                return 0
        return 0

    def _process_batch(self, jobs: list[dict[str, Any]]) -> bool:
        drain_requested = {"value": False}
        released_job_ids: set[str] = set()
        completed_or_failed: set[str] = set()
        inflight: dict[str, Future[StagedWSI]] = {}
        reserved_bytes: dict[str, int] = {}
        next_submit = 0

        def progress_for(job_id: str):
            def _progress(completed_units: float, total_units: float, message: str | None) -> None:
                if job_id in released_job_ids:
                    return
                try:
                    response = self.client.report_progress(
                        worker_id=self.worker_id,
                        job_id=job_id,
                        completed_units=completed_units,
                        total_units=total_units,
                        message=message,
                        lease_seconds=self.lease_seconds,
                    )
                    if response.get("command") == "drain":
                        drain_requested["value"] = True
                except WorkerClientError:
                    if job_id not in released_job_ids:
                        raise

            return _progress

        def job_reserved_size(job: dict[str, Any]) -> int:
            size = parse_optional_int((job.get("payload") or {}).get("expected_size_bytes"))
            return size if size is not None else self.unknown_prefetch_size_bytes

        def submit_until_full(executor: ThreadPoolExecutor) -> None:
            nonlocal next_submit
            while next_submit < len(jobs) and len(inflight) < max(1, self.prefetch_jobs):
                job = jobs[next_submit]
                payload = dict(job.get("payload") or {})
                if should_skip_existing(payload, self.workspace_root):
                    next_submit += 1
                    continue
                size = job_reserved_size(job)
                current_reserved = sum(reserved_bytes.values())
                if inflight and current_reserved + size > self.prefetch_max_bytes:
                    return
                next_submit += 1
                job_id = job["id"]
                payload.setdefault("local_cache_dir", str(self.local_cache_dir))
                inflight[job_id] = executor.submit(
                    stage_wsi,
                    payload,
                    job_id=job_id,
                    workspace_root=self.workspace_root,
                    local_cache_dir=self.local_cache_dir,
                    progress=progress_for(job_id),
                )
                reserved_bytes[job_id] = size

        with ThreadPoolExecutor(max_workers=max(1, self.prefetch_jobs), thread_name_prefix="wsi-prefetch") as executor:
            submit_until_full(executor)
            for index, job in enumerate(jobs):
                job_id = job["id"]
                payload = dict(job.get("payload") or {})
                payload.setdefault("local_cache_dir", str(self.local_cache_dir))
                staged: StagedWSI | None = None
                try:
                    if should_skip_existing(payload, self.workspace_root):
                        paths = planned_artifact_paths(payload, self.workspace_root)
                        self.client.report_progress(
                            worker_id=self.worker_id,
                            job_id=job_id,
                            completed_units=TASK_TOTAL_UNITS,
                            total_units=TASK_TOTAL_UNITS,
                            message="existing artifacts found; skipped",
                            lease_seconds=self.lease_seconds,
                        )
                        self.client.complete_job(
                            worker_id=self.worker_id,
                            job_id=job_id,
                            result=TaskResult(
                                str(paths["features_h5"]),
                                {
                                    "task_type": "wsi_uni2h",
                                    "status": "skipped_existing",
                                    "manifest_path": str(paths["manifest_json"]),
                                },
                            ),
                        )
                        completed_or_failed.add(job_id)
                    else:
                        if job_id not in inflight:
                            inflight[job_id] = executor.submit(
                                stage_wsi,
                                payload,
                                job_id=job_id,
                                workspace_root=self.workspace_root,
                                local_cache_dir=self.local_cache_dir,
                                progress=progress_for(job_id),
                            )
                            reserved_bytes[job_id] = job_reserved_size(job)
                        staged = inflight.pop(job_id).result()
                        reserved_bytes.pop(job_id, None)
                        result = execute_staged_wsi_task(
                            payload,
                            staged,
                            job_id=job_id,
                            workspace_root=self.workspace_root,
                            progress=progress_for(job_id),
                        )
                        self.client.complete_job(
                            worker_id=self.worker_id,
                            job_id=job_id,
                            result=TaskResult(result.output_path, result.metadata),
                        )
                        completed_or_failed.add(job_id)
                except Exception as exc:
                    completed_or_failed.add(job_id)
                    try:
                        self.client.fail_job(worker_id=self.worker_id, job_id=job_id, error=repr(exc), retryable=True)
                    finally:
                        if staged is not None:
                            cleanup_staged_wsi(staged, payload)
                finally:
                    if staged is not None:
                        cleanup_staged_wsi(staged, payload)
                    submit_until_full(executor)

                if drain_requested["value"]:
                    self._release_unstarted_jobs(
                        jobs[index + 1 :],
                        inflight,
                        released_job_ids,
                        completed_or_failed,
                    )
                    return True
        return False

    def _release_unstarted_jobs(
        self,
        remaining_jobs: list[dict[str, Any]],
        inflight: dict[str, Future[StagedWSI]],
        released_job_ids: set[str],
        completed_or_failed: set[str],
    ) -> None:
        for job in remaining_jobs:
            job_id = job["id"]
            if job_id in completed_or_failed or job_id in released_job_ids:
                continue
            released_job_ids.add(job_id)
            future = inflight.get(job_id)
            if future is not None:
                payload = dict(job.get("payload") or {})
                future.cancel()
                if future.done() and not future.cancelled():
                    try:
                        cleanup_staged_wsi(future.result(), payload)
                    except Exception:
                        pass
                elif not future.cancelled():
                    future.add_done_callback(lambda item, job_payload=payload: self._cleanup_released_future(item, job_payload))
            try:
                self.client.release_job(
                    worker_id=self.worker_id,
                    job_id=job_id,
                    reason="worker drain requested before job execution",
                )
            except WorkerClientError:
                pass

    @staticmethod
    def _cleanup_released_future(future: Future[StagedWSI], payload: dict[str, Any]) -> None:
        if future.cancelled():
            return
        try:
            cleanup_staged_wsi(future.result(), payload)
        except Exception:
            pass
