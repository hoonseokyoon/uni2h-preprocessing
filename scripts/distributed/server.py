"""HTTP control-plane app for the Server Pod."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .cost import CostManager
from .models import Job, Worker
from .scheduler import Scheduler
from .store import SQLiteStore


def create_app(store: SQLiteStore, *, worker_token: str, scheduler: Scheduler | None = None):
    """Create a FastAPI app for worker control-plane calls.

    FastAPI is imported inside this function so the store, scheduler, and tests
    remain usable when FastAPI is absent.
    """

    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Query
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for the HTTP server; install fastapi and uvicorn") from exc

    scheduler = scheduler or Scheduler(store)
    cost = CostManager(store)
    app = FastAPI(title="WSI-RNA Distributed Execution Server", version="0.1")

    def require_worker_token(
        authorization: str | None = Header(default=None),
        x_worker_token: str | None = Header(default=None),
    ) -> None:
        candidate = None
        if authorization and authorization.lower().startswith("bearer "):
            candidate = authorization.split(" ", 1)[1]
        elif x_worker_token:
            candidate = x_worker_token
        if not worker_token or candidate != worker_token:
            raise HTTPException(status_code=401, detail="invalid worker token")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/workers/register", dependencies=[Depends(require_worker_token)])
    def register_worker(body: dict[str, Any]) -> dict[str, Any]:
        worker = store.register_worker(
            worker_id=body.get("worker_id"),
            run_id=body.get("run_id", "default"),
            role=body.get("role", "worker"),
            host=body.get("host"),
            labels=body.get("labels") or {},
        )
        return {"worker": _worker_dict(worker), "command": store.get_worker_command(worker.worker_id)}

    @app.post("/workers/{worker_id}/heartbeat", dependencies=[Depends(require_worker_token)])
    def heartbeat(worker_id: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        body = body or {}
        worker = store.heartbeat_worker(worker_id, current_job_id=body.get("current_job_id"))
        return {"worker": _worker_dict(worker), "command": store.get_worker_command(worker_id)}

    @app.get("/workers/{worker_id}/command", dependencies=[Depends(require_worker_token)])
    def worker_command(worker_id: str) -> dict[str, Any]:
        return {"command": store.get_worker_command(worker_id)}

    @app.post("/workers/{worker_id}/drain-ack", dependencies=[Depends(require_worker_token)])
    def drain_ack(worker_id: str) -> dict[str, Any]:
        worker = store.ack_worker_drained(worker_id)
        return {"worker": _worker_dict(worker)}

    @app.post("/workers/{worker_id}/drain", dependencies=[Depends(require_worker_token)])
    def request_drain(worker_id: str) -> dict[str, Any]:
        worker = store.request_worker_drain(worker_id)
        return {"worker": _worker_dict(worker)}

    @app.post("/jobs/claim", dependencies=[Depends(require_worker_token)])
    def claim_job(body: dict[str, Any]) -> dict[str, Any]:
        worker_id = body["worker_id"]
        command = store.get_worker_command(worker_id)
        if command == "drain":
            return {"job": None, "command": command}
        job = store.claim_job(
            worker_id,
            lease_seconds=int(body.get("lease_seconds", 300)),
            run_id=body.get("run_id"),
        )
        return {"job": _job_dict(job) if job else None, "command": store.get_worker_command(worker_id)}

    @app.post("/jobs/claim-batch", dependencies=[Depends(require_worker_token)])
    def claim_jobs(body: dict[str, Any]) -> dict[str, Any]:
        worker_id = body["worker_id"]
        command = store.get_worker_command(worker_id)
        if command == "drain":
            return {"jobs": [], "command": command}
        jobs = store.claim_jobs(
            worker_id,
            max_jobs=int(body.get("max_jobs", 1)),
            lease_seconds=int(body.get("lease_seconds", 300)),
            run_id=body.get("run_id"),
            task_type=body.get("task_type"),
        )
        return {"jobs": [_job_dict(job) for job in jobs], "command": store.get_worker_command(worker_id)}

    @app.post("/jobs/{job_id}/progress", dependencies=[Depends(require_worker_token)])
    def report_progress(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
        job = store.report_progress(
            body["worker_id"],
            job_id,
            completed_units=float(body["completed_units"]),
            total_units=body.get("total_units"),
            message=body.get("message"),
            lease_seconds=body.get("lease_seconds"),
        )
        return {"job": _job_dict(job), "command": store.get_worker_command(body["worker_id"])}

    @app.post("/jobs/{job_id}/release", dependencies=[Depends(require_worker_token)])
    def release_job(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
        job = store.release_claimed_job(
            body["worker_id"],
            job_id,
            reason=str(body.get("reason", "released by worker")),
        )
        return {"job": _job_dict(job), "command": store.get_worker_command(body["worker_id"])}

    @app.post("/jobs/{job_id}/complete", dependencies=[Depends(require_worker_token)])
    def complete_job(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
        job = store.complete_job(body["worker_id"], job_id, result=body.get("result") or {})
        return {"job": _job_dict(job), "command": store.get_worker_command(body["worker_id"])}

    @app.post("/jobs/{job_id}/fail", dependencies=[Depends(require_worker_token)])
    def fail_job(job_id: str, body: dict[str, Any]) -> dict[str, Any]:
        job = store.fail_job(
            body["worker_id"],
            job_id,
            error=str(body.get("error", "worker reported failure")),
            retryable=bool(body.get("retryable", True)),
        )
        return {"job": _job_dict(job), "command": store.get_worker_command(body["worker_id"])}

    @app.get("/status", dependencies=[Depends(require_worker_token)])
    def status() -> dict[str, Any]:
        summary = cost.summarize()
        return {
            "jobs": store.job_state_counts(),
            "workers": [_worker_dict(worker) for worker in store.list_workers()],
            "cost": asdict(summary),
            "progress": asdict(store.progress_snapshot()),
        }

    @app.get("/events", dependencies=[Depends(require_worker_token)])
    def events(
        job_id: str | None = None,
        worker_id: str | None = None,
        event_type: str | None = None,
        since_id: int | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        newest_first: bool = False,
    ) -> dict[str, Any]:
        rows = store.list_events(
            job_id=job_id,
            worker_id=worker_id,
            event_type=event_type,
            since_id=since_id,
            limit=limit,
            newest_first=newest_first,
        )
        return {"events": [_event_dict(row) for row in rows]}

    @app.get("/jobs/{job_id}", dependencies=[Depends(require_worker_token)])
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            job = store.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        latest = store.latest_progress_by_job([job_id]).get(job_id)
        events = store.list_events(job_id=job_id, limit=100)
        return {
            "job": _job_dict(job),
            "latest_progress": _event_dict(latest) if latest else None,
            "events": [_event_dict(row) for row in events],
        }

    @app.get("/workers/{worker_id}", dependencies=[Depends(require_worker_token)])
    def get_worker(worker_id: str) -> dict[str, Any]:
        try:
            worker = store.get_worker(worker_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        events = store.list_events(worker_id=worker_id, limit=100)
        return {"worker": _worker_dict(worker), "events": [_event_dict(row) for row in events]}

    @app.post("/config/cost-cap", dependencies=[Depends(require_worker_token)])
    def set_cost_cap(body: dict[str, Any]) -> dict[str, Any]:
        cap = body.get("cost_cap_per_hour", body.get("cap_per_hour"))
        store.set_cost_cap(None if cap is None else float(cap), hard_cap=bool(body.get("hard_cap", False)))
        drained = [asdict(item) for item in scheduler.enforce_hard_cap()]
        return {"cost": store.get_cost_config(), "hard_cap_drains": drained}

    @app.post("/recover-stale", dependencies=[Depends(require_worker_token)])
    def recover_stale(body: dict[str, Any] | None = None) -> dict[str, int]:
        timeout = int((body or {}).get("worker_timeout_seconds", 600))
        return store.recover_stale_jobs(worker_timeout_seconds=timeout)

    return app


def _job_dict(job: Job | None) -> dict[str, Any] | None:
    if job is None:
        return None
    data = asdict(job)
    data["state"] = job.state.value
    return data


def _worker_dict(worker: Worker) -> dict[str, Any]:
    data = asdict(worker)
    data["state"] = worker.state.value
    return data


def _event_dict(event: Any) -> dict[str, Any]:
    data = asdict(event)
    return data
