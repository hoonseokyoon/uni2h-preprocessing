"""Command-line entry point for distributed execution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.distributed.config import load_config
from scripts.distributed.cost import CostManager
from scripts.distributed.export_artifacts import ExportConfig, export_artifacts
from scripts.distributed.models import JobState, PodRole, WorkerState
from scripts.distributed.runpod_client import RunPodClient
from scripts.distributed.scheduler import Scheduler
from scripts.distributed.server import create_app
from scripts.distributed.store import DEFAULT_RUN_ID, SQLiteStore
from scripts.distributed.worker import WorkerRunner
from scripts.distributed.wsi_batch_worker import WSIBatchWorkerRunner
from scripts.distributed.wsi_preprocess import planned_artifact_paths, safe_path_part
from scripts.downloader.download import DownloadConfig, load_plan_rows
from scripts.downloader.planner import parse_datasets, parse_size


DEFAULT_DB = "runpod_distributed.sqlite"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    return args.handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunPod distributed execution control plane")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite server state path")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("init-db", help="Initialize the server SQLite database")
    p.add_argument("--reset", action="store_true", help="Drop existing distributed tables first")
    p.set_defaults(handler=cmd_init_db)

    p = sub.add_parser("enqueue-demo-jobs", help="Enqueue deterministic demo file jobs")
    p.add_argument("--count", type=int, default=4)
    p.add_argument("--output-dir", default="distributed_demo_outputs")
    p.add_argument("--work-units", type=int, default=5)
    p.add_argument("--delay-seconds", type=float, default=0.0)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.set_defaults(handler=cmd_enqueue_demo_jobs)

    p = sub.add_parser("enqueue-wsi-uni2h-jobs", help="Enqueue WSI stage-in + UNI2-h extraction jobs from raw_wsi download plans")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan-dir", type=Path)
    source.add_argument("--plan-file", type=Path)
    source.add_argument("--index-file", type=Path)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--datasets", default="", help="Comma-separated dataset filter")
    p.add_argument("--source-adapter", default="", help="Optional adapter filter, e.g. gdc")
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--max-bytes", default="", help="Optional selected-byte limit, e.g. 500GB")
    p.add_argument("--artifact-root", default="artifacts/wsi_uni2h_v0")
    p.add_argument("--config-path", default="configs/uni2h_w8yi_style.yaml")
    p.add_argument("--extract-script", default="scripts/extract_uni2h_features.py")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--device")
    p.add_argument("--local-cache-dir", default="", help="Optional worker-local cache override embedded into jobs")
    p.add_argument("--download-timeout-seconds", type=int, default=180)
    p.add_argument("--download-retries", type=int, default=4)
    p.add_argument("--max-attempts", type=int, default=2)
    p.add_argument("--simulate", action="store_true", help="Write small fake artifacts for local server tests")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--overwrite-artifacts", action="store_true")
    p.set_defaults(handler=cmd_enqueue_wsi_uni2h_jobs)

    p = sub.add_parser("serve", help="Run the FastAPI Server Pod control plane")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--token", default=os.environ.get("WORKER_TOKEN"))
    p.set_defaults(handler=cmd_serve)

    p = sub.add_parser("worker", help="Run a worker loop against a Server Pod")
    p.add_argument("--server-url", default=os.environ.get("SERVER_URL"))
    p.add_argument("--token", default=os.environ.get("WORKER_TOKEN"))
    p.add_argument("--run-id", default=os.environ.get("RUN_ID", DEFAULT_RUN_ID))
    p.add_argument("--worker-id", default=os.environ.get("WORKER_ID"))
    p.add_argument("--worker-role", default=os.environ.get("WORKER_ROLE", "wsi-preprocess"))
    p.add_argument("--workspace-root", default=os.environ.get("WORKSPACE_ROOT", "."))
    p.add_argument("--max-jobs", type=int)
    p.add_argument("--idle-sleep-seconds", type=float, default=3.0)
    p.add_argument("--lease-seconds", type=int, default=300)
    p.set_defaults(handler=cmd_worker)

    p = sub.add_parser("wsi-worker", help="Run a WSI batch worker with local async stage-in and UNI2-h extraction")
    p.add_argument("--server-url", default=os.environ.get("SERVER_URL"))
    p.add_argument("--token", default=os.environ.get("WORKER_TOKEN"))
    p.add_argument("--run-id", default=os.environ.get("RUN_ID", DEFAULT_RUN_ID))
    p.add_argument("--worker-id", default=os.environ.get("WORKER_ID"))
    p.add_argument("--worker-role", default=os.environ.get("WORKER_ROLE", "wsi-preprocess"))
    p.add_argument("--workspace-root", default=os.environ.get("WORKSPACE_ROOT", "."))
    p.add_argument("--local-cache-dir", default=os.environ.get("LOCAL_WSI_CACHE_DIR", "local_wsi_cache"))
    p.add_argument("--batch-jobs", type=int, default=4)
    p.add_argument("--prefetch-jobs", type=int, default=2)
    p.add_argument("--prefetch-max-bytes", default=os.environ.get("PREFETCH_MAX_BYTES", "250GB"))
    p.add_argument("--unknown-prefetch-size", default=os.environ.get("UNKNOWN_PREFETCH_SIZE", "20GB"))
    p.add_argument("--max-batches", type=int)
    p.add_argument("--idle-sleep-seconds", type=float, default=5.0)
    p.add_argument("--lease-seconds", type=int, default=1800)
    p.set_defaults(handler=cmd_wsi_worker)

    p = sub.add_parser("status", help="Print job, worker, progress, and cost status")
    p.add_argument("--plain", action="store_true", help="Force plain table output")
    p.set_defaults(handler=cmd_status)

    p = sub.add_parser("set-cost-cap", help="Set or clear the hourly cost cap")
    p.add_argument("--cap", required=True, help="Hourly cap in dollars, or 'none'")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--hard-cap", action="store_true", help="Drain idle/excess workers when burn exceeds cap")
    mode.add_argument("--soft-cap", action="store_true", help="Only block new workers when burn exceeds cap")
    p.set_defaults(handler=cmd_set_cost_cap)

    p = sub.add_parser("pause-run", help="Pause new job claims for a run")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--drain-workers", action="store_true", help="Also request graceful drain for active workers")
    p.set_defaults(handler=cmd_pause_run)

    p = sub.add_parser("resume-run", help="Resume job claims for a paused run")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.set_defaults(handler=cmd_resume_run)

    p = sub.add_parser("drain-run", help="Gracefully drain all active workers in a run")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--worker-role", default="", help="Optional worker role filter")
    p.set_defaults(handler=cmd_drain_run)

    p = sub.add_parser("terminate-drained-pods", help="Stop or delete RunPod worker pods whose workers are drained")
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument("--execute", action="store_true", help="Call RunPod API; otherwise print planned actions")
    p.add_argument("--delete", action="store_true", help="Delete pods instead of stopping them")
    p.set_defaults(handler=cmd_terminate_drained_pods)

    p = sub.add_parser("export-artifacts", help="Export completed artifact trees to S3-compatible storage")
    p.add_argument("--artifact-root", required=True, type=Path)
    p.add_argument("--destination", required=True, help="s3://bucket/prefix")
    p.add_argument("--endpoint-url", default=os.environ.get("AWS_ENDPOINT_URL"))
    p.add_argument("--access-key-id", default=os.environ.get("AWS_ACCESS_KEY_ID"))
    p.add_argument("--secret-access-key", default=os.environ.get("AWS_SECRET_ACCESS_KEY"))
    p.add_argument("--region-name", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    p.add_argument("--inventory-path", type=Path, default=Path("manifests") / "export_inventory_v0.csv")
    p.add_argument("--include", action="append", help="Artifact filename glob; can repeat. Default exports known WSI artifacts.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Upload even when a same-size object already exists")
    p.set_defaults(handler=cmd_export_artifacts)

    for name in ("add-worker", "scale"):
        p = sub.add_parser(name, help="Create or dry-run a RunPod worker pod")
        p.add_argument("--config", help="Optional JSON/YAML config with runpod.worker defaults")
        p.add_argument("--execute", action="store_true", help="Call the RunPod API instead of dry-run")
        p.add_argument("--name", default="wsi-rna-worker")
        p.add_argument("--image-name")
        p.add_argument("--server-pod-id")
        p.add_argument("--server-port", type=int, default=8080)
        p.add_argument("--run-id", default=DEFAULT_RUN_ID)
        p.add_argument("--workspace-root", default="/workspace")
        p.add_argument("--gpu-type-id", action="append", dest="gpu_type_ids")
        p.add_argument("--gpu-count", type=int, default=1)
        p.add_argument("--network-volume-id")
        p.add_argument("--data-center-id", action="append", dest="data_center_ids")
        p.add_argument("--hourly-cost", type=float)
        p.add_argument("--adjusted-hourly-cost", type=float)
        p.add_argument("--worker-token", default=os.environ.get("WORKER_TOKEN"))
        p.add_argument("--worker-role", default="wsi-preprocess")
        p.add_argument("--worker-id", help="Optional stable worker id to inject into the pod environment")
        p.set_defaults(handler=cmd_add_worker)

    p = sub.add_parser("drain-worker", help="Ask a worker to finish its current job and stop claiming")
    p.add_argument("--worker-id", required=True)
    p.set_defaults(handler=cmd_drain_worker)

    p = sub.add_parser("recover-stale", help="Recover expired leases and stale workers")
    p.add_argument("--worker-timeout-seconds", type=int, default=600)
    p.set_defaults(handler=cmd_recover_stale)

    return parser


def cmd_init_db(args: argparse.Namespace) -> int:
    store = SQLiteStore(args.db)
    store.initialize(reset=args.reset)
    print(f"initialized {Path(args.db).resolve()}")
    return 0


def cmd_enqueue_demo_jobs(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    scheduler = Scheduler(store)
    job_ids = scheduler.enqueue_demo_jobs(
        count=args.count,
        output_dir=args.output_dir,
        run_id=args.run_id,
        work_units=args.work_units,
        delay_seconds=args.delay_seconds,
    )
    print(f"enqueued {len(job_ids)} demo jobs")
    for job_id in job_ids:
        print(job_id)
    return 0


def cmd_enqueue_wsi_uni2h_jobs(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    rows = load_plan_rows(
        DownloadConfig(
            plan_dir=args.plan_dir,
            plan_file=args.plan_file,
            index_file=args.index_file,
            asset_kind="raw_wsi",
            source_adapter=args.source_adapter,
            datasets=parse_datasets(args.datasets),
            max_files=args.max_jobs,
            max_bytes=parse_size(args.max_bytes) if args.max_bytes else None,
        )
    )
    enqueued = 0
    skipped_existing_jobs = 0
    skipped_non_raw_wsi = 0
    for row in rows:
        if row.get("asset_kind") != "raw_wsi":
            skipped_non_raw_wsi += 1
            continue
        download_id = row.get("download_id") or row.get("wsi_id") or row.get("file_id")
        job_id = f"wsi_uni2h:{download_id}"
        try:
            store.get_job(job_id)
            skipped_existing_jobs += 1
            continue
        except KeyError:
            pass

        payload = build_wsi_uni2h_payload_from_plan_row(row, args)
        output_path = str(planned_artifact_paths(payload, Path("."))["features_h5"])
        store.enqueue_job(
            "wsi_uni2h",
            payload,
            run_id=args.run_id,
            total_units=100.0,
            priority=int(row.get("priority") or 50),
            max_attempts=args.max_attempts,
            output_path=output_path,
            job_id=job_id,
        )
        enqueued += 1

    print(
        json.dumps(
            {
                "selected_rows": len(rows),
                "enqueued": enqueued,
                "skipped_existing_jobs": skipped_existing_jobs,
                "skipped_non_raw_wsi": skipped_non_raw_wsi,
                "run_id": args.run_id,
                "task_type": "wsi_uni2h",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    if not args.token:
        raise SystemExit("serve requires --token or WORKER_TOKEN")
    store = initialized_store(args.db)
    app = create_app(store, worker_token=args.token)
    try:
        import uvicorn  # type: ignore
    except ImportError as exc:
        raise SystemExit("serve requires uvicorn; install uvicorn or run core tests without HTTP") from exc
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    if not args.server_url:
        raise SystemExit("worker requires --server-url or SERVER_URL")
    if not args.token:
        raise SystemExit("worker requires --token or WORKER_TOKEN")
    runner = WorkerRunner.from_env(
        server_url=args.server_url,
        token=args.token,
        run_id=args.run_id,
        worker_role=args.worker_role,
        worker_id=args.worker_id,
        workspace_root=args.workspace_root,
    )
    runner.max_jobs = args.max_jobs
    runner.idle_sleep_seconds = args.idle_sleep_seconds
    runner.lease_seconds = args.lease_seconds
    return runner.run_forever()


def cmd_wsi_worker(args: argparse.Namespace) -> int:
    if not args.server_url:
        raise SystemExit("wsi-worker requires --server-url or SERVER_URL")
    if not args.token:
        raise SystemExit("wsi-worker requires --token or WORKER_TOKEN")
    runner = WSIBatchWorkerRunner.from_env(
        server_url=args.server_url,
        token=args.token,
        run_id=args.run_id,
        worker_role=args.worker_role,
        worker_id=args.worker_id,
        workspace_root=args.workspace_root,
        local_cache_dir=args.local_cache_dir,
    )
    runner.batch_jobs = max(1, args.batch_jobs)
    runner.prefetch_jobs = max(1, args.prefetch_jobs)
    runner.prefetch_max_bytes = parse_size(args.prefetch_max_bytes)
    runner.unknown_prefetch_size_bytes = parse_size(args.unknown_prefetch_size)
    runner.max_batches = args.max_batches
    runner.idle_sleep_seconds = args.idle_sleep_seconds
    runner.lease_seconds = args.lease_seconds
    return runner.run_forever()


def cmd_status(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    render_status(store, force_plain=args.plain)
    return 0


def cmd_set_cost_cap(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    cap = None if args.cap.lower() in {"none", "null", "off"} else float(args.cap)
    store.set_cost_cap(cap, hard_cap=bool(args.hard_cap))
    drained = Scheduler(store).enforce_hard_cap()
    mode = "hard" if args.hard_cap else "soft"
    cap_text = "none" if cap is None else f"${cap:.2f}/hr"
    print(f"cost cap set to {cap_text} ({mode})")
    if drained:
        print("requested drains:")
        for item in drained:
            print(f"{item.worker_id}\t{item.pod_id or '-'}\t${item.hourly_rate:.2f}/hr")
    return 0


def cmd_pause_run(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    store.set_run_paused(args.run_id, True)
    drained: list[Any] = []
    if args.drain_workers:
        drained = store.request_run_drain(run_id=args.run_id)
    print(json.dumps({"run_id": args.run_id, "paused": True, "drain_requested": len(drained)}, indent=2))
    return 0


def cmd_resume_run(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    store.set_run_paused(args.run_id, False)
    print(json.dumps({"run_id": args.run_id, "paused": False}, indent=2))
    return 0


def cmd_drain_run(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    drained = store.request_run_drain(run_id=args.run_id, role=args.worker_role or None)
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "worker_role": args.worker_role or None,
                "drain_requested": len(drained),
                "workers": [worker.worker_id for worker in drained],
            },
            indent=2,
        )
    )
    return 0


def cmd_terminate_drained_pods(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    drained_worker_ids = {
        worker.worker_id
        for worker in store.list_workers(states=[WorkerState.DRAINED])
        if worker.run_id == args.run_id
    }
    pods = [
        pod
        for pod in store.list_pods()
        if pod.role == PodRole.WORKER
        and pod.run_id == args.run_id
        and pod.worker_id in drained_worker_ids
        and pod.provider_status.lower() not in {"stopped", "terminated", "deleted"}
    ]
    action = "delete" if args.delete else "stop"
    results: list[dict[str, Any]] = []
    client = RunPodClient() if args.execute else None
    for pod in pods:
        item: dict[str, Any] = {"pod_id": pod.pod_id, "worker_id": pod.worker_id, "action": action}
        if args.execute and client is not None:
            response = client.delete_pod(pod.pod_id) if args.delete else client.stop_pod(pod.pod_id)
            store.stop_pod_record(pod.pod_id, provider_status="deleted" if args.delete else "stopped")
            item["response"] = response
        results.append(item)
    print(json.dumps({"run_id": args.run_id, "execute": bool(args.execute), "planned_or_done": results}, indent=2))
    return 0


def cmd_export_artifacts(args: argparse.Namespace) -> int:
    result = export_artifacts(
        ExportConfig(
            artifact_root=args.artifact_root,
            destination=args.destination,
            endpoint_url=args.endpoint_url,
            access_key_id=args.access_key_id,
            secret_access_key=args.secret_access_key,
            region_name=args.region_name,
            inventory_path=args.inventory_path,
            include=tuple(args.include) if args.include else ("features.h5", "overlay.png", "thumbnail.jpg", "tissue_mask.png", "qc_preview.jpg", "manifest.json"),
            dry_run=bool(args.dry_run),
            skip_existing=not bool(args.overwrite),
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_add_worker(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    values = _worker_args_with_config(args)
    required = ["image_name", "server_pod_id", "gpu_type_ids", "hourly_cost"]
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise SystemExit(f"missing required add-worker settings: {', '.join(missing)}")
    scheduler = Scheduler(store)
    plan = scheduler.add_worker(
        name=values["name"],
        image_name=values["image_name"],
        server_pod_id=values["server_pod_id"],
        server_port=int(values["server_port"]),
        run_id=values["run_id"],
        workspace_root=values["workspace_root"],
        gpu_type_ids=list(values["gpu_type_ids"]),
        hourly_cost=float(values["hourly_cost"]),
        worker_token=values.get("worker_token"),
        worker_role=values["worker_role"],
        gpu_count=int(values["gpu_count"]),
        network_volume_id=values.get("network_volume_id"),
        data_center_ids=values.get("data_center_ids"),
        adjusted_hourly_cost=values.get("adjusted_hourly_cost"),
        worker_id=values.get("worker_id"),
        dry_run=not args.execute,
    )
    if not plan.allowed:
        print(f"blocked: {plan.reason}")
        print(json.dumps(plan.payload, indent=2, sort_keys=True))
        return 1
    if plan.dry_run:
        print("dry-run worker payload:")
        print(json.dumps(plan.payload, indent=2, sort_keys=True))
    else:
        print("created worker pod:")
        print(json.dumps(plan.pod_response, indent=2, sort_keys=True))
    return 0


def cmd_drain_worker(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    worker = store.request_worker_drain(args.worker_id)
    print(f"worker {worker.worker_id} state={worker.state.value}")
    return 0


def cmd_recover_stale(args: argparse.Namespace) -> int:
    store = initialized_store(args.db)
    result = store.recover_stale_jobs(worker_timeout_seconds=args.worker_timeout_seconds)
    print(json.dumps(result, sort_keys=True))
    return 0


def render_status(store: SQLiteStore, *, force_plain: bool = False) -> None:
    jobs = store.list_jobs()
    workers = store.list_workers()
    counts = store.job_state_counts()
    progress = store.progress_snapshot()
    cost = CostManager(store).summarize()
    if not force_plain:
        try:
            _render_status_rich(jobs, workers, counts, progress, cost)
            return
        except ImportError:
            pass
    _render_status_plain(jobs, workers, counts, progress, cost)


def _render_status_rich(jobs: list[Any], workers: list[Any], counts: dict[str, int], progress: Any, cost: Any) -> None:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table

    console = Console()
    console.rule("Distributed Execution Status")
    table = Table(title="Jobs")
    table.add_column("State")
    table.add_column("Count", justify="right")
    for state in [item.value for item in JobState]:
        table.add_row(state, str(counts.get(state, 0)))
    console.print(table)
    total = max(progress.total_units, 1.0)
    with Progress(TextColumn("Progress"), BarColumn(), TextColumn("{task.completed:.1f}/{task.total:.1f} units"), console=console) as bar:
        bar.add_task("units", total=total, completed=progress.completed_units)
    worker_table = Table(title="Workers")
    worker_table.add_column("Worker")
    worker_table.add_column("State")
    worker_table.add_column("Current Job")
    worker_table.add_column("Last Heartbeat")
    for worker in workers:
        worker_table.add_row(
            worker.worker_id,
            worker.state.value,
            worker.current_job_id or "-",
            _fmt_ts(worker.last_heartbeat_at),
        )
    console.print(worker_table)
    console.print(
        f"burn=${cost.current_burn_rate_per_hr:.2f}/hr spent=${cost.spent_so_far:.2f} "
        f"eta={_fmt_duration(cost.eta_seconds)} est_remaining={_fmt_optional_money(cost.estimated_cost_to_completion)} "
        f"cap={_fmt_optional_cap(cost.cap_per_hour)}"
    )


def _render_status_plain(jobs: list[Any], workers: list[Any], counts: dict[str, int], progress: Any, cost: Any) -> None:
    print("Distributed Execution Status")
    print("Jobs")
    print("state\tcount")
    for state in [item.value for item in JobState]:
        print(f"{state}\t{counts.get(state, 0)}")
    print(f"progress_units\t{progress.completed_units:.1f}/{progress.total_units:.1f}")
    print("Workers")
    print("worker_id\tstate\tcurrent_job\tlast_heartbeat")
    for worker in workers:
        print(f"{worker.worker_id}\t{worker.state.value}\t{worker.current_job_id or '-'}\t{_fmt_ts(worker.last_heartbeat_at)}")
    print("Cost")
    print(f"burn_per_hr\t${cost.current_burn_rate_per_hr:.2f}")
    print(f"spent_so_far\t${cost.spent_so_far:.2f}")
    print(f"eta\t{_fmt_duration(cost.eta_seconds)}")
    print(f"estimated_cost_to_completion\t{_fmt_optional_money(cost.estimated_cost_to_completion)}")
    print(f"cap_per_hr\t{_fmt_optional_cap(cost.cap_per_hour)}")
    if not cost.scale_up_allowed:
        print(f"scale_up_blocked\t{cost.scale_up_block_reason}")


def initialized_store(db_path: str) -> SQLiteStore:
    store = SQLiteStore(db_path)
    store.initialize()
    return store


def build_wsi_uni2h_payload_from_plan_row(row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "download_id",
        "chunk_id",
        "asset_kind",
        "source_adapter",
        "source_system",
        "dataset",
        "cohort_role",
        "case_submitter_id",
        "case_uuid",
        "sample_or_slide_id",
        "wsi_id",
        "file_id",
        "remote_url",
        "expected_file_name",
        "expected_size_bytes",
        "expected_md5",
        "target_rel_path",
        "access",
        "qc_policy",
        "source_table",
        "source_row_id",
    ]
    payload: dict[str, Any] = {key: row.get(key, "") for key in keys}
    payload.update(
        {
            "slide_key": safe_path_part(row.get("wsi_id") or row.get("sample_or_slide_id") or row.get("expected_file_name") or ""),
            "artifact_root": args.artifact_root,
            "config_path": args.config_path,
            "extract_script": args.extract_script,
            "download_timeout_seconds": args.download_timeout_seconds,
            "download_retries": args.download_retries,
            "simulate": bool(args.simulate),
            "no_overlay": bool(args.no_overlay),
            "overwrite_artifacts": bool(args.overwrite_artifacts),
        }
    )
    if args.batch_size:
        payload["batch_size"] = args.batch_size
    if args.device:
        payload["device"] = args.device
    if args.local_cache_dir:
        payload["local_cache_dir"] = args.local_cache_dir
    return payload


def _worker_args_with_config(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if args.config:
        config = load_config(args.config)
        runpod = config.get("runpod", {})
        worker = runpod.get("worker", {})
        values.update(worker)
    cli_map = {
        "name": args.name,
        "image_name": args.image_name,
        "server_pod_id": args.server_pod_id,
        "server_port": args.server_port,
        "run_id": args.run_id,
        "workspace_root": args.workspace_root,
        "gpu_type_ids": args.gpu_type_ids,
        "gpu_count": args.gpu_count,
        "network_volume_id": args.network_volume_id,
        "data_center_ids": args.data_center_ids,
        "hourly_cost": args.hourly_cost,
        "adjusted_hourly_cost": args.adjusted_hourly_cost,
        "worker_token": args.worker_token,
        "worker_role": args.worker_role,
        "worker_id": args.worker_id,
    }
    for key, value in cli_map.items():
        if value is not None:
            values[key] = value
    values.setdefault("gpu_count", 1)
    values.setdefault("worker_role", "wsi-preprocess")
    values.setdefault("workspace_root", "/workspace")
    values.setdefault("run_id", DEFAULT_RUN_ID)
    return values


def _fmt_ts(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.0f}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _fmt_optional_money(value: float | None) -> str:
    return "unknown" if value is None else f"${value:.2f}"


def _fmt_optional_cap(value: float | None) -> str:
    return "none" if value is None else f"${value:.2f}/hr"


if __name__ == "__main__":
    raise SystemExit(main())
