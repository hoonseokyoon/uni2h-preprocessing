#!/usr/bin/env python
"""Simple RunPod WSI workflow wrapper.

This script keeps the common WSI preprocessing operations behind a small set of
commands. Values come from a dotenv-style config file, then environment
variables, then explicit CLI flags.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.distributed import cli as distributed_cli
from scripts.distributed.artifact_validation import validate_wsi_artifacts
from scripts.distributed.models import JobState
from scripts.distributed.store import SQLiteStore
from scripts.downloader.planner import PlanBuildConfig, build_and_write_plans, parse_datasets, parse_size


DEFAULT_ENV_PATH = Path("configs") / "runpod_wsi.env"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 2
    env = load_settings(args.env_file)
    apply_env_exports(env)
    return args.handler(args, env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-command wrapper for RunPod WSI preprocessing")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH, help="Dotenv-style workflow config")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="Check required config values")
    p.add_argument("--deep", action="store_true", help="Check runtime dependencies, paths, CUDA, and plans")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON for deep checks")
    p.set_defaults(handler=cmd_doctor)

    p = sub.add_parser("build-plan", help="Build raw WSI download plans from QC tables")
    p.set_defaults(handler=cmd_build_plan)

    p = sub.add_parser("enqueue", help="Enqueue WSI UNI2-h jobs from the configured plan")
    p.add_argument("--simulate", action="store_true", help="Enqueue fake extraction jobs for local smoke tests")
    p.set_defaults(handler=cmd_enqueue)

    p = sub.add_parser("start", help="Build plan, enqueue jobs, and optionally create worker pods")
    p.add_argument("--workers", type=int, default=None, help="Number of workers to create after enqueue")
    p.add_argument("--execute", action="store_true", help="Actually create RunPod workers; otherwise dry-run")
    p.add_argument("--skip-plan", action="store_true")
    p.add_argument("--skip-enqueue", action="store_true")
    p.add_argument("--simulate", action="store_true", help="Enqueue fake extraction jobs for local smoke tests")
    p.set_defaults(handler=cmd_start)

    p = sub.add_parser("add-workers", help="Create or dry-run additional worker pods")
    p.add_argument("count", nargs="?", type=int, default=None)
    p.add_argument("--execute", action="store_true")
    p.set_defaults(handler=cmd_add_workers)

    p = sub.add_parser("status", help="Show jobs, workers, progress, and cost")
    p.add_argument("--plain", action="store_true")
    p.set_defaults(handler=cmd_status)

    p = sub.add_parser("pause", help="Pause claims and optionally drain workers")
    p.add_argument("--drain", action="store_true", help="Ask active workers to finish current slide and stop")
    p.set_defaults(handler=cmd_pause)

    p = sub.add_parser("resume", help="Resume claims")
    p.set_defaults(handler=cmd_resume)

    p = sub.add_parser("stop-drained", help="Stop drained worker pods through RunPod")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--delete", action="store_true")
    p.set_defaults(handler=cmd_stop_drained)

    p = sub.add_parser("recover", help="Recover expired leases/stale workers")
    p.add_argument("--worker-timeout-seconds", type=int, default=None)
    p.set_defaults(handler=cmd_recover)

    p = sub.add_parser("export", help="Export artifacts to S3/MinIO")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(handler=cmd_export)

    p = sub.add_parser("serve", help="Run the Server Pod control plane using env config")
    p.set_defaults(handler=cmd_serve)

    p = sub.add_parser("reconcile-pods", help="Refresh stored RunPod provider status")
    p.set_defaults(handler=cmd_reconcile_pods)

    p = sub.add_parser("self-test", help="Run server-local worker canary on this Pod")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true", help="Use simulated WSI artifacts")
    mode.add_argument("--real", action="store_true", help="Run one real WSI download/extraction")
    p.add_argument("--max-jobs", type=int, default=1)
    p.add_argument("--batch-jobs", type=int, default=1)
    p.add_argument("--prefetch-jobs", type=int, default=1)
    p.add_argument("--port", type=int, default=18080)
    p.add_argument("--worker-id", default="server-selftest-gpu0")
    p.add_argument("--timeout-seconds", type=int)
    p.add_argument("--keep-db", action="store_true")
    p.add_argument("--skip-plan", action="store_true")
    p.set_defaults(handler=cmd_self_test)

    return parser


def load_settings(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            values[key.strip()] = strip_quotes(value.strip())
    for key, value in os.environ.items():
        if key == "RUNPOD_API_KEY" and values.get("RUNPOD_API_KEY"):
            continue
        if key.startswith(("RUNPOD_", "WSI_", "EXPORT_", "AWS_", "HF_")) or key in ENV_KEYS:
            values[key] = value
    return values


ENV_KEYS = {
    "DB_PATH",
    "RUN_ID",
    "WORKER_TOKEN",
    "SERVER_POD_ID",
    "SERVER_PORT",
    "SERVER_URL",
    "APP_DIR",
    "IMAGE_NAME",
    "GPU_TYPE_ID",
    "GPU_TYPE_IDS",
    "GPU_COUNT",
    "HOURLY_COST",
    "ADJUSTED_HOURLY_COST",
    "NETWORK_VOLUME_ID",
    "DATA_CENTER_IDS",
    "WORKSPACE_ROOT",
    "WORKER_ROLE",
    "WORKER_NAME_PREFIX",
    "WORKER_COUNT",
    "WORKER_BOOTSTRAP_MODE",
    "WORKER_DOCKER_ENTRYPOINT_JSON",
    "WORKER_DOCKER_START_CMD_JSON",
    "DATASETS",
    "WSI_QC_POLICY",
    "PLAN_DIR",
    "DATA_ROOT",
    "MAX_FILES_PER_CHUNK",
    "MAX_BYTES_PER_CHUNK",
    "SOURCE_ADAPTER",
    "ARTIFACT_ROOT",
    "UNI2H_CONFIG_PATH",
    "EXTRACT_SCRIPT",
    "BATCH_SIZE",
    "BATCH_JOBS",
    "PREFETCH_JOBS",
    "LEASE_SECONDS",
    "DEVICE",
    "LOCAL_WSI_CACHE_DIR",
    "MAX_ATTEMPTS",
    "MAX_JOBS",
    "MAX_BYTES",
    "OVERWRITE_ARTIFACTS",
    "COST_CAP_PER_HOUR",
    "EXPORT_DESTINATION",
    "EXPORT_ENDPOINT_URL",
    "EXPORT_INVENTORY_PATH",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
}


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def apply_env_exports(env: dict[str, str]) -> None:
    for key in ("RUNPOD_API_KEY", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "HF_TOKEN"):
        if env.get(key):
            os.environ[key] = env[key]


def get(env: dict[str, str], key: str, default: str = "") -> str:
    return env.get(key, default).strip()


def get_int(env: dict[str, str], key: str, default: int) -> int:
    value = get(env, key)
    return int(value) if value else default


def get_list(env: dict[str, str], key: str) -> list[str]:
    return [item.strip() for item in get(env, key).split(",") if item.strip()]


def db_args(env: dict[str, str]) -> list[str]:
    return ["--db", get(env, "DB_PATH", "/workspace/server_state/runpod_distributed.sqlite")]


def run_id(env: dict[str, str]) -> str:
    return get(env, "RUN_ID", "default")


def cmd_doctor(args: argparse.Namespace, env: dict[str, str]) -> int:
    checks = {
        "DB_PATH": bool(get(env, "DB_PATH")),
        "RUN_ID": bool(run_id(env)),
        "WORKER_TOKEN": bool(get(env, "WORKER_TOKEN")),
        "PLAN_DIR": bool(get(env, "PLAN_DIR")),
        "DATASETS": bool(get(env, "DATASETS")),
        "ARTIFACT_ROOT": bool(get(env, "ARTIFACT_ROOT")),
        "IMAGE_NAME": bool(get(env, "IMAGE_NAME")),
        "SERVER_POD_ID": bool(get(env, "SERVER_POD_ID")),
        "GPU_TYPE_ID(S)": bool(get(env, "GPU_TYPE_ID") or get(env, "GPU_TYPE_IDS")),
        "HOURLY_COST": bool(get(env, "HOURLY_COST")),
        "NETWORK_VOLUME_ID": bool(get(env, "NETWORK_VOLUME_ID")),
        "EXPORT_DESTINATION": bool(get(env, "EXPORT_DESTINATION")),
    }
    result: dict[str, Any] = {"env_file": str(args.env_file), "checks": checks}
    if args.deep:
        result["deep_checks"] = run_deep_checks(env)
    if args.deep and not args.json:
        print("RunPod WSI doctor")
        for key, ok in checks.items():
            print(f"{'ok' if ok else 'missing'}\t{key}")
        deep = result.get("deep_checks") or {}
        for section, value in deep.items():
            if isinstance(value, dict):
                status = value.get("status", "ok" if value.get("ok") else "fail")
                detail = value.get("detail") or value.get("path") or value.get("count") or ""
                print(f"{status}\t{section}\t{detail}")
            else:
                print(f"info\t{section}\t{value}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    missing = [key for key, ok in checks.items() if not ok and key not in {"EXPORT_DESTINATION"}]
    deep_failed = any(
        isinstance(item, dict) and item.get("status") == "fail"
        for item in (result.get("deep_checks") or {}).values()
    )
    return 1 if missing or deep_failed else 0


def run_deep_checks(env: dict[str, str]) -> dict[str, Any]:
    workspace = Path.cwd()
    db_path = Path(get(env, "DB_PATH", "/workspace/server_state/runpod_distributed.sqlite"))
    artifact_root = resolve_config_path(get(env, "ARTIFACT_ROOT", "/workspace/artifacts/wsi_uni2h_v0"), workspace)
    local_cache = resolve_config_path(get(env, "LOCAL_WSI_CACHE_DIR", "/workspace/local_wsi_cache"), workspace)
    extract_script = resolve_config_path(get(env, "EXTRACT_SCRIPT", "scripts/extract_uni2h_features.py"), workspace)
    uni2h_config = resolve_config_path(get(env, "UNI2H_CONFIG_PATH", "configs/uni2h_w8yi_style.yaml"), workspace)
    plan_dir = resolve_config_path(get(env, "PLAN_DIR", "manifests/download_plans_v0_wsi"), workspace)
    return {
        "db_parent": check_writable_path(db_path.parent),
        "artifact_root": check_writable_path(artifact_root),
        "local_cache": check_writable_path(local_cache),
        "plan_rows": count_plan_rows(plan_dir),
        "extract_script": check_file_exists(extract_script),
        "uni2h_config": check_file_exists(uni2h_config),
        "imports": check_imports(("torch", "openslide", "h5py", "PIL", "huggingface_hub", "fastapi", "uvicorn", "rich")),
        "cuda": check_cuda(),
        "secrets": {
            "status": "ok",
            "HF_TOKEN": "present" if get(env, "HF_TOKEN") else "missing",
            "RUNPOD_API_KEY": "present" if get(env, "RUNPOD_API_KEY") else "missing",
            "AWS_ACCESS_KEY_ID": "present" if get(env, "AWS_ACCESS_KEY_ID") else "missing",
            "AWS_SECRET_ACCESS_KEY": "present" if get(env, "AWS_SECRET_ACCESS_KEY") else "missing",
        },
    }


def resolve_config_path(path_value: str, workspace: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else workspace / path


def check_file_exists(path: Path) -> dict[str, Any]:
    return {"status": "ok" if path.exists() else "fail", "path": str(path)}


def check_writable_path(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".doctor_", dir=path, delete=True):
            pass
        return {"status": "ok", "path": str(path)}
    except Exception as exc:
        return {"status": "fail", "path": str(path), "detail": repr(exc)}


def count_plan_rows(plan_dir: Path) -> dict[str, Any]:
    if not plan_dir.exists():
        return {"status": "warn", "path": str(plan_dir), "count": 0, "detail": "plan dir missing"}
    count = 0
    files = 0
    for path in sorted(plan_dir.glob("chunk-*.csv")):
        files += 1
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            count += sum(1 for _ in reader)
    return {"status": "ok" if count else "warn", "path": str(plan_dir), "count": count, "files": files}


def check_imports(module_names: tuple[str, ...]) -> dict[str, Any]:
    modules: dict[str, str] = {}
    status = "ok"
    for name in module_names:
        try:
            module = importlib.import_module(name)
            modules[name] = str(getattr(module, "__version__", "ok"))
        except Exception as exc:
            modules[name] = f"missing: {exc!r}"
            status = "warn"
    return {"status": status, "modules": modules}


def check_cuda() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"status": "warn", "available": False, "detail": f"torch import failed: {exc!r}"}
    available = bool(torch.cuda.is_available())
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(index))
    return {"status": "ok" if available else "warn", "available": available, "devices": devices}


def cmd_build_plan(args: argparse.Namespace, env: dict[str, str]) -> int:
    config = PlanBuildConfig(
        assets=("raw_wsi",),
        datasets=parse_datasets(get(env, "DATASETS")),
        wsi_qc_policy=get(env, "WSI_QC_POLICY", "main_strict"),
        max_files_per_chunk=get_int(env, "MAX_FILES_PER_CHUNK", 50),
        max_bytes_per_chunk=parse_size(get(env, "MAX_BYTES_PER_CHUNK", "50GB")),
        out_dir=Path(get(env, "PLAN_DIR", "manifests/download_plans_v0_wsi")),
        data_root=get(env, "DATA_ROOT", "data"),
    )
    result = build_and_write_plans(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_enqueue(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [
        *db_args(env),
        "enqueue-wsi-uni2h-jobs",
        "--plan-dir",
        get(env, "PLAN_DIR", "manifests/download_plans_v0_wsi"),
        "--run-id",
        run_id(env),
        "--source-adapter",
        get(env, "SOURCE_ADAPTER", "gdc"),
        "--artifact-root",
        get(env, "ARTIFACT_ROOT", "/workspace/artifacts/wsi_uni2h_v0"),
        "--config-path",
        get(env, "UNI2H_CONFIG_PATH", "configs/uni2h_w8yi_style.yaml"),
        "--extract-script",
        get(env, "EXTRACT_SCRIPT", "scripts/extract_uni2h_features.py"),
        "--max-attempts",
        get(env, "MAX_ATTEMPTS", "2"),
    ]
    append_optional(argv, "--datasets", get(env, "DATASETS"))
    append_optional(argv, "--batch-size", get(env, "BATCH_SIZE"))
    append_optional(argv, "--device", get(env, "DEVICE"))
    append_optional(argv, "--local-cache-dir", get(env, "LOCAL_WSI_CACHE_DIR"))
    append_optional(argv, "--max-jobs", get(env, "MAX_JOBS"))
    append_optional(argv, "--max-bytes", get(env, "MAX_BYTES"))
    if truthy(get(env, "OVERWRITE_ARTIFACTS")):
        argv.append("--overwrite-artifacts")
    if args.simulate:
        argv.append("--simulate")
    return distributed_cli.main(argv)


def cmd_start(args: argparse.Namespace, env: dict[str, str]) -> int:
    if not args.skip_plan:
        code = cmd_build_plan(args, env)
        if code != 0:
            return code
    if not args.skip_enqueue:
        code = cmd_enqueue(args, env)
        if code != 0:
            return code
    workers = args.workers
    if workers is None:
        workers = get_int(env, "WORKER_COUNT", 0)
    if workers > 0:
        return add_workers(env, workers, execute=args.execute)
    return 0


def cmd_add_workers(args: argparse.Namespace, env: dict[str, str]) -> int:
    count = args.count if args.count is not None else get_int(env, "WORKER_COUNT", 1)
    return add_workers(env, count, execute=args.execute)


def add_workers(env: dict[str, str], count: int, *, execute: bool) -> int:
    if count <= 0:
        print("worker count is 0; nothing to add")
        return 0
    prefix = get(env, "WORKER_NAME_PREFIX", "wsi-worker")
    status = 0
    for index in range(1, count + 1):
        argv = [
            *db_args(env),
            "add-worker",
            "--name",
            f"{prefix}-{index}",
            "--image-name",
            require(env, "IMAGE_NAME"),
            "--server-pod-id",
            require(env, "SERVER_POD_ID"),
            "--server-port",
            get(env, "SERVER_PORT", "8080"),
            "--run-id",
            run_id(env),
            "--workspace-root",
            get(env, "WORKSPACE_ROOT", "/workspace"),
            "--gpu-count",
            get(env, "GPU_COUNT", "1"),
            "--hourly-cost",
            require(env, "HOURLY_COST"),
            "--worker-token",
            require(env, "WORKER_TOKEN"),
            "--worker-role",
            get(env, "WORKER_ROLE", "wsi-preprocess"),
        ]
        for gpu_type in gpu_types(env):
            argv.extend(["--gpu-type-id", gpu_type])
        append_optional(argv, "--network-volume-id", get(env, "NETWORK_VOLUME_ID"))
        append_optional(argv, "--adjusted-hourly-cost", get(env, "ADJUSTED_HOURLY_COST"))
        for item in worker_env_items(env):
            argv.extend(["--worker-env", item])
        entrypoint_json, start_cmd_json = worker_start_command(env)
        append_optional(argv, "--docker-entrypoint-json", entrypoint_json)
        append_optional(argv, "--docker-start-cmd-json", start_cmd_json)
        for data_center in get_list(env, "DATA_CENTER_IDS"):
            argv.extend(["--data-center-id", data_center])
        if execute:
            argv.append("--execute")
        code = distributed_cli.main(argv)
        status = status or code
    return status


def worker_env_items(env: dict[str, str]) -> list[str]:
    keys = [
        "HF_TOKEN",
        "APP_DIR",
        "LOCAL_WSI_CACHE_DIR",
        "BATCH_JOBS",
        "PREFETCH_JOBS",
        "PREFETCH_MAX_BYTES",
        "LEASE_SECONDS",
    ]
    items = []
    for key in keys:
        value = get(env, key)
        if value:
            items.append(f"{key}={value}")
    return items


def worker_start_command(env: dict[str, str]) -> tuple[str, str]:
    explicit_entrypoint = get(env, "WORKER_DOCKER_ENTRYPOINT_JSON")
    explicit_start_cmd = get(env, "WORKER_DOCKER_START_CMD_JSON")
    if explicit_entrypoint or explicit_start_cmd:
        return explicit_entrypoint, explicit_start_cmd
    if get(env, "WORKER_BOOTSTRAP_MODE") != "network-volume":
        return "", ""
    app_dir = get(env, "APP_DIR", get(env, "WORKSPACE_ROOT", "/workspace/uni2h-preprocessing"))
    return json.dumps(["/bin/bash"]), json.dumps([f"{app_dir}/scripts/runpod_worker_bootstrap.sh"])


def gpu_types(env: dict[str, str]) -> list[str]:
    values = get_list(env, "GPU_TYPE_IDS")
    if values:
        return values
    value = get(env, "GPU_TYPE_ID")
    return [value] if value else []


def cmd_status(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [*db_args(env), "status"]
    if args.plain:
        argv.append("--plain")
    return distributed_cli.main(argv)


def cmd_pause(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [*db_args(env), "pause-run", "--run-id", run_id(env)]
    if args.drain:
        argv.append("--drain-workers")
    return distributed_cli.main(argv)


def cmd_resume(args: argparse.Namespace, env: dict[str, str]) -> int:
    return distributed_cli.main([*db_args(env), "resume-run", "--run-id", run_id(env)])


def cmd_stop_drained(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [*db_args(env), "terminate-drained-pods", "--run-id", run_id(env)]
    if args.execute:
        argv.append("--execute")
    if args.delete:
        argv.append("--delete")
    return distributed_cli.main(argv)


def cmd_recover(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [*db_args(env), "recover-stale"]
    timeout = args.worker_timeout_seconds or get(env, "WORKER_TIMEOUT_SECONDS")
    append_optional(argv, "--worker-timeout-seconds", str(timeout) if timeout else "")
    return distributed_cli.main(argv)


def cmd_reconcile_pods(args: argparse.Namespace, env: dict[str, str]) -> int:
    return distributed_cli.main([*db_args(env), "reconcile-pods", "--run-id", run_id(env)])


def cmd_self_test(args: argparse.Namespace, env: dict[str, str]) -> int:
    simulate = not args.real
    timeout = args.timeout_seconds or (600 if simulate else 4 * 3600)
    self_env = make_self_test_env(env, args)
    artifact_root = Path(self_env["ARTIFACT_ROOT"])
    log_dir = artifact_root / "_selftest_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.keep_db:
        code = distributed_cli.main([*db_args(self_env), "init-db", "--reset"])
        if code != 0:
            return code
    if not args.skip_plan:
        code = cmd_build_plan(argparse.Namespace(), self_env)
        if code != 0:
            return code
    code = cmd_enqueue(argparse.Namespace(simulate=simulate), self_env)
    if code != 0:
        return code

    store = SQLiteStore(self_env["DB_PATH"])
    jobs = store.list_jobs()
    if not jobs:
        print(json.dumps({"passed": False, "error": "self-test enqueued no jobs"}, indent=2))
        return 1

    server_log = (log_dir / "server.log").open("w", encoding="utf-8")
    worker_log = (log_dir / "worker.log").open("w", encoding="utf-8")
    server_proc: subprocess.Popen[str] | None = None
    worker_proc: subprocess.Popen[str] | None = None
    try:
        server_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.distributed.cli",
                *db_args(self_env),
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--token",
                require(self_env, "WORKER_TOKEN"),
            ],
            cwd=Path.cwd(),
            env=subprocess_env(self_env),
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        wait_for_health(f"http://127.0.0.1:{args.port}/health", server_proc, timeout_seconds=60)
        worker_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scripts.distributed.cli",
                "wsi-worker",
                "--server-url",
                f"http://127.0.0.1:{args.port}",
                "--token",
                require(self_env, "WORKER_TOKEN"),
                "--run-id",
                run_id(self_env),
                "--worker-id",
                args.worker_id,
                "--workspace-root",
                str(Path.cwd()),
                "--local-cache-dir",
                self_env["LOCAL_WSI_CACHE_DIR"],
                "--batch-jobs",
                str(args.batch_jobs),
                "--prefetch-jobs",
                str(args.prefetch_jobs),
                "--prefetch-max-bytes",
                get(self_env, "PREFETCH_MAX_BYTES", "50GB"),
                "--max-batches",
                "1",
                "--lease-seconds",
                get(self_env, "LEASE_SECONDS", "3600"),
            ],
            cwd=Path.cwd(),
            env=subprocess_env(self_env),
            stdout=worker_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        passed, reason = wait_for_self_test_completion(store, worker_proc, timeout_seconds=timeout)
        validation = validate_wsi_artifacts(artifact_root, simulate_ok=simulate)
        report = {
            "passed": bool(passed and validation["passed"]),
            "mode": "simulate" if simulate else "real",
            "reason": reason,
            "db_path": self_env["DB_PATH"],
            "artifact_root": str(artifact_root),
            "server_log": str(log_dir / "server.log"),
            "worker_log": str(log_dir / "worker.log"),
            "jobs": [job_summary(job) for job in store.list_jobs()],
            "validation": validation,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["passed"] else 1
    finally:
        terminate_process(worker_proc)
        terminate_process(server_proc)
        worker_log.close()
        server_log.close()


def cmd_export(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [
        "export-artifacts",
        "--artifact-root",
        get(env, "ARTIFACT_ROOT", "/workspace/artifacts/wsi_uni2h_v0"),
        "--destination",
        require(env, "EXPORT_DESTINATION"),
        "--inventory-path",
        get(env, "EXPORT_INVENTORY_PATH", "/workspace/manifests/export_inventory_wsi_uni2h_v0.csv"),
    ]
    append_optional(argv, "--endpoint-url", get(env, "EXPORT_ENDPOINT_URL") or get(env, "AWS_ENDPOINT_URL"))
    append_optional(argv, "--access-key-id", get(env, "AWS_ACCESS_KEY_ID"))
    append_optional(argv, "--secret-access-key", get(env, "AWS_SECRET_ACCESS_KEY"))
    append_optional(argv, "--region-name", get(env, "AWS_DEFAULT_REGION"))
    if args.dry_run:
        argv.append("--dry-run")
    if args.overwrite:
        argv.append("--overwrite")
    return distributed_cli.main(argv)


def cmd_serve(args: argparse.Namespace, env: dict[str, str]) -> int:
    argv = [
        *db_args(env),
        "serve",
        "--host",
        get(env, "SERVER_HOST", "0.0.0.0"),
        "--port",
        get(env, "SERVER_PORT", "8080"),
        "--token",
        require(env, "WORKER_TOKEN"),
    ]
    return distributed_cli.main(argv)


def make_self_test_env(env: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    values = dict(env)
    base_db = Path(get(values, "DB_PATH", "/workspace/server_state/runpod_distributed.sqlite"))
    artifact_root = resolve_config_path(get(values, "ARTIFACT_ROOT", "/workspace/artifacts/wsi_uni2h_v0"), Path.cwd()) / "_selftest"
    local_cache = resolve_config_path(get(values, "LOCAL_WSI_CACHE_DIR", "/workspace/local_wsi_cache"), Path.cwd()) / "_selftest"
    values["DB_PATH"] = str(Path(str(base_db) + ".selftest.sqlite"))
    values["ARTIFACT_ROOT"] = str(artifact_root)
    values["LOCAL_WSI_CACHE_DIR"] = str(local_cache)
    values["RUN_ID"] = f"{run_id(values)}-selftest"
    values["SERVER_URL"] = f"http://127.0.0.1:{args.port}"
    values["SERVER_PORT"] = str(args.port)
    values["MAX_JOBS"] = str(args.max_jobs)
    values["BATCH_JOBS"] = str(args.batch_jobs)
    values["PREFETCH_JOBS"] = str(args.prefetch_jobs)
    values["OVERWRITE_ARTIFACTS"] = "1"
    if not get(values, "LEASE_SECONDS"):
        values["LEASE_SECONDS"] = "3600"
    if not get(values, "PREFETCH_MAX_BYTES"):
        values["PREFETCH_MAX_BYTES"] = "50GB"
    return values


def subprocess_env(env: dict[str, str]) -> dict[str, str]:
    merged = dict(os.environ)
    for key, value in env.items():
        if value is not None:
            merged[key] = str(value)
    return merged


def wait_for_health(url: str, process: subprocess.Popen[str], *, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited before health check passed with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"server health check timed out: {last_error}")


def wait_for_self_test_completion(
    store: SQLiteStore,
    worker_proc: subprocess.Popen[str],
    *,
    timeout_seconds: int,
) -> tuple[bool, str]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        jobs = store.list_jobs()
        if jobs and all(job.state == JobState.COMPLETED for job in jobs):
            return True, "all jobs completed"
        failed = [job for job in jobs if job.state == JobState.FAILED]
        if failed:
            return False, f"{len(failed)} job(s) failed"
        if worker_proc.poll() is not None and any(job.state != JobState.COMPLETED for job in jobs):
            return False, f"worker exited before completion with code {worker_proc.returncode}"
        time.sleep(5)
    return False, "self-test timed out"


def terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def job_summary(job: Any) -> dict[str, Any]:
    return {
        "id": job.id,
        "state": job.state.value,
        "worker_id": job.worker_id,
        "attempt": job.attempt,
        "max_attempts": job.max_attempts,
        "completed_units": job.completed_units,
        "total_units": job.total_units,
        "error": job.error,
        "output_path": job.output_path,
    }


def append_optional(argv: list[str], flag: str, value: str) -> None:
    if value:
        argv.extend([flag, value])


def require(env: dict[str, str], key: str) -> str:
    value = get(env, key)
    if not value:
        raise SystemExit(f"Missing required setting: {key}")
    return value


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
