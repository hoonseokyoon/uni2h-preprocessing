#!/usr/bin/env python
"""Simple RunPod WSI workflow wrapper.

This script keeps the common WSI preprocessing operations behind a small set of
commands. Values come from a dotenv-style config file, then environment
variables, then explicit CLI flags.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.distributed import cli as distributed_cli
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
    "DEVICE",
    "LOCAL_WSI_CACHE_DIR",
    "MAX_ATTEMPTS",
    "MAX_JOBS",
    "MAX_BYTES",
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
    print(json.dumps({"env_file": str(args.env_file), "checks": checks}, indent=2, ensure_ascii=False))
    missing = [key for key, ok in checks.items() if not ok and key not in {"EXPORT_DESTINATION"}]
    return 1 if missing else 0


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
        for data_center in get_list(env, "DATA_CENTER_IDS"):
            argv.extend(["--data-center-id", data_center])
        if execute:
            argv.append("--execute")
        code = distributed_cli.main(argv)
        status = status or code
    return status


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


def append_optional(argv: list[str], flag: str, value: str) -> None:
    if value:
        argv.extend([flag, value])


def require(env: dict[str, str], key: str) -> str:
    value = get(env, key)
    if not value:
        raise SystemExit(f"Missing required setting: {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
