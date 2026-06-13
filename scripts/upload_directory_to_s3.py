#!/usr/bin/env python
"""Upload a local directory to S3-compatible storage.

This is intentionally generic so it can be used for RunPod Network Volume S3
imports, MinIO exports, and small manifest/config sync tasks.
"""

from __future__ import annotations

import argparse
import csv
import mimetypes
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_env_file(path: Path | None) -> dict[str, str]:
    values: dict[str, str] = {}
    if path is None or not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        values[key.strip()] = strip_quotes(value.strip())
    return values


def strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"destination must be s3://bucket/prefix, got: {uri}")
    return parsed.netloc, parsed.path.strip("/")


def discover_files(source_dir: Path) -> list[Path]:
    return sorted(path for path in source_dir.rglob("*") if path.is_file())


def make_client(*, env: dict[str, str], endpoint_url: str | None, region_name: str):
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("boto3 and botocore are required") from exc

    access_key = env.get("AWS_ACCESS_KEY_ID") or env.get("S3_ACCESS_KEY_ID") or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = env.get("AWS_SECRET_ACCESS_KEY") or env.get("S3_SECRET_ACCESS_KEY") or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise ValueError("Missing AWS_ACCESS_KEY_ID/S3_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY/S3_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url or env.get("AWS_ENDPOINT_URL") or env.get("S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL"),
        region_name=region_name or env.get("AWS_DEFAULT_REGION") or env.get("S3_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": "path"}),
    )


def upload_directory(args: argparse.Namespace) -> int:
    env = load_env_file(args.env_file)
    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    destination = args.destination or env.get("S3_DESTINATION")
    if not destination:
        raise ValueError("Provide --destination or S3_DESTINATION in env file")
    bucket, prefix = parse_s3_uri(destination)
    files = discover_files(source_dir)
    client = None if args.dry_run else make_client(env=env, endpoint_url=args.endpoint_url, region_name=args.region_name)
    rows: list[dict[str, object]] = []
    for path in files:
        rel = path.relative_to(source_dir).as_posix()
        key = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
        status = "planned"
        error = ""
        try:
            if not args.dry_run and client is not None:
                extra_args = {}
                content_type = mimetypes.guess_type(path.name)[0]
                if content_type:
                    extra_args["ContentType"] = content_type
                if args.skip_existing and object_exists(client, bucket, key, path.stat().st_size):
                    status = "skipped_existing"
                else:
                    client.upload_file(str(path), bucket, key, ExtraArgs=extra_args or None)
                    status = "uploaded"
        except Exception as exc:
            status = "failed"
            error = repr(exc)
        rows.append(
            {
                "source_path": str(path),
                "relative_path": rel,
                "destination_uri": f"s3://{bucket}/{key}",
                "size_bytes": path.stat().st_size,
                "status": status,
                "created_at": utc_now(),
                "error_message": error,
            }
        )
    write_inventory(args.inventory_path, rows)
    print_summary(rows, args.inventory_path)
    return 1 if any(row["status"] == "failed" for row in rows) else 0


def object_exists(client, bucket: str, key: str, size: int) -> bool:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return int(response.get("ContentLength") or -1) == int(size)


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["source_path", "relative_path", "destination_uri", "size_bytes", "status", "created_at", "error_message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], inventory_path: Path) -> None:
    by_status: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        status = str(row["status"])
        by_status[status] = by_status.get(status, 0) + 1
        total_bytes += int(row["size_bytes"])
    print(
        {
            "files": len(rows),
            "bytes": total_bytes,
            "by_status": by_status,
            "inventory_path": str(inventory_path),
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--destination", help="s3://bucket/prefix")
    parser.add_argument("--env-file", type=Path, default=Path("configs/runpod_network_volume_s3.env"))
    parser.add_argument("--endpoint-url", default="")
    parser.add_argument("--region-name", default="")
    parser.add_argument("--inventory-path", type=Path, default=Path("manifests/s3_upload_inventory.csv"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(skip_existing=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.skip_existing = not args.overwrite
    return upload_directory(args)


if __name__ == "__main__":
    raise SystemExit(main())
