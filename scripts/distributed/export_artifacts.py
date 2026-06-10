"""Export completed artifact trees to S3-compatible storage."""

from __future__ import annotations

import csv
import fnmatch
import mimetypes
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


EXPORT_INVENTORY_COLUMNS = [
    "source_path",
    "relative_path",
    "destination_uri",
    "size_bytes",
    "status",
    "started_at",
    "finished_at",
    "error_message",
]

DEFAULT_ARTIFACT_NAMES = (
    "features.h5",
    "overlay.png",
    "thumbnail.jpg",
    "tissue_mask.png",
    "qc_preview.jpg",
    "manifest.json",
)


@dataclass(frozen=True)
class ExportConfig:
    artifact_root: Path
    destination: str
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region_name: str = "us-east-1"
    inventory_path: Path = Path("manifests/export_inventory_v0.csv")
    include: tuple[str, ...] = DEFAULT_ARTIFACT_NAMES
    dry_run: bool = False
    skip_existing: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"destination must be an s3://bucket/prefix URI: {uri}")
    prefix = parsed.path.strip("/")
    return parsed.netloc, prefix


def discover_artifacts(root: Path, include: Iterable[str] = DEFAULT_ARTIFACT_NAMES) -> list[Path]:
    root = root.resolve()
    patterns = tuple(include)
    files: list[Path] = []
    for manifest in sorted(root.rglob("manifest.json")):
        slide_dir = manifest.parent
        for name in DEFAULT_ARTIFACT_NAMES:
            path = slide_dir / name
            if path.exists() and matches_any(path.name, patterns):
                files.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def matches_any(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def destination_key(root: Path, source: Path, prefix: str) -> str:
    rel = source.resolve().relative_to(root.resolve()).as_posix()
    return f"{prefix.rstrip('/')}/{rel}" if prefix else rel


def destination_uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in EXPORT_INVENTORY_COLUMNS})


def make_s3_client(config: ExportConfig):
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3/MinIO export. Install boto3 or use --dry-run.") from exc
    kwargs = {
        "service_name": "s3",
        "endpoint_url": config.endpoint_url,
        "region_name": config.region_name,
    }
    access_key = config.access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = config.secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    if access_key:
        kwargs["aws_access_key_id"] = access_key
    if secret_key:
        kwargs["aws_secret_access_key"] = secret_key
    return boto3.client(**kwargs)


def object_exists_with_size(client, bucket: str, key: str, size: int) -> bool:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return False
    return int(response.get("ContentLength") or -1) == int(size)


def export_artifacts(config: ExportConfig) -> dict[str, object]:
    root = config.artifact_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"artifact root not found: {root}")
    bucket, prefix = parse_s3_uri(config.destination)
    files = discover_artifacts(root, config.include)
    client = None if config.dry_run else make_s3_client(config)
    rows: list[dict[str, object]] = []
    for source in files:
        started = utc_now()
        key = destination_key(root, source, prefix)
        size = source.stat().st_size
        status = "planned"
        error = ""
        try:
            if not config.dry_run and client is not None:
                if config.skip_existing and object_exists_with_size(client, bucket, key, size):
                    status = "skipped_existing"
                else:
                    extra_args = {}
                    content_type = mimetypes.guess_type(source.name)[0]
                    if content_type:
                        extra_args["ContentType"] = content_type
                    client.upload_file(
                        str(source),
                        bucket,
                        key,
                        ExtraArgs=extra_args or None,
                    )
                    status = "uploaded"
        except Exception as exc:
            status = "failed"
            error = repr(exc)
        rows.append(
            {
                "source_path": str(source),
                "relative_path": source.resolve().relative_to(root).as_posix(),
                "destination_uri": destination_uri(bucket, key),
                "size_bytes": size,
                "status": status,
                "started_at": started,
                "finished_at": utc_now(),
                "error_message": error,
            }
        )
    write_inventory(config.inventory_path, rows)
    return summarize(rows, config)


def summarize(rows: list[dict[str, object]], config: ExportConfig) -> dict[str, object]:
    by_status: dict[str, int] = {}
    total_bytes = 0
    for row in rows:
        status = str(row.get("status") or "")
        by_status[status] = by_status.get(status, 0) + 1
        total_bytes += int(row.get("size_bytes") or 0)
    return {
        "artifact_root": str(config.artifact_root),
        "destination": config.destination,
        "dry_run": config.dry_run,
        "files": len(rows),
        "bytes": total_bytes,
        "by_status": by_status,
        "inventory_path": str(config.inventory_path),
        "created_at": utc_now(),
    }
