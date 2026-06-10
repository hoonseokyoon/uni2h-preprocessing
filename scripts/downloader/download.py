"""Execute and verify download plan rows."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable


INVENTORY_COLUMNS = [
    "download_id",
    "chunk_id",
    "asset_kind",
    "source_adapter",
    "dataset",
    "case_submitter_id",
    "file_id",
    "target_path",
    "expected_size_bytes",
    "actual_size_bytes",
    "expected_md5",
    "actual_md5",
    "download_status",
    "verification_status",
    "attempt_count",
    "started_at",
    "finished_at",
    "error_message",
]


@dataclass(frozen=True)
class DownloadConfig:
    plan_dir: Path | None = None
    plan_file: Path | None = None
    index_file: Path | None = None
    output_root: Path = Path(".")
    inventory_path: Path = Path("manifests") / "download_inventory_v0.csv"
    asset_kind: str = ""
    source_adapter: str = ""
    datasets: tuple[str, ...] = ()
    concurrency: int = 4
    timeout_seconds: int = 120
    retries: int = 3
    max_files: int | None = None
    max_bytes: int | None = None
    verify_only: bool = False
    resume: bool = True
    overwrite: bool = False
    progress_interval_seconds: float = 5.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "na", "n/a", "unknown", "not reported", "--"}:
        return ""
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in INVENTORY_COLUMNS})


def plan_files_from_config(config: DownloadConfig) -> list[Path]:
    if config.plan_file is not None:
        return [config.plan_file]
    if config.index_file is not None:
        index_rows = read_csv(config.index_file)
        return [Path(row["plan_path"]) for row in index_rows]
    if config.plan_dir is not None:
        index = config.plan_dir / "download_plan_v0_index.csv"
        if index.exists():
            return [Path(row["plan_path"]) for row in read_csv(index)]
        return sorted(path for path in config.plan_dir.glob("download_plan_v0__*.csv") if not path.name.endswith("_index.csv"))
    raise ValueError("Provide plan_dir, plan_file, or index_file")


def row_size(row: dict[str, str]) -> int | None:
    text = clean(row.get("expected_size_bytes"))
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value >= 0 else None


def load_plan_rows(config: DownloadConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in plan_files_from_config(config):
        rows.extend(read_csv(path))

    filtered: list[dict[str, str]] = []
    selected_bytes = 0
    for row in rows:
        if config.asset_kind and clean(row.get("asset_kind")) != config.asset_kind:
            continue
        if config.source_adapter and clean(row.get("source_adapter")) != config.source_adapter:
            continue
        if config.datasets and clean(row.get("dataset")) not in config.datasets:
            continue
        size = row_size(row)
        if config.max_files is not None and len(filtered) >= config.max_files:
            break
        if config.max_bytes is not None and size is not None and selected_bytes + size > config.max_bytes:
            break
        filtered.append(row)
        if size is not None:
            selected_bytes += size
    return filtered


def target_path(row: dict[str, str], output_root: Path) -> Path:
    rel = clean(row.get("target_rel_path"))
    if not rel:
        raise ValueError(f"Missing target_rel_path for {row.get('download_id')}")
    path = Path(rel)
    if path.is_absolute():
        return path
    return output_root / path


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def verify_file(path: Path, row: dict[str, str]) -> tuple[bool, str, int, str]:
    if not path.exists():
        return False, "missing", 0, ""
    actual_size = path.stat().st_size
    expected_size = row_size(row)
    if expected_size is not None and actual_size != expected_size:
        return False, "size_mismatch", actual_size, ""
    expected_md5 = clean(row.get("expected_md5"))
    actual_md5 = file_md5(path) if expected_md5 else ""
    if expected_md5 and actual_md5.lower() != expected_md5.lower():
        return False, "md5_mismatch", actual_size, actual_md5
    return True, "pass", actual_size, actual_md5


def data_url_for_row(row: dict[str, str]) -> str:
    adapter = clean(row.get("source_adapter"))
    file_id = clean(row.get("file_id"))
    if adapter != "gdc":
        raise ValueError(f"Unsupported download adapter for executor: {adapter}")
    if not file_id:
        raise ValueError(f"Missing file_id for {row.get('download_id')}")
    return f"https://api.gdc.cancer.gov/data/{file_id}"


def open_download(url: str, timeout: int, start_byte: int = 0):
    headers = {"User-Agent": "path-rna-fusion-downloader/0.1"}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, timeout=timeout)


def atomic_download(row: dict[str, str], config: DownloadConfig) -> dict[str, object]:
    started_at = utc_now()
    attempts = 0
    path = target_path(row, config.output_root)
    tmp_path = path.with_name(path.name + f".part.{os.getpid()}")
    expected_size = row_size(row)
    expected_md5 = clean(row.get("expected_md5"))
    path.parent.mkdir(parents=True, exist_ok=True)

    if config.verify_only:
        ok, reason, actual_size, actual_md5 = verify_file(path, row)
        return inventory_row(
            row,
            path,
            expected_size,
            actual_size,
            expected_md5,
            actual_md5,
            "verified" if ok else "failed",
            reason,
            attempts,
            started_at,
            utc_now(),
            "" if ok else reason,
        )

    if path.exists() and not config.overwrite:
        ok, reason, actual_size, actual_md5 = verify_file(path, row)
        if ok:
            return inventory_row(row, path, expected_size, actual_size, expected_md5, actual_md5, "skipped", "pass", attempts, started_at, utc_now(), "")
        path.unlink()

    url = data_url_for_row(row)
    last_error = ""
    for attempt in range(1, config.retries + 1):
        attempts = attempt
        try:
            start = tmp_path.stat().st_size if config.resume and tmp_path.exists() else 0
            with open_download(url, config.timeout_seconds, start_byte=start) as response:
                status = getattr(response, "status", None)
                mode = "ab" if start > 0 and status == 206 else "wb"
                with tmp_path.open(mode) as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            ok, reason, actual_size, actual_md5 = verify_file(tmp_path, row)
            if not ok:
                last_error = reason
                if reason == "md5_mismatch" and tmp_path.exists():
                    tmp_path.unlink()
                time.sleep(min(2**attempt, 30))
                continue
            os.replace(tmp_path, path)
            return inventory_row(row, path, expected_size, actual_size, expected_md5, actual_md5, "downloaded", "pass", attempts, started_at, utc_now(), "")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = str(exc)
            time.sleep(min(2**attempt, 30))

    actual_size = tmp_path.stat().st_size if tmp_path.exists() else 0
    actual_md5 = ""
    return inventory_row(row, path, expected_size, actual_size, expected_md5, actual_md5, "failed", "fail", attempts, started_at, utc_now(), last_error)


def inventory_row(
    row: dict[str, str],
    path: Path,
    expected_size: int | None,
    actual_size: int,
    expected_md5: str,
    actual_md5: str,
    download_status: str,
    verification_status: str,
    attempt_count: int,
    started_at: str,
    finished_at: str,
    error_message: str,
) -> dict[str, object]:
    return {
        "download_id": clean(row.get("download_id")),
        "chunk_id": clean(row.get("chunk_id")),
        "asset_kind": clean(row.get("asset_kind")),
        "source_adapter": clean(row.get("source_adapter")),
        "dataset": clean(row.get("dataset")),
        "case_submitter_id": clean(row.get("case_submitter_id")),
        "file_id": clean(row.get("file_id")),
        "target_path": str(path),
        "expected_size_bytes": "" if expected_size is None else expected_size,
        "actual_size_bytes": actual_size,
        "expected_md5": expected_md5,
        "actual_md5": actual_md5,
        "download_status": download_status,
        "verification_status": verification_status,
        "attempt_count": attempt_count,
        "started_at": started_at,
        "finished_at": finished_at,
        "error_message": error_message,
    }


def summarize_inventory(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"total": len(rows), "by_download_status": {}, "by_verification_status": {}, "bytes": 0}
    for row in rows:
        status = clean(row.get("download_status"))
        verify = clean(row.get("verification_status"))
        result["by_download_status"][status] = result["by_download_status"].get(status, 0) + 1  # type: ignore[index, union-attr]
        result["by_verification_status"][verify] = result["by_verification_status"].get(verify, 0) + 1  # type: ignore[index, union-attr]
        try:
            result["bytes"] = int(result["bytes"]) + int(row.get("actual_size_bytes") or 0)
        except ValueError:
            pass
    return result


def run_download_plan(config: DownloadConfig) -> dict[str, object]:
    rows = load_plan_rows(config)
    if not rows:
        summary = {"selected_rows": 0, "inventory_path": str(config.inventory_path), "inventory": summarize_inventory([])}
        write_inventory(config.inventory_path, [])
        return summary

    print(
        json.dumps(
            {
                "selected_rows": len(rows),
                "concurrency": config.concurrency,
                "verify_only": config.verify_only,
                "inventory_path": str(config.inventory_path),
            },
            ensure_ascii=False,
        )
    )
    results: list[dict[str, object]] = []
    lock = Lock()
    completed = 0
    started = time.time()
    last_report = started

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = [executor.submit(atomic_download, row, config) for row in rows]
        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                completed += 1
                now = time.time()
                if now - last_report >= config.progress_interval_seconds or completed == len(rows):
                    elapsed = max(now - started, 0.001)
                    rate = completed / elapsed
                    remaining = len(rows) - completed
                    eta_seconds = remaining / rate if rate > 0 else 0
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(rows),
                                "rate_files_per_sec": round(rate, 3),
                                "eta_seconds": int(eta_seconds),
                                "last_status": result.get("download_status"),
                                "last_verify": result.get("verification_status"),
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_report = now

    write_inventory(config.inventory_path, results)
    return {"selected_rows": len(rows), "inventory_path": str(config.inventory_path), "inventory": summarize_inventory(results)}


def find_complete_paths(inventory_path: Path) -> list[Path]:
    rows = read_csv(inventory_path)
    paths: list[Path] = []
    for row in rows:
        if clean(row.get("verification_status")) == "pass":
            paths.append(Path(row["target_path"]))
    return paths
