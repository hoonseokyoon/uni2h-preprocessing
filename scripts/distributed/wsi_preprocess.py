"""WSI stage-in and UNI2-h preprocessing task implementation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[float, float, str | None], None]
TASK_TOTAL_UNITS = 100.0


@dataclass(frozen=True)
class StagedWSI:
    path: Path
    bytes: int
    source: str
    cleanup_root: Path | None = None


@dataclass(frozen=True)
class WSITaskOutput:
    output_path: str | None
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "na", "n/a", "unknown", "not reported", "--"}:
        return ""
    return text


def safe_path_part(text: str) -> str:
    value = clean(text)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value) if value else "unknown"


def parse_optional_int(value: object) -> int | None:
    text = clean(value)
    if not text:
        return None
    try:
        parsed = int(float(text))
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def planned_artifact_paths(payload: dict[str, Any], workspace_root: str | Path) -> dict[str, Path]:
    workspace = Path(workspace_root)
    artifact_root = resolve_path(payload.get("artifact_root", "artifacts/wsi_uni2h_v0"), workspace)
    dataset = safe_path_part(clean(payload.get("dataset")) or "unknown_dataset")
    case_id = safe_path_part(clean(payload.get("case_submitter_id")) or "unknown_case")
    slide_key = safe_path_part(slide_key_from_payload(payload))
    slide_dir = artifact_root / dataset / case_id / slide_key
    return {
        "slide_dir": slide_dir,
        "features_h5": slide_dir / "features.h5",
        "overlay_png": slide_dir / "overlay.png",
        "thumbnail_jpg": slide_dir / "thumbnail.jpg",
        "tissue_mask_png": slide_dir / "tissue_mask.png",
        "qc_preview_jpg": slide_dir / "qc_preview.jpg",
        "manifest_json": slide_dir / "manifest.json",
    }


def should_skip_existing(payload: dict[str, Any], workspace_root: str | Path) -> bool:
    if bool(payload.get("overwrite_artifacts", False)):
        return False
    paths = planned_artifact_paths(payload, workspace_root)
    return paths["features_h5"].exists() and paths["manifest_json"].exists()


def run_wsi_uni2h_task(
    payload: dict[str, Any],
    *,
    job_id: str,
    workspace_root: str | Path,
    progress: ProgressCallback,
) -> WSITaskOutput:
    """Run a complete worker-local stage-in -> extraction -> publish cycle."""

    if should_skip_existing(payload, workspace_root):
        paths = planned_artifact_paths(payload, workspace_root)
        progress(TASK_TOTAL_UNITS, TASK_TOTAL_UNITS, "existing artifacts found; skipped")
        return WSITaskOutput(
            str(paths["features_h5"]),
            {
                "task_type": "wsi_uni2h",
                "status": "skipped_existing",
                "manifest_path": str(paths["manifest_json"]),
            },
        )

    local_cache_dir = resolve_path(
        payload.get("local_cache_dir") or os.environ.get("LOCAL_WSI_CACHE_DIR") or "local_wsi_cache",
        workspace_root,
    )
    staged = stage_wsi(payload, job_id=job_id, workspace_root=workspace_root, local_cache_dir=local_cache_dir, progress=progress)
    try:
        return execute_staged_wsi_task(payload, staged, job_id=job_id, workspace_root=workspace_root, progress=progress)
    finally:
        cleanup_staged_wsi(staged, payload)


def stage_wsi(
    payload: dict[str, Any],
    *,
    job_id: str,
    workspace_root: str | Path,
    local_cache_dir: str | Path,
    progress: ProgressCallback,
) -> StagedWSI:
    """Stage one WSI into worker-local storage."""

    local_root = Path(local_cache_dir) / "wsi_stage" / safe_path_part(job_id)
    raw_dir = local_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_wsi_path = clean(payload.get("raw_wsi_path"))
    if raw_wsi_path:
        path = resolve_path(raw_wsi_path, workspace_root)
        if not path.exists():
            raise FileNotFoundError(f"raw_wsi_path does not exist: {path}")
        verify_expected_file(path, payload)
        progress(35, TASK_TOTAL_UNITS, f"using local WSI {path.name}")
        cleanup_root = path.parent if bool(payload.get("delete_raw_wsi_after_extract", False)) else None
        return StagedWSI(path=path, bytes=path.stat().st_size, source="local", cleanup_root=cleanup_root)

    adapter = clean(payload.get("source_adapter"))
    filename = clean(payload.get("expected_file_name")) or f"{safe_path_part(job_id)}.svs"
    staged_path = raw_dir / filename

    if adapter == "gdc":
        file_id = clean(payload.get("file_id"))
        if not file_id:
            raise ValueError(f"missing GDC file_id for job {job_id}")
        url = f"https://api.gdc.cancer.gov/data/{file_id}"
    elif adapter in {"http", "https", "url"} or clean(payload.get("remote_url")).startswith(("http://", "https://")):
        url = clean(payload.get("remote_url"))
        if not url:
            raise ValueError(f"missing remote_url for job {job_id}")
    else:
        raise NotImplementedError(
            f"WSI source_adapter={adapter!r} requires pre-staged raw_wsi_path or a dedicated adapter"
        )

    if staged_path.exists() and verify_expected_file(staged_path, payload, strict=False):
        progress(35, TASK_TOTAL_UNITS, f"reused staged WSI {staged_path.name}")
        return StagedWSI(path=staged_path, bytes=staged_path.stat().st_size, source=adapter, cleanup_root=local_root)

    download_file(
        url,
        staged_path,
        payload,
        progress=lambda downloaded, total, message: progress(
            5 + 30 * min(1.0, downloaded / max(total or downloaded or 1, 1)),
            TASK_TOTAL_UNITS,
            message,
        ),
        timeout_seconds=int(payload.get("download_timeout_seconds", 180)),
        retries=int(payload.get("download_retries", 4)),
    )
    progress(35, TASK_TOTAL_UNITS, f"staged WSI {staged_path.name}")
    return StagedWSI(path=staged_path, bytes=staged_path.stat().st_size, source=adapter, cleanup_root=local_root)


def execute_staged_wsi_task(
    payload: dict[str, Any],
    staged: StagedWSI,
    *,
    job_id: str,
    workspace_root: str | Path,
    progress: ProgressCallback,
) -> WSITaskOutput:
    workspace = Path(workspace_root)
    paths = planned_artifact_paths(payload, workspace)
    local_job_dir = resolve_path(
        payload.get("local_cache_dir") or os.environ.get("LOCAL_WSI_CACHE_DIR") or "local_wsi_cache",
        workspace,
    ) / "wsi_extract" / safe_path_part(job_id)
    local_job_dir.mkdir(parents=True, exist_ok=True)
    local_h5 = local_job_dir / "features.h5"
    local_overlay = local_job_dir / "overlay.png"
    local_thumbnail = local_job_dir / "thumbnail.jpg"
    local_tissue_mask = local_job_dir / "tissue_mask.png"
    local_qc_preview = local_job_dir / "qc_preview.jpg"
    stdout_log = local_job_dir / "extract_stdout.log"
    stderr_log = local_job_dir / "extract_stderr.log"

    if bool(payload.get("simulate", False)):
        progress(55, TASK_TOTAL_UNITS, "simulating UNI2-h extraction")
        write_simulated_h5(local_h5, payload, staged, job_id)
        if not bool(payload.get("no_overlay", False)):
            local_overlay.write_bytes(b"simulated overlay\n")
        local_thumbnail.write_bytes(b"simulated thumbnail\n")
        local_tissue_mask.write_bytes(b"simulated tissue mask\n")
        local_qc_preview.write_bytes(b"simulated qc preview\n")
        stdout_log.write_text(json.dumps({"simulated": True, "out_h5": str(local_h5)}), encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
    else:
        progress(40, TASK_TOTAL_UNITS, "starting UNI2-h extraction")
        run_extract_script(
            payload,
            staged.path,
            local_h5,
            local_overlay,
            local_thumbnail,
            local_tissue_mask,
            local_qc_preview,
            stdout_log,
            stderr_log,
            workspace,
            progress=progress,
        )

    validate_nonempty(local_h5, "local UNI2-h H5")
    progress(90, TASK_TOTAL_UNITS, "publishing artifacts")
    copy_atomically(local_h5, paths["features_h5"], job_id=job_id)
    overlay_published = False
    if local_overlay.exists() and not bool(payload.get("no_overlay", False)):
        copy_atomically(local_overlay, paths["overlay_png"], job_id=job_id)
        overlay_published = True
    thumbnail_published = publish_optional_artifact(local_thumbnail, paths["thumbnail_jpg"], job_id)
    tissue_mask_published = publish_optional_artifact(local_tissue_mask, paths["tissue_mask_png"], job_id)
    qc_preview_published = publish_optional_artifact(local_qc_preview, paths["qc_preview_jpg"], job_id)

    manifest = build_manifest(
        payload,
        staged,
        paths,
        stdout_log,
        stderr_log,
        overlay_published,
        thumbnail_published,
        tissue_mask_published,
        qc_preview_published,
    )
    write_json_atomically(paths["manifest_json"], manifest, job_id=job_id)
    progress(TASK_TOTAL_UNITS, TASK_TOTAL_UNITS, "wsi_uni2h complete")

    if bool(payload.get("cleanup_local_extract", True)):
        shutil.rmtree(local_job_dir, ignore_errors=True)

    return WSITaskOutput(
        str(paths["features_h5"]),
        {
            "task_type": "wsi_uni2h",
            "status": "completed",
            "manifest_path": str(paths["manifest_json"]),
            "overlay_path": str(paths["overlay_png"]) if overlay_published else None,
            "thumbnail_path": str(paths["thumbnail_jpg"]) if thumbnail_published else None,
            "tissue_mask_path": str(paths["tissue_mask_png"]) if tissue_mask_published else None,
            "qc_preview_path": str(paths["qc_preview_jpg"]) if qc_preview_published else None,
            "slide_dir": str(paths["slide_dir"]),
            "staged_bytes": staged.bytes,
        },
    )


def run_extract_script(
    payload: dict[str, Any],
    wsi_path: Path,
    out_h5: Path,
    overlay_path: Path,
    thumbnail_path: Path,
    tissue_mask_path: Path,
    qc_preview_path: Path,
    stdout_log: Path,
    stderr_log: Path,
    workspace: Path,
    *,
    progress: ProgressCallback,
) -> None:
    script = resolve_path(payload.get("extract_script", "scripts/extract_uni2h_features.py"), workspace)
    config = resolve_path(payload.get("config_path", "configs/uni2h_w8yi_style.yaml"), workspace)
    if not script.exists():
        raise FileNotFoundError(f"extract script not found: {script}")
    if not config.exists():
        raise FileNotFoundError(f"UNI2-h config not found: {config}")

    cmd = [
        sys.executable,
        str(script),
        "--wsi",
        str(wsi_path),
        "--out",
        str(out_h5),
        "--config",
        str(config),
        "--slide-id",
        slide_key_from_payload(payload),
    ]
    if payload.get("objective_power"):
        cmd.extend(["--objective-power", str(payload["objective_power"])])
    if payload.get("batch_size"):
        cmd.extend(["--batch-size", str(payload["batch_size"])])
    if payload.get("device"):
        cmd.extend(["--device", str(payload["device"])])
    if bool(payload.get("no_overlay", False)):
        cmd.append("--no-overlay")
    else:
        cmd.extend(["--overlay", str(overlay_path)])
    if not bool(payload.get("no_qc_images", False)):
        cmd.extend(["--thumbnail", str(thumbnail_path)])
        cmd.extend(["--tissue-mask", str(tissue_mask_path)])
        cmd.extend(["--qc-preview", str(qc_preview_path)])
    else:
        cmd.append("--no-qc-images")

    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(cmd, cwd=workspace, stdout=out, stderr=err, text=True)
        last_report = time.time()
        while process.poll() is None:
            now = time.time()
            if now - last_report >= float(payload.get("extract_progress_interval_seconds", 30)):
                progress(60, TASK_TOTAL_UNITS, "UNI2-h extraction running")
                last_report = now
            time.sleep(2.0)
        returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "message": "UNI2-h extractor failed",
                    "returncode": returncode,
                    "stdout_tail": read_tail(stdout_log),
                    "stderr_tail": read_tail(stderr_log),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def download_file(
    url: str,
    target_path: Path,
    payload: dict[str, Any],
    *,
    progress: Callable[[int, int | None, str], None],
    timeout_seconds: int,
    retries: int,
) -> None:
    expected_size = parse_optional_int(payload.get("expected_size_bytes"))
    expected_md5 = clean(payload.get("expected_md5")).lower()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(target_path.name + f".part.{os.getpid()}")
    last_error = ""

    for attempt in range(1, max(1, retries) + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            downloaded = 0
            request = urllib.request.Request(url, headers={"User-Agent": "path-rna-fusion-wsi-worker/0.1"})
            progress(0, expected_size, f"downloading WSI attempt {attempt}")
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response, tmp_path.open("wb") as handle:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    downloaded += len(block)
                    progress(downloaded, expected_size, f"downloading WSI {downloaded / 1024**3:.1f}GB")
            verify_expected_file(tmp_path, payload)
            os.replace(tmp_path, target_path)
            return
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            last_error = str(exc)
            if tmp_path.exists() and expected_md5:
                tmp_path.unlink()
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"WSI download failed after {retries} attempts: {last_error}")


def verify_expected_file(path: Path, payload: dict[str, Any], *, strict: bool = True) -> bool:
    if not path.exists():
        if strict:
            raise FileNotFoundError(path)
        return False
    expected_size = parse_optional_int(payload.get("expected_size_bytes"))
    if expected_size is not None and path.stat().st_size != expected_size:
        if strict:
            raise ValueError(f"size mismatch for {path}: expected {expected_size}, got {path.stat().st_size}")
        return False
    expected_md5 = clean(payload.get("expected_md5")).lower()
    if expected_md5:
        actual_md5 = file_md5(path)
        if actual_md5.lower() != expected_md5:
            if strict:
                raise ValueError(f"md5 mismatch for {path}: expected {expected_md5}, got {actual_md5}")
            return False
    return True


def file_md5(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
    return md5.hexdigest()


def copy_atomically(source: Path, final_path: Path, *, job_id: str) -> None:
    validate_nonempty(source, f"source artifact {source}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = final_path.with_name(final_path.name + f".tmp.{safe_path_part(job_id)}.{os.getpid()}")
    try:
        shutil.copyfile(source, tmp)
        validate_nonempty(tmp, f"temp artifact {tmp}")
        os.replace(tmp, final_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def publish_optional_artifact(source: Path, final_path: Path, job_id: str) -> bool:
    if not source.exists() or source.stat().st_size <= 0:
        return False
    copy_atomically(source, final_path, job_id=job_id)
    return True


def write_json_atomically(path: Path, data: dict[str, Any], *, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{safe_path_part(job_id)}.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        validate_nonempty(tmp, f"temp manifest {tmp}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_simulated_h5(path: Path, payload: dict[str, Any], staged: StagedWSI, job_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        json.dumps(
            {
                "simulated": True,
                "job_id": job_id,
                "wsi_path": str(staged.path),
                "dataset": clean(payload.get("dataset")),
                "wsi_id": clean(payload.get("wsi_id")),
            },
            sort_keys=True,
        ).encode("utf-8")
    )


def build_manifest(
    payload: dict[str, Any],
    staged: StagedWSI,
    paths: dict[str, Path],
    stdout_log: Path,
    stderr_log: Path,
    overlay_published: bool,
    thumbnail_published: bool,
    tissue_mask_published: bool,
    qc_preview_published: bool,
) -> dict[str, Any]:
    features = paths["features_h5"]
    overlay = paths["overlay_png"]
    thumbnail = paths["thumbnail_jpg"]
    tissue_mask = paths["tissue_mask_png"]
    qc_preview = paths["qc_preview_jpg"]
    return {
        "manifest_version": "wsi_uni2h_artifact_manifest_v0",
        "created_at": utc_now(),
        "task_type": "wsi_uni2h",
        "dataset": clean(payload.get("dataset")),
        "case_submitter_id": clean(payload.get("case_submitter_id")),
        "wsi_id": clean(payload.get("wsi_id")),
        "slide_key": slide_key_from_payload(payload),
        "source_adapter": clean(payload.get("source_adapter")),
        "source_file_id": clean(payload.get("file_id")),
        "expected_file_name": clean(payload.get("expected_file_name")),
        "config_path": clean(payload.get("config_path")) or "configs/uni2h_w8yi_style.yaml",
        "features_h5": str(features),
        "features_size_bytes": features.stat().st_size if features.exists() else None,
        "features_md5": file_md5(features) if features.exists() else "",
        "overlay_png": str(overlay) if overlay_published else "",
        "overlay_size_bytes": overlay.stat().st_size if overlay_published and overlay.exists() else None,
        "thumbnail_jpg": str(thumbnail) if thumbnail_published else "",
        "thumbnail_size_bytes": thumbnail.stat().st_size if thumbnail_published and thumbnail.exists() else None,
        "tissue_mask_png": str(tissue_mask) if tissue_mask_published else "",
        "tissue_mask_size_bytes": tissue_mask.stat().st_size if tissue_mask_published and tissue_mask.exists() else None,
        "qc_preview_jpg": str(qc_preview) if qc_preview_published else "",
        "qc_preview_size_bytes": qc_preview.stat().st_size if qc_preview_published and qc_preview.exists() else None,
        "staged_source": staged.source,
        "staged_bytes": staged.bytes,
        "extract_stdout_tail": read_tail(stdout_log),
        "extract_stderr_tail": read_tail(stderr_log),
    }


def cleanup_staged_wsi(staged: StagedWSI, payload: dict[str, Any]) -> None:
    if bool(payload.get("keep_raw_wsi_local", False)):
        return
    if staged.cleanup_root is None:
        return
    if staged.cleanup_root.exists():
        shutil.rmtree(staged.cleanup_root, ignore_errors=True)


def validate_nonempty(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.stat().st_size <= 0:
        raise RuntimeError(f"{label} is empty: {path}")


def read_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def slide_key_from_payload(payload: dict[str, Any]) -> str:
    return (
        clean(payload.get("slide_key"))
        or clean(payload.get("sample_or_slide_id"))
        or clean(payload.get("wsi_id"))
        or Path(clean(payload.get("expected_file_name")) or "unknown_slide").stem
    )


def resolve_path(path_value: object, workspace_root: str | Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return Path(workspace_root) / path
