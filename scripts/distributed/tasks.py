"""Demo and utility tasks for the worker runtime."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[float, float, str | None], None]


@dataclass(frozen=True)
class TaskResult:
    output_path: str | None
    metadata: dict[str, Any]


def run_task(
    task_type: str,
    payload: dict[str, Any],
    *,
    job_id: str,
    workspace_root: str | Path,
    progress: ProgressCallback,
) -> TaskResult:
    if task_type == "demo_file":
        return run_demo_file_task(payload, job_id=job_id, workspace_root=workspace_root, progress=progress)
    if task_type == "shell":
        return run_shell_task(payload, job_id=job_id, workspace_root=workspace_root, progress=progress)
    if task_type == "mock":
        total = float(payload.get("total_units", 1))
        progress(total, total, "mock complete")
        return TaskResult(None, {"task_type": "mock"})
    if task_type == "wsi_uni2h":
        from .wsi_preprocess import run_wsi_uni2h_task

        result = run_wsi_uni2h_task(payload, job_id=job_id, workspace_root=workspace_root, progress=progress)
        return TaskResult(result.output_path, result.metadata)
    raise ValueError(f"unknown task type: {task_type}")


def run_demo_file_task(
    payload: dict[str, Any],
    *,
    job_id: str,
    workspace_root: str | Path,
    progress: ProgressCallback,
) -> TaskResult:
    """Write a deterministic artifact using temp-file validation and rename."""

    workspace = Path(workspace_root)
    output_path = _resolve_output_path(payload, workspace, job_id)
    units = int(payload.get("work_units", payload.get("total_units", 5)))
    units = max(1, units)
    delay_seconds = float(payload.get("delay_seconds", 0))
    content = str(payload.get("content", f"demo output for {job_id}"))
    lines: list[str] = []
    for index in range(units):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        line = f"{job_id}\t{index + 1}\t{content}"
        lines.append(line)
        progress(index + 1, units, f"demo unit {index + 1}/{units}")
    atomic_write_text(
        output_path,
        "\n".join(lines) + "\n",
        validator=lambda path: path.exists() and path.stat().st_size > 0,
        temp_suffix=f".tmp.{job_id}.{os.getpid()}",
    )
    return TaskResult(
        str(output_path),
        {
            "task_type": "demo_file",
            "line_count": units,
            "bytes": output_path.stat().st_size,
        },
    )


def run_shell_task(
    payload: dict[str, Any],
    *,
    job_id: str,
    workspace_root: str | Path,
    progress: ProgressCallback,
) -> TaskResult:
    """Run a local shell command and optionally persist stdout atomically."""

    command = payload.get("command")
    if not command:
        raise ValueError("shell task requires payload.command")
    workspace = Path(workspace_root)
    cwd = Path(payload.get("cwd", workspace))
    progress(0, 1, "shell command starting")
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=isinstance(command, str),
        capture_output=True,
        text=True,
        check=False,
    )
    output_path: Path | None = None
    if payload.get("output_path"):
        output_path = _safe_workspace_path(workspace, payload["output_path"])
        atomic_write_text(
            output_path,
            completed.stdout,
            validator=lambda path: path.exists(),
            temp_suffix=f".tmp.{job_id}.{os.getpid()}",
        )
    progress(1, 1, "shell command finished")
    if completed.returncode != 0:
        raise RuntimeError(
            json.dumps(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-4000:],
                    "stderr": completed.stderr[-4000:],
                },
                sort_keys=True,
            )
        )
    return TaskResult(
        str(output_path) if output_path else None,
        {
            "task_type": "shell",
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        },
    )


def atomic_write_text(
    final_path: str | Path,
    text: str,
    *,
    validator: Callable[[Path], bool],
    temp_suffix: str,
) -> None:
    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(final.name + temp_suffix)
    try:
        tmp.write_text(text, encoding="utf-8")
        if not validator(tmp):
            raise RuntimeError(f"validation failed for temp artifact: {tmp}")
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def _resolve_output_path(payload: dict[str, Any], workspace: Path, job_id: str) -> Path:
    output_path = payload.get("output_path")
    if output_path:
        return _safe_workspace_path(workspace, str(output_path))
    return workspace / "outputs" / f"{job_id}.txt"


def _safe_workspace_path(workspace: Path, user_path: str) -> Path:
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    workspace_resolved = workspace.resolve()
    candidate_parent = candidate.parent.resolve()
    if workspace_resolved != candidate_parent and workspace_resolved not in candidate_parent.parents:
        raise ValueError(f"output path escapes workspace root: {candidate}")
    return candidate
