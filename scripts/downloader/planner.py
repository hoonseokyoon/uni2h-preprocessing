"""Build chunked download plan tables.

The planner is intentionally source-agnostic. Each output row records enough
metadata for a later adapter to download and verify the file, while chunk files
keep large runs manageable.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PLAN_VERSION = "download_plan_v0"
DEFAULT_OUTPUT_DIR = Path("manifests") / "download_plans_v0"

PLAN_COLUMNS = [
    "download_id",
    "plan_version",
    "chunk_id",
    "asset_kind",
    "source_adapter",
    "source_system",
    "dataset",
    "cohort_role",
    "case_submitter_id",
    "case_uuid",
    "sample_or_slide_id",
    "rna_id",
    "wsi_id",
    "file_id",
    "repo_id",
    "repo_revision",
    "repo_file_path",
    "remote_url",
    "expected_file_name",
    "expected_size_bytes",
    "expected_md5",
    "target_rel_path",
    "access",
    "qc_policy",
    "priority",
    "status",
    "reason",
    "source_table",
    "source_row_id",
]

INDEX_COLUMNS = [
    "chunk_id",
    "plan_path",
    "asset_kind",
    "source_adapter",
    "dataset",
    "qc_policy",
    "row_count",
    "known_bytes",
    "unknown_size_count",
    "first_download_id",
    "last_download_id",
]


@dataclass(frozen=True)
class PlanBuildConfig:
    assets: tuple[str, ...] = ("raw_wsi", "rna_star_counts")
    datasets: tuple[str, ...] = ()
    wsi_qc_policy: str = "main_strict"
    rna_policy: str = "representative"
    max_files_per_chunk: int = 250
    max_bytes_per_chunk: int = 50 * 1024**3
    out_dir: Path = DEFAULT_OUTPUT_DIR
    data_root: str = "data"
    wsi_table: Path = Path("manifests") / "wsi_slide_pre_qc_table_v0.csv"
    rna_table: Path = Path("manifests") / "rna_qc_table_v0.csv"
    hf_index: Path | None = None
    hf_source_name: str = "hf_uni2h"


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
        raise FileNotFoundError(f"Input table not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in fieldnames})


def parse_assets(text: str) -> tuple[str, ...]:
    assets = tuple(item.strip() for item in text.split(",") if item.strip())
    valid = {"raw_wsi", "rna_star_counts", "hf_uni2h_embedding"}
    invalid = sorted(set(assets) - valid)
    if invalid:
        raise ValueError(f"Unsupported asset kind(s): {invalid}. Supported: {sorted(valid)}")
    return assets


def parse_datasets(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_size(text: str) -> int:
    value = text.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?B?)?", value, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid byte size: {text}")
    number = float(match.group(1))
    unit = match.group(2).upper()
    multipliers = {
        "": 1,
        "B": 1,
        "K": 1024,
        "KB": 1024,
        "M": 1024**2,
        "MB": 1024**2,
        "G": 1024**3,
        "GB": 1024**3,
        "T": 1024**4,
        "TB": 1024**4,
    }
    return int(number * multipliers[unit])


def safe_path_part(text: str) -> str:
    value = clean(text)
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value) if value else "unknown"


def source_adapter_for_wsi(row: dict[str, str]) -> str:
    source = clean(row.get("source_system"))
    if source == "GDC":
        return "gdc"
    if "TCIA" in source or "PathDB" in source:
        return "tcia_pathdb"
    return "unknown"


def should_keep_dataset(row: dict[str, str], datasets: tuple[str, ...]) -> bool:
    return not datasets or clean(row.get("dataset")) in datasets


def wsi_policy_pass(row: dict[str, str], policy: str) -> bool:
    if policy == "main_strict":
        return clean(row.get("main_strict_pre_qc_v0")) == "yes"
    if policy == "inclusive":
        return clean(row.get("inclusive_tumor_pre_qc_v0")) == "yes"
    if policy == "single_best":
        return clean(row.get("single_best_pre_qc_v0")) == "yes"
    if policy == "all_in_scope":
        return clean(row.get("in_scope_for_main")) == "yes"
    raise ValueError(f"Unsupported WSI QC policy: {policy}")


def rna_policy_pass(row: dict[str, str], policy: str) -> bool:
    if policy == "representative":
        return clean(row.get("representative_rna_v0")) == "yes"
    if policy == "metadata_pass":
        return clean(row.get("rna_metadata_qc_pass_v0")) == "yes"
    if policy == "all_in_scope":
        return clean(row.get("in_scope_for_main")) == "yes"
    raise ValueError(f"Unsupported RNA policy: {policy}")


def build_raw_wsi_rows(config: PlanBuildConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(config.wsi_table):
        if not should_keep_dataset(row, config.datasets) or not wsi_policy_pass(row, config.wsi_qc_policy):
            continue
        dataset = clean(row.get("dataset"))
        case_id = clean(row.get("case_submitter_id"))
        file_name = clean(row.get("expected_svs_filename")) or clean(row.get("slide_file_name"))
        source_adapter = source_adapter_for_wsi(row)
        file_id = clean(row.get("slide_file_id"))
        target = Path(config.data_root) / "raw" / source_adapter / "wsi" / dataset / case_id / file_name
        download_id = f"raw_wsi:{clean(row.get('wsi_id'))}"
        rows.append(
            {
                "download_id": download_id,
                "plan_version": PLAN_VERSION,
                "asset_kind": "raw_wsi",
                "source_adapter": source_adapter,
                "source_system": clean(row.get("source_system")),
                "dataset": dataset,
                "cohort_role": clean(row.get("cohort_role")),
                "case_submitter_id": case_id,
                "case_uuid": clean(row.get("case_uuid")),
                "sample_or_slide_id": clean(row.get("slide_id")) or clean(row.get("sample_submitter_id")),
                "rna_id": "",
                "wsi_id": clean(row.get("wsi_id")),
                "file_id": file_id,
                "repo_id": "",
                "repo_revision": "",
                "repo_file_path": "",
                "remote_url": clean(row.get("gdc_link")),
                "expected_file_name": file_name,
                "expected_size_bytes": clean(row.get("file_size_bytes")),
                "expected_md5": clean(row.get("md5sum")),
                "target_rel_path": target.as_posix(),
                "access": clean(row.get("access")) or "open",
                "qc_policy": config.wsi_qc_policy,
                "priority": "50",
                "status": "planned",
                "reason": "planned_from_wsi_pre_qc_table",
                "source_table": str(config.wsi_table),
                "source_row_id": clean(row.get("wsi_id")),
            }
        )
    return rows


def build_rna_rows(config: PlanBuildConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(config.rna_table):
        if not should_keep_dataset(row, config.datasets) or not rna_policy_pass(row, config.rna_policy):
            continue
        dataset = clean(row.get("dataset"))
        case_id = clean(row.get("case_submitter_id"))
        file_name = clean(row.get("file_name"))
        target = Path(config.data_root) / "raw" / "gdc" / "rna_star_counts" / dataset / case_id / file_name
        download_id = f"rna_star_counts:{clean(row.get('rna_id'))}"
        rows.append(
            {
                "download_id": download_id,
                "plan_version": PLAN_VERSION,
                "asset_kind": "rna_star_counts",
                "source_adapter": "gdc",
                "source_system": clean(row.get("source_system")) or "GDC",
                "dataset": dataset,
                "cohort_role": clean(row.get("cohort_role")),
                "case_submitter_id": case_id,
                "case_uuid": clean(row.get("case_uuid")),
                "sample_or_slide_id": clean(row.get("sample_submitter_id")),
                "rna_id": clean(row.get("rna_id")),
                "wsi_id": "",
                "file_id": clean(row.get("file_id")),
                "repo_id": "",
                "repo_revision": "",
                "repo_file_path": "",
                "remote_url": "",
                "expected_file_name": file_name,
                "expected_size_bytes": clean(row.get("file_size_bytes")),
                "expected_md5": clean(row.get("md5sum")),
                "target_rel_path": target.as_posix(),
                "access": clean(row.get("access")) or "open",
                "qc_policy": config.rna_policy,
                "priority": "40",
                "status": "planned",
                "reason": "planned_from_rna_qc_table",
                "source_table": str(config.rna_table),
                "source_row_id": clean(row.get("rna_id")),
            }
        )
    return rows


def read_hf_index(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("files", [])
        if not isinstance(data, list):
            raise ValueError(f"HF index JSON must contain a list or a files list: {path}")
        return [{str(k): clean(v) for k, v in item.items()} for item in data]
    return read_csv(path)


def build_hf_embedding_rows(config: PlanBuildConfig) -> list[dict[str, str]]:
    if config.hf_index is None:
        return []
    rows: list[dict[str, str]] = []
    for row in read_hf_index(config.hf_index):
        if not should_keep_dataset(row, config.datasets):
            continue
        dataset = clean(row.get("dataset"))
        case_id = clean(row.get("case_submitter_id"))
        slide_id = clean(row.get("slide_id")) or clean(row.get("sample_or_slide_id"))
        repo_id = clean(row.get("repo_id"))
        revision = clean(row.get("repo_revision")) or clean(row.get("revision"))
        repo_file_path = clean(row.get("repo_file_path")) or clean(row.get("path"))
        file_name = clean(row.get("expected_file_name")) or Path(repo_file_path).name
        target = (
            Path(config.data_root)
            / "external_embeddings"
            / config.hf_source_name
            / dataset
            / safe_path_part(case_id or slide_id)
            / file_name
        )
        source_row_id = clean(row.get("download_id")) or f"{repo_id}:{revision}:{repo_file_path}"
        rows.append(
            {
                "download_id": f"hf_uni2h_embedding:{dataset}:{source_row_id}",
                "plan_version": PLAN_VERSION,
                "asset_kind": "hf_uni2h_embedding",
                "source_adapter": "huggingface",
                "source_system": "HuggingFace",
                "dataset": dataset,
                "cohort_role": clean(row.get("cohort_role")),
                "case_submitter_id": case_id,
                "case_uuid": clean(row.get("case_uuid")),
                "sample_or_slide_id": slide_id,
                "rna_id": "",
                "wsi_id": clean(row.get("wsi_id")),
                "file_id": clean(row.get("file_id")),
                "repo_id": repo_id,
                "repo_revision": revision,
                "repo_file_path": repo_file_path,
                "remote_url": clean(row.get("remote_url")),
                "expected_file_name": file_name,
                "expected_size_bytes": clean(row.get("expected_size_bytes")) or clean(row.get("size")),
                "expected_md5": clean(row.get("expected_md5")) or clean(row.get("md5")),
                "target_rel_path": target.as_posix(),
                "access": clean(row.get("access")) or "open_or_gated",
                "qc_policy": "hf_index",
                "priority": clean(row.get("priority")) or "60",
                "status": "planned",
                "reason": "planned_from_hf_index",
                "source_table": str(config.hf_index),
                "source_row_id": source_row_id,
            }
        )
    return rows


def build_plan_rows(config: PlanBuildConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if "raw_wsi" in config.assets:
        rows.extend(build_raw_wsi_rows(config))
    if "rna_star_counts" in config.assets:
        rows.extend(build_rna_rows(config))
    if "hf_uni2h_embedding" in config.assets:
        rows.extend(build_hf_embedding_rows(config))
    rows.sort(key=lambda item: (item["asset_kind"], item["source_adapter"], item["dataset"], item["case_submitter_id"], item["download_id"]))
    return rows


def row_size(row: dict[str, str]) -> int | None:
    text = clean(row.get("expected_size_bytes"))
    if not text:
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return value if value >= 0 else None


def chunk_rows(rows: list[dict[str, str]], config: PlanBuildConfig) -> list[list[dict[str, str]]]:
    chunks: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_bytes = 0
    current_key: tuple[str, str, str, str] | None = None

    for row in rows:
        key = (row["asset_kind"], row["source_adapter"], row["dataset"], row["qc_policy"])
        size = row_size(row)
        should_split = False
        if current and key != current_key:
            should_split = True
        if current and len(current) >= config.max_files_per_chunk:
            should_split = True
        if current and size is not None and current_bytes + size > config.max_bytes_per_chunk:
            should_split = True
        if should_split:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_key = key
        if size is not None:
            current_bytes += size

    if current:
        chunks.append(current)
    return chunks


def chunk_id_for(rows: list[dict[str, str]], ordinal: int) -> str:
    first = rows[0]
    return "__".join(
        [
            PLAN_VERSION,
            safe_path_part(first["asset_kind"]),
            safe_path_part(first["source_adapter"]),
            safe_path_part(first["dataset"]),
            safe_path_part(first["qc_policy"]),
            f"chunk-{ordinal:04d}",
        ]
    )


def summarize_chunk(chunk_id: str, path: Path, rows: list[dict[str, str]]) -> dict[str, str]:
    known_bytes = 0
    unknown = 0
    for row in rows:
        size = row_size(row)
        if size is None:
            unknown += 1
        else:
            known_bytes += size
    first = rows[0]
    return {
        "chunk_id": chunk_id,
        "plan_path": path.as_posix(),
        "asset_kind": first["asset_kind"],
        "source_adapter": first["source_adapter"],
        "dataset": first["dataset"],
        "qc_policy": first["qc_policy"],
        "row_count": str(len(rows)),
        "known_bytes": str(known_bytes),
        "unknown_size_count": str(unknown),
        "first_download_id": rows[0]["download_id"],
        "last_download_id": rows[-1]["download_id"],
    }


def build_and_write_plans(config: PlanBuildConfig) -> dict[str, object]:
    rows = build_plan_rows(config)
    chunks = chunk_rows(rows, config)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    for ordinal, chunk in enumerate(chunks, start=1):
        chunk_id = chunk_id_for(chunk, ordinal)
        for row in chunk:
            row["chunk_id"] = chunk_id
        path = config.out_dir / f"{chunk_id}.csv"
        write_csv(path, chunk, PLAN_COLUMNS)
        index_rows.append(summarize_chunk(chunk_id, path, chunk))

    index_path = config.out_dir / f"{PLAN_VERSION}_index.csv"
    write_csv(index_path, index_rows, INDEX_COLUMNS)
    summary = build_summary(config, rows, index_rows)
    summary_path = config.out_dir / f"{PLAN_VERSION}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "rows": len(rows),
        "chunks": len(index_rows),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def build_summary(config: PlanBuildConfig, rows: list[dict[str, str]], index_rows: list[dict[str, str]]) -> dict[str, object]:
    by_asset_dataset: dict[str, dict[str, int]] = {}
    total_known_bytes = 0
    total_unknown = 0
    for row in rows:
        key = f"{row['asset_kind']}|{row['source_adapter']}|{row['dataset']}"
        item = by_asset_dataset.setdefault(key, {"rows": 0, "known_bytes": 0, "unknown_size_count": 0})
        item["rows"] += 1
        size = row_size(row)
        if size is None:
            item["unknown_size_count"] += 1
            total_unknown += 1
        else:
            item["known_bytes"] += size
            total_known_bytes += size
    return {
        "plan_version": PLAN_VERSION,
        "created_at": utc_now(),
        "assets": list(config.assets),
        "datasets": list(config.datasets),
        "wsi_qc_policy": config.wsi_qc_policy,
        "rna_policy": config.rna_policy,
        "max_files_per_chunk": config.max_files_per_chunk,
        "max_bytes_per_chunk": config.max_bytes_per_chunk,
        "total_rows": len(rows),
        "total_chunks": len(index_rows),
        "total_known_bytes": total_known_bytes,
        "total_unknown_size_count": total_unknown,
        "by_asset_adapter_dataset": by_asset_dataset,
    }
