"""CLI for controlled download planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .download import DownloadConfig, run_download_plan
from .planner import (
    DEFAULT_OUTPUT_DIR,
    PlanBuildConfig,
    build_and_write_plans,
    build_plan_rows,
    chunk_rows,
    parse_assets,
    parse_datasets,
    parse_size,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and inspect chunked download plans.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-plan", help="Create chunked download plan CSV files.")
    add_plan_args(build)
    build.add_argument("--dry-run", action="store_true", help="Print summary without writing plan files.")

    estimate = subparsers.add_parser("estimate", help="Estimate rows/chunks without writing plan files.")
    add_plan_args(estimate)

    run = subparsers.add_parser("run-plan", help="Download files from a chunked plan.")
    add_download_args(run)

    verify = subparsers.add_parser("verify-plan", help="Verify existing files from a chunked plan without downloading.")
    add_download_args(verify)

    return parser


def add_plan_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--assets",
        default="raw_wsi,rna_star_counts",
        help="Comma-separated asset kinds: raw_wsi,rna_star_counts,hf_uni2h_embedding.",
    )
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter, e.g. CPTAC-UCEC,TCGA-UCEC.")
    parser.add_argument(
        "--wsi-qc-policy",
        default="main_strict",
        choices=["main_strict", "inclusive", "single_best", "all_in_scope"],
    )
    parser.add_argument(
        "--rna-policy",
        default="representative",
        choices=["representative", "metadata_pass", "all_in_scope"],
    )
    parser.add_argument("--wsi-table", type=Path, default=Path("manifests") / "wsi_slide_pre_qc_table_v0.csv")
    parser.add_argument("--rna-table", type=Path, default=Path("manifests") / "rna_qc_table_v0.csv")
    parser.add_argument("--hf-index", type=Path, default=None, help="Optional CSV/JSON index for HuggingFace embedding files.")
    parser.add_argument("--hf-source-name", default="hf_uni2h", help="Name used under data/external_embeddings.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-files-per-chunk", type=int, default=250)
    parser.add_argument("--max-bytes-per-chunk", default="50GB")


def add_download_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan-dir", type=Path)
    source.add_argument("--plan-file", type=Path)
    source.add_argument("--index-file", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--inventory-path", type=Path, default=Path("manifests") / "download_inventory_v0.csv")
    parser.add_argument("--asset-kind", default="", help="Optional asset filter, e.g. rna_star_counts.")
    parser.add_argument("--source-adapter", default="", help="Optional adapter filter, e.g. gdc.")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset filter.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-bytes", default="", help="Optional selected-byte limit, e.g. 2GB.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)


def config_from_args(args: argparse.Namespace) -> PlanBuildConfig:
    if args.max_files_per_chunk <= 0:
        raise ValueError("--max-files-per-chunk must be positive")
    return PlanBuildConfig(
        assets=parse_assets(args.assets),
        datasets=parse_datasets(args.datasets),
        wsi_qc_policy=args.wsi_qc_policy,
        rna_policy=args.rna_policy,
        max_files_per_chunk=args.max_files_per_chunk,
        max_bytes_per_chunk=parse_size(args.max_bytes_per_chunk),
        out_dir=args.out_dir,
        data_root=args.data_root,
        wsi_table=args.wsi_table,
        rna_table=args.rna_table,
        hf_index=args.hf_index,
        hf_source_name=args.hf_source_name,
    )


def download_config_from_args(args: argparse.Namespace, *, verify_only: bool) -> DownloadConfig:
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    max_bytes = parse_size(args.max_bytes) if args.max_bytes else None
    return DownloadConfig(
        plan_dir=args.plan_dir,
        plan_file=args.plan_file,
        index_file=args.index_file,
        output_root=args.output_root,
        inventory_path=args.inventory_path,
        asset_kind=args.asset_kind,
        source_adapter=args.source_adapter,
        datasets=parse_datasets(args.datasets),
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        max_files=args.max_files,
        max_bytes=max_bytes,
        verify_only=verify_only,
        resume=not args.no_resume,
        overwrite=args.overwrite,
        progress_interval_seconds=args.progress_interval_seconds,
    )


def estimate_only(config: PlanBuildConfig) -> dict[str, object]:
    rows = build_plan_rows(config)
    chunks = chunk_rows(rows, config)
    known_bytes = 0
    unknown = 0
    for row in rows:
        text = row.get("expected_size_bytes", "")
        if not text:
            unknown += 1
            continue
        try:
            known_bytes += int(float(text))
        except ValueError:
            unknown += 1
    return {
        "rows": len(rows),
        "chunks": len(chunks),
        "known_bytes": known_bytes,
        "unknown_size_count": unknown,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in {"run-plan", "verify-plan"}:
        config = download_config_from_args(args, verify_only=args.command == "verify-plan")
        result = run_download_plan(config)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    config = config_from_args(args)
    if args.command == "estimate" or getattr(args, "dry_run", False):
        summary = estimate_only(config)
        if args.command == "build-plan":
            summary["dry_run"] = True
            summary["out_dir"] = str(config.out_dir)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    result = build_and_write_plans(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
