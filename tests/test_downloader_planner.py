import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.downloader.download import DownloadConfig, run_download_plan
from scripts.downloader.planner import PlanBuildConfig, build_and_write_plans, build_plan_rows, parse_size


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class DownloaderPlannerTests(unittest.TestCase):
    def test_parse_size(self) -> None:
        self.assertEqual(parse_size("1KB"), 1024)
        self.assertEqual(parse_size("1.5GB"), int(1.5 * 1024**3))

    def test_builds_chunked_wsi_and_rna_plans(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wsi = root / "wsi.csv"
            rna = root / "rna.csv"
            out = root / "plans"
            write_csv(
                wsi,
                [
                    {
                        "wsi_id": "w1",
                        "dataset": "TCGA-UCEC",
                        "cohort_role": "development",
                        "source_system": "GDC",
                        "case_submitter_id": "case-1",
                        "case_uuid": "case-uuid-1",
                        "slide_id": "slide-1",
                        "sample_submitter_id": "sample-1",
                        "slide_file_id": "file-w1",
                        "expected_svs_filename": "w1.svs",
                        "slide_file_name": "w1.svs",
                        "gdc_link": "",
                        "file_size_bytes": str(10 * 1024**3),
                        "md5sum": "abc",
                        "access": "open",
                        "main_strict_pre_qc_v0": "yes",
                        "inclusive_tumor_pre_qc_v0": "yes",
                        "single_best_pre_qc_v0": "yes",
                        "in_scope_for_main": "yes",
                    },
                    {
                        "wsi_id": "w2",
                        "dataset": "TCGA-UCEC",
                        "cohort_role": "development",
                        "source_system": "GDC",
                        "case_submitter_id": "case-2",
                        "case_uuid": "case-uuid-2",
                        "slide_id": "slide-2",
                        "sample_submitter_id": "sample-2",
                        "slide_file_id": "file-w2",
                        "expected_svs_filename": "w2.svs",
                        "slide_file_name": "w2.svs",
                        "gdc_link": "",
                        "file_size_bytes": str(10 * 1024**3),
                        "md5sum": "def",
                        "access": "open",
                        "main_strict_pre_qc_v0": "yes",
                        "inclusive_tumor_pre_qc_v0": "yes",
                        "single_best_pre_qc_v0": "no",
                        "in_scope_for_main": "yes",
                    },
                ],
            )
            write_csv(
                rna,
                [
                    {
                        "rna_id": "r1",
                        "dataset": "TCGA-UCEC",
                        "cohort_role": "development",
                        "source_system": "GDC",
                        "case_submitter_id": "case-1",
                        "case_uuid": "case-uuid-1",
                        "sample_submitter_id": "sample-1",
                        "file_id": "file-r1",
                        "file_name": "r1.tsv",
                        "file_size_bytes": "100",
                        "md5sum": "xyz",
                        "access": "open",
                        "representative_rna_v0": "yes",
                        "rna_metadata_qc_pass_v0": "yes",
                        "in_scope_for_main": "yes",
                    }
                ],
            )
            config = PlanBuildConfig(
                wsi_table=wsi,
                rna_table=rna,
                out_dir=out,
                datasets=("TCGA-UCEC",),
                max_files_per_chunk=1,
                max_bytes_per_chunk=50 * 1024**3,
            )
            result = build_and_write_plans(config)
            self.assertEqual(result["rows"], 3)
            self.assertEqual(result["chunks"], 3)
            index = out / "download_plan_v0_index.csv"
            self.assertTrue(index.exists())
            with index.open(newline="", encoding="utf-8") as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual(len(index_rows), 3)

    def test_single_best_policy_filters_wsi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wsi = root / "wsi.csv"
            rna = root / "rna.csv"
            write_csv(
                wsi,
                [
                    {
                        "wsi_id": "w1",
                        "dataset": "D",
                        "cohort_role": "external",
                        "source_system": "PathDB/TCIA",
                        "case_submitter_id": "c1",
                        "case_uuid": "",
                        "slide_id": "s1",
                        "sample_submitter_id": "",
                        "slide_file_id": "f1",
                        "expected_svs_filename": "s1.svs",
                        "slide_file_name": "s1.svs",
                        "gdc_link": "",
                        "file_size_bytes": "",
                        "md5sum": "",
                        "access": "open",
                        "main_strict_pre_qc_v0": "yes",
                        "inclusive_tumor_pre_qc_v0": "yes",
                        "single_best_pre_qc_v0": "yes",
                        "in_scope_for_main": "yes",
                    },
                    {
                        "wsi_id": "w2",
                        "dataset": "D",
                        "cohort_role": "external",
                        "source_system": "PathDB/TCIA",
                        "case_submitter_id": "c1",
                        "case_uuid": "",
                        "slide_id": "s2",
                        "sample_submitter_id": "",
                        "slide_file_id": "f2",
                        "expected_svs_filename": "s2.svs",
                        "slide_file_name": "s2.svs",
                        "gdc_link": "",
                        "file_size_bytes": "",
                        "md5sum": "",
                        "access": "open",
                        "main_strict_pre_qc_v0": "yes",
                        "inclusive_tumor_pre_qc_v0": "yes",
                        "single_best_pre_qc_v0": "no",
                        "in_scope_for_main": "yes",
                    },
                ],
            )
            write_csv(
                rna,
                [
                    {
                        "rna_id": "r1",
                        "dataset": "D",
                        "cohort_role": "external",
                        "source_system": "GDC",
                        "case_submitter_id": "c1",
                        "case_uuid": "",
                        "sample_submitter_id": "sample",
                        "file_id": "rf1",
                        "file_name": "r.tsv",
                        "file_size_bytes": "1",
                        "md5sum": "",
                        "access": "open",
                        "representative_rna_v0": "yes",
                        "rna_metadata_qc_pass_v0": "yes",
                        "in_scope_for_main": "yes",
                    }
                ],
            )
            config = PlanBuildConfig(
                assets=("raw_wsi",),
                wsi_table=wsi,
                rna_table=rna,
                wsi_qc_policy="single_best",
            )
            rows = build_plan_rows(config)
            self.assertEqual([row["wsi_id"] for row in rows], ["w1"])
            self.assertEqual(rows[0]["source_adapter"], "tcia_pathdb")

    def test_hf_index_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hf_index = root / "hf.json"
            hf_index.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "repo_id": "owner/repo",
                                "repo_revision": "abc123",
                                "repo_file_path": "TCGA-UCEC/slide.pt",
                                "dataset": "TCGA-UCEC",
                                "case_submitter_id": "case-1",
                                "slide_id": "slide-1",
                                "expected_size_bytes": "123",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config = PlanBuildConfig(assets=("hf_uni2h_embedding",), hf_index=hf_index)
            rows = build_plan_rows(config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_adapter"], "huggingface")
            self.assertEqual(rows[0]["repo_revision"], "abc123")

    def test_cli_estimate_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wsi = root / "wsi.csv"
            write_csv(
                wsi,
                [
                    {
                        "wsi_id": "w1",
                        "dataset": "TCGA-UCEC",
                        "cohort_role": "development",
                        "source_system": "GDC",
                        "case_submitter_id": "case-1",
                        "case_uuid": "case-uuid-1",
                        "slide_id": "slide-1",
                        "sample_submitter_id": "sample-1",
                        "slide_file_id": "file-w1",
                        "expected_svs_filename": "w1.svs",
                        "slide_file_name": "w1.svs",
                        "gdc_link": "",
                        "file_size_bytes": "100",
                        "md5sum": "abc",
                        "access": "open",
                        "main_strict_pre_qc_v0": "yes",
                        "inclusive_tumor_pre_qc_v0": "yes",
                        "single_best_pre_qc_v0": "yes",
                        "in_scope_for_main": "yes",
                    }
                ],
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.downloader.cli",
                    "estimate",
                    "--assets",
                    "raw_wsi",
                    "--datasets",
                    "TCGA-UCEC",
                    "--wsi-table",
                    str(wsi),
                    "--out-dir",
                    str(root / "plans"),
                    "--max-files-per-chunk",
                    "100",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        payload = json.loads(result.stdout)
        self.assertIn("rows", payload)
        self.assertIn("chunks", payload)

    def test_verify_existing_file_from_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data" / "raw" / "gdc" / "rna.tsv"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"gene_id\tunstranded\ttpm_unstranded\nENSG000001\t1\t1000000\n")
            plan = root / "plan.csv"
            write_csv(
                plan,
                [
                    {
                        "download_id": "rna:1",
                        "plan_version": "download_plan_v0",
                        "chunk_id": "chunk",
                        "asset_kind": "rna_star_counts",
                        "source_adapter": "gdc",
                        "source_system": "GDC",
                        "dataset": "D",
                        "cohort_role": "development",
                        "case_submitter_id": "C",
                        "case_uuid": "",
                        "sample_or_slide_id": "S",
                        "rna_id": "R",
                        "wsi_id": "",
                        "file_id": "file-id",
                        "repo_id": "",
                        "repo_revision": "",
                        "repo_file_path": "",
                        "remote_url": "",
                        "expected_file_name": "rna.tsv",
                        "expected_size_bytes": str(target.stat().st_size),
                        "expected_md5": "338a1a73c68d163e1f88d180c76dd6fa",
                        "target_rel_path": str(target.relative_to(root)).replace("\\", "/"),
                        "access": "open",
                        "qc_policy": "representative",
                        "priority": "40",
                        "status": "planned",
                        "reason": "fixture",
                        "source_table": "fixture",
                        "source_row_id": "R",
                    }
                ],
            )
            inventory = root / "inventory.csv"
            result = run_download_plan(
                DownloadConfig(plan_file=plan, output_root=root, inventory_path=inventory, verify_only=True, concurrency=1)
            )
            self.assertEqual(result["inventory"]["by_verification_status"], {"pass": 1})
            with inventory.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["download_status"], "verified")
            self.assertEqual(rows[0]["verification_status"], "pass")


if __name__ == "__main__":
    unittest.main()
