from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from scripts import runpod_wsi


class RunPodWSIWrapperTests(unittest.TestCase):
    def test_load_settings_strips_quotes_and_uses_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / "run.env"
            env_path.write_text(
                "\n".join(
                    [
                        "DB_PATH='state.sqlite'",
                        'RUN_ID="run-a"',
                        "DATASETS=TCGA-UCEC",
                        "# ignored",
                    ]
                ),
                encoding="utf-8",
            )
            settings = runpod_wsi.load_settings(env_path)
            self.assertEqual(settings["DB_PATH"], "state.sqlite")
            self.assertEqual(settings["RUN_ID"], "run-a")
            self.assertEqual(settings["DATASETS"], "TCGA-UCEC")

    def test_doctor_reports_missing_required_values(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            env_path = Path(td) / "run.env"
            env_path.write_text("RUN_ID=run-a\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = runpod_wsi.main(["--env-file", str(env_path), "doctor"])
            self.assertEqual(code, 1)
            self.assertIn('"RUN_ID": true', output.getvalue())
            self.assertIn('"WORKER_TOKEN": false', output.getvalue())

    def test_enqueue_wrapper_accepts_minimal_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_dir = root / "plans"
            plan_dir.mkdir()
            plan_path = plan_dir / "download_plan_v0__raw_wsi__gdc__TCGA-TEST__main_strict__chunk-0001.csv"
            columns = [
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
            with plan_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "download_id": "raw_wsi:test-file",
                        "plan_version": "download_plan_v0",
                        "chunk_id": "chunk-1",
                        "asset_kind": "raw_wsi",
                        "source_adapter": "gdc",
                        "source_system": "GDC",
                        "dataset": "TCGA-TEST",
                        "cohort_role": "development",
                        "case_submitter_id": "CASE-1",
                        "sample_or_slide_id": "SLIDE-1",
                        "wsi_id": "TCGA-TEST:test-file",
                        "file_id": "test-file",
                        "expected_file_name": "slide.svs",
                        "expected_size_bytes": "123",
                        "target_rel_path": "data/raw/gdc/wsi/slide.svs",
                        "access": "open",
                        "qc_policy": "main_strict",
                        "priority": "50",
                        "status": "planned",
                        "reason": "test",
                    }
                )
            env_path = root / "run.env"
            db_path = root / "state.sqlite"
            env_path.write_text(
                "\n".join(
                    [
                        f"DB_PATH={db_path}",
                        "RUN_ID=run-a",
                        "WORKER_TOKEN=token",
                        f"PLAN_DIR={plan_dir}",
                        "DATASETS=TCGA-TEST",
                        "SOURCE_ADAPTER=gdc",
                        f"ARTIFACT_ROOT={root / 'artifacts'}",
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = runpod_wsi.main(["--env-file", str(env_path), "enqueue", "--simulate"])
            self.assertEqual(code, 0)
            self.assertIn('"enqueued": 1', output.getvalue())


if __name__ == "__main__":
    unittest.main()
