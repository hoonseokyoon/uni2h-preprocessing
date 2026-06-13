from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts import runpod_wsi
from scripts.distributed import cli
from scripts.distributed.artifact_validation import validate_wsi_artifacts
from scripts.distributed.cost import CostManager
from scripts.distributed.export_artifacts import ExportConfig, export_artifacts
from scripts.distributed.models import JobState, PodRole, WorkerState
from scripts.distributed.runpod_client import RunPodClient, WorkerPodSpec
from scripts.distributed.store import SQLiteStore
from scripts.distributed.tasks import run_demo_file_task
from scripts.distributed.wsi_preprocess import run_wsi_uni2h_task


class DistributedFrameworkTests(unittest.TestCase):
    def test_job_lifecycle_claim_progress_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            job_id = store.enqueue_job(
                "mock",
                {"total_units": 3},
                run_id="run-a",
                total_units=3,
                max_attempts=2,
                now=101.0,
            )

            claimed = store.claim_job("worker-1", run_id="run-a", lease_seconds=30, now=102.0)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.id, job_id)
            self.assertEqual(claimed.state, JobState.RUNNING)
            self.assertEqual(claimed.attempt, 1)

            progressed = store.report_progress(
                "worker-1",
                job_id,
                completed_units=2,
                total_units=3,
                message="two units",
                now=103.0,
            )
            self.assertEqual(progressed.completed_units, 2)

            completed = store.complete_job("worker-1", job_id, result={"ok": True}, now=104.0)
            self.assertEqual(completed.state, JobState.COMPLETED)
            self.assertEqual(completed.completed_units, 3)
            self.assertEqual(store.get_worker("worker-1").current_job_id, None)
            events = store.list_events(job_id=job_id)
            self.assertEqual([event.event_type for event in events], ["queued", "claimed", "progress", "completed"])
            latest = store.latest_progress_by_job([job_id])[job_id]
            self.assertEqual(latest.message, "two units")

    def test_lease_timeout_recovery_reclaims_retryable_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            store.register_worker(worker_id="worker-2", run_id="run-a", now=100.0)
            job_id = store.enqueue_job(
                "mock",
                {"total_units": 1},
                run_id="run-a",
                total_units=1,
                max_attempts=2,
                now=100.0,
            )
            store.claim_job("worker-1", run_id="run-a", lease_seconds=5, now=101.0)

            result = store.recover_stale_jobs(now=107.0, worker_timeout_seconds=999)
            self.assertEqual(result["recovered_jobs"], 1)
            self.assertEqual(store.get_job(job_id).state, JobState.RETRYABLE)

            reclaimed = store.claim_job("worker-2", run_id="run-a", lease_seconds=5, now=108.0)
            self.assertIsNotNone(reclaimed)
            self.assertEqual(reclaimed.id, job_id)
            self.assertEqual(reclaimed.attempt, 2)
            self.assertEqual(reclaimed.worker_id, "worker-2")

    def test_worker_drain_finishes_current_job_then_acks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            first_id = store.enqueue_job("mock", {}, run_id="run-a", total_units=1, now=101.0)
            second_id = store.enqueue_job("mock", {}, run_id="run-a", total_units=1, now=102.0)
            store.claim_job("worker-1", run_id="run-a", lease_seconds=60, now=103.0)

            draining = store.request_worker_drain("worker-1", now=104.0)
            self.assertEqual(draining.state, WorkerState.DRAINING)
            self.assertEqual(store.get_worker_command("worker-1"), "drain")
            self.assertIsNone(store.claim_job("worker-1", run_id="run-a", lease_seconds=60, now=105.0))

            store.complete_job("worker-1", first_id, result={}, now=106.0)
            drained = store.ack_worker_drained("worker-1", now=107.0)
            self.assertEqual(drained.state, WorkerState.DRAINED)
            self.assertEqual(store.get_job(second_id).state, JobState.QUEUED)

    def test_pause_resume_run_blocks_new_claims(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            job_id = store.enqueue_job("mock", {}, run_id="run-a", total_units=1, now=101.0)

            store.set_run_paused("run-a", True, now=102.0)
            self.assertIsNone(store.claim_job("worker-1", run_id="run-a", now=103.0))
            self.assertEqual(store.get_job(job_id).state, JobState.QUEUED)

            store.set_run_paused("run-a", False, now=104.0)
            claimed = store.claim_job("worker-1", run_id="run-a", now=105.0)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed.id, job_id)

    def test_batch_claim_and_release_unstarted_job(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            job_ids = [
                store.enqueue_job(
                    "wsi_uni2h",
                    {"expected_size_bytes": "10"},
                    run_id="run-a",
                    total_units=100,
                    max_attempts=2,
                    now=101.0 + index,
                )
                for index in range(3)
            ]

            claimed = store.claim_jobs(
                "worker-1",
                run_id="run-a",
                task_type="wsi_uni2h",
                max_jobs=2,
                lease_seconds=60,
                now=110.0,
            )
            self.assertEqual([job.id for job in claimed], job_ids[:2])
            self.assertEqual(store.get_job(job_ids[0]).state, JobState.RUNNING)
            self.assertEqual(store.get_job(job_ids[1]).state, JobState.RUNNING)
            self.assertEqual(store.get_job(job_ids[2]).state, JobState.QUEUED)

            released = store.release_claimed_job("worker-1", job_ids[1], reason="drain", now=111.0)
            self.assertEqual(released.state, JobState.QUEUED)
            self.assertEqual(released.attempt, 0)
            self.assertEqual(store.running_job_count_for_worker("worker-1"), 1)

    def test_cost_burn_eta_and_cap_logic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self.make_store_from_path(Path(td) / "state.sqlite")
            store.upsert_pod(
                pod_id="server-pod",
                role=PodRole.SERVER,
                provider_status="running",
                cost_per_hr=1.0,
                start_time=0.0,
                now=0.0,
            )
            store.upsert_pod(
                pod_id="worker-pod",
                role=PodRole.WORKER,
                provider_status="running",
                cost_per_hr=2.0,
                adjusted_cost_per_hr=2.5,
                start_time=0.0,
                worker_id="worker-1",
                now=0.0,
            )
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            first_id = store.enqueue_job("mock", {}, run_id="run-a", total_units=10, now=100.0)
            store.enqueue_job("mock", {}, run_id="run-a", total_units=10, now=100.0)
            store.claim_job("worker-1", run_id="run-a", lease_seconds=60, now=100.0)
            store.complete_job("worker-1", first_id, result={}, now=200.0)
            store.set_cost_cap(3.5, hard_cap=False)

            manager = CostManager(store)
            summary = manager.summarize(now=300.0, additional_hourly_rate=0.6)
            self.assertAlmostEqual(summary.current_burn_rate_per_hr, 3.5)
            self.assertAlmostEqual(summary.spent_so_far, 3.5 * (300.0 / 3600.0))
            self.assertAlmostEqual(summary.eta_seconds or 0, 200.0)
            self.assertFalse(summary.scale_up_allowed)
            self.assertIn("exceeds cap", summary.scale_up_block_reason or "")

    def test_runpod_worker_payload_generation(self) -> None:
        spec = WorkerPodSpec(
            name="worker-a",
            image_name="registry/image:tag",
            server_pod_id="srv123",
            server_port=8080,
            run_id="run-a",
            worker_token="token",
            worker_role="wsi-preprocess",
            workspace_root="/workspace",
            gpu_type_ids=["NVIDIA RTX A4000"],
            network_volume_id="vol123",
            data_center_ids=["US-KS-2"],
        )
        payload = RunPodClient.create_worker_payload(spec)
        self.assertEqual(payload["name"], "worker-a")
        self.assertEqual(payload["networkVolumeId"], "vol123")
        self.assertEqual(payload["dataCenterIds"], ["US-KS-2"])
        self.assertTrue(payload["globalNetworking"])
        self.assertEqual(payload["env"]["SERVER_URL"], "http://srv123.runpod.internal:8080")
        self.assertEqual(payload["env"]["RUN_ID"], "run-a")
        self.assertEqual(payload["env"]["WORKER_TOKEN"], "token")

    def test_add_worker_injects_stable_worker_id(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "state.sqlite")
            self.assertEqual(cli.main(["--db", db, "init-db"]), 0)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(
                    [
                        "--db",
                        db,
                        "add-worker",
                        "--name",
                        "worker-a",
                        "--image-name",
                        "registry/image:tag",
                        "--server-pod-id",
                        "srv123",
                        "--gpu-type-id",
                        "NVIDIA RTX A4000",
                        "--hourly-cost",
                        "1.0",
                        "--worker-token",
                        "token",
                        "--worker-id",
                        "worker-fixed",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertIn('"WORKER_ID": "worker-fixed"', output.getvalue())

    def test_cli_status_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "state.sqlite")
            self.assertEqual(cli.main(["--db", db, "init-db"]), 0)
            self.assertEqual(
                cli.main(["--db", db, "enqueue-demo-jobs", "--count", "1", "--output-dir", str(Path(td) / "out")]),
                0,
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = cli.main(["--db", db, "status", "--plain"])
            self.assertEqual(code, 0)
            text = output.getvalue()
            self.assertIn("Distributed Execution Status", text)
            self.assertIn("queued\t1", text)

    def test_cli_events_and_inspect_job_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / "state.sqlite")
            store = self.make_store_from_path(Path(db))
            store.register_worker(worker_id="worker-1", run_id="run-a", now=100.0)
            job_id = store.enqueue_job("mock", {"total_units": 1}, run_id="run-a", total_units=1, now=101.0)
            store.claim_job("worker-1", run_id="run-a", now=102.0)
            store.report_progress("worker-1", job_id, completed_units=1, total_units=1, message="done", now=103.0)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--db", db, "events", "--job-id", job_id]), 0)
            self.assertIn("progress", output.getvalue())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--db", db, "inspect-job", job_id]), 0)
            data = json.loads(output.getvalue())
            self.assertEqual(data["job"]["id"], job_id)
            self.assertEqual(data["latest_progress"]["message"], "done")

    def test_demo_file_task_atomic_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            progress_calls: list[tuple[float, float, str | None]] = []
            result = run_demo_file_task(
                {"output_path": "outputs/demo.txt", "work_units": 3, "content": "abc"},
                job_id="job-1",
                workspace_root=td,
                progress=lambda completed, total, message: progress_calls.append((completed, total, message)),
            )
            output_path = Path(result.output_path or "")
            self.assertTrue(output_path.exists())
            self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(len(progress_calls), 3)
            self.assertEqual(list(output_path.parent.glob("*.tmp.*")), [])

    def test_wsi_uni2h_simulated_task_publishes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "raw" / "slide.svs"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"fake svs bytes")
            md5 = hashlib.md5(raw.read_bytes()).hexdigest()
            progress_calls: list[tuple[float, float, str | None]] = []
            result = run_wsi_uni2h_task(
                {
                    "raw_wsi_path": str(raw),
                    "expected_file_name": raw.name,
                    "expected_size_bytes": str(raw.stat().st_size),
                    "expected_md5": md5,
                    "artifact_root": str(root / "shared" / "wsi_uni2h_v0"),
                    "local_cache_dir": str(root / "scratch"),
                    "dataset": "TCGA-TEST",
                    "case_submitter_id": "CASE-1",
                    "wsi_id": "TCGA-TEST:slide-1",
                    "simulate": True,
                },
                job_id="job-1",
                workspace_root=root,
                progress=lambda completed, total, message: progress_calls.append((completed, total, message)),
            )
            output_path = Path(result.output_path or "")
            manifest_path = output_path.with_name("manifest.json")
            overlay_path = output_path.with_name("overlay.png")
            thumbnail_path = output_path.with_name("thumbnail.jpg")
            tissue_mask_path = output_path.with_name("tissue_mask.png")
            qc_preview_path = output_path.with_name("qc_preview.jpg")
            stdout_log_path = output_path.with_name("extract_stdout.log")
            stderr_log_path = output_path.with_name("extract_stderr.log")
            self.assertTrue(output_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(overlay_path.exists())
            self.assertTrue(thumbnail_path.exists())
            self.assertTrue(tissue_mask_path.exists())
            self.assertTrue(qc_preview_path.exists())
            self.assertTrue(stdout_log_path.exists())
            self.assertTrue(stderr_log_path.exists())
            self.assertEqual(list(output_path.parent.glob("*.tmp.*")), [])
            self.assertGreaterEqual(progress_calls[-1][0], 100)
            validation = validate_wsi_artifacts(root / "shared" / "wsi_uni2h_v0", simulate_ok=True)
            self.assertTrue(validation["passed"], validation)

    def test_validate_artifacts_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            slide_dir = Path(td) / "artifacts" / "DATASET" / "CASE" / "SLIDE"
            slide_dir.mkdir(parents=True)
            (slide_dir / "manifest.json").write_text("{}", encoding="utf-8")
            report = validate_wsi_artifacts(slide_dir, simulate_ok=True)
            self.assertFalse(report["passed"])
            self.assertTrue(report["slides"][0]["issues"])

    def test_doctor_deep_helper_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            env = {
                "DB_PATH": str(root / "state.sqlite"),
                "ARTIFACT_ROOT": str(root / "artifacts"),
                "LOCAL_WSI_CACHE_DIR": str(root / "cache"),
                "PLAN_DIR": str(root / "plans"),
                "EXTRACT_SCRIPT": "scripts/extract_uni2h_features.py",
                "UNI2H_CONFIG_PATH": "configs/uni2h_w8yi_style.yaml",
            }
            report = runpod_wsi.run_deep_checks(env)
            self.assertEqual(report["db_parent"]["status"], "ok")
            self.assertEqual(report["artifact_root"]["status"], "ok")
            self.assertIn(report["imports"]["status"], {"ok", "warn"})

    def test_export_artifacts_dry_run_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "artifacts"
            slide_dir = root / "TCGA-TEST" / "CASE-1" / "SLIDE-1"
            slide_dir.mkdir(parents=True)
            for name in ("features.h5", "overlay.png", "thumbnail.jpg", "tissue_mask.png", "qc_preview.jpg", "manifest.json"):
                (slide_dir / name).write_text(name, encoding="utf-8")
            inventory = Path(td) / "export_inventory.csv"
            result = export_artifacts(
                ExportConfig(
                    artifact_root=root,
                    destination="s3://bucket/prefix",
                    inventory_path=inventory,
                    dry_run=True,
                )
            )
            self.assertEqual(result["files"], 6)
            text = inventory.read_text(encoding="utf-8")
            self.assertIn("s3://bucket/prefix/TCGA-TEST/CASE-1/SLIDE-1/features.h5", text)
            self.assertIn("planned", text)

    def make_store_from_path(self, db_path: Path) -> SQLiteStore:
        store = SQLiteStore(db_path)
        store.initialize()
        return store


if __name__ == "__main__":
    unittest.main()
