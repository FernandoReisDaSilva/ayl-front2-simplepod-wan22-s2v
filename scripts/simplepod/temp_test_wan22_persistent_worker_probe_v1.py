import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "docker" / "simplepod-wan22-s2v-fastapi-v2-blackwell"


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_effective_config() -> None:
    sys.path.insert(0, str(APP_ROOT))
    from app.wan22_s2v_persistent_worker import (
        Wan22S2VPersistentWorker,
        Wan22S2VPersistentWorkerConfig,
    )

    config = Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model"))
    worker = Wan22S2VPersistentWorker(config)
    result = worker.validate_effective_config()
    assert_true(result["status"] == "passed", "expected default persistent worker config to pass")
    assert_true(result["effective"]["t5_cpu"] is False, "expected t5_cpu=False")
    assert_true(result["effective"]["offload_model"] is True, "expected offload_model=True")
    assert_true(result["effective"]["convert_model_dtype"] is True, "expected convert_model_dtype=True")
    assert_true(result["effective"]["max_concurrent_jobs"] == 1, "expected MAX_CONCURRENT_JOBS=1")

    bad = Wan22S2VPersistentWorker(
        Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model"), t5_cpu=True)
    )
    bad_result = bad.validate_effective_config()
    assert_true(bad_result["status"] == "failed", "expected t5_cpu=True config to fail")
    assert_true("t5_cpu" in bad_result["mismatches"], "expected t5_cpu mismatch")


def test_cleanup_and_recycle_policy() -> None:
    from app.wan22_s2v_persistent_worker import (
        Wan22S2VPersistentWorker,
        Wan22S2VPersistentWorkerConfig,
    )

    worker = Wan22S2VPersistentWorker(Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model")))
    worker.worker_state = "ready"
    status = worker.status()
    assert_true("last_load_seconds" in status, "expected last_load_seconds in status")
    assert_true("resident_vram_allocated_gb" in status, "expected resident_vram_allocated_gb in status")
    assert_true("resident_vram_reserved_gb" in status, "expected resident_vram_reserved_gb in status")
    worker.worker_baseline_memory = {
        "memory_allocated_gib": 10.0,
        "memory_reserved_gib": 20.0,
        "free_margin_gib": 70.0,
    }
    status = worker.status()
    assert_true(status["resident_vram_allocated_gb"] == 10.0, "expected resident allocated VRAM alias")
    assert_true(status["resident_vram_reserved_gb"] == 20.0, "expected resident reserved VRAM alias")
    temporary_objects = [object(), object()]
    cleanup = worker.cleanup_after_job(temporary_objects)
    assert_true(cleanup["status"] == "succeeded", "expected cleanup to succeed")
    assert_true(temporary_objects == [], "expected cleanup to clear temporary object list")
    assert_true(cleanup["ipc_collect_used"] is False, "expected no ipc_collect in normal cleanup")

    worker.last_job_memory = {
        "residual_growth_vs_worker_baseline_gib": 2.1,
        "memory_after_cleanup": {"free_margin_gib": 70.0},
    }
    decision = worker.should_recycle()
    assert_true(decision["should_recycle"] is True, "expected recycle for residual growth >=2GB")
    assert_true("residual_growth_recycle" in decision["reasons"], "expected residual growth reason")

    worker.last_job_memory = {
        "residual_growth_vs_worker_baseline_gib": 0.0,
        "memory_after_cleanup": {"free_margin_gib": 7.9},
    }
    decision = worker.should_recycle()
    assert_true(decision["should_recycle"] is True, "expected recycle for low free margin")
    assert_true("minimum_free_margin_below_threshold" in decision["reasons"], "expected free margin reason")


def test_unload_state_guards() -> None:
    from app.wan22_s2v_persistent_worker import (
        Wan22S2VPersistentWorker,
        Wan22S2VPersistentWorkerConfig,
    )

    for state in ("loading", "running"):
        worker = Wan22S2VPersistentWorker(Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model")))
        worker.worker_state = state
        worker._pipeline = object()
        result = worker.unload()
        assert_true(result["status"] == "busy", f"expected busy unload for state={state}")
        assert_true(result["worker_state"] == state, f"expected worker_state preserved for state={state}")
        assert_true(result["unload_executed"] is False, f"expected unload_executed=false for state={state}")
        assert_true(worker._pipeline is not None, f"expected pipeline preserved for state={state}")

    worker = Wan22S2VPersistentWorker(Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model")))
    restored = {"called": False}
    worker.worker_state = "ready"
    worker._pipeline = object()
    import app.wan22_s2v_persistent_worker as worker_module

    def restore_attention():
        restored["called"] = True
        worker_module.RUNTIME_PATCH_REPORT["attention_sdpa_patch"] = {
            "attention_patch_status": "restored",
        }

    worker._restore_attention_patch = restore_attention
    worker.last_load_seconds = 12.345
    result = worker.unload()
    assert_true(result["status"] == "unloaded", "expected ready unload to succeed")
    assert_true(worker._pipeline is None, "expected pipeline cleared on unload")
    assert_true(restored["called"] is True, "expected SDPA restore callback on unload")
    assert_true(worker.worker_state == "unloaded", "expected worker state unloaded")
    assert_true(worker.last_load_seconds is None, "expected last_load_seconds cleared on unload")
    assert_true(
        result["attention_sdpa_patch"]["attention_patch_status"] == "restored",
        "expected unload result to expose restored attention patch status",
    )


def test_run_job_two_jobs_simulated() -> None:
    import app.wan22_s2v_persistent_worker as worker_module
    from app.wan22_s2v_persistent_worker import (
        Wan22S2VPersistentWorker,
        Wan22S2VPersistentWorkerConfig,
    )

    class FakeVideo:
        def __getitem__(self, item):
            return self

    class FakePipeline:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return FakeVideo()

    uploads = []
    original_download_file = worker_module.download_file
    original_upload_file = worker_module.upload_file
    original_save_video = worker_module._save_video_and_merge_audio
    original_work_root = worker_module.WORK_ROOT

    def fake_download_file(key, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"input")

    def fake_upload_file(source, key):
        uploads.append({"source": str(source), "key": key})

    def fake_save_video(video, output_path, audio_path, fps):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"mp4")
        return worker_module._file_facts(Path(output_path))

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            worker_module.WORK_ROOT = Path(tmpdir)
            worker_module.download_file = fake_download_file
            worker_module.upload_file = fake_upload_file
            worker_module._save_video_and_merge_audio = fake_save_video

            pipeline = FakePipeline()
            worker = Wan22S2VPersistentWorker(Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model")))
            worker.worker_state = "ready"
            worker._pipeline = pipeline
            worker.load_count = 1
            worker.t5_cpu_effective = False
            worker.offload_model_effective = True
            worker.convert_model_dtype_effective = True
            worker.worker_baseline_memory = {
                "memory_allocated_gib": 10.0,
                "memory_reserved_gib": 20.0,
                "free_margin_gib": 70.0,
            }

            base_payload = {
                "character_id": "mae",
                "base_taught_language": "FR",
                "reference_image_key": "tests/input/reference.png",
                "audio_key": "tests/input/audio.wav",
                "target_width": 720,
                "target_height": 720,
                "fps": 16,
                "target_duration_seconds": 15.0,
                "confirm_inference": "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL",
                "allow_oom_fallback": False,
                "seed": 42,
                "steps": 5,
                "cfg": 1.0,
                "shift": 4.0,
                "offload_model": True,
            }
            first = worker.run_job(
                {
                    **base_payload,
                    "job_id": "persistent_job_1",
                    "output_video_key": "tests/output/job1.mp4",
                    "output_report_key": "tests/output/job1_report.json",
                }
            )
            second = worker.run_job(
                {
                    **base_payload,
                    "job_id": "persistent_job_2",
                    "output_video_key": "tests/output/job2.mp4",
                    "output_report_key": "tests/output/job2_report.json",
                }
            )
            assert_true(first["status"] == "succeeded", "expected first persistent job to succeed")
            assert_true(second["status"] == "succeeded", "expected second persistent job to succeed")
            assert_true(first["uses_subprocess"] is False, "expected no subprocess for persistent job")
            assert_true(second["uses_subprocess"] is False, "expected no subprocess for persistent job")
            assert_true(first["load_count_before"] == 1 and first["load_count_after"] == 1, "expected first job not to reload")
            assert_true(second["load_count_before"] == 1 and second["load_count_after"] == 1, "expected second job not to reload")
            assert_true(first["jobs_completed"] == 1, "expected jobs_completed=1 after first job")
            assert_true(second["jobs_completed"] == 2, "expected jobs_completed=2 after second job")
            assert_true(len(pipeline.calls) == 2, "expected two generate calls on the same pipeline object")
            assert_true(pipeline.calls[0]["input_prompt"], "expected resolved Maé prompt")
            assert_true(pipeline.calls[0]["max_area"] == 720 * 720, "expected 720x720 max_area")
            assert_true(first["video_generated"] is True and second["video_generated"] is True, "expected MP4 generation")
            assert_true(worker.worker_state == "ready", "expected worker ready after sequential jobs")
            assert_true(len(uploads) == 4, "expected two video uploads and two report uploads")
            assert_true(first["t5_cpu_effective"] is False, "expected t5_cpu false")
    finally:
        worker_module.download_file = original_download_file
        worker_module.upload_file = original_upload_file
        worker_module._save_video_and_merge_audio = original_save_video
        worker_module.WORK_ROOT = original_work_root


def test_endpoint_contracts_present() -> None:
    text = (APP_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    for route in (
        '@app.post("/admin/run-mae-wan22-s2v-async"',
        '@app.get("/admin/jobs/{job_id}")',
        '@app.post("/jobs/wan22-s2v/run")',
        '@app.post("/admin/persistent-worker/load-probe")',
        '@app.get("/admin/persistent-worker/status")',
        '@app.post("/admin/persistent-worker/run-job")',
        '@app.post("/admin/persistent-worker/jobs")',
        '@app.get("/admin/persistent-worker/jobs/{job_id}")',
        '@app.post("/admin/persistent-worker/unload")',
    ):
        assert_true(route in text, f"missing endpoint route: {route}")


def test_batch_async_polling_helpers() -> None:
    scripts_root = REPO_ROOT / "scripts" / "simplepod"
    sys.path.insert(0, str(scripts_root))
    import temp_simplepod_persistent_worker_batch_probe_v1 as batch

    class Args:
        job_timeout_seconds = 60
        job_poll_interval_seconds = 1

    submit_calls = []
    poll_calls = []
    original_simple_post = batch.base.simple_post
    original_simple_get = batch.base.simple_get
    original_sleep = batch.time.sleep

    def fake_simple_post(url, payload, timeout_seconds):
        submit_calls.append({"url": url, "payload": payload, "timeout_seconds": timeout_seconds})
        return {
            "status": "succeeded",
            "http_status_code": 202,
            "json": {"status": "queued", "job_id": payload["job_id"]},
        }

    poll_sequence = [
        {"status": "failed", "error_type": "URLError", "error_truncated": "temporary poll failure"},
        {"status": "succeeded", "http_status_code": 200, "json": {"job_id": "job1", "status": "queued"}},
        {"status": "succeeded", "http_status_code": 200, "json": {"job_id": "job1", "status": "running"}},
        {
            "status": "succeeded",
            "http_status_code": 200,
            "json": {
                "job_id": "job1",
                "status": "succeeded",
                "started_at": "start",
                "completed_at": "done",
                "runtime_seconds": 10,
                "generation_seconds": 8,
                "peak_vram_gb": 70,
                "worker_state": "ready",
                "jobs_completed": 1,
                "load_count": 1,
                "recycle_required": False,
            },
        },
    ]

    def fake_simple_get(url, timeout_seconds):
        poll_calls.append({"url": url, "timeout_seconds": timeout_seconds})
        return poll_sequence.pop(0)

    try:
        batch.base.simple_post = fake_simple_post
        batch.base.simple_get = fake_simple_get
        batch.time.sleep = lambda _seconds: None
        submission = batch.submit_persistent_worker_job("http://pod", {"job_id": "job1"})
        assert_true(submission["result"]["http_status_code"] == 202, "expected async submission HTTP 202")
        assert_true(len(submit_calls) == 1, "expected job submitted exactly once")

        poll_result = batch.poll_persistent_worker_job("http://pod", "job1", Args())
        assert_true(poll_result["status"] == "succeeded", "expected queued->running->succeeded polling")
        assert_true(poll_result["json"]["status"] == "succeeded", "expected terminal succeeded")
        assert_true(poll_result["poll_count"] == 4, "expected one failed poll plus three status polls")
        assert_true(len(poll_result["poll_failures"]) == 1, "expected one recorded poll failure")
        assert_true(poll_result["recovered_after_poll_failure"] is True, "expected recovery after poll failure")
        assert_true(len(submit_calls) == 1, "expected polling not to resubmit job")
    finally:
        batch.base.simple_post = original_simple_post
        batch.base.simple_get = original_simple_get
        batch.time.sleep = original_sleep


def main() -> int:
    test_effective_config()
    test_cleanup_and_recycle_policy()
    test_unload_state_guards()
    test_run_job_two_jobs_simulated()
    test_endpoint_contracts_present()
    test_batch_async_polling_helpers()
    print("persistent worker probe tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
