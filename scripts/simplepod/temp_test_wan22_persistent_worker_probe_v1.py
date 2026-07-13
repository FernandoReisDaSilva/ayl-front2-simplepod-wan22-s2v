import sys
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
    worker._restore_attention_patch = lambda: restored.__setitem__("called", True)
    worker.last_load_seconds = 12.345
    result = worker.unload()
    assert_true(result["status"] == "unloaded", "expected ready unload to succeed")
    assert_true(worker._pipeline is None, "expected pipeline cleared on unload")
    assert_true(restored["called"] is True, "expected SDPA restore callback on unload")
    assert_true(worker.worker_state == "unloaded", "expected worker state unloaded")
    assert_true(worker.last_load_seconds is None, "expected last_load_seconds cleared on unload")


def test_run_job_probe_only() -> None:
    from app.wan22_s2v_persistent_worker import (
        Wan22S2VPersistentWorker,
        Wan22S2VPersistentWorkerConfig,
    )

    worker = Wan22S2VPersistentWorker(Wan22S2VPersistentWorkerConfig(model_dir=Path("/tmp/no-model")))
    result = worker.run_job({"job_id": "probe_job"})
    assert_true(result["status"] == "not_implemented_probe_only", "expected probe-only run_job")
    assert_true(result["inference_executed"] is False, "expected run_job to avoid inference")
    assert_true(result["video_generated"] is False, "expected run_job to avoid video generation")


def test_endpoint_contracts_present() -> None:
    text = (APP_ROOT / "app" / "main.py").read_text(encoding="utf-8")
    for route in (
        '@app.post("/admin/run-mae-wan22-s2v-async"',
        '@app.get("/admin/jobs/{job_id}")',
        '@app.post("/jobs/wan22-s2v/run")',
        '@app.post("/admin/persistent-worker/load-probe")',
        '@app.get("/admin/persistent-worker/status")',
        '@app.post("/admin/persistent-worker/unload")',
    ):
        assert_true(route in text, f"missing endpoint route: {route}")


def main() -> int:
    test_effective_config()
    test_cleanup_and_recycle_policy()
    test_unload_state_guards()
    test_run_job_probe_only()
    test_endpoint_contracts_present()
    print("persistent worker probe tests OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
