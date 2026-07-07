import gc
import inspect
import uuid
import importlib
import importlib.metadata
import os
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .r2_client import r2_env_alias_presence, r2_env_ready
from .reporting import file_facts, now_iso, stub_final_report
from .settings import SERVICE_NAME, SERVICE_VERSION, get_settings, is_secret_key
from .wan22_s2v_runner import run_wan22_s2v_single_job


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

REQUIRED_JOB_FIELDS = ("reference_image_key", "audio_key")
OPTIONAL_JOB_FIELDS = (
    "output_video_key",
    "final_report_key",
    "positive_prompt",
    "negative_prompt",
    "prompt",
    "seed",
    "steps",
    "cfg",
    "shift",
    "denoise_strength",
    "audio_scale",
    "pose_start_percent",
    "pose_end_percent",
    "num_frames",
    "duration_seconds",
)
WAN22_S2V_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
DOWNLOAD_CONFIRMATION = "DOWNLOAD_WAN22_S2V_WEIGHTS"
INFERENCE_CONFIRMATIONS = {
    "RUN_WAN22_S2V_MAE_14_8S_1080",
    "RUN_WAN22_S2V_MAE_14_8S_1080_BLACKWELL",
    "RUN_WAN22_S2V_MAE_14_8S_1080_BLACKWELL_NATURAL_V5",
    "RUN_WAN22_S2V_MAE_14_8S_1080_BLACKWELL_NATURAL_V5_NATIVE_PARTIAL",
    "RUN_WAN22_S2V_MAE_14_8S_720_BLACKWELL_NATURAL_V5_NATIVE_PARTIAL",
}
ADMIN_DOWNLOAD_ENV = "AYL_ENABLE_ADMIN_DOWNLOADS"
ADMIN_VERIFY_ENV = "AYL_ENABLE_ADMIN_VERIFY"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 7200
OUTPUT_TRUNCATE_CHARS = 4000
WAN22_REPO_DIR = Path(os.getenv("WAN22_REPO_DIR", "/opt/Wan2.2"))
RUN_JOB_REQUIRED_FIELDS = (
    "job_id",
    "character_id",
    "base_taught_language",
    "reference_image_key",
    "audio_key",
    "target_width",
    "target_height",
    "fps",
    "target_duration_seconds",
    "output_video_key",
    "output_report_key",
    "confirm_inference",
    "allow_oom_fallback",
)
RUN_JOB_OPTIONAL_FIELDS = (
    "positive_prompt",
    "negative_prompt",
    "prompt",
    "seed",
    "steps",
    "cfg",
    "shift",
    "denoise_strength",
    "audio_scale",
    "pose_start_percent",
    "pose_end_percent",
    "num_frames",
    "timeout_seconds",
)
ASYNC_JOBS_LOCK = threading.Lock()
ASYNC_JOBS: dict[str, dict[str, Any]] = {}
ASYNC_RUNNING_JOB_IDS: set[str] = set()
RELEVANT_PACKAGES = (
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "dashscope",
    "diffusers",
    "accelerate",
    "opencv-python-headless",
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "Pillow",
    "regex",
    "tqdm",
    "scipy",
    "safetensors",
    "librosa",
    "decord",
    "moviepy",
    "omegaconf",
    "peft",
    "easydict",
    "einops",
    "ftfy",
)
WAN_CODE_IMPORT_ATTEMPTED_MODULES = (
    "wan",
    "wan.configs",
    "wan.modules",
    "generate",
)
WAN_MODEL_S2V_CLASS_CANDIDATES = (
    ("wan.modules.s2v.model_s2v", "WanModel_S2V"),
)


def torch_probe() -> dict:
    result = {
        "torch_import_status": "not_attempted",
        "torch_version": "",
        "torch_cuda_version": "",
        "cuda_available": False,
        "device_name": "",
        "device_capability": None,
        "device_capability_major": None,
        "device_capability_minor": None,
        "device_capability_string": "",
        "vram_total_gb": None,
        "error_truncated": "",
    }
    try:
        import torch

        result["torch_import_status"] = "ok"
        result["torch_version"] = getattr(torch, "__version__", "") or ""
        result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", "") or ""
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["device_name"] = torch.cuda.get_device_name(0)
            capability = torch.cuda.get_device_capability(0)
            result["device_capability"] = list(capability)
            result["device_capability_major"] = int(capability[0])
            result["device_capability_minor"] = int(capability[1])
            result["device_capability_string"] = f"{capability[0]}.{capability[1]}"
            props = torch.cuda.get_device_properties(0)
            result["vram_total_gb"] = round(float(props.total_memory) / (1024**3), 2)
    except Exception as exc:
        result["torch_import_status"] = "failed"
        result["error_truncated"] = str(exc)[:1000]
    return result


def redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            redacted[key] = "<redacted>" if is_secret_key(str(key)) else redact_payload(item)
        return redacted
    if isinstance(value, list):
        return [redact_payload(item) for item in value]
    return value


def directory_inventory(path: Path) -> dict:
    safetensors = []
    marker_names = {
        "config.json",
        "generation_config.json",
        "model_index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "scheduler_config.json",
        "preprocessor_config.json",
    }
    marker_files = []
    inventory = {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "recursive_file_count": 0,
        "recursive_total_size_bytes": 0,
        "recursive_total_size_gb": 0.0,
        "file_count": 0,
        "total_bytes": 0,
        "total_gb": 0.0,
        "sample_files": [],
        "safetensors_files": [],
        "marker_files": [],
    }
    if not path.exists() or not path.is_dir():
        return inventory

    sample_files = []
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            size = item.stat().st_size
        except OSError:
            continue
        relative = str(item.relative_to(path))
        inventory["recursive_file_count"] += 1
        inventory["recursive_total_size_bytes"] += size
        if len(sample_files) < 20:
            sample_files.append(relative)
        if item.suffix == ".safetensors":
            safetensors.append({"path": relative, "size_bytes": size, "size_gb": round(size / (1024**3), 3)})
        if item.name in marker_names:
            marker_files.append({"path": relative, "size_bytes": size})
    safetensors.sort(key=lambda file: file["size_bytes"], reverse=True)
    marker_files.sort(key=lambda file: file["path"])
    inventory["recursive_total_size_gb"] = round(inventory["recursive_total_size_bytes"] / (1024**3), 3)
    inventory["file_count"] = inventory["recursive_file_count"]
    inventory["total_bytes"] = inventory["recursive_total_size_bytes"]
    inventory["total_gb"] = inventory["recursive_total_size_gb"]
    inventory["sample_files"] = sample_files
    inventory["safetensors_files"] = safetensors[:30]
    inventory["marker_files"] = marker_files
    return inventory


def safe_import_module(module_name: str, extra_path: Path | None = None) -> dict:
    try:
        if extra_path is not None and str(extra_path) not in sys.path:
            sys.path.insert(0, str(extra_path))
        module = importlib.import_module(module_name)
        return {
            "status": "ok",
            "module": module_name,
            "module_file": str(getattr(module, "__file__", "") or ""),
        }
    except Exception as exc:
        traceback_lines = traceback.format_exc().splitlines()
        return {
            "status": "failed",
            "module": module_name,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "traceback_tail": traceback_lines[-12:],
        }


def installed_packages_relevant() -> dict:
    packages = {}
    for package_name in RELEVANT_PACKAGES:
        try:
            packages[package_name] = {
                "status": "installed",
                "version": importlib.metadata.version(package_name),
            }
        except importlib.metadata.PackageNotFoundError:
            packages[package_name] = {
                "status": "missing",
                "version": "",
            }
        except Exception as exc:
            packages[package_name] = {
                "status": "error",
                "version": "",
                "error_type": type(exc).__name__,
                "error_truncated": str(exc)[:500],
            }
    return packages


def runtime_import_checks() -> dict:
    wan_code_imports = {
        module_name: safe_import_module(module_name, WAN22_REPO_DIR)
        for module_name in WAN_CODE_IMPORT_ATTEMPTED_MODULES
    }
    runner_imports = {
        "app.wan22_s2v_runner": safe_import_module("app.wan22_s2v_runner"),
        "app.wan22_s2v_generate_wrapper": safe_import_module("app.wan22_s2v_generate_wrapper"),
    }
    wan_code_ok = any(item["status"] == "ok" for item in wan_code_imports.values())
    runner_ok = all(item["status"] == "ok" for item in runner_imports.values())
    first_wan_failure = next(
        (item for item in wan_code_imports.values() if item.get("status") == "failed"),
        {},
    )
    return {
        "wan_code_import_status": "ok" if wan_code_ok else "failed",
        "wan_code_import_error_type": "" if wan_code_ok else first_wan_failure.get("error_type", ""),
        "wan_code_import_error_truncated": "" if wan_code_ok else first_wan_failure.get("error_truncated", ""),
        "wan_code_import_traceback_tail": [] if wan_code_ok else first_wan_failure.get("traceback_tail", []),
        "wan_code_import_attempted_modules": list(WAN_CODE_IMPORT_ATTEMPTED_MODULES),
        "runner_import_status": "ok" if runner_ok else "failed",
        "wan_code_imports": wan_code_imports,
        "runner_imports": runner_imports,
    }


def require_admin_download_enabled(payload: dict[str, Any]) -> None:
    if os.getenv(ADMIN_DOWNLOAD_ENV, "") != "1":
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Admin model download endpoint is disabled.",
                "required_env": ADMIN_DOWNLOAD_ENV,
                "expected_env_value": "1",
            },
        )
    if payload.get("confirm_download") != DOWNLOAD_CONFIRMATION:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Explicit download confirmation is required.",
                "required_field": "confirm_download",
                "expected_value": DOWNLOAD_CONFIRMATION,
            },
        )


def require_admin_verify_enabled() -> None:
    if os.getenv(ADMIN_DOWNLOAD_ENV, "") == "1" or os.getenv(ADMIN_VERIFY_ENV, "") == "1":
        return
    raise HTTPException(
        status_code=403,
        detail={
            "message": "Admin weight verify endpoint is disabled.",
            "required_env_any_of": [ADMIN_DOWNLOAD_ENV, ADMIN_VERIFY_ENV],
            "expected_env_value": "1",
        },
    )


def truncate_output(value: str) -> str:
    if len(value) <= OUTPUT_TRUNCATE_CHARS:
        return value
    return value[-OUTPUT_TRUNCATE_CHARS:]


def positive_int(value: Any, default: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return default
    return min(parsed, max_value)


def validate_job_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Job body must be a JSON object.")

    missing = [field for field in REQUIRED_JOB_FIELDS if not str(payload.get(field, "")).strip()]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required Wan2.2 S2V stub job field(s).",
                "missing": missing,
                "required": list(REQUIRED_JOB_FIELDS),
                "optional": list(OPTIONAL_JOB_FIELDS),
            },
        )
    return payload


def validate_run_job_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Job body must be a JSON object.")
    payload = dict(payload)
    if payload.get("target_width") in ("", None) and payload.get("width") in ("", None) and isinstance(payload.get("resolution"), str):
        try:
            width_text, height_text = payload["resolution"].lower().replace("*", "x").split("x", 1)
            payload["width"] = width_text
            payload["height"] = height_text
        except ValueError:
            pass
    if payload.get("target_width") in ("", None) and payload.get("width") not in ("", None):
        payload["target_width"] = payload["width"]
    if payload.get("target_height") in ("", None) and payload.get("height") not in ("", None):
        payload["target_height"] = payload["height"]
    if payload.get("confirm_inference") not in INFERENCE_CONFIRMATIONS:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Explicit inference confirmation is required.",
                "required_field": "confirm_inference",
                "expected_values": sorted(INFERENCE_CONFIRMATIONS),
            },
        )
    missing = [field for field in RUN_JOB_REQUIRED_FIELDS if payload.get(field) in ("", None)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Missing required Wan2.2 S2V run field(s).",
                "missing": missing,
                "required": list(RUN_JOB_REQUIRED_FIELDS),
                "optional": list(RUN_JOB_OPTIONAL_FIELDS),
            },
        )
    expected_values = {"fps": 16, "target_duration_seconds": 14.8}
    mismatches = {}
    for key, expected in expected_values.items():
        value = payload.get(key)
        if isinstance(expected, float):
            try:
                matches = abs(float(value) - expected) < 0.001
            except (TypeError, ValueError):
                matches = False
        else:
            matches = value == expected
        if not matches:
            mismatches[key] = {"expected": expected, "received": value}
    allowed_job_ids = {
        "mae_fr_wan22_s2v_14_8s_1080_v1",
        "mae_fr_wan22_s2v_14_8s_1080_blackwell_v1",
        "mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5",
        "mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial",
        "mae_fr_wan22_s2v_14_8s_720_blackwell_natural_v5_native_partial",
    }
    if payload.get("job_id") not in allowed_job_ids:
        mismatches["job_id"] = {"expected_one_of": sorted(allowed_job_ids), "received": payload.get("job_id")}
    try:
        width = int(payload.get("target_width"))
        height = int(payload.get("target_height"))
    except (TypeError, ValueError):
        mismatches["resolution"] = {"expected": "integer target_width/target_height or width/height", "received": {
            "target_width": payload.get("target_width"),
            "target_height": payload.get("target_height"),
        }}
    else:
        if width <= 0 or height <= 0 or width > 1080 or height > 1080:
            mismatches["resolution"] = {"expected": "positive width/height up to 1080", "received": {"width": width, "height": height}}
        payload["target_width"] = width
        payload["target_height"] = height
        payload["width"] = width
        payload["height"] = height
        payload["resolution"] = f"{width}x{height}"
    if mismatches:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected Maé first inference field(s).",
                "mismatches": mismatches,
            },
        )
    return payload


def summarize_run_job_report(report: Any) -> dict:
    if not isinstance(report, dict):
        return {"json_type": type(report).__name__}
    keys = (
        "job_id",
        "job_status",
        "status",
        "width",
        "height",
        "resolution",
        "requested_width",
        "requested_height",
        "requested_resolution",
        "message",
        "requested_resolution_detail",
        "actual_generation_resolution",
        "output_resolution",
        "output_width",
        "output_height",
        "fallback_used",
        "fps",
        "target_duration_seconds",
        "output_video_key",
        "output_report_key",
        "runtime_seconds",
        "peak_vram_gb",
        "estimated_cost",
        "inference_executed",
        "placeholder_generated",
        "video_generated",
        "r2_upload_attempted",
        "report_uploaded_to_r2",
        "error_type",
        "error_truncated",
        "error_message_truncated",
        "r2_env_check_status",
        "r2_reference_head_status",
        "r2_audio_head_status",
        "r2_upload_permission_check_status",
        "received_parameters",
        "forwarded_parameters",
        "unsupported_parameters",
        "supported_parameter_fields",
        "unsupported_parameter_fields",
        "safetensors_cuda_to_cpu_patch",
        "attention_sdpa_patch",
        "attention_backend_used",
        "attention_fallback_applied",
        "attention_patch_status",
        "attention_patch_calls_count",
        "attention_patched_modules",
        "max_concurrent_jobs",
        "active_jobs_at_submission",
    )
    summary = {key: report.get(key) for key in keys if key in report}
    attention = report.get("attention_sdpa_patch")
    if isinstance(attention, dict):
        summary.setdefault("attention_backend_used", attention.get("attention_backend_used"))
        summary.setdefault("attention_fallback_applied", attention.get("attention_fallback_applied"))
        summary.setdefault("attention_patch_status", attention.get("attention_patch_status"))
        summary.setdefault("attention_patch_calls_count", attention.get("attention_patch_calls_count"))
        summary.setdefault("attention_patched_modules", attention.get("patched_modules", []))
    primary = report.get("primary_inference")
    if isinstance(primary, dict):
        summary["subprocess_returncode"] = primary.get("returncode")
        summary["stdout_truncated"] = primary.get("stdout_truncated", "")
        summary["stderr_truncated"] = primary.get("stderr_truncated", "")
    return summary


def async_job_snapshot(job_id: str) -> dict:
    with ASYNC_JOBS_LOCK:
        job = ASYNC_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail={"message": "Async job not found.", "job_id": job_id})
        return dict(job)


def update_async_job(job_id: str, **updates: Any) -> None:
    with ASYNC_JOBS_LOCK:
        if job_id in ASYNC_JOBS:
            ASYNC_JOBS[job_id].update(updates)


def run_async_wan22_job(job_id: str, payload: dict[str, Any]) -> None:
    started_monotonic = time.monotonic()
    update_async_job(job_id, status="running", started_at=now_iso())
    try:
        report = run_wan22_s2v_single_job(payload)
        terminal_status = "succeeded" if isinstance(report, dict) and report.get("status") in {"succeeded", "succeeded_with_960_fallback"} else "failed"
        update_async_job(
            job_id,
            status=terminal_status,
            finished_at=now_iso(),
            runtime_seconds=round(time.monotonic() - started_monotonic, 3),
            summary=summarize_run_job_report(report),
            error_type=None if terminal_status == "succeeded" else (report.get("error_type") if isinstance(report, dict) else "unknown"),
            error_truncated=None if terminal_status == "succeeded" else (report.get("error_truncated") if isinstance(report, dict) else ""),
            result=report if isinstance(report, dict) else {},
        )
    except Exception as exc:
        update_async_job(
            job_id,
            status="failed",
            finished_at=now_iso(),
            runtime_seconds=round(time.monotonic() - started_monotonic, 3),
            summary={},
            error_type=type(exc).__name__,
            error_truncated=str(exc)[:1000],
        )
    finally:
        with ASYNC_JOBS_LOCK:
            ASYNC_RUNNING_JOB_IDS.discard(job_id)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "version": SERVICE_VERSION,
    }


@app.get("/")
def root() -> dict:
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "hint": "use /health, /gpu, or admin endpoints",
    }


@app.get("/gpu")
def gpu() -> dict:
    return torch_probe()


@app.get("/models")
def models() -> dict:
    settings = get_settings()
    return {
        "status": "checked",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "simplepod_models_root": file_facts(settings.simplepod_models_root),
        "wan22_s2v_model_dir": file_facts(settings.wan22_s2v_model_dir),
        "r2_env_present_redacted": r2_env_alias_presence(),
        "r2_client_configured": r2_env_ready(),
        "no_downloads_attempted": True,
    }


def safe_call(label: str, fn) -> dict:
    try:
        value = fn()
        return {"status": "succeeded", "label": label, "result": value}
    except Exception as exc:
        traceback_lines = traceback.format_exc().splitlines()
        return {
            "status": "failed",
            "label": label,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "traceback_tail": traceback_lines[-10:],
        }


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except Exception:
        return ""


def safetensors_device_check() -> dict:
    gpu_status = torch_probe()
    result = {
        "status": "started",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "torch": {
            "torch_import_status": gpu_status.get("torch_import_status"),
            "torch_version": gpu_status.get("torch_version", ""),
            "torch_cuda_version": gpu_status.get("torch_cuda_version", ""),
            "cuda_available": gpu_status.get("cuda_available"),
            "device_name": gpu_status.get("device_name", ""),
            "device_capability": gpu_status.get("device_capability"),
        },
        "versions": {
            "safetensors": package_version("safetensors"),
            "accelerate": package_version("accelerate"),
        },
        "downloads_model_weights": False,
        "loads_full_model": False,
        "inference_executed": False,
        "video_generated": False,
    }
    try:
        import torch
        import safetensors
        import safetensors.torch
        from safetensors import safe_open
    except Exception as exc:
        result["status"] = "failed_import"
        result["error_type"] = type(exc).__name__
        result["error_truncated"] = str(exc)[:1000]
        return result

    test_path = Path("/tmp/ayl_safetensors_device_check.safetensors")
    tensor_payload = {"tiny": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    save_result = safe_call(
        "safetensors.torch.save_file",
        lambda: (safetensors.torch.save_file(tensor_payload, str(test_path)), {"path": str(test_path), "size_bytes": test_path.stat().st_size})[1],
    )
    result["save_file"] = save_result

    def load_file_device(device: str) -> dict:
        loaded = safetensors.torch.load_file(str(test_path), device=device)
        tensor = loaded["tiny"]
        return {"keys": sorted(loaded.keys()), "tensor_device": str(tensor.device), "shape": list(tensor.shape)}

    def safe_open_device(device: str) -> dict:
        with safe_open(str(test_path), framework="pt", device=device) as handle:
            keys = list(handle.keys())
            tensor = handle.get_tensor("tiny")
        return {"keys": keys, "tensor_device": str(tensor.device), "shape": list(tensor.shape)}

    result["load_file_cpu"] = safe_call("safetensors.torch.load_file(device='cpu')", lambda: load_file_device("cpu"))
    result["load_file_cuda0"] = safe_call("safetensors.torch.load_file(device='cuda:0')", lambda: load_file_device("cuda:0"))
    result["safe_open_cpu"] = safe_call("safetensors.safe_open(device='cpu')", lambda: safe_open_device("cpu"))
    result["safe_open_cuda0"] = safe_call("safetensors.safe_open(device='cuda:0')", lambda: safe_open_device("cuda:0"))

    result["monkeypatch_cuda_to_cpu_simulation"] = {
        "enabled": True,
        "load_file_cuda0_redirected": safe_call("patched load_file cuda:0->cpu", lambda: load_file_device("cpu")),
        "safe_open_cuda0_redirected": safe_call("patched safe_open cuda:0->cpu", lambda: safe_open_device("cpu")),
    }
    cuda_checks = (result["load_file_cuda0"], result["safe_open_cuda0"])
    cpu_checks = (result["load_file_cpu"], result["safe_open_cpu"])
    if all(item["status"] == "succeeded" for item in cuda_checks):
        result["status"] = "cuda_device_supported"
    elif all(item["status"] == "succeeded" for item in cpu_checks):
        result["status"] = "cuda_device_failed_cpu_ok"
    else:
        result["status"] = "failed_cpu_and_cuda"
    return result


@app.get("/admin/check-safetensors-device")
def check_safetensors_device() -> dict:
    require_admin_verify_enabled()
    return safetensors_device_check()


def find_wan22_diffusion_shard(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.rglob("diffusion_pytorch_model-*.safetensors"))
    return candidates[0] if candidates else None


def tensor_slice_facts(handle, key: str) -> dict:
    tensor_slice = handle.get_slice(key)
    shape = list(tensor_slice.get_shape())
    dtype = str(tensor_slice.get_dtype())
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return {"key": key, "shape": shape, "dtype": dtype, "numel": numel}


def safe_open_shard_facts(path: Path, device: str, max_tensor_numel: int = 262_144) -> dict:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device=device) as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
        key_facts = []
        selected_key = ""
        for key in keys:
            facts = tensor_slice_facts(handle, key)
            if len(key_facts) < 20:
                key_facts.append(facts)
            if not selected_key and facts["numel"] <= max_tensor_numel:
                selected_key = key
        tensor_result = {
            "status": "skipped_no_small_tensor",
            "max_tensor_numel": max_tensor_numel,
        }
        if selected_key:
            tensor = handle.get_tensor(selected_key)
            tensor_result = {
                "status": "succeeded",
                "key": selected_key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "numel": int(tensor.numel()),
            }
    return {
        "keys_count": len(keys),
        "sample_keys": keys[:20],
        "metadata": metadata,
        "sample_tensor_slices": key_facts,
        "small_tensor_read": tensor_result,
    }


def wan22_safetensors_shard_check(allow_load_file_cpu: bool = False) -> dict:
    settings = get_settings()
    gpu_status = torch_probe()
    model_dir = settings.wan22_s2v_model_dir
    shard_path = find_wan22_diffusion_shard(model_dir)
    result = {
        "status": "started",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "model_dir": str(model_dir),
        "torch": {
            "torch_import_status": gpu_status.get("torch_import_status"),
            "torch_version": gpu_status.get("torch_version", ""),
            "torch_cuda_version": gpu_status.get("torch_cuda_version", ""),
            "cuda_available": gpu_status.get("cuda_available"),
            "device_name": gpu_status.get("device_name", ""),
            "device_capability": gpu_status.get("device_capability"),
        },
        "versions": {
            "safetensors": package_version("safetensors"),
            "accelerate": package_version("accelerate"),
        },
        "allow_load_file_cpu": allow_load_file_cpu,
        "downloads_model_weights": False,
        "loads_full_model": False,
        "load_file_cuda_attempted": False,
        "inference_executed": False,
        "video_generated": False,
    }
    if shard_path is None:
        result["status"] = "missing_diffusion_safetensors_shard"
        result["shard"] = {
            "exists": False,
            "pattern": "diffusion_pytorch_model-*.safetensors",
        }
        return result

    result["shard"] = {
        "path": str(shard_path),
        "relative_path": str(shard_path.relative_to(model_dir)) if shard_path.is_relative_to(model_dir) else str(shard_path),
        "exists": shard_path.exists(),
        "readable": os.access(shard_path, os.R_OK),
        "size_bytes": shard_path.stat().st_size if shard_path.exists() else None,
        "size_gb": round(shard_path.stat().st_size / (1024**3), 3) if shard_path.exists() else None,
    }
    result["safe_open_cpu"] = safe_call(
        "safe_open(real shard, device='cpu')",
        lambda: safe_open_shard_facts(shard_path, "cpu"),
    )
    result["safe_open_cuda0"] = safe_call(
        "safe_open(real shard, device='cuda:0')",
        lambda: safe_open_shard_facts(shard_path, "cuda:0"),
    )
    if allow_load_file_cpu:
        def load_file_cpu() -> dict:
            import safetensors.torch

            loaded = safetensors.torch.load_file(str(shard_path), device="cpu")
            sample = []
            for key, tensor in list(loaded.items())[:10]:
                sample.append({"key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device)})
            return {"keys_count": len(loaded), "sample_tensors": sample}

        result["loads_full_model"] = True
        result["load_file_cpu"] = safe_call("load_file(real shard, device='cpu')", load_file_cpu)
    else:
        result["load_file_cpu"] = {
            "status": "skipped_requires_explicit_allow_load_file_cpu",
            "reason": "Full shard load may be large; enable only for a dedicated diagnostic.",
        }
    result["load_file_cuda0"] = {
        "status": "not_attempted_by_design",
        "reason": "Full CUDA shard load is intentionally disabled.",
    }

    if result["safe_open_cpu"]["status"] != "succeeded":
        result["status"] = "failed_safe_open_cpu"
    elif result["safe_open_cuda0"]["status"] == "succeeded":
        result["status"] = "cuda_safe_open_shard_supported"
    else:
        result["status"] = "cuda_safe_open_shard_failed_cpu_ok"
    return result


@app.get("/admin/check-wan22-safetensors-shard")
def check_wan22_safetensors_shard(allow_load_file_cpu: bool = False) -> dict:
    require_admin_verify_enabled()
    return wan22_safetensors_shard_check(allow_load_file_cpu=allow_load_file_cpu)


def locate_wan_s2v_runtime_path() -> dict:
    if str(WAN22_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(WAN22_REPO_DIR))
    klass = None
    result = {
        "status": "started",
        "generate_entrypoint": {},
        "wan_package": {},
        "wan_s2v_pipeline": {},
        "noise_model": {},
    }
    result["generate_entrypoint"] = safe_import_module("generate", WAN22_REPO_DIR)
    result["wan_package"] = safe_import_module("wan", WAN22_REPO_DIR)
    try:
        import wan

        wan_s2v_class = getattr(wan, "WanS2V")
        wan_s2v_init = getattr(wan_s2v_class, "__init__", None)
        try:
            wan_s2v_signature = str(inspect.signature(wan_s2v_init)) if callable(wan_s2v_init) else ""
        except Exception as signature_exc:
            wan_s2v_signature = f"<signature_unavailable:{type(signature_exc).__name__}>"
        result["wan_s2v_pipeline"] = {
            "status": "ok",
            "module": str(getattr(wan_s2v_class, "__module__", "") or ""),
            "class_name": "WanS2V",
            "class_qualname": str(getattr(wan_s2v_class, "__qualname__", "WanS2V") or "WanS2V"),
            "module_file": str(sys.modules.get(getattr(wan_s2v_class, "__module__", ""), object()).__dict__.get("__file__", "")),
            "init_signature": wan_s2v_signature,
            "real_init_noise_model_call": (
                "WanModel_S2V.from_pretrained(checkpoint_dir, "
                "torch_dtype=config.param_dtype, device_map=torch.device('cuda:{device_id}'))"
            ),
        }
    except Exception as exc:
        traceback_lines = traceback.format_exc().splitlines()
        result["wan_s2v_pipeline"] = {
            "status": "failed",
            "module": "wan",
            "class_name": "WanS2V",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "traceback_tail": traceback_lines[-8:],
        }

    attempts = []
    for module_name, class_name in WAN_MODEL_S2V_CLASS_CANDIDATES:
        attempt = {
            "module": module_name,
            "class_name": class_name,
            "status": "started",
        }
        try:
            module = importlib.import_module(module_name)
            klass = getattr(module, class_name)
            from_pretrained = getattr(klass, "from_pretrained", None)
            try:
                signature = str(inspect.signature(from_pretrained)) if callable(from_pretrained) else ""
            except Exception as signature_exc:
                signature = f"<signature_unavailable:{type(signature_exc).__name__}>"
            attempt.update(
                {
                    "status": "ok",
                    "module_file": str(getattr(module, "__file__", "") or ""),
                    "class_module": str(getattr(klass, "__module__", "") or ""),
                    "class_qualname": str(getattr(klass, "__qualname__", class_name) or class_name),
                    "has_from_pretrained": callable(from_pretrained),
                    "from_pretrained_signature": signature,
                }
            )
            attempt["_class"] = klass
        except Exception as exc:
            traceback_lines = traceback.format_exc().splitlines()
            attempt.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_truncated": str(exc)[:1000],
                    "traceback_tail": traceback_lines[-8:],
                }
            )
        attempts.append(attempt)
    selected = next((attempt for attempt in attempts if attempt.get("status") == "ok"), {})
    klass = selected.pop("_class", None) if selected else None
    result["noise_model"] = {
        "status": "ok" if selected else "failed",
        "selected": selected,
        "attempts": attempts,
    }
    result["status"] = (
        "ok"
        if result["wan_s2v_pipeline"].get("status") == "ok" and result["noise_model"]["status"] == "ok"
        else "failed"
    )
    result["_class"] = klass
    return result


def wan22_checkpoint_inventory(model_dir: Path) -> dict:
    safetensors_paths = sorted(model_dir.rglob("*.safetensors")) if model_dir.exists() else []
    index_paths = sorted(model_dir.rglob("*.index.json")) if model_dir.exists() else []
    config_paths = sorted(model_dir.rglob("config.json")) if model_dir.exists() else []
    return {
        "model_dir": str(model_dir),
        "model_dir_exists": model_dir.exists(),
        "model_dir_is_dir": model_dir.is_dir(),
        "safetensors_count": len(safetensors_paths),
        "safetensors_sample": [
            {
                "path": str(path),
                "relative_path": str(path.relative_to(model_dir)) if path.is_relative_to(model_dir) else str(path),
                "size_gb": round(path.stat().st_size / (1024**3), 3),
            }
            for path in safetensors_paths[:20]
        ],
        "index_json_files": [
            str(path.relative_to(model_dir)) if path.is_relative_to(model_dir) else str(path)
            for path in index_paths[:20]
        ],
        "config_json_files": [
            str(path.relative_to(model_dir)) if path.is_relative_to(model_dir) else str(path)
            for path in config_paths[:20]
        ],
    }


def sanitize_dispatch_kwargs(kwargs: dict[str, Any]) -> dict:
    sanitized = {}
    for key, value in kwargs.items():
        if key == "torch_dtype":
            sanitized[key] = str(value)
        elif key == "device_map":
            sanitized[key] = str(value)
        else:
            sanitized[key] = value
    return sanitized


def should_redirect_safetensors_device(device: Any) -> bool:
    if isinstance(device, str):
        return device.startswith("cuda")
    try:
        return str(device).startswith("cuda")
    except Exception:
        return False


@contextmanager
def scoped_safetensors_cuda_to_cpu_patch(enabled: bool):
    state = {
        "patch_requested": bool(enabled),
        "patch_applied": False,
        "patched_calls_count": 0,
        "redirected_devices": [],
        "restored": False,
        "error_type": "",
        "error_truncated": "",
    }
    if not enabled:
        yield state
        return

    try:
        import safetensors
        import safetensors.torch
    except Exception as exc:
        state["error_type"] = type(exc).__name__
        state["error_truncated"] = str(exc)[:1000]
        yield state
        return

    original_load_file = safetensors.torch.load_file
    original_safe_open = safetensors.safe_open
    original_torch_safe_open = getattr(safetensors.torch, "safe_open", original_safe_open)

    def redirect_device(device: Any) -> Any:
        if should_redirect_safetensors_device(device):
            state["patched_calls_count"] += 1
            device_text = str(device)
            if device_text not in state["redirected_devices"]:
                state["redirected_devices"].append(device_text)
            return "cpu"
        return device

    def patched_load_file(filename, device="cpu", *args, **kwargs):
        return original_load_file(filename, device=redirect_device(device), *args, **kwargs)

    def patched_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    def patched_torch_safe_open(filename, framework, device="cpu", *args, **kwargs):
        return original_torch_safe_open(filename, framework=framework, device=redirect_device(device), *args, **kwargs)

    safetensors.torch.load_file = patched_load_file
    safetensors.safe_open = patched_safe_open
    safetensors.torch.safe_open = patched_torch_safe_open
    state["patch_applied"] = True
    try:
        yield state
    finally:
        safetensors.torch.load_file = original_load_file
        safetensors.safe_open = original_safe_open
        safetensors.torch.safe_open = original_torch_safe_open
        state["restored"] = True


def model_parameter_device_summary(model_obj: Any, sample_limit: int = 2000) -> dict:
    summary = {
        "first_parameter_device": "",
        "first_parameter_dtype": "",
        "any_parameter_on_cuda": False,
        "parameter_device_counts": {},
        "sampled_parameters": 0,
        "sample_limit": sample_limit,
    }
    if model_obj is None or not hasattr(model_obj, "parameters"):
        return summary
    for idx, parameter in enumerate(model_obj.parameters()):
        if idx >= sample_limit:
            break
        device = str(getattr(parameter, "device", ""))
        dtype = str(getattr(parameter, "dtype", ""))
        if idx == 0:
            summary["first_parameter_device"] = device
            summary["first_parameter_dtype"] = dtype
        summary["parameter_device_counts"][device] = summary["parameter_device_counts"].get(device, 0) + 1
        if device.startswith("cuda"):
            summary["any_parameter_on_cuda"] = True
        summary["sampled_parameters"] += 1
    return summary


def wan22_accelerate_dispatch_check(apply_safetensors_cuda_to_cpu_patch: bool = False) -> dict:
    settings = get_settings()
    gpu_status = torch_probe()
    model_dir = settings.wan22_s2v_model_dir
    checkpoint_inventory = wan22_checkpoint_inventory(model_dir)
    patch_requested = os.getenv("AYL_SAFETENSORS_CUDA_TO_CPU_PATCH", "") == "1" or apply_safetensors_cuda_to_cpu_patch
    result = {
        "status": "started",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "model_dir": str(model_dir),
        "wan_repo_path": str(WAN22_REPO_DIR),
        "wan_repo_path_exists": WAN22_REPO_DIR.exists(),
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "sys_path_tail": sys.path[-12:],
        "torch": {
            "torch_import_status": gpu_status.get("torch_import_status"),
            "torch_version": gpu_status.get("torch_version", ""),
            "torch_cuda_version": gpu_status.get("torch_cuda_version", ""),
            "cuda_available": gpu_status.get("cuda_available"),
            "device_name": gpu_status.get("device_name", ""),
            "device_capability": gpu_status.get("device_capability"),
        },
        "versions": {
            "torch": package_version("torch"),
            "diffusers": package_version("diffusers"),
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
            "transformers": package_version("transformers"),
        },
        "checkpoint_inventory": checkpoint_inventory,
        "device_map": "cuda:0",
        "offload": False,
        "dtype": "",
        "low_cpu_mem_usage": "not_forwarded_by_real_wan_s2v_path",
        "local_files_only": "not_forwarded_by_real_wan_s2v_path",
        "download_attempted": False,
        "downloads_attempted": False,
        "loads_full_model": True,
        "sampling_executed": False,
        "generate_called": False,
        "inference_executed": False,
        "video_generated": False,
        "placeholder_generated": False,
        "safetensors_cuda_to_cpu_patch": {
            "patch_requested": patch_requested,
            "patch_applied": False,
            "patched_calls_count": 0,
            "redirected_devices": [],
        },
    }
    if not model_dir.exists() or not model_dir.is_dir():
        result["status"] = "missing_model_dir"
        return result
    locate_result = locate_wan_s2v_runtime_path()
    klass = locate_result.pop("_class", None)
    result["wan_s2v_runtime_path"] = locate_result
    if locate_result.get("status") != "ok" or klass is None:
        result["status"] = "failed_import_wan_model_s2v"
        return result

    model_obj = None
    patch_state_result = dict(result["safetensors_cuda_to_cpu_patch"])
    try:
        import torch
        from wan.configs import WAN_CONFIGS

        cfg = WAN_CONFIGS["s2v-14B"]
        device_map = torch.device("cuda:0")

        dispatch_kwargs = {
            "torch_dtype": cfg.param_dtype,
            "device_map": device_map,
        }
        result["dtype"] = str(cfg.param_dtype)
        result["config"] = {
            "task": "s2v-14B",
            "param_dtype": str(cfg.param_dtype),
            "t5_checkpoint": str(getattr(cfg, "t5_checkpoint", "")),
            "vae_checkpoint": str(getattr(cfg, "vae_checkpoint", "")),
            "transformer": {
                "motion_frames": getattr(getattr(cfg, "transformer", {}), "motion_frames", None),
            },
        }
        result["from_pretrained_call"] = {
            "class_module": str(getattr(klass, "__module__", "") or ""),
            "class_qualname": str(getattr(klass, "__qualname__", "") or ""),
            "pretrained_model_name_or_path": str(model_dir),
            "kwargs": sanitize_dispatch_kwargs(dispatch_kwargs),
            "matches_real_wan_s2v_init_noise_model_call": True,
            "purpose": "diagnose WanS2V noise_model checkpoint dispatch only; no WanS2V.generate, sampling, audio encode, VAE decode, or video save",
        }
        with scoped_safetensors_cuda_to_cpu_patch(patch_requested) as patch_state:
            model_obj = klass.from_pretrained(str(model_dir), **dispatch_kwargs)
        patch_state_result = dict(patch_state)
        result["safetensors_cuda_to_cpu_patch"] = patch_state_result
        parameter_summary = model_parameter_device_summary(model_obj)
        result["status"] = "dispatch_succeeded"
        result["from_pretrained_result"] = {
            "status": "succeeded",
            "object_type": type(model_obj).__name__,
            "object_module": type(model_obj).__module__,
            **parameter_summary,
        }
    except Exception as exc:
        traceback_lines = traceback.format_exc().splitlines()
        if "patch_state" in locals():
            patch_state_result = dict(patch_state)
        result["safetensors_cuda_to_cpu_patch"] = patch_state_result
        result["status"] = "failed_accelerate_dispatch"
        result["from_pretrained_result"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:2000],
            "traceback_tail": traceback_lines[-24:],
        }
    finally:
        try:
            del model_obj
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return result


@app.get("/admin/check-wan22-accelerate-dispatch")
def check_wan22_accelerate_dispatch(apply_safetensors_cuda_to_cpu_patch: bool = False) -> dict:
    require_admin_verify_enabled()
    return wan22_accelerate_dispatch_check(
        apply_safetensors_cuda_to_cpu_patch=apply_safetensors_cuda_to_cpu_patch,
    )


@app.post("/admin/download-wan22-s2v-weights")
def download_wan22_s2v_weights(payload: dict[str, Any]) -> dict:
    require_admin_download_enabled(payload)
    settings = get_settings()
    target_dir = Path(str(payload.get("target_dir") or settings.wan22_s2v_model_dir))
    timeout_seconds = positive_int(
        payload.get("timeout_seconds"),
        DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
        DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
    if str(target_dir) != str(settings.wan22_s2v_model_dir):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Unexpected target_dir. This gate only downloads to WAN22_S2V_MODEL_DIR.",
                "expected_target_dir": str(settings.wan22_s2v_model_dir),
            },
        )

    before = directory_inventory(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "huggingface-cli",
        "download",
        WAN22_S2V_REPO_ID,
        "--local-dir",
        str(target_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        status = "succeeded" if completed.returncode == 0 else "failed_download"
        command_result = {
            "status": status,
            "returncode": completed.returncode,
            "timeout_seconds": timeout_seconds,
            "stdout_truncated": truncate_output(completed.stdout or ""),
            "stderr_truncated": truncate_output(completed.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        command_result = {
            "status": status,
            "returncode": None,
            "timeout_seconds": timeout_seconds,
            "stdout_truncated": truncate_output((exc.stdout or "") if isinstance(exc.stdout, str) else ""),
            "stderr_truncated": truncate_output((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
        }
    except Exception as exc:
        status = "failed_download"
        command_result = {
            "status": status,
            "returncode": None,
            "timeout_seconds": timeout_seconds,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "stdout_truncated": "",
            "stderr_truncated": "",
        }

    after = directory_inventory(target_dir)
    return {
        "status": status,
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "repo_id": WAN22_S2V_REPO_ID,
        "target_dir": str(target_dir),
        "timeout_seconds": timeout_seconds,
        "command": command,
        "command_result": command_result,
        "download_command_equivalent": (
            f"huggingface-cli download {WAN22_S2V_REPO_ID} "
            f"--local-dir {target_dir}"
        ),
        "before": before,
        "after": after,
        "inference_executed": False,
        "video_generated": False,
    }


@app.get("/admin/verify-wan22-s2v-weights")
def verify_wan22_s2v_weights() -> dict:
    require_admin_verify_enabled()
    settings = get_settings()
    inventory = directory_inventory(settings.wan22_s2v_model_dir)
    return {
        "status": "verified" if inventory["exists"] and inventory["is_dir"] else "missing",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "repo_id": WAN22_S2V_REPO_ID,
        "target_dir": str(settings.wan22_s2v_model_dir),
        "path": str(settings.wan22_s2v_model_dir),
        "exists": inventory["exists"],
        "is_dir": inventory["is_dir"],
        "recursive_file_count": inventory["recursive_file_count"],
        "recursive_total_size_bytes": inventory["recursive_total_size_bytes"],
        "recursive_total_size_gb": inventory["recursive_total_size_gb"],
        "safetensors_files": inventory["safetensors_files"],
        "marker_files": inventory["marker_files"],
        "sample_files": inventory["sample_files"],
        "downloads_attempted": False,
        "inference_executed": False,
        "video_generated": False,
    }


@app.get("/admin/verify-wan22-s2v-runtime")
def verify_wan22_s2v_runtime() -> dict:
    require_admin_verify_enabled()
    settings = get_settings()
    gpu_status = torch_probe()
    models_root_inventory = directory_inventory(settings.simplepod_models_root)
    model_inventory = directory_inventory(settings.wan22_s2v_model_dir)
    import_checks = runtime_import_checks()
    required_files_found = {
        "safetensors_count": len(model_inventory["safetensors_files"]),
        "has_safetensors": bool(model_inventory["safetensors_files"]),
        "marker_file_count": len(model_inventory["marker_files"]),
        "marker_files": model_inventory["marker_files"],
    }
    checks = {
        "torch_import_ok": gpu_status.get("torch_import_status") == "ok",
        "cuda_available": gpu_status.get("cuda_available") is True,
        "models_root_exists": models_root_inventory["exists"] and models_root_inventory["is_dir"],
        "wan22_model_dir_exists": model_inventory["exists"] and model_inventory["is_dir"],
        "weights_file_count_positive": model_inventory["recursive_file_count"] > 0,
        "weights_size_positive": model_inventory["recursive_total_size_bytes"] > 0,
        "required_files_found": required_files_found["has_safetensors"],
        "wan_code_import_ok": import_checks["wan_code_import_status"] == "ok",
        "runner_import_ok": import_checks["runner_import_status"] == "ok",
    }
    return {
        "status": "verified" if all(checks.values()) else "failed_runtime_verify",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "torch_version": gpu_status.get("torch_version", ""),
        "torch_cuda_version": gpu_status.get("torch_cuda_version", ""),
        "device_name": gpu_status.get("device_name", ""),
        "device_capability": gpu_status.get("device_capability"),
        "gpu": gpu_status,
        "models_root": str(settings.simplepod_models_root),
        "models_root_exists": models_root_inventory["exists"],
        "models_root_is_dir": models_root_inventory["is_dir"],
        "wan22_model_dir": str(settings.wan22_s2v_model_dir),
        "wan22_model_dir_exists": model_inventory["exists"],
        "wan22_model_dir_is_dir": model_inventory["is_dir"],
        "wan_repo_path": str(WAN22_REPO_DIR),
        "wan_repo_path_exists": WAN22_REPO_DIR.exists(),
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "sys_path_tail": sys.path[-12:],
        "installed_packages_relevant": installed_packages_relevant(),
        "recursive_file_count": model_inventory["recursive_file_count"],
        "recursive_total_size_bytes": model_inventory["recursive_total_size_bytes"],
        "recursive_total_size_gb": model_inventory["recursive_total_size_gb"],
        "required_files_found": required_files_found,
        "safetensors_files": model_inventory["safetensors_files"],
        "marker_files": model_inventory["marker_files"],
        "sample_files": model_inventory["sample_files"],
        "wan_code_import_status": import_checks["wan_code_import_status"],
        "wan_code_import_error_type": import_checks["wan_code_import_error_type"],
        "wan_code_import_error_truncated": import_checks["wan_code_import_error_truncated"],
        "wan_code_import_traceback_tail": import_checks["wan_code_import_traceback_tail"],
        "wan_code_import_attempted_modules": import_checks["wan_code_import_attempted_modules"],
        "runner_import_status": import_checks["runner_import_status"],
        "import_checks": import_checks,
        "checks": checks,
        "download_attempted": False,
        "downloads_attempted": False,
        "inference_executed": False,
        "video_generated": False,
        "placeholder_generated": False,
    }


@app.post("/admin/run-mae-wan22-s2v-async", status_code=202)
def run_mae_wan22_s2v_async(payload: dict[str, Any]) -> dict:
    require_admin_verify_enabled()
    job_payload = validate_run_job_payload(payload)
    settings = get_settings()
    model_inventory = directory_inventory(settings.wan22_s2v_model_dir)
    if not model_inventory["exists"] or not model_inventory["is_dir"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Wan2.2 S2V model directory is missing.",
                "model_dir": str(settings.wan22_s2v_model_dir),
                "model_inventory": model_inventory,
            },
        )
    if not r2_env_ready():
        raise HTTPException(
            status_code=409,
            detail={
                "message": "R2 env is not configured for input/output.",
                "r2_env_present_redacted": r2_env_alias_presence(),
            },
        )

    job_id = str(job_payload["job_id"])
    created_at = now_iso()
    with ASYNC_JOBS_LOCK:
        active_job_ids = sorted(ASYNC_RUNNING_JOB_IDS)
        active_jobs_at_submission = len(active_job_ids)
        if active_jobs_at_submission >= settings.max_concurrent_jobs:
            return JSONResponse(
                status_code=409,
                content={
                    "status": "rejected",
                    "reason": "max_concurrent_jobs_reached",
                    "max_concurrent_jobs": settings.max_concurrent_jobs,
                    "active_job_ids": active_job_ids,
                },
            )
        job_payload["max_concurrent_jobs"] = settings.max_concurrent_jobs
        job_payload["active_jobs_at_submission"] = active_jobs_at_submission
        ASYNC_RUNNING_JOB_IDS.add(job_id)
        ASYNC_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "runtime_seconds": None,
            "summary": {},
            "error_type": None,
            "error_truncated": None,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "active_jobs_at_submission": active_jobs_at_submission,
            "width": job_payload["target_width"],
            "height": job_payload["target_height"],
            "resolution": job_payload["resolution"],
            "output_video_key": job_payload["output_video_key"],
            "output_report_key": job_payload["output_report_key"],
        }

    thread = threading.Thread(target=run_async_wan22_job, args=(job_id, dict(job_payload)), daemon=True)
    thread.start()
    return {
        "status": "accepted",
        "job_id": job_id,
        "created_at": created_at,
        "status_url": f"/admin/jobs/{job_id}",
        "max_concurrent_jobs": settings.max_concurrent_jobs,
        "active_jobs_at_submission": active_jobs_at_submission,
    }


@app.get("/admin/jobs/{job_id}")
def get_async_job(job_id: str) -> dict:
    require_admin_verify_enabled()
    job = async_job_snapshot(job_id)
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "runtime_seconds": job.get("runtime_seconds"),
        "summary": job.get("summary", {}),
        "error_type": job.get("error_type"),
        "error_truncated": job.get("error_truncated"),
        "width": job.get("width"),
        "height": job.get("height"),
        "resolution": job.get("resolution"),
        "max_concurrent_jobs": job.get("max_concurrent_jobs"),
        "active_jobs_at_submission": job.get("active_jobs_at_submission"),
        "output_video_key": job.get("output_video_key"),
        "output_report_key": job.get("output_report_key"),
    }


@app.post("/jobs/wan22-s2v/run")
def run_wan22_s2v_job(payload: dict[str, Any]) -> dict:
    job_payload = validate_run_job_payload(payload)
    settings = get_settings()
    model_inventory = directory_inventory(settings.wan22_s2v_model_dir)
    gpu_status = torch_probe()
    if not model_inventory["exists"] or not model_inventory["is_dir"]:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Wan2.2 S2V model directory is missing.",
                "model_dir": str(settings.wan22_s2v_model_dir),
                "model_inventory": model_inventory,
            },
        )
    if not r2_env_ready():
        raise HTTPException(
            status_code=409,
            detail={
                "message": "R2 env is not configured for input/output.",
                "r2_env_present_redacted": r2_env_alias_presence(),
            },
        )
    report = run_wan22_s2v_single_job(job_payload)
    report["service"] = SERVICE_NAME
    report["timestamp"] = now_iso()
    report["gpu"] = gpu_status
    return report


@app.post("/jobs/wan22-s2v")
def create_wan22_s2v_job(payload: dict[str, Any]) -> dict:
    job_payload = validate_job_payload(payload)
    job_id = str(job_payload.get("job_id") or uuid.uuid4())
    report = stub_final_report(job_id, redact_payload(job_payload))
    return {
        "job_id": job_id,
        "status": "stub_created",
        "received": True,
        "final_report": report,
    }
