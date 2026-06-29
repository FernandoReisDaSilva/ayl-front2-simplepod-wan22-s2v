import uuid
import os
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from .r2_client import r2_env_presence, r2_env_ready
from .reporting import file_facts, now_iso, stub_final_report
from .settings import SERVICE_NAME, SERVICE_VERSION, get_settings, is_secret_key


app = FastAPI(title=SERVICE_NAME, version=SERVICE_VERSION)

REQUIRED_JOB_FIELDS = ("reference_image_key", "audio_key")
OPTIONAL_JOB_FIELDS = ("output_video_key", "final_report_key", "prompt", "seed", "duration_seconds")
WAN22_S2V_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
DOWNLOAD_CONFIRMATION = "DOWNLOAD_WAN22_S2V_WEIGHTS"
ADMIN_DOWNLOAD_ENV = "AYL_ENABLE_ADMIN_DOWNLOADS"
ADMIN_VERIFY_ENV = "AYL_ENABLE_ADMIN_VERIFY"
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 7200
OUTPUT_TRUNCATE_CHARS = 4000


def torch_probe() -> dict:
    result = {
        "torch_import_status": "not_attempted",
        "torch_version": "",
        "cuda_available": False,
        "device_name": "",
        "vram_total_gb": None,
        "error_truncated": "",
    }
    try:
        import torch

        result["torch_import_status"] = "ok"
        result["torch_version"] = getattr(torch, "__version__", "") or ""
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["device_name"] = torch.cuda.get_device_name(0)
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


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "timestamp": now_iso(),
        "version": SERVICE_VERSION,
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
        "r2_env_present_redacted": r2_env_presence(),
        "r2_client_configured": r2_env_ready(),
        "no_downloads_attempted": True,
    }


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
