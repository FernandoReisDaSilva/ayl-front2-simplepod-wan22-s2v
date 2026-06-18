import argparse
import json
import os
import shutil
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_PREFIX = "tests/runpod_latentsync_image_v1_1"
DEFAULT_PROGRESS_KEY = f"{TEST_PREFIX}/progress/container_started.json"
DEFAULT_FINAL_KEY = f"{TEST_PREFIX}/output/final_report.json"
LATENTSYNC_PATH_CANDIDATES = (
    "/workspace/LatentSync",
    "/opt/LatentSync",
    "/app/LatentSync",
)
R2_ENV_KEYS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_presence() -> dict:
    keys = (
        "AYL_RUN_MODE",
        "AYL_IMAGE_TAG",
        "R2_PROGRESS_KEY",
        "R2_FINAL_REPORT_KEY",
        *R2_ENV_KEYS,
    )
    return {key: bool(os.getenv(key, "")) for key in keys}


def require_r2_env() -> None:
    missing = [key for key in R2_ENV_KEYS if not os.getenv(key, "")]
    if missing:
        raise RuntimeError("Missing required R2 env var(s): " + ", ".join(missing))


def r2_client():
    import boto3

    require_r2_env()
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.environ["R2_REGION"],
    )


def upload_json(key: str, payload: dict) -> None:
    path = Path("/tmp") / (Path(key).name + ".json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r2_client().upload_file(str(path), os.environ["R2_BUCKET"], key)


def base_report(mode: str) -> dict:
    return {
        "test_id": "TEST_RUNPOD_LATENTSYNC_IMAGE_V1_1_ENTRYPOINT_PROBE",
        "mode": mode,
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "image_tag": os.getenv("AYL_IMAGE_TAG", ""),
        "marker_nonce": os.getenv("AYL_MARKER_NONCE", ""),
        "env_present_redacted": env_presence(),
        "download_checkpoints": os.getenv("DOWNLOAD_CHECKPOINTS", "0"),
        "no_checkpoint_downloads": True,
        "no_inference": True,
    }


def write_progress(mode: str, status: str, extra: dict | None = None) -> None:
    progress_key = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    payload = {
        **base_report(mode),
        "status": status,
    }
    if extra:
        payload.update(extra)
    upload_json(
        progress_key,
        payload,
    )


def torch_probe() -> dict:
    result = {
        "torch_import_status": "not_attempted",
        "torch_version": "",
        "cuda_available": False,
        "gpu_name": "",
        "error_truncated": "",
    }
    try:
        import torch

        result["torch_import_status"] = "ok"
        result["torch_version"] = getattr(torch, "__version__", "") or ""
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_import_status"] = "failed"
        result["error_truncated"] = str(exc)[:1000]
    return result


def latentsync_paths() -> dict:
    return {path: Path(path).exists() for path in LATENTSYNC_PATH_CANDIDATES}


def build_report(mode: str) -> dict:
    report = base_report(mode)
    if mode == "r2_probe":
        report.update(
            {
                "probe_scope": "r2_only",
                "runtime_probe_status": "ok",
            }
        )
    elif mode == "latentsync_probe":
        write_progress(mode, "torch_check_started")
        torch_result = torch_probe()
        write_progress(
            mode,
            "torch_check_done",
            {
                "torch_import_status": torch_result["torch_import_status"],
                "torch_version": torch_result["torch_version"],
                "cuda_available": torch_result["cuda_available"],
                "gpu_name": torch_result["gpu_name"],
            },
        )
        ffmpeg_exists = shutil.which("ffmpeg") is not None
        write_progress(mode, "ffmpeg_check_done", {"ffmpeg_exists": ffmpeg_exists})
        path_candidates = latentsync_paths()
        path_exists = any(path_candidates.values())
        write_progress(
            mode,
            "latentsync_path_check_done",
            {
                "latentsync_path_candidates": path_candidates,
                "latentsync_path_exists": path_exists,
            },
        )
        report.update(
            {
                "probe_scope": "latentsync_runtime_import_check",
                "torch_import_status": torch_result["torch_import_status"],
                "torch_version": torch_result["torch_version"],
                "cuda_available": torch_result["cuda_available"],
                "gpu_name": torch_result["gpu_name"],
                "torch_error_truncated": torch_result["error_truncated"],
                "ffmpeg_exists": ffmpeg_exists,
                "latentsync_path_candidates": path_candidates,
                "latentsync_path_exists": path_exists,
            }
        )
    else:
        raise RuntimeError(f"Unsupported runtime probe mode: {mode}")
    return report


def run(mode: str) -> int:
    print(f"[AYL_RUNTIME_PROBE] start mode={mode}", flush=True)
    write_progress(mode, "container_started")
    final_key = os.getenv("R2_FINAL_REPORT_KEY", DEFAULT_FINAL_KEY)
    report = build_report(mode)
    report["r2_progress_key"] = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    report["r2_final_report_key"] = final_key
    report["r2_upload_status"] = "ok"
    upload_json(final_key, report)
    write_progress(mode, "final_report_written", {"r2_final_report_key": final_key})
    print(f"[AYL_RUNTIME_PROBE] done mode={mode} status=ok", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AYL LatentSync RunPod image runtime probe.")
    parser.add_argument("--mode", choices=("r2_probe", "latentsync_probe"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args.mode)
    except Exception as exc:
        print(f"[AYL_RUNTIME_PROBE] error={str(exc)[:300]}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
