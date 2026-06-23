import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROGRESS_KEY = "tests/runpod_wan27_probe_v1/progress/container_started.json"
DEFAULT_FINAL_KEY = "tests/runpod_wan27_probe_v1/output/final_report.json"
DEFAULT_REFERENCE_KEY = "tests/runpod_wan27_probe_v1/input/mae_reference.png"
DEFAULT_AUDIO_KEY = "tests/runpod_wan27_probe_v1/input/audio.wav"
DEFAULT_SOURCE_VIDEO_KEY = "tests/runpod_wan27_probe_v1/input/mae_5s.mp4"
DEFAULT_OUTPUT_KEY = "tests/runpod_wan27_probe_v1/output/video_out.mp4"

R2_ENV_KEYS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
)

WORKSPACE = Path("/workspace")
INPUT_DIR = WORKSPACE / "input"
OUTPUT_DIR = WORKSPACE / "output"
REFERENCE_PATH = INPUT_DIR / "mae_reference.png"
AUDIO_PATH = INPUT_DIR / "audio.wav"
SOURCE_VIDEO_PATH = INPUT_DIR / "mae_5s.mp4"
OUTPUT_PATH = OUTPUT_DIR / "video_out.mp4"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_presence() -> dict:
    keys = (
        "AYL_RUN_MODE",
        "AYL_IMAGE_TAG",
        "AYL_MARKER_NONCE",
        "R2_PROGRESS_KEY",
        "R2_FINAL_REPORT_KEY",
        "R2_INPUT_REFERENCE_IMAGE_KEY",
        "R2_INPUT_AUDIO_KEY",
        "R2_INPUT_SOURCE_VIDEO_KEY",
        "R2_OUTPUT_VIDEO_KEY",
        "WAN27_COMMAND",
        "WAN27_DURATION_SECONDS",
        "WAN27_RESOLUTION",
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


def download_r2_file(key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    r2_client().download_file(os.environ["R2_BUCKET"], key, str(destination))


def upload_r2_file(source: Path, key: str) -> None:
    r2_client().upload_file(str(source), os.environ["R2_BUCKET"], key)


def file_facts(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else 0,
    }


def base_report(mode: str) -> dict:
    return {
        "test_id": "TEST_RUNPOD_WAN27_PROBE_V1",
        "mode": mode,
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "image_tag": os.getenv("AYL_IMAGE_TAG", ""),
        "marker_nonce": os.getenv("AYL_MARKER_NONCE", ""),
        "env_present_redacted": env_presence(),
        "network_volume_required": False,
        "dockerArgs_used": False,
        "not_latentsync": True,
    }


def write_progress(mode: str, status: str, extra: dict | None = None) -> None:
    progress_key = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    payload = {**base_report(mode), "status": status}
    if extra:
        payload.update(extra)
    upload_json(progress_key, payload)


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


def r2_keys() -> dict:
    return {
        "reference_image": os.getenv("R2_INPUT_REFERENCE_IMAGE_KEY", DEFAULT_REFERENCE_KEY),
        "audio": os.getenv("R2_INPUT_AUDIO_KEY", DEFAULT_AUDIO_KEY),
        "source_video": os.getenv("R2_INPUT_SOURCE_VIDEO_KEY", DEFAULT_SOURCE_VIDEO_KEY),
        "output_video": os.getenv("R2_OUTPUT_VIDEO_KEY", DEFAULT_OUTPUT_KEY),
    }


def wan27_command() -> str:
    command = os.getenv("WAN27_COMMAND", "").strip()
    if command:
        return command
    return ""


def run_wan27_command(command: str) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "WAN27_REFERENCE_IMAGE_PATH": str(REFERENCE_PATH),
            "WAN27_AUDIO_PATH": str(AUDIO_PATH),
            "WAN27_SOURCE_VIDEO_PATH": str(SOURCE_VIDEO_PATH),
            "WAN27_OUTPUT_VIDEO_PATH": str(OUTPUT_PATH),
            "WAN27_DURATION_SECONDS": os.getenv("WAN27_DURATION_SECONDS", "5"),
            "WAN27_RESOLUTION": os.getenv("WAN27_RESOLUTION", "480p"),
        }
    )
    started_at = now_iso()
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(WORKSPACE),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.getenv("WAN27_COMMAND_TIMEOUT_SECONDS", "1500")),
        check=False,
    )
    return {
        "started_at": started_at,
        "finished_at": now_iso(),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_file": file_facts(OUTPUT_PATH),
    }


def build_wan27_report(mode: str) -> dict:
    report = base_report(mode)
    keys = r2_keys()
    paths = {
        "reference_image": str(REFERENCE_PATH),
        "audio": str(AUDIO_PATH),
        "source_video": str(SOURCE_VIDEO_PATH),
        "output_video": str(OUTPUT_PATH),
    }
    report.update(
        {
            "probe_scope": "wan27_minimal_functional_probe",
            "r2_input_keys": keys,
            "container_paths": paths,
            "target_duration_seconds": os.getenv("WAN27_DURATION_SECONDS", "5"),
            "target_resolution": os.getenv("WAN27_RESOLUTION", "480p"),
        }
    )

    torch_result = torch_probe()
    write_progress(mode, "gpu_check_done", torch_result)
    ffmpeg_exists = shutil.which("ffmpeg") is not None
    write_progress(mode, "ffmpeg_check_done", {"ffmpeg_exists": ffmpeg_exists})

    download_r2_file(keys["reference_image"], REFERENCE_PATH)
    download_r2_file(keys["audio"], AUDIO_PATH)
    download_r2_file(keys["source_video"], SOURCE_VIDEO_PATH)
    input_files = {
        "reference_image": file_facts(REFERENCE_PATH),
        "audio": file_facts(AUDIO_PATH),
        "source_video": file_facts(SOURCE_VIDEO_PATH),
    }
    write_progress(mode, "input_download_done", {"input_files": input_files})

    command = wan27_command()
    report["wan27_command_configured"] = bool(command)
    report["wan27_command_redacted"] = command if command else ""
    if not command:
        report.update(
            {
                "runtime_probe_status": "wan27_command_not_configured",
                "torch_probe": torch_result,
                "ffmpeg_exists": ffmpeg_exists,
                "input_files": input_files,
                "wan27_run_status": "not_attempted",
                "output_upload_status": "not_attempted",
            }
        )
        return report

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_progress(mode, "wan27_run_started", {"wan27_command_configured": True})
    result = run_wan27_command(command)
    write_progress(mode, "wan27_run_done", {"wan27_result": result})
    if result["returncode"] != 0:
        report.update(
            {
                "runtime_probe_status": "wan27_run_failed",
                "torch_probe": torch_result,
                "ffmpeg_exists": ffmpeg_exists,
                "input_files": input_files,
                "wan27_result": result,
                "output_upload_status": "not_attempted",
            }
        )
        return report
    if not OUTPUT_PATH.exists():
        report.update(
            {
                "runtime_probe_status": "wan27_output_missing",
                "torch_probe": torch_result,
                "ffmpeg_exists": ffmpeg_exists,
                "input_files": input_files,
                "wan27_result": result,
                "output_upload_status": "not_attempted",
            }
        )
        return report

    upload_r2_file(OUTPUT_PATH, keys["output_video"])
    output_file = file_facts(OUTPUT_PATH)
    write_progress(mode, "output_upload_done", {"r2_output_video_key": keys["output_video"], "output_file": output_file})
    report.update(
        {
            "runtime_probe_status": "ok",
            "torch_probe": torch_result,
            "ffmpeg_exists": ffmpeg_exists,
            "input_files": input_files,
            "wan27_result": result,
            "wan27_run_status": "ok",
            "output_file": output_file,
            "r2_output_video_key": keys["output_video"],
            "output_upload_status": "ok",
        }
    )
    return report


def run(mode: str) -> int:
    print(f"[AYL_WAN27_RUNTIME_PROBE] start mode={mode}", flush=True)
    write_progress(mode, "container_started")
    final_key = os.getenv("R2_FINAL_REPORT_KEY", DEFAULT_FINAL_KEY)
    try:
        if mode != "wan27_probe":
            raise RuntimeError(f"Unsupported runtime probe mode: {mode}")
        report = build_wan27_report(mode)
    except Exception as exc:
        report = base_report(mode)
        report.update({"runtime_probe_status": "failed", "error_truncated": str(exc)[:2000]})
    report["r2_progress_key"] = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    report["r2_final_report_key"] = final_key
    report["r2_upload_status"] = "ok"
    upload_json(final_key, report)
    write_progress(mode, "final_report_written", {"r2_final_report_key": final_key})
    status = report.get("runtime_probe_status", "ok")
    print(f"[AYL_WAN27_RUNTIME_PROBE] done mode={mode} status={status}", flush=True)
    return 0 if status == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AYL WAN 2.7 RunPod runtime probe.")
    parser.add_argument("--mode", choices=("wan27_probe",), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(args.mode)
    except Exception as exc:
        print(f"[AYL_WAN27_RUNTIME_PROBE] error={str(exc)[:300]}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
