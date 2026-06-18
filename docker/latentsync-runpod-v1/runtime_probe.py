import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_PREFIX = "tests/runpod_latentsync_image_v1_1"
DEFAULT_PROGRESS_KEY = f"{TEST_PREFIX}/progress/container_started.json"
DEFAULT_FINAL_KEY = f"{TEST_PREFIX}/output/final_report.json"
DEFAULT_SMOKE_UNET_KEY = "checkpoints/latentsync/latentsync_unet.pt"
DEFAULT_SMOKE_WHISPER_KEY = "checkpoints/latentsync/whisper/tiny.pt"
DEFAULT_SMOKE_VIDEO_KEY = "tests/runpod_latentsync_smoke_run_0001/input/video.mp4"
DEFAULT_SMOKE_AUDIO_KEY = "tests/runpod_latentsync_smoke_run_0001/input/audio.wav"
DEFAULT_SMOKE_OUTPUT_KEY = "tests/runpod_latentsync_smoke_run_0001/output/video_out.mp4"
LATENTSYNC_ROOT = Path("/opt/LatentSync")
SMOKE_PATHS = {
    "unet": LATENTSYNC_ROOT / "checkpoints" / "latentsync_unet.pt",
    "whisper": LATENTSYNC_ROOT / "checkpoints" / "whisper" / "tiny.pt",
    "video": Path("/workspace/input/video.mp4"),
    "audio": Path("/workspace/input/audio.wav"),
    "output": Path("/workspace/output/video_out.mp4"),
}
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
        "R2_CHECKPOINT_UNET_KEY",
        "R2_CHECKPOINT_WHISPER_KEY",
        "R2_INPUT_VIDEO_KEY",
        "R2_INPUT_AUDIO_KEY",
        "R2_OUTPUT_VIDEO_KEY",
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
    smoke_mode = mode == "latentsync_smoke_run"
    return {
        "test_id": "TEST_RUNPOD_LATENTSYNC_SMOKE_RUN_0001" if smoke_mode else "TEST_RUNPOD_LATENTSYNC_IMAGE_V1_1_ENTRYPOINT_PROBE",
        "mode": mode,
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "image_tag": os.getenv("AYL_IMAGE_TAG", ""),
        "marker_nonce": os.getenv("AYL_MARKER_NONCE", ""),
        "env_present_redacted": env_presence(),
        "download_checkpoints": os.getenv("DOWNLOAD_CHECKPOINTS", "0"),
        "no_checkpoint_downloads": not smoke_mode,
        "no_inference": not smoke_mode,
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


def smoke_r2_keys() -> dict:
    return {
        "checkpoint_unet": os.getenv("R2_CHECKPOINT_UNET_KEY", DEFAULT_SMOKE_UNET_KEY),
        "checkpoint_whisper": os.getenv("R2_CHECKPOINT_WHISPER_KEY", DEFAULT_SMOKE_WHISPER_KEY),
        "input_video": os.getenv("R2_INPUT_VIDEO_KEY", DEFAULT_SMOKE_VIDEO_KEY),
        "input_audio": os.getenv("R2_INPUT_AUDIO_KEY", DEFAULT_SMOKE_AUDIO_KEY),
        "output_video": os.getenv("R2_OUTPUT_VIDEO_KEY", DEFAULT_SMOKE_OUTPUT_KEY),
    }


def smoke_inference_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "scripts.inference",
        "--unet_config_path",
        "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path",
        "checkpoints/latentsync_unet.pt",
        "--inference_steps",
        "20",
        "--guidance_scale",
        "1.5",
        "--enable_deepcache",
        "--video_path",
        str(SMOKE_PATHS["video"]),
        "--audio_path",
        str(SMOKE_PATHS["audio"]),
        "--video_out_path",
        str(SMOKE_PATHS["output"]),
    ]


def run_smoke_report(mode: str) -> dict:
    report = base_report(mode)
    r2_keys = smoke_r2_keys()
    command = smoke_inference_command()
    paths = {name: str(path) for name, path in SMOKE_PATHS.items()}

    report.update(
        {
            "probe_scope": "latentsync_functional_smoke_run",
            "r2_input_keys": r2_keys,
            "container_paths": paths,
            "inference_command": command,
            "latentsync_root": str(LATENTSYNC_ROOT),
        }
    )

    download_r2_file(r2_keys["checkpoint_unet"], SMOKE_PATHS["unet"])
    download_r2_file(r2_keys["checkpoint_whisper"], SMOKE_PATHS["whisper"])
    checkpoint_facts = {
        "unet": file_facts(SMOKE_PATHS["unet"]),
        "whisper": file_facts(SMOKE_PATHS["whisper"]),
    }
    write_progress(mode, "checkpoint_download_done", {"checkpoint_files": checkpoint_facts})

    download_r2_file(r2_keys["input_video"], SMOKE_PATHS["video"])
    download_r2_file(r2_keys["input_audio"], SMOKE_PATHS["audio"])
    input_facts = {
        "video": file_facts(SMOKE_PATHS["video"]),
        "audio": file_facts(SMOKE_PATHS["audio"]),
    }
    write_progress(mode, "input_download_done", {"input_files": input_facts})

    SMOKE_PATHS["output"].parent.mkdir(parents=True, exist_ok=True)
    write_progress(mode, "inference_started", {"inference_command": command})
    started_at = now_iso()
    completed = subprocess.run(
        command,
        cwd=str(LATENTSYNC_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=float(os.getenv("AYL_LATENTSYNC_SMOKE_INFERENCE_TIMEOUT_SECONDS", "900")),
        check=False,
    )
    finished_at = now_iso()
    inference_result = {
        "started_at": started_at,
        "finished_at": finished_at,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
        "output_file": file_facts(SMOKE_PATHS["output"]),
    }
    write_progress(mode, "inference_done", {"inference_result": inference_result})

    if completed.returncode != 0:
        report.update(
            {
                "runtime_probe_status": "inference_failed",
                "checkpoint_files": checkpoint_facts,
                "input_files": input_facts,
                "inference_result": inference_result,
                "output_upload_status": "not_attempted",
            }
        )
        return report

    if not SMOKE_PATHS["output"].exists():
        report.update(
            {
                "runtime_probe_status": "output_missing",
                "checkpoint_files": checkpoint_facts,
                "input_files": input_facts,
                "inference_result": inference_result,
                "output_upload_status": "not_attempted",
            }
        )
        return report

    upload_r2_file(SMOKE_PATHS["output"], r2_keys["output_video"])
    output_facts = file_facts(SMOKE_PATHS["output"])
    write_progress(
        mode,
        "output_upload_done",
        {
            "r2_output_video_key": r2_keys["output_video"],
            "output_file": output_facts,
        },
    )
    report.update(
        {
            "runtime_probe_status": "ok",
            "checkpoint_files": checkpoint_facts,
            "input_files": input_facts,
            "inference_result": inference_result,
            "output_file": output_facts,
            "r2_output_video_key": r2_keys["output_video"],
            "output_upload_status": "ok",
        }
    )
    return report


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
    elif mode == "latentsync_smoke_run":
        report = run_smoke_report(mode)
    else:
        raise RuntimeError(f"Unsupported runtime probe mode: {mode}")
    return report


def run(mode: str) -> int:
    print(f"[AYL_RUNTIME_PROBE] start mode={mode}", flush=True)
    write_progress(mode, "container_started")
    final_key = os.getenv("R2_FINAL_REPORT_KEY", DEFAULT_FINAL_KEY)
    try:
        report = build_report(mode)
    except Exception as exc:
        report = base_report(mode)
        report.update(
            {
                "runtime_probe_status": "failed",
                "error_truncated": str(exc)[:2000],
            }
        )
        if mode == "latentsync_smoke_run":
            report.update(
                {
                    "probe_scope": "latentsync_functional_smoke_run",
                    "r2_input_keys": smoke_r2_keys(),
                    "container_paths": {name: str(path) for name, path in SMOKE_PATHS.items()},
                    "inference_command": smoke_inference_command(),
                    "output_upload_status": "not_attempted",
                }
            )
    report["r2_progress_key"] = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    report["r2_final_report_key"] = final_key
    report["r2_upload_status"] = "ok"
    upload_json(final_key, report)
    write_progress(mode, "final_report_written", {"r2_final_report_key": final_key})
    status = report.get("runtime_probe_status", "ok")
    print(f"[AYL_RUNTIME_PROBE] done mode={mode} status={status}", flush=True)
    return 0 if status == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AYL LatentSync RunPod image runtime probe.")
    parser.add_argument("--mode", choices=("r2_probe", "latentsync_probe", "latentsync_smoke_run"), required=True)
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
