import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .r2_client import download_file, get_r2_client, head_object, r2_env_alias_presence, r2_env_ready, resolved_r2_env, upload_file
from .reporting import now_iso
from .settings import get_settings


OUTPUT_TRUNCATE_CHARS = 4000
WORK_ROOT = Path("/tmp/ayl_wan22_s2v_jobs")
OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def truncate_output(value: str) -> str:
    if len(value) <= OUTPUT_TRUNCATE_CHARS:
        return value
    return value[-OUTPUT_TRUNCATE_CHARS:]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gpu_snapshot() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {"status": "failed", "error_type": type(exc).__name__, "error_truncated": str(exc)[:500]}
    if completed.returncode != 0:
        return {"status": "failed", "stderr_truncated": truncate_output(completed.stderr or "")}
    line = (completed.stdout or "").strip().splitlines()[0] if completed.stdout else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {"status": "failed", "stdout_truncated": truncate_output(completed.stdout or "")}
    try:
        total_mib = float(parts[1])
        used_mib = float(parts[2])
        util = float(parts[3])
    except ValueError:
        total_mib = used_mib = util = None
    return {
        "status": "succeeded",
        "gpuModel": parts[0],
        "memory_total_mib": total_mib,
        "memory_total_gib": round(total_mib / 1024.0, 3) if total_mib is not None else None,
        "memory_used_mib": used_mib,
        "memory_used_gib": round(used_mib / 1024.0, 3) if used_mib is not None else None,
        "utilization_gpu_percent": util,
    }


class GpuMonitor:
    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = gpu_snapshot()
            sample["timestamp"] = now_iso()
            self.samples.append(sample)
            self._stop.wait(self.interval_seconds)

    def summary(self) -> dict:
        usable = [sample for sample in self.samples if sample.get("status") == "succeeded"]
        used_values = [sample.get("memory_used_gib") for sample in usable if sample.get("memory_used_gib") is not None]
        util_values = [sample.get("utilization_gpu_percent") for sample in usable if sample.get("utilization_gpu_percent") is not None]
        total_values = [sample.get("memory_total_gib") for sample in usable if sample.get("memory_total_gib") is not None]
        return {
            "samples": len(self.samples),
            "gpuModel": usable[-1].get("gpuModel", "") if usable else "",
            "runtime_vram_total_gib": total_values[-1] if total_values else None,
            "peak_vram_gb": max(used_values) if used_values else None,
            "avg_gpu_util": round(sum(util_values) / len(util_values), 3) if util_values else None,
        }


def is_oom_result(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def build_command(image_path: Path, audio_path: Path, output_path: Path, model_dir: Path, width: int, height: int) -> list[str]:
    return [
        "python",
        "-m",
        "app.wan22_s2v_generate_wrapper",
        "--task",
        "s2v-14B",
        "--size",
        f"{width}*{height}",
        "--ckpt_dir",
        str(model_dir),
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--prompt",
        "A natural, stable talking-head lip sync video of Maé speaking French.",
        "--image",
        str(image_path),
        "--audio",
        str(audio_path),
        "--save_file",
        str(output_path),
    ]


def run_command(command: list[str], timeout_seconds: int) -> dict:
    monitor = GpuMonitor()
    started = time.monotonic()
    monitor.start()
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd="/opt/ayl-simplepod-wan22-s2v-fastapi-v2",
            env={**os.environ, "PYTHONPATH": "/opt/Wan2.2:/opt/ayl-simplepod-wan22-s2v-fastapi-v2"},
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        status = "succeeded" if completed.returncode == 0 else "oom" if is_oom_result(stdout, stderr) else "failed"
        return {
            "status": status,
            "returncode": completed.returncode,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "stdout_truncated": truncate_output(stdout),
            "stderr_truncated": truncate_output(stderr),
            "telemetry": monitor.summary(),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return {
            "status": "timeout",
            "returncode": None,
            "runtime_seconds": round(time.monotonic() - started, 3),
            "stdout_truncated": truncate_output(stdout),
            "stderr_truncated": truncate_output(stderr),
            "telemetry": monitor.summary(),
        }
    finally:
        monitor.stop()


def file_facts(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def safe_head_object(key: str) -> dict:
    try:
        return head_object(key)
    except Exception as exc:
        return {
            "status": "failed",
            "key": key,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
        }


def safe_upload_permission_check(job_id: str, output_report_key: str) -> dict:
    if not r2_env_ready():
        return {"status": "skipped_missing_r2_env"}
    check_key = f"{output_report_key}.preflight_write_check_{job_id}.json"
    body = json.dumps({"job_id": job_id, "purpose": "preflight_write_check"}).encode("utf-8")
    resolved = resolved_r2_env()
    try:
        client = get_r2_client()
        client.put_object(Bucket=resolved["bucket"], Key=check_key, Body=body, ContentType="application/json")
        delete_status = "not_attempted"
        try:
            client.delete_object(Bucket=resolved["bucket"], Key=check_key)
            delete_status = "succeeded"
        except Exception:
            delete_status = "failed_non_blocking"
        return {"status": "succeeded", "key": check_key, "delete_status": delete_status}
    except Exception as exc:
        return {
            "status": "failed",
            "key": check_key,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
        }


def r2_preflight(payload: dict[str, Any]) -> dict:
    env_status = "succeeded" if r2_env_ready() else "missing_env"
    result = {
        "r2_env_check_status": env_status,
        "r2_env_present_redacted": r2_env_alias_presence(),
        "r2_reference_head_status": safe_head_object(payload["reference_image_key"]) if env_status == "succeeded" else {"status": "skipped_missing_r2_env"},
        "r2_audio_head_status": safe_head_object(payload["audio_key"]) if env_status == "succeeded" else {"status": "skipped_missing_r2_env"},
        "r2_upload_permission_check_status": (
            safe_upload_permission_check(str(payload["job_id"]), payload["output_report_key"])
            if env_status == "succeeded"
            else {"status": "skipped_missing_r2_env"}
        ),
    }
    checks = (
        result["r2_env_check_status"] == "succeeded",
        result["r2_reference_head_status"].get("status") == "succeeded",
        result["r2_audio_head_status"].get("status") == "succeeded",
        result["r2_upload_permission_check_status"].get("status") == "succeeded",
    )
    result["status"] = "succeeded" if all(checks) else "failed"
    return result


def run_wan22_s2v_single_job(payload: dict[str, Any]) -> dict:
    settings = get_settings()
    job_id = str(payload["job_id"])
    work_dir = WORK_ROOT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    input_image = work_dir / "reference.png"
    input_audio = work_dir / "audio.wav"
    output_1080 = work_dir / f"{job_id}_1080x1080.mp4"
    output_960 = work_dir / f"{job_id}_960x960.mp4"
    local_report_path = work_dir / "final_report.json"

    started = time.monotonic()
    report: dict[str, Any] = {
        "job_id": job_id,
        "status": "started",
        "created_at": now_iso(),
        "reference_image_key": payload["reference_image_key"],
        "audio_key": payload["audio_key"],
        "requested_resolution": {"width": payload["target_width"], "height": payload["target_height"]},
        "actual_generation_resolution": None,
        "fallback_resolution": {"width": 960, "height": 960},
        "fallback_used": False,
        "fallback_allowed": bool(payload.get("allow_oom_fallback", False)),
        "fps": payload["fps"],
        "target_duration_seconds": payload["target_duration_seconds"],
        "output_video_key": payload["output_video_key"],
        "output_report_key": payload["output_report_key"],
        "r2_env_present_redacted": r2_env_alias_presence(),
        "r2_client_configured": r2_env_ready(),
        "model_dir": str(settings.wan22_s2v_model_dir),
        "downloads_model_weights": False,
        "placeholder_generated": False,
        "video_generated": False,
        "r2_upload_attempted": False,
    }

    try:
        preflight = r2_preflight(payload)
        report["r2_preflight"] = preflight
        report["r2_env_check_status"] = preflight["r2_env_check_status"]
        report["r2_reference_head_status"] = preflight["r2_reference_head_status"]
        report["r2_audio_head_status"] = preflight["r2_audio_head_status"]
        report["r2_upload_permission_check_status"] = preflight["r2_upload_permission_check_status"]
        report["r2_upload_permission_check_status"] = preflight["r2_upload_permission_check_status"]
        if preflight["status"] != "succeeded":
            report["status"] = "failed_r2_preflight"
            report["error_type"] = "r2_preflight_failed"
            report["runtime_seconds"] = round(time.monotonic() - started, 3)
            write_json(local_report_path, report)
            return report

        download_file(payload["reference_image_key"], input_image)
        download_file(payload["audio_key"], input_audio)
        report["input_files"] = {
            "reference_image": file_facts(input_image),
            "audio": file_facts(input_audio),
        }

        primary_command = build_command(
            input_image,
            input_audio,
            output_1080,
            settings.wan22_s2v_model_dir,
            int(payload["target_width"]),
            int(payload["target_height"]),
        )
        primary_result = run_command(primary_command, int(payload.get("timeout_seconds") or 7200))
        report["primary_inference"] = primary_result
        report["command"] = primary_command
        report.update(primary_result.get("telemetry", {}))

        selected_output = output_1080
        if primary_result["status"] == "succeeded" and output_1080.exists():
            report["status"] = "succeeded"
            report["actual_generation_resolution"] = {"width": payload["target_width"], "height": payload["target_height"]}
        elif primary_result["status"] == "oom" and payload.get("allow_oom_fallback"):
            fallback_command = build_command(input_image, input_audio, output_960, settings.wan22_s2v_model_dir, 960, 960)
            fallback_result = run_command(fallback_command, int(payload.get("timeout_seconds") or 7200))
            report["fallback_inference"] = fallback_result
            report["fallback_command"] = fallback_command
            selected_output = output_960
            report["fallback_used"] = fallback_result["status"] == "succeeded" and output_960.exists()
            if report["fallback_used"]:
                report["status"] = "succeeded_with_960_fallback"
                report["actual_generation_resolution"] = {"width": 960, "height": 960}
                report.update(fallback_result.get("telemetry", {}))
            else:
                report["status"] = "failed_oom_fallback_failed"
        elif primary_result["status"] == "oom":
            report["status"] = "failed_oom_1080_no_fallback"
            report["error_type"] = "cuda_oom"
        else:
            report["status"] = "failed_inference"
            report["error_type"] = primary_result["status"]

        if report["status"] in {"succeeded", "succeeded_with_960_fallback"} and selected_output.exists():
            upload_file(selected_output, payload["output_video_key"])
            report["video_generated"] = True
            report["r2_upload_attempted"] = True
            report["output_file"] = file_facts(selected_output)
        report["runtime_seconds"] = round(time.monotonic() - started, 3)
        report["estimated_cost"] = None
        write_json(local_report_path, report)
        upload_file(local_report_path, payload["output_report_key"])
        report["report_uploaded_to_r2"] = True
        return report
    except Exception as exc:
        report["status"] = "failed_exception"
        report["error_type"] = type(exc).__name__
        report["error_truncated"] = str(exc)[:1000]
        report["runtime_seconds"] = round(time.monotonic() - started, 3)
        write_json(local_report_path, report)
        if r2_env_ready():
            try:
                upload_file(local_report_path, payload["output_report_key"])
                report["report_uploaded_to_r2"] = True
            except Exception as upload_exc:
                report["report_upload_error_type"] = type(upload_exc).__name__
                report["report_upload_error_truncated"] = str(upload_exc)[:1000]
        return report
