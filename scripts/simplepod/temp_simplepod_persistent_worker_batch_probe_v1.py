import argparse
import json
import mimetypes
import os
import time
import wave
from pathlib import Path

import simplepod_wan22_s2v_runtime_base as base
import temp_simplepod_persistent_worker_multi_job_probe_v1 as multi
import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


SCRIPT_ID = "TEMP_SIMPLEPOD_PERSISTENT_WORKER_BATCH_PROBE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_persistent_worker_batch_probe_v1.json"

IMAGE_TAG = multi.IMAGE_TAG
IMAGE_REF = multi.IMAGE_REF
TEMPLATE_ID = multi.TEMPLATE_ID
LOCAL_OUTPUT_DIR = REPO_ROOT / "data" / "character_cast" / "persistent_worker_batch_probe_v1" / "outputs"

INPUT_PREFIX = "tests/simplepod_persistent_worker_batch_probe_v1/inputs"
OUTPUT_PREFIX = "tests/simplepod_persistent_worker_batch_probe_v1/outputs"

LOAD_PROBE_ENDPOINT = multi.LOAD_PROBE_ENDPOINT
STATUS_ENDPOINT = multi.STATUS_ENDPOINT
RUN_JOB_ENDPOINT = multi.RUN_JOB_ENDPOINT
UNLOAD_ENDPOINT = multi.UNLOAD_ENDPOINT


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def file_summary(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def wav_duration_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return round(handle.getnframes() / float(handle.getframerate()), 3)
    except Exception:
        return None


def default_manifest() -> dict:
    jobs = [
        {
            "name": "Alex",
            "character_id": "alex",
            "base_taught_language": "EN",
            "job_id": "alex_en_persistent_batch_probe_v1",
            "local_image_path": "data/character_cast/parallel_round1/alex/reference/alex_lipsync_optimized_reference.png",
            "local_audio_path": "data/character_cast/parallel_round1/alex/audio/alex_en_cast_voice_v1_wan15s.wav",
        },
        {
            "name": "Sofi",
            "character_id": "sofi",
            "base_taught_language": "ES",
            "job_id": "sofi_es_persistent_batch_probe_v1",
            "local_image_path": "data/character_cast/parallel_round1/sofi/reference/sofi_lipsync_optimized_reference.png",
            "local_audio_path": "data/character_cast/parallel_round1/sofi/audio/sofi_es_cast_voice_v1_wan15s.wav",
        },
        {
            "name": "Fernando",
            "character_id": "fernando",
            "base_taught_language": "PT",
            "job_id": "fernando_pt_persistent_batch_probe_v1",
            "local_image_path": "data/character_cast/parallel_round1/fernando/reference/fernando_lipsync_optimized_reference.png",
            "local_audio_path": "data/character_cast/voice_profile_v1/audio/fernando_pt_cast_voice_v1.wav",
        },
        {
            "name": "Maé",
            "character_id": "mae",
            "base_taught_language": "FR",
            "job_id": "mae_fr_persistent_batch_probe_v1",
            "local_image_path": "data/character_cast/parallel_round1/mae/reference/mae_lipsync_optimized_reference.png",
            "local_audio_path": "data/character_cast/voice_profile_v1/audio/mae_fr_cast_voice_v1.wav",
        },
        {
            "name": "Luca",
            "character_id": "luca",
            "base_taught_language": "IT",
            "job_id": "luca_it_persistent_batch_probe_v1",
            "local_image_path": "data/character_cast/parallel_round1/luca/reference/luca_lipsync_optimized_reference.png",
            "local_audio_path": "data/character_cast/voice_profile_v1/audio/luca_it_cast_voice_v1.wav",
        },
    ]
    enriched = []
    for job in jobs:
        character_id = job["character_id"]
        image_name = Path(job["local_image_path"]).name
        audio_name = Path(job["local_audio_path"]).name
        output_stem = f"{job['job_id']}_720x720"
        enriched.append(
            {
                **job,
                "width": 720,
                "height": 720,
                "fps": 16,
                "target_duration_seconds": wav_duration_seconds(repo_path(job["local_audio_path"])) or 15.0,
                "reference_image_key": f"{INPUT_PREFIX}/{character_id}/reference/{image_name}",
                "audio_key": f"{INPUT_PREFIX}/{character_id}/audio/{audio_name}",
                "output_video_key": f"{OUTPUT_PREFIX}/{output_stem}.mp4",
                "output_report_key": f"{OUTPUT_PREFIX}/{output_stem}_final_report.json",
                "seed": 42,
                "steps": 5,
                "cfg": 1.0,
                "shift": 4.0,
                "offload_model": True,
                "convert_model_dtype": True,
                "t5_cpu": False,
                "max_concurrent_jobs": 1,
            }
        )
    return {
        "manifest_id": "simplepod_persistent_worker_batch_probe_v1",
        "description": "Five sequential character-cast jobs for one persistent Wan2.2 S2V worker.",
        "image_ref": IMAGE_REF,
        "template_id": TEMPLATE_ID,
        "jobs": enriched,
    }


def load_manifest(path_text: str) -> dict:
    if not path_text:
        return default_manifest()
    path = repo_path(path_text)
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_jobs(manifest: dict) -> list[dict]:
    jobs = manifest.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("Manifest field 'jobs' must be a list.")
    return jobs


def validate_manifest(manifest: dict) -> dict:
    missing = []
    local_inputs = {}
    for index, job in enumerate(manifest_jobs(manifest), start=1):
        image_path = repo_path(str(job.get("local_image_path", "")))
        audio_path = repo_path(str(job.get("local_audio_path", "")))
        local_inputs[str(index)] = {
            "name": job.get("name", ""),
            "character_id": job.get("character_id", ""),
            "reference_image": file_summary(image_path),
            "audio": file_summary(audio_path),
            "audio_duration_seconds": wav_duration_seconds(audio_path),
        }
        for path in (image_path, audio_path):
            if not path.exists() or not path.is_file():
                missing.append(str(path))
    return {
        "status": "passed" if not missing else "failed_missing_local_inputs",
        "missing_inputs": missing,
        "local_inputs": local_inputs,
    }


def execute_allowed(args: argparse.Namespace) -> bool:
    return (
        args.execute
        and args.confirm_start
        and args.confirm_load_probe
        and args.confirm_inference
        and args.confirm_unload
        and args.confirm_delete
    )


def blocked_status(args: argparse.Namespace, manifest: dict) -> str:
    if not manifest_jobs(manifest):
        return "blocked_empty_manifest"
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_load_probe:
        return "blocked_missing_confirm_load_probe"
    if args.execute and not args.confirm_inference:
        return "blocked_missing_confirm_inference"
    if args.execute and not args.confirm_unload:
        return "blocked_missing_confirm_unload"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def r2_transfer_file(local_path: Path, key: str, *, upload: bool, dry_run: bool) -> dict:
    action = "upload" if upload else "download"
    result = {
        "action": action,
        "attempted": not dry_run,
        "dry_run": dry_run,
        "local_path": str(local_path),
        "key": key,
        "status": "planned" if dry_run else "pending",
    }
    if upload:
        result.update(file_summary(local_path))
    if dry_run:
        return result
    config = base.r2_client_config()
    missing = [name for name, value in config.items() if name != "region" and not value]
    if missing:
        return {**result, "status": "failed_missing_r2_env", "missing_config": missing}
    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint_url"],
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
            region_name=config["region"] or "auto",
        )
        if upload:
            if not local_path.exists() or not local_path.is_file():
                return {**result, "status": "failed_missing_local_file"}
            extra_args = {}
            content_type, _ = mimetypes.guess_type(local_path.name)
            if content_type:
                extra_args["ContentType"] = content_type
            client.upload_file(str(local_path), config["bucket"], key, ExtraArgs=extra_args or None)
            return {**result, "status": "succeeded", "size_bytes": local_path.stat().st_size}
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(config["bucket"], key, str(local_path))
        return {**result, "status": "succeeded", **file_summary(local_path)}
    except Exception as exc:
        return {
            **result,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


def r2_object_exists(key: str) -> dict:
    result = {
        "attempted": True,
        "key": key,
        "exists": False,
        "status": "pending",
        "size_bytes": None,
    }
    config = base.r2_client_config()
    missing = [name for name, value in config.items() if name != "region" and not value]
    if missing:
        return {**result, "status": "failed_missing_r2_env", "missing_config": missing}
    try:
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=config["endpoint_url"],
            aws_access_key_id=config["access_key_id"],
            aws_secret_access_key=config["secret_access_key"],
            region_name=config["region"] or "auto",
        )
        response = client.head_object(Bucket=config["bucket"], Key=key)
        return {
            **result,
            "exists": True,
            "status": "found",
            "size_bytes": response.get("ContentLength"),
            "etag_present": bool(response.get("ETag")),
        }
    except Exception as exc:
        status_code = ""
        try:
            status_code = str(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", ""))
        except Exception:
            status_code = ""
        if status_code in {"403", "404"}:
            return {
                **result,
                "exists": False,
                "status": "not_found_or_forbidden",
                "http_status_code": status_code,
            }
        return {
            **result,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
            "http_status_code": status_code,
        }


def runtime_payload(instance_market: str) -> dict:
    payload = multi.runtime_payload(instance_market)
    for item in payload.get("envVariables", []):
        if item.get("name") == "AYL_RUNTIME_VERSION":
            item["value"] = "v2-blackwell-persistent-batch-probe-v1"
    return payload


def redact_instance_payload(payload: dict) -> dict:
    redacted = smoke.redact_value("", payload)
    for item in redacted.get("envVariables", []):
        if item.get("name") in base.LOCAL_R2_ENV_KEYS and item.get("value"):
            item["value"] = "<present_redacted>"
    return redacted


def job_payload(job: dict) -> dict:
    width = int(job.get("width", 720))
    height = int(job.get("height", 720))
    return {
        "job_id": str(job["job_id"]),
        "character_id": str(job["character_id"]),
        "base_taught_language": str(job["base_taught_language"]),
        "reference_image_key": str(job["reference_image_key"]),
        "audio_key": str(job["audio_key"]),
        "width": width,
        "height": height,
        "target_width": width,
        "target_height": height,
        "resolution": f"{width}x{height}",
        "fps": int(job.get("fps", 16)),
        "target_duration_seconds": float(job.get("target_duration_seconds", 15.0)),
        "output_video_key": str(job["output_video_key"]),
        "output_report_key": str(job["output_report_key"]),
        "confirm_inference": "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL",
        "allow_oom_fallback": False,
        "seed": int(job.get("seed", 42)),
        "steps": int(job.get("steps", 5)),
        "cfg": float(job.get("cfg", 1.0)),
        "shift": float(job.get("shift", 4.0)),
        "offload_model": bool(job.get("offload_model", True)),
        "convert_model_dtype": bool(job.get("convert_model_dtype", True)),
        "t5_cpu": bool(job.get("t5_cpu", False)),
        "max_concurrent_jobs": 1,
    }


def local_output_paths(job: dict) -> dict:
    payload = job_payload(job)
    return {
        "video": LOCAL_OUTPUT_DIR / Path(payload["output_video_key"]).name,
        "report": LOCAL_OUTPUT_DIR / Path(payload["output_report_key"]).name,
    }


def compact_http_result(result: dict) -> dict:
    return multi.compact_http_result(result)


def planned_jobs(manifest: dict) -> dict:
    return {
        str(index): {
            "name": job.get("name", ""),
            "payload": job_payload(job),
            "local_image_path": str(repo_path(str(job.get("local_image_path", "")))),
            "local_audio_path": str(repo_path(str(job.get("local_audio_path", "")))),
        }
        for index, job in enumerate(manifest_jobs(manifest), start=1)
    }


def planned_uploads(manifest: dict) -> dict:
    uploads = {}
    for index, job in enumerate(manifest_jobs(manifest), start=1):
        payload = job_payload(job)
        uploads[str(index)] = {
            "reference_image": r2_transfer_file(
                repo_path(str(job["local_image_path"])),
                payload["reference_image_key"],
                upload=True,
                dry_run=True,
            ),
            "audio": r2_transfer_file(
                repo_path(str(job["local_audio_path"])),
                payload["audio_key"],
                upload=True,
                dry_run=True,
            ),
        }
    return uploads


def planned_downloads(manifest: dict) -> dict:
    downloads = {}
    for index, job in enumerate(manifest_jobs(manifest), start=1):
        payload = job_payload(job)
        paths = local_output_paths(job)
        downloads[str(index)] = {
            "video": r2_transfer_file(paths["video"], payload["output_video_key"], upload=False, dry_run=True),
            "report": r2_transfer_file(paths["report"], payload["output_report_key"], upload=False, dry_run=True),
        }
    return downloads


def numeric_values(job_results: list[dict], key: str) -> list[float]:
    values = []
    for result in job_results:
        value = result.get(key)
        try:
            if value is not None:
                values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def stats(values: list[float]) -> dict:
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "avg": round(sum(values) / len(values), 3),
    }


def per_job_statistics(data: dict) -> dict:
    output = {}
    for index, record in sorted(data.get("jobs_by_index", {}).items(), key=lambda item: int(item[0])):
        result = record.get("result", {}).get("json", {})
        status_after = record.get("status_after", {}).get("json", {})
        output[index] = {
            "job_id": result.get("job_id") or record.get("payload", {}).get("job_id"),
            "character_id": record.get("payload", {}).get("character_id"),
            "status": result.get("status"),
            "runtime_seconds": result.get("runtime_seconds"),
            "generation_seconds": result.get("generation_seconds"),
            "save_merge_seconds": result.get("save_merge_seconds"),
            "peak_vram_gb": result.get("peak_vram_gb"),
            "resident_vram_reserved_gb": result.get("resident_vram_reserved_gb"),
            "load_count_before": result.get("load_count_before"),
            "load_count_after": result.get("load_count_after"),
            "jobs_completed_after_status": status_after.get("jobs_completed"),
            "recycle_required": result.get("recycle_required"),
            "http_response_received": record.get("http_response_received"),
            "client_timeout_occurred": record.get("client_timeout_occurred"),
            "recovered_from_worker_status": record.get("recovered_from_worker_status"),
            "recovered_from_r2_outputs": record.get("recovered_from_r2_outputs"),
            "recovery_seconds": record.get("recovery_seconds"),
            "original_error_type": record.get("original_error_type"),
            "safety_gate": record.get("safety_gate", {}),
        }
    return output


def report_summary(manifest: dict, data: dict) -> dict:
    job_records = data.get("jobs_by_index", {})
    job_results = [
        record.get("result", {}).get("json", {})
        for _, record in sorted(job_records.items(), key=lambda item: int(item[0]))
        if isinstance(record.get("result", {}).get("json"), dict)
    ]
    succeeded = [
        result
        for result in job_results
        if result.get("status") in {"succeeded", "succeeded_recovered_from_status"}
    ]
    load_counts = [
        result.get("load_count_after")
        for result in job_results
        if result.get("load_count_after") is not None
    ]
    reloads = 0
    for result in succeeded:
        if result.get("status") == "succeeded_recovered_from_status":
            if result.get("load_count_after") != 1:
                reloads += 1
        elif result.get("load_count_before") != result.get("load_count_after"):
            reloads += 1
    return {
        "jobs_requested": len(manifest_jobs(manifest)),
        "jobs_completed": len(succeeded),
        "worker_reloads": reloads,
        "recycles": sum(1 for result in job_results if result.get("recycle_required") is True),
        "load_count": 1 if set(load_counts) == {1} or not load_counts else sorted(set(load_counts)),
        "peak_vram_gb_max": round(max(numeric_values(job_results, "peak_vram_gb")), 3) if numeric_values(job_results, "peak_vram_gb") else None,
        "resident_vram_gb": round(max(numeric_values(job_results, "resident_vram_reserved_gb")), 3) if numeric_values(job_results, "resident_vram_reserved_gb") else None,
        "generation_seconds": stats(numeric_values(succeeded, "generation_seconds")),
    }


def safety_gate(job_index: int, job_result: dict, status_json: dict) -> tuple[bool, list[str]]:
    reasons = []
    recovered = isinstance(job_result, dict) and job_result.get("status") == "succeeded_recovered_from_status"
    if not isinstance(job_result, dict) or job_result.get("status") not in {"succeeded", "succeeded_recovered_from_status"}:
        reasons.append("job_not_succeeded")
    if job_result.get("recycle_required") is True:
        reasons.append("job_requested_recycle")
    if not recovered and job_result.get("load_count_before") != job_result.get("load_count_after"):
        reasons.append("load_count_changed")
    if job_result.get("load_count_after") != 1:
        reasons.append("load_count_not_1")
    if isinstance(status_json, dict):
        if status_json.get("worker_state") != "ready":
            reasons.append("worker_state_not_ready")
        if status_json.get("load_count") != 1:
            reasons.append("status_load_count_not_1")
        if status_json.get("jobs_completed") != job_index:
            reasons.append("jobs_completed_mismatch")
        if status_json.get("last_error") is not None:
            reasons.append("last_error_not_null")
        if status_json.get("t5_cpu_effective") is not False:
            reasons.append("t5_cpu_effective_not_false")
        if not status_json.get("resident_model_objects"):
            reasons.append("resident_model_objects_missing")
    else:
        reasons.append("status_json_missing")
    return not reasons, reasons


def response_lost_or_timed_out(result: dict) -> bool:
    if result.get("status") == "timeout":
        return True
    if result.get("http_status_code") is not None:
        return False
    error_text = " ".join(
        str(result.get(key, ""))
        for key in ("status", "error_type", "error_truncated")
    ).lower()
    return any(
        marker in error_text
        for marker in (
            "timeout",
            "timed out",
            "urlerror",
            "connection reset",
            "connection aborted",
            "remote end closed",
            "broken pipe",
        )
    )


def recover_job_after_lost_response(proxy_url: str, index: int, payload: dict) -> dict:
    started = time.monotonic()
    status_result = base.simple_get(proxy_url + STATUS_ENDPOINT, timeout_seconds=30)
    status_json = status_result.get("json") if isinstance(status_result.get("json"), dict) else {}
    video_head = r2_object_exists(payload["output_video_key"])
    report_head = r2_object_exists(payload["output_report_key"])
    worker_ready = (
        isinstance(status_json, dict)
        and status_json.get("worker_state") == "ready"
        and status_json.get("current_job_id") is None
        and status_json.get("jobs_completed") == index
        and status_json.get("last_error") is None
    )
    r2_outputs_exist = bool(video_head.get("exists") and report_head.get("exists"))
    recovered = bool(worker_ready and r2_outputs_exist)
    recovered_json = {
        "job_id": payload["job_id"],
        "status": "succeeded_recovered_from_status" if recovered else "failed_recovery_checks",
        "output_video_key": payload["output_video_key"],
        "output_report_key": payload["output_report_key"],
        "load_count_after": status_json.get("load_count") if isinstance(status_json, dict) else None,
        "jobs_completed": status_json.get("jobs_completed") if isinstance(status_json, dict) else None,
        "recycle_required": False,
        "worker_state_after_job": status_json.get("worker_state") if isinstance(status_json, dict) else None,
        "video_generated": r2_outputs_exist,
        "report_uploaded_to_r2": bool(report_head.get("exists")),
        "recovered_from_worker_status": worker_ready,
        "recovered_from_r2_outputs": r2_outputs_exist,
        "recovery_seconds": round(time.monotonic() - started, 3),
    }
    return {
        "recovered": recovered,
        "status_result": compact_http_result(status_result),
        "status_json": status_json,
        "video_head": video_head,
        "report_head": report_head,
        "json": recovered_json,
        "recovery_seconds": recovered_json["recovery_seconds"],
    }


def delete_instance(base_url: str, api_key: str, instance_id: int | None, data: dict, timer: PhaseTimer) -> None:
    if instance_id is None:
        data["delete_result"] = {"attempted": False, "status": "skipped_no_instance_id"}
        return
    print(f"[{SCRIPT_ID}] cleanup_started pod_id={instance_id}", flush=True)
    with timer.phase("delete_instance"):
        result = smoke.http_request(
            base_url,
            smoke.DELETE_INSTANCE_PATH.format(id=instance_id),
            api_key,
            method="DELETE",
        )
    data["delete_result"] = compact_http_result(result)
    print(f"[{SCRIPT_ID}] cleanup_completed delete_status={data['delete_result'].get('status')}", flush=True)
    if data["delete_result"].get("status") != "succeeded":
        print(f"[{SCRIPT_ID}] DELETE FAILED - manual cleanup required instance_id={instance_id}", flush=True)


def build_report(args: argparse.Namespace, manifest: dict, status: str, data: dict, error: str = "") -> dict:
    dry_run = not execute_allowed(args)
    return {
        "script_id": SCRIPT_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "manifest_source": args.manifest or "embedded_default",
        "manifest_id": manifest.get("manifest_id", ""),
        "summary": report_summary(manifest, data),
        "per_job_statistics": per_job_statistics(data),
        "template_id": TEMPLATE_ID,
        "image_ref": IMAGE_REF,
        "image_tag": IMAGE_TAG,
        "instance_id": data.get("instance_id"),
        "public_api_base_url": data.get("public_api_base_url", ""),
        "selected_api_port_mapping": data.get("selected_api_port_mapping", {}),
        "manifest_validation": data.get("manifest_validation", {}),
        "r2_uploads": data.get("r2_uploads", planned_uploads(manifest)),
        "jobs": planned_jobs(manifest),
        "jobs_by_index": data.get("jobs_by_index", {}),
        "downloads": {
            "by_index": data.get("downloads_by_index", planned_downloads(manifest)),
            "local_output_dir": str(LOCAL_OUTPUT_DIR),
        },
        "market_selection": data.get("market_selection", {}),
        "start_result": data.get("start_result", {}),
        "health_result": data.get("health_result", {}),
        "load_probe_result": data.get("load_probe_result", {}),
        "status_after_load": data.get("status_after_load", {}),
        "unload_result": data.get("unload_result", {}),
        "delete_result": data.get("delete_result", {}),
        "runtime_seconds": data.get("runtime_seconds"),
        "instance_payload_dryrun": redact_instance_payload(runtime_payload(args.instance_market or "<selected_full_blackwell_96gb_market>")),
        "endpoints": {
            "health": "/health",
            "load_probe": LOAD_PROBE_ENDPOINT,
            "status": STATUS_ENDPOINT,
            "run_job": RUN_JOB_ENDPOINT,
            "unload": UNLOAD_ENDPOINT,
        },
        "safety_guards": {
            "starts_simplepod": bool(execute_allowed(args)),
            "uploads_r2": bool(execute_allowed(args)),
            "runs_inference": bool(execute_allowed(args)),
            "downloads_outputs": bool(execute_allowed(args)),
            "downloads_model_weights": False,
            "builds_image": False,
            "publishes_image": False,
            "worker_loaded_once": True,
            "max_concurrent_jobs": 1,
            "parallelism_enabled": False,
            "threads_created": False,
            "asyncio_used": False,
            "deletes_instance": bool(data.get("delete_result", {}).get("attempted")),
            "secrets_printed": False,
        },
        "phase_timings": data.get("phase_timings", []),
    }


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    data = {
        "delete_result": {"attempted": False, "status": "not_started"},
        "jobs_by_index": {},
        "downloads_by_index": planned_downloads(manifest),
    }
    timer = PhaseTimer()
    data["phase_timings"] = timer.phases
    started = time.monotonic()
    instance_id = None
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    status = "failed"
    error = ""
    try:
        dry_run = not execute_allowed(args)
        print(f"[{SCRIPT_ID}] START dry_run={str(dry_run).lower()} jobs={len(manifest_jobs(manifest))} image={IMAGE_REF}", flush=True)

        blocked = blocked_status(args, manifest)
        if blocked:
            status = blocked
            return 1

        with timer.phase("validate_local_inputs"):
            validation = validate_manifest(manifest)
            data["manifest_validation"] = validation
        if validation["status"] != "passed":
            status = validation["status"]
            return 1

        with timer.phase("load_auth_env"):
            base.load_local_env()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)
            data["r2_env_presence"] = base.local_r2_env_presence()

        if dry_run:
            status = "dry_run_ready"
            return 0

        if not api_key:
            status = "missing_api_key"
            return 1
        missing_r2 = base.missing_local_r2_env()
        if missing_r2:
            data["missing_r2_env"] = missing_r2
            status = "missing_r2_env"
            return 1

        with timer.phase("upload_inputs_to_r2"):
            uploads = {}
            for index, job in enumerate(manifest_jobs(manifest), start=1):
                payload = job_payload(job)
                uploads[str(index)] = {
                    "reference_image": r2_transfer_file(repo_path(str(job["local_image_path"])), payload["reference_image_key"], upload=True, dry_run=False),
                    "audio": r2_transfer_file(repo_path(str(job["local_audio_path"])), payload["audio_key"], upload=True, dry_run=False),
                }
            data["r2_uploads"] = uploads
        failed_uploads = [
            result
            for record in data["r2_uploads"].values()
            for result in record.values()
            if result.get("status") != "succeeded"
        ]
        if failed_uploads:
            status = "input_upload_failed"
            return 1

        with timer.phase("market_selection"):
            market = base.choose_market(args, base_url, api_key, data)
        if not market:
            status = "blocked_no_full_blackwell_96gb_market_selected"
            return 1

        print(f"[{SCRIPT_ID}] pod_creation_requested", flush=True)
        start_payload = runtime_payload(market)
        with timer.phase("start_instance"):
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=start_payload)
        data["start_result"] = compact_http_result(start_result)
        data["start_result"]["request_payload_redacted"] = redact_instance_payload(start_payload)
        instance_id = smoke.extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        print(f"[{SCRIPT_ID}] pod_id={instance_id}", flush=True)
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            return 1

        with timer.phase("wait_public_url"):
            proxy_url = base.wait_for_public_url(base_url, api_key, instance_id, args, data)
        if not proxy_url:
            status = "blocked_no_proxy_url_for_port_8000"
            return 1

        with timer.phase("wait_health"):
            readiness, attempts, _health_wait_result = smoke.wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
        data["api_readiness"] = {"status": readiness, "attempts": attempts}
        if readiness != "ready":
            status = "api_not_ready"
            return 1

        with timer.phase("health_check"):
            health_result = base.simple_get(proxy_url + "/health", timeout_seconds=30)
        data["health_result"] = compact_http_result(health_result)
        if health_result.get("http_status_code") != 200:
            status = "health_check_failed"
            return 1

        with timer.phase("persistent_worker_load_probe"):
            load_probe = base.simple_post(proxy_url + LOAD_PROBE_ENDPOINT, {}, timeout_seconds=args.load_probe_timeout_seconds)
        data["load_probe_result"] = compact_http_result(load_probe)
        if load_probe.get("http_status_code") != 200 or not isinstance(load_probe.get("json"), dict):
            status = "load_probe_failed"
            return 1
        load_json = load_probe["json"]
        if load_json.get("worker_state") != "ready" or load_json.get("load_count") != 1 or load_json.get("t5_cpu_effective") is not False:
            status = "load_probe_not_ready_or_wrong_effective_config"
            return 1

        with timer.phase("persistent_worker_status_after_load"):
            status_after_load = base.simple_get(proxy_url + STATUS_ENDPOINT, timeout_seconds=30)
        data["status_after_load"] = compact_http_result(status_after_load)
        if status_after_load.get("http_status_code") != 200:
            status = "status_after_load_failed"
            return 1

        for index, job in enumerate(manifest_jobs(manifest), start=1):
            payload = job_payload(job)
            print(f"[{SCRIPT_ID}] run_job index={index}/{len(manifest_jobs(manifest))} job_id={payload['job_id']}", flush=True)
            with timer.phase(f"persistent_worker_run_job_{index}"):
                job_result = base.simple_post(proxy_url + RUN_JOB_ENDPOINT, payload, timeout_seconds=args.job_timeout_seconds)
            http_response_received = job_result.get("http_status_code") is not None and isinstance(job_result.get("json"), dict)
            client_timeout_occurred = response_lost_or_timed_out(job_result)
            job_record = {
                "name": job.get("name", ""),
                "payload": payload,
                "result": compact_http_result(job_result),
                "http_response_received": http_response_received,
                "client_timeout_occurred": client_timeout_occurred,
                "recovered_from_worker_status": False,
                "recovered_from_r2_outputs": False,
                "recovery_seconds": None,
                "original_error_type": job_result.get("error_type", ""),
            }
            data["jobs_by_index"][str(index)] = job_record
            job_json = job_result.get("json") if isinstance(job_result.get("json"), dict) else {}
            if (
                (job_result.get("http_status_code") != 200 or job_json.get("status") != "succeeded")
                and client_timeout_occurred
            ):
                with timer.phase(f"persistent_worker_recover_job_{index}"):
                    recovery = recover_job_after_lost_response(proxy_url, index, payload)
                job_record["recovery"] = recovery
                job_record["recovered_from_worker_status"] = bool(recovery.get("json", {}).get("recovered_from_worker_status"))
                job_record["recovered_from_r2_outputs"] = bool(recovery.get("json", {}).get("recovered_from_r2_outputs"))
                job_record["recovery_seconds"] = recovery.get("recovery_seconds")
                if recovery.get("recovered"):
                    job_record["result"] = {
                        "status": "succeeded",
                        "http_status_code": None,
                        "json": recovery["json"],
                        "recovered_from_lost_http_response": True,
                    }
                    job_json = recovery["json"]
                    print(f"[{SCRIPT_ID}] recovered_job index={index} job_id={payload['job_id']}", flush=True)
                else:
                    status = f"job_{index}_failed_recovery_checks"
                    return 1
            if job_result.get("http_status_code") != 200 or job_json.get("status") != "succeeded":
                if job_json.get("status") != "succeeded_recovered_from_status":
                    status = f"job_{index}_failed"
                    return 1

            with timer.phase(f"persistent_worker_status_after_job_{index}"):
                status_after_job = base.simple_get(proxy_url + STATUS_ENDPOINT, timeout_seconds=30)
            status_json = status_after_job.get("json") if isinstance(status_after_job.get("json"), dict) else {}
            job_record["status_after"] = compact_http_result(status_after_job)
            can_continue, reasons = safety_gate(index, job_json, status_json)
            job_record["safety_gate"] = {
                "passed": can_continue,
                "reasons": reasons,
            }
            if not can_continue:
                status = f"aborted_after_job_{index}_safety_gate"
                return 1

        with timer.phase("download_outputs_from_r2"):
            downloads = {}
            for index, job in enumerate(manifest_jobs(manifest), start=1):
                payload = job_payload(job)
                paths = local_output_paths(job)
                downloads[str(index)] = {
                    "video": r2_transfer_file(paths["video"], payload["output_video_key"], upload=False, dry_run=False),
                    "report": r2_transfer_file(paths["report"], payload["output_report_key"], upload=False, dry_run=False),
                }
            data["downloads_by_index"] = downloads

        status = "succeeded"
        return 0
    except KeyboardInterrupt:
        status = "interrupted_delete_attempted"
        error = "KeyboardInterrupt"
        print(f"[{SCRIPT_ID}] INTERRUPTED cleanup_will_run=true", flush=True)
        return 130
    except Exception as exc:
        status = "failed"
        error = str(exc)
        print(f"[{SCRIPT_ID}] ERROR {error[:300]}", flush=True)
        return 1
    finally:
        if args.execute and args.confirm_unload and instance_id is not None and data.get("public_api_base_url"):
            with timer.phase("persistent_worker_unload"):
                unload = base.simple_post(data["public_api_base_url"] + UNLOAD_ENDPOINT, {}, timeout_seconds=args.unload_timeout_seconds)
            data["unload_result"] = compact_http_result(unload)
        if args.execute and args.confirm_delete and api_key:
            multi.delete_instance(base_url, api_key, instance_id, data, timer)
        elif instance_id is not None:
            data["manual_cleanup_required"] = True
        data["runtime_seconds"] = round(time.monotonic() - started, 3)
        write_json(REPORT_PATH, build_report(args, manifest, status, data, error))
        print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manifest batch through one persistent Wan2.2 S2V worker.")
    parser.add_argument("--manifest", default="", help="Path to batch manifest JSON. Defaults to the embedded five-character manifest.")
    parser.add_argument("--create-example-manifest", action="store_true", help="Write an example five-job manifest and exit.")
    parser.add_argument("--output", default="", help="Output path for --create-example-manifest.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and run the manifest.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-load-probe", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-unload", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=900)
    parser.add_argument("--load-probe-timeout-seconds", type=int, default=1800)
    parser.add_argument("--job-timeout-seconds", type=int, default=1800)
    parser.add_argument("--unload-timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.create_example_manifest:
        output = repo_path(args.output or "data/character_cast/persistent_worker_batch_probe_v1/example_manifest.json")
        write_json(output, default_manifest())
        print(f"[{SCRIPT_ID}] example_manifest_written path={output}", flush=True)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
