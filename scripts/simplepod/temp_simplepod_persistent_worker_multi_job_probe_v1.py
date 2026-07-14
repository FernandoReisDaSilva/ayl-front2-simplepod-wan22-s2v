import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

import simplepod_wan22_s2v_runtime_base as base
import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


SCRIPT_ID = "TEMP_SIMPLEPOD_PERSISTENT_WORKER_MULTI_JOB_PROBE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_persistent_worker_multi_job_probe_v1.json"

IMAGE_TAG = "0.2.29-blackwell-persistent-two-job-probe-v1"
IMAGE_REF = f"ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:{IMAGE_TAG}"
TEMPLATE_ID = 25138
PORT = 8000
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"

REFERENCE_IMAGE_LOCAL = REPO_ROOT / "data" / "character_cast" / "parallel_round1" / "mae" / "reference" / "mae_lipsync_optimized_reference.png"
AUDIO_LOCAL = REPO_ROOT / "data" / "character_cast" / "duration_test_v1" / "mae" / "audio" / "mae_fr_duration_test_A_15s.wav"
LOCAL_OUTPUT_DIR = REPO_ROOT / "data" / "character_cast" / "persistent_worker_probe_v1" / "mae" / "outputs"

INPUT_REFERENCE_KEY = "tests/simplepod_persistent_worker_probe_v1/inputs/mae/reference/mae_lipsync_optimized_reference.png"
INPUT_AUDIO_KEY = "tests/simplepod_persistent_worker_probe_v1/inputs/mae/audio/mae_fr_duration_test_A_15s.wav"
OUTPUT_PREFIX = "tests/simplepod_persistent_worker_probe_v1/outputs"

LOAD_PROBE_ENDPOINT = "/admin/persistent-worker/load-probe"
STATUS_ENDPOINT = "/admin/persistent-worker/status"
RUN_JOB_ENDPOINT = "/admin/persistent-worker/run-job"
UNLOAD_ENDPOINT = "/admin/persistent-worker/unload"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def execute_allowed(args: argparse.Namespace) -> bool:
    return (
        args.execute
        and args.confirm_start
        and args.confirm_load_probe
        and args.confirm_inference
        and args.confirm_unload
        and args.confirm_delete
    )


def blocked_status(args: argparse.Namespace) -> str:
    if args.jobs < 1:
        return "blocked_jobs_must_be_positive"
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


def file_summary(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def runtime_payload(instance_market: str) -> dict:
    payload = {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_full_blackwell_96gb_market>",
        "instanceTemplate": f"/instances/templates/{TEMPLATE_ID}",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
            {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
            {"name": "HF_HOME", "value": HF_HOME},
            {"name": "AYL_ENABLE_ADMIN_VERIFY", "value": "1"},
            {"name": "AYL_IMAGE_TAG", "value": IMAGE_TAG},
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-persistent-multi-job-probe-v1"},
            {"name": "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH", "value": "1"},
            {"name": "MAX_CONCURRENT_JOBS", "value": "1"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
            *base.r2_env_variables_for_instance(),
        ],
    }
    payload["envVariables"] = [item for item in payload["envVariables"] if item.get("value") not in ("", None)]
    return payload


def redact_instance_payload(payload: dict) -> dict:
    redacted = smoke.redact_value("", payload)
    for item in redacted.get("envVariables", []):
        if item.get("name") in base.LOCAL_R2_ENV_KEYS and item.get("value"):
            item["value"] = "<present_redacted>"
    return redacted


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


def job_payload(job_index: int) -> dict:
    job_id = f"mae_fr_persistent_probe_15s_job{job_index}_v1"
    stem = f"{job_id}_720x720"
    return {
        "job_id": job_id,
        "character_id": "mae",
        "base_taught_language": "FR",
        "reference_image_key": INPUT_REFERENCE_KEY,
        "audio_key": INPUT_AUDIO_KEY,
        "width": 720,
        "height": 720,
        "target_width": 720,
        "target_height": 720,
        "resolution": "720x720",
        "fps": 16,
        "target_duration_seconds": 15.0,
        "output_video_key": f"{OUTPUT_PREFIX}/{stem}.mp4",
        "output_report_key": f"{OUTPUT_PREFIX}/{stem}_final_report.json",
        "confirm_inference": "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL",
        "allow_oom_fallback": False,
        "seed": 42,
        "steps": 5,
        "cfg": 1.0,
        "shift": 4.0,
        "offload_model": True,
        "convert_model_dtype": True,
        "t5_cpu": False,
        "max_concurrent_jobs": 1,
    }


def local_output_paths(job_index: int) -> dict:
    payload = job_payload(job_index)
    return {
        "video": LOCAL_OUTPUT_DIR / Path(payload["output_video_key"]).name,
        "report": LOCAL_OUTPUT_DIR / Path(payload["output_report_key"]).name,
    }


def compact_http_result(result: dict) -> dict:
    keys = (
        "attempted",
        "status",
        "method",
        "path",
        "http_status_code",
        "endpoint_host",
        "error_type",
        "error_truncated",
        "response_body_truncated",
    )
    compact = {key: result.get(key) for key in keys if key in result}
    if isinstance(result.get("json"), dict):
        compact["json"] = result["json"]
    return compact


def planned_jobs(args: argparse.Namespace) -> dict:
    return {str(index): {"payload": job_payload(index)} for index in range(1, args.jobs + 1)}


def planned_downloads(args: argparse.Namespace) -> dict:
    downloads = {}
    for index in range(1, args.jobs + 1):
        payload = job_payload(index)
        paths = local_output_paths(index)
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


def generation_stats(job_results: list[dict]) -> dict:
    values = numeric_values(job_results, "generation_seconds")
    if not values:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "avg": round(sum(values) / len(values), 3),
    }


def report_summary(args: argparse.Namespace, data: dict) -> dict:
    job_records = data.get("jobs_by_index", {})
    job_results = [
        record.get("result", {}).get("json", {})
        for _, record in sorted(job_records.items(), key=lambda item: int(item[0]))
        if isinstance(record.get("result", {}).get("json"), dict)
    ]
    succeeded = [result for result in job_results if result.get("status") == "succeeded"]
    load_counts = {
        result.get("load_count_after")
        for result in job_results
        if result.get("load_count_after") is not None
    }
    peak_values = numeric_values(job_results, "peak_vram_gb")
    resident_values = numeric_values(job_results, "resident_vram_reserved_gb")
    return {
        "jobs_requested": args.jobs,
        "jobs_completed": len(succeeded),
        "load_count": 1 if load_counts == {1} or not load_counts else sorted(load_counts),
        "worker_reloads": 0 if all(result.get("load_count_before") == result.get("load_count_after") for result in succeeded) else None,
        "recycles": sum(1 for result in job_results if result.get("recycle_required") is True),
        "peak_vram_gb_max": round(max(peak_values), 3) if peak_values else None,
        "resident_vram_gb": round(max(resident_values), 3) if resident_values else None,
        "generation_seconds": generation_stats(succeeded),
    }


def safety_gate(job_index: int, job_result: dict, status_json: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not isinstance(job_result, dict) or job_result.get("status") != "succeeded":
        reasons.append("job_not_succeeded")
    if job_result.get("recycle_required") is True:
        reasons.append("job_requested_recycle")
    if job_result.get("load_count_before") != job_result.get("load_count_after"):
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
    error_text = json.dumps(job_result, ensure_ascii=False).lower()
    if "out of memory" in error_text or "cuda oom" in error_text:
        reasons.append("oom_seen")
    if "accelerate" in error_text and "hook" in error_text:
        reasons.append("accelerate_hook_error_seen")
    return not reasons, reasons


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


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    dry_run = not execute_allowed(args)
    return {
        "script_id": SCRIPT_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "summary": report_summary(args, data),
        "template_id": TEMPLATE_ID,
        "image_ref": IMAGE_REF,
        "image_tag": IMAGE_TAG,
        "instance_id": data.get("instance_id"),
        "public_api_base_url": data.get("public_api_base_url", ""),
        "selected_api_port_mapping": data.get("selected_api_port_mapping", {}),
        "local_inputs": {
            "reference_image": file_summary(REFERENCE_IMAGE_LOCAL),
            "audio": file_summary(AUDIO_LOCAL),
        },
        "r2_inputs": {
            "reference_image_key": INPUT_REFERENCE_KEY,
            "audio_key": INPUT_AUDIO_KEY,
            "upload_reference_result": data.get("upload_reference_result", r2_transfer_file(REFERENCE_IMAGE_LOCAL, INPUT_REFERENCE_KEY, upload=True, dry_run=True)),
            "upload_audio_result": data.get("upload_audio_result", r2_transfer_file(AUDIO_LOCAL, INPUT_AUDIO_KEY, upload=True, dry_run=True)),
        },
        "jobs": planned_jobs(args),
        "jobs_by_index": data.get("jobs_by_index", {}),
        "downloads": {
            "by_index": data.get("downloads_by_index", planned_downloads(args)),
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
        "success_conditions": {
            "load_count_must_remain_1": True,
            "worker_state_after_each_job": "ready",
            "abort_immediately_on_failed_safety_gate": True,
            "subprocess_runtime_preserved": True,
        },
        "safety_guards": {
            "starts_simplepod": bool(execute_allowed(args)),
            "uploads_r2": bool(execute_allowed(args)),
            "runs_inference": bool(execute_allowed(args)),
            "downloads_outputs": bool(execute_allowed(args)),
            "downloads_model_weights": False,
            "builds_image": False,
            "publishes_image": False,
            "startScript_sent": False,
            "uses_image_cmd": True,
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
    data = {
        "delete_result": {"attempted": False, "status": "not_started"},
        "jobs_by_index": {},
        "downloads_by_index": planned_downloads(args),
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
        print(f"[{SCRIPT_ID}] START dry_run={str(dry_run).lower()} jobs={args.jobs} image={IMAGE_REF}", flush=True)
        print(f"[{SCRIPT_ID}] local_reference={REFERENCE_IMAGE_LOCAL}", flush=True)
        print(f"[{SCRIPT_ID}] local_audio={AUDIO_LOCAL}", flush=True)

        blocked = blocked_status(args)
        if blocked:
            status = blocked
            return 1

        with timer.phase("validate_local_inputs"):
            missing_inputs = [
                str(path)
                for path in (REFERENCE_IMAGE_LOCAL, AUDIO_LOCAL)
                if not path.exists() or not path.is_file()
            ]
        if missing_inputs:
            data["missing_inputs"] = missing_inputs
            status = "blocked_missing_local_inputs"
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
            data["upload_reference_result"] = r2_transfer_file(REFERENCE_IMAGE_LOCAL, INPUT_REFERENCE_KEY, upload=True, dry_run=False)
            data["upload_audio_result"] = r2_transfer_file(AUDIO_LOCAL, INPUT_AUDIO_KEY, upload=True, dry_run=False)
        if data["upload_reference_result"].get("status") != "succeeded" or data["upload_audio_result"].get("status") != "succeeded":
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

        for index in range(1, args.jobs + 1):
            print(f"[{SCRIPT_ID}] run_job index={index}/{args.jobs}", flush=True)
            with timer.phase(f"persistent_worker_run_job_{index}"):
                job_result = base.simple_post(proxy_url + RUN_JOB_ENDPOINT, job_payload(index), timeout_seconds=args.job_timeout_seconds)
            data["jobs_by_index"][str(index)] = {"payload": job_payload(index), "result": compact_http_result(job_result)}
            job_json = job_result.get("json") if isinstance(job_result.get("json"), dict) else {}
            if job_result.get("http_status_code") != 200 or job_json.get("status") != "succeeded":
                status = f"job_{index}_failed"
                return 1

            with timer.phase(f"persistent_worker_status_after_job_{index}"):
                status_after_job = base.simple_get(proxy_url + STATUS_ENDPOINT, timeout_seconds=30)
            status_json = status_after_job.get("json") if isinstance(status_after_job.get("json"), dict) else {}
            data["jobs_by_index"][str(index)]["status_after"] = compact_http_result(status_after_job)
            can_continue, reasons = safety_gate(index, job_json, status_json)
            data["jobs_by_index"][str(index)]["safety_gate"] = {
                "passed": can_continue,
                "reasons": reasons,
            }
            if not can_continue:
                status = f"aborted_after_job_{index}_safety_gate"
                return 1

        with timer.phase("download_outputs_from_r2"):
            downloads = {}
            for index in range(1, args.jobs + 1):
                payload = job_payload(index)
                paths = local_output_paths(index)
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
            delete_instance(base_url, api_key, instance_id, data, timer)
        elif instance_id is not None:
            data["manual_cleanup_required"] = True
        data["runtime_seconds"] = round(time.monotonic() - started, 3)
        write_json(REPORT_PATH, build_report(args, status, data, error))
        print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sequential multi-job persistent Wan2.2 S2V worker probe on one SimplePod.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and run real sequential jobs.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-load-probe", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-unload", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute.")
    parser.add_argument("--jobs", type=int, default=2, help="Number of sequential jobs to run on the same loaded worker.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=60)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=900)
    parser.add_argument("--load-probe-timeout-seconds", type=int, default=1800)
    parser.add_argument("--job-timeout-seconds", type=int, default=2400)
    parser.add_argument("--unload-timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
