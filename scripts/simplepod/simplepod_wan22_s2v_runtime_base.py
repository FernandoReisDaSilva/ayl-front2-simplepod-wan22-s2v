import argparse
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import temp_simplepod_blackwell_smoke_v1 as blackwell_smoke
import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


TEST_ID = "SIMPLEPOD_WAN22_S2V_RUNTIME"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_wan22_s2v_runtime_base.json"

TEMPLATE_ID = 25138
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.22-blackwell"
STABLE_TEMPLATE_ID = 25114
STABLE_IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.6"
DATACENTER = "EU-PL-01"
PORT = 8000
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"
VERIFY_ENDPOINT = "/admin/verify-wan22-s2v-runtime"
INFERENCE_ENDPOINT = "/jobs/wan22-s2v/run"
ASYNC_INFERENCE_ENDPOINT = "/admin/run-mae-wan22-s2v-async"
GPU_POLICY = "blackwell_full_96gb_inference_policy"

JOB_ID_PREFIX = "simplepod_wan22_s2v"
JOB_ID_SUFFIX = "runtime_base_native_partial"
DEFAULT_CHARACTER_ID = ""
DEFAULT_TAUGHT_LANGUAGE = ""
REFERENCE_IMAGE_KEY = ""
AUDIO_KEY = ""
CONFIRM_INFERENCE_1080 = "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL"
CONFIRM_INFERENCE_720 = "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL"
CONFIRM_INFERENCE_GENERIC = "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL"
NATIVE_PARTIAL_REASON = "Native Wan2.2 runner does not have confirmed support for ComfyUI/WanVideoWrapper conditioning parameters."
NATURAL_V5_REFERENCE_UNSUPPORTED_PARAMETERS = {
    "denoise_strength": 0.80,
    "audio_scale": 1.55,
    "pose_start_percent": 0.0,
    "pose_end_percent": 0.45,
    "num_frames": 237,
}
LOCAL_R2_ENV_KEYS = (
    "R2_ENDPOINT",
    "R2_ENDPOINT_URL",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_BUCKET_NAME",
    "R2_REGION",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_local_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=REPO_ROOT / ".env")
    except Exception:
        smoke.load_repo_dotenv()


def local_r2_env_presence() -> dict:
    return {
        key: {
            "status": "PRESENT" if os.getenv(key, "") else "MISSING",
            "value": "<present_redacted>" if os.getenv(key, "") else "",
        }
        for key in LOCAL_R2_ENV_KEYS
    }


def missing_local_r2_env() -> list[str]:
    missing = []
    if not r2_value("R2_ENDPOINT", "R2_ENDPOINT_URL"):
        missing.append("R2_ENDPOINT_or_R2_ENDPOINT_URL")
    if not r2_value("R2_BUCKET", "R2_BUCKET_NAME"):
        missing.append("R2_BUCKET_or_R2_BUCKET_NAME")
    for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        if not os.getenv(key, ""):
            missing.append(key)
    return missing


def r2_env_variables_for_instance() -> list[dict]:
    return [{"name": key, "value": os.getenv(key, "")} for key in LOCAL_R2_ENV_KEYS if os.getenv(key, "")]


def r2_value(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name, "")
        if value:
            return value
    return default


def r2_client_config() -> dict:
    return {
        "endpoint_url": r2_value("R2_ENDPOINT", "R2_ENDPOINT_URL"),
        "bucket": r2_value("R2_BUCKET", "R2_BUCKET_NAME"),
        "access_key_id": r2_value("R2_ACCESS_KEY_ID"),
        "secret_access_key": r2_value("R2_SECRET_ACCESS_KEY"),
        "region": r2_value("R2_REGION", default="auto"),
    }


def r2_upload_file(local_path: Path, key: str, dry_run: bool) -> dict:
    result = {
        "attempted": not dry_run,
        "dry_run": dry_run,
        "local_path": str(local_path),
        "key": key,
        "status": "planned" if dry_run else "pending",
        "exists": local_path.exists(),
        "size_bytes": local_path.stat().st_size if local_path.exists() else None,
    }
    if dry_run:
        return result
    if not local_path.exists() or not local_path.is_file():
        return {**result, "status": "failed_missing_local_file", "error_type": "FileNotFoundError"}
    config = r2_client_config()
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
        extra_args = {}
        content_type, _ = mimetypes.guess_type(local_path.name)
        if content_type:
            extra_args["ContentType"] = content_type
        client.upload_file(str(local_path), config["bucket"], key, ExtraArgs=extra_args or None)
        return {**result, "status": "succeeded", "attempted": True}
    except Exception as exc:
        return {
            **result,
            "status": "failed",
            "attempted": True,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


def prepare_input_uploads(args: argparse.Namespace, dry_run: bool) -> dict:
    if not args.upload_inputs:
        return {
            "upload_inputs": False,
            "uploaded_image_result": {"attempted": False, "status": "skipped_upload_inputs_false"},
            "uploaded_audio_result": {"attempted": False, "status": "skipped_upload_inputs_false"},
        }
    image_path = local_path_value(args.local_image_path) if args.local_image_path else Path("")
    audio_path = local_path_value(args.local_audio_path) if args.local_audio_path else Path("")
    image_result = r2_upload_file(image_path, input_image_key(args), dry_run)
    audio_result = r2_upload_file(audio_path, input_audio_key(args), dry_run)
    return {
        "upload_inputs": True,
        "uploaded_image_result": image_result,
        "uploaded_audio_result": audio_result,
    }


def redact_instance_payload(payload: dict) -> dict:
    redacted = smoke.redact_value("", payload)
    for item in redacted.get("envVariables", []):
        if item.get("name") in LOCAL_R2_ENV_KEYS and item.get("value"):
            item["value"] = "<present_redacted>"
    return redacted


def resolution_token(width: int, height: int) -> str:
    return str(width) if width == height else f"{width}x{height}"


def default_output_stem(width: int, height: int) -> str:
    return f"{JOB_ID_PREFIX}_{resolution_token(width, height)}_{JOB_ID_SUFFIX}"


def output_stem(args: argparse.Namespace) -> str:
    if args.output_stem:
        return args.output_stem
    return default_output_stem(int(args.width), int(args.height))


def test_id_for_args(args: argparse.Namespace) -> str:
    return args.test_id or output_stem(args)


def output_video_key(args: argparse.Namespace) -> str:
    return f"tests/simplepod_wan22_s2v/outputs/{output_stem(args)}.mp4"


def output_report_key(args: argparse.Namespace) -> str:
    return f"tests/simplepod_wan22_s2v/outputs/{output_stem(args)}_final_report.json"


def local_path_value(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def input_key_from_local_file(args: argparse.Namespace, kind: str, local_path: Path) -> str:
    return f"tests/simplepod_wan22_s2v/inputs/{test_id_for_args(args)}/{kind}/{local_path.name}"


def input_image_key(args: argparse.Namespace) -> str:
    if args.input_image_key:
        return args.input_image_key
    if args.upload_inputs and args.local_image_path:
        return input_key_from_local_file(args, "reference", local_path_value(args.local_image_path))
    return REFERENCE_IMAGE_KEY


def input_audio_key(args: argparse.Namespace) -> str:
    if args.input_audio_key:
        return args.input_audio_key
    if args.upload_inputs and args.local_audio_path:
        return input_key_from_local_file(args, "audio", local_path_value(args.local_audio_path))
    return AUDIO_KEY


def confirm_inference_for_args(args: argparse.Namespace) -> str:
    width = int(args.width)
    height = int(args.height)
    if args.character_id == DEFAULT_CHARACTER_ID and args.taught_language == DEFAULT_TAUGHT_LANGUAGE and width == 720 and height == 720:
        return CONFIRM_INFERENCE_720
    if args.character_id == DEFAULT_CHARACTER_ID and args.taught_language == DEFAULT_TAUGHT_LANGUAGE and width == 1080 and height == 1080:
        return CONFIRM_INFERENCE_1080
    return CONFIRM_INFERENCE_GENERIC


def inference_payload(args: argparse.Namespace) -> dict:
    width = int(args.width)
    height = int(args.height)
    return {
        "job_id": output_stem(args),
        "character_id": args.character_id,
        "base_taught_language": args.taught_language,
        "reference_image_key": input_image_key(args),
        "audio_key": input_audio_key(args),
        "width": width,
        "height": height,
        "target_width": width,
        "target_height": height,
        "resolution": f"{width}x{height}",
        "fps": 16,
        "target_duration_seconds": 14.8,
        "output_video_key": output_video_key(args),
        "output_report_key": output_report_key(args),
        "confirm_inference": confirm_inference_for_args(args),
        "allow_oom_fallback": False,
        "seed": 42,
        "steps": 5,
        "cfg": 1.0,
        "shift": 4.0,
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
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-runtime-base"},
            {"name": "AYL_SAFETENSORS_CUDA_TO_CPU_PATCH", "value": "1"},
            {"name": "MAX_CONCURRENT_JOBS", "value": "1"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
            *r2_env_variables_for_instance(),
        ],
    }
    payload["envVariables"] = [
        item for item in payload["envVariables"] if item.get("value") not in ("", None)
    ]
    return payload


def parse_json_body(body: bytes, content_type: str):
    if "json" not in content_type.lower():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def simple_post(url: str, payload: dict, timeout_seconds: int) -> dict:
    request = Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ayl-front2-simplepod-wan22-s2v-runtime-base",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            return {
                "status": "succeeded",
                "http_status_code": response.status,
                "content_type": content_type,
                "body_bytes": len(body),
                "json": parse_json_body(body, content_type),
            }
    except HTTPError as exc:
        body = exc.read(262_144)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return {
            "status": "failed",
            "http_status_code": exc.code,
            "content_type": content_type,
            "body_bytes": len(body),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
            "json": parse_json_body(body, content_type),
        }
    except (TimeoutError, URLError) as exc:
        return {
            "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


def simple_get(url: str, timeout_seconds: int) -> dict:
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "ayl-front2-simplepod-wan22-s2v-runtime-base",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            return {
                "status": "succeeded",
                "http_status_code": response.status,
                "content_type": content_type,
                "body_bytes": len(body),
                "json": parse_json_body(body, content_type),
            }
    except HTTPError as exc:
        body = exc.read(262_144)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return {
            "status": "failed",
            "http_status_code": exc.code,
            "content_type": content_type,
            "body_bytes": len(body),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
            "json": parse_json_body(body, content_type),
        }
    except (TimeoutError, URLError) as exc:
        return {
            "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:1000],
        }


def summarize_runtime_verify(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    keys = (
        "status",
        "torch_version",
        "torch_cuda_version",
        "device_name",
        "device_capability",
        "models_root_exists",
        "wan22_model_dir_exists",
        "recursive_file_count",
        "recursive_total_size_gb",
        "wan_code_import_status",
        "runner_import_status",
        "download_attempted",
        "inference_executed",
        "video_generated",
        "placeholder_generated",
    )
    return {key: value.get(key) for key in keys if key in value}


def runtime_verify_passed(result: dict) -> bool:
    value = result.get("json")
    return (
        result.get("http_status_code") == 200
        and isinstance(value, dict)
        and value.get("status") == "verified"
        and value.get("wan_code_import_status") == "ok"
        and value.get("runner_import_status") == "ok"
        and value.get("download_attempted") is False
        and value.get("inference_executed") is False
        and value.get("video_generated") is False
    )


def summarize_inference(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    keys = (
        "job_id",
        "job_status",
        "status",
        "message",
        "width",
        "height",
        "resolution",
        "requested_width",
        "requested_height",
        "requested_resolution",
        "requested_resolution_detail",
        "actual_generation_resolution",
        "output_width",
        "output_height",
        "output_resolution",
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
    summary = {key: value.get(key) for key in keys if key in value}
    primary = value.get("primary_inference")
    if isinstance(primary, dict):
        summary["subprocess_returncode"] = primary.get("returncode")
        summary["stdout_truncated"] = primary.get("stdout_truncated", "")
        summary["stderr_truncated"] = primary.get("stderr_truncated", "")
    return summary


def r2_preflight_from_inference(value) -> dict:
    if not isinstance(value, dict):
        return {"status": "not_available_before_inference_response"}
    preflight = value.get("r2_preflight")
    if isinstance(preflight, dict):
        return preflight
    return {
        "status": "summarized_from_inference_result",
        "r2_env_check_status": value.get("r2_env_check_status"),
        "r2_reference_head_status": value.get("r2_reference_head_status"),
        "r2_audio_head_status": value.get("r2_audio_head_status"),
        "r2_upload_permission_check_status": value.get("r2_upload_permission_check_status"),
    }


def poll_async_job(proxy_url: str, job_id: str, args: argparse.Namespace, log_prefix: str | None = None) -> dict:
    started = time.monotonic()
    attempts = []
    status_url = proxy_url + f"/admin/jobs/{job_id}"
    while True:
        elapsed = round(time.monotonic() - started, 3)
        if elapsed > args.job_timeout_seconds:
            return {
                "status": "timeout",
                "job_id": job_id,
                "elapsed_seconds": elapsed,
                "attempts": attempts[-20:],
                "error_type": "JobTimeout",
                "error_truncated": f"Async job did not finish within {args.job_timeout_seconds}s.",
            }
        result = simple_get(status_url, timeout_seconds=30)
        body = result.get("json")
        job_status = body.get("status") if isinstance(body, dict) else result.get("status")
        attempts.append(
            {
                "elapsed_seconds": elapsed,
                "http_status_code": result.get("http_status_code"),
                "job_status": job_status,
                "error_type": result.get("error_type", ""),
            }
        )
        print(f"[{log_prefix or TEST_ID}] job_status={job_status} elapsed={int(elapsed)}s")
        if isinstance(body, dict) and job_status in {"succeeded", "failed"}:
            return {
                "status": "succeeded",
                "http_status_code": result.get("http_status_code"),
                "job_id": job_id,
                "elapsed_seconds": elapsed,
                "attempts": attempts[-20:],
                "json": body,
            }
        if result.get("status") not in {"succeeded", "timeout"} and result.get("http_status_code") != 404:
            return {
                "status": "failed",
                "http_status_code": result.get("http_status_code"),
                "job_id": job_id,
                "elapsed_seconds": elapsed,
                "attempts": attempts[-20:],
                "error_type": result.get("error_type", ""),
                "error_truncated": result.get("error_truncated", ""),
                "json": body,
            }
        time.sleep(max(1, args.job_poll_interval_seconds))


def is_full_blackwell_inference_candidate(item: dict) -> tuple[bool, str]:
    model = blackwell_smoke.gpu_model(item)
    memory_mb = blackwell_smoke.gpu_memory_mb(item)
    gpu_count = item.get("gpuCount")
    text = json.dumps(item, ensure_ascii=False).lower()
    if not blackwell_smoke.market_iri(item):
        return False, "missing_market_id"
    if str(item.get("rentalStatus") or item.get("status") or "active").lower() != "active":
        return False, "rentalStatus_not_active"
    if DATACENTER.lower() not in text:
        return False, "datacenter_not_EU_PL_01"
    if gpu_count not in {1, "1"}:
        return False, "gpuCount_not_1"
    if memory_mb is None or memory_mb < 90_000:
        return False, "gpuMemorySize_below_90000_for_full_blackwell"
    if "blackwell" not in model.lower() or "rtx pro 6000" not in model.lower():
        return False, "gpuModel_not_RTX_PRO_6000_Blackwell"
    if blackwell_smoke.is_mig_model(model):
        return False, "MIG_rejected_for_real_inference"
    return True, "full_RTX_PRO_6000_Blackwell_96GB_selected_for_single_real_inference"


def select_full_blackwell_market(items: list[dict]) -> dict:
    accepted = []
    rejected = []
    for item in items:
        if not isinstance(item, dict):
            continue
        accepted_candidate, reason = is_full_blackwell_inference_candidate(item)
        summary = blackwell_smoke.candidate_summary(item, reason)
        if accepted_candidate:
            accepted.append({"item": item, "price": blackwell_smoke.price_value(item), "summary": summary})
        else:
            rejected.append(summary)
    accepted.sort(
        key=lambda candidate: (
            candidate["price"] is None,
            candidate["price"] if candidate["price"] is not None else 999999,
            candidate["summary"]["market_iri"],
        )
    )
    selected = accepted[0] if accepted else None
    selected_item = selected["item"] if selected else {}
    return {
        "selected_policy": GPU_POLICY,
        "selected_market": blackwell_smoke.market_iri(selected_item),
        "selected_market_id": blackwell_smoke.market_id(selected_item),
        "selected_summary": (
            blackwell_smoke.candidate_summary(selected_item, "lowest_price_full_blackwell_96gb_candidate")
            if selected_item
            else {}
        ),
        "primary_datacenter": DATACENTER,
        "selected_datacenter": DATACENTER if selected_item else "",
        "fallback_datacenter_used": False,
        "searched_datacenters": [DATACENTER],
        "selection_rule": "EU-PL-01 active gpuCount=1 gpuMemorySize>=90000 gpuModel contains RTX PRO 6000 Blackwell and rejects MIG.",
        "accepted_candidates_observed": len(accepted),
        "accepted_candidates_summary": [candidate["summary"] for candidate in accepted[:10]],
        "rejected_candidates_observed": len(rejected),
        "rejected_candidates_summary": rejected[:30],
        "reason_selected": (
            "Full RTX PRO 6000 Blackwell 96GB selected; MIG is not allowed for real inference."
            if selected_item
            else "No full RTX PRO 6000 Blackwell 96GB market available in EU-PL-01."
        ),
    }


def selected_summary_from_data(data: dict) -> dict:
    selection = data.get("market_selection", {})
    selected = selection.get("selected", {}) if isinstance(selection, dict) else {}
    return selected.get("selected_summary", {}) if isinstance(selected, dict) else {}


def estimate_cost(selected_summary: dict, runtime_seconds: float | None) -> dict:
    price = blackwell_smoke.price_value(selected_summary)
    estimated = None
    if price is not None and runtime_seconds is not None:
        estimated = price * (runtime_seconds / 3600.0)
    return {
        "pricePerGpu": price,
        "runtime_seconds": runtime_seconds,
        "estimated_cost": estimated,
        "source": "SimplePod market API pricePerGpu" if price is not None else "",
    }


def choose_market(args: argparse.Namespace, base_url: str, api_key: str, data: dict) -> str:
    if args.instance_market:
        data["market_selection"] = {
            "selected": {
                "selected_policy": GPU_POLICY,
                "selected_market": args.instance_market,
                "selected_market_id": args.instance_market.rsplit("/", 1)[-1],
                "reason_selected": "Manual --instance-market override; runtime /gpu still rejects non-full or MIG Blackwell.",
            }
        }
        return args.instance_market

    query = {
        "mode": "docker",
        "rentalStatus": "active",
        "region": DATACENTER,
        "gpuCount[gte]": 1,
        "gpuCount[lte]": 1,
        "gpuMemorySize[gte]": 90_000,
        "itemsPerPage": 100,
        "order[pricePerGpu]": "asc",
    }
    market_result = smoke.http_request(base_url, f"{blackwell_smoke.MARKET_LIST_PATH}?{urlencode(query)}", api_key)
    items = smoke.extract_items(market_result.get("json"))
    selected = select_full_blackwell_market(items)
    data["market_selection"] = {
        "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path", "error_type", "error_truncated")},
        "items_observed": len(items),
        "selected": selected,
        "selected_policy": GPU_POLICY,
        "selected_market_id": selected.get("selected_market_id", ""),
        "gpuModel": selected.get("selected_summary", {}).get("gpuModel", ""),
        "gpuMemorySize": selected.get("selected_summary", {}).get("gpuMemorySize"),
        "pricePerGpu": selected.get("selected_summary", {}).get("pricePerGpu"),
        "selected_datacenter": selected.get("selected_datacenter", ""),
        "searched_datacenters": selected.get("searched_datacenters", [DATACENTER]),
        "reason_selected": selected.get("reason_selected", ""),
        "rejected_candidates_summary": selected.get("rejected_candidates_summary", []),
    }
    return selected.get("selected_market", "")


def wait_for_public_url(base_url: str, api_key: str, instance_id: int, args: argparse.Namespace, data: dict) -> str:
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    proxy_url = ""
    attempts = []
    for _ in range(max(1, args.detail_attempts)):
        detail_result = smoke.http_request(base_url, detail_path, api_key)
        attempts.append({key: detail_result.get(key) for key in ("status", "http_status_code", "error_type")})
        if isinstance(detail_result.get("json"), dict):
            selected_mapping = smoke.extract_api_port_mapping(detail_result["json"], PORT)
            if selected_mapping:
                data["selected_api_port_mapping"] = selected_mapping
            proxy_url = smoke.extract_proxy_url_for_port(detail_result["json"], PORT)
            if proxy_url:
                break
        time.sleep(args.poll_interval_seconds)
    data["detail_attempts"] = attempts
    data["public_api_base_url"] = proxy_url
    return proxy_url


def runtime_gpu_is_full_blackwell(gpu_json) -> dict:
    validation = blackwell_smoke.validate_blackwell_gpu(gpu_json)
    device_name = ""
    if isinstance(gpu_json, dict):
        device_name = str(gpu_json.get("device_name", ""))
    is_mig = "mig" in device_name.lower()
    return {
        "status": "passed" if validation.get("status") == "passed" and not is_mig else "blocked",
        "blackwell_validation": validation,
        "device_name": device_name,
        "mig_rejected": is_mig,
        "reason": "" if validation.get("status") == "passed" and not is_mig else "runtime GPU is not full Blackwell 96GB or appears to be MIG",
    }

