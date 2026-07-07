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


TEST_ID = "TEMP_SIMPLEPOD_RUN_MAE_WAN22_S2V_14_8S_1080_BLACKWELL_NATURAL_V5"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_mae_wan22_s2v_14_8s_1080_blackwell_natural_v5_inference_v1.json"

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

JOB_ID_PREFIX = "mae_fr_wan22_s2v_14_8s"
JOB_ID_SUFFIX = "blackwell_natural_v5_native_partial"
DEFAULT_CHARACTER_ID = "mae"
DEFAULT_TAUGHT_LANGUAGE = "FR"
REFERENCE_IMAGE_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/reference/Mae_para_Wan_V3.png"
AUDIO_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/audio/mae_fr_14_8s_cut_for_wan.wav"
CONFIRM_INFERENCE_1080 = "RUN_WAN22_S2V_MAE_14_8S_1080_BLACKWELL_NATURAL_V5_NATIVE_PARTIAL"
CONFIRM_INFERENCE_720 = "RUN_WAN22_S2V_MAE_14_8S_720_BLACKWELL_NATURAL_V5_NATIVE_PARTIAL"
CONFIRM_INFERENCE_GENERIC = "RUN_WAN22_S2V_BLACKWELL_NATIVE_PARTIAL"
NATURAL_V5_POSITIVE_PROMPT = (
    "stable square close-up talking head portrait of the same woman, natural French speech articulation, "
    "stronger and more active accurate lip sync, clear mouth openings closures rounded vowels and labial consonants, "
    "slower more natural head motion, subtle controlled head turns and nods, gentle eye neck shoulder and torso micro movements, "
    "preserved identity, high quality face, natural conversational delivery in French"
)
NATURAL_V5_NEGATIVE_PROMPT = (
    "fast head movement, head bobbing, jerky motion, excessive body swaying, exaggerated motion, overacting, "
    "distorted mouth, weak lip sync, blurry face, identity drift, singing performance, subtitles"
)
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
        "positive_prompt": NATURAL_V5_POSITIVE_PROMPT,
        "negative_prompt": NATURAL_V5_NEGATIVE_PROMPT,
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
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-mae-natural-v5-inference"},
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
            "User-Agent": "ayl-front2-simplepod-mae-wan22-s2v-blackwell-v1",
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
            "User-Agent": "ayl-front2-simplepod-mae-wan22-s2v-blackwell-v1",
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


def poll_async_job(proxy_url: str, job_id: str, args: argparse.Namespace) -> dict:
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
        print(f"[{TEST_ID}] job_status={job_status} elapsed={int(elapsed)}s")
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


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    execute_allowed = args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete
    selected_summary = selected_summary_from_data(data)
    runtime_seconds = data.get("runtime_seconds")
    requested_width = int(args.width)
    requested_height = int(args.height)
    requested_resolution = f"{requested_width}x{requested_height}"
    payload = inference_payload(args)
    inference_summary = data.get("inference_result", {}).get("summary", {})
    async_start_json = data.get("async_job_start_result", {}).get("json")
    if not isinstance(async_start_json, dict):
        async_start_json = {}
    return {
        "script_id": TEST_ID,
        "test_id": test_id_for_args(args),
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not execute_allowed,
        "job_id": inference_summary.get("job_id") or payload["job_id"],
        "job_status": inference_summary.get("job_status"),
        "character_id": args.character_id,
        "taught_language": args.taught_language,
        "local_image_path": str(local_path_value(args.local_image_path)) if args.local_image_path else "",
        "local_audio_path": str(local_path_value(args.local_audio_path)) if args.local_audio_path else "",
        "input_image_key": payload["reference_image_key"],
        "input_audio_key": payload["audio_key"],
        "upload_inputs": bool(args.upload_inputs),
        "uploaded_image_result": data.get("uploaded_image_result", {}),
        "uploaded_audio_result": data.get("uploaded_audio_result", {}),
        "output_stem": output_stem(args),
        "width": inference_summary.get("width", requested_width),
        "height": inference_summary.get("height", requested_height),
        "resolution": inference_summary.get("resolution", requested_resolution),
        "requested_width": requested_width,
        "requested_height": requested_height,
        "requested_resolution": requested_resolution,
        "output_width": inference_summary.get("output_width"),
        "output_height": inference_summary.get("output_height"),
        "output_resolution": inference_summary.get("output_resolution"),
        "template_id": TEMPLATE_ID,
        "image_ref": IMAGE,
        "stable_template_unchanged": STABLE_TEMPLATE_ID,
        "stable_image_unchanged": STABLE_IMAGE,
        "gpu_policy": GPU_POLICY,
        "selected_market_id": data.get("market_selection", {}).get("selected_market_id") or selected_summary.get("market_id", ""),
        "gpuModel": selected_summary.get("gpuModel", ""),
        "gpuMemorySize": selected_summary.get("gpuMemorySize"),
        "pricePerGpu": selected_summary.get("pricePerGpu"),
        "instance_id": data.get("instance_id"),
        "public_url": data.get("public_api_base_url", ""),
        "health_result": data.get("health_result", {}),
        "gpu_result": data.get("gpu_check", {}),
        "runtime_verify_result": data.get("runtime_verify_result", {}),
        "r2_preflight_result": data.get("r2_preflight_result", {}),
        "inference_result": data.get("inference_result", {}),
        "async_inference_endpoint": ASYNC_INFERENCE_ENDPOINT,
        "async_job_id": data.get("async_job_id", ""),
        "async_job_start_result": data.get("async_job_start_result", {}),
        "async_job_poll_result": data.get("async_job_poll_result", {}),
        "job_timeout_seconds": args.job_timeout_seconds,
        "job_poll_interval_seconds": args.job_poll_interval_seconds,
        "max_concurrent_jobs": 1,
        "active_jobs_at_submission": async_start_json.get("active_jobs_at_submission"),
        "output_video_key": payload["output_video_key"],
        "output_report_key": payload["output_report_key"],
        "payload_dryrun": payload,
        "native_partial_reason": NATIVE_PARTIAL_REASON,
        "natural_v5_reference_unsupported_parameters": NATURAL_V5_REFERENCE_UNSUPPORTED_PARAMETERS,
        "r2_env_local_check": data.get("r2_env_local_check", local_r2_env_presence()),
        "instance_payload_dryrun": redact_instance_payload(runtime_payload(args.instance_market or "<selected_full_blackwell_96gb_market>")),
        "startScript_sent": False,
        "uses_image_cmd": True,
        "docker_entrypoint_arguments_sent": False,
        "runtime_seconds": runtime_seconds,
        "estimated_cost": estimate_cost(selected_summary, runtime_seconds),
        "delete_result": data.get("delete_result", {}),
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "downloads_model_weights": False,
            "runs_inference": bool(data.get("inference_result", {}).get("attempted")),
            "generates_video": bool(data.get("inference_result", {}).get("summary", {}).get("video_generated") is True),
            "placeholder_generated": False,
            "uses_scheduler": False,
            "parallel_jobs": False,
            "uses_mig": False,
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "calls_simplepod": bool(execute_allowed),
            "secrets_printed": False,
        },
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    if args.width <= 0 or args.height <= 0:
        return "blocked_invalid_resolution"
    if args.width > 1080 or args.height > 1080:
        return "blocked_resolution_above_1080"
    if args.upload_inputs and (not args.local_image_path or not args.local_audio_path):
        return "blocked_upload_inputs_missing_local_paths"
    if args.execute and args.upload_inputs:
        if not local_path_value(args.local_image_path).is_file() or not local_path_value(args.local_audio_path).is_file():
            return "blocked_upload_inputs_local_files_missing"
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_inference:
        return "blocked_missing_confirm_inference"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def run(args: argparse.Namespace) -> int:
    data = {
        "delete_result": {
            "attempted": False,
            "status": "not_started",
            "http_status_code": None,
            "error_type": "",
            "error_truncated": "",
        }
    }
    timer = PhaseTimer()
    data["phase_timings"] = timer.phases
    instance_id = None
    proxy_url = ""
    async_job_id = ""
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    started_monotonic = time.monotonic()
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] gpu_policy={GPU_POLICY} target={args.width}x{args.height} allow_oom_fallback=false")

        status = blocked_status(args)
        if status:
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("load_auth_env"):
            load_local_env()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)
            data["r2_env_local_check"] = local_r2_env_presence()

        with timer.phase("prepare_input_uploads"):
            upload_results = prepare_input_uploads(args, dry_run=not execute_allowed)
            data.update(upload_results)
        if execute_allowed and args.upload_inputs:
            upload_ok = (
                data.get("uploaded_image_result", {}).get("status") == "succeeded"
                and data.get("uploaded_audio_result", {}).get("status") == "succeeded"
            )
            if not upload_ok:
                status = "failed_upload_inputs"
                data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

        if api_key:
            with timer.phase("market_selection"):
                market = choose_market(args, base_url, api_key, data)
        else:
            market = args.instance_market
            data["market_selection"] = {
                "status": "skipped_missing_api_key",
                "selected": select_full_blackwell_market([]),
            }

        if not execute_allowed:
            with timer.phase("dry_run_report"):
                pass
            status = "dry_run_ready"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        if not api_key:
            status = "missing_api_key"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        missing_r2 = missing_local_r2_env()
        if missing_r2:
            status = "missing_local_r2_env"
            data["missing_local_r2_env"] = missing_r2
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not market:
            status = "blocked_no_full_blackwell_96gb_market_selected"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("start_instance"):
            start_payload = runtime_payload(market)
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=start_payload)
        data["start_result"] = {
            key: start_result.get(key)
            for key in (
                "attempted",
                "status",
                "method",
                "path",
                "http_status_code",
                "endpoint_host",
                "content_type",
                "body_bytes",
                "error_type",
                "error_truncated",
                "response_body_truncated",
            )
        }
        data["start_result"]["request_payload_redacted"] = redact_instance_payload(start_payload)
        data["start_result"]["json"] = start_result.get("json")
        instance_id = smoke.extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("wait_public_url"):
            proxy_url = wait_for_public_url(base_url, api_key, instance_id, args, data)
        if not proxy_url:
            status = "blocked_no_proxy_url_for_port_8000"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("wait_health"):
            readiness, readiness_attempts, _ = smoke.wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
        data["api_readiness"] = {"status": readiness, "attempts": readiness_attempts}
        if readiness != "ready":
            status = "api_not_ready"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("health_check"):
            health_result = smoke.simple_get(proxy_url + "/health")
        data["health_result"] = {
            "status": health_result.get("status"),
            "http_status_code": health_result.get("http_status_code"),
            "summary": smoke.summarize_api_response(health_result.get("json")),
        }
        if health_result.get("http_status_code") != 200:
            status = "health_check_failed"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("gpu_check"):
            gpu_result = smoke.simple_get(proxy_url + "/gpu")
        data["gpu_check"] = {
            "status": gpu_result.get("status"),
            "http_status_code": gpu_result.get("http_status_code"),
            "summary": blackwell_smoke.summarize_gpu(gpu_result.get("json")),
            "full_blackwell_runtime_check": runtime_gpu_is_full_blackwell(gpu_result.get("json")),
        }
        if data["gpu_check"]["full_blackwell_runtime_check"].get("status") != "passed":
            status = "blocked_runtime_not_full_blackwell_96gb_or_mig"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("wan22_runtime_verify"):
            verify_result = smoke.simple_get(proxy_url + VERIFY_ENDPOINT)
        data["runtime_verify_result"] = {
            "attempted": True,
            "status": verify_result.get("status"),
            "http_status_code": verify_result.get("http_status_code"),
            "error_type": verify_result.get("error_type", ""),
            "error_truncated": verify_result.get("error_truncated", ""),
            "summary": summarize_runtime_verify(verify_result.get("json")),
        }
        if not runtime_verify_passed(verify_result):
            status = "failed_wan22_runtime_verify_before_inference"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("start_async_inference_job"):
            async_start_result = simple_post(
                proxy_url + ASYNC_INFERENCE_ENDPOINT,
                inference_payload(args),
                60,
            )
        async_start_body = async_start_result.get("json")
        data["async_job_start_result"] = {
            "attempted": True,
            "status": async_start_result.get("status"),
            "http_status_code": async_start_result.get("http_status_code"),
            "error_type": async_start_result.get("error_type", ""),
            "error_truncated": async_start_result.get("error_truncated", ""),
            "json": async_start_body,
        }
        if async_start_result.get("http_status_code") not in {200, 202} or not isinstance(async_start_body, dict):
            status = "failed_start_async_inference_job"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        async_job_id = str(async_start_body.get("job_id") or "")
        data["async_job_id"] = async_job_id
        if async_start_body.get("status") != "accepted" or not async_job_id:
            status = "async_inference_not_accepted"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("poll_async_inference_job"):
            async_poll_result = poll_async_job(proxy_url, async_job_id, args)
        data["async_job_poll_result"] = async_poll_result
        body = async_poll_result.get("json")
        job_summary = body.get("summary", {}) if isinstance(body, dict) else {}
        data["r2_preflight_result"] = r2_preflight_from_inference(job_summary)
        data["inference_result"] = {
            "attempted": True,
            "mode": "async_polling",
            "status": body.get("status") if isinstance(body, dict) else async_poll_result.get("status"),
            "http_status_code": async_poll_result.get("http_status_code"),
            "error_type": (body.get("error_type") if isinstance(body, dict) else async_poll_result.get("error_type", "")) or "",
            "error_truncated": (body.get("error_truncated") if isinstance(body, dict) else async_poll_result.get("error_truncated", "")) or "",
            "summary": summarize_inference(job_summary),
            "job_status": body,
        }

        status = "succeeded" if isinstance(job_summary, dict) and job_summary.get("video_generated") is True else "failed_inference_endpoint"
        data["_status_for_finally"] = status
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, status, data))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
        return 0 if status == "succeeded" else 1
    except KeyboardInterrupt:
        status = "interrupted_delete_attempted" if instance_id is not None else "interrupted"
        if proxy_url and async_job_id:
            last_job_status = simple_get(proxy_url + f"/admin/jobs/{async_job_id}", timeout_seconds=20)
            data["last_async_job_status_on_interrupt"] = last_job_status
            if "inference_result" not in data:
                body = last_job_status.get("json")
                summary = body.get("summary", {}) if isinstance(body, dict) else {}
                data["inference_result"] = {
                    "attempted": True,
                    "mode": "async_polling",
                    "status": body.get("status") if isinstance(body, dict) else "interrupted",
                    "http_status_code": last_job_status.get("http_status_code"),
                    "error_type": "KeyboardInterrupt",
                    "error_truncated": "Interrupted locally after async job was started.",
                    "summary": summarize_inference(summary),
                    "job_status": body,
                }
                data["r2_preflight_result"] = r2_preflight_from_inference(summary)
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        data["_status_for_finally"] = status
        write_json(REPORT_PATH, build_report(args, status, data, "KeyboardInterrupt"))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", file=sys.stderr)
        return 130
    except Exception as exc:
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, "failed", data, str(exc)))
        print(f"[{TEST_ID}] ERROR {str(exc)[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1
    finally:
        if instance_id is not None:
            delete_path = smoke.DELETE_INSTANCE_PATH.format(id=instance_id)
            try:
                with timer.phase("delete_instance"):
                    delete_result = smoke.http_request(base_url, delete_path, api_key, method="DELETE")
            except Exception as delete_exc:
                delete_result = {
                    "attempted": True,
                    "status": "failed",
                    "method": "DELETE",
                    "path": delete_path,
                    "http_status_code": None,
                    "endpoint_host": base_url,
                    "error_type": type(delete_exc).__name__,
                    "error_truncated": str(delete_exc)[:1000],
                }
            data["delete_result"] = {
                key: delete_result.get(key)
                for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
            }
            final_status = data.get("_status_for_finally")
            if delete_result.get("http_status_code") not in {200, 202, 204}:
                final_status = "delete_failed_manual_required"
                print(f"[{TEST_ID}] DELETE FAILED - manual cleanup required instance_id={instance_id}", file=sys.stderr)
            if final_status:
                data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
                write_json(REPORT_PATH, build_report(args, final_status, data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute Maé Wan2.2 S2V Blackwell natural_v5 inference gate.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and call the inference endpoint.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute to call the inference endpoint.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; runtime still rejects MIG.")
    parser.add_argument("--character-id", default=DEFAULT_CHARACTER_ID)
    parser.add_argument("--taught-language", default=DEFAULT_TAUGHT_LANGUAGE)
    parser.add_argument("--local-image-path", default="")
    parser.add_argument("--local-audio-path", default="")
    parser.add_argument("--input-image-key", default="")
    parser.add_argument("--input-audio-key", default="")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--test-id", default="")
    parser.add_argument("--upload-inputs", action="store_true", help="Upload local image/audio to R2 before starting SimplePod. Dry-run only plans.")
    parser.add_argument("--width", type=int, default=720, help="Requested generation width. Default: 720.")
    parser.add_argument("--height", type=int, default=720, help="Requested generation height. Default: 720.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    parser.add_argument("--inference-timeout-seconds", type=int, default=7200)
    parser.add_argument("--job-timeout-seconds", type=int, default=3600)
    parser.add_argument("--job-poll-interval-seconds", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
