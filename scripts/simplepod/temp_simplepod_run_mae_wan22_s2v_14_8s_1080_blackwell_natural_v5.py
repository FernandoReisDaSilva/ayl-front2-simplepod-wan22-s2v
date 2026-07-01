import argparse
import json
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
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.6-blackwell"
STABLE_TEMPLATE_ID = 25114
STABLE_IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.6"
DATACENTER = "EU-PL-01"
PORT = 8000
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"
VERIFY_ENDPOINT = "/admin/verify-wan22-s2v-runtime"
INFERENCE_ENDPOINT = "/jobs/wan22-s2v/run"
GPU_POLICY = "blackwell_full_96gb_inference_policy"

JOB_ID = "mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial"
REFERENCE_IMAGE_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/reference/Mae_para_Wan_V3.png"
AUDIO_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/audio/mae_fr_14_8s_cut_for_wan.wav"
OUTPUT_VIDEO_KEY = "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial.mp4"
OUTPUT_REPORT_KEY = "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_blackwell_natural_v5_native_partial_final_report.json"
CONFIRM_INFERENCE = "RUN_WAN22_S2V_MAE_14_8S_1080_BLACKWELL_NATURAL_V5_NATIVE_PARTIAL"
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
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
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
    return [key for key in LOCAL_R2_ENV_KEYS if not os.getenv(key, "")]


def r2_env_variables_for_instance() -> list[dict]:
    return [{"name": key, "value": os.getenv(key, "")} for key in LOCAL_R2_ENV_KEYS]


def redact_instance_payload(payload: dict) -> dict:
    redacted = smoke.redact_value("", payload)
    for item in redacted.get("envVariables", []):
        if item.get("name") in LOCAL_R2_ENV_KEYS and item.get("value"):
            item["value"] = "<present_redacted>"
    return redacted


def inference_payload() -> dict:
    return {
        "job_id": JOB_ID,
        "character_id": "mae",
        "base_taught_language": "FR",
        "reference_image_key": REFERENCE_IMAGE_KEY,
        "audio_key": AUDIO_KEY,
        "target_width": 1080,
        "target_height": 1080,
        "fps": 16,
        "target_duration_seconds": 14.8,
        "output_video_key": OUTPUT_VIDEO_KEY,
        "output_report_key": OUTPUT_REPORT_KEY,
        "confirm_inference": CONFIRM_INFERENCE,
        "allow_oom_fallback": False,
        "positive_prompt": NATURAL_V5_POSITIVE_PROMPT,
        "negative_prompt": NATURAL_V5_NEGATIVE_PROMPT,
        "seed": 42,
        "steps": 5,
        "cfg": 1.0,
        "shift": 4.0,
    }


def runtime_payload(instance_market: str) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_full_blackwell_96gb_market>",
        "instanceTemplate": f"/instances/templates/{TEMPLATE_ID}",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
            {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
            {"name": "HF_HOME", "value": HF_HOME},
            {"name": "AYL_ENABLE_ADMIN_VERIFY", "value": "1"},
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-mae-natural-v5-inference"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
            *r2_env_variables_for_instance(),
        ],
    }


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
        "status",
        "message",
        "requested_resolution",
        "actual_generation_resolution",
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
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not execute_allowed,
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
        "output_video_key": OUTPUT_VIDEO_KEY,
        "output_report_key": OUTPUT_REPORT_KEY,
        "payload_dryrun": inference_payload(),
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
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_inference:
        return "blocked_missing_confirm_inference"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def run(args: argparse.Namespace) -> int:
    data = {}
    timer = PhaseTimer()
    data["phase_timings"] = timer.phases
    instance_id = None
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    started_monotonic = time.monotonic()
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] gpu_policy={GPU_POLICY} target=1080x1080 allow_oom_fallback=false")

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
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=runtime_payload(market))
        data["start_result"] = {
            key: start_result.get(key)
            for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
        }
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

        with timer.phase("run_inference_endpoint"):
            inference_result = simple_post(
                proxy_url + INFERENCE_ENDPOINT,
                inference_payload(),
                args.inference_timeout_seconds,
            )
        body = inference_result.get("json")
        data["r2_preflight_result"] = r2_preflight_from_inference(body)
        data["inference_result"] = {
            "attempted": True,
            "status": inference_result.get("status"),
            "http_status_code": inference_result.get("http_status_code"),
            "error_type": inference_result.get("error_type", ""),
            "error_truncated": inference_result.get("error_truncated", ""),
            "summary": summarize_inference(body),
        }

        status = "succeeded" if isinstance(body, dict) and body.get("video_generated") is True else "failed_inference_endpoint"
        data["_status_for_finally"] = status
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, status, data))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
        return 0 if status == "succeeded" else 1
    except KeyboardInterrupt:
        status = "interrupted_delete_attempted" if instance_id is not None else "interrupted"
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
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
            with timer.phase("delete_instance"):
                delete_path = smoke.DELETE_INSTANCE_PATH.format(id=instance_id)
                delete_result = smoke.http_request(base_url, delete_path, api_key, method="DELETE")
            data["delete_result"] = {
                key: delete_result.get(key)
                for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
            }
            final_status = data.get("_status_for_finally")
            if delete_result.get("http_status_code") not in {200, 202, 204}:
                final_status = "delete_failed_manual_required"
            if final_status:
                data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
                write_json(REPORT_PATH, build_report(args, final_status, data))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute Maé Wan2.2 S2V 1080 Blackwell natural_v5 inference gate.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and call the inference endpoint.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute to call the inference endpoint.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; runtime still rejects MIG.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    parser.add_argument("--inference-timeout-seconds", type=int, default=7200)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
