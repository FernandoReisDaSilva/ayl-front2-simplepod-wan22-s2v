import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import simplepod_gpu_policies as gpu_policies
from simplepod_phase_timing import PhaseTimer, now_iso
import temp_simplepod_runtime_smoke_v2 as smoke
from temp_simplepod_verify_wan22_s2v_weights_v1 import summarize_verify, verify_result_looks_successful


TEST_ID = "TEMP_SIMPLEPOD_RUN_MAE_WAN22_S2V_14_8S_1080_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_mae_wan22_s2v_14_8s_1080_inference_v1.json"

TEMPLATE_ID = 25114
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.5"
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"
GPU_POLICY = "production_single_job_policy"
VERIFY_ENDPOINT = "/admin/verify-wan22-s2v-weights"
INFERENCE_ENDPOINT = "/jobs/wan22-s2v/run"
CONFIRM_INFERENCE = "RUN_WAN22_S2V_MAE_14_8S_1080"

JOB_ID = "mae_fr_wan22_s2v_14_8s_1080_v1"
REFERENCE_IMAGE_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/reference/Mae_para_Wan_V3.png"
AUDIO_KEY = "tests/simplepod_wan22_s2v/inputs/mae_fr_wan22_s2v_14_8s_1080_v1/audio/mae_fr_14_8s_cut_for_wan.wav"
OUTPUT_VIDEO_KEY = "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_v1.mp4"
OUTPUT_REPORT_KEY = "tests/simplepod_wan22_s2v/outputs/mae_fr_wan22_s2v_14_8s_1080_v1_final_report.json"
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
    return [
        {"name": key, "value": os.getenv(key, "")}
        for key in LOCAL_R2_ENV_KEYS
    ]


def redact_instance_payload(payload: dict) -> dict:
    redacted = smoke.redact_value("", payload)
    for item in redacted.get("envVariables", []):
        if item.get("name") in LOCAL_R2_ENV_KEYS and item.get("value"):
            item["value"] = "<present_redacted>"
    return redacted


def inference_payload(allow_oom_fallback: bool = False) -> dict:
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
        "allow_oom_fallback": allow_oom_fallback,
    }


def runtime_payload(instance_market: str) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_from_GET_/instances/market/list>",
        "instanceTemplate": f"/instances/templates/{TEMPLATE_ID}",
        "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
            {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
            {"name": "HF_HOME", "value": HF_HOME},
            {"name": "AYL_ENABLE_ADMIN_VERIFY", "value": "1"},
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-first-inference-gate"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
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
            "User-Agent": "ayl-front2-simplepod-mae-wan22-s2v-1080-v1",
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
        "inference_executed",
        "placeholder_generated",
        "video_generated",
        "r2_upload_attempted",
    )
    return {key: value.get(key) for key in keys if key in value}


def gpu_passes_runtime_sanity(gpu_json: dict) -> bool:
    report = gpu_policies.vram_policy_report(GPU_POLICY, {}, gpu_json)
    return bool(report.get("runtime_sanity_passed"))


def choose_market(args: argparse.Namespace, base_url: str, api_key: str, data: dict) -> str:
    if args.instance_market:
        data["market_selection"] = {
            "selected_policy": GPU_POLICY,
            "selected_market_id": args.instance_market.rsplit("/", 1)[-1],
            "selected_market": args.instance_market,
            "gpuModel": "",
            "gpuMemorySize": None,
            "pricePerGpu": None,
            "datacenter": smoke.DATACENTER,
            "reason_selected": "Manual --instance-market override; GPU details were not fetched from market selection.",
            "rejected_candidates_summary": [],
            "estimated_cost": gpu_policies.estimated_cost({}),
        }
        data["resolution_policy"] = gpu_policies.resolution_report("inference", GPU_POLICY)
        return args.instance_market

    query = urlencode(gpu_policies.market_query(GPU_POLICY))
    market_result = smoke.http_request(base_url, f"{smoke.MARKET_LIST_PATH}?{query}", api_key)
    items = smoke.extract_items(market_result.get("json"))
    selected = gpu_policies.select_market(items, GPU_POLICY)
    data["market_selection"] = {
        "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path")},
        "items_observed": len(items),
        "selected": selected,
        "selected_policy": GPU_POLICY,
        "selected_market_id": selected.get("selected_market_id", ""),
        "gpuModel": selected.get("selected_summary", {}).get("gpuModel", ""),
        "gpuMemorySize": selected.get("selected_summary", {}).get("gpuMemorySize"),
        "pricePerGpu": selected.get("selected_summary", {}).get("pricePerGpu"),
        "datacenter": selected.get("selected_summary", {}).get("datacenter"),
        "reason_selected": selected.get("reason_selected", ""),
        "rejected_candidates_summary": selected.get("rejected_candidates_summary", []),
        "estimated_cost": gpu_policies.estimated_cost(selected),
    }
    data["resolution_policy"] = gpu_policies.resolution_report(
        "inference",
        GPU_POLICY,
        selected.get("selected_summary", {}),
    )
    return gpu_policies.selected_market_iri(selected)


def wait_for_public_url(base_url: str, api_key: str, instance_id: int, args: argparse.Namespace, data: dict) -> str:
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    proxy_url = ""
    attempts = []
    for _ in range(max(1, args.detail_attempts)):
        detail_result = smoke.http_request(base_url, detail_path, api_key)
        attempts.append({key: detail_result.get(key) for key in ("status", "http_status_code", "error_type")})
        if isinstance(detail_result.get("json"), dict):
            selected_mapping = smoke.extract_api_port_mapping(detail_result["json"], smoke.PORT)
            if selected_mapping:
                data["selected_api_port_mapping"] = selected_mapping
            proxy_url = smoke.extract_proxy_url_for_port(detail_result["json"], smoke.PORT)
            if proxy_url:
                break
        time.sleep(args.poll_interval_seconds)
    data["detail_attempts"] = attempts
    data["public_api_base_url"] = proxy_url
    return proxy_url


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    runtime_seconds = data.get("runtime_seconds")
    market_selection = data.get("market_selection", {})
    selected = market_selection.get("selected") if isinstance(market_selection, dict) else {}
    actual_generation_resolution = (
        data["actual_generation_resolution"]
        if "actual_generation_resolution" in data
        else {"width": 1080, "height": 1080}
    )
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not (args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete),
        "template": {
            "id": TEMPLATE_ID,
            "iri": f"/instances/templates/{TEMPLATE_ID}",
            "required_image": IMAGE,
            "note": "Template must point at V2 image tag 0.1.5 before real execution.",
        },
        "inference_gate": {
            "endpoint": f"POST {INFERENCE_ENDPOINT}",
            "endpoint_mode": "real_single_job_no_scheduler",
            "requires_real_wan22_integration": False,
            "gpu_policy": GPU_POLICY,
            "not_allowed_gpu_note": "RTX 3060 and 24GB-class GPUs must not be used for this first Maé 1080 test; require production_single_job_policy >=48GB marketplace VRAM.",
            "downloads_model_weights": False,
            "placeholder_generated": False,
            "runs_inference": status == "succeeded",
        },
        "payload_dryrun": inference_payload(args.allow_oom_fallback),
        "r2_env_local_check": data.get("r2_env_local_check", local_r2_env_presence()),
        "instance_payload_dryrun": redact_instance_payload(runtime_payload(args.instance_market or "<selected_from_market_api>")),
        "gpu_selection_policy": gpu_policies.select_market([], GPU_POLICY),
        "resolution_policy": {
            **gpu_policies.resolution_report("inference", GPU_POLICY),
            "requested_resolution": {"width": 1080, "height": 1080},
            "actual_generation_resolution": actual_generation_resolution,
            "fallback_used": bool(data.get("fallback_used", False)),
            "oom_or_error_status": data.get("oom_or_error_status", ""),
        },
        "vram_policy": data.get(
            "vram_policy",
            gpu_policies.vram_policy_report(GPU_POLICY, {}, None),
        ),
        "runtime_seconds": runtime_seconds,
        "estimated_cost": gpu_policies.estimated_cost(selected if isinstance(selected, dict) else {}, runtime_seconds),
        "confirmations": {
            "execute": args.execute,
            "confirm_start": args.confirm_start,
            "confirm_inference": args.confirm_inference,
            "confirm_delete": args.confirm_delete,
        },
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "verify_called": bool(data.get("verify_result", {}).get("attempted")),
            "inference_endpoint_called": bool(data.get("inference_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "model_weights_downloaded": False,
            "placeholder_generated": False,
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
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    instance_id = None
    started_monotonic = time.monotonic()
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_inference and args.confirm_delete
        load_local_env()
        data["r2_env_local_check"] = local_r2_env_presence()
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] gpu_policy={GPU_POLICY} target_resolution=1080x1080")

        status = blocked_status(args)
        if status:
            with timer.phase("blocked_preflight"):
                pass
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not execute_allowed:
            status = "dry_run_ready"
            with timer.phase("dry_run_report"):
                pass
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        with timer.phase("load_auth"):
            load_local_env()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)
            data["r2_env_local_check"] = local_r2_env_presence()
        if not api_key:
            status = "missing_api_key"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        missing_r2 = missing_local_r2_env()
        if missing_r2:
            status = "missing_local_r2_env"
            data["missing_local_r2_env"] = missing_r2
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        try:
            with timer.phase("market_selection"):
                market = choose_market(args, base_url, api_key, data)
            if not market:
                status = "blocked_no_instance_market_selected"
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
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            with timer.phase("wait_public_url"):
                proxy_url = wait_for_public_url(base_url, api_key, instance_id, args, data)
            if not proxy_url:
                status = "blocked_no_proxy_url_for_port_8000"
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            with timer.phase("wait_health"):
                readiness, readiness_attempts, _ = smoke.wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
            data["api_readiness"] = {"status": readiness, "attempts": readiness_attempts}
            if readiness != "ready":
                status = "api_not_ready"
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            with timer.phase("gpu_check"):
                gpu_result = smoke.simple_get(proxy_url + "/gpu")
            data["gpu_check"] = {
                "status": gpu_result.get("status"),
                "http_status_code": gpu_result.get("http_status_code"),
                "summary": smoke.summarize_api_response(gpu_result.get("json")),
            }
            selected_summary = data.get("market_selection", {}).get("selected", {}).get("selected_summary", {})
            data["vram_policy"] = gpu_policies.vram_policy_report(
                GPU_POLICY,
                selected_summary,
                gpu_result.get("json") if isinstance(gpu_result, dict) else {},
            )
            if not data["vram_policy"].get("marketplace_policy_passed"):
                status = "blocked_marketplace_vram_below_48000mb_or_unknown"
                data["oom_or_error_status"] = status
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1
            if not gpu_passes_runtime_sanity(gpu_result.get("json")):
                status = "blocked_runtime_vram_below_46gib_or_unknown"
                data["oom_or_error_status"] = status
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            with timer.phase("verify_weights"):
                verify_result = smoke.simple_get(proxy_url + VERIFY_ENDPOINT)
            data["verify_result"] = {
                "attempted": True,
                "status": verify_result.get("status"),
                "http_status_code": verify_result.get("http_status_code"),
                "error_type": verify_result.get("error_type", ""),
                "error_truncated": verify_result.get("error_truncated", ""),
                "summary": summarize_verify(verify_result.get("json")),
            }
            if not verify_result_looks_successful(verify_result):
                status = "failed_verify_weights_before_inference"
                data["oom_or_error_status"] = status
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            with timer.phase("run_inference_endpoint"):
                inference_result = simple_post(
                    proxy_url + INFERENCE_ENDPOINT,
                    inference_payload(args.allow_oom_fallback),
                    args.inference_timeout_seconds,
                )
            body = inference_result.get("json")
            data["inference_result"] = {
                "attempted": True,
                "status": inference_result.get("status"),
                "http_status_code": inference_result.get("http_status_code"),
                "error_type": inference_result.get("error_type", ""),
                "error_truncated": inference_result.get("error_truncated", ""),
                "summary": summarize_inference(body),
            }
            if isinstance(body, dict):
                data["actual_generation_resolution"] = body.get("actual_generation_resolution")
                data["fallback_used"] = body.get("fallback_used", False)
                data["oom_or_error_status"] = body.get("status", "")

            status = "succeeded" if isinstance(body, dict) and body.get("video_generated") is True else "failed_inference_endpoint"
            data["_status_for_finally"] = status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0 if status == "succeeded" else 1
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
    except KeyboardInterrupt:
        status = "interrupted_delete_attempted" if instance_id is not None else "interrupted"
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, status, data, "KeyboardInterrupt"))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc)
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, "failed", data, message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute first SimplePod Maé Wan2.2 S2V inference gate.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and call the inference endpoint.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-inference", action="store_true", help="Required with --execute to call the inference endpoint.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; otherwise select by first inference GPU policy.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    parser.add_argument("--inference-timeout-seconds", type=int, default=7200)
    parser.add_argument("--allow-oom-fallback", action="store_true", help="Allow one 960x960 retry only after real 1080 OOM.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
