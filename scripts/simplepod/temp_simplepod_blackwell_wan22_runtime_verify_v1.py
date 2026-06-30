import argparse
import json
import os
import time
from pathlib import Path

import temp_simplepod_blackwell_smoke_v1 as blackwell_smoke
import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


TEST_ID = "TEMP_SIMPLEPOD_BLACKWELL_WAN22_RUNTIME_VERIFY_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_blackwell_wan22_runtime_verify_v1.json"

TEMPLATE_ID = 25138
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.2-blackwell"
STABLE_IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.6"
VERIFY_ENDPOINT = "/admin/verify-wan22-s2v-runtime"
PORT = 8000


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def runtime_payload(instance_market: str) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_blackwell_market>",
        "instanceTemplate": f"/instances/templates/{TEMPLATE_ID}",
        "envVariables": [
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-wan22-runtime-verify"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
            {"name": "AYL_ENABLE_ADMIN_VERIFY", "value": "1"},
        ],
    }


def summarize_verify(value) -> dict:
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
        "required_files_found",
        "wan_code_import_status",
        "wan_code_import_error_type",
        "wan_code_import_error_truncated",
        "wan_code_import_traceback_tail",
        "wan_code_import_attempted_modules",
        "sys_path_tail",
        "wan_repo_path_exists",
        "wan_repo_path",
        "cwd",
        "python_version",
        "installed_packages_relevant",
        "runner_import_status",
        "download_attempted",
        "inference_executed",
        "video_generated",
        "placeholder_generated",
    )
    return {key: value.get(key) for key in keys if key in value}


def verify_passed(result: dict) -> bool:
    value = result.get("json")
    return (
        result.get("http_status_code") == 200
        and isinstance(value, dict)
        and value.get("status") == "verified"
        and value.get("download_attempted") is False
        and value.get("inference_executed") is False
        and value.get("video_generated") is False
    )


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


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    execute_allowed = args.execute and args.confirm_start and args.confirm_delete
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
        "stable_image_unchanged": STABLE_IMAGE,
        "requires_new_image_tag": "0.2.2-blackwell",
        "why_new_tag_needed": "Adds detailed Wan2.2 import diagnostics to GET /admin/verify-wan22-s2v-runtime.",
        "selected_market_id": data.get("market_selection", {}).get("selected_market_id") or selected_summary.get("market_id", ""),
        "gpuModel": selected_summary.get("gpuModel", ""),
        "gpuMemorySize": selected_summary.get("gpuMemorySize"),
        "pricePerGpu": selected_summary.get("pricePerGpu"),
        "instance_id": data.get("instance_id"),
        "public_url": data.get("public_api_base_url", ""),
        "health_result": data.get("health_result", {}),
        "gpu_result": data.get("gpu_check", {}),
        "runtime_verify_result": data.get("runtime_verify_result", {}),
        "delete_result": data.get("delete_result", {}),
        "runtime_seconds": runtime_seconds,
        "estimated_cost": estimate_cost(selected_summary, runtime_seconds),
        "startScript_sent": False,
        "uses_image_cmd": True,
        "docker_entrypoint_arguments_sent": False,
        "instance_payload_dryrun": smoke.redact_value("", runtime_payload(args.instance_market or "<selected_blackwell_market>")),
        "safety_guards": {
            "downloads_model_weights": False,
            "runs_inference": False,
            "generates_video": False,
            "placeholder_generated": False,
            "calls_jobs_wan22_s2v_run": False,
            "calls_download_endpoint": False,
            "calls_simplepod": bool(execute_allowed),
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "deletes_instance": bool(data.get("delete_result", {}).get("attempted")),
            "secrets_printed": False,
        },
        "phase_timings": data.get("phase_timings", []),
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def wait_for_public_url(base_url: str, api_key: str, instance_id: int, args: argparse.Namespace, data: dict) -> str:
    proxy_url = ""
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
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


def run(args: argparse.Namespace) -> int:
    data = {}
    timer = PhaseTimer()
    data["phase_timings"] = timer.phases
    instance_id = None
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    started_monotonic = time.monotonic()
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] no_downloads=true no_inference=true no_video=true")

        status = blocked_status(args)
        if status:
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("load_auth"):
            smoke.load_repo_dotenv()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)

        market = args.instance_market
        if args.instance_market:
            data["market_selection"] = {
                "selected_market": market,
                "selected_market_id": market.rsplit("/", 1)[-1],
                "reason_selected": "Manual --instance-market override.",
            }
        elif api_key:
            with timer.phase("market_selection"):
                query = {
                    "mode": "docker",
                    "rentalStatus": "active",
                    "region": blackwell_smoke.DATACENTER,
                    "gpuCount[gte]": 1,
                    "gpuCount[lte]": 1,
                    "gpuMemorySize[gte]": 48_000,
                    "itemsPerPage": 100,
                    "order[pricePerGpu]": "asc",
                }
                market_result = smoke.http_request(base_url, f"{blackwell_smoke.MARKET_LIST_PATH}?{blackwell_smoke.urlencode(query)}", api_key)
                items = smoke.extract_items(market_result.get("json"))
                selected = blackwell_smoke.select_blackwell_market(items)
            data["market_selection"] = {
                "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path", "error_type", "error_truncated")},
                "items_observed": len(items),
                "selected": selected,
            }
            market = selected.get("selected_market", "")
        else:
            data["market_selection"] = {
                "status": "skipped_missing_api_key",
                "selected": blackwell_smoke.select_blackwell_market([]),
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
        if not market:
            status = "blocked_no_blackwell_market_selected"
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
            "blackwell_validation": blackwell_smoke.validate_blackwell_gpu(gpu_result.get("json")),
        }
        if data["gpu_check"]["blackwell_validation"].get("status") != "passed":
            status = "failed_blackwell_gpu_validation"
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
            "summary": summarize_verify(verify_result.get("json")),
        }
        status = "succeeded" if verify_passed(verify_result) else "failed_wan22_runtime_verify"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute Blackwell Wan2.2 runtime verify without inference.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and run runtime verify.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute; deletes instance in finally.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
