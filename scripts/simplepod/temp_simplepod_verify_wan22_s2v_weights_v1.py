import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import simplepod_gpu_policies as gpu_policies
from simplepod_phase_timing import PhaseTimer, now_iso
import temp_simplepod_runtime_smoke_v2 as smoke


TEST_ID = "TEMP_SIMPLEPOD_VERIFY_WAN22_S2V_WEIGHTS_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_wan22_s2v_weights_verify_v1.json"

TEMPLATE_ID = 25114
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.2"
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"
VERIFY_ENDPOINT = "/admin/verify-wan22-s2v-weights"
GPU_POLICY = "download_gpu_policy"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize_verify(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    return {
        "status": value.get("status"),
        "target_dir": value.get("target_dir"),
        "exists": value.get("exists"),
        "is_dir": value.get("is_dir"),
        "recursive_file_count": value.get("recursive_file_count"),
        "recursive_total_size_bytes": value.get("recursive_total_size_bytes"),
        "recursive_total_size_gb": value.get("recursive_total_size_gb"),
        "safetensors_files": value.get("safetensors_files", [])[:30],
        "marker_files": value.get("marker_files", []),
        "downloads_attempted": value.get("downloads_attempted"),
        "inference_executed": value.get("inference_executed"),
        "video_generated": value.get("video_generated"),
    }


def verify_result_looks_successful(verify_result: dict) -> bool:
    body = verify_result.get("json")
    if not isinstance(body, dict):
        return False
    return (
        verify_result.get("status") == "succeeded"
        and verify_result.get("http_status_code") == 200
        and body.get("exists") is True
        and body.get("is_dir") is True
        and int(body.get("recursive_file_count") or 0) > 0
        and int(body.get("recursive_total_size_bytes") or 0) > 0
    )


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
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-verify-weights-gate"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
    }


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
        data["resolution_policy"] = gpu_policies.resolution_report("check", GPU_POLICY)
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
        "check",
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
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not (args.execute and args.confirm_start and args.confirm_verify and args.confirm_delete),
        "template": {
            "id": TEMPLATE_ID,
            "iri": f"/instances/templates/{TEMPLATE_ID}",
            "required_image": IMAGE,
            "note": "Template must point at V2 image tag 0.1.2 before real execution.",
        },
        "verify_plan": {
            "target_dir": MODEL_DIR,
            "verify_endpoint": f"GET {VERIFY_ENDPOINT}",
            "gpu_policy": GPU_POLICY,
            "downloads_model_weights": False,
            "runs_inference": False,
            "generates_video": False,
            "prints_secrets": False,
        },
        "payload_dryrun": smoke.redact_value("", runtime_payload(args.instance_market or "<selected_from_market_api>")),
        "gpu_selection_policy": gpu_policies.select_market([], GPU_POLICY),
        "resolution_policy": gpu_policies.resolution_report("check", GPU_POLICY),
        "confirmations": {
            "execute": args.execute,
            "confirm_start": args.confirm_start,
            "confirm_verify": args.confirm_verify,
            "confirm_delete": args.confirm_delete,
        },
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "verify_called": bool(data.get("verify_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "model_weights_downloaded": False,
            "inference_executed": False,
            "video_generated": False,
            "secrets_printed": False,
        },
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_verify:
        return "blocked_missing_confirm_verify"
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
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_verify and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] verify_endpoint=GET {VERIFY_ENDPOINT}")

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
            smoke.load_repo_dotenv()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)
        if not api_key:
            status = "missing_api_key"
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
                payload = runtime_payload(market)
                start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=payload)
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
            status = "succeeded" if verify_result_looks_successful(verify_result) else "failed_verify"
            data["_status_for_finally"] = status
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
                    write_json(REPORT_PATH, build_report(args, final_status, data))
    except KeyboardInterrupt:
        status = "interrupted_delete_attempted" if instance_id is not None else "interrupted"
        write_json(REPORT_PATH, build_report(args, status, data, "KeyboardInterrupt"))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc)
        write_json(REPORT_PATH, build_report(args, "failed", data, message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Wan2.2 S2V weights verification gate.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and verify weights.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-verify", action="store_true", help="Required with --execute to call the admin verify endpoint.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; otherwise select by GPU policy.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
