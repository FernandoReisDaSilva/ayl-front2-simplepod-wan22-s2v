import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import temp_simplepod_runtime_smoke_v2 as smoke


TEST_ID = "TEMP_SIMPLEPOD_DOWNLOAD_WAN22_S2V_WEIGHTS_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_wan22_s2v_weights_download_v1.json"

TEMPLATE_ID = 25114
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.1"
WAN22_S2V_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
HF_HOME = "/mnt/ayl_models/caches/huggingface"
ADMIN_ENDPOINT = "/admin/download-wan22-s2v-weights"
DOWNLOAD_CONFIRMATION = "DOWNLOAD_WAN22_S2V_WEIGHTS"
DOWNLOAD_COMMAND = (
    "huggingface-cli download "
    f"{WAN22_S2V_REPO_ID} "
    f"--local-dir {MODEL_DIR}"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_body(body: bytes, content_type: str):
    if "json" not in content_type.lower():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def summarize_response(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    summary = {}
    for key in sorted(value.keys()):
        item = value.get(key)
        if key in {"before", "after", "after_failure"} and isinstance(item, dict):
            summary[key] = {
                "path": item.get("path"),
                "exists": item.get("exists"),
                "is_dir": item.get("is_dir"),
                "file_count": item.get("file_count"),
                "total_bytes": item.get("total_bytes"),
                "sample_files": item.get("sample_files", [])[:20],
            }
        elif isinstance(item, (str, int, float, bool)) or item is None:
            summary[key] = item
        elif isinstance(item, dict):
            summary[key] = smoke.redact_value(key, item)
    return summary


def simple_post(url: str, payload: dict, timeout_seconds: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        method="POST",
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ayl-front2-simplepod-download-wan22-s2v-weights-v1",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            return {
                "status": "succeeded",
                "http_status_code": response.status,
                "content_type": content_type,
                "body_bytes": len(response_body),
                "json": parse_json_body(response_body, content_type),
            }
    except HTTPError as exc:
        response_body = exc.read(262_144)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return {
            "status": "failed",
            "http_status_code": exc.code,
            "content_type": content_type,
            "body_bytes": len(response_body),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
            "json": parse_json_body(response_body, content_type),
        }
    except URLError as exc:
        return {
            "status": "failed",
            "error_type": "URLError",
            "error_truncated": str(exc)[:1000],
        }


def runtime_payload(instance_market: str) -> dict:
    payload = smoke.runtime_payload(instance_market)
    payload["instanceTemplate"] = f"/instances/templates/{TEMPLATE_ID}"
    payload["envVariables"] = [
        {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
        {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
        {"name": "HF_HOME", "value": HF_HOME},
        {"name": "AYL_ENABLE_ADMIN_DOWNLOADS", "value": "1"},
        {"name": "AYL_RUNTIME_VERSION", "value": "v2-download-gate"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
    ]
    return payload


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not (args.execute and args.confirm_start and args.confirm_download and args.confirm_delete),
        "template": {
            "id": TEMPLATE_ID,
            "iri": f"/instances/templates/{TEMPLATE_ID}",
            "required_image": IMAGE,
            "note": "Template must point at V2 image tag 0.1.1 before real execution.",
        },
        "download_plan": {
            "repo_id": WAN22_S2V_REPO_ID,
            "target_dir": MODEL_DIR,
            "hf_home": HF_HOME,
            "admin_endpoint": f"POST {ADMIN_ENDPOINT}",
            "command_equivalent_inside_container": DOWNLOAD_COMMAND,
            "requires_new_image_tag": "0.1.1",
            "runs_inference": False,
            "generates_video": False,
            "prints_secrets": False,
        },
        "payload_dryrun": smoke.redact_value("", runtime_payload(args.instance_market or "<selected_from_market_api>")),
        "confirmations": {
            "execute": args.execute,
            "confirm_start": args.confirm_start,
            "confirm_download": args.confirm_download,
            "confirm_delete": args.confirm_delete,
        },
        "safety_guards": {
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "download_called": bool(data.get("download_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "inference_executed": False,
            "video_generated": False,
            "secrets_printed": False,
        },
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
    if args.execute and not args.confirm_download:
        return "blocked_missing_confirm_download"
    if args.execute and not args.confirm_delete:
        return "blocked_missing_confirm_delete"
    return ""


def choose_market(args: argparse.Namespace, base_url: str, api_key: str, data: dict) -> str:
    if args.instance_market:
        return args.instance_market
    query = urlencode(
        {
            "mode": "docker",
            "rentalStatus": "active",
            "region": smoke.DATACENTER,
            "gpuCount[gte]": 1,
            "gpuCount[lte]": 1,
            "itemsPerPage": 100,
            "order[pricePerGpu]": "asc",
        }
    )
    market_result = smoke.http_request(base_url, f"{smoke.MARKET_LIST_PATH}?{query}", api_key)
    items = smoke.extract_items(market_result.get("json"))
    selected = smoke.select_lowest_cost_market(items, smoke.DATACENTER)
    data["market_selection"] = {
        "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path")},
        "items_observed": len(items),
        "selected": selected,
    }
    return selected.get("iri", "")


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


def run(args: argparse.Namespace) -> int:
    data = {}
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_download and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] command_inside_container={DOWNLOAD_COMMAND}")

        status = blocked_status(args)
        if status:
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not execute_allowed:
            status = "dry_run_ready"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        smoke.load_repo_dotenv()
        api_key = os.getenv(smoke.API_KEY_ENV, "")
        base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)
        if not api_key:
            status = "missing_api_key"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        instance_id = None
        try:
            market = choose_market(args, base_url, api_key, data)
            if not market:
                status = "blocked_no_instance_market_selected"
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

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
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            proxy_url = wait_for_public_url(base_url, api_key, instance_id, args, data)
            if not proxy_url:
                status = "blocked_no_proxy_url_for_port_8000"
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            readiness, readiness_attempts, _ = smoke.wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
            data["api_readiness"] = {"status": readiness, "attempts": readiness_attempts}
            if readiness != "ready":
                status = "api_not_ready"
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            before_models = smoke.simple_get(proxy_url + "/models")
            data["models_before_download"] = {
                "status": before_models.get("status"),
                "http_status_code": before_models.get("http_status_code"),
                "summary": smoke.summarize_api_response(before_models.get("json")),
            }

            download_payload = {
                "confirm_download": DOWNLOAD_CONFIRMATION,
                "target_dir": MODEL_DIR,
            }
            download_result = simple_post(proxy_url + ADMIN_ENDPOINT, download_payload, args.download_timeout_seconds)
            data["download_result"] = {
                "attempted": True,
                "status": download_result.get("status"),
                "http_status_code": download_result.get("http_status_code"),
                "error_type": download_result.get("error_type", ""),
                "error_truncated": download_result.get("error_truncated", ""),
                "summary": summarize_response(download_result.get("json")),
            }
            if download_result.get("status") != "succeeded" or download_result.get("http_status_code") != 200:
                status = "download_failed"
                data["_status_for_finally"] = status
                write_json(REPORT_PATH, build_report(args, status, data))
                print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
                return 1

            after_models = smoke.simple_get(proxy_url + "/models")
            data["models_after_download"] = {
                "status": after_models.get("status"),
                "http_status_code": after_models.get("http_status_code"),
                "summary": smoke.summarize_api_response(after_models.get("json")),
            }
            status = "download_succeeded"
            data["_status_for_finally"] = status
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0
        finally:
            if instance_id is not None:
                delete_path = smoke.DELETE_INSTANCE_PATH.format(id=instance_id)
                delete_result = smoke.http_request(base_url, delete_path, api_key, method="DELETE")
                data["delete_result"] = {
                    key: delete_result.get(key)
                    for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
                }
                final_status = data.get("_status_for_finally")
                if final_status:
                    write_json(REPORT_PATH, build_report(args, final_status, data))
    except Exception as exc:
        message = str(exc)
        write_json(REPORT_PATH, build_report(args, "failed", data, message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Wan2.2 S2V weight download gate.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance and download weights.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-download", action="store_true", help="Required with --execute to call the admin download endpoint.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; otherwise select lowest-cost observed market.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    parser.add_argument("--download-timeout-seconds", type=int, default=7200)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
