import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


TEST_ID = "TEMP_SIMPLEPOD_BLACKWELL_SMOKE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_blackwell_smoke_v1.json"

IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.0-blackwell"
DEFAULT_TEMPLATE_ID = int(os.getenv("SIMPLEPOD_BLACKWELL_TEMPLATE_ID", os.getenv("SIMPLEPOD_TEMPLATE_ID_V2", "25114")) or "25114")
DATACENTER = "EU-PL-01"
PORT = 8000
MARKET_LIST_PATH = "/instances/market/list"
BLACKWELL_MARKERS = ("Blackwell", "RTX PRO 6000")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gpu_model(item: dict) -> str:
    for key in ("gpuModel", "gpuName", "gpu"):
        value = item.get(key)
        if isinstance(value, str):
            return value
    return ""


def gpu_memory_mb(item: dict) -> int | None:
    value = item.get("gpuMemorySize")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def price_value(item: dict):
    for key in ("pricePerGpu", "pricePerHour", "price"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def market_iri(item: dict) -> str:
    for key in ("@id", "instanceMarket"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith("/instances/market/"):
            return value
    value = item.get("id")
    if isinstance(value, int):
        return f"/instances/market/{value}"
    if isinstance(value, str) and value.isdigit():
        return f"/instances/market/{value}"
    return ""


def market_id(item: dict) -> str:
    value = item.get("id")
    if value is not None:
        return str(value)
    iri = market_iri(item)
    return iri.rsplit("/", 1)[-1] if iri else ""


def candidate_summary(item: dict, reason: str = "") -> dict:
    return {
        "market_id": market_id(item),
        "market_iri": market_iri(item),
        "gpuModel": gpu_model(item),
        "gpuMemorySize": item.get("gpuMemorySize"),
        "gpuMemorySize_mb_normalized": gpu_memory_mb(item),
        "gpuCount": item.get("gpuCount"),
        "pricePerGpu": item.get("pricePerGpu"),
        "datacenter": item.get("datacenter") or item.get("region") or item.get("dataCenter"),
        "rentalStatus": item.get("rentalStatus") or item.get("status"),
        "reason": reason,
    }


def select_blackwell_market(items: list[dict]) -> dict:
    accepted = []
    rejected = []
    for item in items:
        if not isinstance(item, dict):
            continue
        iri = market_iri(item)
        model = gpu_model(item)
        memory_mb = gpu_memory_mb(item)
        gpu_count = item.get("gpuCount")
        text = json.dumps(item, ensure_ascii=False).lower()
        if not iri:
            rejected.append(candidate_summary(item, "missing_market_id"))
        elif str(item.get("rentalStatus") or item.get("status") or "active").lower() != "active":
            rejected.append(candidate_summary(item, "rentalStatus_not_active"))
        elif DATACENTER.lower() not in text:
            rejected.append(candidate_summary(item, "datacenter_not_matched"))
        elif gpu_count not in {1, "1"}:
            rejected.append(candidate_summary(item, "gpuCount_not_1"))
        elif memory_mb is None or memory_mb < 48_000:
            rejected.append(candidate_summary(item, "gpuMemorySize_below_48000"))
        elif not any(marker.lower() in model.lower() for marker in BLACKWELL_MARKERS):
            rejected.append(candidate_summary(item, "gpuModel_not_blackwell_target"))
        else:
            accepted.append({"item": item, "price": price_value(item), "summary": candidate_summary(item)})
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
        "selected_market": market_iri(selected_item),
        "selected_market_id": market_id(selected_item),
        "selected_summary": candidate_summary(selected_item, "lowest_price_blackwell_candidate") if selected_item else {},
        "accepted_candidates_observed": len(accepted),
        "rejected_candidates_observed": len(rejected),
        "rejected_candidates_summary": rejected[:20],
    }


def runtime_payload(instance_market: str, template_id: int) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_blackwell_market>",
        "instanceTemplate": f"/instances/templates/{template_id}",
        "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
        "envVariables": [
            {"name": "AYL_RUNTIME_VERSION", "value": "v2-blackwell-smoke"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
        ],
    }


def summarize_gpu(gpu_json) -> dict:
    if not isinstance(gpu_json, dict):
        return {"json_type": type(gpu_json).__name__}
    keys = (
        "torch_import_status",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "device_name",
        "device_capability",
        "device_capability_string",
        "vram_total_gb",
        "error_truncated",
    )
    return {key: gpu_json.get(key) for key in keys if key in gpu_json}


def validate_blackwell_gpu(gpu_json) -> dict:
    summary = summarize_gpu(gpu_json)
    device_name = str(summary.get("device_name") or "")
    capability = summary.get("device_capability")
    capability_string = str(summary.get("device_capability_string") or "")
    capability_ok = capability == [12, 0] or capability == (12, 0) or capability_string == "12.0"
    checks = {
        "torch_import_ok": summary.get("torch_import_status") == "ok",
        "cuda_available": summary.get("cuda_available") is True,
        "device_name_contains_blackwell": "blackwell" in device_name.lower(),
        "device_capability_is_12_0": capability_ok,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "gpu_summary": summary,
    }


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    execute_allowed = args.execute and args.confirm_start and args.confirm_delete
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": not execute_allowed,
        "read_only_until_execute": True,
        "image": IMAGE,
        "stable_image_unchanged": "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.6",
        "template_id": args.template_id,
        "template_note": "Template must point at the Blackwell experimental image before execute smoke.",
        "smoke_checks": ["/health", "/gpu", "torch_version", "torch_cuda_version", "torch.cuda.get_device_capability"],
        "safety_guards": {
            "runs_inference": False,
            "downloads_model_weights": False,
            "generates_video": False,
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "secrets_printed": False,
        },
        "instance_payload_dryrun": smoke.redact_value("", runtime_payload(args.instance_market or "<selected_blackwell_market>", args.template_id)),
        "phase_timings": data.get("phase_timings", []),
        "runtime": data,
    }


def blocked_status(args: argparse.Namespace) -> str:
    if args.execute and not args.confirm_start:
        return "blocked_missing_confirm_start"
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
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} image={IMAGE}")
        print(f"[{TEST_ID}] smoke_checks=/health,/gpu capability=12.0 no_inference=true")
        status = blocked_status(args)
        if status:
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not execute_allowed:
            with timer.phase("dry_run_report"):
                pass
            status = "dry_run_ready"
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

        if args.instance_market:
            market = args.instance_market
            data["market_selection"] = {
                "selected_market": market,
                "selected_market_id": market.rsplit("/", 1)[-1],
                "reason_selected": "Manual --instance-market override.",
            }
        else:
            with timer.phase("market_selection"):
                query = {
                    "mode": "docker",
                    "rentalStatus": "active",
                    "region": DATACENTER,
                    "gpuCount[gte]": 1,
                    "gpuCount[lte]": 1,
                    "gpuMemorySize[gte]": 48_000,
                    "itemsPerPage": 100,
                    "order[pricePerGpu]": "asc",
                }
                market_result = smoke.http_request(base_url, f"{MARKET_LIST_PATH}?{urlencode(query)}", api_key)
                items = smoke.extract_items(market_result.get("json"))
                selected = select_blackwell_market(items)
            data["market_selection"] = {
                "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path")},
                "items_observed": len(items),
                "selected": selected,
            }
            market = selected.get("selected_market", "")
        if not market:
            status = "blocked_no_blackwell_market_selected"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("start_instance"):
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=runtime_payload(market, args.template_id))
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
            proxy_url = ""
            detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
            for _ in range(max(1, args.detail_attempts)):
                detail_result = smoke.http_request(base_url, detail_path, api_key)
                if isinstance(detail_result.get("json"), dict):
                    selected_mapping = smoke.extract_api_port_mapping(detail_result["json"], PORT)
                    if selected_mapping:
                        data["selected_api_port_mapping"] = selected_mapping
                    proxy_url = smoke.extract_proxy_url_for_port(detail_result["json"], PORT)
                    if proxy_url:
                        break
                time.sleep(args.poll_interval_seconds)
        data["public_api_base_url"] = proxy_url
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
            "summary": summarize_gpu(gpu_result.get("json")),
            "blackwell_validation": validate_blackwell_gpu(gpu_result.get("json")),
        }
        status = "succeeded" if data["gpu_check"]["blackwell_validation"]["status"] == "passed" else "failed_blackwell_gpu_validation"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Blackwell sm_120 FastAPI smoke.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance for Blackwell smoke.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute; deletes instance in finally.")
    parser.add_argument("--template-id", type=int, default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
