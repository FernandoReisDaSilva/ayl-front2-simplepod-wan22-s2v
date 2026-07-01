import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import temp_simplepod_blackwell_smoke_v1 as blackwell_smoke
import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


TEST_ID = "TEMP_SIMPLEPOD_CHECK_SAFETENSORS_DEVICE_BLACKWELL_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_safetensors_device_blackwell_v1.json"

TEMPLATE_ID = 25138
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.2.11-blackwell"
DATACENTER = "EU-PL-01"
PORT = 8000
CHECK_ENDPOINT = "/admin/check-safetensors-device"
GPU_POLICY = "blackwell_full_96gb_diagnostic_policy"
RUNTIME_VERSION = "v2-blackwell-safetensors-device-diagnostic"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_full_blackwell_diagnostic_candidate(item: dict) -> tuple[bool, str]:
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
        return False, "MIG_rejected_for_diagnostic"
    return True, "full_RTX_PRO_6000_Blackwell_96GB_selected_for_safetensors_diagnostic"


def select_full_blackwell_market(items: list[dict]) -> dict:
    accepted = []
    rejected = []
    for item in items:
        if not isinstance(item, dict):
            continue
        accepted_candidate, reason = is_full_blackwell_diagnostic_candidate(item)
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
        "searched_datacenters": [DATACENTER],
        "selection_rule": "EU-PL-01 active gpuCount=1 gpuMemorySize>=90000 gpuModel contains RTX PRO 6000 Blackwell and rejects MIG.",
        "accepted_candidates_observed": len(accepted),
        "accepted_candidates_summary": [candidate["summary"] for candidate in accepted[:10]],
        "rejected_candidates_observed": len(rejected),
        "rejected_candidates_summary": rejected[:30],
        "reason_selected": (
            "Full RTX PRO 6000 Blackwell 96GB selected; MIG is not allowed for this diagnostic."
            if selected_item
            else "No full RTX PRO 6000 Blackwell 96GB market available in EU-PL-01."
        ),
    }


def runtime_payload(instance_market: str) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_full_blackwell_96gb_market>",
        "instanceTemplate": f"/instances/templates/{TEMPLATE_ID}",
        "envVariables": [
            {"name": "AYL_ENABLE_ADMIN_VERIFY", "value": "1"},
            {"name": "AYL_RUNTIME_VERSION", "value": RUNTIME_VERSION},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
        ],
    }


def select_market(args: argparse.Namespace, base_url: str, api_key: str, data: dict) -> str:
    if args.instance_market:
        data["market_selection"] = {
            "selected": {
                "selected_policy": GPU_POLICY,
                "selected_market": args.instance_market,
                "selected_market_id": args.instance_market.rsplit("/", 1)[-1],
                "reason_selected": "Manual --instance-market override; runtime /gpu still validates full Blackwell non-MIG.",
            },
            "selected_policy": GPU_POLICY,
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
        "reason_selected": selected.get("reason_selected", ""),
    }
    return selected.get("selected_market", "")


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
    device_name = str(gpu_json.get("device_name", "")) if isinstance(gpu_json, dict) else ""
    is_mig = "mig" in device_name.lower()
    return {
        "status": "passed" if validation.get("status") == "passed" and not is_mig else "blocked",
        "blackwell_validation": validation,
        "device_name": device_name,
        "mig_rejected": is_mig,
        "reason": "" if validation.get("status") == "passed" and not is_mig else "runtime GPU is not full Blackwell 96GB or appears to be MIG",
    }


def summarize_safetensors_check(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    summary = {
        "status": value.get("status"),
        "torch": value.get("torch", {}),
        "versions": value.get("versions", {}),
        "downloads_model_weights": value.get("downloads_model_weights"),
        "loads_full_model": value.get("loads_full_model"),
        "inference_executed": value.get("inference_executed"),
        "video_generated": value.get("video_generated"),
    }
    for key in (
        "load_file_cpu",
        "load_file_cuda0",
        "safe_open_cpu",
        "safe_open_cuda0",
    ):
        item = value.get(key)
        if isinstance(item, dict):
            summary[key] = {
                "status": item.get("status"),
                "error_type": item.get("error_type"),
                "error_truncated": item.get("error_truncated"),
                "result": item.get("result"),
            }
    patch = value.get("monkeypatch_cuda_to_cpu_simulation")
    if isinstance(patch, dict):
        summary["monkeypatch_cuda_to_cpu_simulation"] = {
            "enabled": patch.get("enabled"),
            "load_file_cuda0_redirected_status": (patch.get("load_file_cuda0_redirected") or {}).get("status"),
            "safe_open_cuda0_redirected_status": (patch.get("safe_open_cuda0_redirected") or {}).get("status"),
        }
    return summary


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
        "gpu_policy": GPU_POLICY,
        "selected_market_id": data.get("market_selection", {}).get("selected_market_id") or selected_summary.get("market_id", ""),
        "gpuModel": selected_summary.get("gpuModel", ""),
        "gpuMemorySize": selected_summary.get("gpuMemorySize"),
        "pricePerGpu": selected_summary.get("pricePerGpu"),
        "instance_id": data.get("instance_id"),
        "public_url": data.get("public_api_base_url", ""),
        "health_result": data.get("health_result", {}),
        "gpu_result": data.get("gpu_check", {}),
        "safetensors_device_result": data.get("safetensors_device_result", {}),
        "delete_result": data.get("delete_result", {}),
        "runtime_seconds": runtime_seconds,
        "estimated_cost": estimate_cost(selected_summary, runtime_seconds),
        "instance_payload_dryrun": smoke.redact_value("", runtime_payload(args.instance_market or "<selected_full_blackwell_96gb_market>")),
        "startScript_sent": False,
        "uses_image_cmd": True,
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "downloads_model_weights": False,
            "loads_full_model": False,
            "runs_inference": False,
            "generates_video": False,
            "calls_simplepod": bool(execute_allowed),
            "simplepod_start_called": bool(data.get("start_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "secrets_printed": False,
        },
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
    started_monotonic = time.monotonic()
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] image_required={IMAGE}")
        print(f"[{TEST_ID}] endpoint={CHECK_ENDPOINT} no_inference=true no_model_load=true")

        status = blocked_status(args)
        if status:
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("load_auth_env"):
            smoke.load_repo_dotenv()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)

        if api_key:
            with timer.phase("market_selection"):
                market = select_market(args, base_url, api_key, data)
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

        with timer.phase("safetensors_device_check"):
            check_result = smoke.simple_get(proxy_url + CHECK_ENDPOINT)
        data["safetensors_device_result"] = {
            "attempted": True,
            "status": check_result.get("status"),
            "http_status_code": check_result.get("http_status_code"),
            "error_type": check_result.get("error_type", ""),
            "error_truncated": check_result.get("error_truncated", ""),
            "summary": summarize_safetensors_check(check_result.get("json")),
            "json": check_result.get("json"),
        }
        body = check_result.get("json")
        status = "succeeded" if check_result.get("http_status_code") == 200 and isinstance(body, dict) else "failed_safetensors_device_check"
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
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod Blackwell safetensors device diagnostic.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance for safetensors diagnostic.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute; deletes instance in finally.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}.")
    parser.add_argument("--detail-attempts", type=int, default=36)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
