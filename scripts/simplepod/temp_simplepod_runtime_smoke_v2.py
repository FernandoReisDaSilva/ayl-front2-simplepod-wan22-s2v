import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


TEST_ID = "TEMP_SIMPLEPOD_RUNTIME_SMOKE_V2"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_runtime_smoke_v2.json"

API_KEY_ENV = "SIMPLEPOD_API_KEY"
BASE_URL_ENV = "SIMPLEPOD_API_BASE_URL"
DEFAULT_BASE_URL = "https://api.simplepod.ai"
AUTH_HEADER = "X-AUTH-TOKEN"
DOCS_URL = "https://api.simplepod.ai/docs"
SENSITIVE_KEY_PARTS = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "HASHID",
    "SSH",
    "AUTH",
)
NETWORK_KEY_PARTS = (
    "PORT",
    "PROXY",
    "EXPOSE",
    "NETWORK",
    "TUNNEL",
    "URL",
    "HOST",
    "IP",
    "STATUS",
    "STATE",
    "MODE",
    "CONTAINER",
)

TEMPLATE_ID = int(os.getenv("SIMPLEPOD_TEMPLATE_ID_V2", "0") or "0")
TEMPLATE_NAME = "ayl-wan22-s2v-fastapi-v2"
IMAGE = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:0.1.0"
DATACENTER = "EU-PL-01"
VOLUME_NAME = "ayl_models_wan22_s2v_v1"
VOLUME_MOUNT_PATH = "/mnt/ayl_models"
PORT = 8000

MARKET_LIST_PATH = "/instances/market/list"
START_INSTANCE_PATH = "/instances"
INSTANCE_DETAIL_PATH = "/instances/{id}"
DELETE_INSTANCE_PATH = "/instances/{id}"
SMOKE_ENDPOINTS = ("/health", "/gpu", "/models")
EXPECTED_API_PORT_MAPPING_FROM_INSPECT = {
    "source": "ports.direct.ip_destPort",
    "srcPort": 8000,
    "destPort": "20008",
    "ip": "194.93.49.14",
    "service": "PORT-8000",
    "protocol": "unknown",
    "url": "",
    "selected_url": "http://194.93.49.14:20008",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_repo_dotenv() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and not os.environ.get(key):
            os.environ[key] = value


def normalize_base_url(url: str) -> str:
    return url.rstrip("/") + "/" if url else ""


def endpoint_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url.split("/")[0]


def redact_value(key: str, value):
    if any(token in key.upper() for token in SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, dict):
        return {item_key: redact_value(str(item_key), item_value) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_value(key, item) for item in value]
    return value


def http_request(
    base_url: str,
    path: str,
    api_key: str = "",
    method: str = "GET",
    payload: dict | None = None,
    headers: dict | None = None,
    timeout_seconds: int = 30,
) -> dict:
    url = urljoin(normalize_base_url(base_url), path.lstrip("/"))
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "ayl-front2-simplepod-runtime-smoke-v2",
    }
    if api_key:
        request_headers[AUTH_HEADER] = api_key
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)

    request = Request(url, method=method, data=body, headers=request_headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            parsed = parse_json_body(response_body, content_type)
            return {
                "attempted": True,
                "status": "succeeded",
                "method": method,
                "path": path,
                "http_status_code": response.status,
                "endpoint_host": endpoint_host(url),
                "content_type": content_type,
                "body_bytes": len(response_body),
                "json": parsed,
            }
    except HTTPError as exc:
        response_body = exc.read(65_536)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return {
            "attempted": True,
            "status": "failed",
            "method": method,
            "path": path,
            "http_status_code": exc.code,
            "endpoint_host": endpoint_host(url),
            "content_type": content_type,
            "body_bytes": len(response_body),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
            "json": parse_json_body(response_body, content_type),
        }
    except URLError as exc:
        return {
            "attempted": True,
            "status": "failed",
            "method": method,
            "path": path,
            "endpoint_host": endpoint_host(url),
            "error_type": "URLError",
            "error_truncated": str(exc)[:1000],
        }


def parse_json_body(body: bytes, content_type: str):
    if "json" not in content_type.lower():
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def extract_items(value) -> list:
    if isinstance(value, list):
        return value
    if not isinstance(value, dict):
        return []
    for key in ("hydra:member", "member", "items", "data", "results", "instances", "markets"):
        items = value.get(key)
        if isinstance(items, list):
            return items
    return []


def safe_market_summary(item: dict) -> dict:
    allowed = (
        "id",
        "@id",
        "instanceMarket",
        "name",
        "gpuModel",
        "gpuCount",
        "gpuMemorySize",
        "pricePerGpu",
        "pricePerHour",
        "price",
        "region",
        "datacenter",
        "rentalStatus",
        "status",
    )
    return {key: item.get(key) for key in allowed if key in item}


def price_value(item: dict) -> float | None:
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
    return ""


def select_lowest_cost_market(items: list[dict], datacenter: str) -> dict:
    candidates = []
    for item in items:
        if not isinstance(item, dict):
            continue
        iri = market_iri(item)
        if not iri:
            continue
        item_text = json.dumps(item, ensure_ascii=False).lower()
        datacenter_match = datacenter.lower() in item_text
        price = price_value(item)
        candidates.append(
            {
                "iri": iri,
                "price": price,
                "datacenter_match": datacenter_match,
                "summary": safe_market_summary(item),
            }
        )
    candidates.sort(
        key=lambda item: (
            not item["datacenter_match"],
            item["price"] is None,
            item["price"] if item["price"] is not None else 999999,
            item["iri"],
        )
    )
    return candidates[0] if candidates else {}


def runtime_payload(instance_market: str) -> dict:
    template_iri = f"/instances/templates/{TEMPLATE_ID}" if TEMPLATE_ID else "<set_--template-id-before_execute>"
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_from_GET_/instances/market/list>",
        "instanceTemplate": template_iri,
        "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": VOLUME_MOUNT_PATH},
            {"name": "WAN22_S2V_MODEL_DIR", "value": f"{VOLUME_MOUNT_PATH}/wan2.2/Wan2.2-S2V-14B"},
            {"name": "AYL_RUNTIME_SMOKE_ONLY", "value": "1"},
            {"name": "AYL_RUNTIME_VERSION", "value": "v2"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
    }


def extract_instance_id(value) -> int | None:
    if isinstance(value, dict):
        for key in ("id", "instanceId"):
            if isinstance(value.get(key), int):
                return value[key]
        for item in value.values():
            found = extract_instance_id(item)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = extract_instance_id(item)
            if found is not None:
                return found
    return None


def src_port_matches(value, port: int) -> bool:
    if isinstance(value, int):
        return value == port
    if isinstance(value, str):
        return value.strip() == str(port)
    return False


def safe_port_mapping(item: dict, source: str, built_url: str) -> dict:
    return {
        "source": source,
        "srcPort": item.get("srcPort"),
        "destPort": item.get("destPort"),
        "ip": item.get("ip"),
        "service": item.get("service"),
        "protocol": item.get("protocol"),
        "url": item.get("url"),
        "selected_url": built_url,
    }


def extract_api_port_mapping(instance: dict, port: int) -> dict:
    ports = instance.get("ports") if isinstance(instance, dict) else None
    if isinstance(ports, dict):
        direct_ports = ports.get("direct") if isinstance(ports.get("direct"), list) else []
        proxy_ports = ports.get("proxy") if isinstance(ports.get("proxy"), list) else []
    elif isinstance(ports, list):
        direct_ports = ports
        proxy_ports = []
    else:
        direct_ports = []
        proxy_ports = []

    for item in direct_ports:
        if not isinstance(item, dict) or not src_port_matches(item.get("srcPort"), port):
            continue
        if str(item.get("srcPort", "")).endswith("console"):
            continue
        raw_url = str(item.get("url") or "").strip()
        if raw_url.startswith(("http://", "https://")):
            return safe_port_mapping(item, "ports.direct.url", raw_url.rstrip("/"))
        ip = str(item.get("ip") or "").strip()
        dest_port = str(item.get("destPort") or "").strip()
        if ip and dest_port:
            return safe_port_mapping(item, "ports.direct.ip_destPort", f"http://{ip}:{dest_port}")

    for item in proxy_ports:
        if not isinstance(item, dict) or not src_port_matches(item.get("srcPort"), port):
            continue
        raw_url = str(item.get("url") or "").strip()
        if raw_url and raw_url.lower() != "closed" and raw_url.startswith(("http://", "https://")):
            return safe_port_mapping(item, "ports.proxy.url", raw_url.rstrip("/"))

    return {}


def extract_proxy_url_for_port(instance: dict, port: int) -> str:
    selected = extract_api_port_mapping(instance, port)
    if selected.get("selected_url"):
        return selected["selected_url"].rstrip("/")
    return ""


def find_proxy_urls(value, path: str = "") -> list[dict]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if isinstance(item, str) and item.startswith(("http://", "https://")):
                if any(token in key.upper() for token in ("PROXY", "URL", "TUNNEL")):
                    found.append({"path": child_path, "value": item, "context": safe_scalar_context(value)})
            found.extend(find_proxy_urls(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_proxy_urls(item, f"{path}[{index}]"))
    return found


def safe_scalar_context(value: dict) -> dict:
    context = {}
    if not isinstance(value, dict):
        return context
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)) or item is None:
            context[key] = redact_value(str(key), item)
    return context


def relevant_field_items(value, path: str = "", depth: int = 0) -> list[dict]:
    if depth > 8:
        return []
    items = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            key_upper = key_text.upper()
            is_relevant = any(token in key_upper for token in NETWORK_KEY_PARTS)
            if is_relevant:
                items.append(
                    {
                        "path": child_path,
                        "value": truncate_safe_value(redact_value(key_text, item)),
                    }
                )
            items.extend(relevant_field_items(item, child_path, depth + 1))
    elif isinstance(value, list):
        for index, item in enumerate(value[:20]):
            items.extend(relevant_field_items(item, f"{path}[{index}]", depth + 1))
    return items


def truncate_safe_value(value):
    if isinstance(value, dict):
        return {
            str(key): truncate_safe_value(item)
            for key, item in list(value.items())[:30]
        }
    if isinstance(value, list):
        return [truncate_safe_value(item) for item in value[:20]]
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "...<truncated>"
    return value


def safe_instance_inspection(value) -> dict:
    if not isinstance(value, dict):
        return {"json_type": type(value).__name__}
    return {
        "top_level_keys": sorted(str(key) for key in value.keys()),
        "status_fields": {
            key: redact_value(key, value.get(key))
            for key in ("id", "name", "status", "state", "mode", "createdAt", "updatedAt", "gpuModel", "gpuCount")
            if key in value
        },
        "port_proxy_related_fields": relevant_field_items(value),
        "proxy_url_candidates": find_proxy_urls(value),
    }


def safe_instance_list_inspection(value) -> dict:
    items = extract_items(value)
    return {
        "top_level_type": type(value).__name__,
        "items_observed": len(items),
        "safe_items": [
            safe_instance_inspection(item)
            for item in items[:10]
            if isinstance(item, dict)
        ],
    }


def likely_proxy_hypothesis(detail_inspection: dict, list_inspection: dict | None = None) -> str:
    fields = detail_inspection.get("port_proxy_related_fields") or []
    urls = detail_inspection.get("proxy_url_candidates") or []
    if urls:
        return "proxyUrl appears to exist but the previous extractor did not match it reliably."
    field_paths = [str(item.get("path", "")).lower() for item in fields]
    if not any("port" in path or "proxy" in path or "expose" in path for path in field_paths):
        return "Most likely no public port mapping was generated for port 8000 in the instance details."
    if any("exposeportmappings" in path for path in field_paths):
        return "Likely the template/runtime needs exposePortMappings rather than only exposePorts."
    if any("status" in path or "state" in path for path in field_paths):
        return "Instance details were available, but port/proxy fields need inspection from the saved safe detail payload."
    if list_inspection:
        return "Instance detail did not expose proxyUrl; inspect GET /instances/list safe fields for alternate location."
    return "Undetermined from safe fields; run inspect-only to capture GET /instances/list as well."


def summarize_api_response(value) -> dict:
    if isinstance(value, dict):
        return redact_value("", {key: value.get(key) for key in sorted(value.keys())[:20]})
    return {"json_type": type(value).__name__}


def wait_for_instance_api(base_url: str, timeout_seconds: int) -> tuple[str, list[dict], dict]:
    attempts = []
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = simple_get(base_url + "/health", timeout_seconds=10)
        attempts.append(
            {
                "url": base_url + "/health",
                "status": result.get("status"),
                "http_status_code": result.get("http_status_code"),
                "error_type": result.get("error_type", ""),
            }
        )
        if result.get("status") == "succeeded" and result.get("http_status_code") == 200:
            return "ready", attempts, result
        time.sleep(5)
    return "timeout", attempts, {}


def simple_get(url: str, timeout_seconds: int = 20) -> dict:
    request = Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            return {
                "status": "succeeded",
                "http_status_code": response.status,
                "content_type": content_type,
                "json": parse_json_body(body, content_type),
                "body_bytes": len(body),
            }
    except Exception as exc:
        return {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
        }


def build_report(args: argparse.Namespace, status: str, data: dict, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "docs_url": DOCS_URL,
        "template": {
            "id": TEMPLATE_ID,
            "iri": f"/instances/templates/{TEMPLATE_ID}" if TEMPLATE_ID else "",
            "name": TEMPLATE_NAME,
            "image": IMAGE,
        },
        "volume": {
            "name": VOLUME_NAME,
            "datacenter": DATACENTER,
            "mount_path": VOLUME_MOUNT_PATH,
            "api_attach_status": "not_documented_in_POST_/instances_body",
            "operator_note": "If SimplePod requires explicit Network Drive selection, use the panel/template UI; the REST rent body inspected exposes instanceMarket, instanceTemplate, gpuCount, startScript, and envVariables.",
        },
        "identified_endpoints": {
            "market_list": "GET /instances/market/list?rentalStatus=active",
            "start_instance": "POST /instances",
            "instance_details": "GET /instances/{id}",
            "delete_instance": "DELETE /instances/{id}",
            "port_mapping_source": "ports field from GET /instances/{id} or GET /instances/list",
        },
        "execute_requested": args.execute,
        "confirm_start": args.confirm_start,
        "confirm_delete": args.confirm_delete,
        "inspect_only": args.inspect_only,
        "dry_run": not (args.execute and args.confirm_start and args.confirm_delete),
        "payload_dryrun": redact_value("", runtime_payload(args.instance_market)),
        "expected_selected_api_port_mapping_from_last_inspect": EXPECTED_API_PORT_MAPPING_FROM_INSPECT,
        "safety_guards": {
            "simplepod_write_called": bool(data.get("start_result", {}).get("attempted")),
            "delete_attempted": bool(data.get("delete_result", {}).get("attempted")),
            "model_weights_downloaded": False,
            "inference_executed": False,
            "secrets_printed": False,
        },
        "runtime": data,
    }


def run(args: argparse.Namespace) -> int:
    data = {}
    try:
        execute_allowed = args.execute and args.confirm_start and args.confirm_delete
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} template_id={TEMPLATE_ID}")
        print(f"[{TEST_ID}] start_endpoint=POST /instances delete_endpoint=DELETE /instances/{{id}}")
        if args.execute and not args.confirm_start:
            status = "blocked_missing_confirm_start"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if args.execute and not args.confirm_delete:
            status = "blocked_missing_confirm_delete"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if args.execute and not TEMPLATE_ID:
            status = "blocked_missing_template_id_v2"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not execute_allowed:
            status = "dry_run_ready"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] volume_attach=not_documented_in_POST_/instances_body")
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        load_repo_dotenv()
        api_key = os.getenv(API_KEY_ENV, "")
        base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL)
        if not api_key:
            status = "missing_api_key"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        market = args.instance_market
        if not market:
            query = urlencode(
                {
                    "mode": "docker",
                    "rentalStatus": "active",
                    "region": DATACENTER,
                    "gpuCount[gte]": 1,
                    "gpuCount[lte]": 1,
                    "itemsPerPage": 100,
                    "order[pricePerGpu]": "asc",
                }
            )
            market_result = http_request(base_url, f"{MARKET_LIST_PATH}?{query}", api_key)
            items = extract_items(market_result.get("json"))
            selected = select_lowest_cost_market(items, DATACENTER)
            data["market_selection"] = {
                "result": {key: market_result.get(key) for key in ("status", "http_status_code", "path")},
                "items_observed": len(items),
                "selected": selected,
                "cost_estimate_source": "SimplePod market API response" if selected.get("price") is not None else "",
            }
            market = selected.get("iri", "")
        if not market:
            status = "blocked_no_instance_market_selected"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        payload = runtime_payload(market)
        start_result = http_request(base_url, START_INSTANCE_PATH, api_key, method="POST", payload=payload)
        data["start_result"] = {
            key: start_result.get(key)
            for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
        }
        instance_id = extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        detail_path = INSTANCE_DETAIL_PATH.format(id=instance_id)
        proxy_url = ""
        detail_attempts = []
        for _ in range(max(1, args.detail_attempts)):
            detail_result = http_request(base_url, detail_path, api_key)
            detail_attempts.append(
                {
                    key: detail_result.get(key)
                    for key in ("status", "http_status_code", "error_type", "error_truncated")
                }
            )
            if isinstance(detail_result.get("json"), dict):
                data["latest_detail_json"] = detail_result["json"]
                selected_mapping = extract_api_port_mapping(detail_result["json"], PORT)
                if selected_mapping:
                    data["selected_api_port_mapping"] = selected_mapping
                proxy_url = extract_proxy_url_for_port(detail_result["json"], PORT)
                if proxy_url:
                    break
            time.sleep(args.poll_interval_seconds)
        data["detail_attempts"] = detail_attempts
        data["public_api_base_url"] = proxy_url
        if "latest_detail_json" in data:
            detail_inspection = safe_instance_inspection(data["latest_detail_json"])
            data["instance_detail_safe_inspection"] = detail_inspection
            data.pop("latest_detail_json", None)

        if args.inspect_only or not proxy_url:
            list_result = http_request(base_url, "/instances/list", api_key)
            list_inspection = safe_instance_list_inspection(list_result.get("json"))
            data["instance_list_safe_inspection"] = {
                "request": {
                    key: list_result.get(key)
                    for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
                },
                "inspection": list_inspection,
            }
            data["proxy_url_diagnosis"] = {
                "likely_hypothesis": likely_proxy_hypothesis(
                    data.get("instance_detail_safe_inspection", {}),
                    list_inspection,
                ),
                "checks": {
                    "detail_requests_returned_200": any(
                        item.get("http_status_code") == 200 for item in detail_attempts
                    ),
                    "proxy_url_found": bool(proxy_url),
                    "inspect_only": args.inspect_only,
                    "health_skipped": args.inspect_only or not proxy_url,
                },
            }

        if args.inspect_only:
            data["api_readiness"] = {"status": "inspect_only_skipped_health"}
        elif proxy_url:
            readiness, readiness_attempts, _ = wait_for_instance_api(proxy_url, args.ready_timeout_seconds)
            data["api_readiness"] = {"status": readiness, "attempts": readiness_attempts}
            endpoint_results = {}
            if readiness == "ready":
                for endpoint in SMOKE_ENDPOINTS:
                    result = simple_get(proxy_url + endpoint)
                    endpoint_results[endpoint] = {
                        "status": result.get("status"),
                        "http_status_code": result.get("http_status_code"),
                        "summary": summarize_api_response(result.get("json")),
                    }
            data["fastapi_smoke"] = endpoint_results
        else:
            data["api_readiness"] = {"status": "blocked_no_proxy_url_for_port_8000"}

        delete_path = DELETE_INSTANCE_PATH.format(id=instance_id)
        delete_result = http_request(base_url, delete_path, api_key, method="DELETE")
        data["delete_result"] = {
            key: delete_result.get(key)
            for key in ("attempted", "status", "method", "path", "http_status_code", "endpoint_host", "error_type", "error_truncated")
        }
        status = "succeeded" if delete_result.get("http_status_code") in {200, 202, 204} else "delete_failed"
        write_json(REPORT_PATH, build_report(args, status, data))
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        write_json(REPORT_PATH, build_report(args, "failed", data, message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    global TEMPLATE_ID
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod runtime smoke for AYL Wan2.2 S2V FastAPI.")
    parser.add_argument("--template-id", type=int, default=TEMPLATE_ID, help="SimplePod private template id for V2.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute to start the instance.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute to delete the instance at the end.")
    parser.add_argument("--inspect-only", action="store_true", help="Start, inspect instance/list port fields, then delete without calling FastAPI.")
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; otherwise select lowest-cost observed market.")
    parser.add_argument("--detail-attempts", type=int, default=24)
    parser.add_argument("--poll-interval-seconds", type=int, default=5)
    parser.add_argument("--ready-timeout-seconds", type=int, default=240)
    args = parser.parse_args()
    TEMPLATE_ID = args.template_id
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
