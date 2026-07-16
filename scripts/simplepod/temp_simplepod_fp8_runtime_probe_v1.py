import argparse
import base64
import binascii
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import temp_simplepod_runtime_smoke_v2 as smoke
from simplepod_phase_timing import PhaseTimer, now_iso


SCRIPT_ID = "TEMP_SIMPLEPOD_FP8_RUNTIME_PROBE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_fp8_runtime_probe_v1.json"
CONTAINER_LOG_PATH = REPO_ROOT / "logs" / "simplepod_fp8_runtime_probe_container.log"

IMAGE_TAG = "0.3.00-blackwell-fp8-runtime-probe-v1"
IMAGE_REF = f"ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:{IMAGE_TAG}"
DATACENTER = "EU-PL-01"
MIN_GPU_MEMORY_MB = 48_000
PROBE_REPORT_PATH = "/tmp/fp8_runtime_probe_v1.json"
CERTIFICATION_REPORT_PATH = "/tmp/fp8_runtime_certification_v1.json"

MARKET_LIST_PATH = "/instances/market/list"
INSTANCE_LIST_PATH = "/instances/list"

CONTAINER_LOG_ENDPOINT_TEMPLATES = (
    "/instances/{id}/logs",
    "/instances/{id}/log",
    "/instances/{id}/container-logs",
    "/instances/{id}/container/logs",
    "/instances/{id}/console",
    "/instances/{id}/events",
    "/instances/{id}/crashlog",
    "/instances/{id}/crashlogs",
    "/instances/{id}/stdout",
    "/instances/{id}/stderr",
)

LOG_FIELD_KEYWORDS = (
    "log",
    "logs",
    "stdout",
    "stderr",
    "console",
    "crash",
    "crashlog",
    "error",
    "errors",
    "warning",
    "warnings",
    "debug",
    "message",
    "messages",
    "output",
    "events",
)

STATUS_FIELD_KEYWORDS = (
    "status",
    "state",
    "phase",
    "exit",
    "exitcode",
    "container",
    "runtime",
    "reason",
    "message",
    "error",
    "warning",
)

TERMINAL_STATE_MARKERS = (
    "completed",
    "complete",
    "exited",
    "failed",
    "finished",
    "stopped",
    "terminated",
    "succeeded",
    "success",
    "error",
)

REPORT_MARKERS = (
    "runtime_certification",
    "Float8WeightOnlyConfig",
    "TEMP_FP8_RUNTIME_PROBE_V1",
    "[TEMP_FP8_RUNTIME_PROBE_V1]",
    "fp8_runtime_probe_v1",
    "fp8_runtime_certification_v1",
    "torchao",
    "quantize_",
)

ERROR_MARKERS = (
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "RuntimeError",
    "CUDA",
    "torchao",
    "Float8WeightOnlyConfig",
)

STARTUP_PULL_MARKERS = (
    "Download starting",
    "Preparing instance",
    "Pulling image",
)

STARTUP_READY_MARKERS = (
    "Image pulled",
    "Starting container",
    "Container started",
    "Running",
    "Ready",
    "[TEMP_FP8_RUNTIME_PROBE_V1]",
)

IMAGE_PULL_ERROR_MARKERS = (
    "Error pulling image",
    "manifest unknown",
    "unauthorized",
)

CREATED_OR_PULLING_STATUSES = (
    "created",
    "preparing",
    "pulling",
    "provisioning",
)

RUNNING_STATUSES = (
    "running",
    "active",
    "ready",
    "started",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def truncate(value, limit: int = 2000):
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\r\n\t")
    return printable / len(text)


def maybe_decode_base64_text(value: str) -> tuple[str, bool]:
    stripped = value.strip()
    if not stripped or len(stripped) < 16:
        return value, False
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r"
    if any(char not in allowed for char in stripped):
        return value, False
    compact = "".join(stripped.split())
    if len(compact) % 4:
        compact += "=" * (4 - (len(compact) % 4))
    try:
        decoded_bytes = base64.b64decode(compact, validate=False)
    except (binascii.Error, ValueError):
        return value, False
    if not decoded_bytes:
        return value, False
    decoded = decoded_bytes.decode("utf-8", errors="replace")
    if printable_ratio(decoded) < 0.85:
        return value, False
    decoded_lower = decoded.lower()
    if not any(token in decoded_lower for token in ("instance", "image", "container", "pull", "ready", "error", "torch", "fp8", "probe", "runtime", "cuda")):
        return value, False
    return decoded, True


def display_log_text(value: str) -> tuple[str, bool]:
    decoded, was_base64 = maybe_decode_base64_text(value)
    return decoded, was_base64


def endpoint_host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url.split("/")[0]


def http_text_request(base_url: str, path: str, api_key: str, timeout_seconds: int = 30) -> dict:
    url = urljoin(smoke.normalize_base_url(base_url), path.lstrip("/"))
    headers = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "ayl-front2-simplepod-fp8-runtime-probe-v1",
    }
    if api_key:
        headers[smoke.AUTH_HEADER] = api_key
    request = Request(url, method="GET", headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(524_288)
            text = body.decode("utf-8", errors="replace")
            return {
                "attempted": True,
                "status": "succeeded",
                "method": "GET",
                "path": path,
                "http_status_code": response.status,
                "endpoint_host": endpoint_host(url),
                "content_type": response.headers.get("Content-Type", ""),
                "body_bytes": len(body),
                "body_text": text,
                "body_truncated": truncate(text, 8000),
            }
    except HTTPError as exc:
        body = exc.read(131_072)
        text = body.decode("utf-8", errors="replace")
        return {
            "attempted": True,
            "status": "failed",
            "method": "GET",
            "path": path,
            "http_status_code": exc.code,
            "endpoint_host": endpoint_host(url),
            "content_type": exc.headers.get("Content-Type", "") if exc.headers else "",
            "body_bytes": len(body),
            "body_text": text,
            "body_truncated": truncate(text, 8000),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
        }
    except URLError as exc:
        return {
            "attempted": True,
            "status": "failed",
            "method": "GET",
            "path": path,
            "endpoint_host": endpoint_host(url),
            "error_type": "URLError",
            "error_truncated": str(exc)[:1000],
            "body_text": "",
            "body_truncated": "",
        }


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


def gpu_count(item: dict):
    value = item.get("gpuCount")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


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
    model = gpu_model(item)
    lower_model = model.lower()
    return {
        "market_id": market_id(item),
        "market_iri": market_iri(item),
        "gpuModel": model,
        "gpuMemorySize": item.get("gpuMemorySize"),
        "gpuMemorySize_mb_normalized": gpu_memory_mb(item),
        "gpuCount": item.get("gpuCount"),
        "pricePerGpu": item.get("pricePerGpu"),
        "datacenter": item.get("datacenter") or item.get("region") or item.get("dataCenter"),
        "rentalStatus": item.get("rentalStatus") or item.get("status"),
        "is_rtx_pro_6000": "rtx pro 6000" in lower_model,
        "is_mig_2g_48gb": "mig 2g.48gb" in lower_model,
        "is_mig": "mig" in lower_model,
        "reason": reason,
    }


def select_fp8_probe_market(items: list[dict]) -> dict:
    accepted = []
    rejected = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model = gpu_model(item)
        lower_model = model.lower()
        memory_mb = gpu_memory_mb(item)
        item_text = json.dumps(item, ensure_ascii=False).lower()
        rental_status = str(item.get("rentalStatus") or item.get("status") or "active").lower()
        iri = market_iri(item)
        count = gpu_count(item)

        if not iri:
            rejected.append(candidate_summary(item, "missing_market_id"))
        elif rental_status != "active":
            rejected.append(candidate_summary(item, "rentalStatus_not_active"))
        elif DATACENTER.lower() not in item_text:
            rejected.append(candidate_summary(item, "datacenter_not_EU_PL_01"))
        elif count != 1:
            rejected.append(candidate_summary(item, "gpuCount_not_1"))
        elif memory_mb is None or memory_mb < MIN_GPU_MEMORY_MB:
            rejected.append(candidate_summary(item, "gpuMemorySize_below_48000"))
        elif "rtx pro 6000" not in lower_model:
            rejected.append(candidate_summary(item, "gpuModel_not_RTX_PRO_6000"))
        else:
            accepted.append(
                {
                    "item": item,
                    "price": price_value(item),
                    "summary": candidate_summary(item),
                }
            )

    accepted.sort(
        key=lambda candidate: (
            not candidate["summary"]["is_mig_2g_48gb"],
            not candidate["summary"]["is_mig"],
            candidate["price"] is None,
            candidate["price"] if candidate["price"] is not None else 999999,
            candidate["summary"]["market_iri"],
        )
    )
    selected = accepted[0] if accepted else None
    selected_item = selected["item"] if selected else {}
    reason = "mig_2g_48gb_preferred_for_fp8_runtime_probe" if selected_item and selected["summary"]["is_mig_2g_48gb"] else "lowest_price_RTX_PRO_6000_48gb_or_higher"
    return {
        "policy": "fp8_runtime_probe_mig_48gb_policy",
        "primary_datacenter": DATACENTER,
        "minimum_gpuMemorySize_mb": MIN_GPU_MEMORY_MB,
        "selection_rule": "active EU-PL-01 gpuCount=1 RTX PRO 6000 >=48GB; prefer MIG 2g.48gb over full 96GB",
        "selected_market": market_iri(selected_item),
        "selected_market_id": market_id(selected_item),
        "selected_summary": candidate_summary(selected_item, reason) if selected_item else {},
        "accepted_candidates_observed": len(accepted),
        "accepted_candidates_summary": [candidate["summary"] for candidate in accepted[:20]],
        "rejected_candidates_observed": len(rejected),
        "rejected_candidates_summary": rejected[:40],
    }


def runtime_payload(instance_market: str, template_id: int) -> dict:
    return {
        "gpuCount": 1,
        "instanceMarket": instance_market or "<selected_fp8_probe_market>",
        "instanceTemplate": f"/instances/templates/{template_id}",
        "envVariables": [
            {"name": "AYL_IMAGE_TAG", "value": IMAGE_TAG},
            {"name": "AYL_RUNTIME_VERSION", "value": IMAGE_TAG},
            {"name": "AYL_FP8_PROBE_REPORT_PATH", "value": PROBE_REPORT_PATH},
            {"name": "AYL_FP8_CERTIFICATION_REPORT_PATH", "value": CERTIFICATION_REPORT_PATH},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "TORCH_CUDA_ARCH_LIST", "value": "12.0"},
        ],
    }


def safe_result(result: dict) -> dict:
    keys = (
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
    return {key: result.get(key) for key in keys if key in result}


def safe_instance_observation(detail_json) -> dict:
    if not isinstance(detail_json, dict):
        return {"json_type": type(detail_json).__name__}
    interesting_parts = (
        "id",
        "status",
        "state",
        "phase",
        "containerStatus",
        "containerState",
        "runtimeStatus",
        "exitCode",
        "startedAt",
        "finishedAt",
        "updatedAt",
        "createdAt",
        "name",
        "image",
        "imageName",
        "imageTag",
        "template",
        "instanceTemplate",
    )
    summary = {key: detail_json.get(key) for key in interesting_parts if key in detail_json}
    for nested_key in ("container", "docker", "pod", "task", "job"):
        value = detail_json.get(nested_key)
        if isinstance(value, dict):
            summary[nested_key] = {
                key: value.get(key)
                for key in ("status", "state", "phase", "exitCode", "startedAt", "finishedAt", "image", "tag")
                if key in value
            }
    return summary


def path_join(parent: str, child) -> str:
    if parent:
        return f"{parent}.{child}"
    return str(child)


def collect_instance_text_fields(value, *, path: str = "", max_depth: int = 10, max_items: int = 80) -> list[dict]:
    if max_depth < 0 or max_items <= 0:
        return []
    found: list[dict] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = path_join(path, key)
            key_lower = str(key).lower()
            if isinstance(item, str):
                decoded_text, decoded_from_base64 = display_log_text(item)
                item_has_marker = any(marker in decoded_text for marker in REPORT_MARKERS + ERROR_MARKERS)
                key_is_loglike = any(token in key_lower for token in LOG_FIELD_KEYWORDS)
                if key_is_loglike or item_has_marker:
                    found.append(
                        {
                            "path": item_path,
                            "key": str(key),
                            "length": len(item),
                            "decoded_from_base64": decoded_from_base64,
                            "decoded_length": len(decoded_text),
                            "contains_fp8_marker": any(marker in decoded_text for marker in REPORT_MARKERS),
                            "contains_error_marker": any(marker in decoded_text for marker in ERROR_MARKERS),
                            "raw_truncated": truncate(item, 1000),
                            "text_truncated": truncate(decoded_text, 8000),
                        }
                    )
                    if len(found) >= max_items:
                        return found
            elif isinstance(item, (dict, list)):
                found.extend(collect_instance_text_fields(item, path=item_path, max_depth=max_depth - 1, max_items=max_items - len(found)))
                if len(found) >= max_items:
                    return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            item_path = path_join(path, index)
            found.extend(collect_instance_text_fields(item, path=item_path, max_depth=max_depth - 1, max_items=max_items - len(found)))
            if len(found) >= max_items:
                return found
    return found


def collect_instance_status_fields(value, *, path: str = "", max_depth: int = 10, max_items: int = 100) -> list[dict]:
    if max_depth < 0 or max_items <= 0:
        return []
    found: list[dict] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = path_join(path, key)
            key_lower = str(key).lower().replace("_", "")
            if any(token in key_lower for token in STATUS_FIELD_KEYWORDS) and not isinstance(item, (dict, list)):
                found.append({"path": item_path, "key": str(key), "value": truncate(item, 1000)})
                if len(found) >= max_items:
                    return found
            if isinstance(item, (dict, list)):
                found.extend(collect_instance_status_fields(item, path=item_path, max_depth=max_depth - 1, max_items=max_items - len(found)))
                if len(found) >= max_items:
                    return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(collect_instance_status_fields(item, path=path_join(path, index), max_depth=max_depth - 1, max_items=max_items - len(found)))
            if len(found) >= max_items:
                return found
    return found


def extract_exit_code(status_fields: list[dict]):
    for field in status_fields:
        key = str(field.get("key") or "").lower()
        path = str(field.get("path") or "").lower()
        if "exit" not in key and "exit" not in path:
            continue
        value = field.get("value")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                pass
    return None


def extract_status_value(status_fields: list[dict]) -> str:
    preferred = ("containerstatus", "containerstate", "runtimestatus", "status", "state", "phase")
    for token in preferred:
        for field in status_fields:
            path_key = (str(field.get("path") or "") + "." + str(field.get("key") or "")).lower().replace("_", "")
            if token in path_key and field.get("value") is not None:
                return str(field.get("value"))
    return ""


def terminal_state_seen(detail_json) -> bool:
    text = json.dumps(safe_instance_observation(detail_json), ensure_ascii=False).lower()
    return any(marker in text for marker in TERMINAL_STATE_MARKERS)


def combined_instance_text(detail_json) -> str:
    fields = collect_instance_text_fields(detail_json)
    return "\n".join(str(field.get("text_truncated") or "") for field in fields)


def marker_seen(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def classify_startup(detail_json) -> dict:
    status_fields = collect_instance_status_fields(detail_json)
    text = combined_instance_text(detail_json)
    status_value = extract_status_value(status_fields)
    status_lower = status_value.lower()
    text_lower = text.lower()
    image_pull_detected = marker_seen(text, STARTUP_PULL_MARKERS) or any(token in status_lower for token in CREATED_OR_PULLING_STATUSES)
    image_pull_completed = marker_seen(text, ("Image pulled",))
    container_start_detected = (
        marker_seen(text, STARTUP_READY_MARKERS)
        or any(token in status_lower for token in RUNNING_STATUSES)
        or any(marker in text for marker in REPORT_MARKERS)
    )
    image_pull_failed = marker_seen(text, IMAGE_PULL_ERROR_MARKERS) or any(marker.lower() in text_lower for marker in IMAGE_PULL_ERROR_MARKERS)
    terminal_seen = terminal_state_seen(detail_json)
    return {
        "status_fields": status_fields,
        "text_fields": collect_instance_text_fields(detail_json),
        "console_text": text,
        "status_value": status_value,
        "container_exit_code": extract_exit_code(status_fields),
        "image_pull_detected": image_pull_detected,
        "image_pull_completed": image_pull_completed,
        "container_start_detected": container_start_detected,
        "image_pull_failed": image_pull_failed,
        "terminal_state_seen": terminal_seen,
    }


def startup_progress_line(classification: dict) -> str:
    text = classification.get("console_text") or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    interesting = []
    for line in lines:
        if marker_seen(line, STARTUP_PULL_MARKERS + STARTUP_READY_MARKERS + IMAGE_PULL_ERROR_MARKERS):
            interesting.append(line)
    if interesting:
        return interesting[-1]
    status_value = classification.get("status_value") or ""
    return f"container_status={status_value}" if status_value else "container_status=unknown"


def maybe_json_from_string(value: str):
    text = value.strip()
    if not text:
        return None
    candidates = []
    if text.startswith("{") and text.endswith("}"):
        candidates.append(text)
    for line in text.splitlines():
        line_text = line.strip()
        if line_text.startswith("{") and line_text.endswith("}"):
            candidates.append(line_text)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def looks_like_fp8_report(value) -> bool:
    if not isinstance(value, dict):
        return False
    text = json.dumps(value, ensure_ascii=False)
    return any(marker in text for marker in REPORT_MARKERS)


def find_fp8_report(value, *, max_depth: int = 10):
    if max_depth < 0:
        return None
    if isinstance(value, dict):
        if looks_like_fp8_report(value):
            return value
        for item in value.values():
            found = find_fp8_report(item, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_fp8_report(item, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(value, str) and any(marker in value for marker in REPORT_MARKERS):
        parsed = maybe_json_from_string(value)
        if parsed is not None and looks_like_fp8_report(parsed):
            return parsed
    return None


def summarize_fp8_report(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {}

    text_report = json.dumps(report, ensure_ascii=False)

    def first_value(keys: tuple[str, ...]):
        stack = [report]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        return value[key]
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        return None

    return {
        "torch_version": first_value(("torch_version", "torch.__version__")),
        "cuda_version": first_value(("cuda_version", "torch_cuda_version", "torch.version.cuda")),
        "torchao_version": first_value(("torchao_version", "torchao.__version__")),
        "device_name": first_value(("device_name", "gpu_name")),
        "device_capability": first_value(("device_capability", "compute_capability")),
        "float8_weight_only_config_available": first_value(("Float8WeightOnlyConfig_available", "float8_weight_only_config_available")),
        "quantize_available": first_value(("quantize_available", "quantize__available", "quantize_ available")),
        "nn_linear_preserved": first_value(("nn_linear_preserved", "linear_preserved")),
        "weight_type": first_value(("weight_type", "weight_class")),
        "weight_dtype": first_value(("weight_dtype", "weight_dtype_after")),
        "allocated_vram_gb": first_value(("allocated_vram_gb", "allocated_gb", "memory_allocated_gb")),
        "reserved_vram_gb": first_value(("reserved_vram_gb", "reserved_gb", "memory_reserved_gb")),
        "peak_vram_gb": first_value(("peak_vram_gb", "peak_allocated_gb", "max_memory_allocated_gb")),
        "runtime_certification": first_value(("runtime_certification", "certification")),
        "error": first_value(("error", "error_truncated")),
        "stacktrace": truncate(first_value(("stacktrace", "traceback", "stacktrace_truncated")), 4000),
        "contains_runtime_certification_marker": "runtime_certification" in text_report,
    }


def parse_runtime_certification_from_text(text: str) -> str:
    lowered = text.lower()
    if "runtime_certification" in lowered:
        lines = [line.strip() for line in text.splitlines() if "runtime_certification" in line.lower()]
        for line in lines:
            line_upper = line.upper()
            if "PASS" in line_upper:
                return "PASS"
            if "FAIL" in line_upper:
                return "FAIL"
    if "runtime_certification = pass" in lowered or "runtime_certification=pass" in lowered:
        return "PASS"
    if "runtime_certification = fail" in lowered or "runtime_certification=fail" in lowered:
        return "FAIL"
    return ""


def analyze_container_log_text(text: str) -> dict:
    markers_found = [marker for marker in REPORT_MARKERS + ERROR_MARKERS if marker in text]
    runtime_certification = parse_runtime_certification_from_text(text)
    status_value = ""
    for line in text.splitlines():
        stripped = line.strip()
        if "status=" in stripped.lower() or '"status"' in stripped.lower():
            status_value = truncate(stripped, 1000)
        if "runtime_certification" in stripped.lower():
            break
    traceback_tail = ""
    if "Traceback" in text:
        traceback_tail = "Traceback" + text.rsplit("Traceback", 1)[-1]
        traceback_tail = truncate(traceback_tail, 6000)
    return {
        "probe_output_markers_found": markers_found,
        "runtime_certification_detected_from_logs": bool(runtime_certification),
        "runtime_certification_value": runtime_certification,
        "status_line_truncated": status_value,
        "traceback_tail": traceback_tail,
        "contains_pass": "PASS" in text,
        "contains_fail": "FAIL" in text,
        "contains_traceback": "Traceback" in text,
        "contains_module_not_found": "ModuleNotFoundError" in text,
        "contains_import_error": "ImportError" in text,
        "contains_runtime_error": "RuntimeError" in text,
    }


def collect_container_logs(base_url: str, api_key: str, instance_id: int, latest_detail_json) -> dict:
    endpoint_attempts = []
    best_text_parts = []
    best_source = ""
    best_http_status = None
    best_path = ""

    for template in CONTAINER_LOG_ENDPOINT_TEMPLATES:
        path = template.format(id=instance_id)
        result = http_text_request(base_url, path, api_key, timeout_seconds=30)
        attempt = {
            "path": path,
            "status": result.get("status"),
            "http_status_code": result.get("http_status_code"),
            "content_type": result.get("content_type"),
            "body_bytes": result.get("body_bytes"),
            "error_type": result.get("error_type"),
            "error_truncated": result.get("error_truncated"),
            "body_truncated": result.get("body_truncated"),
        }
        endpoint_attempts.append(attempt)
        body_text = result.get("body_text") or ""
        if result.get("http_status_code") == 200 and body_text.strip():
            best_text_parts.append(f"===== {path} =====\n{body_text}")
            best_source = "simplepod_log_endpoint"
            best_http_status = result.get("http_status_code")
            best_path = path
            if any(marker in body_text for marker in REPORT_MARKERS + ERROR_MARKERS):
                break

    detail_text_fields = collect_instance_text_fields(latest_detail_json)
    if detail_text_fields:
        detail_log_text = "\n\n".join(
            f"===== instance_detail:{field['path']} =====\n{field.get('text_truncated') or ''}"
            for field in detail_text_fields
            if field.get("text_truncated")
        )
        if detail_log_text.strip():
            best_text_parts.append(detail_log_text)
            if not best_source:
                best_source = "instance_detail_fields"
                best_path = "GET /instances/{id}"

    combined_text = "\n\n".join(part for part in best_text_parts if part.strip())
    if combined_text:
        write_text(CONTAINER_LOG_PATH, combined_text)

    analysis = analyze_container_log_text(combined_text)
    return {
        "container_logs_attempted": True,
        "container_logs_retrieved": bool(combined_text.strip()),
        "container_logs_source": best_source,
        "container_logs_http_status": best_http_status,
        "container_logs_path": best_path,
        "container_logs_local_path": str(CONTAINER_LOG_PATH) if combined_text else "",
        "container_logs_truncated": truncate(combined_text, 12000),
        "container_log_endpoint_attempts": endpoint_attempts,
        "instance_text_fields": detail_text_fields[:40],
        **analysis,
    }


def decoded_field_by_name(text_fields: list[dict], name: str) -> str:
    target = name.lower()
    for field in text_fields:
        if str(field.get("key") or "").lower() == target:
            return str(field.get("text_truncated") or "")
    return ""


def build_report(args: argparse.Namespace, status: str, data: dict) -> dict:
    fp8_report = data.get("fp8_runtime_report")
    fp8_summary = summarize_fp8_report(fp8_report)
    selected = data.get("market_selection", {}).get("selected", {})
    selected_summary = selected.get("selected_summary", {})
    return {
        "script_id": SCRIPT_ID,
        "created_at": now_iso(),
        "status": status,
        "dry_run": not args.execute,
        "image_ref": IMAGE_REF,
        "template_id": args.template_id,
        "datacenter": DATACENTER,
        "report_path": str(REPORT_PATH),
        "container_report_paths": {
            "probe": PROBE_REPORT_PATH,
            "certification": CERTIFICATION_REPORT_PATH,
        },
        "selected_gpu": {
            "selected_market_id": selected.get("selected_market_id") or data.get("selected_market_id"),
            "selected_market": selected.get("selected_market") or args.instance_market,
            "gpuModel": selected_summary.get("gpuModel"),
            "gpuMemorySize": selected_summary.get("gpuMemorySize"),
            "gpuMemorySize_mb_normalized": selected_summary.get("gpuMemorySize_mb_normalized"),
            "pricePerGpu": selected_summary.get("pricePerGpu"),
            "datacenter": selected_summary.get("datacenter"),
            "reason": selected_summary.get("reason"),
        },
        "market_selection": data.get("market_selection"),
        "instance_id": data.get("instance_id"),
        "request_payload_redacted": data.get("request_payload_redacted"),
        "start_result": data.get("start_result"),
        "monitoring": data.get("monitoring"),
        "fp8_report_retrieval": data.get("fp8_report_retrieval"),
        "startup_seconds": data.get("startup_seconds"),
        "probe_seconds": data.get("probe_seconds"),
        "image_pull_detected": data.get("image_pull_detected", False),
        "image_pull_completed": data.get("image_pull_completed", False),
        "container_start_detected": data.get("container_start_detected", False),
        "decoded_console": data.get("decoded_console", ""),
        "decoded_console_system": data.get("decoded_console_system", ""),
        "startup_timeout_seconds": args.startup_timeout_seconds,
        "probe_timeout_seconds": args.probe_timeout_seconds,
        "status_history": data.get("status_history", []),
        "container_logs_attempted": data.get("container_logs_attempted", False),
        "container_logs_retrieved": data.get("container_logs_retrieved", False),
        "container_logs_source": data.get("container_logs_source", ""),
        "container_logs_http_status": data.get("container_logs_http_status"),
        "container_logs_path": data.get("container_logs_path", ""),
        "container_logs_local_path": data.get("container_logs_local_path", ""),
        "container_logs_truncated": data.get("container_logs_truncated", ""),
        "container_exit_detected": data.get("container_exit_detected", False),
        "container_exit_code": data.get("container_exit_code"),
        "container_status": data.get("container_status", ""),
        "instance_errors": data.get("instance_errors", []),
        "instance_warnings": data.get("instance_warnings", []),
        "probe_output_markers_found": data.get("probe_output_markers_found", []),
        "runtime_certification_detected_from_logs": data.get("runtime_certification_detected_from_logs", False),
        "runtime_certification_value": data.get("runtime_certification_value", ""),
        "original_status": data.get("original_status", ""),
        "fp8_runtime_summary": fp8_summary,
        "fp8_runtime_report": fp8_report,
        "delete_result": data.get("delete_result"),
        "runtime_seconds": data.get("runtime_seconds"),
        "phase_timings": data.get("phase_timings", []),
        "safety_guards": {
            "loads_wan": False,
            "loads_wan_models": False,
            "uses_r2": False,
            "uploads_files": False,
            "runs_inference": False,
            "generates_audio": False,
            "generates_video": False,
            "uses_network_drive": False,
            "opens_fastapi": False,
            "exposes_port": False,
            "sends_startup_script": False,
            "overrides_cmd_or_entrypoint": False,
            "calls_simplepod": bool(args.execute),
            "deletes_instance": bool(args.execute and args.confirm_delete),
        },
    }


def query_market(base_url: str, api_key: str) -> dict:
    query = {
        "mode": "docker",
        "rentalStatus": "active",
        "region": DATACENTER,
        "gpuCount[gte]": "1",
        "gpuCount[lte]": "1",
        "gpuMemorySize[gte]": str(MIN_GPU_MEMORY_MB),
        "itemsPerPage": "100",
        "order[pricePerGpu]": "asc",
    }
    return smoke.http_request(base_url, f"{MARKET_LIST_PATH}?{urlencode(query)}", api_key)


def wait_for_instance_observable(base_url: str, api_key: str, instance_id: int, timeout_seconds: int, poll_interval_seconds: int) -> tuple[str, list[dict]]:
    observations = []
    deadline = time.monotonic() + max(1, timeout_seconds)
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    while time.monotonic() < deadline:
        detail_result = smoke.http_request(base_url, detail_path, api_key, timeout_seconds=30)
        observation = {
            "observed_at": now_iso(),
            "request": safe_result(detail_result),
            "instance": safe_instance_observation(detail_result.get("json")),
        }
        observations.append(observation)
        if detail_result.get("status") == "succeeded" and isinstance(detail_result.get("json"), dict):
            return "observable", observations
        time.sleep(max(1, poll_interval_seconds))
    return "timeout", observations


def wait_for_container_start(base_url: str, api_key: str, instance_id: int, timeout_seconds: int, poll_interval_seconds: int) -> dict:
    started_monotonic = time.monotonic()
    deadline = started_monotonic + max(1, timeout_seconds)
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    status_history = []
    latest_detail_json = None
    latest_classification = {}
    last_progress_line = ""
    polls = 0

    while time.monotonic() < deadline:
        polls += 1
        detail_result = smoke.http_request(base_url, detail_path, api_key, timeout_seconds=30)
        latest_detail_json = detail_result.get("json")
        classification = classify_startup(latest_detail_json)
        latest_classification = classification
        progress_line = startup_progress_line(classification)
        if progress_line != last_progress_line:
            print(f"[{SCRIPT_ID}] startup {progress_line}", flush=True)
            last_progress_line = progress_line
        status_history.append(
            {
                "observed_at": now_iso(),
                "poll": polls,
                "request": safe_result(detail_result),
                "container_status": classification.get("status_value"),
                "container_exit_code": classification.get("container_exit_code"),
                "image_pull_detected": classification.get("image_pull_detected"),
                "image_pull_completed": classification.get("image_pull_completed"),
                "container_start_detected": classification.get("container_start_detected"),
                "image_pull_failed": classification.get("image_pull_failed"),
                "terminal_state_seen": classification.get("terminal_state_seen"),
                "progress_line": truncate(progress_line, 1000),
            }
        )
        if classification.get("image_pull_failed"):
            return {
                "status": "failed_image_pull",
                "startup_seconds": round(time.monotonic() - started_monotonic, 3),
                "latest_detail_json": latest_detail_json,
                "latest_classification": latest_classification,
                "status_history": status_history,
            }
        if classification.get("container_start_detected"):
            return {
                "status": "container_started",
                "startup_seconds": round(time.monotonic() - started_monotonic, 3),
                "latest_detail_json": latest_detail_json,
                "latest_classification": latest_classification,
                "status_history": status_history,
            }
        if classification.get("terminal_state_seen"):
            return {
                "status": "container_terminal_before_probe_monitor",
                "startup_seconds": round(time.monotonic() - started_monotonic, 3),
                "latest_detail_json": latest_detail_json,
                "latest_classification": latest_classification,
                "status_history": status_history,
            }
        time.sleep(max(1, poll_interval_seconds))

    timeout_status = "blocked_image_pull_timeout" if latest_classification.get("image_pull_detected") else "blocked_startup_timeout"
    return {
        "status": timeout_status,
        "startup_seconds": round(time.monotonic() - started_monotonic, 3),
        "latest_detail_json": latest_detail_json,
        "latest_classification": latest_classification,
        "status_history": status_history,
    }


def monitor_probe(base_url: str, api_key: str, instance_id: int, timeout_seconds: int, poll_interval_seconds: int) -> dict:
    started_monotonic = time.monotonic()
    deadline = time.monotonic() + max(1, timeout_seconds)
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    observations = []
    found_report = None
    terminal_seen = False
    latest_detail_json = None
    latest_status_fields = []
    polls = 0
    while time.monotonic() < deadline:
        polls += 1
        detail_result = smoke.http_request(base_url, detail_path, api_key, timeout_seconds=30)
        detail_json = detail_result.get("json")
        latest_detail_json = detail_json
        latest_status_fields = collect_instance_status_fields(detail_json)
        report = find_fp8_report(detail_json)
        observation = {
            "observed_at": now_iso(),
            "poll": polls,
            "request": safe_result(detail_result),
            "instance": safe_instance_observation(detail_json),
            "status_fields": latest_status_fields[:30],
            "fp8_report_marker_seen": report is not None,
            "terminal_state_seen": terminal_state_seen(detail_json),
        }
        observations.append(observation)
        if report is not None:
            found_report = report
            break
        if observation["terminal_state_seen"]:
            terminal_seen = True
            break
        time.sleep(max(1, poll_interval_seconds))

    if found_report is not None:
        status = "report_found"
    elif terminal_seen:
        status = "terminal_state_seen_without_report"
    else:
        status = "timeout_without_report"
    return {
        "status": status,
        "polls": polls,
        "observations": observations[-20:],
        "report_found": found_report is not None,
        "terminal_state_seen": terminal_seen,
        "report_retrieval_note": "The FP8 image does not expose FastAPI/R2. This script can only capture the report if SimplePod exposes it in instance details/log fields.",
        "fp8_runtime_report": found_report,
        "latest_detail_json": latest_detail_json,
        "latest_status_fields": latest_status_fields,
        "container_exit_code": extract_exit_code(latest_status_fields),
        "container_status": extract_status_value(latest_status_fields),
        "probe_seconds": round(time.monotonic() - started_monotonic, 3),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute SimplePod FP8 runtime probe on RTX PRO 6000 MIG 2g.48gb.")
    parser.add_argument("--execute", action="store_true", help="Create a real SimplePod instance.")
    parser.add_argument("--confirm-start", action="store_true", help="Required with --execute.")
    parser.add_argument("--confirm-delete", action="store_true", help="Required with --execute; deletes instance in finally.")
    parser.add_argument("--template-id", type=int, required=True, help="Experimental SimplePod template id. Do not use BF16 template 25138.")
    parser.add_argument("--startup-timeout-seconds", type=int, default=1200)
    parser.add_argument("--ready-timeout-seconds", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--probe-timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    parser.add_argument("--instance-market", default="", help="Optional explicit /instances/market/{id}; normally auto-selected.")
    args = parser.parse_args()
    if args.ready_timeout_seconds is not None:
        args.startup_timeout_seconds = args.ready_timeout_seconds
    return args


def validate_args(args: argparse.Namespace) -> tuple[bool, str]:
    if not args.execute:
        return True, "dry_run"
    if not args.confirm_start:
        return False, "blocked_missing_confirm_start"
    if not args.confirm_delete:
        return False, "blocked_missing_confirm_delete"
    if args.template_id <= 0:
        return False, "blocked_invalid_template_id"
    if args.template_id == 25138:
        return False, "blocked_refuses_bf16_template_25138"
    return True, "execute_confirmed"


def main() -> int:
    args = parse_args()
    started_monotonic = time.monotonic()
    timer = PhaseTimer(emit=True)
    data: dict = {"phase_timings": timer.phases}
    instance_id = None
    status = "unknown"
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL

    print(f"[{SCRIPT_ID}] START dry_run={str(not args.execute).lower()} template_id={args.template_id}", flush=True)
    print(f"[{SCRIPT_ID}] image_ref={IMAGE_REF}", flush=True)
    print(f"[{SCRIPT_ID}] no_wan=true no_models=true no_r2=true no_inference=true no_video=true", flush=True)

    valid, validation_status = validate_args(args)
    payload = runtime_payload(args.instance_market, args.template_id)
    data["request_payload_redacted"] = smoke.redact_value("request_payload", payload)
    data["fp8_report_retrieval"] = {
        "method": "instance_detail_or_log_field_scan",
        "container_report_paths": [PROBE_REPORT_PATH, CERTIFICATION_REPORT_PATH],
        "requires_simplepod_report_visibility": True,
    }

    try:
        if not valid:
            status = validation_status
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1

        if not args.execute:
            data["market_selection"] = {
                "status": "dry_run_not_queried",
                "policy": "fp8_runtime_probe_mig_48gb_policy",
                "planned_query": {
                    "endpoint": MARKET_LIST_PATH,
                    "datacenter": DATACENTER,
                    "gpuModel_contains": "RTX PRO 6000",
                    "gpuMemorySize_min_mb": MIN_GPU_MEMORY_MB,
                    "prefer": "RTX PRO 6000 MIG 2g.48gb",
                    "reject_below_48gb": True,
                },
                "selected": select_fp8_probe_market([]),
            }
            status = "dry_run_ready"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 0

        with timer.phase("load_auth_env"):
            smoke.load_repo_dotenv()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)

        if not api_key:
            status = "missing_simplepod_api_key"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1

        if args.instance_market:
            selected = {
                "policy": "fp8_runtime_probe_mig_48gb_policy",
                "selected_market": args.instance_market,
                "selected_market_id": args.instance_market.rstrip("/").rsplit("/", 1)[-1],
                "selected_summary": {"reason": "explicit_instance_market_argument"},
            }
            market_result = {"status": "skipped_explicit_instance_market"}
        else:
            with timer.phase("market_selection"):
                market_result = query_market(base_url, api_key)
            items = smoke.extract_items(market_result.get("json"))
            selected = select_fp8_probe_market(items)
        data["market_selection"] = {
            "result": safe_result(market_result),
            "items_observed": len(smoke.extract_items(market_result.get("json"))) if isinstance(market_result, dict) else 0,
            "selected": selected,
        }
        market = selected.get("selected_market", "")
        if not market:
            status = "blocked_no_fp8_probe_market_selected"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1

        payload = runtime_payload(market, args.template_id)
        data["request_payload_redacted"] = smoke.redact_value("request_payload", payload)
        with timer.phase("start_instance"):
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=payload, timeout_seconds=60)
        data["start_result"] = safe_result(start_result)
        instance_id = smoke.extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 1

        with timer.phase("wait_container_startup"):
            startup_result = wait_for_container_start(
                base_url,
                api_key,
                instance_id,
                args.startup_timeout_seconds,
                args.poll_interval_seconds,
            )
        data["monitoring"] = {
            "startup": {
                key: value
                for key, value in startup_result.items()
                if key not in {"latest_detail_json", "latest_classification"}
            },
        }
        latest_detail_json = startup_result.get("latest_detail_json")
        latest_classification = startup_result.get("latest_classification") or {}
        startup_text_fields = latest_classification.get("text_fields") or []
        data["startup_seconds"] = startup_result.get("startup_seconds")
        data["image_pull_detected"] = bool(latest_classification.get("image_pull_detected"))
        data["image_pull_completed"] = bool(latest_classification.get("image_pull_completed"))
        data["container_start_detected"] = bool(latest_classification.get("container_start_detected"))
        data["decoded_console"] = decoded_field_by_name(startup_text_fields, "console")
        data["decoded_console_system"] = decoded_field_by_name(startup_text_fields, "consoleSystem")
        data["status_history"] = startup_result.get("status_history", [])[-80:]

        startup_status = startup_result.get("status")
        if startup_status in {"failed_image_pull", "blocked_image_pull_timeout", "blocked_startup_timeout"}:
            status = startup_status
            data["original_status"] = startup_status
            latest_status_fields = latest_classification.get("status_fields") or []
            data["container_exit_code"] = latest_classification.get("container_exit_code")
            data["container_status"] = latest_classification.get("status_value") or ""
            data["container_exit_detected"] = data["container_exit_code"] is not None or latest_classification.get("terminal_state_seen") is True
            data["instance_errors"] = [
                field for field in latest_status_fields if "error" in str(field.get("path") or field.get("key") or "").lower()
            ][:20]
            data["instance_warnings"] = [
                field for field in latest_status_fields if "warning" in str(field.get("path") or field.get("key") or "").lower()
            ][:20]
            with timer.phase("collect_container_logs"):
                log_collection = collect_container_logs(base_url, api_key, instance_id, latest_detail_json)
            for key, value in log_collection.items():
                if key not in {"container_log_endpoint_attempts", "instance_text_fields"}:
                    data[key] = value
            data["fp8_report_retrieval"].update(
                {
                    "container_log_endpoint_attempts": log_collection.get("container_log_endpoint_attempts", []),
                    "instance_text_fields": log_collection.get("instance_text_fields", []),
                    "container_logs_attempted": log_collection.get("container_logs_attempted", False),
                    "container_logs_retrieved": log_collection.get("container_logs_retrieved", False),
                }
            )
            log_certification = str(log_collection.get("runtime_certification_value") or "").upper()
            if log_certification == "PASS":
                status = "succeeded_recovered_from_container_logs"
            elif log_certification == "FAIL":
                status = "failed_recovered_from_container_logs"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
            return 0 if status == "succeeded_recovered_from_container_logs" else 1

        if startup_status == "container_terminal_before_probe_monitor":
            monitor_result = {
                "status": "terminal_state_seen_without_report",
                "polls": 0,
                "observations": [],
                "report_found": find_fp8_report(latest_detail_json) is not None,
                "terminal_state_seen": True,
                "report_retrieval_note": "Container reached a terminal state during startup monitoring; collecting logs immediately.",
                "fp8_runtime_report": find_fp8_report(latest_detail_json),
                "latest_detail_json": latest_detail_json,
                "latest_status_fields": latest_classification.get("status_fields") or [],
                "container_exit_code": latest_classification.get("container_exit_code"),
                "container_status": latest_classification.get("status_value") or "",
                "probe_seconds": 0.0,
            }
        else:
            with timer.phase("monitor_probe_report"):
                monitor_result = monitor_probe(
                    base_url,
                    api_key,
                    instance_id,
                    args.probe_timeout_seconds,
                    args.poll_interval_seconds,
                )
        data["monitoring"]["probe"] = {
            key: value
            for key, value in monitor_result.items()
            if key not in {"fp8_runtime_report", "latest_detail_json"}
        }
        data["fp8_runtime_report"] = monitor_result.get("fp8_runtime_report")
        data["fp8_report_retrieval"].update(
            {
                "status": monitor_result.get("status"),
                "report_found": monitor_result.get("report_found"),
                "terminal_state_seen": monitor_result.get("terminal_state_seen"),
            }
        )

        fp8_summary = summarize_fp8_report(data.get("fp8_runtime_report"))
        certification = str(fp8_summary.get("runtime_certification") or "").upper()
        if certification == "PASS":
            status = "succeeded"
        elif monitor_result.get("report_found"):
            status = "probe_completed_certification_failed"
        elif monitor_result.get("terminal_state_seen"):
            status = "blocked_report_retrieval_not_exposed_by_simplepod_api"
        else:
            status = "probe_timeout_no_report"
        data["original_status"] = status

        latest_detail_json = monitor_result.get("latest_detail_json")
        latest_status_fields = monitor_result.get("latest_status_fields") or []
        data["probe_seconds"] = monitor_result.get("probe_seconds")
        data["container_exit_code"] = monitor_result.get("container_exit_code")
        data["container_status"] = monitor_result.get("container_status") or ""
        data["container_exit_detected"] = data["container_exit_code"] is not None or monitor_result.get("terminal_state_seen") is True
        data["instance_errors"] = [
            field for field in latest_status_fields if "error" in str(field.get("path") or field.get("key") or "").lower()
        ][:20]
        data["instance_warnings"] = [
            field for field in latest_status_fields if "warning" in str(field.get("path") or field.get("key") or "").lower()
        ][:20]

        with timer.phase("collect_container_logs"):
            log_collection = collect_container_logs(base_url, api_key, instance_id, latest_detail_json)
        for key, value in log_collection.items():
            if key != "container_log_endpoint_attempts" and key != "instance_text_fields":
                data[key] = value
        data["fp8_report_retrieval"].update(
            {
                "container_log_endpoint_attempts": log_collection.get("container_log_endpoint_attempts", []),
                "instance_text_fields": log_collection.get("instance_text_fields", []),
                "container_logs_attempted": log_collection.get("container_logs_attempted", False),
                "container_logs_retrieved": log_collection.get("container_logs_retrieved", False),
            }
        )

        log_certification = str(log_collection.get("runtime_certification_value") or "").upper()
        if certification != "PASS" and log_certification == "PASS":
            status = "succeeded_recovered_from_container_logs"
        elif certification != "PASS" and log_certification == "FAIL":
            status = "failed_recovered_from_container_logs"
        elif (
            status != "probe_timeout_no_report"
            and not monitor_result.get("report_found")
            and not log_collection.get("container_logs_retrieved")
        ):
            status = "blocked_container_logs_not_exposed_by_simplepod_api"
        data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
        write_json(REPORT_PATH, build_report(args, status, data))
        print(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}", flush=True)
        return 0 if status in {"succeeded", "succeeded_recovered_from_container_logs"} else 1
    finally:
        if instance_id is not None and args.confirm_delete:
            with timer.phase("delete_instance"):
                delete_path = smoke.DELETE_INSTANCE_PATH.format(id=instance_id)
                delete_result = smoke.http_request(base_url, delete_path, api_key, method="DELETE", timeout_seconds=60)
            data["delete_result"] = safe_result(delete_result)
            if delete_result.get("http_status_code") not in {200, 202, 204}:
                status = "delete_failed_manual_required"
                print(f"[{SCRIPT_ID}] DELETE FAILED - manual cleanup required instance_id={instance_id}", flush=True)
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            data["phase_timings"] = timer.phases
            write_json(REPORT_PATH, build_report(args, status, data))


if __name__ == "__main__":
    sys.exit(main())
