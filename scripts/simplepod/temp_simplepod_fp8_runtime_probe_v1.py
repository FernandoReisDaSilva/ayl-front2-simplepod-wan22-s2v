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
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
REPORT_PATH = REPO_ROOT / "logs" / f"fp8_gate0_{RUN_TIMESTAMP}_report.json"
RUNNER_LOG_PATH = REPO_ROOT / "logs" / f"fp8_gate0_{RUN_TIMESTAMP}_runner.log"
CONTAINER_LOG_PATH = REPO_ROOT / "logs" / f"fp8_gate0_{RUN_TIMESTAMP}_container.log"
INSTANCE_DETAIL_PATH_LOCAL = REPO_ROOT / "logs" / f"fp8_gate0_{RUN_TIMESTAMP}_instance.json"
CREATE_PAYLOAD_PATH = REPO_ROOT / "logs" / f"fp8_gate0_{RUN_TIMESTAMP}_create_payload_sanitized.json"

IMAGE_TAG = "0.3.06-blackwell-fp8-wan-gate0-path-resolution-v1"
IMAGE_REF = f"ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2:{IMAGE_TAG}"
EXPECTED_IMAGE_REF = IMAGE_REF
DATACENTER = "EU-PL-01"
MIN_GPU_MEMORY_MB = 48_000
PROBE_REPORT_PATH = "/tmp/fp8_wan_gate0_probe_v1.json"
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
    "deleted",
    "succeeded",
    "success",
    "error",
)

REPORT_MARKERS = (
    "runtime_certification",
    "Float8WeightOnlyConfig",
    "TEMP_FP8_RUNTIME_PROBE_V1",
    "[TEMP_FP8_RUNTIME_PROBE_V1]",
    "TEMP_FP8_WAN_GATE0_PROBE_V1",
    "[TEMP_FP8_WAN_GATE0_PROBE_V1]",
    "probe_build_id",
    "report_schema_version",
    "fp8_runtime_probe_v1",
    "fp8_runtime_certification_v1",
    "fp8_wan_gate0_probe_v1",
    "torchao",
    "quantize_",
)

ERROR_MARKERS = (
    "Traceback",
    "ModuleNotFoundError",
    "ImportError",
    "RuntimeError",
    "Failed to load",
    "_C_cutlass_90a",
    "_C_mxfp8",
    "CUDA",
    "torchao",
    "Float8WeightOnlyConfig",
)

STRUCTURED_LOG_FIELDS = (
    "probe_build_id",
    "report_schema_version",
    "failure_stage",
    "exception_type",
    "exception_message",
    "missing_path",
    "resolved_path",
    "exception_filename",
    "exception_errno",
    "cwd",
    "probe_file",
    "probe_script_path",
    "loader_entrypoint",
    "environment",
    "loader_preflight",
    "path_checks",
    "detected_mount_points_json",
    "candidate_model_roots_json",
    "model_search_results_json",
    "storage_direct_inventory_json",
    "expected_model_path",
    "configured_model_path",
    "resolved_model_path",
    "model_path_source",
    "model_path_resolution_status",
    "model_path_candidates_json",
    "model_path_validation_json",
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
    "[TEMP_FP8_WAN_GATE0_PROBE_V1]",
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

NON_TERMINAL_STARTUP_STATES = {
    "",
    "unknown",
    "created",
    "pending",
    "queued",
    "preparing",
    "provisioning",
    "pulling",
    "starting",
    "initializing",
}

RUNNING_STATES = {
    "running",
    "active",
    "ready",
}

TERMINAL_STATES = {
    "deleted",
    "terminated",
    "exited",
    "stopped",
    "failed",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def log_runner(message: str) -> None:
    RUNNER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUNNER_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def print_status(message: str) -> None:
    print(message, flush=True)
    log_runner(message)


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
    if not stripped or len(stripped) < 4:
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
    if len(compact) <= 12:
        return decoded, True
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


def collect_image_fields(value, *, path: str = "", max_depth: int = 10, max_items: int = 80) -> list[dict]:
    if max_depth < 0 or max_items <= 0:
        return []
    found: list[dict] = []
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = path_join(path, key)
            key_lower = str(key).lower().replace("_", "")
            if isinstance(item, str) and any(token in key_lower for token in ("image", "tag", "defaulttag")):
                found.append({"path": item_path, "key": str(key), "value": item})
                if len(found) >= max_items:
                    return found
            elif isinstance(item, (dict, list)):
                found.extend(collect_image_fields(item, path=item_path, max_depth=max_depth - 1, max_items=max_items - len(found)))
                if len(found) >= max_items:
                    return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(collect_image_fields(item, path=path_join(path, index), max_depth=max_depth - 1, max_items=max_items - len(found)))
            if len(found) >= max_items:
                return found
    return found


def normalize_image_ref(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.startswith("docker://"):
        text = text[len("docker://"):]
    return text


def image_ref_from_fields(fields: list[dict]) -> tuple[str, str]:
    for field in fields:
        value = normalize_image_ref(str(field.get("value") or ""))
        if "ghcr.io/" in value and ":" in value.rsplit("/", 1)[-1]:
            return value, str(field.get("path") or "")
    image_name = ""
    image_name_path = ""
    image_tag = ""
    image_tag_path = ""
    for field in fields:
        key = str(field.get("key") or "").lower().replace("_", "")
        value = normalize_image_ref(str(field.get("value") or ""))
        if not value:
            continue
        if not image_name and key in {"imagename", "image"} and "ghcr.io/" in value:
            image_name = value.split(":", 1)[0]
            image_name_path = str(field.get("path") or "")
        if not image_tag and key in {"imagetag", "defaulttag", "tag"} and not value.startswith("ghcr.io/"):
            image_tag = value
            image_tag_path = str(field.get("path") or "")
    if image_name and image_tag:
        return f"{image_name}:{image_tag}", f"{image_name_path}+{image_tag_path}"
    return "", ""


def extract_effective_image_ref(detail_json) -> dict:
    fields = collect_image_fields(detail_json)
    image_ref, source_path = image_ref_from_fields(fields)
    return {
        "effective_image_ref": image_ref,
        "effective_image_source_path": source_path,
        "image_fields": fields[:40],
    }


def verify_effective_image(detail_json, expected_image_ref: str = EXPECTED_IMAGE_REF) -> dict:
    extracted = extract_effective_image_ref(detail_json)
    effective = extracted.get("effective_image_ref") or ""
    status = "matched" if effective == expected_image_ref else ("missing_effective_image_ref" if not effective else "mismatch")
    return {
        "status": status,
        "expected_image_ref": expected_image_ref,
        "effective_image_ref": effective,
        **extracted,
    }


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


def normalize_state(value) -> str:
    return str(value or "").strip().lower().replace("_", "-").replace(" ", "-")


def extract_current_status_value(detail_json, status_fields: list[dict] | None = None) -> str:
    if isinstance(detail_json, dict):
        for key in ("containerStatus", "status", "state", "phase", "containerState", "runtimeStatus"):
            value = detail_json.get(key)
            if value is not None:
                return str(value)
        for parent in ("container", "docker", "pod", "task", "job"):
            nested = detail_json.get(parent)
            if isinstance(nested, dict):
                for key in ("containerStatus", "status", "state", "phase", "containerState", "runtimeStatus"):
                    value = nested.get(key)
                    if value is not None:
                        return str(value)
    return extract_status_value(status_fields or [])


def terminal_state_seen(detail_json) -> bool:
    status_fields = collect_instance_status_fields(detail_json)
    current_state = normalize_state(extract_current_status_value(detail_json, status_fields))
    return current_state in TERMINAL_STATES


def combined_instance_text(detail_json) -> str:
    fields = collect_instance_text_fields(detail_json)
    return "\n".join(str(field.get("text_truncated") or "") for field in fields)


def marker_seen(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def classify_startup(detail_json) -> dict:
    status_fields = collect_instance_status_fields(detail_json)
    text = combined_instance_text(detail_json)
    status_value = extract_current_status_value(detail_json, status_fields)
    current_state = normalize_state(status_value)
    text_lower = text.lower()
    matched_probe_markers = [
        marker
        for marker in ("[TEMP_FP8_RUNTIME_PROBE_V1]", "[TEMP_FP8_WAN_GATE0_PROBE_V1]", "runtime_certification", "probe_build_id", "report_schema_version")
        if marker in text or marker in text_lower
    ]
    matched_running_markers = [
        marker
        for marker in ("Image pulled", "Starting container", "Container started")
        if marker in text
    ]
    if any(line.strip().lower() == "running" for line in text.splitlines()):
        matched_running_markers.append("Running")
    if any(line.strip().lower() == "ready" for line in text.splitlines()):
        matched_running_markers.append("Ready")
    matched_terminal_markers = [
        state for state in TERMINAL_STATES if current_state == state
    ]
    image_pull_detected = marker_seen(text, STARTUP_PULL_MARKERS) or current_state in CREATED_OR_PULLING_STATUSES or current_state in {
        "created",
        "pending",
        "queued",
        "preparing",
        "provisioning",
        "pulling",
        "starting",
        "initializing",
    }
    image_pull_completed = marker_seen(text, ("Image pulled",))
    probe_output_detected = bool(matched_probe_markers)
    image_pull_failed = marker_seen(text, IMAGE_PULL_ERROR_MARKERS) or any(marker.lower() in text_lower for marker in IMAGE_PULL_ERROR_MARKERS)
    if current_state in TERMINAL_STATES:
        terminal_seen = True
        container_start_detected = False
    elif current_state in RUNNING_STATES:
        terminal_seen = False
        container_start_detected = True
    elif current_state in NON_TERMINAL_STARTUP_STATES:
        terminal_seen = False
        container_start_detected = False
    else:
        terminal_seen = False
        container_start_detected = bool(matched_running_markers or matched_probe_markers)
    return {
        "status_fields": status_fields,
        "text_fields": collect_instance_text_fields(detail_json),
        "console_text": text,
        "status_value": status_value,
        "current_state_normalized": current_state,
        "container_exit_code": extract_exit_code(status_fields),
        "image_pull_detected": image_pull_detected,
        "image_pull_completed": image_pull_completed,
        "probe_output_detected": probe_output_detected,
        "matched_probe_markers": matched_probe_markers,
        "matched_running_markers": matched_running_markers,
        "matched_terminal_markers": matched_terminal_markers,
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
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if any(str(key).lower() == "runtime_certification" for key in current):
                return True
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return False


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


def parse_structured_probe_fields_from_text(text: str) -> dict:
    fields = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("[TEMP_FP8_"):
            continue
        try:
            _, payload = stripped.split("]", 1)
        except ValueError:
            continue
        payload = payload.strip()
        for key in STRUCTURED_LOG_FIELDS:
            prefix = f"{key}="
            if payload.startswith(prefix):
                fields[key] = payload[len(prefix):]
    return fields


def structured_value(fields: dict, key: str, fallback=None):
    value = fields.get(key)
    if value is None:
        return fallback
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value)
        except Exception:
            parsed = maybe_json_from_string(value)
            if parsed is not None:
                return parsed
    return value


def structured_value_with_parse_error(fields: dict, key: str, fallback=None) -> tuple[object, str]:
    value = fields.get(key)
    if value is None:
        return fallback, ""
    if isinstance(value, str) and value and value[0] in "[{":
        try:
            return json.loads(value), ""
        except Exception as exc:
            return value, f"{type(exc).__name__}: {truncate(exc, 300)}"
    return value, ""


def find_fp8_report_in_text(text: str):
    for line in text.splitlines():
        parsed = maybe_json_from_string(line)
        if parsed is not None:
            report = find_fp8_report(parsed)
            if report is not None:
                return report
    parsed = maybe_json_from_string(text)
    if parsed is not None:
        report = find_fp8_report(parsed)
        if report is not None:
            return report
    return None


def analyze_container_log_text(text: str) -> dict:
    markers_found = [marker for marker in REPORT_MARKERS + ERROR_MARKERS if marker in text]
    failure_markers = [
        marker
        for marker in (
            "Failed to load",
            "_C_cutlass_90a",
            "_C_mxfp8",
            "Traceback",
            "ModuleNotFoundError",
            "ImportError",
            "RuntimeError",
            "FileNotFoundError",
            "manifest unknown",
            "unauthorized",
            "Error pulling image",
        )
        if marker in text
    ]
    torchao_extension_load_errors = []
    for line in text.splitlines():
        if "Failed to load" in line and ("torchao" in line or "_C_cutlass_90a" in line or "_C_mxfp8" in line):
            torchao_extension_load_errors.append(truncate(line.strip(), 1200))
    runtime_certification = parse_runtime_certification_from_text(text)
    structured_probe_fields = parse_structured_probe_fields_from_text(text)
    status_value = ""
    for line in text.splitlines():
        stripped = line.strip()
        if "status=" in stripped.lower() or '"status"' in stripped.lower():
            status_value = truncate(stripped, 1000)
        if "runtime_certification" in stripped.lower():
            break
    traceback_tail = ""
    exception_type_from_logs = ""
    exception_message_from_logs = ""
    if "Traceback" in text:
        traceback_tail = "Traceback" + text.rsplit("Traceback", 1)[-1]
        traceback_tail = truncate(traceback_tail, 6000)
        for line in reversed(traceback_tail.splitlines()):
            stripped = line.strip()
            if ":" not in stripped:
                continue
            candidate_type, candidate_message = stripped.split(":", 1)
            if candidate_type.endswith("Error") or candidate_type.endswith("Exception"):
                exception_type_from_logs = candidate_type.strip()
                exception_message_from_logs = candidate_message.strip()
                break
    return {
        "probe_output_markers_found": markers_found,
        "failure_markers": failure_markers,
        "torchao_extension_load_errors": torchao_extension_load_errors,
        "runtime_certification_detected_from_logs": bool(runtime_certification),
        "runtime_certification_value": runtime_certification,
        "structured_probe_fields": structured_probe_fields,
        "status_line_truncated": status_value,
        "traceback_tail": traceback_tail,
        "exception_type_from_logs": exception_type_from_logs,
        "exception_message_from_logs": exception_message_from_logs,
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
        body_text, body_decoded_from_base64 = display_log_text(result.get("body_text") or "")
        attempt["decoded_from_base64"] = body_decoded_from_base64
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
    report_from_logs = find_fp8_report_in_text(combined_text)
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
        "fp8_runtime_report_from_logs": report_from_logs,
        "report_available_from_logs": report_from_logs is not None,
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
    structured_probe_fields = data.get("structured_probe_fields") or {}
    environment, environment_parse_error = structured_value_with_parse_error(structured_probe_fields, "environment", data.get("environment"))
    loader_preflight, loader_preflight_parse_error = structured_value_with_parse_error(structured_probe_fields, "loader_preflight", data.get("loader_preflight"))
    path_checks, path_checks_parse_error = structured_value_with_parse_error(structured_probe_fields, "path_checks", data.get("path_checks"))
    detected_mount_points, detected_mount_points_parse_error = structured_value_with_parse_error(structured_probe_fields, "detected_mount_points_json", data.get("detected_mount_points"))
    candidate_roots, candidate_roots_parse_error = structured_value_with_parse_error(structured_probe_fields, "candidate_model_roots_json", data.get("candidate_model_roots"))
    model_search_results, model_search_results_parse_error = structured_value_with_parse_error(structured_probe_fields, "model_search_results_json", data.get("model_search_results"))
    storage_direct_inventory, storage_direct_inventory_parse_error = structured_value_with_parse_error(structured_probe_fields, "storage_direct_inventory_json", data.get("storage_direct_inventory"))
    model_path_candidates, model_path_candidates_parse_error = structured_value_with_parse_error(structured_probe_fields, "model_path_candidates_json", data.get("model_path_candidates"))
    model_path_validation, model_path_validation_parse_error = structured_value_with_parse_error(structured_probe_fields, "model_path_validation_json", data.get("model_path_validation"))
    structured_parse_errors = {
        key: value
        for key, value in {
            "environment": environment_parse_error,
            "loader_preflight": loader_preflight_parse_error,
            "path_checks": path_checks_parse_error,
            "detected_mount_points_json": detected_mount_points_parse_error,
            "candidate_model_roots_json": candidate_roots_parse_error,
            "model_search_results_json": model_search_results_parse_error,
            "storage_direct_inventory_json": storage_direct_inventory_parse_error,
            "model_path_candidates_json": model_path_candidates_parse_error,
            "model_path_validation_json": model_path_validation_parse_error,
        }.items()
        if value
    }
    selected = data.get("market_selection", {}).get("selected", {})
    selected_summary = selected.get("selected_summary", {})
    if isinstance(environment, dict):
        internal_image_tag = environment.get("image_tag", "")
    elif isinstance(structured_probe_fields.get("environment"), str) and '"image_tag":"' in structured_probe_fields.get("environment", ""):
        internal_image_tag = structured_probe_fields.get("environment", "").split('"image_tag":"', 1)[1].split('"', 1)[0]
    else:
        internal_image_tag = data.get("internal_image_tag", "")
    return {
        "script_id": SCRIPT_ID,
        "created_at": now_iso(),
        "status": status,
        "dry_run": not args.execute,
        "image_ref": IMAGE_REF,
        "expected_image_ref": EXPECTED_IMAGE_REF,
        "effective_image_ref": data.get("effective_image_ref", ""),
        "effective_image_source_path": data.get("effective_image_source_path", ""),
        "image_verification": data.get("image_verification", {}),
        "template_result": data.get("template_result"),
        "template_id": args.template_id,
        "datacenter": DATACENTER,
        "report_path": str(REPORT_PATH),
        "runner_log_path": str(RUNNER_LOG_PATH),
        "container_log_path": str(CONTAINER_LOG_PATH),
        "instance_json_path": str(INSTANCE_DETAIL_PATH_LOCAL),
        "create_payload_sanitized_path": str(CREATE_PAYLOAD_PATH),
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
        "container_terminal_state_detected": data.get("container_terminal_state_detected", False),
        "terminal_state": data.get("terminal_state", ""),
        "terminal_state_seconds": data.get("terminal_state_seconds"),
        "instance_errors": data.get("instance_errors", []),
        "instance_warnings": data.get("instance_warnings", []),
        "probe_output_markers_found": data.get("probe_output_markers_found", []),
        "failure_markers": data.get("failure_markers", []),
        "torchao_extension_load_errors": data.get("torchao_extension_load_errors", []),
        "final_decoded_console": data.get("final_decoded_console", ""),
        "runtime_certification_detected_from_logs": data.get("runtime_certification_detected_from_logs", False),
        "runtime_certification_value": data.get("runtime_certification_value", ""),
        "structured_probe_fields": structured_probe_fields,
        "probe_build_id": structured_probe_fields.get("probe_build_id") or data.get("probe_build_id"),
        "report_schema_version": structured_probe_fields.get("report_schema_version") or data.get("report_schema_version"),
        "failure_stage": structured_probe_fields.get("failure_stage") or data.get("failure_stage"),
        "exception_type": structured_probe_fields.get("exception_type") or data.get("exception_type") or data.get("exception_type_from_logs"),
        "exception_message": structured_probe_fields.get("exception_message") or data.get("exception_message") or data.get("exception_message_from_logs"),
        "traceback": structured_probe_fields.get("traceback") or data.get("traceback") or data.get("traceback_tail"),
        "traceback_tail": data.get("traceback_tail", ""),
        "internal_image_tag": internal_image_tag,
        "missing_path": structured_probe_fields.get("missing_path") or data.get("missing_path"),
        "resolved_path": structured_probe_fields.get("resolved_path") or data.get("resolved_path"),
        "exception_filename": structured_probe_fields.get("exception_filename") or data.get("exception_filename"),
        "exception_errno": structured_probe_fields.get("exception_errno") or data.get("exception_errno"),
        "cwd": structured_probe_fields.get("cwd") or data.get("cwd"),
        "probe_file": structured_probe_fields.get("probe_file") or data.get("probe_file"),
        "probe_script_path": structured_probe_fields.get("probe_script_path") or data.get("probe_script_path"),
        "loader_entrypoint": structured_probe_fields.get("loader_entrypoint") or data.get("loader_entrypoint"),
        "environment": environment,
        "loader_preflight": loader_preflight,
        "path_checks": path_checks,
        "detected_mount_points": detected_mount_points,
        "candidate_model_roots": candidate_roots,
        "model_search_results": model_search_results,
        "storage_direct_inventory": storage_direct_inventory,
        "expected_model_path": structured_probe_fields.get("expected_model_path") or data.get("expected_model_path"),
        "configured_model_path": structured_probe_fields.get("configured_model_path") or data.get("configured_model_path"),
        "resolved_model_path": structured_probe_fields.get("resolved_model_path") or data.get("resolved_model_path"),
        "model_path_source": structured_probe_fields.get("model_path_source") or data.get("model_path_source"),
        "model_path_candidates": model_path_candidates,
        "model_path_validation": model_path_validation,
        "model_path_resolution_status": structured_probe_fields.get("model_path_resolution_status") or data.get("model_path_resolution_status"),
        "structured_probe_parse_errors": structured_parse_errors,
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


def wait_for_container_start(
    base_url: str,
    api_key: str,
    instance_id: int,
    timeout_seconds: int,
    poll_interval_seconds: int,
    *,
    debug_startup_classification: bool = False,
) -> dict:
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
        if debug_startup_classification:
            print(
                json.dumps(
                    {
                        "status_value": classification.get("status_value"),
                        "container_start_detected": classification.get("container_start_detected"),
                        "terminal_state_seen": classification.get("terminal_state_seen"),
                        "image_pull_detected": classification.get("image_pull_detected"),
                        "image_pull_completed": classification.get("image_pull_completed"),
                        "image_pull_failed": classification.get("image_pull_failed"),
                        "matched_probe_markers": classification.get("matched_probe_markers", []),
                        "matched_running_markers": classification.get("matched_running_markers", []),
                        "matched_terminal_markers": classification.get("matched_terminal_markers", []),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
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
        if classification.get("terminal_state_seen"):
            return {
                "status": "container_terminal_before_probe_monitor",
                "startup_seconds": round(time.monotonic() - started_monotonic, 3),
                "terminal_state_seconds": round(time.monotonic() - started_monotonic, 3),
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
        time.sleep(max(1, poll_interval_seconds))

    timeout_status = "blocked_image_pull_timeout" if latest_classification.get("image_pull_detected") else "blocked_startup_timeout"
    return {
        "status": timeout_status,
        "startup_seconds": round(time.monotonic() - started_monotonic, 3),
        "latest_detail_json": latest_detail_json,
        "latest_classification": latest_classification,
        "status_history": status_history,
    }


def monitor_probe(
    base_url: str,
    api_key: str,
    instance_id: int,
    timeout_seconds: int,
    poll_interval_seconds: int,
    *,
    debug_probe_monitor: bool = False,
) -> dict:
    started_monotonic = time.monotonic()
    deadline = time.monotonic() + max(1, timeout_seconds)
    failure_grace_deadline = None
    failure_grace_seconds = 30
    detail_path = smoke.INSTANCE_DETAIL_PATH.format(id=instance_id)
    observations = []
    found_report = None
    terminal_seen = False
    logs_endpoint_unavailable_seen = False
    report_channel_unavailable_seen = False
    runtime_certification_value = ""
    runtime_certification_detected = False
    explicit_failure_seen = False
    matched_failure_markers = []
    latest_log_collection = {}
    latest_structured_probe_fields = {}
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
        log_collection = collect_container_logs(base_url, api_key, instance_id, detail_json)
        latest_log_collection = log_collection
        latest_structured_probe_fields = dict(log_collection.get("structured_probe_fields") or {})
        if log_collection.get("fp8_runtime_report_from_logs") is not None and report is None:
            report = log_collection.get("fp8_runtime_report_from_logs")
        log_certification = str(log_collection.get("runtime_certification_value") or "").upper()
        report_summary = summarize_fp8_report(report)
        report_certification = str(report_summary.get("runtime_certification") or "").upper()
        runtime_certification_value = report_certification or log_certification
        runtime_certification_detected = bool(runtime_certification_value)
        matched_failure_markers = list(log_collection.get("failure_markers") or [])
        structured_failure_seen = bool(
            latest_structured_probe_fields.get("failure_stage")
            or latest_structured_probe_fields.get("exception_type")
        )
        explicit_failure_seen = runtime_certification_value == "FAIL" or bool(matched_failure_markers) or structured_failure_seen
        endpoint_attempts = log_collection.get("container_log_endpoint_attempts") or []
        logs_endpoint_unavailable = bool(endpoint_attempts) and not any(
            attempt.get("http_status_code") == 200 and (attempt.get("body_truncated") or "").strip()
            for attempt in endpoint_attempts
        )
        logs_endpoint_unavailable_seen = logs_endpoint_unavailable_seen or logs_endpoint_unavailable
        report_channel_unavailable = report is None and not log_collection.get("container_logs_retrieved")
        report_channel_unavailable_seen = report_channel_unavailable_seen or report_channel_unavailable
        terminal_now = terminal_state_seen(detail_json)
        container_status = extract_current_status_value(detail_json, latest_status_fields)
        if report is not None:
            next_action = "return_report_found"
        elif runtime_certification_value == "PASS":
            next_action = "return_runtime_certification_pass"
        elif explicit_failure_seen and runtime_certification_value == "FAIL" and report is None and not terminal_now:
            if failure_grace_deadline is None:
                failure_grace_deadline = min(deadline, time.monotonic() + failure_grace_seconds)
            next_action = "wait_for_failure_report_after_fail_marker" if time.monotonic() < failure_grace_deadline else "return_runtime_certification_failed"
        elif explicit_failure_seen:
            next_action = "return_probe_failed" if structured_failure_seen and runtime_certification_value != "FAIL" else "return_runtime_certification_failed"
        elif terminal_now:
            next_action = "return_terminal_without_report"
        else:
            next_action = "continue_polling"
        observation = {
            "observed_at": now_iso(),
            "poll": polls,
            "request": safe_result(detail_result),
            "instance": safe_instance_observation(detail_json),
            "status_fields": latest_status_fields[:30],
            "fp8_report_marker_seen": report is not None,
            "terminal_state_seen": terminal_now,
            "container_status": container_status,
            "logs_available": bool(log_collection.get("container_logs_retrieved")),
            "report_available": report is not None,
            "runtime_certification_detected": runtime_certification_detected,
            "runtime_certification_value": runtime_certification_value,
            "matched_failure_markers": matched_failure_markers,
            "structured_failure_seen": structured_failure_seen,
            "failure_grace_active": bool(failure_grace_deadline and time.monotonic() < failure_grace_deadline),
            "probe_state": "probe_report_found" if report is not None else ("probe_terminal_without_report" if terminal_now else "probe_running_no_report_yet"),
            "next_action": next_action,
        }
        observations.append(observation)
        print(
            f"[{SCRIPT_ID}] probe poll={polls} status={container_status or 'unknown'} "
            f"logs={str(observation['logs_available']).lower()} "
            f"report={str(observation['report_available']).lower()} "
            f"cert={runtime_certification_value or 'none'} next={next_action}",
            flush=True,
        )
        if debug_probe_monitor:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                        "container_status": container_status,
                        "terminal_state_seen": terminal_now,
                        "logs_available": bool(log_collection.get("container_logs_retrieved")),
                        "report_available": report is not None,
                        "runtime_certification_detected": runtime_certification_detected,
                        "runtime_certification_value": runtime_certification_value,
                        "matched_failure_markers": matched_failure_markers,
                        "structured_failure_seen": structured_failure_seen,
                        "failure_grace_active": bool(failure_grace_deadline and time.monotonic() < failure_grace_deadline),
                        "next_action": next_action,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if report is not None:
            found_report = report
            break
        if runtime_certification_value == "PASS":
            found_report = {
                "runtime_certification": "PASS",
                "status": "succeeded_recovered_from_container_logs",
                "source": "container_logs",
            }
            break
        if explicit_failure_seen and runtime_certification_value == "FAIL" and report is None and not terminal_now:
            if failure_grace_deadline is not None and time.monotonic() < failure_grace_deadline:
                time.sleep(max(1, poll_interval_seconds))
                continue
            break
        if explicit_failure_seen:
            break
        if terminal_now:
            terminal_seen = True
            break
        time.sleep(max(1, poll_interval_seconds))

    if found_report is not None:
        status = "completed"
    elif explicit_failure_seen:
        status = "probe_failed" if latest_structured_probe_fields.get("failure_stage") and runtime_certification_value != "FAIL" else "runtime_certification_failed"
    elif terminal_seen:
        status = "probe_terminal_without_report"
    else:
        status = "probe_timeout"
    if status == "probe_timeout" and logs_endpoint_unavailable_seen:
        probe_state = "probe_logs_endpoint_unavailable"
    elif status == "probe_timeout" and report_channel_unavailable_seen:
        probe_state = "probe_report_channel_unavailable"
    elif status == "probe_timeout":
        probe_state = "probe_timeout"
    elif status == "probe_terminal_without_report":
        probe_state = "probe_terminal_without_report"
    elif status == "runtime_certification_failed":
        probe_state = "runtime_certification_failed"
    elif status == "probe_failed":
        probe_state = "probe_failed"
    else:
        probe_state = "probe_report_found"
    return {
        "status": status,
        "probe_state": probe_state,
        "polls": polls,
        "observations": observations[-20:],
        "report_found": found_report is not None,
        "terminal_state_seen": terminal_seen,
        "terminal_state_seconds": round(time.monotonic() - started_monotonic, 3) if terminal_seen else None,
        "report_retrieval_note": "The FP8 image does not expose FastAPI/R2. Polling continues until explicit report/certification, terminal state, or probe timeout.",
        "fp8_runtime_report": found_report,
        "latest_detail_json": latest_detail_json,
        "latest_status_fields": latest_status_fields,
        "container_exit_code": extract_exit_code(latest_status_fields),
        "container_status": extract_current_status_value(latest_detail_json, latest_status_fields),
        "probe_seconds": round(time.monotonic() - started_monotonic, 3),
        "runtime_certification_detected": runtime_certification_detected,
        "runtime_certification_value": runtime_certification_value,
        "matched_failure_markers": matched_failure_markers,
        "structured_probe_fields": latest_structured_probe_fields,
        "logs_endpoint_unavailable_seen": logs_endpoint_unavailable_seen,
        "report_channel_unavailable_seen": report_channel_unavailable_seen,
        "latest_log_collection": latest_log_collection,
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
    parser.add_argument("--debug-startup-classification", action="store_true", help="Print safe startup classification details on each poll.")
    parser.add_argument("--debug-probe-monitor", action="store_true", help="Print safe probe monitor details on each poll.")
    parser.add_argument("--run-mock-tests", action="store_true", help=argparse.SUPPRESS)
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


def b64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def make_mock_instance(status_value: str, console_text: str = "", **extra) -> dict:
    payload = {
        "id": 123,
        "status": status_value,
        "containerStatus": status_value,
        "console": b64_text(console_text) if console_text else "",
        "consoleSystem": b64_text(console_text) if console_text else "",
    }
    payload.update(extra)
    return payload


def run_mock_tests() -> int:
    original_http_request = smoke.http_request
    original_sleep = time.sleep
    original_monotonic = time.monotonic
    original_http_text_request = globals()["http_text_request"]

    def announce(name: str, extra: str = "") -> None:
        suffix = f" {extra}" if extra else ""
        print(f"{name}: PASS{suffix}", flush=True)

    def require_classification(name: str, payload: dict, expected: dict) -> dict:
        classification = classify_startup(payload)
        for key, value in expected.items():
            assert classification.get(key) == value, {"name": name, "expected": expected, "actual": classification}
        announce(name)
        return classification

    class FakeClock:
        def __init__(self) -> None:
            self.current = 0.0

        def monotonic(self) -> float:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += max(0.001, float(seconds))

    def make_text_response(path: str, text: str = "", *, http_status_code: int = 200) -> dict:
        return {
            "attempted": True,
            "status": "succeeded" if http_status_code == 200 else "failed",
            "method": "GET",
            "path": path,
            "http_status_code": http_status_code,
            "content_type": "text/plain",
            "body_bytes": len(text.encode("utf-8")),
            "body_text": text,
            "body_truncated": truncate(text, 8000),
        }

    def run_monitor_mock(detail_sequence: list[dict], log_sequence: list[str | None], *, timeout_seconds: int = 30) -> tuple[dict, dict]:
        clock = FakeClock()
        calls = {"detail": 0, "text": 0}

        def fake_monitor_http_request(_base_url, _path, _api_key="", **_kwargs):
            index = min(calls["detail"], len(detail_sequence) - 1)
            calls["detail"] += 1
            return {
                "attempted": True,
                "status": "succeeded",
                "method": "GET",
                "path": _path,
                "http_status_code": 200,
                "json": detail_sequence[index],
            }

        def fake_monitor_text_request(_base_url, path, _api_key, timeout_seconds=30):
            poll_index = max(0, calls["detail"] - 1)
            text = log_sequence[min(poll_index, len(log_sequence) - 1)] if log_sequence else ""
            calls["text"] += 1
            if text is None:
                return make_text_response(path, "", http_status_code=404)
            return make_text_response(path, text, http_status_code=200)

        smoke.http_request = fake_monitor_http_request
        globals()["http_text_request"] = fake_monitor_text_request
        time.monotonic = clock.monotonic
        time.sleep = clock.sleep
        result = monitor_probe("https://mock.simplepod.local", "token", 123, timeout_seconds, 0)
        return result, calls

    try:
        time.sleep = lambda _seconds: None

        payload = runtime_payload("/instances/market/mock", 26108)
        env_names = {item.get("name") for item in payload.get("envVariables", [])}
        assert "AYL_IMAGE_TAG" not in env_names, payload
        assert "AYL_RUNTIME_VERSION" not in env_names, payload
        payload_text = json.dumps(smoke.redact_value("payload", payload), ensure_ascii=False)
        assert "SECRET" not in payload_text.upper() and "TOKEN" not in payload_text.upper(), payload_text
        announce("payload_uses_template_image_without_env_override")

        matching_instance = {
            "imageName": "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2",
            "imageTag": "0.3.06-blackwell-fp8-wan-gate0-path-resolution-v1",
        }
        matching_verify = verify_effective_image(matching_instance)
        assert matching_verify["status"] == "matched", matching_verify
        assert matching_verify["effective_image_ref"] == EXPECTED_IMAGE_REF, matching_verify
        announce("image_verification_expected_template_image_matches")

        mismatch_instance = {
            "imageName": "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2",
            "imageTag": "0.3.04-blackwell-fp8-wan-gate0-mount-audit-v1",
        }
        mismatch_verify = verify_effective_image(mismatch_instance)
        assert mismatch_verify["status"] == "mismatch", mismatch_verify
        assert mismatch_verify["effective_image_ref"].endswith(":0.3.04-blackwell-fp8-wan-gate0-mount-audit-v1"), mismatch_verify
        announce("image_verification_rejects_old_0304_tag")

        missing_image_verify = verify_effective_image({"status": "running"})
        assert missing_image_verify["status"] == "missing_effective_image_ref", missing_image_verify
        announce("image_verification_requires_effective_image")

        module_not_found_log = "\n".join(
            [
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] runtime_certification=FAIL",
                "Traceback (most recent call last):",
                "  File \"/opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe/temp_fp8_wan_gate0_probe_v1.py\", line 1, in <module>",
                "    import missing_wan_dependency",
                "ModuleNotFoundError: No module named 'missing_wan_dependency'",
            ]
        )
        module_not_found_analysis = analyze_container_log_text(module_not_found_log)
        assert module_not_found_analysis["exception_type_from_logs"] == "ModuleNotFoundError", module_not_found_analysis
        assert "missing_wan_dependency" in module_not_found_analysis["exception_message_from_logs"], module_not_found_analysis
        assert "Traceback" in module_not_found_analysis["traceback_tail"], module_not_found_analysis
        announce("module_not_found_traceback_preserved_from_logs")

        require_classification(
            "startup_classification_created_pure",
            {
                "id": 123,
                "instanceId": 123,
                "containerId": "abc123",
                "status": "created",
                "containerStatus": "created",
                "startedAt": "2026-07-16T00:00:00Z",
                "history": [{"status": "queued"}, {"status": "created"}],
                "console": b64_text("Your instance will be ready shortly. Download starting, please wait up to 5 minutes.\n"),
            },
            {
                "status_value": "created",
                "container_start_detected": False,
                "terminal_state_seen": False,
                "image_pull_failed": False,
            },
        )

        require_classification(
            "startup_classification_created_historical_running",
            {
                "status": "created",
                "containerStatus": "created",
                "history": [{"status": "running"}, {"message": "Container started previously"}],
                "console": b64_text("Historical state: running\n"),
            },
            {
                "status_value": "created",
                "container_start_detected": False,
                "terminal_state_seen": False,
            },
        )

        require_classification(
            "startup_classification_created_torchao_error",
            make_mock_instance(
                "created",
                "torchao failed to load _C_cutlass_90a and _C_mxfp8\n",
            ),
            {
                "status_value": "created",
                "container_start_detected": False,
                "terminal_state_seen": False,
            },
        )

        require_classification(
            "startup_classification_running",
            make_mock_instance("running", "Running\n"),
            {
                "status_value": "running",
                "container_start_detected": True,
                "terminal_state_seen": False,
            },
        )

        require_classification(
            "startup_classification_deleted_with_old_probe_marker",
            make_mock_instance(
                "deleted",
                "[TEMP_FP8_RUNTIME_PROBE_V1] runtime_certification=PASS\n",
            ),
            {
                "status_value": "deleted",
                "container_start_detected": False,
                "terminal_state_seen": True,
            },
        )

        created = make_mock_instance("created", "[mock] Preparing instance...\n")
        created_again = make_mock_instance("created", "[mock] Preparing instance...\n")
        pulling = make_mock_instance("created", "[mock] Pulling image...\n")
        running = make_mock_instance("running", "[mock] Running\n")
        sequence = [created, created_again, pulling, running]
        calls = {"count": 0}

        def fake_startup_http_request(_base_url, _path, _api_key="", **_kwargs):
            index = min(calls["count"], len(sequence) - 1)
            calls["count"] += 1
            return {
                "attempted": True,
                "status": "succeeded",
                "method": "GET",
                "path": _path,
                "http_status_code": 200,
                "json": sequence[index],
            }

        smoke.http_request = fake_startup_http_request
        startup_result = wait_for_container_start("https://mock.simplepod.local", "token", 123, 30, 0)
        assert startup_result["status"] == "container_started", startup_result
        assert calls["count"] == 4, startup_result
        assert startup_result["status_history"][0]["container_status"] == "created", startup_result
        assert startup_result["status_history"][2]["image_pull_detected"] is True, startup_result
        assert startup_result["latest_classification"]["container_start_detected"] is True, startup_result
        announce("startup_poll_created_created_pulling_running", f"polls={calls['count']}")

        probe_sequence = [
            make_mock_instance(
                "running",
                "[TEMP_FP8_RUNTIME_PROBE_V1] status=succeeded runtime_certification=PASS\n",
                report={"runtime_certification": "PASS"},
            )
        ]

        def fake_probe_http_request(_base_url, _path, _api_key="", **_kwargs):
            return {
                "attempted": True,
                "status": "succeeded",
                "method": "GET",
                "path": _path,
                "http_status_code": 200,
                "json": probe_sequence[0],
            }

        smoke.http_request = fake_probe_http_request
        globals()["http_text_request"] = lambda _base_url, path, _api_key, timeout_seconds=30: make_text_response(path, "", http_status_code=404)
        probe_result = monitor_probe("https://mock.simplepod.local", "token", 123, 30, 0)
        assert probe_result["status"] == "completed", probe_result
        assert probe_result["report_found"] is True, probe_result

        deleted_sequence = [
            make_mock_instance("created", "[mock] Preparing instance...\n"),
            make_mock_instance("created", "[mock] Pulling image...\n"),
            make_mock_instance("deleted", "[mock] Failed to load /usr/local/lib/python3.10/dist-packages/torchao/_C_cutlass_90a.abi3.so\n"),
        ]
        deleted_calls = {"count": 0}

        def fake_deleted_http_request(_base_url, _path, _api_key="", **_kwargs):
            index = min(deleted_calls["count"], len(deleted_sequence) - 1)
            deleted_calls["count"] += 1
            return {
                "attempted": True,
                "status": "succeeded",
                "method": "GET",
                "path": _path,
                "http_status_code": 200,
                "json": deleted_sequence[index],
            }

        def fake_text_request(_base_url, path, _api_key, timeout_seconds=30):
            return {
                "attempted": True,
                "status": "failed",
                "method": "GET",
                "path": path,
                "http_status_code": 404,
                "body_text": "",
                "body_truncated": "",
            }

        smoke.http_request = fake_deleted_http_request
        globals()["http_text_request"] = fake_text_request
        deleted_result = wait_for_container_start("https://mock.simplepod.local", "token", 123, 30, 0)
        assert deleted_result["status"] == "container_terminal_before_probe_monitor", deleted_result
        assert deleted_calls["count"] == 3, deleted_result
        assert deleted_result["latest_classification"]["terminal_state_seen"] is True, deleted_result
        assert deleted_result["latest_classification"]["container_start_detected"] is False, deleted_result
        log_collection = collect_container_logs("https://mock.simplepod.local", "token", 123, deleted_result["latest_detail_json"])
        assert log_collection["container_logs_retrieved"] is True, log_collection
        assert log_collection["torchao_extension_load_errors"], log_collection
        assert "_C_cutlass_90a" in log_collection["container_logs_truncated"], log_collection
        announce("startup_poll_created_pulling_deleted", f"polls={deleted_calls['count']}")

        idempotent_delete_result = safe_result(
            {
                "attempted": True,
                "status": "failed",
                "method": "DELETE",
                "path": "/instances/123",
                "http_status_code": 404,
                "error_type": "HTTPError",
                "error_truncated": "HTTP Error 404: Not Found",
            }
        )
        assert idempotent_delete_result["http_status_code"] == 404, idempotent_delete_result

        running_empty = make_mock_instance("running", "")
        pass_log = "[TEMP_FP8_RUNTIME_PROBE_V1] status=succeeded runtime_certification=PASS\n"
        result, calls = run_monitor_mock(
            [running_empty, running_empty, running_empty, running_empty],
            ["", "", "", pass_log],
        )
        assert result["status"] == "completed", result
        assert result["runtime_certification_value"] == "PASS", result
        assert calls["detail"] == 4, result
        announce("probe_monitor_running_no_logs_then_pass", f"polls={calls['detail']}")

        result, calls = run_monitor_mock(
            [running_empty, running_empty, running_empty, running_empty],
            ["", "", "", ""],
            timeout_seconds=3,
        )
        assert result["status"] == "probe_timeout", result
        assert calls["detail"] >= 2, result
        announce("probe_monitor_timeout_waits_full_window")

        detail_pass = make_mock_instance(
            "running",
            "",
            report={"runtime_certification": "PASS", "status": "succeeded"},
        )
        result, calls = run_monitor_mock(
            [running_empty, running_empty, detail_pass],
            [None, None, None],
        )
        assert result["status"] == "completed", result
        assert result["report_found"] is True, result
        assert result["logs_endpoint_unavailable_seen"] is True, result
        assert calls["detail"] == 3, result
        announce("probe_monitor_logs_endpoint_unavailable_continues")

        deleted_no_report = make_mock_instance("deleted", "")
        result, _calls = run_monitor_mock(
            [running_empty, deleted_no_report],
            ["", ""],
        )
        assert result["status"] == "probe_terminal_without_report", result
        announce("probe_monitor_terminal_without_report")

        fail_log = "[TEMP_FP8_RUNTIME_PROBE_V1] status=failed runtime_certification=FAIL error=_C_mxfp8\n"
        result, _calls = run_monitor_mock(
            [running_empty],
            [fail_log],
        )
        assert result["status"] == "runtime_certification_failed", result
        assert result["runtime_certification_value"] == "FAIL", result
        announce("probe_monitor_explicit_fail")

        result, calls = run_monitor_mock(
            [
                running_empty,
                running_empty,
                make_mock_instance("running", "", report={"runtime_certification": "FAIL", "status": "failed", "exception_type": "ModuleNotFoundError"}),
            ],
            [fail_log, fail_log, None],
            timeout_seconds=45,
        )
        assert result["status"] == "completed", result
        assert result["report_found"] is True, result
        assert calls["detail"] == 3, result
        announce("probe_monitor_waits_briefly_for_report_after_fail_marker")

        structured_failure_no_cert = "\n".join(
            [
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] probe_build_id=gate0-mount-audit-v1",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] report_schema_version=fp8-wan-gate0-v3",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] failure_stage=wan_load",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_type=FileNotFoundError",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_message=[Errno 2] No structurally valid Wan model dir found: '/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B'",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] configured_model_path=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] resolved_model_path=",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] model_path_resolution_status=failed_no_structurally_valid_model_dir",
            ]
        )
        result, _calls = run_monitor_mock([running_empty], [structured_failure_no_cert])
        assert result["status"] == "probe_failed", result
        assert result["probe_state"] == "probe_failed", result
        assert result["structured_probe_fields"]["failure_stage"] == "wan_load", result
        assert result["structured_probe_fields"]["model_path_resolution_status"] == "failed_no_structurally_valid_model_dir", result
        announce("probe_monitor_structured_failure_without_certification_classified")

        gate0_fail_log = "\n".join(
            [
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] probe_build_id=gate0-mount-audit-v1",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] report_schema_version=fp8-wan-gate0-v3",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] failure_stage=wan_load",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_type=FileNotFoundError",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_message=[Errno 2] Wan model dir not found: '/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B'",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] missing_path=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] resolved_path=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_filename=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] exception_errno=2",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] cwd=/opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] probe_file=/opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe/temp_fp8_wan_gate0_probe_v1.py",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] probe_script_path=/opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe/temp_fp8_wan_gate0_probe_v1.py",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] loader_entrypoint=wan.speech2video.WanS2V",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] environment={\"image_tag\":\"0.3.04-blackwell-fp8-wan-gate0-mount-audit-v1\",\"wan_commit\":\"42bf4cfaa384bc21833865abc2f9e6c0e67233dc\"}",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] loader_preflight={\"loader_entrypoint\":\"wan.speech2video.WanS2V\",\"path_checks\":[{\"label\":\"model_dir\",\"exists\":false}]}",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] path_checks=[{\"label\":\"model_dir\",\"exists\":false}]",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] detected_mount_points_json=[\"/mnt\",\"/runpod-volume\"]",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] candidate_model_roots_json=[\"/mnt\",\"/runpod-volume\"]",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] model_search_results_json=[{\"path\":\"/runpod-volume/wan2.2/Wan2.2-S2V-14B\",\"exists\":true,\"is_dir\":true,\"depth\":2,\"matched_name\":\"Wan2.2-S2V-14B\"}]",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] expected_model_path=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B",
                "[TEMP_FP8_WAN_GATE0_PROBE_V1] runtime_certification=FAIL",
            ]
        )
        result, _calls = run_monitor_mock([running_empty], [gate0_fail_log])
        assert result["status"] == "runtime_certification_failed", result
        assert result["runtime_certification_value"] == "FAIL", result
        assert result["structured_probe_fields"]["probe_build_id"] == "gate0-mount-audit-v1", result
        assert result["structured_probe_fields"]["missing_path"] == "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B", result
        mock_args = argparse.Namespace(
            execute=True,
            template_id=26108,
            startup_timeout_seconds=1200,
            probe_timeout_seconds=300,
            instance_market="",
            confirm_delete=True,
        )
        recovered_report = build_report(
            mock_args,
            "failed_recovered_from_container_logs",
            {
                "structured_probe_fields": result["structured_probe_fields"],
                "runtime_certification_detected_from_logs": True,
                "runtime_certification_value": "FAIL",
            },
        )
        assert recovered_report["probe_build_id"] == "gate0-mount-audit-v1", recovered_report
        assert recovered_report["report_schema_version"] == "fp8-wan-gate0-v3", recovered_report
        assert recovered_report["failure_stage"] == "wan_load", recovered_report
        assert recovered_report["exception_type"] == "FileNotFoundError", recovered_report
        assert recovered_report["probe_script_path"].endswith("temp_fp8_wan_gate0_probe_v1.py"), recovered_report
        assert recovered_report["environment"]["wan_commit"] == "42bf4cfaa384bc21833865abc2f9e6c0e67233dc", recovered_report
        assert recovered_report["loader_preflight"]["path_checks"][0]["exists"] is False, recovered_report
        assert recovered_report["path_checks"][0]["label"] == "model_dir", recovered_report
        assert recovered_report["detected_mount_points"] == ["/mnt", "/runpod-volume"], recovered_report
        assert recovered_report["candidate_model_roots"] == ["/mnt", "/runpod-volume"], recovered_report
        assert recovered_report["model_search_results"][0]["matched_name"] == "Wan2.2-S2V-14B", recovered_report
        assert recovered_report["expected_model_path"] == "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B", recovered_report
        assert not recovered_report["structured_probe_parse_errors"], recovered_report
        assert recovered_report["exception_message"], recovered_report
        assert recovered_report["missing_path"] == "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B", recovered_report
        assert recovered_report["resolved_path"] == "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B", recovered_report
        announce("recovery_mount_fields_preserved")
        announce("probe_monitor_gate0_structured_failure_fields_preserved")

        json_pass = make_mock_instance(
            "running",
            "",
            report={"runtime_certification": "PASS", "status": "succeeded", "torch_version": "mock"},
        )
        result, _calls = run_monitor_mock(
            [json_pass],
            [""],
        )
        assert result["status"] == "completed", result
        assert result["runtime_certification_value"] == "PASS", result
        announce("probe_monitor_json_pass")

        start_scripts = make_mock_instance("running", "Running start scripts...\n")
        result, calls = run_monitor_mock(
            [start_scripts, start_scripts, make_mock_instance("running", "")],
            ["", "", pass_log],
        )
        assert result["status"] == "completed", result
        assert calls["detail"] == 3, result
        first_observation = result["observations"][0]
        assert first_observation["next_action"] == "continue_polling", result
        announce("probe_monitor_start_scripts_only_continues")

        print(f"[{SCRIPT_ID}] mock_tests=passed", flush=True)
        return 0
    finally:
        smoke.http_request = original_http_request
        time.sleep = original_sleep
        time.monotonic = original_monotonic
        globals()["http_text_request"] = original_http_text_request


def main() -> int:
    args = parse_args()
    if args.run_mock_tests:
        return run_mock_tests()

    started_monotonic = time.monotonic()
    timer = PhaseTimer(emit=True)
    data: dict = {"phase_timings": timer.phases}
    instance_id = None
    status = "unknown"
    api_key = ""
    base_url = smoke.DEFAULT_BASE_URL
    startup_status = "not_started"
    startup_completed = False
    container_terminal_state_detected = False
    terminal_state = None
    terminal_state_seconds = None
    latest_detail_json = None
    latest_classification = {}
    startup_result = {}
    monitor_result = {}
    log_collection = {}

    print_status(f"[{SCRIPT_ID}] START dry_run={str(not args.execute).lower()} template_id={args.template_id}")
    print_status(f"[{SCRIPT_ID}] expected_image_ref={EXPECTED_IMAGE_REF}")
    print_status(f"[{SCRIPT_ID}] image_source_of_truth=template_{args.template_id}_plus_effective_instance_verification")
    print_status(f"[{SCRIPT_ID}] no_wan=true no_models=true no_r2=true no_inference=true no_video=true")

    valid, validation_status = validate_args(args)
    payload = runtime_payload(args.instance_market, args.template_id)
    data["request_payload_redacted"] = smoke.redact_value("request_payload", payload)
    write_json(CREATE_PAYLOAD_PATH, data["request_payload_redacted"])
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
            print_status(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}")
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
            print_status(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        with timer.phase("load_auth_env"):
            smoke.load_repo_dotenv()
            api_key = os.getenv(smoke.API_KEY_ENV, "")
            base_url = os.getenv(smoke.BASE_URL_ENV, smoke.DEFAULT_BASE_URL)

        if not api_key:
            status = "missing_simplepod_api_key"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print_status(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("template_readonly_audit"):
            template_result = smoke.http_request(base_url, f"/instances/templates/{args.template_id}", api_key, timeout_seconds=30)
        data["template_result"] = {
            "request": safe_result(template_result),
            "image_extraction": extract_effective_image_ref(template_result.get("json")),
        }

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
            print_status(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        payload = runtime_payload(market, args.template_id)
        data["request_payload_redacted"] = smoke.redact_value("request_payload", payload)
        write_json(CREATE_PAYLOAD_PATH, data["request_payload_redacted"])
        with timer.phase("start_instance"):
            start_result = smoke.http_request(base_url, smoke.START_INSTANCE_PATH, api_key, method="POST", payload=payload, timeout_seconds=60)
        data["start_result"] = safe_result(start_result)
        instance_id = smoke.extract_instance_id(start_result.get("json"))
        data["instance_id"] = instance_id
        if start_result.get("status") != "succeeded" or instance_id is None:
            status = "start_failed"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print_status(f"[{SCRIPT_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        with timer.phase("verify_effective_image"):
            observable_status, observable_observations = wait_for_instance_observable(
                base_url,
                api_key,
                instance_id,
                timeout_seconds=60,
                poll_interval_seconds=max(1, args.poll_interval_seconds),
            )
        latest_observable_json = observable_observations[-1].get("request", {}) if observable_observations else {}
        detail_result = smoke.http_request(base_url, smoke.INSTANCE_DETAIL_PATH.format(id=instance_id), api_key, timeout_seconds=30)
        detail_json_for_image = detail_result.get("json")
        if isinstance(detail_json_for_image, dict):
            write_json(INSTANCE_DETAIL_PATH_LOCAL, detail_json_for_image)
        image_verification = verify_effective_image(detail_json_for_image)
        data["image_verification"] = image_verification
        data["effective_image_ref"] = image_verification.get("effective_image_ref", "")
        data["effective_image_source_path"] = image_verification.get("effective_image_source_path", "")
        data["monitoring"] = {
            "image_observable": {
                "status": observable_status,
                "observations": observable_observations[-10:],
                "latest_observable_request": latest_observable_json,
            }
        }
        if image_verification["status"] != "matched":
            status = "image_mismatch"
            data["runtime_seconds"] = round(time.monotonic() - started_monotonic, 3)
            write_json(REPORT_PATH, build_report(args, status, data))
            print_status(
                f"[{SCRIPT_ID}] DONE status={status} expected_image_ref={EXPECTED_IMAGE_REF} "
                f"effective_image_ref={image_verification.get('effective_image_ref') or '<missing>'} report={REPORT_PATH}"
            )
            return 1

        with timer.phase("wait_container_startup"):
            startup_result = wait_for_container_start(
                base_url,
                api_key,
                instance_id,
                args.startup_timeout_seconds,
                args.poll_interval_seconds,
                debug_startup_classification=args.debug_startup_classification,
            )
        data.setdefault("monitoring", {})
        data["monitoring"]["startup"] = {
            key: value
            for key, value in startup_result.items()
            if key not in {"latest_detail_json", "latest_classification"}
        }
        latest_detail_json = startup_result.get("latest_detail_json")
        latest_classification = startup_result.get("latest_classification") or {}
        startup_status = startup_result.get("status") or "unknown"
        startup_completed = startup_status == "container_started"
        container_terminal_state_detected = startup_status == "container_terminal_before_probe_monitor"
        terminal_state = latest_classification.get("status_value") or None
        terminal_state_seconds = startup_result.get("terminal_state_seconds")
        startup_text_fields = latest_classification.get("text_fields") or []
        data["startup_seconds"] = startup_result.get("startup_seconds")
        data["image_pull_detected"] = bool(latest_classification.get("image_pull_detected"))
        data["image_pull_completed"] = bool(latest_classification.get("image_pull_completed"))
        data["container_start_detected"] = bool(latest_classification.get("container_start_detected"))
        data["decoded_console"] = decoded_field_by_name(startup_text_fields, "console")
        data["decoded_console_system"] = decoded_field_by_name(startup_text_fields, "consoleSystem")
        data["final_decoded_console"] = latest_classification.get("console_text") or ""
        data["status_history"] = startup_result.get("status_history", [])[-80:]
        data["container_terminal_state_detected"] = container_terminal_state_detected
        data["terminal_state"] = terminal_state or ""
        data["terminal_state_seconds"] = terminal_state_seconds
        data["startup_completed"] = startup_completed
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
            data["final_decoded_console"] = log_collection.get("container_logs_truncated") or data.get("final_decoded_console", "")
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
            elif log_certification == "FAIL" or log_collection.get("failure_markers"):
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
                    debug_probe_monitor=args.debug_probe_monitor,
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
                "probe_state": monitor_result.get("probe_state"),
                "report_found": monitor_result.get("report_found"),
                "terminal_state_seen": monitor_result.get("terminal_state_seen"),
            }
        )

        fp8_summary = summarize_fp8_report(data.get("fp8_runtime_report"))
        certification = str(fp8_summary.get("runtime_certification") or "").upper()
        if certification == "PASS":
            status = "succeeded"
        elif monitor_result.get("status") in {"runtime_certification_failed", "probe_failed"}:
            status = "failed_recovered_from_container_logs"
        elif monitor_result.get("report_found"):
            status = "probe_completed_certification_failed"
        elif monitor_result.get("terminal_state_seen"):
            status = "probe_terminal_without_report"
        else:
            status = "probe_timeout"
        data["original_status"] = status

        latest_detail_json = monitor_result.get("latest_detail_json")
        latest_status_fields = monitor_result.get("latest_status_fields") or []
        data["probe_seconds"] = monitor_result.get("probe_seconds")
        data["container_exit_code"] = monitor_result.get("container_exit_code")
        data["container_status"] = monitor_result.get("container_status") or ""
        data["container_exit_detected"] = data["container_exit_code"] is not None or monitor_result.get("terminal_state_seen") is True
        data["container_terminal_state_detected"] = monitor_result.get("terminal_state_seen") is True
        data["terminal_state"] = data["container_status"]
        data["terminal_state_seconds"] = monitor_result.get("terminal_state_seconds")
        data["instance_errors"] = [
            field for field in latest_status_fields if "error" in str(field.get("path") or field.get("key") or "").lower()
        ][:20]
        data["instance_warnings"] = [
            field for field in latest_status_fields if "warning" in str(field.get("path") or field.get("key") or "").lower()
        ][:20]

        with timer.phase("collect_container_logs"):
            log_collection = monitor_result.get("latest_log_collection") or collect_container_logs(base_url, api_key, instance_id, latest_detail_json)
        for key, value in log_collection.items():
            if key != "container_log_endpoint_attempts" and key != "instance_text_fields":
                data[key] = value
        data["final_decoded_console"] = log_collection.get("container_logs_truncated") or data.get("final_decoded_console", "")
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
        elif certification != "PASS" and (log_certification == "FAIL" or log_collection.get("failure_markers")):
            status = "failed_recovered_from_container_logs"
        elif status == "probe_timeout" and not log_collection.get("container_logs_retrieved"):
            status = "probe_timeout"
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
