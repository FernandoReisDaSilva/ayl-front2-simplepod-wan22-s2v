import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


TEST_ID = "TEMP_CREATE_SIMPLEPOD_TEMPLATE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_template_create_v1.json"

API_KEY_ENV = "SIMPLEPOD_API_KEY"
BASE_URL_ENV = "SIMPLEPOD_API_BASE_URL"
DEFAULT_BASE_URL = "https://api.simplepod.ai"
AUTH_HEADER = "X-AUTH-TOKEN"
CREATE_TEMPLATE_PATH = "/instances/templates"
CREATE_TEMPLATE_ENDPOINT = "POST /instances/templates"
DOCS_URL = "https://api.simplepod.ai/docs"

TEMPLATE_NAME = "ayl-wan22-s2v-fastapi-v1"
IMAGE_NAME = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1"
IMAGE_TAG = "0.1.0"
DOCKER_IMAGE = f"{IMAGE_NAME}:{IMAGE_TAG}"
DATACENTER = "EU-PL-01"
VOLUME_NAME = "ayl_models_wan22_s2v_v1"
MODELS_ROOT = "/mnt/ayl_models"
MODEL_DIR = "/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B"
PORT = 8000


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


def template_payload() -> dict:
    return {
        "name": TEMPLATE_NAME,
        "imageName": IMAGE_NAME,
        "categoryName": "ayl-wan22-s2v",
        "defaultTag": IMAGE_TAG,
        "diskSize": 32,
        "exposePorts": str(PORT),
        "startScript": "uvicorn app.main:app --host 0.0.0.0 --port 8000",
        "argOptions": "",
        "envVariables": [
            {"name": "SIMPLEPOD_MODELS_ROOT", "value": MODELS_ROOT},
            {"name": "WAN22_S2V_MODEL_DIR", "value": MODEL_DIR},
            {"name": "AYL_IMAGE_TAG", "value": IMAGE_TAG},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
        ],
        "notes": (
            "AYL Wan2.2 S2V FastAPI template. "
            "No inference, no model download, no secrets in template payload."
        ),
        "isPasswordProtected": False,
        "isRunSshServerOn": False,
        "isRunJupyterOn": False,
    }


def redacted_payload(payload: dict) -> dict:
    redacted = json.loads(json.dumps(payload))
    for item in redacted.get("envVariables", []):
        name = str(item.get("name", ""))
        if any(token in name.upper() for token in ("KEY", "SECRET", "TOKEN", "PASSWORD")):
            item["value"] = "<redacted>"
    return redacted


def response_summary(body: bytes, content_type: str) -> dict:
    summary = {
        "content_type": content_type,
        "body_bytes": len(body),
        "json_parse_status": "not_json",
    }
    if "json" not in content_type.lower():
        return summary
    try:
        parsed = json.loads(body.decode("utf-8"))
    except Exception:
        summary["json_parse_status"] = "failed"
        return summary
    summary["json_parse_status"] = "succeeded"
    if isinstance(parsed, dict):
        safe_keys = [
            "id",
            "hashId",
            "name",
            "imageName",
            "categoryName",
            "defaultTag",
            "status",
            "createdAt",
        ]
        summary["top_level_keys"] = sorted(str(key) for key in parsed.keys())
        summary["safe_fields"] = {key: parsed[key] for key in safe_keys if key in parsed}
    elif isinstance(parsed, list):
        summary["json_type"] = "list"
        summary["items_observed"] = len(parsed)
    else:
        summary["json_type"] = type(parsed).__name__
    return summary


def create_template(base_url: str, api_key: str, payload: dict, timeout_seconds: int) -> dict:
    url = urljoin(normalize_base_url(base_url), CREATE_TEMPLATE_PATH.lstrip("/"))
    request = Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            AUTH_HEADER: api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "ayl-front2-simplepod-template-create-v1",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(262_144)
            content_type = response.headers.get("Content-Type", "")
            return {
                "attempted": True,
                "status": "succeeded",
                "http_status_code": response.status,
                "endpoint_host": endpoint_host(url),
                "response_summary": response_summary(body, content_type),
            }
    except HTTPError as exc:
        body = exc.read(65_536)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        return {
            "attempted": True,
            "status": "failed",
            "http_status_code": exc.code,
            "endpoint_host": endpoint_host(url),
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:1000],
            "response_summary": response_summary(body, content_type),
        }
    except URLError as exc:
        return {
            "attempted": True,
            "status": "failed",
            "endpoint_host": endpoint_host(url),
            "error_type": "URLError",
            "error_truncated": str(exc)[:1000],
        }


def build_report(args: argparse.Namespace, payload: dict, create_result: dict, status: str, error: str = "") -> dict:
    base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL)
    api_key_present = bool(os.getenv(API_KEY_ENV, ""))
    execute_allowed = args.execute and args.confirm_create
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "docs_url": DOCS_URL,
        "endpoint": CREATE_TEMPLATE_ENDPOINT,
        "auth_method": f"apiKey header {AUTH_HEADER}",
        "api_key_present": api_key_present if args.execute else False,
        "base_url_host": endpoint_host(base_url) if args.execute else "",
        "execute_requested": args.execute,
        "confirm_create": args.confirm_create,
        "execute_allowed": execute_allowed,
        "dry_run": not execute_allowed,
        "docker_image": DOCKER_IMAGE,
        "payload_redacted": redacted_payload(payload),
        "planned_runtime_context_not_in_template_post_body": {
            "datacenter": DATACENTER,
            "volume_name": VOLUME_NAME,
            "mount_path": MODELS_ROOT,
            "wan22_s2v_model_dir": MODEL_DIR,
            "port": PORT,
        },
        "create_result": create_result,
        "safety_guards": {
            "template_create_endpoint_called": bool(create_result.get("attempted")),
            "instance_started": False,
            "pod_created": False,
            "model_weights_downloaded": False,
            "inference_executed": False,
            "secrets_printed": False,
        },
    }


def run(args: argparse.Namespace) -> int:
    payload = template_payload()
    create_result = {"attempted": False, "status": "not_attempted"}
    execute_allowed = args.execute and args.confirm_create
    try:
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()}")
        print(f"[{TEST_ID}] endpoint={CREATE_TEMPLATE_ENDPOINT}")
        print(f"[{TEST_ID}] template_name={TEMPLATE_NAME} image={DOCKER_IMAGE}")

        if args.execute and not args.confirm_create:
            status = "blocked_missing_confirm_create"
            report = build_report(args, payload, create_result, status)
            write_json(REPORT_PATH, report)
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        if not execute_allowed:
            status = "dry_run_ready"
            report = build_report(args, payload, create_result, status)
            write_json(REPORT_PATH, report)
            print(f"[{TEST_ID}] template_create_endpoint_called=false")
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 0

        load_repo_dotenv()
        api_key = os.getenv(API_KEY_ENV, "")
        base_url = os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL)
        if not api_key:
            status = "missing_api_key"
            report = build_report(args, payload, create_result, status)
            write_json(REPORT_PATH, report)
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1
        if not base_url:
            status = "missing_base_url"
            report = build_report(args, payload, create_result, status)
            write_json(REPORT_PATH, report)
            print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
            return 1

        create_result = create_template(base_url, api_key, payload, args.timeout_seconds)
        status = "succeeded" if create_result.get("status") == "succeeded" else "failed"
        report = build_report(args, payload, create_result, status)
        write_json(REPORT_PATH, report)
        print(f"[{TEST_ID}] template_create_endpoint_called=true")
        print(f"[{TEST_ID}] DONE status={status} report={REPORT_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        report = build_report(args, payload, create_result, "failed", message)
        write_json(REPORT_PATH, report)
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed report={REPORT_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or create the private SimplePod template for AYL Wan2.2 S2V FastAPI.")
    parser.add_argument("--execute", action="store_true", help="Call POST /instances/templates.")
    parser.add_argument("--confirm-create", action="store_true", help="Required with --execute before creating the template.")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
