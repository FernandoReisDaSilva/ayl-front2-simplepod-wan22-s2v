import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_WAN22_S2V_PROMPT_PAYLOAD_DEBUG_DOWNLOAD_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
PRIMARY_DESTINATION = REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_local_debug_v1.json"
COMPAT_DESTINATION = REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_debug_v1.json"
LOG_PATH = REPO_ROOT / "logs" / "wan22_s2v_prompt_payload_debug_download_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"
PROMPT_DEBUG_KEY = "tests/runpod_wan22_s2v_probe_v1/debug/prompt_payload_debug.json"
REQUIRED_ENV_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_host_only(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint.split("/")[0]


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'python-dotenv'. Install it with: {DEPENDENCY_NOTE}") from exc
    if not ENV_PATH.exists():
        raise RuntimeError(f"Repo .env file not found: {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH, override=False)


def import_boto3():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'boto3'. Install it with: {DEPENDENCY_NOTE}") from exc
    return boto3


def env_config() -> dict:
    missing = [key for key in REQUIRED_ENV_VARS if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing required .env variable(s): {', '.join(missing)}")
    return {
        "endpoint": os.environ["R2_ENDPOINT"],
        "access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "bucket": os.environ["R2_BUCKET"],
        "region": os.environ["R2_REGION"],
    }


def r2_client(config: dict):
    boto3 = import_boto3()
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key_id"],
        aws_secret_access_key=config["secret_access_key"],
        region_name=config["region"],
    )


def missing_object_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0)
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def head_object(client, bucket: str, key: str) -> dict:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
        return {
            "status": "found",
            "exists": True,
            "size_bytes": int(response.get("ContentLength", 0)),
            "etag_present": bool(response.get("ETag")),
            "error": "",
        }
    except Exception as exc:
        if missing_object_error(exc):
            return {"status": "missing", "exists": False, "size_bytes": 0, "etag_present": False, "error": ""}
        return {
            "status": "error",
            "exists": False,
            "size_bytes": 0,
            "etag_present": False,
            "error": str(exc)[:1000],
        }


def build_log(config: dict, status: str, *, dry_run: bool, head: dict | None = None, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "r2_key": PROMPT_DEBUG_KEY,
        "primary_destination": str(PRIMARY_DESTINATION),
        "compat_destination": str(COMPAT_DESTINATION),
        "downloaded_size_bytes": PRIMARY_DESTINATION.stat().st_size if PRIMARY_DESTINATION.exists() else 0,
        "r2_head": head or {},
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "dry_run_plan_only" if dry_run else "head_then_download_file",
        "no_upload": True,
        "no_delete": True,
        "no_runpod": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
    }


def run(args: argparse.Namespace) -> int:
    config = {}
    dry_run = not args.execute
    try:
        if dry_run:
            print(f"[{TEST_ID}] START dry_run=true")
            print(f"[{TEST_ID}] PLANNED key={PROMPT_DEBUG_KEY}")
            print(f"[{TEST_ID}] PLANNED primary_destination={PRIMARY_DESTINATION}")
            print(f"[{TEST_ID}] PLANNED compat_destination={COMPAT_DESTINATION}")
            print(f"[{TEST_ID}] PASS --execute to download")
            write_json(LOG_PATH, build_log(config, "dry_run_ready", dry_run=True))
            print(f"[{TEST_ID}] DONE status=dry_run_ready log={LOG_PATH}")
            return 0

        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        print(f"[{TEST_ID}] START execute=true key={PROMPT_DEBUG_KEY}")
        head = head_object(client, config["bucket"], PROMPT_DEBUG_KEY)
        if head["status"] == "missing":
            write_json(LOG_PATH, build_log(config, "missing", dry_run=False, head=head))
            print(f"[{TEST_ID}] DONE status=missing key={PROMPT_DEBUG_KEY} destination={PRIMARY_DESTINATION} size=0 log={LOG_PATH}")
            return 0
        if head["status"] == "error":
            raise RuntimeError(head["error"] or "head_object failed")

        PRIMARY_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(config["bucket"], PROMPT_DEBUG_KEY, str(PRIMARY_DESTINATION))
        shutil.copyfile(PRIMARY_DESTINATION, COMPAT_DESTINATION)
        size_bytes = PRIMARY_DESTINATION.stat().st_size
        write_json(LOG_PATH, build_log(config, "succeeded", dry_run=False, head=head))
        print(f"[{TEST_ID}] DONE status=succeeded key={PROMPT_DEBUG_KEY} destination={PRIMARY_DESTINATION} size={size_bytes} compat_destination={COMPAT_DESTINATION} log={LOG_PATH}")
        return 0
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(config, "failed", dry_run=dry_run, error=message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Wan2.2 S2V prompt payload debug JSON from R2.")
    parser.add_argument("--execute", action="store_true", help="Download the real R2 prompt payload debug JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
