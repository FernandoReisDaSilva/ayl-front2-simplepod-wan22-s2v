import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_WAN22_S2V_PROBE_INPUTS_CHECK_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = REPO_ROOT / "logs" / "r2_wan22_s2v_probe_inputs_check_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"
REQUIRED_ENV_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")
OBJECT_KEYS = (
    "tests/runpod_wan22_s2v_probe_v1/input/mae_reference.png",
    "tests/runpod_wan22_s2v_probe_v1/input/mae_audio_5s.wav",
)


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
    return boto3.client("s3", endpoint_url=config["endpoint"], aws_access_key_id=config["access_key_id"], aws_secret_access_key=config["secret_access_key"], region_name=config["region"])


def head_object(client, bucket: str, key: str) -> dict:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
        return {"key": key, "exists": True, "check_status": "found", "size_bytes": int(response.get("ContentLength", 0)), "error": ""}
    except Exception as exc:
        response = getattr(exc, "response", None)
        error_code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        return {"key": key, "exists": False, "check_status": "missing" if error_code in {"404", "NoSuchKey", "NotFound"} else "error", "size_bytes": 0, "error": error_code or type(exc).__name__}


def build_log(config: dict, results: list[dict], status: str, *, dry_run: bool, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "dry_run_key_list_only" if dry_run else "head_object_only",
        "no_upload": True,
        "no_delete": True,
        "no_runpod": True,
        "not_latentsync": True,
        "not_wan27": True,
        "objects_checked": results,
        "all_exist": all(item.get("exists") for item in results) if results else False,
    }


def run(args: argparse.Namespace) -> int:
    config = {}
    results = []
    dry_run = not args.execute
    try:
        if dry_run:
            print(f"[{TEST_ID}] START dry_run=true planned_keys={len(OBJECT_KEYS)}")
            for key in OBJECT_KEYS:
                results.append({"key": key, "exists": False, "check_status": "planned", "size_bytes": 0, "error": ""})
                print(f"[{TEST_ID}] PLANNED key={key}")
            write_json(LOG_PATH, build_log(config, results, "dry_run_ready", dry_run=True))
            print(f"[{TEST_ID}] DONE status=dry_run_ready log={LOG_PATH}")
            return 0
        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        print(f"[{TEST_ID}] START execute=true checking={len(OBJECT_KEYS)}")
        for key in OBJECT_KEYS:
            result = head_object(client, config["bucket"], key)
            results.append(result)
            print(f"[{TEST_ID}] {str(result['check_status']).upper()} size={result['size_bytes']} key={key}")
        all_exist = all(item["exists"] for item in results)
        status = "succeeded" if all_exist else "missing_objects"
        write_json(LOG_PATH, build_log(config, results, status, dry_run=False))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if all_exist else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(config, results, "failed", dry_run=dry_run, error=message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or HEAD-check required Wan2.2 S2V probe inputs in R2.")
    parser.add_argument("--execute", action="store_true", help="Perform real R2 HEAD checks.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
