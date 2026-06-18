import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_LATENTSYNC_SMOKE_INPUTS_0001"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = REPO_ROOT / "logs" / "r2_latentsync_smoke_inputs_0001_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"

REQUIRED_ENV_VARS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
)

OBJECT_KEYS = (
    "checkpoints/latentsync/latentsync_unet.pt",
    "checkpoints/latentsync/whisper/tiny.pt",
    "checkpoints/latentsync/vae/sd-vae-ft-mse/config.json",
    "checkpoints/latentsync/vae/sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
    "tests/runpod_latentsync_smoke_run_0001/input/video.mp4",
    "tests/runpod_latentsync_smoke_run_0001/input/audio.wav",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_host_only(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.netloc:
        return parsed.netloc
    return endpoint.split("/")[0]


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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def r2_client(config: dict):
    boto3 = import_boto3()
    return boto3.client(
        "s3",
        endpoint_url=config["endpoint"],
        aws_access_key_id=config["access_key_id"],
        aws_secret_access_key=config["secret_access_key"],
        region_name=config["region"],
    )


def head_object(client, bucket: str, key: str) -> dict:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
        return {
            "key": key,
            "exists": True,
            "check_status": "found",
            "size_bytes": int(response.get("ContentLength", 0)),
            "last_modified": response.get("LastModified").isoformat() if response.get("LastModified") else "",
            "etag_present": bool(response.get("ETag")),
            "error": "",
        }
    except Exception as exc:
        error_code = ""
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            error_code = str(response.get("Error", {}).get("Code", ""))
        missing_codes = {"404", "NoSuchKey", "NotFound"}
        is_missing = error_code in missing_codes
        return {
            "key": key,
            "exists": False,
            "check_status": "missing" if is_missing else "error",
            "size_bytes": 0,
            "last_modified": "",
            "etag_present": False,
            "error": error_code or type(exc).__name__,
        }


def build_log(config: dict, results: list[dict], status: str, error: str = "") -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "head_object_only",
        "no_upload": True,
        "no_delete": True,
        "no_runpod": True,
        "objects_checked": results,
        "all_exist": all(item.get("exists") for item in results) if results else False,
    }


def run() -> int:
    config: dict = {}
    results: list[dict] = []
    try:
        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        print(f"[{TEST_ID}] START checking {len(OBJECT_KEYS)} R2 object(s)")
        for key in OBJECT_KEYS:
            result = head_object(client, config["bucket"], key)
            results.append(result)
            status = str(result["check_status"]).upper()
            size = result["size_bytes"]
            print(f"[{TEST_ID}] {status} size={size} key={key}")

        all_exist = all(item["exists"] for item in results)
        any_check_error = any(item.get("check_status") == "error" for item in results)
        status = "succeeded" if all_exist else "check_failed" if any_check_error else "missing_objects"
        write_json(LOG_PATH, build_log(config, results, status))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if all_exist else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(config, results, "failed", message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
