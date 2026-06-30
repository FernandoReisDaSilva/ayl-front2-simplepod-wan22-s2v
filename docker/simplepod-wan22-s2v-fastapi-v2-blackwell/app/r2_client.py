import os
from pathlib import Path

from .settings import R2_ENV_KEYS


def _first_env(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key, "")
        if value:
            return value
    return default


def resolved_r2_env() -> dict:
    return {
        "endpoint": _first_env("R2_ENDPOINT", "R2_ENDPOINT_URL"),
        "access_key_id": _first_env("R2_ACCESS_KEY_ID"),
        "secret_access_key": _first_env("R2_SECRET_ACCESS_KEY"),
        "bucket": _first_env("R2_BUCKET", "R2_BUCKET_NAME"),
        "region": _first_env("R2_REGION", default="auto"),
    }


def r2_env_presence() -> dict:
    return {key: bool(os.getenv(key, "")) for key in R2_ENV_KEYS}


def r2_env_alias_presence() -> dict:
    resolved = resolved_r2_env()
    return {
        "endpoint": bool(resolved["endpoint"]),
        "access_key_id": bool(resolved["access_key_id"]),
        "secret_access_key": bool(resolved["secret_access_key"]),
        "bucket": bool(resolved["bucket"]),
        "region": bool(resolved["region"]),
        "accepted_aliases": {
            "endpoint": ["R2_ENDPOINT", "R2_ENDPOINT_URL"],
            "bucket": ["R2_BUCKET", "R2_BUCKET_NAME"],
            "access_key_id": ["R2_ACCESS_KEY_ID"],
            "secret_access_key": ["R2_SECRET_ACCESS_KEY"],
            "region": ["R2_REGION"],
        },
    }


def r2_env_ready() -> bool:
    resolved = resolved_r2_env()
    return all(resolved[key] for key in ("endpoint", "access_key_id", "secret_access_key", "bucket", "region"))


def missing_r2_env() -> list[str]:
    resolved = resolved_r2_env()
    missing = []
    if not resolved["endpoint"]:
        missing.append("R2_ENDPOINT or R2_ENDPOINT_URL")
    if not resolved["access_key_id"]:
        missing.append("R2_ACCESS_KEY_ID")
    if not resolved["secret_access_key"]:
        missing.append("R2_SECRET_ACCESS_KEY")
    if not resolved["bucket"]:
        missing.append("R2_BUCKET or R2_BUCKET_NAME")
    return missing


def get_r2_client():
    if not r2_env_ready():
        raise RuntimeError("Missing required R2 env var(s): " + ", ".join(missing_r2_env()))

    resolved = resolved_r2_env()
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=resolved["endpoint"],
        aws_access_key_id=resolved["access_key_id"],
        aws_secret_access_key=resolved["secret_access_key"],
        region_name=resolved["region"],
    )


def head_object(key: str) -> dict:
    resolved = resolved_r2_env()
    response = get_r2_client().head_object(Bucket=resolved["bucket"], Key=key)
    return {
        "status": "succeeded",
        "key": key,
        "content_length": response.get("ContentLength"),
        "content_type": response.get("ContentType", ""),
        "etag_present": bool(response.get("ETag")),
    }


def upload_file(source: Path, key: str) -> None:
    get_r2_client().upload_file(str(source), resolved_r2_env()["bucket"], key)


def download_file(key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    get_r2_client().download_file(resolved_r2_env()["bucket"], key, str(destination))
