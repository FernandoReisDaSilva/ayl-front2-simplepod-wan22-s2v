import os
from pathlib import Path

from .settings import R2_ENV_KEYS


def r2_env_presence() -> dict:
    return {key: bool(os.getenv(key, "")) for key in R2_ENV_KEYS}


def r2_env_ready() -> bool:
    return all(os.getenv(key, "") for key in R2_ENV_KEYS)


def missing_r2_env() -> list[str]:
    return [key for key in R2_ENV_KEYS if not os.getenv(key, "")]


def get_r2_client():
    if not r2_env_ready():
        raise RuntimeError("Missing required R2 env var(s): " + ", ".join(missing_r2_env()))

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.environ["R2_REGION"],
    )


def upload_file(source: Path, key: str) -> None:
    get_r2_client().upload_file(str(source), os.environ["R2_BUCKET"], key)


def download_file(key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    get_r2_client().download_file(os.environ["R2_BUCKET"], key, str(destination))
