import os
from dataclasses import dataclass
from pathlib import Path


SERVICE_NAME = "ayl-simplepod-wan22-s2v-fastapi-v2"
SERVICE_VERSION = "0.2.3-blackwell"

R2_ENV_KEYS = (
    "R2_ENDPOINT",
    "R2_ENDPOINT_URL",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_BUCKET_NAME",
    "R2_REGION",
)

APP_ENV_KEYS = (
    "SIMPLEPOD_MODELS_ROOT",
    "WAN22_S2V_MODEL_DIR",
    "AYL_IMAGE_TAG",
    "AYL_MARKER_NONCE",
    "AYL_ENABLE_ADMIN_DOWNLOADS",
    "AYL_ENABLE_ADMIN_VERIFY",
    "HF_HOME",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    *R2_ENV_KEYS,
)

SECRET_TOKENS = (
    "SECRET",
    "TOKEN",
    "KEY",
    "PASSWORD",
    "AUTH",
    "CREDENTIAL",
)


@dataclass(frozen=True)
class Settings:
    simplepod_models_root: Path
    wan22_s2v_model_dir: Path
    image_tag: str
    marker_nonce: str


def get_settings() -> Settings:
    models_root = Path(os.getenv("SIMPLEPOD_MODELS_ROOT", "/mnt/ayl-models"))
    model_dir = Path(os.getenv("WAN22_S2V_MODEL_DIR", str(models_root / "wan22_s2v")))
    return Settings(
        simplepod_models_root=models_root,
        wan22_s2v_model_dir=model_dir,
        image_tag=os.getenv("AYL_IMAGE_TAG", ""),
        marker_nonce=os.getenv("AYL_MARKER_NONCE", ""),
    )


def is_secret_key(key: str) -> bool:
    upper_key = key.upper()
    return any(token in upper_key for token in SECRET_TOKENS)


def redact_env_value(key: str, value: str) -> str:
    if not value:
        return ""
    if is_secret_key(key):
        return "<redacted>"
    return value


def env_presence() -> dict:
    return {key: bool(os.getenv(key, "")) for key in APP_ENV_KEYS}


def env_redacted() -> dict:
    return {key: redact_env_value(key, os.getenv(key, "")) for key in APP_ENV_KEYS}
