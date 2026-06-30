import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from .settings import SERVICE_NAME, SERVICE_VERSION, env_presence, get_settings


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_facts(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir() if exists else False,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
    }


def base_report(mode: str, status: str) -> dict:
    settings = get_settings()
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "mode": mode,
        "status": status,
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "image_tag": settings.image_tag,
        "marker_nonce": settings.marker_nonce,
        "env_present_redacted": env_presence(),
        "no_inference": True,
        "no_model_downloads": True,
        "not_runpod": True,
        "not_latentsync": True,
        "not_wan27": True,
    }


def stub_final_report(job_id: str, request_redacted: dict) -> dict:
    return {
        **base_report("wan22_s2v_stub", "stub_created"),
        "job_id": job_id,
        "received": True,
        "request_redacted": request_redacted,
        "output_video": None,
        "manual_cleanup_required": False,
    }
