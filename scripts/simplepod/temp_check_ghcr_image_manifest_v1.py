import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


TEST_ID = "TEMP_CHECK_GHCR_IMAGE_MANIFEST_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_ghcr_image_manifest_v1.json"

IMAGE_NAME = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v1"
IMAGE_TAG = "0.1.0"
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
MAX_BODY_BYTES = 262_144


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def manifest_url(image_name: str, tag: str) -> str:
    parsed = urlparse("https://" + image_name)
    return f"https://{parsed.netloc}/v2{parsed.path}/manifests/{tag}"


def summarize_manifest(body: bytes, content_type: str) -> dict:
    summary = {
        "content_type": content_type,
        "body_bytes_read": len(body),
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
        summary["mediaType"] = parsed.get("mediaType", "")
        summary["schemaVersion"] = parsed.get("schemaVersion")
        manifests = parsed.get("manifests")
        layers = parsed.get("layers")
        if isinstance(manifests, list):
            summary["manifest_count"] = len(manifests)
            summary["platforms"] = [
                item.get("platform", {})
                for item in manifests[:10]
                if isinstance(item, dict)
            ]
        if isinstance(layers, list):
            summary["layer_count"] = len(layers)
        config = parsed.get("config")
        if isinstance(config, dict):
            summary["config_mediaType"] = config.get("mediaType", "")
            summary["config_size"] = config.get("size")
    return summary


def check_manifest() -> dict:
    url = manifest_url(IMAGE_NAME, IMAGE_TAG)
    request = Request(
        url,
        method="GET",
        headers={
            "Accept": MANIFEST_ACCEPT,
            "User-Agent": "ayl-front2-ghcr-manifest-check-v1",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read(MAX_BODY_BYTES)
            content_type = response.headers.get("Content-Type", "")
            digest = response.headers.get("Docker-Content-Digest", "")
            return {
                "attempted": True,
                "status": "image_tag_found",
                "http_status_code": response.status,
                "manifest_url": url,
                "docker_content_digest": digest,
                "manifest_summary": summarize_manifest(body, content_type),
            }
    except HTTPError as exc:
        body = exc.read(8192)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        status = "http_error"
        if exc.code == 404:
            status = "image_tag_not_found"
        elif exc.code in {401, 403}:
            status = "image_tag_requires_auth_or_package_private"
        return {
            "attempted": True,
            "status": status,
            "http_status_code": exc.code,
            "manifest_url": url,
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:500],
            "manifest_summary": summarize_manifest(body, content_type),
        }
    except URLError as exc:
        return {
            "attempted": True,
            "status": "network_error",
            "manifest_url": url,
            "error_type": "URLError",
            "error_truncated": str(exc)[:500],
        }


def build_report(result: dict) -> dict:
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "image": IMAGE_NAME,
        "tag": IMAGE_TAG,
        "image_ref": f"{IMAGE_NAME}:{IMAGE_TAG}",
        "check_type": "ghcr_registry_manifest_metadata_only",
        "downloads_image_layers": False,
        "uses_secrets": False,
        "calls_simplepod": False,
        "downloads_model_weights": False,
        "runs_inference": False,
        "result": result,
    }


def main() -> int:
    result = check_manifest()
    report = build_report(result)
    write_json(REPORT_PATH, report)

    status = result.get("status", "unknown")
    print(f"[{TEST_ID}] image={IMAGE_NAME}:{IMAGE_TAG}")
    print(f"[{TEST_ID}] status={status}")
    print(f"[{TEST_ID}] downloads_image_layers=false")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 1 if status == "network_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
