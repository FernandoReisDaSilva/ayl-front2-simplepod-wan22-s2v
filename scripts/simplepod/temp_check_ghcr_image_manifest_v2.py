import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


TEST_ID = "TEMP_CHECK_GHCR_IMAGE_MANIFEST_V2"
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO_ROOT / "logs" / "simplepod_ghcr_image_manifest_v2.json"

IMAGE_NAME = "ghcr.io/fernandoreisdasilva/ayl-simplepod-wan22-s2v-fastapi-v2"
IMAGE_TAG = "0.1.4"
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


def image_scope(image_name: str) -> str:
    parsed = urlparse("https://" + image_name)
    return f"repository:{parsed.path.lstrip('/')}:pull"


def parse_www_authenticate(value: str) -> dict:
    if not value:
        return {"ok": False, "error": "missing_www_authenticate_header"}
    scheme, _, rest = value.partition(" ")
    if scheme.lower() != "bearer" or not rest:
        return {"ok": False, "error": "unsupported_auth_scheme", "scheme": scheme}

    params = {}
    current = ""
    in_quotes = False
    for char in rest:
        if char == '"':
            in_quotes = not in_quotes
        if char == "," and not in_quotes:
            key, sep, raw_value = current.partition("=")
            if sep:
                params[key.strip()] = raw_value.strip().strip('"')
            current = ""
        else:
            current += char
    if current:
        key, sep, raw_value = current.partition("=")
        if sep:
            params[key.strip()] = raw_value.strip().strip('"')

    realm = params.get("realm", "")
    service = params.get("service", "")
    scope = params.get("scope", "")
    ok = bool(realm and service)
    return {
        "ok": ok,
        "scheme": scheme,
        "realm": realm,
        "service": service,
        "scope": scope,
        "error": "" if ok else "missing_realm_or_service",
    }


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


def manifest_request(url: str, token: str = "") -> dict:
    headers = {
        "Accept": MANIFEST_ACCEPT,
        "User-Agent": "ayl-front2-ghcr-manifest-check-v2",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        method="GET",
        headers=headers,
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
        www_authenticate = exc.headers.get("WWW-Authenticate", "") if exc.headers else ""
        status = "http_error"
        if exc.code == 404:
            status = "image_tag_not_found"
        elif exc.code in {401, 403}:
            status = "auth_required"
        return {
            "attempted": True,
            "status": status,
            "http_status_code": exc.code,
            "manifest_url": url,
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:500],
            "www_authenticate_present": bool(www_authenticate),
            "www_authenticate_scheme": www_authenticate.split(" ", 1)[0] if www_authenticate else "",
            "auth_challenge": parse_www_authenticate(www_authenticate),
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


def request_anonymous_token(challenge: dict, fallback_scope: str) -> dict:
    if not challenge.get("ok"):
        return {
            "attempted": False,
            "status": "challenge_parse_failed",
            "error": challenge.get("error", ""),
        }

    scope = challenge.get("scope") or fallback_scope
    query = {
        "service": challenge["service"],
        "scope": scope,
    }
    token_url = challenge["realm"] + "?" + urlencode(query, quote_via=quote)
    request = Request(
        token_url,
        method="GET",
        headers={
            "Accept": "application/json",
            "User-Agent": "ayl-front2-ghcr-manifest-check-v2",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            body = response.read(65_536)
            parsed = json.loads(body.decode("utf-8"))
            token = parsed.get("token") or parsed.get("access_token") or ""
            return {
                "attempted": True,
                "status": "succeeded" if token else "missing_token_in_response",
                "http_status_code": response.status,
                "token_url_host": urlparse(token_url).netloc,
                "service": challenge["service"],
                "scope": scope,
                "token_present": bool(token),
                "token": token,
            }
    except HTTPError as exc:
        exc.read(8192)
        return {
            "attempted": True,
            "status": "failed",
            "http_status_code": exc.code,
            "token_url_host": urlparse(token_url).netloc,
            "service": challenge["service"],
            "scope": scope,
            "error_type": "HTTPError",
            "error_truncated": str(exc)[:500],
            "token_present": False,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "status": "failed",
            "token_url_host": urlparse(token_url).netloc,
            "service": challenge["service"],
            "scope": scope,
            "error_type": type(exc).__name__,
            "error_truncated": str(exc)[:500],
            "token_present": False,
        }


def public_token_summary(token_result: dict) -> dict:
    return {key: value for key, value in token_result.items() if key != "token"}


def final_status(initial: dict, token_result: dict, retry: dict) -> str:
    if initial.get("status") == "image_tag_found":
        return "image_tag_found"
    if initial.get("status") == "image_tag_not_found":
        return "image_tag_not_found"
    if initial.get("status") != "auth_required":
        return initial.get("status", "http_error")
    challenge = initial.get("auth_challenge") or {}
    if not challenge.get("ok"):
        return "ghcr_auth_challenge_parse_failed"
    if token_result.get("status") != "succeeded":
        return "ghcr_token_request_failed"
    if retry.get("status") == "image_tag_found":
        return "image_tag_found"
    if retry.get("status") == "image_tag_not_found":
        return "image_tag_not_found"
    if retry.get("http_status_code") in {401, 403}:
        return "image_tag_private_or_auth_required"
    return retry.get("status", "http_error")


def check_manifest() -> dict:
    url = manifest_url(IMAGE_NAME, IMAGE_TAG)
    initial = manifest_request(url)
    token_result = {"attempted": False, "status": "not_needed"}
    retry = {"attempted": False, "status": "not_needed"}

    if initial.get("status") == "auth_required":
        challenge = initial.get("auth_challenge") or {}
        token_result = request_anonymous_token(challenge, image_scope(IMAGE_NAME))
        token = token_result.get("token", "")
        if token:
            retry = manifest_request(url, token)

    status = final_status(initial, token_result, retry)
    return {
        "attempted": True,
        "status": status,
        "manifest_url": url,
        "initial_manifest_request": initial,
        "anonymous_token_request": public_token_summary(token_result),
        "anonymous_token_manifest_request": retry,
        "manifest_found": status == "image_tag_found",
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
    print(
        f"[{TEST_ID}] initial_http_status="
        f"{result.get('initial_manifest_request', {}).get('http_status_code', '')}"
    )
    print(
        f"[{TEST_ID}] anonymous_http_status="
        f"{result.get('anonymous_token_manifest_request', {}).get('http_status_code', '')}"
    )
    print(f"[{TEST_ID}] manifest_found={str(result.get('manifest_found', False)).lower()}")
    print(f"[{TEST_ID}] downloads_image_layers=false")
    print(f"[{TEST_ID}] report={REPORT_PATH}")
    return 1 if status == "network_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
