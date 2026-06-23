import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_WAN27_PROBE_INPUTS_UPLOAD_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = REPO_ROOT / "logs" / "r2_wan27_probe_inputs_upload_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"

REQUIRED_ENV_VARS = (
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
)

UPLOAD_TARGETS = (
    ("reference_image", "reference_image_local", "tests/runpod_wan27_probe_v1/input/mae_reference.png"),
    ("audio", "audio_local", "tests/runpod_wan27_probe_v1/input/audio.wav"),
    ("video", "video_local", "tests/runpod_wan27_probe_v1/input/mae_5s.mp4"),
)

TARGET_GROUPS = {
    "reference": {"reference_image"},
    "audio": {"audio"},
    "video": {"video"},
    "inputs": {"reference_image", "audio", "video"},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_host_only(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint.split("/")[0]


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        response = getattr(exc, "response", None)
        error_code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
        return {
            "key": key,
            "exists": False,
            "check_status": "missing" if error_code in {"404", "NoSuchKey", "NotFound"} else "error",
            "size_bytes": 0,
            "last_modified": "",
            "etag_present": False,
            "error": error_code or type(exc).__name__,
        }


def local_path_from_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def selected_scope(args: argparse.Namespace) -> str:
    selected = [
        name
        for name, enabled in (
            ("reference", args.only_reference),
            ("audio", args.only_audio),
            ("video", args.only_video),
            ("inputs", args.only_inputs),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise RuntimeError("Use only one partial upload selector at a time.")
    return selected[0] if selected else "inputs"


def selected_targets(args: argparse.Namespace) -> tuple[str, tuple[tuple[str, str, str], ...]]:
    scope = selected_scope(args)
    names = TARGET_GROUPS[scope]
    return scope, tuple(target for target in UPLOAD_TARGETS if target[0] in names)


def build_upload_plan(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    items: list[dict] = []
    problems: list[str] = []
    _, targets = selected_targets(args)
    for name, arg_name, r2_key in targets:
        raw_value = getattr(args, arg_name)
        path = local_path_from_arg(raw_value) if raw_value else None
        exists = bool(path and path.exists())
        is_file = bool(path and path.is_file())
        size_bytes = path.stat().st_size if is_file else 0
        sha256 = sha256_file(path) if is_file else ""
        if not raw_value:
            problems.append(f"missing --{arg_name.replace('_', '-')}")
        elif not exists:
            problems.append(f"local file not found for --{arg_name.replace('_', '-')}: {path}")
        elif not is_file:
            problems.append(f"local path is not a file for --{arg_name.replace('_', '-')}: {path}")
        items.append(
            {
                "name": name,
                "arg_name": arg_name,
                "local_path": str(path) if path else "",
                "local_exists": exists,
                "local_is_file": is_file,
                "local_size_bytes": size_bytes,
                "local_sha256": sha256,
                "r2_key": r2_key,
                "upload_attempted": False,
                "upload_status": "not_attempted",
                "remote_before": {},
                "remote_after": {},
            }
        )
    return items, problems


def build_log(*, args: argparse.Namespace, config: dict, items: list[dict], status: str, problems: list[str], error: str = "") -> dict:
    execute_allowed = args.execute and args.confirm_upload
    try:
        scope, _ = selected_targets(args)
    except RuntimeError:
        scope = "invalid_multiple_selectors"
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "problems": problems,
        "execute_requested": args.execute,
        "confirm_upload": args.confirm_upload,
        "execute_allowed": execute_allowed,
        "overwrite": args.overwrite,
        "upload_scope": scope,
        "dry_run": not execute_allowed,
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "upload_file_then_head_object" if execute_allowed else "dry_run_plan_only",
        "no_runpod": True,
        "no_delete": True,
        "not_latentsync": True,
        "upload_targets": items,
    }


def run(args: argparse.Namespace) -> int:
    config: dict = {}
    items: list[dict] = []
    problems: list[str] = []
    execute_allowed = args.execute and args.confirm_upload
    try:
        scope, _ = selected_targets(args)
        items, problems = build_upload_plan(args)
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} scope={scope} targets={len(items)}")
        for item in items:
            present = "ok" if item["local_is_file"] else "missing"
            print(f"[{TEST_ID}] LOCAL {present} size={item['local_size_bytes']} key={item['r2_key']}")

        if problems:
            status = "missing_local_inputs"
            write_json(LOG_PATH, build_log(args=args, config=config, items=items, status=status, problems=problems))
            print(f"[{TEST_ID}] MISSING_ARGS_OR_FILES count={len(problems)}")
            for problem in problems:
                print(f"[{TEST_ID}] {problem}")
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 1

        if not execute_allowed:
            status = "dry_run_ready"
            write_json(LOG_PATH, build_log(args=args, config=config, items=items, status=status, problems=[]))
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        upload_errors: list[str] = []
        for item in items:
            remote_before = head_object(client, config["bucket"], item["r2_key"])
            item["remote_before"] = remote_before
            if remote_before["exists"] and not args.overwrite:
                item["upload_status"] = "blocked_existing_remote_without_overwrite"
                upload_errors.append(f"remote object already exists: {item['r2_key']}")
                continue
            if remote_before["check_status"] == "error":
                item["upload_status"] = "blocked_remote_head_error"
                upload_errors.append(f"remote HEAD error for {item['r2_key']}: {remote_before['error']}")
                continue
            item["upload_attempted"] = True
            client.upload_file(item["local_path"], config["bucket"], item["r2_key"])
            remote_after = head_object(client, config["bucket"], item["r2_key"])
            item["remote_after"] = remote_after
            item["upload_status"] = "uploaded_and_head_ok" if remote_after["exists"] else "uploaded_but_head_missing"
            if not remote_after["exists"]:
                upload_errors.append(f"remote HEAD after upload failed for {item['r2_key']}")

        status = "succeeded" if not upload_errors else "failed"
        write_json(LOG_PATH, build_log(args=args, config=config, items=items, status=status, problems=upload_errors))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(args=args, config=config, items=items, status="failed", problems=problems, error=message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or upload required WAN 2.7 probe inputs to R2.")
    parser.add_argument("--execute", action="store_true", help="Perform real R2 uploads.")
    parser.add_argument("--confirm-upload", action="store_true", help="Required with --execute for real uploads.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing R2 objects.")
    parser.add_argument("--only-reference", action="store_true", help="Upload only mae_reference.png.")
    parser.add_argument("--only-audio", action="store_true", help="Upload only audio.wav.")
    parser.add_argument("--only-video", action="store_true", help="Upload only mae_5s.mp4.")
    parser.add_argument("--only-inputs", action="store_true", help="Upload reference image, audio, and source video.")
    parser.add_argument("--reference-image-local", default="data/wan27/inputs/mae_reference.png", help="Local Mae reference PNG path.")
    parser.add_argument("--audio-local", default="~/Downloads/mae_audio_5s.wav", help="Local Mae audio WAV path.")
    parser.add_argument("--video-local", default="~/Downloads/mae_5s.mp4", help="Local Mae source video MP4 path.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
