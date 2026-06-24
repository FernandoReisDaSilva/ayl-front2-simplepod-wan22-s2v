import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_WAN22_S2V_MODEL_WEIGHTS_UPLOAD_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = REPO_ROOT / "logs" / "r2_wan22_s2v_model_weights_upload_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"
R2_PREFIX = "checkpoints/wan22_s2v/comfyui_models/"
LOCAL_ROOT = REPO_ROOT / "data" / "checkpoints" / "wan22_s2v" / "comfyui_models"

REQUIRED_ENV_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")
UPLOAD_TARGETS = (
    {
        "name": "transformer",
        "arg_name": "transformer_local",
        "comfy_models_relative_path": "diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
    },
    {
        "name": "vae",
        "arg_name": "vae_local",
        "comfy_models_relative_path": "vae/wanvideo/Wan2_1_VAE_bf16.safetensors",
    },
    {
        "name": "umt5",
        "arg_name": "umt5_local",
        "comfy_models_relative_path": "text_encoders/umt5-xxl-enc-bf16.safetensors",
    },
    {
        "name": "wav2vec",
        "arg_name": "wav2vec_local",
        "comfy_models_relative_path": "audio_encoders/wav2vec_xlsr_53_english_fp32.safetensors",
    },
)
TARGET_GROUPS = {
    "transformer": {"transformer"},
    "vae": {"vae"},
    "umt5": {"umt5"},
    "wav2vec": {"wav2vec"},
    "all": {"transformer", "vae", "umt5", "wav2vec"},
}


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


def head_object(client, bucket: str, key: str) -> dict:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
        return {
            "key": key,
            "exists": True,
            "check_status": "found",
            "size_bytes": int(response.get("ContentLength", 0)),
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
            "etag_present": False,
            "error": error_code or type(exc).__name__,
        }


def local_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def default_local(relative_path: str) -> str:
    return str(LOCAL_ROOT / relative_path)


def selected_scope(args: argparse.Namespace) -> str:
    selected = [
        name
        for name, enabled in (
            ("transformer", args.only_transformer),
            ("vae", args.only_vae),
            ("umt5", args.only_umt5),
            ("wav2vec", args.only_wav2vec),
            ("all", args.only_all),
        )
        if enabled
    ]
    if len(selected) > 1:
        raise RuntimeError("Use only one upload selector at a time.")
    return selected[0] if selected else "all"


def selected_targets(args: argparse.Namespace) -> tuple[str, tuple[dict, ...]]:
    scope = selected_scope(args)
    names = TARGET_GROUPS[scope]
    return scope, tuple(target for target in UPLOAD_TARGETS if target["name"] in names)


def build_upload_plan(args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    items = []
    problems = []
    _, targets = selected_targets(args)
    for target in targets:
        raw_value = getattr(args, target["arg_name"])
        path = local_path(raw_value) if raw_value else None
        is_file = bool(path and path.is_file())
        size_bytes = path.stat().st_size if is_file else 0
        if not raw_value:
            problems.append(f"missing --{target['arg_name'].replace('_', '-')}")
        elif not is_file:
            problems.append(f"local file not found for --{target['arg_name'].replace('_', '-')}: {path}")
        elif size_bytes <= 0:
            problems.append(f"local file has invalid size for --{target['arg_name'].replace('_', '-')}: {path}")
        r2_key = R2_PREFIX + target["comfy_models_relative_path"]
        items.append(
            {
                "name": target["name"],
                "arg_name": target["arg_name"],
                "local_path": str(path) if path else "",
                "local_is_file": is_file,
                "local_size_bytes": size_bytes,
                "r2_key": r2_key,
                "container_path": "/opt/ComfyUI/models/" + target["comfy_models_relative_path"],
                "upload_attempted": False,
                "upload_status": "not_attempted",
                "remote_before": {},
                "remote_after": {},
                "remote_size_matches_local": False,
            }
        )
    return items, problems


def build_log(
    args: argparse.Namespace,
    config: dict,
    items: list[dict],
    status: str,
    problems: list[str],
    error: str = "",
) -> dict:
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
        "execute_allowed": execute_allowed,
        "dry_run": not execute_allowed,
        "overwrite": args.overwrite,
        "upload_scope": scope,
        "r2_prefix": R2_PREFIX,
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "upload_file_then_head_object" if execute_allowed else "dry_run_plan_only",
        "no_download": True,
        "no_runpod": True,
        "not_latentsync": True,
        "not_wan27": True,
        "mirrors_comfyui_models_root": True,
        "upload_targets": items,
    }


def run(args: argparse.Namespace) -> int:
    config = {}
    items = []
    problems = []
    execute_allowed = args.execute and args.confirm_upload
    try:
        scope, _ = selected_targets(args)
        items, problems = build_upload_plan(args)
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} scope={scope} targets={len(items)}")
        for item in items:
            local_state = "ok" if item["local_is_file"] and item["local_size_bytes"] > 0 else "missing_or_invalid"
            print(f"[{TEST_ID}] LOCAL {local_state} size={item['local_size_bytes']} key={item['r2_key']}")
        if args.execute and not args.confirm_upload:
            problems.append("real upload requires --execute --confirm-upload")
        if problems:
            status = "blocked_before_upload" if args.execute else "dry_run_local_inputs_missing"
            write_json(LOG_PATH, build_log(args, config, items, status, problems))
            for problem in problems:
                print(f"[{TEST_ID}] {problem}")
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 1 if args.execute else 0
        if not execute_allowed:
            status = "dry_run_ready"
            write_json(LOG_PATH, build_log(args, config, items, status, []))
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        upload_errors = []
        for item in items:
            before = head_object(client, config["bucket"], item["r2_key"])
            item["remote_before"] = before
            if before["exists"] and not args.overwrite:
                item["upload_status"] = "blocked_existing_remote_without_overwrite"
                upload_errors.append(f"remote object already exists: {item['r2_key']}")
                continue
            if before["check_status"] == "error":
                item["upload_status"] = "blocked_remote_head_error"
                upload_errors.append(f"remote HEAD error for {item['r2_key']}: {before['error']}")
                continue
            item["upload_attempted"] = True
            client.upload_file(item["local_path"], config["bucket"], item["r2_key"])
            after = head_object(client, config["bucket"], item["r2_key"])
            item["remote_after"] = after
            item["remote_size_matches_local"] = after.get("size_bytes") == item["local_size_bytes"]
            item["upload_status"] = (
                "uploaded_and_head_size_ok"
                if after.get("exists") and item["remote_size_matches_local"]
                else "uploaded_but_head_size_mismatch"
            )
            print(
                f"[{TEST_ID}] UPLOADED status={item['upload_status']} "
                f"local_size={item['local_size_bytes']} remote_size={after.get('size_bytes', 0)} key={item['r2_key']}"
            )
            if item["upload_status"] != "uploaded_and_head_size_ok":
                upload_errors.append(f"remote HEAD after upload did not match local size: {item['r2_key']}")
        status = "succeeded" if not upload_errors and all(item["upload_status"] == "uploaded_and_head_size_ok" for item in items) else "upload_failed"
        write_json(LOG_PATH, build_log(args, config, items, status, upload_errors))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(args, config, items, "failed", problems, message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or upload Wan2.2 S2V V1 minimum model weights to R2.")
    for target in UPLOAD_TARGETS:
        parser.add_argument(
            "--" + target["arg_name"].replace("_", "-"),
            default=default_local(target["comfy_models_relative_path"]),
            help="Local prepared safetensors path. The script never downloads weights.",
        )
    parser.add_argument("--only-transformer", action="store_true")
    parser.add_argument("--only-vae", action="store_true")
    parser.add_argument("--only-umt5", action="store_true")
    parser.add_argument("--only-wav2vec", action="store_true")
    parser.add_argument("--only-all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Perform real R2 uploads only with --confirm-upload.")
    parser.add_argument("--confirm-upload", action="store_true", help="Required with --execute for real uploads.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
