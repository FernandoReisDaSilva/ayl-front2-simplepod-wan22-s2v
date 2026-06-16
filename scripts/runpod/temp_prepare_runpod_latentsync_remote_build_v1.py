import argparse
import json
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_RUNPOD_LATENTSYNC_REMOTE_BUILD_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"
OUTPUT_DIR = REPO_ROOT / "tmp" / "runpod_latentsync_remote_build_v1"
INTENDED_PAYLOAD_PATH = OUTPUT_DIR / "intended_payload.json"
LOCAL_REPORT_PATH = OUTPUT_DIR / "output" / "remote_build_report.json"
LOG_PATH = REPO_ROOT / "logs" / "runpod_latentsync_remote_build_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install requests python-dotenv boto3"

REQUIRED_EXEC_ENV_VARS = (
    "RUNPOD_API_KEY",
    "R2_ENDPOINT",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_REGION",
    "REGISTRY_USERNAME",
)
SECRET_ENV_VARS = (
    "RUNPOD_API_KEY",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "REGISTRY_PASSWORD",
    "REGISTRY_TOKEN",
)
EXPENSIVE_GPU_MARKERS = (
    "A100",
    "H100",
    "H200",
    "B200",
    "B300",
    "L40",
    "L40S",
    "4090",
    "5090",
    "80GB",
    "96GB",
    "141",
    "143",
    "180",
    "192",
    "288",
)

DEFAULT_TEMPLATE_ID = "runpod-ubuntu-2404"
DEFAULT_GPU_TYPE_ID = ""
DEFAULT_CLOUD_TYPE = "COMMUNITY"
DEFAULT_POD_NAME = "ayl-latentsync-remote-build-v1"
DEFAULT_BUILDER_IMAGE = "gcr.io/kaniko-project/executor:debug"
DEFAULT_SOURCE_GIT_REF = "main"
DEFAULT_R2_PREFIX = "tests/runpod_latentsync_remote_build_v1"

CREATE_POD_MUTATION = """
mutation RunpodLatentSyncRemoteBuildV1Create($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    machineId
  }
}
"""

POD_STATUS_QUERY = """
query RunpodLatentSyncRemoteBuildV1Status($podId: String!) {
  pod(input: { podId: $podId }) {
    id
    name
    desiredStatus
    machineId
    runtime {
      uptimeInSeconds
    }
  }
}
"""

TERMINATE_POD_MUTATION = """
mutation RunpodLatentSyncRemoteBuildV1Terminate($podId: String!) {
  podTerminate(input: { podId: $podId })
}
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(DEPENDENCY_NOTE) from exc
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)


def import_requests():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(DEPENDENCY_NOTE) from exc
    return requests


def import_boto3():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(DEPENDENCY_NOTE) from exc
    return boto3


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sanitize_string(value: str) -> str:
    value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
    value = re.sub(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    value = re.sub(r"(?i)(access[_-]?key[_-]?id['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    value = re.sub(r"(?i)(secret[_-]?access[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    value = re.sub(r"(?i)(registry[_-]?(password|token)['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    return value


def redact_env_entry(entry: dict) -> dict:
    key = str(entry.get("key", ""))
    value = str(entry.get("value", ""))
    if key in SECRET_ENV_VARS or "SECRET" in key or "PASSWORD" in key or "TOKEN" in key:
        value = "<redacted>"
    return {"key": key, "value": value}


def response_shape(payload) -> dict:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    shape = {"top_level_keys": sorted(payload.keys())}
    if isinstance(payload.get("data"), dict):
        shape["data_keys"] = sorted(payload["data"].keys())
    if isinstance(payload.get("errors"), list):
        shape["graphql_errors"] = [
            {"message": sanitize_string(str(error.get("message", error)))}
            if isinstance(error, dict)
            else {"message": sanitize_string(str(error))}
            for error in payload["errors"]
        ]
    return shape


def safe_event(event_type: str, message: str = "", payload=None, http_status_code: int | None = None) -> dict:
    event = {
        "event": event_type,
        "created_at": now_iso(),
        "http_status_code": http_status_code,
    }
    if message:
        event["message"] = sanitize_string(message)
    if payload is not None:
        event["response_shape"] = response_shape(payload)
    return event


def endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint.split("/")[0]


def registry_password_env_name() -> str:
    if os.getenv("REGISTRY_TOKEN"):
        return "REGISTRY_TOKEN"
    return "REGISTRY_PASSWORD"


def env_present() -> dict:
    keys = list(REQUIRED_EXEC_ENV_VARS) + ["REGISTRY_PASSWORD", "REGISTRY_TOKEN"]
    return {key: bool(os.getenv(key, "")) for key in keys}


def missing_env_vars(args: argparse.Namespace) -> list[str]:
    missing = [key for key in REQUIRED_EXEC_ENV_VARS if not os.getenv(key, "")]
    if not os.getenv("REGISTRY_PASSWORD", "") and not os.getenv("REGISTRY_TOKEN", ""):
        missing.append("REGISTRY_PASSWORD or REGISTRY_TOKEN")
    if not args.source_git_url:
        missing.append("--source-git-url or LATENTSYNC_SOURCE_GIT_URL")
    if not args.remote_image_tag:
        missing.append("--remote-image-tag or LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG")
    if args.gpu_count > 0 and not args.gpu_type_id:
        missing.append("--gpu-type-id when --gpu-count is greater than 0")
    return missing


def expensive_gpu_selected(gpu_type_id: str) -> bool:
    upper = gpu_type_id.upper()
    return any(marker in upper for marker in EXPENSIVE_GPU_MARKERS)


def build_r2_keys(prefix: str) -> dict:
    prefix = prefix.strip("/")
    return {
        "started": f"{prefix}/progress/started.json",
        "build_started": f"{prefix}/progress/build_started.json",
        "build_finished": f"{prefix}/progress/build_finished.json",
        "final_report": f"{prefix}/output/remote_build_report.json",
    }


def remote_command(args: argparse.Namespace, r2_keys: dict) -> str:
    # The command is intentionally plain shell. It is a prepared future execution
    # path, not exercised by dry-run. Kaniko avoids Docker daemon assumptions.
    dockerfile = "docker/latentsync-runpod-v1/Dockerfile"
    password_name = registry_password_env_name()
    source_ref = args.source_git_ref
    download_checkpoints = "1" if args.download_checkpoints == "1" else "0"
    return f"""set -euo pipefail
workdir=/workspace/latentsync_remote_build_v1
mkdir -p "$workdir"
report="$workdir/remote_build_report.json"
write_report() {{
  status="$1"
  message="$2"
  cat > "$report" <<JSON
{{"test_id":"{TEST_ID}","status":"$status","message":"$message","created_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","download_checkpoints":"{download_checkpoints}","remote_image_tag":"{args.remote_image_tag}"}}
JSON
}}
upload_report() {{
  python3 - "$report" "$R2_FINAL_REPORT_KEY" <<'PY'
import os, sys
import boto3
path, key = sys.argv[1], sys.argv[2]
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name=os.environ["R2_REGION"],
)
client.upload_file(path, os.environ["R2_BUCKET"], key)
PY
}}
write_progress() {{
  key="$1"
  message="$2"
  progress="$workdir/progress.json"
  echo "{{\\"test_id\\":\\"{TEST_ID}\\",\\"created_at\\":\\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\\",\\"message\\":\\"$message\\"}}" > "$progress"
  python3 - "$progress" "$key" <<'PY'
import os, sys
import boto3
path, key = sys.argv[1], sys.argv[2]
client = boto3.client(
    "s3",
    endpoint_url=os.environ["R2_ENDPOINT"],
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    region_name=os.environ["R2_REGION"],
)
client.upload_file(path, os.environ["R2_BUCKET"], key)
PY
}}
trap 'rc=$?; if [ "$rc" -ne 0 ]; then write_report failed "remote build command failed with exit code $rc"; upload_report || true; fi' EXIT
write_progress "$R2_STARTED_KEY" "remote build pod command started"
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required for R2 reporting in this prepared route" >&2
  exit 20
fi
python3 -m pip install --no-cache-dir boto3 >/tmp/remote_build_boto3_install.log 2>&1 || true
mkdir -p /kaniko/.docker
auth="$(printf '%s:%s' "$REGISTRY_USERNAME" "${{{password_name}}}" | base64 | tr -d '\\n')"
registry_host="$(python3 - <<'PY'
import os
tag = os.environ["REMOTE_IMAGE_TAG"]
print(tag.split("/")[0])
PY
)"
printf '{{"auths":{{"%s":{{"auth":"%s"}}}}}}' "$registry_host" "$auth" > /kaniko/.docker/config.json
write_progress "$R2_BUILD_STARTED_KEY" "kaniko build started"
/kaniko/executor \\
  --context "git://{args.source_git_url}#{source_ref}" \\
  --dockerfile "{dockerfile}" \\
  --destination "$REMOTE_IMAGE_TAG" \\
  --build-arg DOWNLOAD_CHECKPOINTS={download_checkpoints} \\
  --verbosity info
write_progress "$R2_BUILD_FINISHED_KEY" "kaniko build finished"
write_report succeeded "remote build completed and pushed image"
upload_report
"""


def remote_env(args: argparse.Namespace, r2_keys: dict) -> list[dict]:
    password_name = registry_password_env_name()
    env = [
        {"key": "R2_ENDPOINT", "value": os.getenv("R2_ENDPOINT", "")},
        {"key": "R2_ACCESS_KEY_ID", "value": os.getenv("R2_ACCESS_KEY_ID", "")},
        {"key": "R2_SECRET_ACCESS_KEY", "value": os.getenv("R2_SECRET_ACCESS_KEY", "")},
        {"key": "R2_BUCKET", "value": os.getenv("R2_BUCKET", "")},
        {"key": "R2_REGION", "value": os.getenv("R2_REGION", "")},
        {"key": "R2_STARTED_KEY", "value": r2_keys["started"]},
        {"key": "R2_BUILD_STARTED_KEY", "value": r2_keys["build_started"]},
        {"key": "R2_BUILD_FINISHED_KEY", "value": r2_keys["build_finished"]},
        {"key": "R2_FINAL_REPORT_KEY", "value": r2_keys["final_report"]},
        {"key": "REGISTRY_USERNAME", "value": os.getenv("REGISTRY_USERNAME", "")},
        {"key": password_name, "value": os.getenv(password_name, "")},
        {"key": "REMOTE_IMAGE_TAG", "value": args.remote_image_tag or ""},
    ]
    return env


def mutation_input(args: argparse.Namespace, r2_keys: dict) -> dict:
    return {
        "cloudType": args.cloud_type,
        "gpuCount": args.gpu_count,
        "volumeInGb": 0,
        "containerDiskInGb": args.container_disk_gb,
        "minVcpuCount": args.min_vcpu_count,
        "minMemoryInGb": args.min_memory_gb,
        "gpuTypeId": args.gpu_type_id,
        "name": args.pod_name,
        "imageName": args.builder_image,
        "dockerArgs": remote_command(args, r2_keys),
        "ports": "",
        "templateId": args.template_id,
        "env": remote_env(args, r2_keys),
    }


def redacted_mutation_input(args: argparse.Namespace, r2_keys: dict) -> dict:
    payload = mutation_input(args, r2_keys)
    payload["env"] = [redact_env_entry(item) for item in payload["env"]]
    return payload


def intended_payload(args: argparse.Namespace, r2_keys: dict) -> dict:
    return {
        "test_id": TEST_ID,
        "purpose": "prepare_controlled_remote_build_for_latentsync_v1",
        "dry_run_default": True,
        "requires_execute_flag": True,
        "requires_confirm_cost_risk_flag": True,
        "does_not_execute_in_dry_run": True,
        "recommended_route": "Prefer GitHub Actions or registry-native build. Use RunPod builder Pod only if necessary.",
        "builder_strategy": args.builder_strategy,
        "docker_in_docker_warning": "Do not assume a normal RunPod Pod can run a Docker daemon.",
        "no_network_volume": True,
        "no_gpu_required_for_build": True,
        "gpu_count": args.gpu_count,
        "gpu_type_id": args.gpu_type_id or "none_requested",
        "download_checkpoints": args.download_checkpoints,
        "r2_keys": r2_keys,
        "mutation_input_redacted": redacted_mutation_input(args, r2_keys),
        "safety_notes": [
            "Default mode sends no mutation and creates no Pod.",
            "Future execution is billable.",
            "The remote command must upload progress/final reports to R2.",
            "The future Pod should terminate automatically after build/push/report.",
            "manual_cleanup_required must be checked after execution.",
        ],
    }


def graphql_request(requests_module, api_key: str, query: str, variables: dict, timeout_seconds: float):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return requests_module.post(
        GRAPHQL_ENDPOINT,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=timeout_seconds,
    )


def parse_json_response(response):
    try:
        return response.json()
    except ValueError:
        return {"non_json_response_text_truncated": sanitize_string(response.text[:1000])}


def r2_client():
    boto3 = import_boto3()
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.environ["R2_REGION"],
    )


def r2_object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def download_r2_report(client, bucket: str, key: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, str(destination))
        return True
    except Exception:
        return False


def extract_pod_id(payload) -> str:
    try:
        return payload["data"]["podFindAndDeployOnDemand"]["id"] or ""
    except Exception:
        return ""


def terminate_pod(requests_module, api_key: str, pod_id: str, timeout_seconds: float, status_events: list[dict]) -> bool:
    if not pod_id:
        return False
    response = graphql_request(
        requests_module,
        api_key,
        TERMINATE_POD_MUTATION,
        {"podId": pod_id},
        timeout_seconds,
    )
    payload = parse_json_response(response)
    status_events.append(safe_event("terminate_pod", payload=payload, http_status_code=response.status_code))
    return bool(payload.get("data", {}).get("podTerminate") is True)


def run(args: argparse.Namespace) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "output").mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    load_repo_dotenv()
    args.source_git_url = args.source_git_url or os.getenv("LATENTSYNC_SOURCE_GIT_URL", "")
    args.source_git_ref = args.source_git_ref or os.getenv("LATENTSYNC_SOURCE_GIT_REF", DEFAULT_SOURCE_GIT_REF)
    args.remote_image_tag = args.remote_image_tag or os.getenv("LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG", "")
    args.download_checkpoints = args.download_checkpoints or os.getenv("LATENTSYNC_DOWNLOAD_CHECKPOINTS", "0")
    args.r2_prefix = args.r2_prefix or os.getenv("R2_REMOTE_BUILD_PREFIX", DEFAULT_R2_PREFIX)
    r2_keys = build_r2_keys(args.r2_prefix)

    status_events = [safe_event("start", "remote build preparation started")]
    log = {
        "test_id": TEST_ID,
        "started_at": now_iso(),
        "execute_requested": args.execute,
        "confirm_cost_risk": args.confirm_cost_risk,
        "manual_cleanup_required": False,
        "env_present": env_present(),
        "status_events": status_events,
    }

    write_json(INTENDED_PAYLOAD_PATH, intended_payload(args, r2_keys))
    status_events.append(safe_event("write_intended_payload", f"Wrote {INTENDED_PAYLOAD_PATH}"))
    write_json(LOG_PATH, log)

    if not args.execute:
        status_events.append(safe_event("dry_run_complete", "No mutation sent; no Pod created."))
        log["finished_at"] = now_iso()
        log["status"] = "dry_run"
        write_json(LOG_PATH, log)
        print("RunPod LatentSync remote build V1 dry-run prepared.")
        print(f"Intended payload: {INTENDED_PAYLOAD_PATH}")
        print(f"Log: {LOG_PATH}")
        return 0

    if not args.confirm_cost_risk:
        status_events.append(safe_event("blocked_missing_confirm_cost_risk"))
        log["status"] = "blocked"
        write_json(LOG_PATH, log)
        print("Blocked: execution requires --confirm-cost-risk.", file=sys.stderr)
        return 2

    if args.gpu_type_id and expensive_gpu_selected(args.gpu_type_id) and not args.allow_expensive_gpu:
        status_events.append(safe_event("blocked_expensive_gpu", args.gpu_type_id))
        log["status"] = "blocked"
        write_json(LOG_PATH, log)
        print("Blocked: selected GPU is considered expensive. Use --allow-expensive-gpu only after explicit approval.", file=sys.stderr)
        return 2

    missing = missing_env_vars(args)
    if missing:
        status_events.append(safe_event("blocked_missing_env", ", ".join(missing)))
        log["status"] = "blocked"
        write_json(LOG_PATH, log)
        print("Blocked: missing required env/config values: " + ", ".join(missing), file=sys.stderr)
        return 2

    requests_module = import_requests()
    api_key = os.environ["RUNPOD_API_KEY"]
    pod_id = ""
    terminated = False
    try:
        client = r2_client()
        create_response = graphql_request(
            requests_module,
            api_key,
            CREATE_POD_MUTATION,
            {"input": mutation_input(args, r2_keys)},
            args.request_timeout_seconds,
        )
        create_payload = parse_json_response(create_response)
        status_events.append(safe_event("create_pod", payload=create_payload, http_status_code=create_response.status_code))
        pod_id = extract_pod_id(create_payload)
        log["pod_id"] = pod_id
        write_json(LOG_PATH, log)
        if not pod_id:
            raise RuntimeError("Pod creation did not return a pod id.")

        deadline = time.time() + args.max_wait_seconds
        while time.time() < deadline:
            if r2_object_exists(client, os.environ["R2_BUCKET"], r2_keys["final_report"]):
                downloaded = download_r2_report(client, os.environ["R2_BUCKET"], r2_keys["final_report"], LOCAL_REPORT_PATH)
                status_events.append(safe_event("r2_final_report_detected", f"downloaded={downloaded}"))
                break
            status_events.append(safe_event("poll_r2", "final report not ready"))
            write_json(LOG_PATH, log)
            time.sleep(args.poll_interval_seconds)
        else:
            status_events.append(safe_event("timeout", "final R2 report not detected before timeout"))

        terminated = terminate_pod(requests_module, api_key, pod_id, args.request_timeout_seconds, status_events)
        log["terminated"] = terminated
        log["manual_cleanup_required"] = not terminated
        log["status"] = "succeeded" if terminated and LOCAL_REPORT_PATH.exists() else "finished_with_warnings"
        log["finished_at"] = now_iso()
        write_json(LOG_PATH, log)
        return 0 if terminated else 1
    except Exception as exc:
        status_events.append(safe_event("error", str(exc)))
        if pod_id and not terminated:
            try:
                terminated = terminate_pod(requests_module, api_key, pod_id, args.request_timeout_seconds, status_events)
            except Exception as term_exc:
                status_events.append(safe_event("terminate_error", str(term_exc)))
        log["terminated"] = terminated
        log["manual_cleanup_required"] = bool(pod_id and not terminated)
        log["status"] = "failed"
        log["error"] = sanitize_string(str(exc))
        log["finished_at"] = now_iso()
        write_json(LOG_PATH, log)
        print(f"RunPod LatentSync remote build V1 failed: {sanitize_string(str(exc))}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, and only when explicitly approved execute, a controlled RunPod remote build for "
            "the LatentSync V1 image. Default is dry-run: no mutation, no Pod, no build, no push."
        )
    )
    parser.add_argument("--execute", action="store_true", help="Send RunPod mutation. Requires --confirm-cost-risk.")
    parser.add_argument("--confirm-cost-risk", action="store_true", help="Confirm billable RunPod risk.")
    parser.add_argument("--allow-expensive-gpu", action="store_true", help="Allow blocked high-cost GPU markers.")
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID)
    parser.add_argument("--gpu-count", type=int, default=0)
    parser.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE)
    parser.add_argument("--pod-name", default=DEFAULT_POD_NAME)
    parser.add_argument("--builder-image", default=DEFAULT_BUILDER_IMAGE)
    parser.add_argument("--builder-strategy", default="kaniko_git_context")
    parser.add_argument("--source-git-url", default="")
    parser.add_argument("--source-git-ref", default="")
    parser.add_argument("--remote-image-tag", default="")
    parser.add_argument("--download-checkpoints", choices=("0", "1"), default="")
    parser.add_argument("--r2-prefix", default="")
    parser.add_argument("--container-disk-gb", type=int, default=40)
    parser.add_argument("--min-vcpu-count", type=int, default=4)
    parser.add_argument("--min-memory-gb", type=int, default=16)
    parser.add_argument("--max-wait-seconds", type=float, default=3600)
    parser.add_argument("--poll-interval-seconds", type=float, default=15)
    parser.add_argument("--request-timeout-seconds", type=float, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
