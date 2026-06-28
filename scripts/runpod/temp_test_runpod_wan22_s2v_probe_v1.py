import argparse
import json
import os
import re
import secrets
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_RUNPOD_WAN22_S2V_PROBE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
GRAPHQL_ENDPOINT = "https://api.runpod.io/graphql"
OUTPUT_DIR = REPO_ROOT / "tmp" / "runpod_wan22_s2v_probe_v1"
INTENDED_PAYLOAD_PATH = OUTPUT_DIR / "intended_payload.json"
LOCAL_FINAL_REPORT_PATH = OUTPUT_DIR / "output" / "final_report.json"
LOG_PATH = REPO_ROOT / "logs" / "runpod_wan22_s2v_probe_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install requests python-dotenv boto3"

DEFAULT_TEMPLATE_ID = "runpod-ubuntu-2404"
DEFAULT_GPU_TYPE_ID = "NVIDIA GeForce RTX 3090"
DEFAULT_CLOUD_TYPE = "COMMUNITY"
DEFAULT_POD_NAME = "ayl-test-wan22-s2v-probe-v1"
DEFAULT_IMAGE_TAG = "ghcr.io/fernandoreisdasilva/ayl-wan22-s2v-runpod:0.1.0"
DEFAULT_CONTAINER_DISK_GB = 80
RUN_MODE = "wan22_s2v_probe"

R2_PROGRESS_KEY = "tests/runpod_wan22_s2v_probe_v1/progress/container_started.json"
R2_FINAL_REPORT_KEY = "tests/runpod_wan22_s2v_probe_v1/output/final_report.json"
R2_INPUT_REFERENCE_IMAGE_KEY = "tests/runpod_wan22_s2v_probe_v1/input/mae_reference.png"
R2_INPUT_AUDIO_KEY = "tests/runpod_wan22_s2v_probe_v1/input/mae_audio_5s.wav"
R2_OUTPUT_VIDEO_KEY = "tests/runpod_wan22_s2v_probe_v1/output/video_out.mp4"
R2_WAN22_MODEL_PREFIX = "checkpoints/wan22_s2v/comfyui_models/"
DEFAULT_OUTPUT_FILENAME_PREFIX = "ayl_wan22_s2v_probe_v1/video_out"
DEFAULT_POSITIVE_PROMPT = "a woman is singing passionately"
DEFAULT_NEGATIVE_PROMPT = ""

REQUIRED_FINAL_REPORT_FIELDS = (
    "runtime_probe_status",
    "r2_upload_status",
    "torch_probe",
    "ffmpeg_exists",
    "input_files",
    "node_validation",
    "wan22_s2v_controls",
    "output_upload_status",
)

REQUIRED_ENV_VARS = ("RUNPOD_API_KEY", "R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")

CREATE_POD_MUTATION = """
mutation TestRunpodWan22S2VProbeV1Create($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) {
    id
    name
    desiredStatus
    machineId
  }
}
"""

TERMINATE_POD_MUTATION = """
mutation TestRunpodWan22S2VProbeV1Terminate($podId: String!) {
  podTerminate(input: { podId: $podId })
}
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint_host(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    return parsed.netloc or endpoint.split("/")[0]


def sanitize_string(value: str) -> str:
    value = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer <redacted>", value)
    value = re.sub(r"(?i)(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    value = re.sub(r"(?i)(access[_-]?key[_-]?id['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    value = re.sub(r"(?i)(secret[_-]?access[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", r"\1<redacted>", value)
    return value


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reset_output_dir() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_repo_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'python-dotenv'. Install it with: {DEPENDENCY_NOTE}") from exc
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=False)


def import_requests():
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'requests'. Install it with: {DEPENDENCY_NOTE}") from exc
    return requests


def import_boto3():
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'boto3'. Install it with: {DEPENDENCY_NOTE}") from exc
    return boto3


def missing_env_vars() -> list[str]:
    return [key for key in REQUIRED_ENV_VARS if not os.getenv(key, "")]


def r2_config() -> dict:
    return {
        "endpoint": os.environ["R2_ENDPOINT"],
        "access_key_id": os.environ["R2_ACCESS_KEY_ID"],
        "secret_access_key": os.environ["R2_SECRET_ACCESS_KEY"],
        "bucket": os.environ["R2_BUCKET"],
        "region": os.environ["R2_REGION"],
    }


def r2_client(config: dict):
    boto3 = import_boto3()
    return boto3.client("s3", endpoint_url=config["endpoint"], aws_access_key_id=config["access_key_id"], aws_secret_access_key=config["secret_access_key"], region_name=config["region"])


def r2_object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def r2_delete_if_exists(client, bucket: str, key: str) -> bool:
    if not r2_object_exists(client, bucket, key):
        return False
    client.delete_object(Bucket=bucket, Key=key)
    return True


def download_final_report(client, bucket: str, key: str, marker_nonce: str) -> tuple[bool, bool]:
    try:
        LOCAL_FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, key, str(LOCAL_FINAL_REPORT_PATH))
        data = json.loads(LOCAL_FINAL_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False, False
    required_fields_present = all(field in data for field in REQUIRED_FINAL_REPORT_FIELDS)
    verified = (
        data.get("r2_upload_status") == "ok"
        and data.get("runtime_probe_status") == "ok"
        and data.get("output_upload_status") == "ok"
        and data.get("marker_nonce") == marker_nonce
        and data.get("env_present_redacted", {}).get("AYL_RUN_MODE") is True
        and required_fields_present
    )
    return True, verified


def control_env(args: argparse.Namespace) -> list[dict]:
    return [
        {"key": "WAN22_S2V_CFG", "value": str(args.cfg)},
        {"key": "WAN22_S2V_SHIFT", "value": str(args.shift)},
        {"key": "WAN22_S2V_SEED", "value": str(args.seed)},
        {"key": "WAN22_S2V_STEPS", "value": str(args.steps)},
        {"key": "WAN22_S2V_DENOISE_STRENGTH", "value": str(args.denoise_strength)},
        {"key": "WAN22_S2V_AUDIO_SCALE", "value": str(args.audio_scale)},
        {"key": "WAN22_S2V_POSE_START_PERCENT", "value": str(args.pose_start_percent)},
        {"key": "WAN22_S2V_POSE_END_PERCENT", "value": str(args.pose_end_percent)},
        {"key": "WAN22_S2V_WIDTH", "value": str(args.width)},
        {"key": "WAN22_S2V_HEIGHT", "value": str(args.height)},
        {"key": "WAN22_S2V_OUTPUT_FILENAME_PREFIX", "value": args.output_filename_prefix},
        {"key": "WAN22_S2V_POSITIVE_PROMPT", "value": args.positive_prompt},
        {"key": "WAN22_S2V_NEGATIVE_PROMPT", "value": args.negative_prompt},
        {"key": "WAN22_S2V_PROMPT_TIMEOUT_SECONDS", "value": str(args.prompt_timeout_seconds)},
    ]


def redacted_env(args: argparse.Namespace | None = None) -> list[dict]:
    image_tag = args.image_tag if args else "<public_image_tag>"
    controls = control_env(args) if args else []
    return [
        {"key": "AYL_RUN_MODE", "value": RUN_MODE},
        {"key": "AYL_IMAGE_TAG", "value": image_tag},
        {"key": "AYL_MARKER_NONCE", "value": "<generated>"},
        *controls,
        {"key": "R2_PROGRESS_KEY", "value": R2_PROGRESS_KEY},
        {"key": "R2_FINAL_REPORT_KEY", "value": R2_FINAL_REPORT_KEY},
        {"key": "R2_INPUT_REFERENCE_IMAGE_KEY", "value": R2_INPUT_REFERENCE_IMAGE_KEY},
        {"key": "R2_INPUT_AUDIO_KEY", "value": R2_INPUT_AUDIO_KEY},
        {"key": "R2_OUTPUT_VIDEO_KEY", "value": args.output_video_key if args else R2_OUTPUT_VIDEO_KEY},
        {"key": "R2_WAN22_MODEL_PREFIX", "value": R2_WAN22_MODEL_PREFIX},
        {"key": "R2_ENDPOINT", "value": "<redacted>"},
        {"key": "R2_ACCESS_KEY_ID", "value": "<redacted>"},
        {"key": "R2_SECRET_ACCESS_KEY", "value": "<redacted>"},
        {"key": "R2_BUCKET", "value": "<redacted>"},
        {"key": "R2_REGION", "value": "<redacted>"},
    ]


def pod_env(config: dict, marker_nonce: str, args: argparse.Namespace) -> list[dict]:
    return [
        {"key": "AYL_RUN_MODE", "value": RUN_MODE},
        {"key": "AYL_IMAGE_TAG", "value": args.image_tag},
        {"key": "AYL_MARKER_NONCE", "value": marker_nonce},
        *control_env(args),
        {"key": "R2_ENDPOINT", "value": config["endpoint"]},
        {"key": "R2_ACCESS_KEY_ID", "value": config["access_key_id"]},
        {"key": "R2_SECRET_ACCESS_KEY", "value": config["secret_access_key"]},
        {"key": "R2_BUCKET", "value": config["bucket"]},
        {"key": "R2_REGION", "value": config["region"]},
        {"key": "R2_PROGRESS_KEY", "value": R2_PROGRESS_KEY},
        {"key": "R2_FINAL_REPORT_KEY", "value": R2_FINAL_REPORT_KEY},
        {"key": "R2_INPUT_REFERENCE_IMAGE_KEY", "value": R2_INPUT_REFERENCE_IMAGE_KEY},
        {"key": "R2_INPUT_AUDIO_KEY", "value": R2_INPUT_AUDIO_KEY},
        {"key": "R2_OUTPUT_VIDEO_KEY", "value": args.output_video_key},
        {"key": "R2_WAN22_MODEL_PREFIX", "value": R2_WAN22_MODEL_PREFIX},
    ]


def mutation_input(args: argparse.Namespace, config: dict, marker_nonce: str) -> dict:
    return {
        "cloudType": args.cloud_type,
        "gpuCount": 1,
        "volumeInGb": 0,
        "containerDiskInGb": args.container_disk_gb,
        "minVcpuCount": 1,
        "minMemoryInGb": 1,
        "gpuTypeId": args.gpu_type_id,
        "name": args.pod_name,
        "imageName": args.image_tag,
        "ports": "",
        "templateId": args.template_id,
        "env": pod_env(config, marker_nonce, args),
    }


def intended_payload(args: argparse.Namespace, marker_nonce: str) -> dict:
    return {
        "test_id": TEST_ID,
        "purpose": "validate_wan22_s2v_comfyui_runpod_community_r2_probe_v1_without_network_volume",
        "image_tag": args.image_tag,
        "run_mode": RUN_MODE,
        "gpu_type_id": args.gpu_type_id,
        "cloud_type": args.cloud_type,
        "template_id": args.template_id,
        "pod_name": args.pod_name,
        "container_disk_gb": args.container_disk_gb,
        "max_wait_seconds": args.max_wait_seconds,
        "poll_interval_seconds": args.poll_interval_seconds,
        "request_timeout_seconds": args.timeout_seconds,
        "workflow_source": "kijai/ComfyUI-WanVideoWrapper/s2v/wanvideo2_2_S2V_context_window_testing.json",
        "workflow_real_node_ids": {"image": 73, "audio": 94, "sampler": 27, "s2v_embeds": 101, "video_combine": [30, 97]},
        "controls": {item["key"]: item["value"] for item in control_env(args)},
        "dry_run_default": True,
        "requires_execute_flag": True,
        "requires_confirm_cost_risk_flag": True,
        "no_dockerArgs": True,
        "no_pod_logs_dependency": True,
        "network_volume_required": False,
        "not_latentsync": True,
        "not_wan27": True,
        "r2_progress_key": R2_PROGRESS_KEY,
        "r2_final_report_key": R2_FINAL_REPORT_KEY,
        "r2_output_video_key": args.output_video_key,
        "r2_input_keys": {"reference_image": R2_INPUT_REFERENCE_IMAGE_KEY, "audio": R2_INPUT_AUDIO_KEY, "model_prefix": R2_WAN22_MODEL_PREFIX},
        "marker_nonce": marker_nonce,
        "mutation_input_redacted": {
            "cloudType": args.cloud_type,
            "gpuCount": 1,
            "volumeInGb": 0,
            "containerDiskInGb": args.container_disk_gb,
            "minVcpuCount": 1,
            "minMemoryInGb": 1,
            "gpuTypeId": args.gpu_type_id,
            "name": args.pod_name,
            "imageName": args.image_tag,
            "ports": "",
            "templateId": args.template_id,
            "env": redacted_env(args),
        },
    }


def graphql_request(requests_module, api_key: str, query: str, variables: dict, timeout_seconds: float):
    return requests_module.post(GRAPHQL_ENDPOINT, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"query": query, "variables": variables}, timeout=timeout_seconds)


def parse_json_response(response):
    try:
        return response.json()
    except ValueError:
        return {"non_json_response": True, "body_truncated_redacted": sanitize_string((getattr(response, "text", "") or "")[:1000])}


def response_shape(payload) -> dict:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    shape = {"top_level_keys": sorted(payload.keys())}
    if isinstance(payload.get("data"), dict):
        shape["data_keys"] = sorted(payload["data"].keys())
    if isinstance(payload.get("errors"), list):
        shape["graphql_errors"] = [{"message": sanitize_string(str(item.get("message", "")))} for item in payload["errors"] if isinstance(item, dict)]
    return shape


def has_graphql_errors(payload) -> bool:
    return isinstance(payload, dict) and bool(payload.get("errors"))


def extract_pod_id(payload) -> str:
    data = payload.get("data") if isinstance(payload, dict) else None
    result = data.get("podFindAndDeployOnDemand") if isinstance(data, dict) else None
    return str(result.get("id") or "") if isinstance(result, dict) else ""


def terminate_pod(requests_module, api_key: str, pod_id: str, timeout_seconds: float, events: list[dict]) -> bool:
    response = graphql_request(requests_module, api_key, TERMINATE_POD_MUTATION, {"podId": pod_id}, timeout_seconds)
    payload = parse_json_response(response)
    events.append({"event": "terminate_pod_mutation", "created_at": now_iso(), "http_status_code": response.status_code, "response_shape": response_shape(payload)})
    return response.status_code == 200 and not has_graphql_errors(payload)


def build_log(args: argparse.Namespace, **values) -> dict:
    data = {
        "test_id": TEST_ID,
        "endpoint_host": endpoint_host(GRAPHQL_ENDPOINT),
        "image_tag": args.image_tag,
        "run_mode": RUN_MODE,
        "gpu_type_id_requested": args.gpu_type_id,
        "cloud_type_requested": args.cloud_type,
        "template_id_requested": args.template_id,
        "pod_name_requested": args.pod_name,
        "container_disk_gb": args.container_disk_gb,
        "max_wait_seconds": args.max_wait_seconds,
        "poll_interval_seconds": args.poll_interval_seconds,
        "request_timeout_seconds": args.timeout_seconds,
        "network_volume_required": False,
        "dockerArgs_used": False,
        "not_latentsync": True,
        "not_wan27": True,
        "r2_progress_key": R2_PROGRESS_KEY,
        "r2_final_report_key": R2_FINAL_REPORT_KEY,
        "r2_output_video_key": args.output_video_key,
    }
    data.update(values)
    return data


def run(args: argparse.Namespace) -> int:
    reset_output_dir()
    created_at = now_iso()
    marker_nonce = f"nonce_{secrets.token_hex(12)}"
    execute_allowed = args.execute and args.confirm_cost_risk
    events = [{"event": "start", "created_at": created_at, "message": f"execute_allowed={execute_allowed}"}]
    errors = []
    pod_id = ""
    pod_created = False
    pod_terminated = None
    mutation_used = False
    create_attempted = False
    terminate_attempted = False
    r2_progress_detected = False
    r2_final_detected = False
    final_report_downloaded = False
    final_report_verified = False
    status = "started"

    def trace(message: str, *, stderr: bool = False) -> None:
        print(f"[{TEST_ID}] {message}", file=sys.stderr if stderr else sys.stdout, flush=True)

    def save_log(log_status: str) -> None:
        manual_cleanup_required = bool(pod_created and not pod_terminated and pod_id)
        write_json(
            LOG_PATH,
            build_log(
                args,
                auth_present=bool(os.getenv("RUNPOD_API_KEY", "")),
                r2_env_present=all(bool(os.getenv(key, "")) for key in REQUIRED_ENV_VARS if key != "RUNPOD_API_KEY"),
                dry_run=not execute_allowed,
                execute_requested=args.execute,
                confirm_cost_risk=args.confirm_cost_risk,
                mutation_used=mutation_used,
                create_mutation_attempted=create_attempted,
                terminate_mutation_attempted=terminate_attempted,
                pod_created=pod_created,
                pod_id=pod_id,
                pod_terminated=pod_terminated,
                r2_progress_detected=r2_progress_detected,
                r2_final_detected=r2_final_detected,
                final_report_downloaded=final_report_downloaded,
                final_report_verified=final_report_verified,
                status_events=events,
                error_messages=errors,
                manual_cleanup_required=manual_cleanup_required,
                created_at=created_at,
                finished_at=now_iso(),
                status=log_status,
            ),
        )

    try:
        load_repo_dotenv()
        write_json(INTENDED_PAYLOAD_PATH, intended_payload(args, marker_nonce))
        if not execute_allowed:
            save_log("dry_run_payload_created")
            print("RunPod Wan2.2 S2V probe V1 dry-run created. No mutation sent and no Pod created.")
            print(f"Intended payload written: {INTENDED_PAYLOAD_PATH}")
            print(f"Log written: {LOG_PATH}")
            return 0

        trace("START execute mode")
        trace(f"CONFIG image={args.image_tag} gpu_type_id={args.gpu_type_id} timeout={args.max_wait_seconds}s run_mode={RUN_MODE}")
        missing = missing_env_vars()
        if missing:
            raise RuntimeError(f"Missing required .env variable(s): {', '.join(missing)}")
        config = r2_config()
        client = r2_client(config)
        r2_delete_if_exists(client, config["bucket"], R2_PROGRESS_KEY)
        r2_delete_if_exists(client, config["bucket"], R2_FINAL_REPORT_KEY)
        r2_delete_if_exists(client, config["bucket"], args.output_video_key)

        requests = import_requests()
        api_key = os.environ["RUNPOD_API_KEY"]
        mutation_used = True
        create_attempted = True
        trace("POD_CREATE requested")
        response = graphql_request(requests, api_key, CREATE_POD_MUTATION, {"input": mutation_input(args, config, marker_nonce)}, args.timeout_seconds)
        payload = parse_json_response(response)
        events.append({"event": "create_pod_mutation", "created_at": now_iso(), "http_status_code": response.status_code, "response_shape": response_shape(payload)})
        if response.status_code != 200 or has_graphql_errors(payload):
            raise RuntimeError(f"create_pod_mutation failed with HTTP {response.status_code}")
        pod_id = extract_pod_id(payload)
        pod_created = bool(pod_id)
        if not pod_created:
            raise RuntimeError("create_pod_mutation: Pod ID not found")
        trace(f"POD_CREATED pod_id={pod_id}")
        save_log("pod_created")

        trace(f"R2_PROGRESS waiting key={R2_PROGRESS_KEY}")
        trace(f"R2_FINAL waiting key={R2_FINAL_REPORT_KEY}")
        deadline = time.monotonic() + args.max_wait_seconds
        poll_index = 0
        while time.monotonic() < deadline:
            poll_index += 1
            r2_progress_detected = r2_progress_detected or r2_object_exists(client, config["bucket"], R2_PROGRESS_KEY)
            r2_final_detected = r2_final_detected or r2_object_exists(client, config["bucket"], R2_FINAL_REPORT_KEY)
            trace(f"poll={poll_index} progress={str(r2_progress_detected).lower()} final={str(r2_final_detected).lower()}")
            save_log("polling_r2")
            if r2_progress_detected and r2_final_detected:
                break
            time.sleep(args.poll_interval_seconds)
        if not r2_progress_detected:
            errors.append(f"R2 progress not detected within {args.max_wait_seconds} seconds.")
        if not r2_final_detected:
            errors.append(f"R2 final report not detected within {args.max_wait_seconds} seconds.")
    except Exception as exc:
        errors.append(sanitize_string(str(exc)))
        events.append({"event": "exception", "created_at": now_iso(), "message": sanitize_string(str(exc))})
        trace(f"error: {sanitize_string(str(exc))}", stderr=True)
    finally:
        if pod_created and pod_id:
            try:
                trace("CLEANUP started")
                requests = import_requests()
                pod_terminated = terminate_pod(requests, os.environ.get("RUNPOD_API_KEY", ""), pod_id, args.timeout_seconds, events)
                terminate_attempted = True
                trace(f"CLEANUP done pod_terminated={str(bool(pod_terminated)).lower()}")
            except Exception as exc:
                pod_terminated = False
                errors.append(f"terminate_pod_mutation: {sanitize_string(str(exc))}")
        if mutation_used and r2_final_detected:
            try:
                config = r2_config()
                client = r2_client(config)
                final_report_downloaded, final_report_verified = download_final_report(client, config["bucket"], R2_FINAL_REPORT_KEY, marker_nonce)
            except Exception as exc:
                errors.append(f"download_final_report: {sanitize_string(str(exc))}")
        if pod_created and pod_terminated and r2_progress_detected and r2_final_detected and final_report_verified:
            status = "succeeded"
        elif mutation_used and not pod_created:
            status = "failed_before_pod_create"
        elif mutation_used:
            status = "failed_cleanup_attempted"
        else:
            status = "failed" if errors else "dry_run_payload_created"
        save_log(status)
        if mutation_used or pod_created:
            trace(f"DONE status={status} manual_cleanup_required={str(bool(pod_created and not pod_terminated and pod_id)).lower()}")
    return 0 if status in {"succeeded", "dry_run_payload_created"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute Wan2.2 S2V ComfyUI RunPod Community + R2 probe V1.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-cost-risk", action="store_true")
    parser.add_argument("--template-id", default=DEFAULT_TEMPLATE_ID)
    parser.add_argument("--gpu-type-id", default=DEFAULT_GPU_TYPE_ID)
    parser.add_argument("--cloud-type", default=DEFAULT_CLOUD_TYPE)
    parser.add_argument("--pod-name", default=DEFAULT_POD_NAME)
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG)
    parser.add_argument("--container-disk-gb", type=int, default=DEFAULT_CONTAINER_DISK_GB)
    parser.add_argument("--max-wait-seconds", type=float, default=2400)
    parser.add_argument("--poll-interval-seconds", type=float, default=15)
    parser.add_argument("--timeout-seconds", type=float, default=20)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--denoise-strength", type=float, default=1.0)
    parser.add_argument("--audio-scale", type=float, default=1.0)
    parser.add_argument("--pose-start-percent", type=float, default=0.0)
    parser.add_argument("--pose-end-percent", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--output-filename-prefix", default=DEFAULT_OUTPUT_FILENAME_PREFIX)
    parser.add_argument("--output-video-key", default=R2_OUTPUT_VIDEO_KEY)
    parser.add_argument("--positive-prompt", default=DEFAULT_POSITIVE_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
