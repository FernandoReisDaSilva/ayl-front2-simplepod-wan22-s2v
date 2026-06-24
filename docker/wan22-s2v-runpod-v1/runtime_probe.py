import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests


TEST_ID = "TEST_RUNPOD_WAN22_S2V_PROBE_V1"
DEFAULT_PROGRESS_KEY = "tests/runpod_wan22_s2v_probe_v1/progress/container_started.json"
DEFAULT_FINAL_KEY = "tests/runpod_wan22_s2v_probe_v1/output/final_report.json"
DEFAULT_REFERENCE_KEY = "tests/runpod_wan22_s2v_probe_v1/input/mae_reference.png"
DEFAULT_AUDIO_KEY = "tests/runpod_wan22_s2v_probe_v1/input/mae_audio_5s.wav"
DEFAULT_OUTPUT_KEY = "tests/runpod_wan22_s2v_probe_v1/output/video_out.mp4"
DEFAULT_MODEL_PREFIX = "checkpoints/wan22_s2v/comfyui_models/"

COMFYUI_PATH = Path(os.getenv("COMFYUI_PATH", "/opt/ComfyUI"))
WORKSPACE = Path("/workspace")
INPUT_DIR = WORKSPACE / "input"
OUTPUT_DIR = WORKSPACE / "output"
REFERENCE_PATH = INPUT_DIR / "mae_reference.png"
AUDIO_PATH = INPUT_DIR / "mae_audio_5s.wav"
OUTPUT_PATH = OUTPUT_DIR / "video_out.mp4"
WORKFLOW_PATH = Path(os.getenv("AYL_WAN22_S2V_WORKFLOW_PATH", "/opt/ayl/workflows/wanvideo2_2_S2V_context_window_testing.json"))
COMFY_HOST = os.getenv("AYL_COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("AYL_COMFY_PORT", "8188"))
COMFY_BASE = f"http://{COMFY_HOST}:{COMFY_PORT}"

R2_ENV_KEYS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")
REQUIRED_NODE_CLASSES = (
    "LoadImage",
    "VHS_LoadAudio",
    "WanVideoModelLoader",
    "WanVideoVAELoader",
    "WanVideoTextEncodeCached",
    "WanVideoSampler",
    "WanVideoAddS2VEmbeds",
    "VHS_VideoCombine",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "")
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_r2_env() -> None:
    missing = [key for key in R2_ENV_KEYS if not os.getenv(key, "")]
    if missing:
        raise RuntimeError("Missing required R2 env var(s): " + ", ".join(missing))


def r2_client():
    import boto3

    require_r2_env()
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name=os.environ["R2_REGION"],
    )


def upload_json(key: str, payload: dict) -> None:
    path = Path("/tmp") / (Path(key).name + ".json")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    r2_client().upload_file(str(path), os.environ["R2_BUCKET"], key)


def upload_r2_file(source: Path, key: str) -> None:
    r2_client().upload_file(str(source), os.environ["R2_BUCKET"], key)


def download_r2_file(key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    r2_client().download_file(os.environ["R2_BUCKET"], key, str(destination))


def download_r2_prefix(prefix: str, destination: Path) -> dict:
    client = r2_client()
    bucket = os.environ["R2_BUCKET"]
    normalized_prefix = prefix.rstrip("/") + "/"
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=normalized_prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key", ""))
            if not key or key.endswith("/"):
                continue
            target = destination / key[len(normalized_prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            downloaded.append({"key": key, "path": str(target), "size_bytes": int(item.get("Size", 0))})
    return {
        "prefix": normalized_prefix,
        "destination": str(destination),
        "file_count": len(downloaded),
        "total_size_bytes": sum(item["size_bytes"] for item in downloaded),
        "files_sample": downloaded[:25],
    }


def file_facts(path: Path) -> dict:
    return {"path": str(path), "exists": path.exists(), "size_bytes": path.stat().st_size if path.exists() else 0}


def env_presence() -> dict:
    keys = (
        "AYL_RUN_MODE",
        "AYL_IMAGE_TAG",
        "AYL_MARKER_NONCE",
        "R2_PROGRESS_KEY",
        "R2_FINAL_REPORT_KEY",
        "R2_INPUT_REFERENCE_IMAGE_KEY",
        "R2_INPUT_AUDIO_KEY",
        "R2_OUTPUT_VIDEO_KEY",
        "R2_WAN22_MODEL_PREFIX",
        "WAN22_S2V_CFG",
        "WAN22_S2V_SHIFT",
        "WAN22_S2V_SEED",
        "WAN22_S2V_STEPS",
        "WAN22_S2V_DENOISE_STRENGTH",
        "WAN22_S2V_AUDIO_SCALE",
        "WAN22_S2V_POSE_START_PERCENT",
        "WAN22_S2V_POSE_END_PERCENT",
        *R2_ENV_KEYS,
    )
    return {key: bool(os.getenv(key, "")) for key in keys}


def base_report(mode: str) -> dict:
    return {
        "test_id": TEST_ID,
        "mode": mode,
        "timestamp": now_iso(),
        "hostname": socket.gethostname(),
        "python_version": sys.version.replace("\n", " "),
        "cwd": str(Path.cwd()),
        "image_tag": os.getenv("AYL_IMAGE_TAG", ""),
        "marker_nonce": os.getenv("AYL_MARKER_NONCE", ""),
        "env_present_redacted": env_presence(),
        "network_volume_required": False,
        "dockerArgs_used": False,
        "not_latentsync": True,
        "not_wan27": True,
    }


def write_progress(mode: str, status: str, extra: dict | None = None) -> None:
    payload = {**base_report(mode), "status": status}
    if extra:
        payload.update(extra)
    upload_json(os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY), payload)


def torch_probe() -> dict:
    result = {"torch_import_status": "not_attempted", "torch_version": "", "cuda_available": False, "gpu_name": "", "error_truncated": ""}
    try:
        import torch

        result["torch_import_status"] = "ok"
        result["torch_version"] = getattr(torch, "__version__", "") or ""
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        result["torch_import_status"] = "failed"
        result["error_truncated"] = str(exc)[:1000]
    return result


def copy_inputs_to_comfy() -> None:
    comfy_input = COMFYUI_PATH / "input"
    comfy_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE_PATH, comfy_input / REFERENCE_PATH.name)
    shutil.copy2(AUDIO_PATH, comfy_input / AUDIO_PATH.name)


def start_comfy() -> subprocess.Popen:
    command = [
        sys.executable,
        "main.py",
        "--listen",
        COMFY_HOST,
        "--port",
        str(COMFY_PORT),
        "--disable-auto-launch",
    ]
    return subprocess.Popen(
        command,
        cwd=str(COMFYUI_PATH),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )


def wait_comfy(timeout_seconds: int = 120) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{COMFY_BASE}/object_info", timeout=5)
            if response.status_code == 200:
                return response.json()
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"ComfyUI did not become ready: {last_error[:500]}")


def validate_nodes(object_info: dict) -> dict:
    missing = [name for name in REQUIRED_NODE_CLASSES if name not in object_info]
    sampler_inputs = object_info.get("WanVideoSampler", {}).get("input", {})
    add_s2v_inputs = object_info.get("WanVideoAddS2VEmbeds", {}).get("input", {})
    return {
        "required_node_classes": list(REQUIRED_NODE_CLASSES),
        "missing_node_classes": missing,
        "sampler_control_fields": {
            "cfg": "cfg" in sampler_inputs.get("required", {}),
            "shift": "shift" in sampler_inputs.get("required", {}),
            "seed": "seed" in sampler_inputs.get("required", {}),
            "denoise_strength": "denoise_strength" in sampler_inputs.get("optional", {}),
        },
        "s2v_control_fields": {
            "audio_scale": "audio_scale" in add_s2v_inputs.get("required", {}),
            "pose_start_percent": "pose_start_percent" in add_s2v_inputs.get("required", {}),
            "pose_end_percent": "pose_end_percent" in add_s2v_inputs.get("required", {}),
        },
        "valid": not missing,
    }


def link_lookup(workflow: dict) -> dict[int, list]:
    return {int(link[0]): [str(link[1]), int(link[2])] for link in workflow.get("links", [])}


def input_order(class_info: dict) -> list[str]:
    info = class_info.get("input", {})
    names = []
    for section in ("required", "optional"):
        values = info.get(section, {})
        if isinstance(values, dict):
            names.extend(values.keys())
    return names


def convert_ui_workflow_to_api(workflow: dict, object_info: dict) -> dict:
    links = link_lookup(workflow)
    prompt = {}
    for node in workflow.get("nodes", []):
        node_type = node.get("type")
        if not node_type or node_type == "Note":
            continue
        node_id = str(node["id"])
        if node_type not in object_info:
            prompt[node_id] = {"class_type": node_type, "inputs": {}}
            continue
        inputs = {}
        linked_names = set()
        for item in node.get("inputs", []):
            if item.get("link") is not None:
                inputs[item["name"]] = links[int(item["link"])]
                linked_names.add(item["name"])
        widgets = node.get("widgets_values")
        if isinstance(widgets, dict):
            for key, value in widgets.items():
                if key != "videopreview":
                    inputs[key] = value
        elif isinstance(widgets, list):
            candidates = [name for name in input_order(object_info[node_type]) if name not in linked_names]
            for name, value in zip(candidates, widgets):
                inputs.setdefault(name, value)
        prompt[node_id] = {"class_type": node_type, "inputs": inputs}
    return prompt


def patch_prompt(prompt: dict) -> dict:
    prompt["73"]["inputs"]["image"] = REFERENCE_PATH.name
    prompt["94"]["inputs"]["audio_file"] = str(Path("input") / AUDIO_PATH.name)
    prompt["27"]["inputs"].update(
        {
            "steps": env_int("WAN22_S2V_STEPS", 4),
            "cfg": env_float("WAN22_S2V_CFG", 1.0),
            "shift": env_float("WAN22_S2V_SHIFT", 4.0),
            "seed": env_int("WAN22_S2V_SEED", 42),
            "denoise_strength": env_float("WAN22_S2V_DENOISE_STRENGTH", 1.0),
        }
    )
    prompt["101"]["inputs"].update(
        {
            "audio_scale": env_float("WAN22_S2V_AUDIO_SCALE", 1.0),
            "pose_start_percent": env_float("WAN22_S2V_POSE_START_PERCENT", 0.0),
            "pose_end_percent": env_float("WAN22_S2V_POSE_END_PERCENT", 1.0),
        }
    )
    for node_id in ("30", "97"):
        if node_id in prompt:
            prompt[node_id]["inputs"].update(
                {
                    "filename_prefix": "ayl_wan22_s2v_probe_v1/video_out",
                    "save_output": True,
                    "format": "video/h264-mp4",
                    "trim_to_audio": True,
                }
            )
    return prompt


def queue_prompt(prompt: dict) -> str:
    payload = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    response = requests.post(f"{COMFY_BASE}/prompt", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    if "prompt_id" not in data:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
    return str(data["prompt_id"])


def wait_prompt(prompt_id: str, timeout_seconds: int) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        response = requests.get(f"{COMFY_BASE}/history/{prompt_id}", timeout=10)
        response.raise_for_status()
        history = response.json()
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(5)
    raise TimeoutError(f"ComfyUI prompt timed out: {prompt_id}")


def find_output_video(history: dict) -> Path | None:
    candidates: list[Path] = []
    for node in history.get("outputs", {}).values():
        for item in node.get("gifs", []) + node.get("videos", []):
            filename = item.get("filename")
            subfolder = item.get("subfolder", "")
            if filename:
                candidates.append(COMFYUI_PATH / "output" / subfolder / filename)
    output_root = COMFYUI_PATH / "output"
    candidates.extend(sorted(output_root.glob("**/*.mp4"), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True))
    for path in candidates:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            return path
    return None


def build_report(mode: str) -> dict:
    report = base_report(mode)
    keys = {
        "reference_image": os.getenv("R2_INPUT_REFERENCE_IMAGE_KEY", DEFAULT_REFERENCE_KEY),
        "audio": os.getenv("R2_INPUT_AUDIO_KEY", DEFAULT_AUDIO_KEY),
        "output_video": os.getenv("R2_OUTPUT_VIDEO_KEY", DEFAULT_OUTPUT_KEY),
        "model_prefix": os.getenv("R2_WAN22_MODEL_PREFIX", DEFAULT_MODEL_PREFIX),
    }
    controls = {
        "cfg": env_float("WAN22_S2V_CFG", 1.0),
        "shift": env_float("WAN22_S2V_SHIFT", 4.0),
        "seed": env_int("WAN22_S2V_SEED", 42),
        "steps": env_int("WAN22_S2V_STEPS", 4),
        "denoise_strength": env_float("WAN22_S2V_DENOISE_STRENGTH", 1.0),
        "audio_scale": env_float("WAN22_S2V_AUDIO_SCALE", 1.0),
        "pose_start_percent": env_float("WAN22_S2V_POSE_START_PERCENT", 0.0),
        "pose_end_percent": env_float("WAN22_S2V_POSE_END_PERCENT", 1.0),
    }
    report.update({"r2_keys": keys, "wan22_s2v_controls": controls, "workflow_path": str(WORKFLOW_PATH)})

    torch_result = torch_probe()
    write_progress(mode, "gpu_check_done", {"torch_probe": torch_result})
    report["torch_probe"] = torch_result
    report["ffmpeg_exists"] = shutil.which("ffmpeg") is not None
    write_progress(mode, "ffmpeg_check_done", {"ffmpeg_exists": report["ffmpeg_exists"]})

    download_r2_file(keys["reference_image"], REFERENCE_PATH)
    download_r2_file(keys["audio"], AUDIO_PATH)
    copy_inputs_to_comfy()
    input_files = {"reference_image": file_facts(REFERENCE_PATH), "audio": file_facts(AUDIO_PATH)}
    report["input_files"] = input_files
    write_progress(mode, "input_download_done", {"input_files": input_files})

    model_download = download_r2_prefix(keys["model_prefix"], COMFYUI_PATH / "models")
    report["model_download"] = model_download
    write_progress(mode, "model_download_done", {"model_download": model_download})

    comfy_proc = start_comfy()
    report["comfyui_started"] = True
    try:
        object_info = wait_comfy(env_int("WAN22_S2V_COMFY_READY_TIMEOUT_SECONDS", 180))
        node_validation = validate_nodes(object_info)
        report["node_validation"] = node_validation
        write_progress(mode, "comfyui_object_info_validated", {"node_validation": node_validation})
        if not node_validation["valid"]:
            report.update({"runtime_probe_status": "missing_comfyui_nodes", "output_upload_status": "not_attempted"})
            return report

        workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
        prompt = patch_prompt(convert_ui_workflow_to_api(workflow, object_info))
        report["prompt_node_count"] = len(prompt)
        report["workflow_control_node_ids"] = {"sampler": 27, "s2v_embeds": 101, "image": 73, "audio": 94, "video_combine": [30, 97]}
        write_progress(mode, "comfyui_prompt_ready", {"prompt_node_count": len(prompt)})

        prompt_id = queue_prompt(prompt)
        report["comfyui_prompt_id"] = prompt_id
        write_progress(mode, "comfyui_prompt_queued", {"prompt_id": prompt_id})
        history = wait_prompt(prompt_id, env_int("WAN22_S2V_PROMPT_TIMEOUT_SECONDS", 1800))
        report["comfyui_history_status"] = history.get("status", {})
        write_progress(mode, "comfyui_prompt_done", {"prompt_id": prompt_id, "status": report["comfyui_history_status"]})

        output_video = find_output_video(history)
        if output_video is None:
            report.update({"runtime_probe_status": "video_output_missing", "output_upload_status": "not_attempted"})
            return report
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_video, OUTPUT_PATH)
        upload_r2_file(OUTPUT_PATH, keys["output_video"])
        report.update(
            {
                "runtime_probe_status": "ok",
                "output_file": file_facts(OUTPUT_PATH),
                "r2_output_video_key": keys["output_video"],
                "output_upload_status": "ok",
                "editorial_gate": {
                    "status": "manual_review_required",
                    "phonetic_mouth_score_required_min": 8,
                    "phonetic_mouth_score": None,
                    "head_motion_artifact": None,
                },
            }
        )
        write_progress(mode, "output_upload_done", {"output_file": report["output_file"], "r2_output_video_key": keys["output_video"]})
        return report
    finally:
        comfy_proc.terminate()
        try:
            comfy_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            comfy_proc.kill()


def run(mode: str) -> int:
    print(f"[AYL_WAN22_S2V_RUNTIME] start mode={mode}", flush=True)
    write_progress(mode, "container_started")
    final_key = os.getenv("R2_FINAL_REPORT_KEY", DEFAULT_FINAL_KEY)
    try:
        if mode != "wan22_s2v_probe":
            raise RuntimeError(f"Unsupported mode: {mode}")
        report = build_report(mode)
    except Exception as exc:
        report = base_report(mode)
        report.update({"runtime_probe_status": "failed", "error_truncated": str(exc)[:2000], "output_upload_status": "not_attempted"})
    report["r2_progress_key"] = os.getenv("R2_PROGRESS_KEY", DEFAULT_PROGRESS_KEY)
    report["r2_final_report_key"] = final_key
    report["r2_upload_status"] = "ok"
    upload_json(final_key, report)
    write_progress(mode, "final_report_written", {"r2_final_report_key": final_key, "runtime_probe_status": report.get("runtime_probe_status")})
    status = report.get("runtime_probe_status", "unknown")
    print(f"[AYL_WAN22_S2V_RUNTIME] done status={status}", flush=True)
    return 0 if status == "ok" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AYL Wan2.2 S2V ComfyUI RunPod runtime probe.")
    parser.add_argument("--mode", choices=("wan22_s2v_probe",), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args().mode))
