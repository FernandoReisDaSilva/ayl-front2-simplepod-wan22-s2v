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
DEFAULT_PROMPT_DEBUG_KEY = "tests/runpod_wan22_s2v_probe_v1/debug/prompt_payload_debug.json"
MAX_COMFYUI_ERROR_TEXT_CHARS = 50000

COMFYUI_PATH = Path(os.getenv("COMFYUI_PATH", "/opt/ComfyUI"))
WORKSPACE = Path("/workspace")
INPUT_DIR = WORKSPACE / "input"
OUTPUT_DIR = WORKSPACE / "output"
REFERENCE_PATH = INPUT_DIR / "mae_reference.png"
AUDIO_PATH = INPUT_DIR / "mae_audio_5s.wav"
OUTPUT_PATH = OUTPUT_DIR / "video_out.mp4"
PROMPT_DEBUG_PATH = WORKSPACE / "wan22_s2v_prompt_payload_debug.json"
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
DECORATIVE_NODE_TYPES = {
    "MarkdownNote",
    "Note",
    "AnythingEverywhere",
    "Reroute",
}
REPORT_ONLY_NON_DECORATIVE_NODE_TYPES = {
    "PrimitiveNode",
}


class ComfyPromptHTTPError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


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
        "R2_PROMPT_PAYLOAD_DEBUG_KEY",
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


def filter_ui_workflow_for_api(workflow: dict) -> tuple[dict, dict]:
    removed_ids = {
        str(node.get("id"))
        for node in workflow.get("nodes", [])
        if node.get("type") in DECORATIVE_NODE_TYPES
    }
    removed_nodes = []
    class_counts: dict[str, int] = {}
    preserved_non_decorative = set()
    filtered_nodes = []
    for node in workflow.get("nodes", []):
        node_id = str(node.get("id"))
        node_type = str(node.get("type", ""))
        if node_id in removed_ids:
            removed_nodes.append({"id": node_id, "class_type": node_type, "title": node.get("title", "")})
            class_counts[node_type] = class_counts.get(node_type, 0) + 1
            continue
        if node_type in REPORT_ONLY_NON_DECORATIVE_NODE_TYPES:
            preserved_non_decorative.add(node_type)
        filtered_nodes.append(node)

    dependent_links = []
    removed_link_ids = set()
    for link in workflow.get("links", []):
        if len(link) < 4:
            continue
        link_id = str(link[0])
        source_node_id = str(link[1])
        target_node_id = str(link[3])
        if source_node_id in removed_ids or target_node_id in removed_ids:
            removed_link_ids.add(link_id)
            dependent_links.append(
                {
                    "link_id": link_id,
                    "source_node_id": source_node_id,
                    "target_node_id": target_node_id,
                    "link": link,
                }
            )

    dependent_inputs = []
    for node in filtered_nodes:
        for item in node.get("inputs", []):
            link_id = item.get("link")
            if link_id is not None and str(link_id) in removed_link_ids:
                dependent_inputs.append(
                    {
                        "node_id": str(node.get("id")),
                        "class_type": node.get("type", ""),
                        "input_name": item.get("name", ""),
                        "link_id": str(link_id),
                    }
                )

    status = "error" if dependent_links or dependent_inputs else "ok"
    filtered_workflow = {**workflow, "nodes": filtered_nodes}
    if status == "ok":
        filtered_workflow["links"] = [
            link for link in workflow.get("links", []) if str(link[0]) not in removed_link_ids
        ]
    return filtered_workflow, {
        "workflow_filter_status": status,
        "workflow_filter_removed_nodes": removed_nodes,
        "workflow_filter_removed_class_type_counts": class_counts,
        "workflow_filter_preserved_non_decorative_node_classes": sorted(preserved_non_decorative),
        "workflow_filter_dependent_links": dependent_links,
        "workflow_filter_dependent_inputs": dependent_inputs,
    }


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
        if not node_type or node_type in DECORATIVE_NODE_TYPES:
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


def is_prompt_link(value) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str | int)


def prompt_links(prompt: dict) -> list[dict]:
    links = []
    for target_node_id, node in prompt.items():
        for input_name, value in node.get("inputs", {}).items():
            if is_prompt_link(value):
                links.append(
                    {
                        "source_node_id": str(value[0]),
                        "source_output_index": int(value[1]),
                        "target_node_id": str(target_node_id),
                        "target_input_name": input_name,
                    }
                )
    return links


def class_type(prompt: dict, node_id: str) -> str:
    return str(prompt.get(str(node_id), {}).get("class_type", ""))


def detect_melband_nodes(prompt: dict) -> dict:
    links = prompt_links(prompt)
    load_audio_nodes = [node_id for node_id, node in prompt.items() if node.get("class_type") == "VHS_LoadAudio"]
    audio_encoder_nodes = [node_id for node_id, node in prompt.items() if node.get("class_type") == "AudioEncoderEncode"]
    s2v_nodes = [node_id for node_id, node in prompt.items() if node.get("class_type") == "WanVideoAddS2VEmbeds"]
    melband_nodes = [
        node_id
        for node_id, node in prompt.items()
        if "MelBandRoFormer" in str(node.get("class_type", ""))
    ]
    normalize_audio_nodes = [
        node_id for node_id, node in prompt.items() if node.get("class_type") == "NormalizeAudioLoudness"
    ]
    return {
        "load_audio_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in load_audio_nodes],
        "audio_encoder_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in audio_encoder_nodes],
        "s2v_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in s2v_nodes],
        "melband_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in melband_nodes],
        "normalize_audio_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in normalize_audio_nodes],
        "melband_related_links": [
            link
            for link in links
            if link["source_node_id"] in set(melband_nodes + normalize_audio_nodes)
            or link["target_node_id"] in set(melband_nodes + normalize_audio_nodes)
        ],
    }


def melband_bypass_error(message: str, prompt: dict, detected: dict) -> dict:
    return {
        "melband_bypass_status": "error",
        "melband_bypass_error": message,
        "melband_bypass_detected_nodes": detected,
        "melband_bypass_detected_links": detected.get("melband_related_links", []),
        "melband_bypass_removed_nodes": [],
        "melband_bypass_removed_links": [],
        "melband_bypass_new_links": [],
        "melband_bypass_audio_source_node": "",
        "melband_bypass_audio_target_node": "",
        "melband_bypass_audio_target_input_name": "",
    }


def apply_melband_bypass(prompt: dict) -> tuple[dict, dict]:
    detected = detect_melband_nodes(prompt)
    melband_ids = {item["id"] for item in detected["melband_nodes"]}
    if not melband_ids:
        return prompt, {
            "melband_bypass_status": "not_needed",
            "melband_bypass_error": "",
            "melband_bypass_detected_nodes": detected,
            "melband_bypass_detected_links": [],
            "melband_bypass_removed_nodes": [],
            "melband_bypass_removed_links": [],
            "melband_bypass_new_links": [],
            "melband_bypass_audio_source_node": "",
            "melband_bypass_audio_target_node": "",
            "melband_bypass_audio_target_input_name": "",
        }

    load_audio_ids = [item["id"] for item in detected["load_audio_nodes"]]
    audio_encoder_ids = [item["id"] for item in detected["audio_encoder_nodes"]]
    if len(load_audio_ids) != 1:
        return prompt, melband_bypass_error("Expected exactly one VHS_LoadAudio node.", prompt, detected)
    if len(audio_encoder_ids) != 1:
        return prompt, melband_bypass_error("Expected exactly one AudioEncoderEncode node.", prompt, detected)

    audio_source_id = load_audio_ids[0]
    audio_target_id = audio_encoder_ids[0]
    audio_target = prompt.get(audio_target_id, {})
    audio_input_name = "audio"
    if audio_input_name not in audio_target.get("inputs", {}):
        return prompt, melband_bypass_error("AudioEncoderEncode input audio not found", prompt, detected)

    links = prompt_links(prompt)
    outgoing: dict[str, list[dict]] = {}
    for link in links:
        outgoing.setdefault(link["source_node_id"], []).append(link)

    removable_ids = set(melband_ids)
    queue = list(melband_ids)
    while queue:
        current = queue.pop(0)
        for link in outgoing.get(current, []):
            target_id = link["target_node_id"]
            target_class = class_type(prompt, target_id)
            if target_id == audio_target_id:
                continue
            if target_class in {"NormalizeAudioLoudness"} or "MelBandRoFormer" in target_class:
                if target_id not in removable_ids:
                    removable_ids.add(target_id)
                    queue.append(target_id)

    removed_nodes = [
        {"id": node_id, "class_type": class_type(prompt, node_id)}
        for node_id in sorted(removable_ids, key=lambda value: int(value) if value.isdigit() else value)
    ]
    removed_links = [
        link
        for link in links
        if link["source_node_id"] in removable_ids or link["target_node_id"] in removable_ids
    ]
    patched_prompt = {
        node_id: node
        for node_id, node in prompt.items()
        if node_id not in removable_ids
    }
    old_audio_input = patched_prompt[audio_target_id]["inputs"].get(audio_input_name)
    patched_prompt[audio_target_id]["inputs"][audio_input_name] = [audio_source_id, 0]
    new_link = {
        "source_node_id": audio_source_id,
        "source_output_index": 0,
        "target_node_id": audio_target_id,
        "target_input_name": audio_input_name,
        "replaced_value": old_audio_input,
    }
    remaining_bad_links = [
        link
        for link in prompt_links(patched_prompt)
        if link["source_node_id"] in removable_ids or link["target_node_id"] in removable_ids
    ]
    if remaining_bad_links:
        return prompt, melband_bypass_error(
            "Bypass left links connected to removed MelBand nodes.",
            prompt,
            {**detected, "remaining_bad_links": remaining_bad_links},
        )
    return patched_prompt, {
        "melband_bypass_status": "ok",
        "melband_bypass_error": "",
        "melband_bypass_detected_nodes": detected,
        "melband_bypass_detected_links": detected.get("melband_related_links", []),
        "melband_bypass_removed_nodes": removed_nodes,
        "melband_bypass_removed_links": removed_links,
        "melband_bypass_new_links": [new_link],
        "melband_bypass_audio_source_node": audio_source_id,
        "melband_bypass_audio_target_node": audio_target_id,
        "melband_bypass_audio_target_input_name": audio_input_name,
    }


def is_gimmvfi_class(class_name: str) -> bool:
    lower_name = class_name.lower()
    return "gimmvfi" in lower_name or "vfi" in lower_name


def detect_gimmvfi_nodes(prompt: dict) -> dict:
    links = prompt_links(prompt)
    gimmvfi_nodes = [
        node_id
        for node_id, node in prompt.items()
        if is_gimmvfi_class(str(node.get("class_type", "")))
    ]
    video_combine_nodes = [
        node_id for node_id, node in prompt.items() if node.get("class_type") == "VHS_VideoCombine"
    ]
    interpolation_nodes = [
        node_id
        for node_id, node in prompt.items()
        if "interpol" in str(node.get("class_type", "")).lower()
    ]
    select_every_nth_nodes = [
        node_id for node_id, node in prompt.items() if node.get("class_type") == "VHS_SelectEveryNthImage"
    ]
    related_ids = set(gimmvfi_nodes + interpolation_nodes + select_every_nth_nodes)
    return {
        "gimmvfi_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in gimmvfi_nodes],
        "interpolation_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in interpolation_nodes],
        "select_every_nth_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in select_every_nth_nodes],
        "video_combine_nodes": [{"id": node_id, "class_type": class_type(prompt, node_id)} for node_id in video_combine_nodes],
        "gimmvfi_related_links": [
            link
            for link in links
            if link["source_node_id"] in related_ids or link["target_node_id"] in related_ids
        ],
    }


def gimmvfi_video_combine_candidates(prompt: dict, interpolated_combine_ids: set[str]) -> list[dict]:
    candidates = []
    links = prompt_links(prompt)
    for node_id, node in prompt.items():
        if node.get("class_type") != "VHS_VideoCombine":
            continue
        input_links = [link for link in links if link["target_node_id"] == node_id]
        candidates.append(
            {
                "id": node_id,
                "class_type": class_type(prompt, node_id),
                "is_interpolated_path": node_id in interpolated_combine_ids,
                "is_preferred_direct_node_97": node_id == "97",
                "input_links": input_links,
            }
        )
    return candidates


def gimmvfi_bypass_error(message: str, detected: dict, candidates: list[dict]) -> dict:
    return {
        "gimmvfi_bypass_status": "error",
        "gimmvfi_bypass_error": message,
        "gimmvfi_bypass_detected_nodes": detected,
        "gimmvfi_bypass_detected_links": detected.get("gimmvfi_related_links", []),
        "gimmvfi_bypass_video_combine_candidates": candidates,
        "gimmvfi_bypass_removed_nodes": [],
        "gimmvfi_bypass_removed_links": [],
        "gimmvfi_bypass_preserved_video_path": {},
        "gimmvfi_bypass_selected_video_combine_node": "",
    }


def apply_gimmvfi_bypass(prompt: dict) -> tuple[dict, dict]:
    detected = detect_gimmvfi_nodes(prompt)
    gimmvfi_ids = {item["id"] for item in detected["gimmvfi_nodes"]}
    if not gimmvfi_ids:
        return prompt, {
            "gimmvfi_bypass_status": "not_needed",
            "gimmvfi_bypass_error": "",
            "gimmvfi_bypass_detected_nodes": detected,
            "gimmvfi_bypass_detected_links": [],
            "gimmvfi_bypass_video_combine_candidates": gimmvfi_video_combine_candidates(prompt, set()),
            "gimmvfi_bypass_removed_nodes": [],
            "gimmvfi_bypass_removed_links": [],
            "gimmvfi_bypass_preserved_video_path": {},
            "gimmvfi_bypass_selected_video_combine_node": "",
        }

    links = prompt_links(prompt)
    outgoing: dict[str, list[dict]] = {}
    for link in links:
        outgoing.setdefault(link["source_node_id"], []).append(link)

    removable_ids = set(gimmvfi_ids)
    interpolated_combine_ids: set[str] = set()
    queue = list(gimmvfi_ids)
    while queue:
        current = queue.pop(0)
        for link in outgoing.get(current, []):
            target_id = link["target_node_id"]
            target_class = class_type(prompt, target_id)
            target_lower = target_class.lower()
            if target_class == "VHS_VideoCombine":
                interpolated_combine_ids.add(target_id)
                continue
            if (
                is_gimmvfi_class(target_class)
                or target_class == "VHS_SelectEveryNthImage"
                or "interpol" in target_lower
            ):
                if target_id not in removable_ids:
                    removable_ids.add(target_id)
                    queue.append(target_id)

    candidates = gimmvfi_video_combine_candidates(prompt, interpolated_combine_ids)
    direct_candidates = [item for item in candidates if not item["is_interpolated_path"]]
    selected_candidate = next((item for item in direct_candidates if item["id"] == "97"), None)
    if selected_candidate is None and len(direct_candidates) == 1:
        selected_candidate = direct_candidates[0]
    if selected_candidate is None:
        return prompt, gimmvfi_bypass_error(
            "Could not safely identify direct non-interpolated VHS_VideoCombine.",
            detected,
            candidates,
        )

    removable_ids.update(node_id for node_id in interpolated_combine_ids if node_id != selected_candidate["id"])
    removed_nodes = [
        {"id": node_id, "class_type": class_type(prompt, node_id)}
        for node_id in sorted(removable_ids, key=lambda value: int(value) if value.isdigit() else value)
    ]
    removed_links = [
        link
        for link in links
        if link["source_node_id"] in removable_ids or link["target_node_id"] in removable_ids
    ]
    patched_prompt = {
        node_id: node
        for node_id, node in prompt.items()
        if node_id not in removable_ids
    }
    remaining_bad_links = [
        link
        for link in prompt_links(patched_prompt)
        if link["source_node_id"] in removable_ids or link["target_node_id"] in removable_ids
    ]
    if remaining_bad_links:
        return prompt, gimmvfi_bypass_error(
            "Bypass left links connected to removed GIMMVFI nodes.",
            {**detected, "remaining_bad_links": remaining_bad_links},
            candidates,
        )

    selected_node_id = selected_candidate["id"]
    preserved_video_path = {
        "selected_video_combine": selected_candidate,
        "selected_video_combine_inputs": patched_prompt.get(selected_node_id, {}).get("inputs", {}),
        "logic": "preferred node 97 when it is not on the GIMMVFI interpolated path; otherwise the only non-interpolated VHS_VideoCombine",
    }
    return patched_prompt, {
        "gimmvfi_bypass_status": "ok",
        "gimmvfi_bypass_error": "",
        "gimmvfi_bypass_detected_nodes": detected,
        "gimmvfi_bypass_detected_links": detected.get("gimmvfi_related_links", []),
        "gimmvfi_bypass_video_combine_candidates": candidates,
        "gimmvfi_bypass_removed_nodes": removed_nodes,
        "gimmvfi_bypass_removed_links": removed_links,
        "gimmvfi_bypass_preserved_video_path": preserved_video_path,
        "gimmvfi_bypass_selected_video_combine_node": selected_node_id,
    }


def workflow_node_lookup(workflow: dict) -> dict[str, dict]:
    return {str(node.get("id")): node for node in workflow.get("nodes", [])}


def primitive_literal_from_workflow_node(node: dict) -> tuple[object, str, bool, str]:
    widgets = node.get("widgets_values")
    if isinstance(widgets, dict):
        for key in ("value", "num_frames", "number", "int", "float"):
            if key in widgets:
                return widgets[key], f"widgets_values.{key}", False, ""
        candidates = [(key, value) for key, value in widgets.items() if key != "videopreview"]
        if len(candidates) == 1:
            key, value = candidates[0]
            return value, f"widgets_values.{key}", False, ""
    if isinstance(widgets, list):
        scalar_values = [
            value
            for value in widgets
            if isinstance(value, str | int | float | bool) or value is None
        ]
        if len(scalar_values) == 1:
            return scalar_values[0], "widgets_values[0]", False, ""

    title = str(node.get("title", "")).strip().lower()
    if title == "num_frames":
        reason = "PrimitiveNode num_frames value not found in API payload; using V1 probe fallback 81"
        return env_int("WAN22_S2V_NUM_FRAMES", 81), "env:WAN22_S2V_NUM_FRAMES default 81", True, reason
    return None, "", False, ""


def detect_primitive_nodes(prompt: dict, workflow: dict) -> dict:
    workflow_nodes = workflow_node_lookup(workflow)
    links = prompt_links(prompt)
    primitive_ids = [node_id for node_id, node in prompt.items() if node.get("class_type") == "PrimitiveNode"]
    detected_nodes = []
    for node_id in primitive_ids:
        workflow_node = workflow_nodes.get(node_id, {})
        outgoing_links = [link for link in links if link["source_node_id"] == node_id]
        incoming_links = [link for link in links if link["target_node_id"] == node_id]
        detected_nodes.append(
            {
                "id": node_id,
                "class_type": class_type(prompt, node_id),
                "title": workflow_node.get("title", ""),
                "widgets_values": workflow_node.get("widgets_values"),
                "prompt_inputs": prompt.get(node_id, {}).get("inputs", {}),
                "outgoing_links": outgoing_links,
                "incoming_links": incoming_links,
            }
        )
    return {
        "primitive_nodes": detected_nodes,
        "primitive_links": [
            link
            for link in links
            if link["source_node_id"] in set(primitive_ids) or link["target_node_id"] in set(primitive_ids)
        ],
    }


def remaining_primitive_nodes(prompt: dict) -> list[dict]:
    return [
        {"id": node_id, "class_type": class_type(prompt, node_id)}
        for node_id, node in prompt.items()
        if node.get("class_type") == "PrimitiveNode"
    ]


def primitive_resolve_error(message: str, detected: dict, prompt: dict) -> dict:
    return {
        "primitive_resolve_status": "error",
        "primitive_resolve_error": message,
        "primitive_resolve_detected_nodes": detected.get("primitive_nodes", []),
        "primitive_resolve_detected_links": detected.get("primitive_links", []),
        "primitive_resolve_resolved_nodes": [],
        "primitive_resolve_replaced_inputs": [],
        "primitive_resolve_removed_links": [],
        "primitive_resolve_node_id": "",
        "primitive_resolve_title": "",
        "primitive_resolve_targets": [],
        "primitive_resolve_fallback_used": False,
        "primitive_resolve_fallback_reason": "",
        "primitive_resolve_remaining_primitive_nodes": remaining_primitive_nodes(prompt),
    }


def resolve_primitive_nodes(prompt: dict, workflow: dict) -> tuple[dict, dict]:
    detected = detect_primitive_nodes(prompt, workflow)
    primitive_nodes = detected["primitive_nodes"]
    if not primitive_nodes:
        return prompt, {
            "primitive_resolve_status": "not_needed",
            "primitive_resolve_error": "",
            "primitive_resolve_detected_nodes": [],
            "primitive_resolve_detected_links": [],
            "primitive_resolve_resolved_nodes": [],
            "primitive_resolve_replaced_inputs": [],
            "primitive_resolve_removed_links": [],
            "primitive_resolve_node_id": "",
            "primitive_resolve_title": "",
            "primitive_resolve_targets": [],
            "primitive_resolve_fallback_used": False,
            "primitive_resolve_fallback_reason": "",
            "primitive_resolve_remaining_primitive_nodes": [],
        }

    workflow_nodes = workflow_node_lookup(workflow)
    patched_prompt = {node_id: node for node_id, node in prompt.items()}
    resolved_nodes = []
    replaced_inputs = []
    removed_links = []
    fallback_used = False
    fallback_reason = ""
    for primitive in primitive_nodes:
        node_id = primitive["id"]
        if primitive["incoming_links"]:
            return prompt, primitive_resolve_error(f"PrimitiveNode {node_id} has incoming links.", detected, prompt)
        value, source, used_fallback, reason = primitive_literal_from_workflow_node(workflow_nodes.get(node_id, {}))
        if primitive["outgoing_links"] and not source:
            return prompt, primitive_resolve_error(f"Could not resolve literal value for PrimitiveNode {node_id}.", detected, prompt)
        if used_fallback:
            fallback_used = True
            fallback_reason = reason
        for link in primitive["outgoing_links"]:
            target_id = link["target_node_id"]
            input_name = link["target_input_name"]
            if target_id not in patched_prompt:
                return prompt, primitive_resolve_error(
                    f"PrimitiveNode {node_id} target node {target_id} is missing.",
                    detected,
                    prompt,
                )
            old_value = patched_prompt[target_id].get("inputs", {}).get(input_name)
            patched_prompt[target_id].setdefault("inputs", {})[input_name] = value
            replaced_inputs.append(
                {
                    "source_node_id": node_id,
                    "source_title": primitive.get("title", ""),
                    "literal_value": value,
                    "literal_source": source,
                    "target_node_id": target_id,
                    "target_class_type": class_type(prompt, target_id),
                    "target_input_name": input_name,
                    "replaced_value": old_value,
                }
            )
            removed_links.append(link)
        patched_prompt.pop(node_id, None)
        resolved_nodes.append(
            {
                "id": node_id,
                "class_type": "PrimitiveNode",
                "title": primitive.get("title", ""),
                "literal_value": value,
                "literal_source": source,
                "outgoing_link_count": len(primitive["outgoing_links"]),
            }
        )

    remaining = remaining_primitive_nodes(patched_prompt)
    if remaining:
        return prompt, {
            "primitive_resolve_status": "error",
            "primitive_resolve_error": "PrimitiveNode remained in final payload",
            "primitive_resolve_detected_nodes": detected.get("primitive_nodes", []),
            "primitive_resolve_detected_links": detected.get("primitive_links", []),
            "primitive_resolve_resolved_nodes": resolved_nodes,
            "primitive_resolve_replaced_inputs": replaced_inputs,
            "primitive_resolve_removed_links": removed_links,
            "primitive_resolve_node_id": "",
            "primitive_resolve_title": "",
            "primitive_resolve_targets": [],
            "primitive_resolve_fallback_used": fallback_used,
            "primitive_resolve_fallback_reason": fallback_reason,
            "primitive_resolve_remaining_primitive_nodes": remaining,
        }

    primary = next((node for node in resolved_nodes if node["id"] == "71"), resolved_nodes[0])
    primary_targets = [
        {
            "node_id": item["target_node_id"],
            "class_type": item["target_class_type"],
            "input_name": item["target_input_name"],
            "literal_value": item["literal_value"],
        }
        for item in replaced_inputs
        if item["source_node_id"] == primary["id"]
    ]
    return patched_prompt, {
        "primitive_resolve_status": "ok",
        "primitive_resolve_error": "",
        "primitive_resolve_detected_nodes": detected.get("primitive_nodes", []),
        "primitive_resolve_detected_links": detected.get("primitive_links", []),
        "primitive_resolve_resolved_nodes": resolved_nodes,
        "primitive_resolve_replaced_inputs": replaced_inputs,
        "primitive_resolve_removed_links": removed_links,
        "primitive_resolve_node_id": primary["id"],
        "primitive_resolve_title": primary.get("title", ""),
        "primitive_resolve_targets": primary_targets,
        "primitive_resolve_fallback_used": fallback_used,
        "primitive_resolve_fallback_reason": fallback_reason,
        "primitive_resolve_remaining_primitive_nodes": remaining,
    }


def object_input_spec(object_info: dict, class_name: str, input_name: str):
    class_info = object_info.get(class_name, {})
    inputs = class_info.get("input", {})
    for section in ("required", "optional"):
        values = inputs.get(section, {})
        if isinstance(values, dict) and input_name in values:
            return values[input_name]
    return None


def object_input_is_optional(object_info: dict, class_name: str, input_name: str) -> bool:
    class_info = object_info.get(class_name, {})
    optional_inputs = class_info.get("input", {}).get("optional", {})
    return isinstance(optional_inputs, dict) and input_name in optional_inputs


def input_options(object_info: dict, class_name: str, input_name: str) -> list:
    spec = object_input_spec(object_info, class_name, input_name)
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return spec[0]
    if isinstance(spec, tuple) and spec and isinstance(spec[0], list):
        return spec[0]
    return []


def prompt_sanitize_change(changes: list[dict], node_id: str, class_name: str, input_name: str, old_value, new_value, reason: str) -> None:
    if old_value == new_value:
        return
    changes.append(
        {
            "node_id": node_id,
            "class_type": class_name,
            "input_name": input_name,
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
        }
    )


def accepted_or_default(options: list, preferred: str, fallback: str = "") -> str:
    if preferred in options:
        return preferred
    if fallback and fallback in options:
        return fallback
    return str(options[0]) if options else preferred


def sanitize_path_like_inputs(prompt: dict, changes: list[dict]) -> None:
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        for input_name, value in list(node.get("inputs", {}).items()):
            if not isinstance(value, str) or "\\" not in value:
                continue
            lower_input = input_name.lower()
            is_path_like = "model" in lower_input or "lora" in lower_input or value.endswith(".safetensors")
            if not is_path_like:
                continue
            new_value = value.replace("\\", "/")
            node["inputs"][input_name] = new_value
            prompt_sanitize_change(changes, node_id, class_name, input_name, value, new_value, "normalize path separators")


def is_attention_control_input(input_name: str) -> bool:
    lower_name = input_name.lower()
    return any(
        token in lower_name
        for token in (
            "sage",
            "sage_attention",
            "use_sage",
            "use_sage_attention",
            "attention",
            "attention_mode",
            "attention_backend",
        )
    )


def safe_attention_value(current_value, options: list):
    safe_string_preferences = ("disabled", "sdpa", "pytorch", "torch", "flash_attn", "none", "off")
    if options:
        for preferred in (False, "disabled", "sdpa", "pytorch", "torch", "flash_attn", "none", "off"):
            if preferred in options:
                return preferred
        return None
    if isinstance(current_value, bool):
        return False
    if isinstance(current_value, str):
        return "sdpa"
    return None


def sanitize_sageattention_inputs(prompt: dict, object_info: dict, changes: list[dict]) -> dict:
    detected_inputs = []
    sage_changes = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        for input_name, old_value in list(node.get("inputs", {}).items()):
            if not is_attention_control_input(input_name):
                continue
            options = input_options(object_info, class_name, input_name)
            detected = {
                "node_id": node_id,
                "class_type": class_name,
                "input_name": input_name,
                "old_value": old_value,
                "object_info_options": options,
            }
            detected_inputs.append(detected)
            new_value = safe_attention_value(old_value, options)
            if new_value is None:
                continue
            node["inputs"][input_name] = new_value
            before = len(changes)
            prompt_sanitize_change(
                changes,
                node_id,
                class_name,
                input_name,
                old_value,
                new_value,
                "disable SageAttention / choose safe attention backend",
            )
            if len(changes) > before:
                sage_changes.append(changes[-1])

    remaining_enabled = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        for input_name, value in node.get("inputs", {}).items():
            if not is_attention_control_input(input_name):
                continue
            value_text = str(value).lower()
            if value is True or "sage" in value_text:
                remaining_enabled.append(
                    {
                        "node_id": node_id,
                        "class_type": class_name,
                        "input_name": input_name,
                        "value": value,
                    }
                )

    if sage_changes:
        policy = "payload_control_applied"
    elif detected_inputs:
        policy = "payload_control_detected_no_safe_option"
    else:
        policy = "no_payload_control_found"
    return {
        "sageattention_policy": policy,
        "sageattention_detected_inputs": detected_inputs,
        "sageattention_sanitize_changes": sage_changes,
        "sageattention_remaining_enabled_values": remaining_enabled,
    }


def safe_torch_precision_value(options: list) -> str:
    if "fp16" in options:
        return "fp16"
    non_fast = [str(item) for item in options if "fast" not in str(item).lower()]
    for preferred in ("bf16", "fp32", "fp8_e4m3fn", "fp8_e4m3fn_scaled"):
        if preferred in non_fast:
            return preferred
    return non_fast[0] if non_fast else ""


def sanitize_torch_precision_inputs(prompt: dict, object_info: dict, changes: list[dict]) -> dict:
    precision_changes = []
    detected = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        if class_name != "WanVideoModelLoader":
            continue
        inputs = node.setdefault("inputs", {})
        if "base_precision" not in inputs:
            continue
        old_value = inputs.get("base_precision")
        options = input_options(object_info, class_name, "base_precision")
        detected.append(
            {
                "node_id": node_id,
                "class_type": class_name,
                "input_name": "base_precision",
                "old_value": old_value,
                "object_info_options": options,
            }
        )
        if "fast" not in str(old_value).lower():
            continue
        new_value = safe_torch_precision_value(options) or "fp16"
        inputs["base_precision"] = new_value
        before = len(changes)
        prompt_sanitize_change(
            changes,
            node_id,
            class_name,
            "base_precision",
            old_value,
            new_value,
            "avoid fp16_fast path requiring torch allow_fp16_accumulation",
        )
        if len(changes) > before:
            precision_changes.append(changes[-1])

    remaining_fast = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        value = node.get("inputs", {}).get("base_precision")
        if value is not None and "fast" in str(value).lower():
            remaining_fast.append(
                {
                    "node_id": node_id,
                    "class_type": class_name,
                    "input_name": "base_precision",
                    "value": value,
                }
            )
    if precision_changes:
        policy = "payload_control_applied"
    elif detected:
        policy = "base_precision_detected_no_change_needed"
    else:
        policy = "no_base_precision_input_found"
    return {
        "torch_precision_policy": policy,
        "torch_precision_detected_inputs": detected,
        "torch_precision_sanitize_changes": precision_changes,
        "torch_precision_remaining_fast_values": remaining_fast,
    }


def sanitize_image_resize_inputs(prompt: dict, object_info: dict, changes: list[dict]) -> dict:
    image_resize_changes = []
    detected_nodes = []

    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        if class_name != "ImageResizeKJv2":
            continue
        inputs = node.setdefault("inputs", {})
        detected_nodes.append({"node_id": node_id, "class_type": class_name})

        old_device = inputs.get("device")
        if old_device not in {"cpu", "gpu"}:
            inputs["device"] = "gpu"
            before = len(changes)
            prompt_sanitize_change(
                changes,
                node_id,
                class_name,
                "device",
                old_device,
                "gpu",
                "force valid resize device",
            )
            if len(changes) > before:
                image_resize_changes.append(changes[-1])

        old_mask = inputs.get("mask")
        if isinstance(old_mask, str):
            before = len(changes)
            if object_input_is_optional(object_info, class_name, "mask"):
                del inputs["mask"]
                prompt_sanitize_change(
                    changes,
                    node_id,
                    class_name,
                    "mask",
                    old_mask,
                    "<removed>",
                    "remove optional mask string; ImageResizeKJv2 expects mask tensor or None",
                )
            else:
                inputs["mask"] = None
                prompt_sanitize_change(
                    changes,
                    node_id,
                    class_name,
                    "mask",
                    old_mask,
                    None,
                    "set invalid mask string to None; ImageResizeKJv2 expects mask tensor or None",
                )
            if len(changes) > before:
                image_resize_changes.append(changes[-1])

    remaining_invalid_mask_values = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        if class_name != "ImageResizeKJv2":
            continue
        value = node.get("inputs", {}).get("mask")
        if isinstance(value, str):
            remaining_invalid_mask_values.append(
                {
                    "node_id": node_id,
                    "class_type": class_name,
                    "input_name": "mask",
                    "value": value,
                }
            )

    if image_resize_changes:
        policy = "payload_control_applied"
    elif detected_nodes:
        policy = "image_resize_detected_no_change_needed"
    else:
        policy = "no_image_resize_node_found"
    return {
        "image_resize_policy": policy,
        "image_resize_detected_nodes": detected_nodes,
        "image_resize_sanitize_changes": image_resize_changes,
        "image_resize_remaining_invalid_mask_values": remaining_invalid_mask_values,
    }


def sanitize_prompt_values(prompt: dict, object_info: dict) -> tuple[dict, dict]:
    changes: list[dict] = []
    errors: list[str] = []

    sanitize_path_like_inputs(prompt, changes)

    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        inputs = node.setdefault("inputs", {})
        if class_name == "WanVideoModelLoader":
            old_value = inputs.get("model")
            new_value = "WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors"
            inputs["model"] = new_value
            prompt_sanitize_change(changes, node_id, class_name, "model", old_value, new_value, "force V1 transformer model")
        elif class_name == "WanVideoVAELoader":
            old_value = inputs.get("model_name")
            new_value = "wanvideo/Wan2_1_VAE_bf16.safetensors"
            inputs["model_name"] = new_value
            prompt_sanitize_change(changes, node_id, class_name, "model_name", old_value, new_value, "force V1 VAE model")
        elif class_name == "WanVideoLoraSelectMulti":
            for input_name, old_value in list(inputs.items()):
                if not input_name.startswith("lora_"):
                    continue
                options = input_options(object_info, class_name, input_name)
                if not options or old_value not in options:
                    new_value = "none"
                    inputs[input_name] = new_value
                    prompt_sanitize_change(
                        changes,
                        node_id,
                        class_name,
                        input_name,
                        old_value,
                        new_value,
                        "disable LoRA not included in V1 minimum",
                    )
        elif class_name == "WanVideoSampler":
            scheduler_options = input_options(object_info, class_name, "scheduler")
            old_scheduler = inputs.get("scheduler")
            new_scheduler = accepted_or_default(scheduler_options, "dpm++_sde")
            inputs["scheduler"] = new_scheduler
            prompt_sanitize_change(
                changes,
                node_id,
                class_name,
                "scheduler",
                old_scheduler,
                new_scheduler,
                "force valid scheduler after UI widget alignment",
            )
            old_riflex = inputs.get("riflex_freq_index")
            inputs["riflex_freq_index"] = 0
            prompt_sanitize_change(
                changes,
                node_id,
                class_name,
                "riflex_freq_index",
                old_riflex,
                0,
                "force integer riflex_freq_index after UI widget alignment",
            )

    image_resize_report = sanitize_image_resize_inputs(prompt, object_info, changes)
    sageattention_report = sanitize_sageattention_inputs(prompt, object_info, changes)
    torch_precision_report = sanitize_torch_precision_inputs(prompt, object_info, changes)
    remaining_suspect_values = []
    for node_id, node in prompt.items():
        class_name = str(node.get("class_type", ""))
        for input_name, value in node.get("inputs", {}).items():
            if isinstance(value, str) and ("<tr" in value.lower() or "<td" in value.lower() or "</" in value.lower()):
                remaining_suspect_values.append(
                    {
                        "node_id": node_id,
                        "class_type": class_name,
                        "input_name": input_name,
                        "value_truncated": value[:1000],
                        "reason": "html_string_value",
                    }
                )

    if remaining_suspect_values:
        errors.append("HTML-like string values remain in prompt inputs after sanitize.")
    if image_resize_report["image_resize_remaining_invalid_mask_values"]:
        errors.append("Invalid ImageResizeKJv2 mask string values remain after sanitize.")
    if torch_precision_report["torch_precision_remaining_fast_values"]:
        errors.append("Fast torch precision values remain in prompt inputs after sanitize.")
    return prompt, {
        "prompt_sanitize_status": "error" if errors else "ok",
        "prompt_sanitize_changes": changes,
        "prompt_sanitize_errors": errors,
        "prompt_sanitize_remaining_suspect_values": remaining_suspect_values,
        **image_resize_report,
        **sageattention_report,
        **torch_precision_report,
    }


def truncate_text(value: str, limit: int = MAX_COMFYUI_ERROR_TEXT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n... truncated {len(value) - limit} chars ..."


def prompt_payload_summary(payload: dict, debug_upload_status: str, debug_key: str) -> dict:
    prompt = payload.get("prompt", {})
    class_counts: dict[str, int] = {}
    for node in prompt.values():
        class_type = str(node.get("class_type", ""))
        if class_type:
            class_counts[class_type] = class_counts.get(class_type, 0) + 1
    return {
        "client_id": payload.get("client_id", ""),
        "prompt_node_count": len(prompt),
        "prompt_node_ids_sample": list(prompt.keys())[:50],
        "class_type_counts": class_counts,
        "payload_debug_local_path": str(PROMPT_DEBUG_PATH),
        "payload_debug_r2_key": debug_key,
        "payload_debug_upload_status": debug_upload_status,
        "payload_json_size_bytes": PROMPT_DEBUG_PATH.stat().st_size if PROMPT_DEBUG_PATH.exists() else 0,
    }


def response_json_or_none(response: requests.Response):
    try:
        return response.json()
    except Exception:
        return None


def relevant_response_headers(response: requests.Response) -> dict:
    relevant = {}
    for key, value in response.headers.items():
        lower_key = key.lower()
        if lower_key in {"content-type", "content-length", "server", "date"} or lower_key.startswith("x-"):
            relevant[key] = value
    return relevant


def queue_prompt(prompt: dict) -> str:
    payload = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    PROMPT_DEBUG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    debug_key = os.getenv("R2_PROMPT_PAYLOAD_DEBUG_KEY", DEFAULT_PROMPT_DEBUG_KEY)
    debug_upload_status = "not_attempted"
    try:
        upload_r2_file(PROMPT_DEBUG_PATH, debug_key)
        debug_upload_status = "ok"
    except Exception as exc:
        debug_upload_status = f"failed: {str(exc)[:500]}"
    response = requests.post(f"{COMFY_BASE}/prompt", json=payload, timeout=30)
    if response.status_code >= 400:
        diagnostics = {
            "comfyui_prompt_status_code": response.status_code,
            "comfyui_prompt_response_text": truncate_text(response.text),
            "comfyui_prompt_response_json": response_json_or_none(response),
            "comfyui_prompt_response_headers": relevant_response_headers(response),
            "comfyui_prompt_payload_summary": prompt_payload_summary(payload, debug_upload_status, debug_key),
        }
        raise ComfyPromptHTTPError(f"ComfyUI /prompt returned HTTP {response.status_code}", diagnostics)
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
        filtered_workflow, workflow_filter = filter_ui_workflow_for_api(workflow)
        report.update(workflow_filter)
        if workflow_filter["workflow_filter_status"] != "ok":
            report.update(
                {
                    "runtime_probe_status": "workflow_filter_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": "Decorative workflow node removal would break one or more links.",
                }
            )
            write_progress(mode, "workflow_filter_error", workflow_filter)
            return report

        prompt = patch_prompt(convert_ui_workflow_to_api(filtered_workflow, object_info))
        prompt, melband_bypass = apply_melband_bypass(prompt)
        report.update(melband_bypass)
        if melband_bypass["melband_bypass_status"] == "error":
            report.update(
                {
                    "runtime_probe_status": "melband_bypass_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": melband_bypass["melband_bypass_error"],
                }
            )
            write_progress(mode, "melband_bypass_error", melband_bypass)
            return report
        prompt, gimmvfi_bypass = apply_gimmvfi_bypass(prompt)
        report.update(gimmvfi_bypass)
        if gimmvfi_bypass["gimmvfi_bypass_status"] == "error":
            report.update(
                {
                    "runtime_probe_status": "gimmvfi_bypass_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": gimmvfi_bypass["gimmvfi_bypass_error"],
                }
            )
            write_progress(mode, "gimmvfi_bypass_error", gimmvfi_bypass)
            return report
        prompt, primitive_resolve = resolve_primitive_nodes(prompt, filtered_workflow)
        report.update(primitive_resolve)
        if primitive_resolve["primitive_resolve_status"] == "error":
            report.update(
                {
                    "runtime_probe_status": "primitive_resolve_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": primitive_resolve["primitive_resolve_error"],
                }
            )
            write_progress(mode, "primitive_resolve_error", primitive_resolve)
            return report
        prompt, prompt_sanitize = sanitize_prompt_values(prompt, object_info)
        report.update(prompt_sanitize)
        if prompt_sanitize["prompt_sanitize_status"] == "error":
            report.update(
                {
                    "runtime_probe_status": "prompt_sanitize_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": "; ".join(prompt_sanitize["prompt_sanitize_errors"])[:2000],
                }
            )
            write_progress(mode, "prompt_sanitize_error", prompt_sanitize)
            return report
        report["prompt_node_count"] = len(prompt)
        report["workflow_control_node_ids"] = {"sampler": 27, "s2v_embeds": 101, "image": 73, "audio": 94, "video_combine": [30, 97]}
        write_progress(mode, "comfyui_prompt_ready", {"prompt_node_count": len(prompt)})

        try:
            prompt_id = queue_prompt(prompt)
        except ComfyPromptHTTPError as exc:
            report.update(
                {
                    "runtime_probe_status": "comfyui_prompt_http_error",
                    "output_upload_status": "not_attempted",
                    "error_truncated": str(exc)[:2000],
                    **exc.diagnostics,
                }
            )
            write_progress(
                mode,
                "comfyui_prompt_http_error",
                {
                    "comfyui_prompt_status_code": exc.diagnostics.get("comfyui_prompt_status_code"),
                    "comfyui_prompt_payload_summary": exc.diagnostics.get("comfyui_prompt_payload_summary", {}),
                },
            )
            return report
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
