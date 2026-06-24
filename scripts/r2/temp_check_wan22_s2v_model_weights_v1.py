import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


TEST_ID = "TEST_R2_WAN22_S2V_MODEL_WEIGHTS_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOG_PATH = REPO_ROOT / "logs" / "r2_wan22_s2v_model_weights_v1_log.json"
DEPENDENCY_NOTE = "python3 -m pip install boto3 python-dotenv"
R2_PREFIX = "checkpoints/wan22_s2v/comfyui_models/"

REQUIRED_ENV_VARS = ("R2_ENDPOINT", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET", "R2_REGION")

WEIGHT_INVENTORY = (
    {
        "id": "wan22_s2v_fp8_transformer",
        "status": "required_confirmed",
        "comfy_models_relative_path": "diffusion_models/WanVideo/S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "workflow_node_id": 22,
        "workflow_node_type": "WanVideoModelLoader",
        "workflow_widget_value": "WanVideo\\S2V\\Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "source_repo": "Kijai/WanVideo_comfy_fp8_scaled",
        "source_repo_path": "S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "expected_size_bytes": None,
        "notes": "Loader reads from ComfyUI/models/diffusion_models. Destination keeps the workflow subfolder WanVideo/S2V.",
    },
    {
        "id": "wan_vae_bf16",
        "status": "required_confirmed",
        "comfy_models_relative_path": "vae/wanvideo/Wan2_1_VAE_bf16.safetensors",
        "workflow_node_id": 38,
        "workflow_node_type": "WanVideoVAELoader",
        "workflow_widget_value": "wanvideo\\Wan2_1_VAE_bf16.safetensors",
        "source_repo": "Kijai/WanVideo_comfy",
        "source_repo_path": "Wan2_1_VAE_bf16.safetensors",
        "expected_size_bytes": None,
        "notes": "Loader reads from ComfyUI/models/vae. Destination keeps the workflow subfolder wanvideo.",
    },
    {
        "id": "umt5_text_encoder",
        "status": "required_confirmed",
        "comfy_models_relative_path": "text_encoders/umt5-xxl-enc-bf16.safetensors",
        "workflow_node_id": 67,
        "workflow_node_type": "WanVideoTextEncodeCached",
        "workflow_widget_value": "umt5-xxl-enc-bf16.safetensors",
        "source_repo": "Kijai/WanVideo_comfy",
        "source_repo_path": "umt5-xxl-enc-bf16.safetensors",
        "expected_size_bytes": None,
        "notes": "Loader reads from ComfyUI/models/text_encoders.",
    },
    {
        "id": "wav2vec_s2v_audio_encoder",
        "status": "required_workflow_unresolved_source",
        "comfy_models_relative_path": "text_encoders/wav2vec_xlsr_53_english_fp32.safetensors",
        "workflow_node_id": 65,
        "workflow_node_type": "AudioEncoderLoader",
        "workflow_widget_value": "wav2vec_xlsr_53_english_fp32.safetensors",
        "source_repo": "Wan-AI/Wan2.2-S2V-14B",
        "source_repo_path": "wav2vec2-large-xlsr-53-english/model.safetensors",
        "expected_size_bytes": None,
        "notes": "Workflow expects a single ComfyUI safetensors filename. The official Wan-AI repo exposes a wav2vec folder; conversion/rename must be confirmed before upload.",
    },
    {
        "id": "melband_roformer_vocal_separation",
        "status": "required_workflow_unresolved_loader",
        "comfy_models_relative_path": "audio_encoders/MelBandRoFormer/MelBandRoformer_fp16.safetensors",
        "workflow_node_id": 81,
        "workflow_node_type": "MelBandRoFormerModelLoader",
        "workflow_widget_value": "MelBandRoFormer\\MelBandRoformer_fp16.safetensors",
        "source_repo": "",
        "source_repo_path": "",
        "expected_size_bytes": None,
        "notes": "Workflow references this node/model, but the loader source was not found in the current Docker custom-node set. Confirm custom node and model folder before paid run.",
    },
    {
        "id": "lightx2v_lora_rank64",
        "status": "optional_or_workflow_aux",
        "comfy_models_relative_path": "loras/WanVideo/Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
        "workflow_node_id": 60,
        "workflow_node_type": "WanVideoLoraSelectMulti",
        "workflow_widget_value": "WanVideo\\Lightx2v\\lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16_.safetensors",
        "source_repo": "Kijai/WanVideo_comfy",
        "source_repo_path": "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
        "expected_size_bytes": None,
        "notes": "Workflow widget appears to include a trailing underscore variant. The current graph has no outgoing links from the LoRA selection path in the inspected workflow; keep as auxiliary until confirmed.",
    },
    {
        "id": "gimmvfi_interpolator",
        "status": "optional_or_workflow_aux",
        "comfy_models_relative_path": "upscale_models/gimmvfi_r_arb_lpips_fp32.safetensors",
        "workflow_node_id": 95,
        "workflow_node_type": "DownloadAndLoadGIMMVFIModel",
        "workflow_widget_value": "gimmvfi_r_arb_lpips_fp32.safetensors",
        "source_repo": "",
        "source_repo_path": "",
        "expected_size_bytes": None,
        "notes": "Workflow uses this for interpolation. Node name suggests auto-download, but V1 should avoid runtime downloads; custom node source/model folder still needs confirmation.",
    },
)


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


def inventory_items() -> list[dict]:
    items = []
    for item in WEIGHT_INVENTORY:
        r2_key = R2_PREFIX + item["comfy_models_relative_path"]
        items.append({**item, "r2_key": r2_key, "container_path": "/opt/ComfyUI/models/" + item["comfy_models_relative_path"]})
    return items


def build_log(config: dict, items: list[dict], status: str, *, dry_run: bool, error: str = "") -> dict:
    required_items = [item for item in items if item["status"].startswith("required")]
    checked_items = [item for item in items if "r2_head" in item]
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "dry_run": dry_run,
        "r2_prefix": R2_PREFIX,
        "r2_endpoint_host": endpoint_host_only(config.get("endpoint", "")) if config else "",
        "r2_bucket_present": bool(config.get("bucket")) if config else False,
        "operation": "dry_run_inventory_only" if dry_run else "head_object_only",
        "no_download": True,
        "no_upload": True,
        "no_delete": True,
        "no_runpod": True,
        "not_latentsync": True,
        "not_wan27": True,
        "mirrors_comfyui_models_root": True,
        "comfyui_models_root_container": "/opt/ComfyUI/models",
        "inventory_count": len(items),
        "required_count": len(required_items),
        "all_required_exist": all(item.get("r2_head", {}).get("exists") for item in required_items) if checked_items else False,
        "items": items,
    }


def run(args: argparse.Namespace) -> int:
    config = {}
    items = inventory_items()
    dry_run = not args.execute
    try:
        if dry_run:
            print(f"[{TEST_ID}] START dry_run=true inventory_items={len(items)}")
            for item in items:
                print(f"[{TEST_ID}] PLANNED status={item['status']} key={item['r2_key']}")
            write_json(LOG_PATH, build_log(config, items, "dry_run_ready", dry_run=True))
            print(f"[{TEST_ID}] DONE status=dry_run_ready log={LOG_PATH}")
            return 0

        load_repo_dotenv()
        config = env_config()
        client = r2_client(config)
        print(f"[{TEST_ID}] START execute=true head_only=true inventory_items={len(items)}")
        for item in items:
            result = head_object(client, config["bucket"], item["r2_key"])
            item["r2_head"] = result
            print(f"[{TEST_ID}] {str(result['check_status']).upper()} status={item['status']} size={result['size_bytes']} key={item['r2_key']}")
        required_items = [item for item in items if item["status"].startswith("required")]
        all_required_exist = all(item.get("r2_head", {}).get("exists") for item in required_items)
        any_error = any(item.get("r2_head", {}).get("check_status") == "error" for item in items)
        status = "succeeded" if all_required_exist else "check_failed" if any_error else "missing_required_objects"
        write_json(LOG_PATH, build_log(config, items, status, dry_run=False))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if all_required_exist else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(config, items, "failed", dry_run=dry_run, error=message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory or HEAD-check Wan2.2 S2V ComfyUI model weights expected in R2.")
    parser.add_argument("--execute", action="store_true", help="Perform real R2 HEAD checks. No download/upload/delete.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
