import importlib.util
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT_PATH = REPO_ROOT / "scripts" / "runpod" / "temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py"
LATENTSYNC_INFERENCE_STEPS = "50"
LATENTSYNC_GUIDANCE_SCALE = "1.5"
LATENTSYNC_ENABLE_DEEPCACHE = "0"


def load_smoke_module():
    spec = importlib.util.spec_from_file_location("ayl_latentsync_smoke_run_0001", SMOKE_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load smoke script: {SMOKE_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quality_inference_command() -> list[str]:
    return [
        "python",
        "-m",
        "scripts.inference",
        "--unet_config_path",
        "configs/unet/stage2_512.yaml",
        "--inference_ckpt_path",
        "checkpoints/latentsync_unet.pt",
        "--inference_steps",
        LATENTSYNC_INFERENCE_STEPS,
        "--guidance_scale",
        LATENTSYNC_GUIDANCE_SCALE,
        "--video_path",
        "/workspace/input/video.mp4",
        "--audio_path",
        "/workspace/input/audio.wav",
        "--video_out_path",
        "/workspace/output/video_out.mp4",
    ]


def configure_quality(module) -> None:
    module.TEST_ID = "TEST_RUNPOD_LATENTSYNC_QUALITY_V1_0001"
    module.OUTPUT_DIR = REPO_ROOT / "tmp" / "runpod_latentsync_quality_v1_0001"
    module.INTENDED_PAYLOAD_PATH = module.OUTPUT_DIR / "intended_payload.json"
    module.LOCAL_FINAL_REPORT_PATH = module.OUTPUT_DIR / "output" / "final_report.json"
    module.LOG_PATH = REPO_ROOT / "logs" / "runpod_latentsync_quality_v1_0001_log.json"
    module.DEFAULT_POD_NAME = "ayl-test-latentsync-quality-v1-0001"
    module.DEFAULT_IMAGE_TAG = "ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.8"
    module.DEFAULT_CONTAINER_DISK_GB = 40
    module.RUN_MODE = "latentsync_smoke_run"

    module.R2_PROGRESS_KEY = "tests/runpod_latentsync_quality_v1_0001/progress/container_started.json"
    module.R2_FINAL_REPORT_KEY = "tests/runpod_latentsync_quality_v1_0001/output/final_report.json"
    module.R2_OUTPUT_VIDEO_KEY = "tests/runpod_latentsync_quality_v1_0001/output/video_out.mp4"
    module.R2_INPUT_VIDEO_KEY = "tests/runpod_latentsync_quality_v1_0001/input/video.mp4"
    module.R2_INPUT_AUDIO_KEY = "tests/runpod_latentsync_quality_v1_0001/input/audio.wav"

    module.R2_CHECKPOINT_UNET_KEY = "checkpoints/latentsync/latentsync_unet.pt"
    module.R2_CHECKPOINT_WHISPER_KEY = "checkpoints/latentsync/whisper/tiny.pt"
    module.R2_VAE_CONFIG_KEY = "checkpoints/latentsync/vae/sd-vae-ft-mse/config.json"
    module.R2_VAE_SAFETENSORS_KEY = "checkpoints/latentsync/vae/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"

    original_redacted_env = module.redacted_env
    original_pod_env = module.pod_env
    original_intended_payload = module.intended_payload
    original_build_log = module.build_log
    original_run = module.run

    quality_env = {
        "LATENTSYNC_INFERENCE_STEPS": LATENTSYNC_INFERENCE_STEPS,
        "LATENTSYNC_GUIDANCE_SCALE": LATENTSYNC_GUIDANCE_SCALE,
        "LATENTSYNC_ENABLE_DEEPCACHE": LATENTSYNC_ENABLE_DEEPCACHE,
    }

    def quality_redacted_env() -> list[dict]:
        items = original_redacted_env()
        items.extend({"key": key, "value": value} for key, value in quality_env.items())
        return items

    def quality_pod_env(config, marker_nonce: str, args) -> list[dict]:
        items = original_pod_env(config, marker_nonce, args)
        items.extend({"key": key, "value": value} for key, value in quality_env.items())
        return items

    def quality_intended_payload(args, marker_nonce: str) -> dict:
        data = original_intended_payload(args, marker_nonce)
        data["quality_test"] = True
        data["max_wait_seconds"] = args.max_wait_seconds
        data["quality_variant"] = "V1A"
        data["latentsync_inference_steps"] = LATENTSYNC_INFERENCE_STEPS
        data["latentsync_guidance_scale"] = LATENTSYNC_GUIDANCE_SCALE
        data["latentsync_enable_deepcache"] = LATENTSYNC_ENABLE_DEEPCACHE
        data["inference_command"] = quality_inference_command()
        return data

    def quality_build_log(args, **values) -> dict:
        data = original_build_log(args, **values)
        data["quality_test"] = True
        data["max_wait_seconds"] = args.max_wait_seconds
        data["quality_variant"] = "V1A"
        data["latentsync_inference_steps"] = LATENTSYNC_INFERENCE_STEPS
        data["latentsync_guidance_scale"] = LATENTSYNC_GUIDANCE_SCALE
        data["latentsync_enable_deepcache"] = LATENTSYNC_ENABLE_DEEPCACHE
        data["expected_inference_command"] = quality_inference_command()
        return data

    def quality_run(args) -> int:
        if module.OUTPUT_DIR.exists():
            shutil.rmtree(module.OUTPUT_DIR)
        module.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return original_run(args)

    module.redacted_env = quality_redacted_env
    module.pod_env = quality_pod_env
    module.intended_payload = quality_intended_payload
    module.build_log = quality_build_log
    module.run = quality_run


def ensure_quality_container_disk_default() -> None:
    if "--container-disk-gb" not in sys.argv:
        sys.argv.extend(["--container-disk-gb", "40"])


def ensure_quality_timeout_default() -> None:
    if "--max-wait-seconds" not in sys.argv:
        sys.argv.extend(["--max-wait-seconds", "1800"])


def main() -> int:
    module = load_smoke_module()
    configure_quality(module)
    ensure_quality_container_disk_default()
    ensure_quality_timeout_default()
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
