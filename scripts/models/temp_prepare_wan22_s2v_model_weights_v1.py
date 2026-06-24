import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LOCAL_WAN22_S2V_MODEL_WEIGHTS_PREPARE_V1"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPO_ROOT / "data" / "checkpoints" / "wan22_s2v" / "comfyui_models"
RAW_ROOT = REPO_ROOT / "data" / "checkpoints" / "wan22_s2v" / "raw"
LOG_PATH = REPO_ROOT / "logs" / "wan22_s2v_model_weights_prepare_v1_log.json"
DEPENDENCY_NOTE = "Install Hugging Face CLI first if needed: python3 -m pip install huggingface_hub"

WEIGHTS = (
    {
        "name": "transformer",
        "repo_id": "Kijai/WanVideo_comfy_fp8_scaled",
        "repo_file": "S2V/Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "download_local_dir": LOCAL_ROOT / "diffusion_models" / "WanVideo",
        "final_path": LOCAL_ROOT / "diffusion_models" / "WanVideo" / "S2V" / "Wan2_2-S2V-14B_fp8_e4m3fn_scaled_KJ.safetensors",
        "min_size_bytes": 1024 * 1024 * 1024,
        "prepare_action": "download_to_final_path",
    },
    {
        "name": "vae",
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_file": "Wan2_1_VAE_bf16.safetensors",
        "download_local_dir": LOCAL_ROOT / "vae" / "wanvideo",
        "final_path": LOCAL_ROOT / "vae" / "wanvideo" / "Wan2_1_VAE_bf16.safetensors",
        "min_size_bytes": 100 * 1024 * 1024,
        "prepare_action": "download_to_final_path",
    },
    {
        "name": "umt5",
        "repo_id": "Kijai/WanVideo_comfy",
        "repo_file": "umt5-xxl-enc-bf16.safetensors",
        "download_local_dir": LOCAL_ROOT / "text_encoders",
        "final_path": LOCAL_ROOT / "text_encoders" / "umt5-xxl-enc-bf16.safetensors",
        "min_size_bytes": 1024 * 1024 * 1024,
        "prepare_action": "download_to_final_path",
    },
    {
        "name": "wav2vec",
        "repo_id": "Wan-AI/Wan2.2-S2V-14B",
        "repo_file": "wav2vec2-large-xlsr-53-english/model.safetensors",
        "download_local_dir": RAW_ROOT,
        "downloaded_path": RAW_ROOT / "wav2vec2-large-xlsr-53-english" / "model.safetensors",
        "final_path": LOCAL_ROOT / "audio_encoders" / "wav2vec_xlsr_53_english_fp32.safetensors",
        "min_size_bytes": 100 * 1024 * 1024,
        "prepare_action": "download_then_copy_rename",
    },
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def size_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def ensure_directories() -> list[str]:
    dirs = {LOCAL_ROOT, RAW_ROOT}
    for weight in WEIGHTS:
        dirs.add(weight["download_local_dir"])
        dirs.add(weight["final_path"].parent)
    created_or_present = []
    for directory in sorted(dirs):
        directory.mkdir(parents=True, exist_ok=True)
        created_or_present.append(str(directory))
    return created_or_present


def run_hf_download(repo_id: str, repo_file: str, local_dir: Path) -> None:
    command = [
        "huggingface-cli",
        "download",
        repo_id,
        repo_file,
        "--local-dir",
        str(local_dir),
    ]
    subprocess.run(command, check=True)


def build_item(weight: dict) -> dict:
    final_path = weight["final_path"]
    downloaded_path = weight.get("downloaded_path", final_path)
    final_size = size_bytes(final_path)
    downloaded_size = size_bytes(downloaded_path)
    return {
        "name": weight["name"],
        "source_repo": weight["repo_id"],
        "source_file": weight["repo_file"],
        "download_local_dir": str(weight["download_local_dir"]),
        "downloaded_path": str(downloaded_path),
        "final_path": str(final_path),
        "prepare_action": weight["prepare_action"],
        "min_size_bytes": weight["min_size_bytes"],
        "downloaded_exists": downloaded_path.is_file(),
        "downloaded_size_bytes": downloaded_size,
        "final_exists": final_path.is_file(),
        "final_size_bytes": final_size,
        "final_size_ok": final_size >= weight["min_size_bytes"],
    }


def build_log(
    *,
    args: argparse.Namespace,
    status: str,
    directories: list[str],
    items: list[dict],
    problems: list[str],
    error: str = "",
) -> dict:
    execute_allowed = args.execute and args.confirm_download
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "problems": problems,
        "execute_allowed": execute_allowed,
        "dry_run": not execute_allowed,
        "local_root": str(LOCAL_ROOT),
        "raw_root": str(RAW_ROOT),
        "created_or_present_directories": directories,
        "no_upload": True,
        "no_runpod": True,
        "no_build_push": True,
        "not_latentsync": True,
        "not_wan27": True,
        "items": items,
    }


def validate_items(items: list[dict]) -> list[str]:
    problems = []
    for item in items:
        if not item["final_exists"]:
            problems.append(f"missing final file for {item['name']}: {item['final_path']}")
        elif not item["final_size_ok"]:
            problems.append(
                f"final file below minimum size for {item['name']}: "
                f"{item['final_size_bytes']} < {item['min_size_bytes']}"
            )
    return problems


def prepare_weight(weight: dict) -> None:
    run_hf_download(weight["repo_id"], weight["repo_file"], weight["download_local_dir"])
    if weight["prepare_action"] == "download_then_copy_rename":
        source = weight["downloaded_path"]
        destination = weight["final_path"]
        if not source.is_file():
            raise RuntimeError(f"downloaded wav2vec source not found: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run(args: argparse.Namespace) -> int:
    directories = []
    items = []
    problems = []
    execute_allowed = args.execute and args.confirm_download
    try:
        directories = ensure_directories()
        items = [build_item(weight) for weight in WEIGHTS]
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} weights={len(items)}")
        print(f"[{TEST_ID}] LOCAL_ROOT {LOCAL_ROOT}")
        for item in items:
            print(
                f"[{TEST_ID}] SOURCE name={item['name']} repo={item['source_repo']} "
                f"file={item['source_file']}"
            )
            print(
                f"[{TEST_ID}] LOCAL name={item['name']} exists={str(item['final_exists']).lower()} "
                f"size={item['final_size_bytes']} min_size={item['min_size_bytes']} path={item['final_path']}"
            )

        if args.execute and not args.confirm_download:
            problems.append("real download requires --execute --confirm-download")
            status = "blocked_before_download"
            write_json(LOG_PATH, build_log(args=args, status=status, directories=directories, items=items, problems=problems))
            print(f"[{TEST_ID}] {problems[-1]}")
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 1

        if not execute_allowed:
            status = "dry_run_ready"
            write_json(LOG_PATH, build_log(args=args, status=status, directories=directories, items=items, problems=[]))
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        for weight in WEIGHTS:
            print(f"[{TEST_ID}] DOWNLOAD name={weight['name']} repo={weight['repo_id']} file={weight['repo_file']}")
            prepare_weight(weight)
            item = build_item(weight)
            if not item["final_size_ok"]:
                problems.append(
                    f"final file below minimum size for {item['name']}: "
                    f"{item['final_size_bytes']} < {item['min_size_bytes']}"
                )
            print(
                f"[{TEST_ID}] PREPARED name={item['name']} exists={str(item['final_exists']).lower()} "
                f"size={item['final_size_bytes']} path={item['final_path']}"
            )

        items = [build_item(weight) for weight in WEIGHTS]
        problems.extend(validate_items(items))
        status = "succeeded" if not problems else "validation_failed"
        write_json(LOG_PATH, build_log(args=args, status=status, directories=directories, items=items, problems=problems))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        write_json(
            LOG_PATH,
            build_log(args=args, status="failed", directories=directories, items=items, problems=problems, error=message),
        )
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Wan2.2 S2V V1 minimum model weights locally. No R2 upload.")
    parser.add_argument("--execute", action="store_true", help="Download files locally only with --confirm-download.")
    parser.add_argument("--confirm-download", action="store_true", help="Required with --execute for real downloads.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
