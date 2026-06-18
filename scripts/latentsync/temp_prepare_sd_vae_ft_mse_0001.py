import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LATENTSYNC_SD_VAE_FT_MSE_PREPARE_0001"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "logs" / "latentsync_sd_vae_ft_mse_prepare_0001_log.json"
VAE_ROOT = REPO_ROOT / "data" / "checkpoints" / "latentsync" / "vae" / "sd-vae-ft-mse"
HF_REPO_ID = "stabilityai/sd-vae-ft-mse"
SOURCE_METHOD = "huggingface_hub.hf_hub_download_hf_transfer_disabled"
DEPENDENCY_NOTE = "python3 -m pip install huggingface-hub"

VAE_FILES = (
    {
        "name": "config",
        "repo_filename": "config.json",
        "local_path": VAE_ROOT / "config.json",
        "source_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/main/config.json",
    },
    {
        "name": "safetensors",
        "repo_filename": "diffusion_pytorch_model.safetensors",
        "local_path": VAE_ROOT / "diffusion_pytorch_model.safetensors",
        "source_url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/main/diffusion_pytorch_model.safetensors",
    },
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_facts(path: Path) -> dict:
    exists = path.exists()
    is_file = path.is_file() if exists else False
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else 0,
        "sha256": sha256_file(path) if is_file else "",
    }


def import_hf_hub_download():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(f"Missing dependency 'huggingface-hub'. Install it with: {DEPENDENCY_NOTE}") from exc
    return hf_hub_download


def vae_result(item: dict, *, download_attempted: bool, download_status: str) -> dict:
    return {
        "name": item["name"],
        "repo_id": HF_REPO_ID,
        "repo_filename": item["repo_filename"],
        "source_url": item["source_url"],
        "source_method": SOURCE_METHOD,
        "local": file_facts(item["local_path"]),
        "download_attempted": download_attempted,
        "download_status": download_status,
    }


def vae_status() -> list[dict]:
    return [vae_result(item, download_attempted=False, download_status="not_attempted") for item in VAE_FILES]


def build_log(args: argparse.Namespace, results: list[dict], status: str, error: str = "") -> dict:
    execute_allowed = args.execute and args.confirm_download
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "execute_requested": args.execute,
        "confirm_download": args.confirm_download,
        "execute_allowed": execute_allowed,
        "dry_run": not execute_allowed,
        "overwrite": args.overwrite,
        "hf_hub_enable_hf_transfer": "0",
        "no_runpod": True,
        "no_r2_upload": True,
        "vae_root": str(VAE_ROOT),
        "source_repo_id": HF_REPO_ID,
        "source_method": SOURCE_METHOD,
        "files": results,
    }


def download_file(hf_hub_download, item: dict, overwrite: bool) -> dict:
    destination = item["local_path"]
    if destination.exists() and not overwrite:
        return vae_result(item, download_attempted=False, download_status="already_exists_skipped")

    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=item["repo_filename"],
            repo_type="model",
        )
    )
    if destination.exists() and overwrite:
        destination.unlink()
    shutil.copy2(downloaded_path, destination)
    return vae_result(item, download_attempted=True, download_status="downloaded")


def run(args: argparse.Namespace) -> int:
    execute_allowed = args.execute and args.confirm_download
    results = vae_status()
    try:
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} files={len(results)}")
        for result in results:
            print(
                f"[{TEST_ID}] LOCAL exists={str(result['local']['exists']).lower()} "
                f"size={result['local']['size_bytes']} path={result['local']['path']}"
            )

        all_exist = all(item["local"]["exists"] and item["local"]["is_file"] for item in results)
        if not execute_allowed:
            status = "succeeded" if all_exist else "dry_run_ready"
            write_json(LOG_PATH, build_log(args, results, status))
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        if all_exist and not args.overwrite:
            updated_results = [vae_result(item, download_attempted=False, download_status="already_exists_skipped") for item in VAE_FILES]
            write_json(LOG_PATH, build_log(args, updated_results, "succeeded"))
            for updated in updated_results:
                print(f"[{TEST_ID}] {updated['download_status']} size={updated['local']['size_bytes']} path={updated['local']['path']}")
            print(f"[{TEST_ID}] DONE status=succeeded log={LOG_PATH}")
            return 0

        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        hf_hub_download = import_hf_hub_download()
        updated_results = []
        for item in VAE_FILES:
            updated = download_file(hf_hub_download, item, args.overwrite)
            updated_results.append(updated)
            print(f"[{TEST_ID}] {updated['download_status']} size={updated['local']['size_bytes']} path={updated['local']['path']}")

        final_all_exist = all(item["local"]["exists"] and item["local"]["is_file"] for item in updated_results)
        status = "succeeded" if final_all_exist else "failed_missing_after_download"
        write_json(LOG_PATH, build_log(args, updated_results, status))
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if final_all_exist else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(args, results, "failed", message))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or download the sd-vae-ft-mse files for LatentSync smoke testing.")
    parser.add_argument("--execute", action="store_true", help="Perform real local VAE downloads.")
    parser.add_argument("--confirm-download", action="store_true", help="Required with --execute for real downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing local VAE files.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
