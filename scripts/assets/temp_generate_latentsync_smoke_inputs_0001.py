import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TEST_ID = "TEST_LATENTSYNC_SMOKE_INPUT_GENERATE_0001"
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "data" / "smoke_inputs" / "latentsync"
VIDEO_PATH = OUTPUT_DIR / "video.mp4"
AUDIO_PATH = OUTPUT_DIR / "audio.wav"
LOG_PATH = REPO_ROOT / "logs" / "latentsync_smoke_input_generate_0001_log.json"
DURATION_SECONDS = 4.0
VIDEO_SIZE = "512x512"
VIDEO_FPS = 25
AUDIO_SAMPLE_RATE = 16000


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


def run_command(command: list[str], timeout_seconds: float = 60) -> tuple[int, str, str]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    return completed.returncode, completed.stdout, completed.stderr


def ffprobe_duration(path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.exists():
        return None
    code, stdout, _ = run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        timeout_seconds=20,
    )
    if code != 0:
        return None
    try:
        return float(stdout.strip())
    except ValueError:
        return None


def file_facts(path: Path) -> dict:
    exists = path.exists()
    is_file = path.is_file() if exists else False
    return {
        "path": str(path),
        "exists": exists,
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else 0,
        "duration_seconds": ffprobe_duration(path) if is_file else None,
        "sha256": sha256_file(path) if is_file else "",
    }


def ffmpeg_paths() -> dict:
    return {
        "ffmpeg": shutil.which("ffmpeg") or "",
        "ffprobe": shutil.which("ffprobe") or "",
    }


def video_command() -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    face_filter = ",".join(
        [
            "drawbox=x=0:y=0:w=512:h=512:color=0x4b6478@1:t=fill",
            "drawbox=x=156:y=86:w=200:h=270:color=0xf2c29a@1:t=fill",
            "drawbox=x=176:y=116:w=160:h=55:color=0x5a382c@1:t=fill",
            "drawbox=x=206:y=203:w=28:h=22:color=0x202020@1:t=fill",
            "drawbox=x=278:y=203:w=28:h=22:color=0x202020@1:t=fill",
            "drawbox=x=250:y=232:w=12:h=38:color=0xc98970@1:t=fill",
            "drawbox=x=218:y=294:w=76:h=14:color=0x8e2f3f@1:t=fill",
            "drawbox=x=148:y=345:w=216:h=88:color=0x26384f@1:t=fill",
            "drawbox=x=121:y=154:w=34:h=98:color=0xf2c29a@1:t=fill",
            "drawbox=x=357:y=154:w=34:h=98:color=0xf2c29a@1:t=fill",
            "drawbox=x=0:y=438:w=512:h=74:color=0x1c2634@1:t=fill",
        ]
    )
    return [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c=0x4b6478:s={VIDEO_SIZE}:r={VIDEO_FPS}:d={DURATION_SECONDS}",
        "-vf",
        face_filter,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(VIDEO_PATH),
    ]


def audio_command() -> list[str]:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    return [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=220:duration={DURATION_SECONDS}:sample_rate={AUDIO_SAMPLE_RATE}",
        "-ac",
        "1",
        "-ar",
        str(AUDIO_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(AUDIO_PATH),
    ]


def build_log(args: argparse.Namespace, status: str, error: str = "", commands: dict | None = None) -> dict:
    execute_allowed = args.execute and args.confirm_generate
    return {
        "test_id": TEST_ID,
        "created_at": now_iso(),
        "status": status,
        "error": error,
        "execute_requested": args.execute,
        "confirm_generate": args.confirm_generate,
        "execute_allowed": execute_allowed,
        "dry_run": not execute_allowed,
        "overwrite": args.overwrite,
        "duration_seconds_requested": DURATION_SECONDS,
        "video_spec": {
            "path": str(VIDEO_PATH),
            "container": "mp4",
            "size": VIDEO_SIZE,
            "fps": VIDEO_FPS,
            "visual_source": "ffmpeg_lavfi_static_placeholder_face",
        },
        "audio_spec": {
            "path": str(AUDIO_PATH),
            "container": "wav",
            "channels": 1,
            "sample_rate_hz": AUDIO_SAMPLE_RATE,
            "audio_source": "ffmpeg_lavfi_sine_signal",
        },
        "ffmpeg": ffmpeg_paths(),
        "no_runpod": True,
        "no_r2_upload": True,
        "no_external_api": True,
        "commands": commands or {},
        "outputs": {
            "video": file_facts(VIDEO_PATH),
            "audio": file_facts(AUDIO_PATH),
        },
    }


def generate_file(kind: str, path: Path, command: list[str], overwrite: bool) -> tuple[str, str]:
    if path.exists() and not overwrite:
        return "already_exists_skipped", ""
    path.parent.mkdir(parents=True, exist_ok=True)
    code, stdout, stderr = run_command(command, timeout_seconds=120)
    if code != 0:
        return "failed", (stderr or stdout)[-2000:]
    return "generated", ""


def run(args: argparse.Namespace) -> int:
    execute_allowed = args.execute and args.confirm_generate
    commands = {
        "video": video_command(),
        "audio": audio_command(),
    }
    try:
        print(f"[{TEST_ID}] START dry_run={str(not execute_allowed).lower()} duration={DURATION_SECONDS:g}s")
        print(f"[{TEST_ID}] VIDEO exists={str(VIDEO_PATH.exists()).lower()} path={VIDEO_PATH}")
        print(f"[{TEST_ID}] AUDIO exists={str(AUDIO_PATH.exists()).lower()} path={AUDIO_PATH}")

        if not execute_allowed:
            status = "dry_run_ready"
            write_json(LOG_PATH, build_log(args, status, commands=commands))
            print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
            return 0

        paths = ffmpeg_paths()
        if not paths["ffmpeg"]:
            raise RuntimeError("ffmpeg is required but was not found in PATH")

        video_status, video_error = generate_file("video", VIDEO_PATH, commands["video"], args.overwrite)
        audio_status, audio_error = generate_file("audio", AUDIO_PATH, commands["audio"], args.overwrite)
        print(f"[{TEST_ID}] VIDEO {video_status} size={file_facts(VIDEO_PATH)['size_bytes']}")
        print(f"[{TEST_ID}] AUDIO {audio_status} size={file_facts(AUDIO_PATH)['size_bytes']}")

        errors = [message for message in (video_error, audio_error) if message]
        outputs = {"video": file_facts(VIDEO_PATH), "audio": file_facts(AUDIO_PATH)}
        all_exist = outputs["video"]["is_file"] and outputs["audio"]["is_file"]
        status = "succeeded" if all_exist and not errors else "failed"
        log = build_log(args, status, error=" | ".join(errors), commands=commands)
        log["generation_status"] = {"video": video_status, "audio": audio_status}
        write_json(LOG_PATH, log)
        print(f"[{TEST_ID}] DONE status={status} log={LOG_PATH}")
        return 0 if status == "succeeded" else 1
    except Exception as exc:
        message = str(exc)
        write_json(LOG_PATH, build_log(args, "failed", error=message, commands=commands))
        print(f"[{TEST_ID}] ERROR {message[:300]}", file=sys.stderr)
        print(f"[{TEST_ID}] DONE status=failed log={LOG_PATH}", file=sys.stderr)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or generate minimal LatentSync smoke input media.")
    parser.add_argument("--execute", action="store_true", help="Generate local smoke input media.")
    parser.add_argument("--confirm-generate", action="store_true", help="Required with --execute for real file generation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated media files.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
