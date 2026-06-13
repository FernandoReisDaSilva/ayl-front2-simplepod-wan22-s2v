import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "video_base" / "idle_clip_library_v1_manifest.json"
DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-fernandoreisdasilva@gmail.com"
    / "Meu Drive"
)
REVIEW_REEL_ROOT = (
    DRIVE_ROOT
    / "AYL_Production"
    / "04_video_jobs"
    / "TEST_WARDROBE_WAN_0001"
    / "video_base"
    / "review_reels"
    / "idle_library_v1"
)

CHARACTER_ORDER = ["alex", "sofi", "fernando", "mae", "luca"]
STATE_ORDER = [
    "neutral_present",
    "encouraging_wait",
    "challenge_focus",
    "listening_check",
    "thinking_pause",
    "result_reveal",
]


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def require_ffmpeg() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg is required to create review reels.")


def clips_for_character(manifest: dict, character_id: str) -> list[dict]:
    clips = [
        clip
        for clip in manifest.get("clips", [])
        if clip.get("character_id") == character_id and clip.get("status") == "succeeded"
    ]
    by_state = {clip["idle_clip_state"]: clip for clip in clips}
    return [by_state[state] for state in STATE_ORDER if state in by_state]


def ping_pong_clip(input_path: Path, output_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        "[0:v]split=2[f][r];[r]reverse[rr];[f][rr]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def concat_clips(inputs: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as file:
        list_path = Path(file.name)
        for input_path in inputs:
            file.write(f"file '{input_path.as_posix()}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
    finally:
        list_path.unlink(missing_ok=True)


def ensure_review_reels(manifest: dict) -> dict:
    return manifest.setdefault("review_reels", {})


def create_reel(manifest: dict, character_id: str, dry_run: bool) -> tuple[bool, str]:
    clips = clips_for_character(manifest, character_id)
    output_path = REVIEW_REEL_ROOT / f"{character_id}_idle_v1_review_reel.mp4"
    if len(clips) != len(STATE_ORDER):
        return False, f"Expected 6 succeeded clips for {character_id}, found {len(clips)}."

    if dry_run:
        return True, str(output_path)

    REVIEW_REEL_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{character_id}_idle_reel_") as tmp:
        tmpdir = Path(tmp)
        pingpong_paths = []
        for index, clip in enumerate(clips, start=1):
            input_path = Path(clip["output_video_path"])
            if not input_path.exists():
                return False, f"Missing clip file: {input_path}"
            pingpong_path = tmpdir / f"{index:03d}_{clip['idle_clip_state']}_pingpong.mp4"
            ping_pong_clip(input_path, pingpong_path)
            pingpong_paths.append(pingpong_path)
        concat_clips(pingpong_paths, output_path)

    return True, str(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create V1 idle review reels from succeeded clips.")
    parser.add_argument("--characters", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    characters = [
        item.strip()
        for item in args.characters.split(",")
        if item.strip()
    ] or CHARACTER_ORDER

    if not args.dry_run:
        require_ffmpeg()

    review_reels = ensure_review_reels(manifest)
    failed = []
    for character_id in characters:
        ok, message = create_reel(manifest, character_id, args.dry_run)
        review_reels[character_id] = {
            "review_reel_path": message if ok else "",
            "status": "draft" if ok and args.dry_run else ("succeeded" if ok else "blocked"),
            "notes": "Dry run only." if ok and args.dry_run else ("" if ok else message),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if ok:
            print(f"{character_id}: {message}")
        else:
            failed.append(f"{character_id}: {message}")
            print(f"{character_id}: {message}")

    write_manifest(manifest)
    if failed:
        print("Review reel creation incomplete:")
        for item in failed:
            print(f"- {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
