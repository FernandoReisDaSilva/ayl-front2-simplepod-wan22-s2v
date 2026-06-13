import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.replicate import run_wan27_i2v as wan27  # noqa: E402


JOB_ID = "TEST_WARDROBE_WAN_0001"
CHARACTER_ID = "mae"
MODEL = wan27.MODEL
MODEL_FAMILY = wan27.MODEL_FAMILY
PROMPT_ROOT = REPO_ROOT / "video_base" / "replicate_inputs" / "idle_library_v1"
TEMP_MANIFEST_PATH = REPO_ROOT / "video_base" / "mae_idle_v1_temp_test_manifest.json"
DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-fernandoreisdasilva@gmail.com"
    / "Meu Drive"
)
JOB_ROOT = DRIVE_ROOT / "AYL_Production" / "04_video_jobs" / JOB_ID / "video_base"
DEFAULT_IMAGE_PATH = JOB_ROOT / "replicate_inputs" / "source_images" / "mae_wan_v3.png"
DEFAULT_OUTPUT_FOLDER = JOB_ROOT / "replicate_outputs" / "mae_idle_v1_temp_test"
DEFAULT_REVIEW_REEL_FOLDER = JOB_ROOT / "review_reels" / "mae_idle_v1_temp_test"
REVIEW_REEL_FILENAME = "mae_idle_v1_temp_test_review_reel.mp4"

STATES = [
    ("001", "neutral_present", "mae_idle_v1_001_neutral_present_wan27_prompt.txt"),
    ("002", "encouraging_wait", "mae_idle_v1_002_encouraging_wait_wan27_prompt.txt"),
    ("003", "challenge_focus", "mae_idle_v1_003_challenge_focus_wan27_prompt.txt"),
    ("004", "listening_check", "mae_idle_v1_004_listening_check_wan27_prompt.txt"),
    ("005", "thinking_pause", "mae_idle_v1_005_thinking_pause_wan27_prompt.txt"),
    ("006", "result_reveal", "mae_idle_v1_006_result_reveal_wan27_prompt.txt"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def base_manifest(args: argparse.Namespace, batch_status: str) -> dict:
    return {
        "job_id": JOB_ID,
        "test_id": "mae_idle_v1_temp_test",
        "character_id": CHARACTER_ID,
        "purpose": "Temporary Maé-only V1.1 idle prompt validation for alertness, quick blink, and no sleepy/bored expression.",
        "model": MODEL,
        "model_family": MODEL_FAMILY,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "batch_status": batch_status,
        "source_image_path": str(args.image_path),
        "prompt_root": str(PROMPT_ROOT),
        "output_folder": str(args.output_folder),
        "review_reel_folder": str(args.review_reel_folder),
        "review_reel_path": "",
        "settings": {
            "duration": args.duration,
            "aspect_ratio": args.aspect_ratio,
            "resolution": args.resolution,
            "omit_last_frame": True,
            "enable_prompt_expansion": False,
        },
        "clips": [],
    }


def load_manifest(args: argparse.Namespace) -> dict:
    if TEMP_MANIFEST_PATH.exists():
        return json.loads(TEMP_MANIFEST_PATH.read_text(encoding="utf-8"))
    return base_manifest(args, "draft")


def clip_record(index: str, state_id: str, prompt_path: Path, args: argparse.Namespace) -> dict:
    run_label = f"mae_idle_v1_temp_test_{index}_{state_id}"
    return {
        "video_id": run_label,
        "character_id": CHARACTER_ID,
        "idle_clip_state": state_id,
        "source_image_path": str(args.image_path),
        "prompt_path": str(prompt_path),
        "output_folder": str(args.output_folder),
        "output_video_path": "",
        "run_log_path": str(args.output_folder / f"{run_label}_wan27_run_log.json"),
        "model": MODEL,
        "model_family": MODEL_FAMILY,
        "duration": args.duration,
        "aspect_ratio": args.aspect_ratio,
        "resolution": args.resolution,
        "omit_last_frame": True,
        "enable_prompt_expansion": False,
        "status": "draft",
        "notes": "Temporary Maé-only V1.1 idle prompt test.",
    }


def upsert_clip(manifest: dict, record: dict) -> None:
    clips = manifest.setdefault("clips", [])
    for index, item in enumerate(clips):
        if item.get("video_id") == record["video_id"]:
            clips[index] = record
            return
    clips.append(record)


def build_wan_args(
    args: argparse.Namespace,
    prompt_path: Path,
    run_label: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        image_path=args.image_path,
        prompt_path=prompt_path,
        prompt_inline="",
        run_label=run_label,
        output_folder=args.output_folder,
        duration=args.duration,
        aspect_ratio=args.aspect_ratio,
        resolution=args.resolution,
        audio_path=None,
        first_frame_param="first_frame",
        last_frame_param="last_frame",
        audio_param="audio",
        omit_last_frame=True,
        enable_prompt_expansion=False,
        omit_param=[],
        extra_input_json="",
        poll_interval=args.poll_interval,
        timeout_seconds=args.timeout_seconds,
        dry_run=args.dry_run,
        notes="Temporary Maé-only V1.1 idle clip test.",
    )


def run_state(
    args: argparse.Namespace,
    index: str,
    state_id: str,
    prompt_filename: str,
) -> dict:
    prompt_path = PROMPT_ROOT / prompt_filename
    record = clip_record(index, state_id, prompt_path, args)
    run_label = record["video_id"]
    run_log_path = Path(record["run_log_path"])
    wan_args = build_wan_args(args, prompt_path, run_label)

    try:
        wan27.require_file(args.image_path, "Input image")
        wan27.require_file(prompt_path, "Prompt file")
        args.output_folder.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            input_payload = wan27.build_input_payload(wan_args, stack)
            if args.dry_run:
                output_path = args.output_folder / wan27.output_filename(run_label, None)
                log = wan27.build_run_log(
                    wan_args,
                    prediction=None,
                    status="dry_run",
                    output_path=output_path,
                    notes="Dry run only. No Replicate request was sent.",
                    input_payload=input_payload,
                )
                write_json(run_log_path, log)
                record.update(
                    {
                        "output_video_path": str(output_path),
                        "status": "draft",
                        "notes": "Dry run only. No Replicate request was sent.",
                    }
                )
                return record

            print(f"[{index}/006] Starting {state_id}")
            print(f"[{index}/006] Payload field types: {wan27.payload_types_for_log(input_payload)}")
            prediction = wan27.create_prediction(input_payload)
            prediction_id = wan27.object_get(prediction, "id", "")
            print(f"[{index}/006] Prediction created: {prediction_id}")
            prediction = wan27.poll_prediction(
                prediction,
                interval_seconds=args.poll_interval,
                timeout_seconds=args.timeout_seconds,
            )

            status = str(wan27.object_get(prediction, "status", "")).lower()
            output_path = args.output_folder / wan27.output_filename(run_label, prediction_id)
            if status not in wan27.SUCCESS_STATUSES:
                error = wan27.object_get(prediction, "error", "")
                log = wan27.build_run_log(
                    wan_args,
                    prediction=prediction,
                    status=status or "failed",
                    output_path=output_path,
                    notes=f"Prediction did not succeed. Replicate error: {error}",
                    input_payload=input_payload,
                )
                write_json(run_log_path, log)
                record.update(
                    {
                        "output_video_path": str(output_path),
                        "replicate_prediction_id": prediction_id,
                        "status": status or "failed",
                        "notes": f"Replicate error: {error}",
                    }
                )
                return record

            wan27.download_output(wan27.object_get(prediction, "output", None), output_path)
            log = wan27.build_run_log(
                wan_args,
                prediction=prediction,
                status="needs_review",
                output_path=output_path,
                notes="Temporary Maé-only V1.1 idle clip downloaded for review.",
                input_payload=input_payload,
            )
            write_json(run_log_path, log)
            record.update(
                {
                    "output_video_path": str(output_path),
                    "replicate_prediction_id": prediction_id,
                    "status": "needs_review",
                    "notes": "Downloaded for temporary Maé-only review.",
                }
            )
            print(f"[{index}/006] Completed {state_id}: {output_path}")
            return record

    except Exception as exc:
        record.update({"status": "failed", "notes": str(exc)})
        print(f"[{index}/006] FAILED {state_id}: {exc}", file=sys.stderr)
        return record


def require_ffmpeg() -> None:
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg is required to create the Maé temporary review reel.")


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


def create_review_reel(args: argparse.Namespace, manifest: dict) -> bool:
    clips_by_state = {
        clip.get("idle_clip_state"): clip
        for clip in manifest.get("clips", [])
        if clip.get("status") in {"needs_review", "approved", "succeeded"}
    }
    ordered = []
    for _, state_id, _ in STATES:
        clip = clips_by_state.get(state_id)
        if not clip:
            print(f"Review reel skipped: missing completed state {state_id}.")
            return False
        output_video_path = Path(clip.get("output_video_path", ""))
        if not output_video_path.exists():
            print(f"Review reel skipped: missing MP4 for {state_id}: {output_video_path}")
            return False
        ordered.append((state_id, output_video_path))

    require_ffmpeg()
    args.review_reel_folder.mkdir(parents=True, exist_ok=True)
    review_reel_path = args.review_reel_folder / REVIEW_REEL_FILENAME
    with tempfile.TemporaryDirectory(prefix="mae_idle_v1_temp_reel_") as tmp:
        tmpdir = Path(tmp)
        pingpong_paths = []
        for number, (state_id, input_path) in enumerate(ordered, start=1):
            pingpong_path = tmpdir / f"{number:03d}_{state_id}_pingpong.mp4"
            ping_pong_clip(input_path, pingpong_path)
            pingpong_paths.append(pingpong_path)
        concat_clips(pingpong_paths, review_reel_path)

    manifest["review_reel_path"] = str(review_reel_path)
    manifest["review_reel_status"] = "needs_review"
    manifest["updated_at"] = now_iso()
    print(f"Review reel written: {review_reel_path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporary Maé-only Wan 2.7 V1.1 idle generation test."
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output-folder", type=Path, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--review-reel-folder", type=Path, default=DEFAULT_REVIEW_REEL_FOLDER)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-review-reel", action="store_true")
    parser.add_argument("--review-reel-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wan27.load_dotenv_if_available()

    if args.review_reel_only:
        manifest = load_manifest(args)
        ok = create_review_reel(args, manifest)
        write_json(TEMP_MANIFEST_PATH, manifest)
        return 0 if ok else 1

    manifest = base_manifest(args, "in_progress" if not args.dry_run else "draft")
    if not args.dry_run and not os.getenv("REPLICATE_API_TOKEN"):
        manifest["batch_status"] = "blocked"
        manifest["notes"] = "REPLICATE_API_TOKEN is not set. No Replicate requests were sent."
        write_json(TEMP_MANIFEST_PATH, manifest)
        print(
            "REPLICATE_API_TOKEN is not set. Export it before running this temporary Maé test.",
            file=sys.stderr,
        )
        print(f"Temp manifest written as blocked: {TEMP_MANIFEST_PATH}")
        return 1

    for index, state_id, prompt_filename in STATES:
        record = run_state(args, index, state_id, prompt_filename)
        upsert_clip(manifest, record)
        manifest["updated_at"] = now_iso()
        write_json(TEMP_MANIFEST_PATH, manifest)
        print(f"[{index}/006] Status: {record['status']}")

    failed = [clip for clip in manifest["clips"] if clip["status"] == "failed"]
    completed = [
        clip
        for clip in manifest["clips"]
        if clip["status"] in {"needs_review", "approved", "succeeded"}
    ]
    manifest["batch_status"] = "needs_review" if len(completed) == len(STATES) else "blocked"

    if not args.skip_review_reel and len(completed) == len(STATES):
        try:
            create_review_reel(args, manifest)
        except Exception as exc:
            manifest["review_reel_status"] = "blocked"
            manifest["review_reel_notes"] = str(exc)
            print(f"Review reel creation failed: {exc}", file=sys.stderr)

    manifest["updated_at"] = now_iso()
    write_json(TEMP_MANIFEST_PATH, manifest)

    print("\nSummary")
    print(f"- Completed: {len(completed)}")
    print(f"- Failed: {len(failed)}")
    print(f"- Temp manifest: {TEMP_MANIFEST_PATH}")
    if manifest.get("review_reel_path"):
        print(f"- Review reel: {manifest['review_reel_path']}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
