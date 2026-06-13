import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "video_base" / "idle_clip_library_v1_manifest.json"
REVIEW_PATH = REPO_ROOT / "review" / "wan27_review.md"
RUNNER_PATH = REPO_ROOT / "scripts" / "replicate" / "run_wan27_i2v.py"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def selected_clips(
    manifest: dict,
    retry_failed_only: bool,
    limit: int | None,
    characters: set[str],
) -> list[dict]:
    clips = manifest.get("clips", [])
    if retry_failed_only:
        return [clip for clip in clips if clip.get("status") == "failed"]
    if characters:
        clips = [clip for clip in clips if clip.get("character_id") in characters]
    if limit is not None:
        clips = clips[:limit]
    return clips


def run_clip(clip: dict) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        str(RUNNER_PATH),
        "--image-path",
        clip["source_image_path"],
        "--prompt-path",
        clip["prompt_path"],
        "--output-folder",
        clip["output_folder"],
        "--duration",
        str(clip.get("duration", 5)),
        "--aspect-ratio",
        clip.get("aspect_ratio", "1:1"),
        "--resolution",
        clip.get("resolution", "720p"),
        "--omit-last-frame",
    ]
    if clip.get("enable_prompt_expansion"):
        cmd.append("--enable-prompt-expansion")

    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    output = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, output


def update_clip_from_run_log(clip: dict, ok: bool, output: str) -> None:
    run_log_path = Path(clip["run_log_path"])
    if ok and run_log_path.exists():
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        clip["replicate_prediction_id"] = run_log.get("prediction_id", "")
        clip["output_video_path"] = run_log.get("output_path", clip["output_video_path"])
        clip["status"] = "succeeded"
        clip["notes"] = "Output downloaded; ready for review."
        return

    clip["status"] = "failed"
    clip["notes"] = output[-2000:] if output else "Runner failed without output."
    failure_log = {
        "model": clip.get("model", ""),
        "prediction_id": clip.get("replicate_prediction_id", ""),
        "input_image": clip["source_image_path"],
        "prompt_path": clip["prompt_path"],
        "output_path": clip.get("output_video_path", ""),
        "status": "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": clip["notes"],
    }
    write_json(run_log_path, failure_log)


def replace_section(text: str, title: str, body: str) -> str:
    start = f"## {title}"
    marker = "\n## "
    if start not in text:
        return text.rstrip() + f"\n\n{start}\n\n{body.strip()}\n"
    before, rest = text.split(start, 1)
    next_index = rest.find(marker)
    if next_index == -1:
        return before.rstrip() + f"\n\n{start}\n\n{body.strip()}\n"
    return before.rstrip() + f"\n\n{start}\n\n{body.strip()}\n" + rest[next_index:]


def update_review(manifest: dict, attempted: int, succeeded: int, failed: int) -> None:
    rows = "\n".join(
        f"| {clip['video_id']} | {clip['character_id']} | {clip['idle_clip_state']} | {clip['status']} | {clip.get('output_video_path', '')} |"
        for clip in manifest.get("clips", [])
    )
    failed_items = [
        clip["video_id"] for clip in manifest.get("clips", []) if clip.get("status") == "failed"
    ]
    failed_text = ", ".join(failed_items) if failed_items else "None"
    body = (
        "Run 11 is the approved reference baseline for V1 neutral idle quality.\n\n"
        "V1 batch of 30 idle clips launched.\n\n"
        "| video_id | character_id | idle_state | status | output |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{rows}\n\n"
        "### Results Summary\n\n"
        f"Attempted: {attempted}\n\n"
        f"Succeeded: {succeeded}\n\n"
        f"Failed: {failed}\n\n"
        f"Failed items: {failed_text}"
    )
    review = REVIEW_PATH.read_text(encoding="utf-8") if REVIEW_PATH.exists() else ""
    REVIEW_PATH.write_text(
        replace_section(review, "Idle Library V1 Batch", body),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V1 idle clip library sequentially.")
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--characters",
        default="",
        help="Comma-separated character ids, e.g. alex,sofi,luca.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    if not os.getenv("REPLICATE_API_TOKEN"):
        manifest["batch_status"] = "blocked"
        manifest["batch_blocked_reason"] = "REPLICATE_API_TOKEN is not set."
        write_manifest(manifest)
        update_review(manifest, attempted=0, succeeded=0, failed=0)
        print("ERROR: REPLICATE_API_TOKEN is not set. Batch blocked before submission.")
        print(f"Manifest: {MANIFEST_PATH}")
        return 2

    retry_failed_only = args.retry_failed_only or args.only_failed
    characters = {
        item.strip()
        for item in args.characters.split(",")
        if item.strip()
    }
    clips = selected_clips(manifest, retry_failed_only, args.limit, characters)
    attempted = 0
    succeeded = 0
    failed = 0
    manifest["batch_status"] = "running"
    manifest["batch_blocked_reason"] = ""
    write_manifest(manifest)

    for index, clip in enumerate(clips, start=1):
        attempted += 1
        clip["status"] = "submitted"
        clip["submitted_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(manifest)

        print(f"[{index}/{len(clips)}] Running {clip['video_id']}")
        ok, output = run_clip(clip)
        update_clip_from_run_log(clip, ok, output)
        if ok:
            succeeded += 1
            print(f"  succeeded: {clip['output_video_path']}")
        else:
            failed += 1
            print(f"  failed: {clip['notes']}")
        write_manifest(manifest)

    update_review(manifest, attempted, succeeded, failed)
    manifest["batch_status"] = "completed" if failed == 0 else "completed_with_failures"
    write_manifest(manifest)
    print("Batch complete")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Output folder: {manifest['clips'][0]['output_folder'] if manifest.get('clips') else ''}")
    print(f"Attempted: {attempted}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
