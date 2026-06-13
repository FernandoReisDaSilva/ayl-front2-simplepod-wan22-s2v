import argparse
import json
import os
import sys
import time
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


MODEL = "wan-video/wan-2.7-i2v"
MODEL_FAMILY = "Wan 2.7 image-to-video via Replicate API"
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "video_base" / "video_base_manifest.json"
SUCCESS_STATUSES = {"succeeded", "successful"}
FAILURE_STATUSES = {"failed", "canceled", "cancelled"}
AYL_REVIEW_STATUS = "needs_review"


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def derive_video_id(prompt_path: Path | None, run_label: str = "") -> str:
    if run_label:
        return run_label
    if prompt_path is None:
        raise ValueError("--run-label is required when using --prompt-inline without --prompt-path")
    stem = prompt_path.stem
    suffix = "_wan27_prompt"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def output_filename(video_id: str, prediction_id: str | None) -> str:
    if prediction_id:
        return f"{video_id}_{prediction_id}.mp4"
    return f"{video_id}.mp4"


def object_get(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def prediction_to_log_dict(prediction: object) -> dict:
    return {
        "id": object_get(prediction, "id", ""),
        "status": object_get(prediction, "status", ""),
        "model": object_get(prediction, "model", ""),
        "version": object_get(prediction, "version", ""),
        "error": object_get(prediction, "error", None),
        "logs": object_get(prediction, "logs", ""),
        "urls": object_get(prediction, "urls", {}),
        "created_at": object_get(prediction, "created_at", ""),
        "completed_at": object_get(prediction, "completed_at", ""),
        "metrics": object_get(prediction, "metrics", {}),
    }


def sanitize_input_for_log(input_payload: dict, args: argparse.Namespace) -> dict:
    sanitized = {}
    for key, value in input_payload.items():
        if key == args.first_frame_param:
            sanitized[key] = str(args.image_path)
        elif not args.omit_last_frame and key == args.last_frame_param:
            sanitized[key] = str(args.image_path)
        elif args.audio_path and key == args.audio_param:
            sanitized[key] = str(args.audio_path)
        else:
            sanitized[key] = value
    return sanitized


def payload_types_for_log(input_payload: dict) -> dict:
    return {key: type(value).__name__ for key, value in input_payload.items()}


def find_downloadable_output(output: object) -> object:
    if output is None:
        return None
    if hasattr(output, "read"):
        return output
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        for item in output:
            found = find_downloadable_output(item)
            if found is not None:
                return found
    if isinstance(output, dict):
        for key in ("video", "mp4", "output", "url", "file"):
            if key in output:
                found = find_downloadable_output(output[key])
                if found is not None:
                    return found
        for value in output.values():
            found = find_downloadable_output(value)
            if found is not None:
                return found
    if hasattr(output, "url"):
        return output
    return None


def download_output(output: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloadable = find_downloadable_output(output)
    if downloadable is None:
        raise RuntimeError(f"Prediction succeeded, but no downloadable MP4 output was found: {output!r}")

    if hasattr(downloadable, "read"):
        with output_path.open("wb") as file:
            for chunk in downloadable:
                if chunk:
                    file.write(chunk)
        return

    url = getattr(downloadable, "url", downloadable)
    if not isinstance(url, str) or not urlparse(url).scheme.startswith("http"):
        raise RuntimeError(f"Unsupported output shape; expected a URL or file-like output: {output!r}")

    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'requests'. Run: pip install replicate requests python-dotenv"
        ) from exc

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with output_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def build_input_payload(args: argparse.Namespace, stack: ExitStack) -> dict:
    if args.prompt_inline:
        prompt = args.prompt_inline.strip()
    elif args.prompt_path:
        prompt = args.prompt_path.read_text(encoding="utf-8").strip()
    else:
        raise ValueError("Provide either --prompt-path or --prompt-inline.")

    input_payload = {
        "prompt": prompt,
        args.first_frame_param: stack.enter_context(args.image_path.open("rb")),
        "duration": args.duration,
        "aspect_ratio": args.aspect_ratio,
        "resolution": args.resolution,
        "enable_prompt_expansion": args.enable_prompt_expansion,
    }

    if not args.omit_last_frame:
        input_payload[args.last_frame_param] = stack.enter_context(args.image_path.open("rb"))

    if args.audio_path:
        input_payload[args.audio_param] = stack.enter_context(args.audio_path.open("rb"))

    for key in args.omit_param:
        input_payload.pop(key, None)

    if args.extra_input_json:
        extra = json.loads(args.extra_input_json)
        if not isinstance(extra, dict):
            raise ValueError("--extra-input-json must be a JSON object")
        input_payload.update(extra)

    return input_payload


def create_prediction(input_payload: dict):
    try:
        import replicate
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'replicate'. Run: pip install replicate requests python-dotenv"
        ) from exc

    try:
        return replicate.predictions.create(model=MODEL, input=input_payload)
    except Exception as exc:
        raise RuntimeError(
            "Replicate rejected the prediction request. "
            "If the error names an unsupported input, rerun with "
            "--omit-param <name> or adjust --first-frame-param/--last-frame-param/--audio-param. "
            f"Original error: {exc}"
        ) from exc


def poll_prediction(prediction: object, interval_seconds: int, timeout_seconds: int):
    import replicate

    prediction_id = object_get(prediction, "id", "")
    if not prediction_id:
        return prediction

    deadline = time.monotonic() + timeout_seconds
    current = prediction
    while str(object_get(current, "status", "")).lower() not in SUCCESS_STATUSES | FAILURE_STATUSES:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Timed out waiting for Replicate prediction {prediction_id}. "
                "Check the run log or Replicate dashboard for the final state."
            )
        time.sleep(interval_seconds)
        current = replicate.predictions.get(prediction_id)
        print(f"Prediction {prediction_id}: {object_get(current, 'status', '')}")
    return current


def build_run_log(
    args: argparse.Namespace,
    prediction: object | None,
    status: str,
    output_path: Path | None,
    notes: str,
    input_payload: dict | None,
) -> dict:
    prediction_id = object_get(prediction, "id", "") if prediction is not None else ""
    return {
        "model": MODEL,
        "model_family": MODEL_FAMILY,
        "prediction_id": prediction_id,
        "prediction": prediction_to_log_dict(prediction) if prediction is not None else {},
        "input_image": str(args.image_path),
        "prompt_path": str(args.prompt_path) if args.prompt_path else "",
        "run_label": args.run_label,
        "audio_path": str(args.audio_path) if args.audio_path else "",
        "input": sanitize_input_for_log(input_payload, args) if input_payload else {},
        "input_types": payload_types_for_log(input_payload) if input_payload else {},
        "output_path": str(output_path) if output_path else "",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
    }


def update_manifest(args: argparse.Namespace, video_id: str, output_path: Path, prediction: object) -> None:
    manifest = load_json(
        MANIFEST_PATH,
        {
            "job_id": "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_family": MODEL_FAMILY,
            "videos": [],
        },
    )
    manifest["model_family"] = MODEL_FAMILY
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("videos", [])

    prediction_id = object_get(prediction, "id", "")
    record = None
    for item in manifest["videos"]:
        if item.get("video_id") == video_id:
            record = item
            break

    if record is None:
        record = {"video_id": video_id}
        manifest["videos"].append(record)

    record.update(
        {
            "video_id": video_id,
            "character_id": video_id.split("_", 1)[0],
            "source_image_path": str(args.image_path),
            "prompt_path": str(args.prompt_path) if args.prompt_path else "",
            "output_path": str(output_path),
            "model": MODEL,
            "model_family": MODEL_FAMILY,
            "prediction_id": prediction_id,
            "omit_last_frame": args.omit_last_frame,
            "enable_prompt_expansion": args.enable_prompt_expansion,
            "status": AYL_REVIEW_STATUS,
            "review_notes": "Replicate API output ready for Wan 2.7 review.",
        }
    )
    write_json(MANIFEST_PATH, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run wan-video/wan-2.7-i2v via Replicate API and save output metadata."
    )
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--prompt-inline", default="")
    parser.add_argument("--run-label", default="")
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--resolution", default="720p")
    parser.add_argument("--audio-path", type=Path, default=None)
    parser.add_argument("--first-frame-param", default="first_frame")
    parser.add_argument("--last-frame-param", default="last_frame")
    parser.add_argument("--audio-param", default="audio")
    parser.add_argument("--omit-last-frame", action="store_true")
    parser.add_argument("--enable-prompt-expansion", action="store_true")
    parser.add_argument("--omit-param", action="append", default=[])
    parser.add_argument("--extra-input-json", default="")
    parser.add_argument("--poll-interval", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv_if_available()

    try:
        require_file(args.image_path, "Input image")
        if args.prompt_path:
            require_file(args.prompt_path, "Prompt file")
        if args.prompt_inline and args.prompt_path:
            raise ValueError("Use either --prompt-path or --prompt-inline, not both.")
        if args.prompt_inline and not args.run_label:
            raise ValueError("--run-label is required when using --prompt-inline.")
        if not args.prompt_inline and not args.prompt_path:
            raise ValueError("Provide either --prompt-path or --prompt-inline.")
        if args.audio_path:
            require_file(args.audio_path, "Audio file")

        video_id = derive_video_id(args.prompt_path, args.run_label)
        args.output_folder.mkdir(parents=True, exist_ok=True)
        run_log_path = args.output_folder / f"{video_id}_wan27_run_log.json"

        with ExitStack() as stack:
            input_payload = build_input_payload(args, stack)
            if args.dry_run:
                output_path = args.output_folder / output_filename(video_id, None)
                log = build_run_log(
                    args,
                    prediction=None,
                    status="dry_run",
                    output_path=output_path,
                    notes="Dry run only. No Replicate request was sent.",
                    input_payload=input_payload,
                )
                write_json(run_log_path, log)
                print(f"Dry run log written: {run_log_path}")
                return 0

            if not os.getenv("REPLICATE_API_TOKEN"):
                raise RuntimeError(
                    "REPLICATE_API_TOKEN is not set. Export it in your shell or load it from .env."
                )

            print(f"Payload field types: {payload_types_for_log(input_payload)}")
            prediction = create_prediction(input_payload)
            prediction_id = object_get(prediction, "id", "")
            print(f"Prediction created: {prediction_id}")
            prediction = poll_prediction(
                prediction,
                interval_seconds=args.poll_interval,
                timeout_seconds=args.timeout_seconds,
            )

            status = str(object_get(prediction, "status", "")).lower()
            output_path = args.output_folder / output_filename(video_id, prediction_id)
            if status not in SUCCESS_STATUSES:
                error = object_get(prediction, "error", "")
                log = build_run_log(
                    args,
                    prediction=prediction,
                    status=status or "failed",
                    output_path=output_path,
                    notes=f"Prediction did not succeed. Replicate error: {error}",
                    input_payload=input_payload,
                )
                write_json(run_log_path, log)
                raise RuntimeError(
                    f"Replicate prediction {prediction_id} ended with status {status}. "
                    f"Error: {error}. Run log: {run_log_path}"
                )

            download_output(object_get(prediction, "output", None), output_path)
            update_manifest(args, video_id, output_path, prediction)
            log = build_run_log(
                args,
                prediction=prediction,
                status=AYL_REVIEW_STATUS,
                output_path=output_path,
                notes=args.notes or "Wan 2.7 I2V output downloaded and marked needs_review.",
                input_payload=input_payload,
            )
            write_json(run_log_path, log)
            print(f"Output written: {output_path}")
            print(f"Run log written: {run_log_path}")
            print(f"Manifest updated: {MANIFEST_PATH}")
            return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
