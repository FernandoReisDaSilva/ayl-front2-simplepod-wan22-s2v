import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-fernandoreisdasilva@gmail.com"
    / "Meu Drive"
)
PRODUCTION_ROOT = DRIVE_ROOT / "AYL_Production"
DEFAULT_JOB_ID = "TEST_WARDROBE_WAN_0001"
VALID_STATUSES = (
    "draft",
    "ready_for_next_stage",
    "in_progress",
    "needs_review",
    "approved",
    "blocked",
    "returned_for_revision",
    "final",
    "archived",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register metadata for a wardrobe test result."
    )
    parser.add_argument("outfit_id", help="Outfit test id, e.g. luca_test_001.")
    parser.add_argument(
        "--status",
        choices=VALID_STATUSES,
        default="needs_review",
        help="Review status for this wardrobe result.",
    )
    parser.add_argument(
        "--result-path",
        default="",
        help="Optional external media path or URI. The script does not copy media.",
    )
    parser.add_argument("--notes", default="", help="Optional review notes.")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wardrobe_root = PRODUCTION_ROOT / "04_video_jobs" / args.job_id / "wardrobe"
    results_root = wardrobe_root / "results"
    result_record_path = results_root / f"{args.outfit_id}_result.json"

    results_root.mkdir(parents=True, exist_ok=True)
    result_record = {
        "job_id": args.job_id,
        "outfit_id": args.outfit_id,
        "status": args.status,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "result_path": args.result_path,
        "notes": args.notes,
        "media_handling": "Metadata only. No generated media is copied into the repo.",
    }
    result_record_path.write_text(
        json.dumps(result_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wardrobe root: {wardrobe_root}")
    print(f"Result record written: {result_record_path}")
    if args.result_path:
        print(f"External result path: {args.result_path}")


if __name__ == "__main__":
    main()
