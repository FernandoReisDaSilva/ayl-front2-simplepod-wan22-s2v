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
DEFAULT_VISUAL_PACKAGE = "clean-card-grammar"
DEFAULT_THUMBNAIL_STYLE = "host-right-grammar-contrast"
DEFAULT_PRIMARY_CHARACTER = "luca"
INITIAL_OUTFIT_STATUS = "draft"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTFITS_PATH = REPO_ROOT / "examples" / "wardrobe_luca_test_outfits.json"
CHARACTER_REFERENCE_FILENAMES = {
    "alex": "AYL_Character_Reference_Alex_EN.png",
    "fernando": "AYL_Character_Reference_Fernando_PT.png",
    "luca": "AYL_Character_Reference_Luca_IT.png",
    "mae": "AYL_Character_Reference_Mae_FR.png",
    "sofi": "AYL_Character_Reference_Sofi_ES.png",
}


def load_outfits(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of outfit tests in {path}")

    required_keys = {"outfit_id", "description", "test_goal"}
    for item in data:
        missing = required_keys - set(item)
        if missing:
            raise ValueError(f"Missing {sorted(missing)} in outfit item: {item}")
    return data


def build_queue(args: argparse.Namespace, outfits: list[dict]) -> dict:
    character_key = args.primary_character.lower()
    if character_key not in CHARACTER_REFERENCE_FILENAMES:
        supported = ", ".join(sorted(CHARACTER_REFERENCE_FILENAMES))
        raise ValueError(
            f"Unsupported primary character '{args.primary_character}'. "
            f"Supported characters: {supported}"
        )

    reference_filename = CHARACTER_REFERENCE_FILENAMES[character_key]
    job_root = PRODUCTION_ROOT / "04_video_jobs" / args.job_id
    wardrobe_root = job_root / "wardrobe"
    prompts_root = wardrobe_root / "prompts"
    reference_image_path = job_root / "character_sources" / reference_filename
    official_reference_image_path = (
        PRODUCTION_ROOT / "00_project_sources" / "character_references" / reference_filename
    )

    return {
        "job_id": args.job_id,
        "visual_package": args.visual_package,
        "thumbnail_style": args.thumbnail_style,
        "primary_character": args.primary_character,
        "reference_image_path": str(reference_image_path),
        "official_reference_image_path": str(official_reference_image_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "instruction": "Preserve character identity and change clothing only.",
        "outfits": [
            {
                "outfit_id": outfit["outfit_id"],
                "description": outfit["description"],
                "reference_image_path": str(reference_image_path),
                "official_reference_image_path": str(official_reference_image_path),
                "visual_package": args.visual_package,
                "thumbnail_style": args.thumbnail_style,
                "prompt_path": str(prompts_root / f"{outfit['outfit_id']}_prompt.txt"),
                "status": INITIAL_OUTFIT_STATUS,
                "test_goal": outfit["test_goal"],
            }
            for outfit in outfits
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a local wardrobe test queue for the Wan 2.1 pilot."
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    parser.add_argument("--visual-package", default=DEFAULT_VISUAL_PACKAGE)
    parser.add_argument("--thumbnail-style", default=DEFAULT_THUMBNAIL_STYLE)
    parser.add_argument("--primary-character", default=DEFAULT_PRIMARY_CHARACTER)
    parser.add_argument(
        "--outfits",
        type=Path,
        default=DEFAULT_OUTFITS_PATH,
        help="Path to a JSON list of outfit tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wardrobe_root = PRODUCTION_ROOT / "04_video_jobs" / args.job_id / "wardrobe"
    queue_path = wardrobe_root / "wardrobe_test_queue.json"

    wardrobe_root.mkdir(parents=True, exist_ok=True)
    outfits = load_outfits(args.outfits)
    queue = build_queue(args, outfits)
    queue_path.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wardrobe root: {wardrobe_root}")
    print(f"Queue written: {queue_path}")
    print(f"Outfits drafted: {len(outfits)}")


if __name__ == "__main__":
    main()
