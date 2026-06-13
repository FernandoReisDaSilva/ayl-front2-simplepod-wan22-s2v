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
DEFAULT_OUTFIT_STATUS = "draft"


PROMPT_TEMPLATE = """TASK
This is an image-editing wardrobe test for {primary_character}.
The output must use the provided Luca reference image as the identity anchor: {reference_image_path}
Change clothing only.

CHARACTER IDENTITY LOCK
Preserve same character identity.
Preserve face, age impression, facial structure, skin tone, hairline, hairstyle, eyes, nose, mouth, jawline, facial proportions, and overall likeness.
Preserve the same realistic photographic style.
Do not make Luca look like a different person.
Do not make him look like a generic stock model.

ALLOWED CHANGE
Change only the outfit described in the wardrobe test.
Keep hair, face, age impression, body proportions, and identity unchanged.
Keep accessories absent unless explicitly requested.

OUTFIT
{description}

VISUAL PACKAGE COMPATIBILITY
visual_package: {visual_package}
thumbnail_style: {thumbnail_style}
Clean educational look.
Light or neutral background.
Simple studio-compatible background.
High readability.
No visual clutter.
Compatible with Remotion card-based layout.
Compatible with host-right composition.

FRAMING AND POSE
Chest-up or upper-torso framing.
Front-facing or slight 3/4 presenter angle.
Calm neutral presenter posture.
Shoulders visible.
Hands either out of frame or natural and not blocking the torso.
Face clearly visible.

LIGHTING AND BACKGROUND
Soft studio lighting.
Clean shadows.
No dramatic cinematic lighting.
No busy background.
No text in the background.
No logos.
No flags.
No national symbols.

WAN 2.1 READINESS
Output should be suitable as an image-to-video input for Wan 2.1.
Stable face.
Clear upper torso.
Clean outfit edges.
No motion blur.
No distorted hands.
No cropped-off head.
No extreme facial expression.

NEGATIVE INSTRUCTIONS
Do not change face.
Do not change age.
Do not change ethnicity or national identity.
Do not add logos.
Do not add text.
Do not add flags.
Do not add national costume.
Do not add extra people.
Do not add dramatic background.
Do not change hairstyle.
Do not change body type.
Do not over-stylize.
Do not create a cartoon or illustration.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create per-outfit wardrobe prompts from a test queue."
    )
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wardrobe_root = PRODUCTION_ROOT / "04_video_jobs" / args.job_id / "wardrobe"
    queue_path = wardrobe_root / "wardrobe_test_queue.json"
    prompts_root = wardrobe_root / "prompts"
    prompt_index_path = prompts_root / "outfit_prompt_index.json"

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    prompts_root.mkdir(parents=True, exist_ok=True)

    prompt_records = []
    for outfit in queue["outfits"]:
        outfit_id = outfit["outfit_id"]
        prompt = PROMPT_TEMPLATE.format(
            primary_character=queue["primary_character"],
            reference_image_path=outfit.get(
                "reference_image_path", queue.get("reference_image_path", "")
            ),
            description=outfit["description"],
            visual_package=outfit.get("visual_package", queue["visual_package"]),
            thumbnail_style=outfit.get("thumbnail_style", queue["thumbnail_style"]),
        ).strip()
        prompt_path = Path(outfit.get("prompt_path", prompts_root / f"{outfit_id}_prompt.txt"))
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        prompt_records.append(
            {
                "outfit_id": outfit_id,
                "description": outfit["description"],
                "test_goal": outfit.get("test_goal", ""),
                "reference_image_path": outfit.get(
                    "reference_image_path", queue.get("reference_image_path", "")
                ),
                "official_reference_image_path": outfit.get(
                    "official_reference_image_path",
                    queue.get("official_reference_image_path", ""),
                ),
                "visual_package": outfit.get("visual_package", queue["visual_package"]),
                "thumbnail_style": outfit.get(
                    "thumbnail_style", queue["thumbnail_style"]
                ),
                "prompt_path": str(prompt_path),
                "status": outfit.get("status", DEFAULT_OUTFIT_STATUS),
            }
        )

    prompt_index = {
        "job_id": queue["job_id"],
        "primary_character": queue["primary_character"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "identity_rule": "Preserve character identity and change clothing only.",
        "prompts": prompt_records,
    }
    prompt_index_path.write_text(
        json.dumps(prompt_index, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wardrobe root: {wardrobe_root}")
    print(f"Prompts directory: {prompts_root}")
    print(f"Prompt index written: {prompt_index_path}")
    print(f"Prompts written: {len(prompt_records)}")


if __name__ == "__main__":
    main()
