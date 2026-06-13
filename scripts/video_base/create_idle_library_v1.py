import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-fernandoreisdasilva@gmail.com"
    / "Meu Drive"
)
PRODUCTION_ROOT = DRIVE_ROOT / "AYL_Production"
JOB_ID = "TEST_WARDROBE_WAN_0001"
MODEL = "wan-video/wan-2.7-i2v"
MODEL_FAMILY = "Wan 2.7 image-to-video via Replicate API"
BLINK_DIRECTIVE = (
    "blinking is optional; at most one quick, alert blink during the full 5-second clip; "
    "eyes reopen immediately; do not blink repeatedly; "
    "if blinking looks unnatural, keep eyes open and alert"
)
ALERT_ENGAGEMENT_POSITIVE = (
    "Calm but alert. Attentive eyes. Eyes stay alert and engaged. "
    "Expression is calm but interested. Quiet energy. Engaged presence. "
    "Looks involved in the learner's response. "
    "Not overly excited, but clearly present. Blinking should be rare. "
    "Eyes stay engaged without excessive blinking."
)
ALERT_EYES_NEGATIVE = (
    "No tired expression. No bored expression. No sleepy eyes. No heavy eyelids. "
    "No disinterest. No passive stare. No drooping eyelids. No low-energy face. "
    "No slow blink. Do not blink repeatedly. Do not blink more than once. "
    "If blinking looks unnatural, keep eyes open and alert. Eyes reopen immediately after blink."
)
PROMPT_ROOT = REPO_ROOT / "video_base" / "replicate_inputs" / "idle_library_v1"
MANIFEST_PATH = REPO_ROOT / "video_base" / "idle_clip_library_v1_manifest.json"
REVIEW_PATH = REPO_ROOT / "review" / "wan27_review.md"
SOURCE_ROOT = (
    PRODUCTION_ROOT
    / "04_video_jobs"
    / JOB_ID
    / "video_base"
    / "replicate_inputs"
    / "source_images"
)
OUTPUT_ROOT = (
    PRODUCTION_ROOT
    / "04_video_jobs"
    / JOB_ID
    / "video_base"
    / "replicate_outputs"
    / "idle_library_v1"
)

CHARACTERS = {
    "alex": "alex_wan_v3.png",
    "sofi": "sofi_wan_v3.png",
    "fernando": "fernando_wan_v3.png",
    "mae": "mae_wan_v3.png",
    "luca": "luca_wan_v3.png",
}

STATES = [
    (
        1,
        "neutral_present",
        "calm but alert, attentive, pleasant",
        f"subtle breathing, {BLINK_DIRECTIVE}, tiny posture adjustment",
    ),
    (
        2,
        "encouraging_wait",
        "supportive, patient and engaged, present and attentive, gently encouraging",
        f"subtle breathing, soft eye focus, {BLINK_DIRECTIVE}, tiny approving micro-expression",
    ),
    (
        3,
        "challenge_focus",
        "focused, firm, calm but alert challenge energy",
        f"controlled but present breathing, focused eye contact, {BLINK_DIRECTIVE}, very small head micro-adjustment",
    ),
    (
        4,
        "listening_check",
        "actively listening with focused eyes, attentive, concentrated",
        f"subtle breathing, small eye movement, {BLINK_DIRECTIVE}, tiny listening posture",
    ),
    (
        5,
        "thinking_pause",
        "thoughtful, actively considering the answer, calm but alert, slightly analytical",
        f"minimal breathing, {BLINK_DIRECTIVE}, tiny eye movement, very slight posture settling",
    ),
    (
        6,
        "result_reveal",
        "subtle approval, calm but alert satisfaction",
        f"subtle breathing, {BLINK_DIRECTIVE}, tiny positive eye expression, minimal head movement",
    ),
]


def prompt_text(character_id: str, expression: str, movement: str) -> str:
    display_name = character_id.capitalize()
    return (
        f"Create a 5 second 1:1 idle character clip of {display_name} from the source image. "
        f"Preserve {display_name} identity and current wardrobe. Preserve clean educational framing. "
        f"{display_name} has this closed-mouth expression: {expression}. "
        "Lips remain fully closed the entire time. "
        "No speaking, no lip movement, no mouth opening, no visible teeth, no hand gestures, "
        "no raised hands, no arm movement, no face distortion.\n\n"
        f"{ALERT_ENGAGEMENT_POSITIVE} "
        f"Use controlled but present natural idle motion only: {movement}. "
        "Keep both arms and hands still and out of action. Do not gesture. Do not raise hands. "
        "Keep head movement small and natural. "
        f"{ALERT_EYES_NEGATIVE} "
        "The clip should feel alive and stable, not frozen, matching Luca Run 11 idle quality."
    )


def load_existing_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def existing_by_video_id(manifest: dict) -> dict:
    return {item["video_id"]: item for item in manifest.get("clips", [])}


def build_clip(character_id: str, image_name: str, state: tuple, existing: dict) -> dict:
    index, state_id, expression, movement = state
    video_id = f"{character_id}_idle_v1_{index:03d}_{state_id}"
    prompt_path = PROMPT_ROOT / f"{video_id}_wan27_prompt.txt"
    source_image_path = SOURCE_ROOT / image_name
    run_log_path = OUTPUT_ROOT / f"{video_id}_wan27_run_log.json"
    prior = existing.get(video_id, {})

    return {
        "video_id": video_id,
        "character_id": character_id,
        "source_image_path": str(source_image_path),
        "idle_clip_state": state_id,
        "prompt_path": str(prompt_path),
        "intended_use": "V1 reusable 1:1 character idle base asset for Remotion composition.",
        "expression_direction": expression,
        "movement_direction": movement,
        "output_folder": str(OUTPUT_ROOT),
        "output_video_path": prior.get(
            "output_video_path", str(OUTPUT_ROOT / f"{video_id}.mp4")
        ),
        "run_log_path": str(run_log_path),
        "replicate_prediction_id": prior.get("replicate_prediction_id", ""),
        "model": MODEL,
        "model_family": MODEL_FAMILY,
        "duration": 5,
        "aspect_ratio": "1:1",
        "resolution": "720p",
        "omit_last_frame": True,
        "enable_prompt_expansion": False,
        "status": prior.get("status", "draft"),
        "notes": prior.get("notes", "Prepared for V1 idle batch."),
    }


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


def update_review(clips: list[dict]) -> None:
    if REVIEW_PATH.exists():
        review = REVIEW_PATH.read_text(encoding="utf-8")
    else:
        review = "# Wan 2.7 Review - TEST_WARDROBE_WAN_0001\n"

    planned_rows = "\n".join(
        f"| {clip['video_id']} | {clip['character_id']} | {clip['idle_clip_state']} | {clip['status']} | {clip['output_video_path']} |"
        for clip in clips
    )
    body = (
        "Run 11 is the approved reference baseline for V1 neutral idle quality.\n\n"
        "V1 batch of 30 idle clips launched/planned.\n\n"
        "| video_id | character_id | idle_state | status | planned_output |\n"
        "| --- | --- | --- | --- | --- |\n"
        f"{planned_rows}\n\n"
        "### Results Summary\n\n"
        "Pending batch execution."
    )
    REVIEW_PATH.write_text(
        replace_section(review, "Idle Library V1 Batch", body),
        encoding="utf-8",
    )


def main() -> None:
    PROMPT_ROOT.mkdir(parents=True, exist_ok=True)
    existing_manifest = load_existing_manifest()
    prior = existing_by_video_id(existing_manifest)

    clips = []
    for character_id, image_name in CHARACTERS.items():
        for state in STATES:
            clip = build_clip(character_id, image_name, state, prior)
            prompt_path = Path(clip["prompt_path"])
            prompt_path.write_text(
                prompt_text(
                    character_id,
                    clip["expression_direction"],
                    clip["movement_direction"],
                )
                + "\n",
                encoding="utf-8",
            )
            clips.append(clip)

    manifest = {
        "job_id": JOB_ID,
        "library_id": "idle_library_v1",
        "created_at": existing_manifest.get(
            "created_at", datetime.now(timezone.utc).isoformat()
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "model_family": MODEL_FAMILY,
        "status_values": [
            "draft",
            "submitted",
            "succeeded",
            "failed",
            "needs_review",
            "approved",
            "archived",
        ],
        "reference_baseline": "Luca Run 11 approved idle quality.",
        "wan27_silent_idle_decision": {
            "approval": "Wan 2.7 is approved for V1 silent visual idle clips only.",
            "scope": "Short silent character presence clips for pause, repeat, listening, challenge, thinking, and reveal moments.",
            "out_of_scope": "Lip-sync, audio-driven video, spoken character clips, and final scaled production pipeline.",
            "replicate_role": "Replicate remains a test harness; future scaled production is expected to move to RunPod API.",
        },
        "official_silent_idle_states": [state_id for _, state_id, _, _ in STATES],
        "recommended_future_manifest_field": "silent_character_state",
        "blink_directive": BLINK_DIRECTIVE,
        "global_positive_guidance": ALERT_ENGAGEMENT_POSITIVE,
        "global_negative_guidance": ALERT_EYES_NEGATIVE,
        "media_handling": "Generated videos remain in production Google Drive output folder, not in git.",
        "clips": clips,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    update_review(clips)

    print(f"Prompts written: {PROMPT_ROOT}")
    print(f"Manifest written: {MANIFEST_PATH}")
    print(f"Clips prepared: {len(clips)}")


if __name__ == "__main__":
    main()
