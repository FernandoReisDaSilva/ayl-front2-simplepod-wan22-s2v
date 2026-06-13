from pathlib import Path

DRIVE_ROOT = (
    Path.home()
    / "Library"
    / "CloudStorage"
    / "GoogleDrive-fernandoreisdasilva@gmail.com"
    / "Meu Drive"
)
PRODUCTION_ROOT = DRIVE_ROOT / "AYL_Production"

folders = [
    "00_project_sources/active_documents",
    "00_project_sources/human_setup_documents",
    "00_project_sources/character_references",
    "00_project_sources/visual_tokens",
    "00_project_sources/schemas",

    "04_video_jobs/TEST_WARDROBE_WAN_0001/character_sources",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/prompts",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/generated",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/approved",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/rejected",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/results",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/replicate_inputs",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/replicate_outputs",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/approved",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/rejected",
    "04_video_jobs/TEST_WARDROBE_WAN_0001/review",
]

for folder in folders:
    path = PRODUCTION_ROOT / folder
    path.mkdir(parents=True, exist_ok=True)
    print(f"OK: {path}")

print()
print(f"AYL Production root: {PRODUCTION_ROOT}")
