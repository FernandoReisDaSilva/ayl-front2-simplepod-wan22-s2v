# AYL Front 2 — Voice, Character & Lip Sync

Local Mac orchestration repo for Amplify Your Language Front 2.

This repo manages:
- Gemini TTS scripts
- wardrobe prompts and metadata
- OpenAI image generation/editing scripts
- Replicate / Wan 2.7 test helpers
- video-base manifests
- character-video handoff manifests

It does not store heavy production assets.
Production assets live in Google Drive under AYL_Production.

## Wardrobe Wan 2.7 Test

Create the production folder structure:

```bash
python3 scripts/setup/create_project_sources_structure.py
```

Create the Luca wardrobe queue:

```bash
python3 scripts/wardrobe/create_wardrobe_test_queue.py
```

Generate per-outfit prompts:

```bash
python3 scripts/wardrobe/create_outfit_prompts.py
```

Register a result status without copying media into the repo:

```bash
python3 scripts/wardrobe/register_wardrobe_result.py luca_test_001 --status needs_review --result-path "/path/or/uri/to/external/result"
```

Outputs are written under:

```text
/Users/fernandoreisdasilva/Library/CloudStorage/GoogleDrive-fernandoreisdasilva@gmail.com/Meu Drive/AYL_Production/04_video_jobs/TEST_WARDROBE_WAN_0001/wardrobe/
```

The scripts only create orchestration JSON and prompt text. They do not hardcode media files into the repo, copy generated media assets, or modify `AYL_Production/00_project_sources/active_documents/`.

## Replicate Wan 2.7 API Runner

Set up a local environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install replicate requests python-dotenv
export REPLICATE_API_TOKEN="..."
```

Run the no-arm-movement Luca V3 idle test:

```bash
python3 scripts/replicate/run_wan27_i2v.py \
  --image-path "$HOME/Library/CloudStorage/GoogleDrive-fernandoreisdasilva@gmail.com/Meu Drive/AYL_Production/04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/replicate_inputs/source_images/luca_wan_v3.png" \
  --prompt-path "video_base/replicate_inputs/luca_test_002_wan27_base_motion_011_no_arm_movement_idle_wan27_prompt.txt" \
  --output-folder "$HOME/Library/CloudStorage/GoogleDrive-fernandoreisdasilva@gmail.com/Meu Drive/AYL_Production/04_video_jobs/TEST_WARDROBE_WAN_0001/video_base/replicate_outputs/" \
  --duration 5 \
  --aspect-ratio "1:1" \
  --resolution "720p" \
  --omit-last-frame
```

If Replicate rejects an input parameter, rerun with `--omit-param <name>` or adjust parameter names with `--first-frame-param`, `--last-frame-param`, and `--audio-param`. Use `--audio-path` only when testing audio/lip-sync readiness.

## Idle Library V1

Wan 2.7 is approved for V1 silent character idle clips without audio. This approval is limited to short silent visual presence states for pauses, repeat moments, listening, challenge, thinking, and reveal moments. It does not approve Wan 2.7 for lip-sync, audio-driven video, spoken character clips, or final scaled production. Replicate remains the test harness for now; future scaled production is expected to move to RunPod API.

Character clips may be generated as `1:1`; Front 3 / Remotion handles the final `16:9` composition. For idle generation, use first frame only and omit `last_frame`, because using the same first and last frame tends to freeze motion. Keep `enable_prompt_expansion=false` for controlled identity, expression, and motion. Use clips around 5 seconds; longer silent moments should be assembled in Remotion or FFmpeg with ping-pong, crossfade, repetition, or state alternation.

Official V1 silent idle states:

- `neutral_present`
- `encouraging_wait`
- `challenge_focus`
- `listening_check`
- `thinking_pause`
- `result_reveal`

Recommended future manifest field: `silent_character_state`.

Generate the 30-prompt idle library and manifest:

```bash
python3 scripts/video_base/create_idle_library_v1.py
```

Idle prompt rule: use `blinking is optional; at most one quick, alert blink during the full 5-second clip; eyes reopen immediately; do not blink repeatedly; if blinking looks unnatural, keep eyes open and alert`. Preserve calmness while adding alert engagement: calm but alert, attentive eyes, quiet energy, engaged presence, and an interested expression. Avoid tired, bored, sleepy, passive, or disinterested expressions; no slow blinks, heavy eyelids, drooping eyelids, low-energy face, excessive blinking, or sleepy eyes. Keep lips fully closed: no speaking, no lip movement, no mouth opening, no teeth. Keep both arms and hands still; do not gesture or raise hands.

The Maé temporary V1.1 test generated 6/6 clips successfully and validated partial per-character testing. It also revealed excessive blinking across several clips, which led to the V1.2 blink correction above. The temporary Maé runner remains a test utility and may be deleted later before the production RunPod workflow.

Run the full sequential 30-video batch:

```bash
export REPLICATE_API_TOKEN="..."
python3 scripts/video_base/run_idle_library_v1_batch.py
```

Rerun only failed items later:

```bash
python3 scripts/video_base/run_idle_library_v1_batch.py --retry-failed-only
```

Limit or filter a test run:

```bash
python3 scripts/video_base/run_idle_library_v1_batch.py --characters alex --limit 2
```

Create the five character review reels from succeeded clips:

```bash
python3 scripts/video_base/create_idle_review_reels.py
```

Review both the 30 individual source clips and the 5 review-only reels before approving.
