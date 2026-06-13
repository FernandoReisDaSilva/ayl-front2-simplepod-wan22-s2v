# Wan 2.7 Silent Idle Clip Decision V1

## Decision Summary

Front 2 approves Wan 2.7 for generating V1 silent character idle clips without audio. This approval is limited to visual character presence clips only. It does not approve Wan 2.7 for lip-sync, voiced clips, spoken character clips, or final scaled production.

Replicate remains a test harness for now. Future scaled production is expected to move to a RunPod API workflow.

## Approved Scope

Wan 2.7 may be used for short silent character presence clips for:

- `neutral_present`
- `encouraging_wait`
- `challenge_focus`
- `listening_check`
- `thinking_pause`
- `result_reveal`

These states cover pauses, repeat moments, listening, challenge, thinking, and reveal moments.

## Out Of Scope

- Lip-sync
- Audio-driven video
- Spoken character clips
- Final scaled production pipeline
- Approval of Replicate as the long-term production backend

## Technical Settings

- Model family: Wan 2.7 image-to-video
- Current harness: Replicate Web/API
- Clip duration: around 5 seconds
- Aspect ratio: `1:1`
- Resolution: `720p`
- Use first frame only
- Omit `last_frame`
- Set `enable_prompt_expansion=false`

Character clips may be generated in `1:1`. Front 3 / Remotion handles the final `16:9` layout composition.

Using the same `first_frame` and `last_frame` tends to freeze motion. For idle clips, use `first_frame` only.

Longer silent moments should be assembled outside Wan with Remotion or FFmpeg using ping-pong, crossfade, repetition, trimming, or state alternation.

## Prompt Rules

Use calm but alert character presence:

- calm but alert
- attentive eyes
- quiet energy
- engaged presence
- interested expression
- eyes stay alert and engaged

Avoid low-energy expressions:

- no tired expression
- no bored expression
- no sleepy eyes
- no passive stare
- no drooping eyelids
- no low-energy face

Keep the mouth closed:

- lips remain fully closed
- no speaking
- no lip movement
- no mouth opening
- no teeth

Keep arms and hands stable:

- keep both arms and hands still
- do not gesture
- do not raise hands

## Blink V1.2 Correction

The Maé temporary V1.1 test showed that forced quick blinks can still become too frequent across several clips. Future prompts must make blinking optional and rare.

Use this blink guidance:

```text
blinking is optional; at most one quick, alert blink during the full 5-second clip; eyes reopen immediately; do not blink repeatedly; if blinking looks unnatural, keep eyes open and alert
```

Additional guidance:

- blinking should be rare
- do not blink repeatedly
- do not blink more than once
- if blinking looks unnatural, keep eyes open and alert
- eyes stay engaged without excessive blinking

## Maé Temp Test Result

The temporary Maé-only idle test generated 6/6 clips successfully and validated partial per-character testing. This confirms Front 2 can test a single character without regenerating the full 30-clip library.

The Maé test also revealed excessive blinking, leading to the V1.2 blink correction.

The temporary Maé runner remains a test utility and may be deleted later before the production RunPod workflow.

## Implications For Front 2

Front 2 can use Wan 2.7 for silent visual idle clip generation in controlled V1 testing. Prompt text should remain the source of truth for identity, closed-mouth behavior, arm/hand stability, alert presence, and rare optional blinking.

Front 2 should keep individual clips as source-of-truth production assets and create review reels only as review artifacts.

Recommended future manifest field:

```text
silent_character_state
```

## Implications For Front 3

Front 3 should treat these clips as silent internal Remotion assets. The clips may remain `1:1`; Front 3 is responsible for `16:9` composition, layout, timing, state alternation, ping-pong/repetition, crossfades, and placement inside the final scene.

Front 3 should not infer that these silent clips are approved for lip-sync or audio-driven character delivery.

## Items For Front 1 Incorporation

Front 1 may incorporate the following policy language into official AYL documentation:

- Wan 2.7 is approved for V1 silent visual idle clips without audio.
- This approval excludes lip-sync, audio-driven video, spoken clips, and scaled final production.
- Use first frame only and omit last frame for idle clips.
- Use `enable_prompt_expansion=false` for controlled identity and motion.
- Use `1:1` character clips for Remotion composition into `16:9`.
- Use rare optional blinking, not forced blinking.
- Keep mouth closed and arms/hands stable.
- Use `silent_character_state` for future state manifests.

## Future RunPod API Note

Replicate is currently a test harness. Future scaled production is expected to move to RunPod API so Front 2 can run controlled batch generation, logging, retries, and production orchestration without relying on Replicate as the long-term backend.
