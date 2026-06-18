# R2 LatentSync Smoke Inputs 0001

## Objective

Prepare the required R2 inputs for the first LatentSync functional smoke run.

This local helper does not create RunPod Pods, does not call RunPod, and does not delete R2 objects.

## Destination Objects

```text
checkpoints/latentsync/latentsync_unet.pt
checkpoints/latentsync/whisper/tiny.pt
tests/runpod_latentsync_smoke_run_0001/input/video.mp4
tests/runpod_latentsync_smoke_run_0001/input/audio.wav
```

## Script

```text
scripts/r2/temp_upload_latentsync_smoke_inputs_0001.py
```

Default mode is dry-run. Real upload requires both:

```text
--execute --confirm-upload
```

Existing R2 objects are not overwritten unless `--overwrite` is also passed.

## Local Inputs

The script accepts:

```text
--checkpoint-unet-local
--checkpoint-whisper-local
--video-local
--audio-local
```

It validates local existence, requires each path to be a file, and calculates local size and SHA-256 before any upload.

## Dry-Run Command

```bash
python3 scripts/r2/temp_upload_latentsync_smoke_inputs_0001.py
```

With files:

```bash
python3 scripts/r2/temp_upload_latentsync_smoke_inputs_0001.py \
  --checkpoint-unet-local /path/to/latentsync_unet.pt \
  --checkpoint-whisper-local /path/to/tiny.pt \
  --video-local /path/to/video.mp4 \
  --audio-local /path/to/audio.wav
```

## Paid/Mutating Command

NÃO executar ainda:

```bash
python3 scripts/r2/temp_upload_latentsync_smoke_inputs_0001.py \
  --execute \
  --confirm-upload \
  --checkpoint-unet-local /path/to/latentsync_unet.pt \
  --checkpoint-whisper-local /path/to/tiny.pt \
  --video-local /path/to/video.mp4 \
  --audio-local /path/to/audio.wav
```

## Log

```text
logs/r2_latentsync_smoke_upload_0001_log.json
```

The log stores redacted R2 context, local file sizes, SHA-256 hashes, destination keys, dry-run/execute state, overwrite state, and post-upload HEAD results when upload is executed.
