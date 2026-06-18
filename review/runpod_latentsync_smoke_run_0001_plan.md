# RunPod LatentSync Smoke Run 0001 Plan

## Objective

Run the first minimal functional LatentSync smoke test on RunPod using the ENTRYPOINT-controlled image path, without `dockerArgs` and without a Network Volume.

This test is the first step that intentionally downloads checkpoints, downloads short media inputs, runs LatentSync inference, uploads an output MP4, and writes a final R2 report.

## Image

Target image:

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.5
```

This requires a new image tag because `AYL_RUN_MODE=latentsync_smoke_run` is added to the image entrypoint/runtime after the validated `0.1.4` environment probe.

## Run Mode

The Pod receives:

```text
AYL_RUN_MODE=latentsync_smoke_run
```

The script does not use `dockerArgs`.

## R2 Inputs

Expected existing R2 objects:

```text
checkpoints/latentsync/latentsync_unet.pt
checkpoints/latentsync/whisper/tiny.pt
tests/runpod_latentsync_smoke_run_0001/input/video.mp4
tests/runpod_latentsync_smoke_run_0001/input/audio.wav
```

The input video/audio should be short, ideally 3-5 seconds.

## Container Paths

```text
/opt/LatentSync/checkpoints/latentsync_unet.pt
/opt/LatentSync/checkpoints/whisper/tiny.pt
/workspace/input/video.mp4
/workspace/input/audio.wav
/workspace/output/video_out.mp4
```

## Inference Command

```bash
cd /opt/LatentSync
python -m scripts.inference \
  --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --inference_steps 20 \
  --guidance_scale 1.5 \
  --enable_deepcache \
  --video_path /workspace/input/video.mp4 \
  --audio_path /workspace/input/audio.wav \
  --video_out_path /workspace/output/video_out.mp4
```

## R2 Progress Phases

The runtime writes progress before any download or heavy operation:

```text
container_started
checkpoint_download_done
input_download_done
inference_started
inference_done
output_upload_done
final_report_written
```

The primary progress object is:

```text
tests/runpod_latentsync_smoke_run_0001/progress/container_started.json
```

## R2 Outputs

```text
tests/runpod_latentsync_smoke_run_0001/output/video_out.mp4
tests/runpod_latentsync_smoke_run_0001/output/final_report.json
```

The final report includes checkpoint file facts, input file facts, inference return code, output file facts, output upload status, image tag, hostname, Python version, and redacted env presence.

## Success Criteria

- Pod is created.
- R2 progress is detected.
- R2 final report is detected and verified.
- Final report has required smoke-run fields.
- Output MP4 is uploaded to R2.
- Pod cleanup succeeds.
- `manual_cleanup_required=false`.

The local orchestration script treats final report presence as a technical signal. The actual LatentSync result quality must be reviewed from the uploaded MP4 and final report.

## Failure Handling

The runtime should write a final report even if inference returns non-zero or the output MP4 is missing.

If the Pod is created, the script must attempt automatic cleanup. If cleanup cannot be verified, terminate the Pod manually in the RunPod console.

## Dry-Run Command

```bash
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

## Paid Command

NÃO executar ainda:

```bash
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py --execute --confirm-cost-risk
```

## Build/Push Next Step

Build and push `0.1.5` through GitHub Actions only after the local validations pass. Keep `push_image=false` for a build-only check first if desired, then run again with `push_image=true` after approval.
