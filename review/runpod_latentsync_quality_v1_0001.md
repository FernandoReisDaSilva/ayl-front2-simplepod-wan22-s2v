# RunPod LatentSync Quality V1 0001

## Objective

Create a separate LatentSync quality test without changing the approved technical smoke test.

Quality V1 uses separate R2 input/output/report/progress keys so quality experiments can evolve without disturbing the smoke baseline.

## RunPod Script

```text
scripts/runpod/temp_test_runpod_latentsync_quality_v1_0001.py
```

It reuses the approved smoke runner implementation but overrides the test ID, logs, output paths, R2 keys, pod name, and default timeout.

## Image And Run Mode

```text
image: ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.8
run_mode: latentsync_smoke_run
```

`latentsync_smoke_run` is reused for now because it is the available runtime mode that downloads checkpoints/VAE/input assets, patches VAE loading to the local path, runs inference, uploads output MP4, and writes the final report.

## Quality R2 Keys

```text
tests/runpod_latentsync_quality_v1_0001/progress/container_started.json
tests/runpod_latentsync_quality_v1_0001/input/video.mp4
tests/runpod_latentsync_quality_v1_0001/input/audio.wav
tests/runpod_latentsync_quality_v1_0001/output/video_out.mp4
tests/runpod_latentsync_quality_v1_0001/output/final_report.json
```

## Quality V1A Parameters

The RunPod script passes these runtime environment variables:

```text
LATENTSYNC_INFERENCE_STEPS=50
LATENTSYNC_GUIDANCE_SCALE=1.5
LATENTSYNC_ENABLE_DEEPCACHE=0
```

Expected effective inference options:

```text
--inference_steps 50
--guidance_scale 1.5
```

`--enable_deepcache` is intentionally omitted for Quality V1A.

## Shared Stable Assets

These remain shared and stable:

```text
checkpoints/latentsync/latentsync_unet.pt
checkpoints/latentsync/whisper/tiny.pt
checkpoints/latentsync/vae/sd-vae-ft-mse/config.json
checkpoints/latentsync/vae/sd-vae-ft-mse/diffusion_pytorch_model.safetensors
```

## Timeout

Quality V1 defaults to:

```text
--container-disk-gb 40
--max-wait-seconds 1800
```

This leaves more room for higher-quality inputs or slower cold-start behavior while keeping automatic cleanup.

## R2 Helpers

Upload quality inputs only:

```text
scripts/r2/temp_upload_latentsync_quality_v1_inputs_0001.py
```

Check quality inputs only:

```text
scripts/r2/temp_check_latentsync_quality_v1_inputs_0001.py
```

These helpers do not upload or check checkpoints/VAE.

## Dry-Run Commands

```bash
python3 scripts/runpod/temp_test_runpod_latentsync_quality_v1_0001.py
python3 scripts/r2/temp_upload_latentsync_quality_v1_inputs_0001.py --video-local /path/to/video.mp4 --audio-local /path/to/audio.wav
```

## Paid Command

NÃO executar ainda:

```bash
python3 scripts/runpod/temp_test_runpod_latentsync_quality_v1_0001.py --execute --confirm-cost-risk
```

## Guardrails

- Do not alter the smoke script for quality experiments.
- Do not use `dockerArgs`.
- Do not use Network Volume.
- Do not depend on pod logs.
- Use R2 for progress and final report.
- Always auto-cleanup Pods in live execution.
