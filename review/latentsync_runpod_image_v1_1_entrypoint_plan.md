# LatentSync RunPod Image V1.1 Entrypoint Plan

Status: prepared, not built, not pushed, not executed

## Diagnosis

The public custom image `ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.0` repeatedly created a RunPod Pod, stayed `RUNNING`, and terminated cleanly, but produced no R2 progress or final report.

This happened with both:

- the LatentSync custom image R2 probe;
- the smaller shell/R2 probe.

The RunPod API, Community RTX 3090 create flow, status polling, and cleanup are working. The remaining failure is inside container startup or command injection. We will stop relying on `dockerArgs` for this image family.

## V1.1 Decision

Build a new tag with an image-owned entrypoint:

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.1
```

The entrypoint is controlled by `AYL_RUN_MODE` and does not require `dockerArgs`.

## Entrypoint Modes

- `idle`: safe default; sleeps indefinitely unless `AYL_IDLE_SECONDS` is set.
- `r2_probe`: runs `/opt/ayl/runtime_probe.py --mode r2_probe`.
- `latentsync_probe`: runs `/opt/ayl/runtime_probe.py --mode latentsync_probe`.
- `latentsync_run`: reserved; exits non-zero for now.

Default without `AYL_RUN_MODE` is `idle`. Test scripts must pass:

```text
AYL_RUN_MODE=r2_probe
```

## R2 Probe

The first runtime action in `r2_probe` is writing:

```text
tests/runpod_latentsync_image_v1_1/progress/container_started.json
```

The final report is written to:

```text
tests/runpod_latentsync_image_v1_1/output/final_report.json
```

The report includes hostname, Python version, cwd, image tag, marker nonce, redacted env presence, and explicit flags that no checkpoints are downloaded and no inference is run.

## LatentSync Probe

`latentsync_probe` adds:

- torch import status;
- torch version;
- CUDA availability;
- GPU name when available;
- `ffmpeg` availability;
- LatentSync path candidates:
  - `/workspace/LatentSync`
  - `/opt/LatentSync`
  - `/app/LatentSync`

It still does not download checkpoints and does not run inference.

## Dockerfile Changes

- Copy `entrypoint.sh` and `runtime_probe.py` to `/opt/ayl/`.
- `chmod +x /opt/ayl/entrypoint.sh`.
- Set `ENTRYPOINT ["/opt/ayl/entrypoint.sh"]`.
- Set `CMD ["idle"]`.
- Keep `DOWNLOAD_CHECKPOINTS=0` by default.
- Keep `boto3` installed in the image.
- Do not embed model weights.

## RunPod Probe Script

New script:

```text
scripts/runpod/temp_test_runpod_latentsync_image_v1_1_entrypoint_r2_probe_0001.py
```

It creates a Pod with image `0.1.1`, passes `AYL_RUN_MODE=r2_probe`, passes R2 env vars, does not use `dockerArgs`, polls R2 progress/final, and always attempts cleanup.

## Validation Commands

```bash
bash -n docker/latentsync-runpod-v1/entrypoint.sh
python3 -m py_compile docker/latentsync-runpod-v1/runtime_probe.py
python3 -m py_compile scripts/runpod/temp_test_runpod_latentsync_image_v1_1_entrypoint_r2_probe_0001.py
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_entrypoint_r2_probe_0001.py
```

## GitHub Actions Next Step

After review, run the workflow:

```text
.github/workflows/build-latentsync-runpod-v1.yml
```

Use:

```text
image_tag=ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.1
download_checkpoints=false
push_image=true
```

Do not build/push until this plan is approved.

## Paid Probe Command

Do not execute yet:

```bash
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_entrypoint_r2_probe_0001.py --execute --confirm-cost-risk
```
