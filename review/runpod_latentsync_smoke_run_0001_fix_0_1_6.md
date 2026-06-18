# RunPod LatentSync Smoke Run 0001 Fix 0.1.6

## Observed Failure

The `0.1.5` smoke run reached LatentSync inference but failed while loading `stabilityai/sd-vae-ft-mse`.

The reported root cause was:

```text
HF_HUB_ENABLE_HF_TRANSFER=1 is active, but hf_transfer is not installed
```

## Fix

Prepare image tag:

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.6
```

The Docker image now installs `hf_transfer` in the active LatentSync Python environment:

```text
/opt/latentsync-venv
```

This is the minimal correction because the current inference path still allows Hugging Face model downloads during runtime.

## Files Updated

```text
docker/latentsync-runpod-v1/Dockerfile
.github/workflows/build-latentsync-runpod-v1.yml
scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

## Deferred Follow-Up

If the project strategy becomes “no Hugging Face downloads during inference,” cache or mirror `stabilityai/sd-vae-ft-mse` through R2 in a later step and map it into the expected Hugging Face cache path inside the container.

That is intentionally not included in `0.1.6`; this fix only installs the missing transfer helper.

## Validation

Local validation only:

```bash
bash -n docker/latentsync-runpod-v1/entrypoint.sh
python3 -m py_compile docker/latentsync-runpod-v1/runtime_probe.py
python3 -m py_compile scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

No Pod execution, build, push, checkpoint download, or R2 mutation is part of this fix validation.
