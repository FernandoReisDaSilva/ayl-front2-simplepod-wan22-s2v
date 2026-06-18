# RunPod LatentSync Smoke Run 0001 Fix 0.1.7

## Observed Failure

Image `0.1.6` reached the LatentSync inference path but failed when Diffusers tried to resolve:

```text
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

The runtime attempted a Hugging Face/Xet download and received a 403/no-permit style failure.

## Goal

Prepare:

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.7
```

The smoke run should not depend on Hugging Face network access during RunPod inference for `stabilityai/sd-vae-ft-mse`.

## Implementation

The Dockerfile pre-caches the VAE during image build using:

```text
HF_HOME=/opt/hf-cache
huggingface_hub.snapshot_download
repo_id=stabilityai/sd-vae-ft-mse
```

The download is restricted with `allow_patterns`:

```text
config.json
diffusion_pytorch_model.safetensors
```

This avoids downloading the full model repository and avoids the legacy pickle `.bin` weight when safetensors is available.

`hf_transfer` remains installed in `/opt/latentsync-venv`.

## Runtime Offline Mode

The image sets:

```text
HF_HOME=/opt/hf-cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

This should force Diffusers/Hugging Face clients to resolve `stabilityai/sd-vae-ft-mse` from the pre-populated cache instead of attempting an external download during inference.

## Files Updated

```text
docker/latentsync-runpod-v1/Dockerfile
.github/workflows/build-latentsync-runpod-v1.yml
scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

## Validation

Local-only validation:

```bash
bash -n docker/latentsync-runpod-v1/entrypoint.sh
python3 -m py_compile docker/latentsync-runpod-v1/runtime_probe.py
python3 -m py_compile scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

No Pod execution, RunPod call, Docker build, image push, local VAE download, or R2 mutation is part of this local validation.

## Remaining Risk

This assumes Diffusers resolves `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` from the standard Hugging Face cache populated by `snapshot_download`.

If LatentSync pins a Diffusers behavior that does not prefer `diffusion_pytorch_model.safetensors`, a follow-up may need to include `diffusion_pytorch_model.bin` too.
