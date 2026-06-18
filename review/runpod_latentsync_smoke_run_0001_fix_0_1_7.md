# RunPod LatentSync Smoke Run 0001 Fix 0.1.7

## Observed Failure

Image `0.1.6` reached the LatentSync inference path but failed when Diffusers tried to resolve:

```text
AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
```

The runtime attempted a Hugging Face/Xet download and received a 403/no-permit style failure.

## Revised Strategy

Do not depend on Hugging Face during Docker build or RunPod inference.

The VAE is handled like the other smoke-test assets:

1. Prepare it locally.
2. Upload it to R2.
3. Download it from R2 inside the container before inference.
4. Patch LatentSync inference locally so `AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")` resolves to the local VAE directory.

## Image Tag

```text
ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.7
```

## R2 VAE Objects

```text
checkpoints/latentsync/vae/sd-vae-ft-mse/config.json
checkpoints/latentsync/vae/sd-vae-ft-mse/diffusion_pytorch_model.safetensors
```

## Container VAE Paths

```text
/opt/LatentSync/checkpoints/vae/sd-vae-ft-mse/config.json
/opt/LatentSync/checkpoints/vae/sd-vae-ft-mse/diffusion_pytorch_model.safetensors
```

## Implementation

- Removed Dockerfile `snapshot_download` for `stabilityai/sd-vae-ft-mse`.
- Kept `hf_transfer` installed, but the smoke path no longer depends on it.
- Kept runtime offline env:

```text
HF_HOME=/opt/hf-cache
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

- Added local preparer:

```text
scripts/latentsync/temp_prepare_sd_vae_ft_mse_0001.py
```

- Updated R2 checker/uploader to include the VAE files.
- Updated `latentsync_smoke_run` to download VAE files from R2 and patch `scripts/inference.py` before inference.
- Updated the smoke RunPod script to pass:

```text
R2_VAE_CONFIG_KEY
R2_VAE_SAFETENSORS_KEY
```

## Validation

Local validation only:

```bash
bash -n docker/latentsync-runpod-v1/entrypoint.sh
python3 -m py_compile docker/latentsync-runpod-v1/runtime_probe.py
python3 -m py_compile scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
python3 scripts/runpod/temp_test_runpod_latentsync_image_v1_1_smoke_run_0001.py
```

No Pod execution, RunPod call, Docker build, image push, local VAE download, or R2 mutation is part of this fix validation.
