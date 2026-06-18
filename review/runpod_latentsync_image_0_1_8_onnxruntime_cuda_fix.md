# RunPod LatentSync Image 0.1.8 ONNXRuntime CUDA Fix

## Objective

Prepare image tag:

`ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.8`

The goal is to prevent InsightFace from silently falling back to:

`Applied providers: ['CPUExecutionProvider']`

when LatentSync runs face analysis through ONNXRuntime.

## Observed Problem

The smoke run reached LatentSync inference, but InsightFace used ONNXRuntime CPU provider only. A previous runtime error also showed that ONNXRuntime CUDA provider could not load because `libnvrtc.so.12` was missing.

## Dockerfile Change

The image now installs:

`nvidia-cuda-nvrtc-cu12==12.1.105`

This provides `libnvrtc.so.12` in the Python environment. The Dockerfile also exposes CUDA-related Python wheel library directories through `LD_LIBRARY_PATH`, including:

- `nvidia/cuda_nvrtc/lib`
- `nvidia/cudnn/lib`
- `nvidia/cublas/lib`
- `nvidia/cuda_runtime/lib`

The existing `onnxruntime-gpu==1.21.0` from the LatentSync requirements is kept.

## Runtime Probe Change

`runtime_probe.py` now records:

- `onnxruntime_import_status`
- `onnxruntime_version`
- `onnxruntime_available_providers`
- `onnxruntime_cuda_available`
- `onnxruntime_cuda_required`
- `onnxruntime_error_truncated`

If `AYL_REQUIRE_ONNXRUNTIME_CUDA=1` and `CUDAExecutionProvider` is absent, the runtime probe returns:

`runtime_probe_status=onnxruntime_cuda_missing`

This allows a future paid probe to fail explicitly instead of accepting CPU fallback.

## Smoke Status

The approved smoke script remains pinned to image `0.1.7` and was not changed by this fix.

## Next Build Step

Run GitHub Actions workflow `Build LatentSync RunPod V1` with:

`image_tag=ghcr.io/fernandoreisdasilva/ayl-latentsync-runpod:0.1.8`

Use `push_image=false` for a build-only check first if desired, then repeat with `push_image=true` after approval.

## Future Probe Recommendation

After image `0.1.8` is published, run the technical LatentSync environment probe:

`scripts/runpod/temp_test_runpod_latentsync_image_v1_1_latentsync_probe_0001.py`

The probe default image was updated to `0.1.8` and now passes:

`AYL_REQUIRE_ONNXRUNTIME_CUDA=1`

Success criterion:

- `CUDAExecutionProvider` appears in `onnxruntime_available_providers`
- `onnxruntime_cuda_available=true`
- final report is written to R2
- RunPod cleanup succeeds

## Not Done

- No RunPod Pod was created.
- No build was run locally.
- No image was pushed.
- The approved `0.1.7` smoke script was not altered.
