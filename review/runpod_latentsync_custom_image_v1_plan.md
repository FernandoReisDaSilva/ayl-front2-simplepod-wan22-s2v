# RunPod LatentSync Custom Image V1 Plan

Date: 2026-06-16

## Decision

Abandon the long PyTorch-image probe chain as the active route and build one custom worker image from `runpod/base:1.0.2-ubuntu2404`.

The fast route is:

```text
runpod/base:1.0.2-ubuntu2404
-> apt system deps
-> micromamba Python 3.10.13 environment
-> LatentSync repo at a229c3948406bc2cf6eaf4873e662e70c6a04746
-> pip install requirements.txt with CUDA 12.1 wheels
-> first build without checkpoints
-> optional later build with minimum inference checkpoints
-> push image
-> run one short paid Pod test only when approved
```

## 1. Real Dependency Map For LatentSync

Validated from local RunPod reports and upstream LatentSync files.

System layer:

- `git`: clone LatentSync or preserve source provenance.
- `ffmpeg`: mandatory for video/audio decoding, resampling, and muxing.
- `build-essential`, `python3.10-dev`: needed for source-build packages such as `insightface`, `python_speech_features`, and `antlr4-python3-runtime`.
- `libgl1`, `libglib2.0-0`: needed by OpenCV/import path on Linux containers.
- `libgomp1`: low-risk runtime support for scientific/image packages.
- `ca-certificates`, `curl`: download and TLS hygiene.

Python/runtime layer:

- Python 3.10 is the cleanest target from the successful resolver probe; V1 uses a `micromamba` Python 3.10.13 environment so build headers are available for source-build packages.
- `torch==2.5.1` and `torchvision==0.20.1` from `https://download.pytorch.org/whl/cu121`.
- GPU/media/model packages from upstream `requirements.txt`: `diffusers==0.32.2`, `transformers==4.48.0`, `decord==0.6.0`, `accelerate==0.26.1`, `einops==0.7.0`, `omegaconf==2.3.0`, `opencv-python==4.9.0.80`, `mediapipe==0.10.11`, `python_speech_features==0.6`, `librosa==0.10.1`, `scenedetect==0.6.1`, `ffmpeg-python==0.2.0`, `imageio==2.31.1`, `imageio-ffmpeg==0.5.1`, `lpips==0.1.4`, `face-alignment==1.4.1`, `gradio==5.24.0`, `huggingface-hub==0.30.2`, `numpy==1.26.4`, `kornia==0.8.0`, `insightface==0.7.3`, `onnxruntime-gpu==1.21.0`, `DeepCache==0.1.1`.
- `boto3`: not a LatentSync dependency, but useful for AYL R2 input/output in the first real worker test.

Operational layer:

- RunPod template: `runpod-ubuntu-2404`.
- Confirmed low-cost GPU candidate: `NVIDIA RTX A4000`.
- Container disk: use at least `40 GB`; `20 GB` was enough for dry-run, but not a comfortable custom-image runtime with checkpoints and outputs.
- Network Volume: not required for the first test.
- R2 remains the input/output side-effect channel.
- The build/push scripts must be executed from the repository root and fail fast if launched elsewhere.
- The default image `CMD` keeps the container alive with `sleep infinity`; this is useful for RunPod command overrides, but any Pod left running will keep billing until manually or automatically terminated.

## 2. Minimum Models / Checkpoints

For the first functional inference test:

- `checkpoints/latentsync_unet.pt` from `ByteDance/LatentSync-1.6` on Hugging Face, about `5.07 GB`.
- `checkpoints/whisper/tiny.pt` from `ByteDance/LatentSync-1.6`, about `75.6 MB`.

Not required for first inference:

- `checkpoints/stable_syncnet.pt`, about `1.61 GB`; this is for training/evaluation, not the shortest inference path.
- `checkpoints/auxiliary/i3d_torchscript.pt` and other auxiliary assets; these are training/TREPA/evaluation oriented.
- `vgg16-397923af.pth`; only needed if LPIPS/training/eval code path pulls it. It is not part of the first CLI inference path.

## 3. Dockerfile V1

Created:

```text
docker/latentsync-runpod-v1/Dockerfile
```

It installs system deps, creates a Python 3.10.13 environment with `micromamba`, installs LatentSync requirements, and performs import smoke checks during build.

By default, `DOWNLOAD_CHECKPOINTS=0`, so V1 builds a lighter dependency image without model downloads. A later explicit build can set `LATENTSYNC_DOWNLOAD_CHECKPOINTS=1` to embed `latentsync_unet.pt` and `whisper/tiny.pt`.

TODO: pin the `micromamba` download URL/version after checking the current official release path. V1 intentionally keeps the upstream `latest` URL until that is verified.

## 4. Build And Push Scripts

Created:

```text
scripts/runpod/build_latentsync_runpod_image_v1.sh
scripts/runpod/push_latentsync_runpod_image_v1.sh
```

Recommended first local build from repo root, without checkpoints:

```bash
LATENTSYNC_RUNPOD_IMAGE_TAG=ayl-latentsync-runpod:0.1.0 \
  bash scripts/runpod/build_latentsync_runpod_image_v1.sh
```

The script defaults to `LATENTSYNC_DOWNLOAD_CHECKPOINTS=0`, prints the selected image tag and checkpoint mode, and never performs a push.

Later build with embedded checkpoints:

```bash
LATENTSYNC_DOWNLOAD_CHECKPOINTS=1 \
LATENTSYNC_RUNPOD_IMAGE_TAG=ayl-latentsync-runpod:0.1.0-with-checkpoints \
  bash scripts/runpod/build_latentsync_runpod_image_v1.sh
```

Push from repo root, only after the local image is intentionally built:

```bash
LATENTSYNC_RUNPOD_IMAGE_TAG=ayl-latentsync-runpod:0.1.0 \
LATENTSYNC_RUNPOD_REMOTE_IMAGE_TAG=docker.io/USER/ayl-latentsync-runpod:0.1.0 \
  bash scripts/runpod/push_latentsync_runpod_image_v1.sh
```

## 5. Estimated Final Image Size

Expected size with checkpoints embedded:

- `runpod/base:1.0.2-ubuntu2404`: unknown local compressed size, likely modest compared with CUDA wheels.
- PyTorch CUDA 12.1 wheel family observed in dry-run: roughly `2.8 GB` downloaded just for torch/CUDA/triton family.
- Other Python packages: roughly `1.5-3.0 GB` installed footprint after scientific/media/model dependencies.
- Apt system dependencies: roughly `0.8 GB` installed footprint, based on the successful system-deps probe.
- LatentSync repo: about `20 MB`.
- Minimum checkpoints: about `5.15 GB`.

Practical estimate:

- Without checkpoints: `7-10 GB`.
- With minimum checkpoints: `12-16 GB`.
- With full Hugging Face repo copied: likely `17-22 GB` because the model repo is listed at `9.64 GB`.

The first recommended build is the no-checkpoint image. It validates dependency installation and image mechanics without adding the extra model-download time and image weight. The checkpoint build should happen only after the light build is accepted.

## 6. Minimum Functional Validation Plan

No optimization and no long test:

1. Local no-cost checks:
   - `docker build` completes.
   - `docker run --gpus all IMAGE python -c "import torch; print(torch.cuda.is_available())"` on a GPU host, if available.
   - `docker run IMAGE python -c "import diffusers, transformers, cv2, mediapipe, insightface, onnxruntime, decord"`.
   - Confirm files exist: `checkpoints/latentsync_unet.pt` and `checkpoints/whisper/tiny.pt`.

2. Push image to registry.

3. Paid RunPod smoke test, only after explicit approval:
   - Create one disposable A4000 Pod from the custom image.
   - Run a command that only prints torch/CUDA status and checkpoint presence, then uploads a JSON marker to R2.
   - Terminate Pod immediately.

4. First real lip-sync test, only after smoke test passes:
   - Use one short input video clip, target `3-5 seconds`.
   - Use one short WAV, 16 kHz if already available.
   - Run `python -m scripts.inference` with `--inference_steps 20`, `--guidance_scale 1.5`, `--enable_deepcache`, and `configs/unet/stage2_512.yaml`.
   - Upload only `video_out.mp4` plus a small JSON run report to R2.
   - Terminate Pod immediately.

## 7. Probes That Can Be Discarded

Keep as historical evidence, but stop extending:

- `temp_test_runpod_latentsync_env_probe_0001.py`
- `temp_test_runpod_latentsync_repo_clone_probe_0001.py`
- `temp_test_runpod_latentsync_dependency_probe_0001.py`
- `temp_test_runpod_latentsync_python310_base_probe_0001.py`
- `temp_test_runpod_latentsync_pip_dry_run_python310_0001.py`
- `temp_test_runpod_latentsync_cuda121_base_image_probe_0001.py`
- `temp_test_runpod_pytorch_image_r2_marker_probe_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_repo_clone_probe_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_requirements_dry_install_probe_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_requirements_real_install_probe_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_install_failure_isolation_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_system_deps_install_probe_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_post_install_import_repair_0001.py`
- `temp_test_runpod_pytorch_image_latentsync_import_repair_split_torchvision_0001.py`

Keep useful infrastructure probes:

- `temp_test_runpod_api_auth_0001.py`
- `temp_test_runpod_template_and_create_fields_discovery_0001.py`
- `temp_select_runpod_template_candidates_0001.py`
- `temp_test_runpod_dockerargs_inline_command_0001.py`
- `temp_test_runpod_dockerargs_inline_r2_marker_0001.py`
- `temp_test_runpod_r2_pull_push_0001.py`

## 8. Shortest Path To First Real Lip-Sync

1. Build the image locally or on a machine with good network.
2. Push it to a registry visible to RunPod.
3. Run one paid A4000 smoke Pod with checkpoint/import/R2 marker only.
4. Upload a tiny input pair to R2: one `3-5s` face video and one `3-5s` WAV.
5. Run one paid A4000 Pod with the custom image and one inline command:

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

6. Upload `/workspace/output/video_out.mp4` and a JSON report to R2.
7. Terminate the Pod.

This skips more dependency probes and converts the next spend into an actual LatentSync result.
