FROM runpod/base:1.0.2-ubuntu2404

ARG DEBIAN_FRONTEND=noninteractive
ARG LATENTSYNC_REPO=https://github.com/bytedance/LatentSync
ARG LATENTSYNC_COMMIT=a229c3948406bc2cf6eaf4873e662e70c6a04746
ARG DOWNLOAD_CHECKPOINTS=0

ENV VENV_PATH=/opt/latentsync-venv
ENV LATENTSYNC_PATH=/opt/LatentSync
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=/opt/latentsync-venv/bin:$PATH
ENV HF_HOME=/opt/hf-cache
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV DOWNLOAD_CHECKPOINTS=${DOWNLOAD_CHECKPOINTS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        bzip2 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba \
    && micromamba create -y -p "$VENV_PATH" -c conda-forge python=3.10.13 pip \
    && micromamba clean -a -y \
    && python -m pip install --upgrade pip setuptools wheel

RUN git clone --depth 1 "$LATENTSYNC_REPO" "$LATENTSYNC_PATH" \
    && cd "$LATENTSYNC_PATH" \
    && git fetch --depth 1 origin "$LATENTSYNC_COMMIT" \
    && git checkout "$LATENTSYNC_COMMIT"

WORKDIR /opt/LatentSync

RUN python -m pip install -r requirements.txt \
    && python -m pip install boto3 hf_transfer

RUN mkdir -p /opt/ayl
COPY docker/latentsync-runpod-v1/entrypoint.sh /opt/ayl/entrypoint.sh
COPY docker/latentsync-runpod-v1/runtime_probe.py /opt/ayl/runtime_probe.py
RUN chmod +x /opt/ayl/entrypoint.sh

RUN if [ "$DOWNLOAD_CHECKPOINTS" = "1" ]; then \
        huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints; \
        huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints; \
    fi \
    && mkdir -p /root/.cache/torch/hub/checkpoints \
    && if [ -f checkpoints/auxiliary/vgg16-397923af.pth ]; then \
        ln -sf "$LATENTSYNC_PATH/checkpoints/auxiliary/vgg16-397923af.pth" /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth; \
    fi

RUN python - <<'PY'
import importlib
import torch

modules = [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "cv2",
    "mediapipe",
    "librosa",
    "huggingface_hub",
    "kornia",
    "insightface",
    "onnxruntime",
    "decord",
]
for name in modules:
    importlib.import_module(name)
print("torch", torch.__version__, "cuda", torch.version.cuda)
PY

ENTRYPOINT ["/opt/ayl/entrypoint.sh"]
CMD ["idle"]
