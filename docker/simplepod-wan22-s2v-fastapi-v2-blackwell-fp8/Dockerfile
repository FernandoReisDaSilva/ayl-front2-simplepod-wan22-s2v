FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG WAN22_REPO_URL=https://github.com/Wan-Video/Wan2.2.git
ARG WAN22_GIT_COMMIT=42bf4cfaa384bc21833865abc2f9e6c0e67233dc

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    AYL_TORCH_VERSION=2.11.0 \
    AYL_TORCHAO_VERSION=0.17.0 \
    AYL_IMAGE_TAG=0.3.03-blackwell-fp8-wan-gate0-diagnostics-v2 \
    AYL_WAN22_GIT_COMMIT=42bf4cfaa384bc21833865abc2f9e6c0e67233dc \
    AYL_FP8_PROBE_REPORT_PATH=/tmp/fp8_runtime_probe_v1.json \
    AYL_FP8_WAN_GATE0_REPORT_PATH=/tmp/fp8_wan_gate0_probe_v1.json \
    SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models \
    WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B \
    WAN22_REPO_DIR=/opt/Wan2.2 \
    HF_HOME=/mnt/ayl_models/caches/huggingface

WORKDIR /opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates ffmpeg git \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "torch==${AYL_TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir "torchao==${AYL_TORCHAO_VERSION}" --index-url https://download.pytorch.org/whl/cu128

COPY docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

RUN git clone --no-checkout "$WAN22_REPO_URL" /opt/Wan2.2 \
    && cd /opt/Wan2.2 \
    && git fetch --depth 1 origin "$WAN22_GIT_COMMIT" \
    && git checkout --detach "$WAN22_GIT_COMMIT" \
    && rm -rf /opt/Wan2.2/.git

COPY docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/patches ./patches
RUN python patches/patch_wan22_autocast_deprecated.py /opt/Wan2.2

COPY docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/app ./app
COPY scripts/fp8/temp_fp8_runtime_probe_v1.py ./temp_fp8_runtime_probe_v1.py
COPY scripts/fp8/temp_fp8_build_smoke_v1.py ./temp_fp8_build_smoke_v1.py
COPY scripts/fp8/temp_fp8_wan_gate0_probe_v1.py ./temp_fp8_wan_gate0_probe_v1.py
RUN python ./temp_fp8_build_smoke_v1.py
RUN python -m py_compile ./temp_fp8_wan_gate0_probe_v1.py

CMD ["python", "temp_fp8_wan_gate0_probe_v1.py"]
