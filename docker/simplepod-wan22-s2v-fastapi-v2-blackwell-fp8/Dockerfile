FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    AYL_TORCH_VERSION=2.11.0 \
    AYL_TORCHAO_VERSION=0.17.0 \
    AYL_IMAGE_TAG=0.3.01-blackwell-fp8-runtime-probe-v2 \
    AYL_FP8_PROBE_REPORT_PATH=/tmp/fp8_runtime_probe_v1.json

WORKDIR /opt/ayl-simplepod-wan22-s2v-fp8-runtime-probe

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir "torch==${AYL_TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir "torchao==${AYL_TORCHAO_VERSION}" --index-url https://download.pytorch.org/whl/cu128

COPY docker/simplepod-wan22-s2v-fastapi-v2-blackwell-fp8/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY scripts/fp8/temp_fp8_runtime_probe_v1.py ./temp_fp8_runtime_probe_v1.py
COPY scripts/fp8/temp_fp8_build_smoke_v1.py ./temp_fp8_build_smoke_v1.py
RUN python ./temp_fp8_build_smoke_v1.py

CMD ["python", "temp_fp8_runtime_probe_v1.py"]
