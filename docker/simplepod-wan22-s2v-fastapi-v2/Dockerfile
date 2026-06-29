FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models \
    WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B \
    HF_HOME=/mnt/ayl_models/caches/huggingface \
    WAN22_REPO_DIR=/opt/Wan2.2 \
    AYL_IMAGE_TAG=0.1.4

WORKDIR /opt/ayl-simplepod-wan22-s2v-fastapi-v2

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git /opt/Wan2.2

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
