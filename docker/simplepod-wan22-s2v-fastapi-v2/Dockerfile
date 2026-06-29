FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SIMPLEPOD_MODELS_ROOT=/mnt/ayl_models \
    WAN22_S2V_MODEL_DIR=/mnt/ayl_models/wan2.2/Wan2.2-S2V-14B \
    AYL_IMAGE_TAG=0.1.0

WORKDIR /opt/ayl-simplepod-wan22-s2v-fastapi-v2

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
