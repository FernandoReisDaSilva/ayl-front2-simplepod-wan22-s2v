FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AYL_RUN_MODE=idle
ENV WAN27_DURATION_SECONDS=5
ENV WAN27_RESOLUTION=480p

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install boto3

RUN mkdir -p /opt/ayl /workspace/input /workspace/output
COPY docker/wan27-runpod-v1/entrypoint.sh /opt/ayl/entrypoint.sh
COPY docker/wan27-runpod-v1/runtime_probe.py /opt/ayl/runtime_probe.py
RUN chmod +x /opt/ayl/entrypoint.sh

ENTRYPOINT ["/opt/ayl/entrypoint.sh"]
CMD ["idle"]
