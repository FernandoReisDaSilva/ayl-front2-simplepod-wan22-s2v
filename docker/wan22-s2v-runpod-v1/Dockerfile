FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG WANVIDEO_WRAPPER_REPO=https://github.com/kijai/ComfyUI-WanVideoWrapper.git
ARG VIDEO_HELPER_REPO=https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
ARG KJ_NODES_REPO=https://github.com/kijai/ComfyUI-KJNodes.git

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV AYL_RUN_MODE=idle
ENV COMFYUI_PATH=/opt/ComfyUI
ENV AYL_COMFY_HOST=127.0.0.1
ENV AYL_COMFY_PORT=8188

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 "${COMFYUI_REPO}" /opt/ComfyUI \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /opt/ComfyUI/requirements.txt \
    && python -m pip install boto3 requests websocket-client

RUN git clone --depth 1 "${WANVIDEO_WRAPPER_REPO}" /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper \
    && git clone --depth 1 "${VIDEO_HELPER_REPO}" /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite \
    && git clone --depth 1 "${KJ_NODES_REPO}" /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes \
    && python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt \
    && python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt \
    && python -m pip install -r /opt/ComfyUI/custom_nodes/ComfyUI-KJNodes/requirements.txt

RUN mkdir -p /opt/ayl/workflows /workspace/input /workspace/output \
    && curl -L --fail --silent \
        https://raw.githubusercontent.com/kijai/ComfyUI-WanVideoWrapper/main/s2v/wanvideo2_2_S2V_context_window_testing.json \
        -o /opt/ayl/workflows/wanvideo2_2_S2V_context_window_testing.json

COPY docker/wan22-s2v-runpod-v1/entrypoint.sh /opt/ayl/entrypoint.sh
COPY docker/wan22-s2v-runpod-v1/runtime_probe.py /opt/ayl/runtime_probe.py
RUN chmod +x /opt/ayl/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/opt/ayl/entrypoint.sh"]
CMD ["idle"]
