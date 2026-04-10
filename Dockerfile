FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG PRELOAD_MODEL=1
ARG HF_MODEL_ID=tiiuae/Falcon-Perception
ARG HF_REVISION=main
ARG BUNDLED_MODEL_DIR=/opt/models/falcon-perception

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH=/app/.venv/bin:/usr/local/bin:$PATH \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PORT=7860 \
    NUM_GPUS=1 \
    DTYPE=bfloat16 \
    COMPILE=0 \
    CUDAGRAPH=0 \
    ENABLE_HR_CACHE=0 \
    MAX_BATCH_SIZE=16 \
    MAX_SEQ_LENGTH=4096 \
    N_PAGES=256 \
    PREFILL_LENGTH_LIMIT=8192 \
    MAX_TOKENS=2048 \
    MAX_IMAGE_SIZE=768 \
    STARTUP_TIMEOUT=900 \
    IMAGES_DIR=/tmp/falcon-perception/images \
    BUNDLED_MODEL_DIR=${BUNDLED_MODEL_DIR} \
    HF_MODEL_ID=${HF_MODEL_ID} \
    HF_REVISION=${HF_REVISION}

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    build-essential \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked --no-dev --extra server --no-install-project --python /usr/bin/python3 \
    && rm -rf /root/.cache /tmp/*

RUN if [ "${PRELOAD_MODEL}" = "1" ]; then \
        python -c "from huggingface_hub import snapshot_download; import os; local_dir=os.environ['BUNDLED_MODEL_DIR']; os.makedirs(local_dir, exist_ok=True); snapshot_download(repo_id=os.environ['HF_MODEL_ID'], repo_type='model', revision=os.environ['HF_REVISION'], local_dir=local_dir, allow_patterns=['model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']); print(f'Bundled model downloaded to {local_dir}')"; \
    fi \
    && rm -rf /root/.cache /tmp/*

COPY falcon_perception /app/falcon_perception
COPY deploy/aliyun_fc/entrypoint.sh /app/deploy/aliyun_fc/entrypoint.sh
COPY deploy/aliyun_fc/prewarm.py /app/deploy/aliyun_fc/prewarm.py

RUN chmod +x /app/deploy/aliyun_fc/entrypoint.sh \
    && mkdir -p "${IMAGES_DIR}" "${BUNDLED_MODEL_DIR}"

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10m --retries=10 \
    CMD python -c "import json, os, sys, urllib.request as u; data = json.load(u.urlopen('http://127.0.0.1:%s/v1/health' % os.environ.get('PORT', '7860'), timeout=3)); sys.exit(0 if data.get('status') == 'ready' else 1)" || exit 1

ENTRYPOINT ["/app/deploy/aliyun_fc/entrypoint.sh"]
