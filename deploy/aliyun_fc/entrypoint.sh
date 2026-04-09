#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export ENABLE_HR_CACHE="${ENABLE_HR_CACHE:-0}"

PORT="${PORT:-7860}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
NUM_GPUS="${NUM_GPUS:-1}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
IMAGES_DIR="${IMAGES_DIR:-/tmp/falcon-perception/images}"
HF_MODEL_ID="${HF_MODEL_ID:-tiiuae/Falcon-Perception}"
HF_REVISION="${HF_REVISION:-main}"

mkdir -p "${IMAGES_DIR}"

cmd=(
  python -m falcon_perception.server
  --config.host "${SERVER_HOST}"
  --config.port "${PORT}"
  --config.num-gpus "${NUM_GPUS}"
  --config.startup-timeout "${STARTUP_TIMEOUT}"
  --config.images-dir "${IMAGES_DIR}"
)

if [[ -n "${HF_LOCAL_DIR:-}" ]]; then
  cmd+=(--config.hf-local-dir "${HF_LOCAL_DIR}")
else
  cmd+=(--config.hf-model-id "${HF_MODEL_ID}" --config.hf-revision "${HF_REVISION}")
fi

if [[ -n "${DTYPE:-}" ]]; then
  cmd+=(--config.dtype "${DTYPE}")
fi

if [[ -n "${MAX_BATCH_SIZE:-}" ]]; then
  cmd+=(--config.max-batch-size "${MAX_BATCH_SIZE}")
fi

if [[ -n "${MAX_IMAGE_SIZE:-}" ]]; then
  cmd+=(--config.max-image-size "${MAX_IMAGE_SIZE}")
fi

if [[ -n "${MIN_IMAGE_SIZE:-}" ]]; then
  cmd+=(--config.min-image-size "${MIN_IMAGE_SIZE}")
fi

if [[ -n "${MAX_TOKENS:-}" ]]; then
  cmd+=(--config.max-tokens "${MAX_TOKENS}")
fi

echo "[entrypoint] launching Falcon Perception server"
echo "[entrypoint] port=${PORT} host=${SERVER_HOST} num_gpus=${NUM_GPUS} images_dir=${IMAGES_DIR}"
if [[ -n "${HF_LOCAL_DIR:-}" ]]; then
  echo "[entrypoint] using local model dir: ${HF_LOCAL_DIR}"
else
  echo "[entrypoint] using HF model: ${HF_MODEL_ID}@${HF_REVISION}"
fi

exec "${cmd[@]}"
