#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
if [ -z "${ACTION}" ]; then
  echo "Usage: $0 <start|stop|restart|status|logs|health> [options]"
  exit 1
fi
shift || true

CONTAINER_NAME="${CONTAINER_NAME:-vllm-endpointing}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL_PATH="${MODEL_PATH:-/home/sagemaker-user/1.0.2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-endpointing-qwen3-0.6b-ft}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

start() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  docker run -d \
    --name "${CONTAINER_NAME}" \
    --network sagemaker \
    --gpus all \
    -v "${MODEL_PATH}:/model:ro" \
    "${IMAGE}" \
    /model \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --max-model-len 512 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 2048 \
    --enable-prefix-caching \
    --generation-config vllm

  echo "Started: ${CONTAINER_NAME}"
}

stop() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "Stopped: ${CONTAINER_NAME}"
}

status() {
  docker ps -a --filter "name=${CONTAINER_NAME}" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'
}

logs() {
  docker logs -f "${CONTAINER_NAME}"
}

health() {
  code="$(curl -s -o /tmp/vllm_health.out -w '%{http_code}' http://127.0.0.1:8000/health || true)"
  echo "health_code=${code}"
}

case "${ACTION}" in
  start)
    start
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  health)
    health
    ;;
  *)
    echo "Unknown action: ${ACTION}"
    exit 1
    ;;
esac
