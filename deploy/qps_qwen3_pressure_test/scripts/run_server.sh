#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
if [ -z "${ACTION}" ]; then
  echo "Usage: $0 <start|stop|restart|status|logs|health|wait-ready> [options]"
  exit 1
fi
shift || true

CONTAINER_NAME="${CONTAINER_NAME:-vllm-endpointing}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
MODEL_PATH="${MODEL_PATH:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-endpointing-qwen3-0.6b-ft}"
RUN_ENV="${RUN_ENV:-local}"
NETWORK_NAME="${NETWORK_NAME:-sagemaker}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_IDS="${GPU_IDS:-all}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-300}"
READY_POLL_INTERVAL_SEC="${READY_POLL_INTERVAL_SEC:-5}"

usage() {
  cat <<'EOF'
Usage:
  run_server.sh <start|stop|restart|status|logs|health|wait-ready> [options]

Options:
  --model-path PATH
  --served-model-name NAME
  --host HOST
  --port PORT
  --gpu-ids VALUE              Docker --gpus value. Example: all or device=0
  --run-env local|sagemaker
  --network-name NAME          Only used in sagemaker mode. Default: sagemaker
  --ready-timeout-sec N
  --ready-poll-interval-sec N
EOF
}

resolve_existing_dir() {
  local value="$1"
  (cd "${value}" && pwd)
}

require_container_exists() {
  if ! docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Container not found: ${CONTAINER_NAME}"
    exit 1
  fi
}

require_model_path() {
  if [ -z "${MODEL_PATH}" ]; then
    echo "MODEL_PATH is required for ${ACTION}."
    exit 1
  fi
  if [ ! -d "${MODEL_PATH}" ]; then
    echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}"
    exit 1
  fi
}

resolve_container_network_ip() {
  local container_ip

  require_container_exists
  container_ip="$(
    docker inspect \
      --format "{{with index .NetworkSettings.Networks \"${NETWORK_NAME}\"}}{{.IPAddress}}{{end}}" \
      "${CONTAINER_NAME}" 2>/dev/null || true
  )"
  if [ -z "${container_ip}" ]; then
    echo "Container ${CONTAINER_NAME} is not attached to Docker network ${NETWORK_NAME}."
    exit 1
  fi

  printf '%s\n' "${container_ip}"
}

resolve_local_health_host() {
  case "${VLLM_HOST}" in
    ""|0.0.0.0|::|[::])
      printf '127.0.0.1\n'
      ;;
    *)
      printf '%s\n' "${VLLM_HOST}"
      ;;
  esac
}

resolve_health_url() {
  local health_host

  case "${RUN_ENV}" in
    local)
      health_host="$(resolve_local_health_host)"
      ;;
    sagemaker)
      health_host="$(resolve_container_network_ip)"
      ;;
    *)
      echo "Unsupported RUN_ENV: ${RUN_ENV} (expected local or sagemaker)"
      exit 1
      ;;
  esac

  printf 'http://%s:%s/health\n' "${health_host}" "${VLLM_PORT}"
}

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
    --host)
      VLLM_HOST="$2"
      shift 2
      ;;
    --port)
      VLLM_PORT="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --run-env)
      RUN_ENV="$2"
      shift 2
      ;;
    --network-name)
      NETWORK_NAME="$2"
      shift 2
      ;;
    --ready-timeout-sec)
      READY_TIMEOUT_SEC="$2"
      shift 2
      ;;
    --ready-poll-interval-sec)
      READY_POLL_INTERVAL_SEC="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

start() {
  require_model_path
  MODEL_PATH="$(resolve_existing_dir "${MODEL_PATH}")"
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

  docker_args=(
    run
    -d
    --name "${CONTAINER_NAME}"
    --gpus "${GPU_IDS}"
  )

  case "${RUN_ENV}" in
    local)
      docker_args+=(--network host)
      ;;
    sagemaker)
      docker_args+=(--network "${NETWORK_NAME}")
      ;;
    *)
      echo "Unsupported RUN_ENV: ${RUN_ENV} (expected local or sagemaker)"
      exit 1
      ;;
  esac

  docker_args+=(
    -v "${MODEL_PATH}:/model:ro"
    "${IMAGE}"
    /model
    --host "${VLLM_HOST}"
    --port "${VLLM_PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --enable-prefix-caching
    --generation-config vllm
  )

  docker "${docker_args[@]}"

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
  local health_url
  local code

  health_url="$(resolve_health_url)"
  code="$(curl -s -o /tmp/vllm_health.out -w '%{http_code}' "${health_url}" || true)"
  echo "health_url=${health_url}"
  echo "health_code=${code}"
}

wait_ready() {
  local attempt=0
  local max_attempts
  local health_url
  local code=""

  health_url="$(resolve_health_url)"
  max_attempts=$(( (READY_TIMEOUT_SEC + READY_POLL_INTERVAL_SEC - 1) / READY_POLL_INTERVAL_SEC ))
  if [ "${max_attempts}" -le 0 ]; then
    echo "READY_TIMEOUT_SEC must be positive."
    exit 1
  fi

  while [ "${attempt}" -lt "${max_attempts}" ]; do
    attempt=$((attempt + 1))
    code="$(curl -s -o /tmp/vllm_health.out -w '%{http_code}' "${health_url}" || true)"
    echo "health_url=${health_url}"
    echo "health_attempt=${attempt}"
    echo "health_code=${code}"
    if [ "${code}" = "200" ]; then
      return 0
    fi
    sleep "${READY_POLL_INTERVAL_SEC}"
  done

  echo "Timed out waiting for vLLM readiness after ${READY_TIMEOUT_SEC}s."
  exit 1
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
  wait-ready)
    wait_ready
    ;;
  *)
    echo "Unknown action: ${ACTION}"
    exit 1
    ;;
esac
