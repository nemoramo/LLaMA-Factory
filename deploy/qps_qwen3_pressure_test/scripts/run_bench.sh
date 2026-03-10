#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENARIO="${1:-}"
if [ -z "${SCENARIO}" ]; then
  echo "Usage: $0 <warm|cold|burst|sweep|all> [options]"
  exit 1
fi
shift || true

RUN_ENV="${RUN_ENV:-local}"
NETWORK_NAME="${NETWORK_NAME:-sagemaker}"
BENCH_IMAGE="${BENCH_IMAGE:-vllm-bench:latest}"
DATASET="${DATASET:-${ROOT_DIR}/data/bench_endpointing_512.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT_DIR}/results}"
MODEL_PATH="${MODEL_PATH:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-endpointing-qwen3-0.6b-ft}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
BENCH_PORT="${BENCH_PORT:-8000}"
RATES="${RATES:-200,250,300,350,400,500}"
REQUEST_RATE="${REQUEST_RATE:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
NUM_WARMUPS="${NUM_WARMUPS:-}"
BURSTINESS="${BURSTINESS:-0.3}"

usage() {
  cat <<'EOF'
Usage:
  run_bench.sh <warm|cold|burst|sweep|all> [options]

Options:
  --dataset PATH
  --results-dir PATH
  --model-path PATH
  --tokenizer-path PATH
  --served-model-name NAME
  --host HOST
  --port PORT
  --request-rate N
  --max-concurrency N
  --num-prompts N
  --num-warmups N
  --burstiness X
  --rates CSV
  --run-env local|sagemaker
  --network-name NAME
EOF
}

resolve_existing_dir() {
  local value="$1"
  (cd "${value}" && pwd)
}

resolve_existing_file() {
  local value="$1"
  local parent
  parent="$(cd "$(dirname "${value}")" && pwd)"
  printf '%s/%s\n' "${parent}" "$(basename "${value}")"
}

value_or_default() {
  local value="$1"
  local default_value="$2"
  if [ -n "${value}" ]; then
    printf '%s\n' "${value}"
  else
    printf '%s\n' "${default_value}"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --host)
      BENCH_HOST="$2"
      shift 2
      ;;
    --port)
      BENCH_PORT="$2"
      shift 2
      ;;
    --request-rate)
      REQUEST_RATE="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --num-warmups)
      NUM_WARMUPS="$2"
      shift 2
      ;;
    --burstiness)
      BURSTINESS="$2"
      shift 2
      ;;
    --rates)
      RATES="$2"
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

mkdir -p "${RESULTS_DIR}"
RESULTS_DIR="$(resolve_existing_dir "${RESULTS_DIR}")"

if [ -n "${MODEL_PATH}" ]; then
  if [ ! -d "${MODEL_PATH}" ]; then
    echo "Model path not found: ${MODEL_PATH}"
    exit 1
  fi
  MODEL_PATH="$(resolve_existing_dir "${MODEL_PATH}")"
fi

if [ -z "${TOKENIZER_PATH}" ] && [ -n "${MODEL_PATH}" ]; then
  TOKENIZER_PATH="${MODEL_PATH}"
fi

if [ -z "${TOKENIZER_PATH}" ]; then
  echo "Set TOKENIZER_PATH or MODEL_PATH."
  exit 1
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "Tokenizer path not found: ${TOKENIZER_PATH}"
  exit 1
fi
TOKENIZER_PATH="$(resolve_existing_dir "${TOKENIZER_PATH}")"

if [ ! -f "${DATASET}" ]; then
  echo "Dataset not found: ${DATASET}"
  exit 1
fi
DATASET="$(resolve_existing_file "${DATASET}")"
DATASET_DIR="$(dirname "${DATASET}")"
DATASET_FILE="$(basename "${DATASET}")"

run_bench_once() {
  local result_file="$1"
  shift

  docker_args=(
    run
    --rm
    --entrypoint vllm
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
    -v "${DATASET_DIR}:/data:ro"
    -v "${RESULTS_DIR}:/results"
    -v "${TOKENIZER_PATH}:/modeltok:ro"
    "${BENCH_IMAGE}"
    bench
    serve
    --backend openai
    --endpoint /v1/completions
    --host "${BENCH_HOST}"
    --port "${BENCH_PORT}"
    --model "${SERVED_MODEL_NAME}"
    --tokenizer /modeltok
    --served-model-name "${SERVED_MODEL_NAME}"
    --dataset-name custom
    --dataset-path "/data/${DATASET_FILE}"
    --custom-output-len 1
    --skip-chat-template
    --percentile-metrics e2el
    --metric-percentiles 50,90,99
    --goodput e2el:60
    --temperature 0
    --top-p 1
    --top-k 1
    --extra-body '{"structured_outputs":{"choice":["<EOU>","<CONT_USER>","<UNADDRESSED>"]}}'
    --save-result
    --save-detailed
    --result-dir /results
    --result-filename "${result_file}"
    --disable-tqdm
  )

  docker "${docker_args[@]}" "$@"
}

run_warm() {
  local num_warmups
  local num_prompts
  local max_concurrency
  local request_rate
  local result_file

  num_warmups="$(value_or_default "${NUM_WARMUPS}" "2000")"
  num_prompts="$(value_or_default "${NUM_PROMPTS}" "20000")"
  max_concurrency="$(value_or_default "${MAX_CONCURRENCY}" "32")"
  request_rate="$(value_or_default "${REQUEST_RATE}" "800")"
  result_file="warm_c${max_concurrency}_r${request_rate}.json"

  run_bench_once "${result_file}" \
    --num-warmups "${num_warmups}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${max_concurrency}" \
    --request-rate "${request_rate}"
}

run_cold() {
  local num_warmups
  local num_prompts
  local max_concurrency
  local request_rate
  local result_file

  num_warmups="$(value_or_default "${NUM_WARMUPS}" "0")"
  num_prompts="$(value_or_default "${NUM_PROMPTS}" "2000")"
  max_concurrency="$(value_or_default "${MAX_CONCURRENCY}" "16")"
  request_rate="$(value_or_default "${REQUEST_RATE}" "200")"
  result_file="cold_c${max_concurrency}_r${request_rate}.json"

  echo "Tip: cold 测试前请先重启服务。"
  run_bench_once "${result_file}" \
    --num-warmups "${num_warmups}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${max_concurrency}" \
    --request-rate "${request_rate}"
}

run_burst() {
  local num_warmups
  local num_prompts
  local max_concurrency
  local request_rate
  local burstiness
  local burst_tag
  local result_file

  num_warmups="$(value_or_default "${NUM_WARMUPS}" "2000")"
  num_prompts="$(value_or_default "${NUM_PROMPTS}" "20000")"
  max_concurrency="$(value_or_default "${MAX_CONCURRENCY}" "32")"
  request_rate="$(value_or_default "${REQUEST_RATE}" "800")"
  burstiness="$(value_or_default "${BURSTINESS}" "0.3")"
  burst_tag="${burstiness//./}"
  result_file="burst_c${max_concurrency}_r${request_rate}_b${burst_tag}.json"

  run_bench_once "${result_file}" \
    --num-warmups "${num_warmups}" \
    --num-prompts "${num_prompts}" \
    --max-concurrency "${max_concurrency}" \
    --request-rate "${request_rate}" \
    --burstiness "${burstiness}"
}

run_sweep() {
  local num_warmups
  local num_prompts
  local max_concurrency
  IFS=',' read -r -a rates_arr <<< "${RATES}"

  num_warmups="$(value_or_default "${NUM_WARMUPS}" "1000")"
  num_prompts="$(value_or_default "${NUM_PROMPTS}" "10000")"
  max_concurrency="$(value_or_default "${MAX_CONCURRENCY}" "32")"

  for r in "${rates_arr[@]}"; do
    run_bench_once "sweep_c${max_concurrency}_r${r}.json" \
      --num-warmups "${num_warmups}" \
      --num-prompts "${num_prompts}" \
      --max-concurrency "${max_concurrency}" \
      --request-rate "${r}"
  done
}

case "${SCENARIO}" in
  warm)
    run_warm
    ;;
  cold)
    run_cold
    ;;
  burst)
    run_burst
    ;;
  sweep)
    run_sweep
    ;;
  all)
    run_warm
    run_burst
    run_sweep
    echo "All done. cold 场景请在重启服务后单独执行。"
    ;;
  *)
    echo "Unknown scenario: ${SCENARIO}"
    exit 1
    ;;
esac
