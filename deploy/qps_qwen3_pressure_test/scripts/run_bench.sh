#!/usr/bin/env bash
set -euo pipefail

SCENARIO="${1:-}"
if [ -z "${SCENARIO}" ]; then
  echo "Usage: $0 <warm|cold|burst|sweep|all> [options]"
  exit 1
fi
shift || true

BENCH_IMAGE="${BENCH_IMAGE:-vllm-bench:latest}"
DATASET="${DATASET:-/home/sagemaker-user/qps_qwen3_pressure_test/data/bench_endpointing_512.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-/home/sagemaker-user/qps_qwen3_pressure_test/results}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/sagemaker-user/1.0.2}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-endpointing-qwen3-0.6b-ft}"
RATES="${RATES:-200,250,300,350,400,500}"

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
    --tokenizer-path)
      TOKENIZER_PATH="$2"
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --rates)
      RATES="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

mkdir -p "${RESULTS_DIR}"

DATASET_DIR="$(dirname "${DATASET}")"
DATASET_FILE="$(basename "${DATASET}")"

if [ ! -f "${DATASET}" ]; then
  echo "Dataset not found: ${DATASET}"
  exit 1
fi

run_bench_once() {
  local result_file="$1"
  shift

  docker run --rm \
    --network sagemaker \
    --entrypoint vllm \
    -v "${DATASET_DIR}:/data:ro" \
    -v "${RESULTS_DIR}:/results" \
    -v "${TOKENIZER_PATH}:/modeltok:ro" \
    "${BENCH_IMAGE}" bench serve \
    --backend openai \
    --endpoint /v1/completions \
    --host 127.0.0.1 \
    --port 8000 \
    --model "${SERVED_MODEL_NAME}" \
    --tokenizer /modeltok \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --dataset-name custom \
    --dataset-path "/data/${DATASET_FILE}" \
    --custom-output-len 1 \
    --skip-chat-template \
    --percentile-metrics e2el \
    --metric-percentiles 50,90,99 \
    --goodput e2el:60 \
    --temperature 0 \
    --top-p 1 \
    --top-k 1 \
    --extra-body '{"structured_outputs":{"choice":["<EOU>","<CONT_USER>","<UNADDRESSED>"]}}' \
    --save-result \
    --save-detailed \
    --result-dir /results \
    --result-filename "${result_file}" \
    --disable-tqdm \
    "$@"
}

run_warm() {
  run_bench_once "warm_c32_r800.json" \
    --num-warmups 2000 \
    --num-prompts 20000 \
    --max-concurrency 32 \
    --request-rate 800
}

run_cold() {
  echo "Tip: cold 测试前请先重启服务。"
  run_bench_once "cold_c16_r200.json" \
    --num-warmups 0 \
    --num-prompts 2000 \
    --max-concurrency 16 \
    --request-rate 200
}

run_burst() {
  run_bench_once "burst_c32_r800_b03.json" \
    --num-warmups 2000 \
    --num-prompts 20000 \
    --max-concurrency 32 \
    --request-rate 800 \
    --burstiness 0.3
}

run_sweep() {
  IFS=',' read -r -a rates_arr <<< "${RATES}"
  for r in "${rates_arr[@]}"; do
    run_bench_once "sweep_c32_r${r}.json" \
      --num-warmups 1000 \
      --num-prompts 10000 \
      --max-concurrency 32 \
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
