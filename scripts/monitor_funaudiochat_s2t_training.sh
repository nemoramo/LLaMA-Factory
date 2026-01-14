#!/bin/bash

################################################################################
# FunAudioChat S2T Training Watchdog
# - Runs training in the foreground (ideal for a byobu tab)
# - If training exits (OOM / disconnect / crash), it restarts automatically
# - If output_dir has checkpoints and overwrite_output_dir=false, LLaMA-Factory
#   will auto-resume from the latest checkpoint.
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${WORK_DIR:-${DEFAULT_WORK_DIR}}"
CONFIG_FILE="${CONFIG_FILE:-examples/funaudiochat/funaudiochat_s2t_sft_full.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-llamafactory}"

PYTHONPATH_OVERRIDE="${WORK_DIR}/src"
if [[ -n "${PYTHONPATH:-}" ]]; then
  PYTHONPATH_OVERRIDE="${PYTHONPATH_OVERRIDE}:${PYTHONPATH}"
fi

GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

DATASET_DIR="${DATASET_DIR:-/data2/mayufeng/manifests/llama_data}"
DATASET="${DATASET:-funaudiochat_asr_v9_train,funaudiochat_africa_fr_en_train,funaudiochat_spgispeech_train,funaudiochat_hindi_english_v3_train}"
EVAL_DATASET="${EVAL_DATASET:-gemma3n_asr_hausa_youtube_test_norm_text}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-30}"
EVAL_STEPS="${EVAL_STEPS:-500}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"

OUTPUT_DIR="${OUTPUT_DIR:-/data2/mayufeng/llamafactory_saves/funaudiochat/s2t_lora_alldata_e3}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3.0}"
MAX_STEPS="${MAX_STEPS:-0}"

# Dynamic prompt packing knobs (helps avoid GPU idle due to huge CPU-side buffers).
DYNAMIC_PROMPT_LAZY_ALIGN="${DYNAMIC_PROMPT_LAZY_ALIGN:-true}"
DYNAMIC_PROMPT_PACKING_BUFFER_SIZE="${DYNAMIC_PROMPT_PACKING_BUFFER_SIZE:-2048}"
DYNAMIC_PROMPT_PACKING_LOG_INTERVAL="${DYNAMIC_PROMPT_PACKING_LOG_INTERVAL:-10}"
DYNAMIC_PROMPT_PACKING_GLOBAL_SHUFFLE="${DYNAMIC_PROMPT_PACKING_GLOBAL_SHUFFLE:-false}"
DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS="${DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS:-4}"
DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS="${DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS:-2}"

# Target ~60GB VRAM on H20; adjust if you see frequent OOMs.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

CUTOFF_LEN="${CUTOFF_LEN:-2048}"
PACKING="${PACKING:-false}"
NEAT_PACKING="${NEAT_PACKING:-false}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"

LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
LOAD_BEST_MODEL_AT_END="${LOAD_BEST_MODEL_AT_END:-false}"
METRIC_FOR_BEST_MODEL="${METRIC_FOR_BEST_MODEL:-eval_loss}"
GREATER_IS_BETTER="${GREATER_IS_BETTER:-}"
INIT_ADAPTER_NAME_OR_PATH="${INIT_ADAPTER_NAME_OR_PATH:-}"
IGNORE_DATA_SKIP="${IGNORE_DATA_SKIP:-true}"
CREATE_NEW_ADAPTER="${CREATE_NEW_ADAPTER:-false}"
FUNAUDIOCHAT_FULL_AUDIO_TUNING="${FUNAUDIOCHAT_FULL_AUDIO_TUNING:-false}"

MAX_RESTARTS="${MAX_RESTARTS:-999999}"
RESTART_SLEEP_SECONDS="${RESTART_SLEEP_SECONDS:-10}"
OOM_BACKOFF_RATIO="${OOM_BACKOFF_RATIO:-0.85}"
MIN_PER_DEVICE_TRAIN_BATCH_SIZE="${MIN_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"

LOG_DIR="${OUTPUT_DIR}/watchdog_logs"
mkdir -p "${LOG_DIR}"

log() {
  local level="$1"
  shift
  local msg="$*"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] [${level}] ${msg}" | tee -a "${LOG_DIR}/watchdog.log"
}

is_oom_log() {
  local file="$1"
  rg -n "(CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED)" "${file}" >/dev/null 2>&1
}

maybe_backoff_bs_on_oom() {
  if [[ "${PER_DEVICE_TRAIN_BATCH_SIZE}" -le "${MIN_PER_DEVICE_TRAIN_BATCH_SIZE}" ]]; then
    return 0
  fi
  local new_bs
  new_bs="$(python - <<PY
import math
bs=int(${PER_DEVICE_TRAIN_BATCH_SIZE})
ratio=float(${OOM_BACKOFF_RATIO})
new=max(int(math.floor(bs*ratio)), bs-1, int(${MIN_PER_DEVICE_TRAIN_BATCH_SIZE}))
print(new)
PY
)"
  if [[ "${new_bs}" -lt "${PER_DEVICE_TRAIN_BATCH_SIZE}" ]]; then
    log "WARN" "OOM detected: backing off per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} -> ${new_bs}"
    PER_DEVICE_TRAIN_BATCH_SIZE="${new_bs}"
  fi
}

run_once() {
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  local train_log="${LOG_DIR}/train_${run_id}.log"

  local adapter_arg=()
  if [[ -n "${INIT_ADAPTER_NAME_OR_PATH}" ]]; then
    if ls -d "${OUTPUT_DIR}/checkpoint-"* >/dev/null 2>&1; then
      log "INFO" "Found existing checkpoints in ${OUTPUT_DIR}; skip INIT_ADAPTER_NAME_OR_PATH."
    else
      adapter_arg=(adapter_name_or_path="${INIT_ADAPTER_NAME_OR_PATH}")
    fi
  fi

  local max_steps_arg=()
  if [[ "${MAX_STEPS}" != "0" && "${MAX_STEPS}" != "" ]]; then
    max_steps_arg=(max_steps="${MAX_STEPS}" num_train_epochs=1.0)
  else
    max_steps_arg=(num_train_epochs="${NUM_TRAIN_EPOCHS}")
  fi

  local packing_args=()
  if [[ "${PACKING}" == "true" ]]; then
    packing_args+=(packing=true)
  fi
  if [[ "${NEAT_PACKING}" == "true" ]]; then
    packing_args+=(neat_packing=true)
  fi

  local greater_is_better_arg=()
  if [[ -n "${GREATER_IS_BETTER}" ]]; then
    greater_is_better_arg=(greater_is_better="${GREATER_IS_BETTER}")
  fi

  log "INFO" "Launching training (run_id=${run_id})"
  log "INFO" "GPUS=${GPUS} NPROC_PER_NODE=${NPROC_PER_NODE} OUTPUT_DIR=${OUTPUT_DIR}"
  log "INFO" "DATASET=${DATASET}"
  log "INFO" "cutoff_len=${CUTOFF_LEN} packing=${PACKING} neat_packing=${NEAT_PACKING}"
  log "INFO" "per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} grad_acc=${GRADIENT_ACCUMULATION_STEPS} lr=${LEARNING_RATE}"
  if [[ "${MAX_STEPS}" != "0" && "${MAX_STEPS}" != "" ]]; then
    log "INFO" "max_steps=${MAX_STEPS} (epoch semantics may vary under packing)"
  else
    log "INFO" "epochs=${NUM_TRAIN_EPOCHS}"
  fi

  mkdir -p "${OUTPUT_DIR}"

  local config_src="${CONFIG_FILE}"
  if [[ "${config_src}" != /* ]]; then
    config_src="${WORK_DIR}/${config_src}"
  fi
  if [[ -f "${config_src}" ]]; then
    cp -f "${config_src}" "${OUTPUT_DIR}/config_base.yaml"
  else
    log "WARN" "Config file not found: ${config_src} (skip saving config_base.yaml)"
  fi

  cat > "${OUTPUT_DIR}/training_command.txt" <<EOF
Saved at: $(date)
Command:
  conda run -n ${CONDA_ENV_NAME} --no-capture-output \\
    env CUDA_VISIBLE_DEVICES=${GPUS} FORCE_TORCHRUN=1 NPROC_PER_NODE=${NPROC_PER_NODE} PYTORCH_ALLOC_CONF=expandable_segments:True \\
      PYTHONPATH=${PYTHONPATH_OVERRIDE} PYTHONNOUSERSITE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false \\
      llamafactory-cli train ${CONFIG_FILE} \\
        dataset=${DATASET} dataset_dir=${DATASET_DIR} dynamic_prompt_sampling=true dynamic_prompt_lazy_align=${DYNAMIC_PROMPT_LAZY_ALIGN} \\
        dynamic_prompt_packing_buffer_size=${DYNAMIC_PROMPT_PACKING_BUFFER_SIZE} dynamic_prompt_packing_log_interval=${DYNAMIC_PROMPT_PACKING_LOG_INTERVAL} \\
        dynamic_prompt_packing_global_shuffle=${DYNAMIC_PROMPT_PACKING_GLOBAL_SHUFFLE} \\
        dynamic_prompt_packing_prefetch_buffers=${DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS} \\
        dynamic_prompt_packing_carryover_packs=${DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS} \\
        cutoff_len=${CUTOFF_LEN} ${packing_args[*]} \\
        ignore_data_skip=${IGNORE_DATA_SKIP} \\
        create_new_adapter=${CREATE_NEW_ADAPTER} funaudiochat_full_audio_tuning=${FUNAUDIOCHAT_FULL_AUDIO_TUNING} \\
        eval_dataset=${EVAL_DATASET} eval_strategy=steps eval_steps=${EVAL_STEPS} per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE} \\
        predict_with_generate=true compute_wer_cer=true eval_num_samples=${EVAL_NUM_SAMPLES} eval_loss_on_full_dataset=false \\
        do_sample=false temperature=0.0 top_p=1.0 num_beams=1 max_new_tokens=${EVAL_MAX_NEW_TOKENS} \\
        overwrite_cache=false preprocessing_num_workers=${PREPROCESSING_NUM_WORKERS} \\
        dataloader_num_workers=${DATALOADER_NUM_WORKERS} dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR} \\
        finetuning_type=lora lora_rank=8 lora_alpha=16 lora_dropout=0.05 ${adapter_arg[*]} \\
        output_dir=${OUTPUT_DIR} overwrite_output_dir=false \\
        learning_rate=${LEARNING_RATE} lr_scheduler_type=${LR_SCHEDULER_TYPE} warmup_ratio=${WARMUP_RATIO} \\
        ${max_steps_arg[*]} \\
        per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \\
        logging_steps=${LOGGING_STEPS} \\
        eval_strategy=steps eval_steps=${EVAL_STEPS} \\
        save_strategy=steps save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT} \\
        load_best_model_at_end=${LOAD_BEST_MODEL_AT_END} metric_for_best_model=${METRIC_FOR_BEST_MODEL} ${greater_is_better_arg[*]}
EOF

  cd "${WORK_DIR}"
  set +e
  stdbuf -oL -eL conda run -n "${CONDA_ENV_NAME}" --no-capture-output \
    env CUDA_VISIBLE_DEVICES="${GPUS}" \
    FORCE_TORCHRUN=1 \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${PYTHONPATH_OVERRIDE}" \
    PYTHONNOUSERSITE=1 \
    DISABLE_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    llamafactory-cli train "${CONFIG_FILE}" \
    dataset="${DATASET}" \
    dataset_dir="${DATASET_DIR}" \
	    dynamic_prompt_sampling=true \
	    dynamic_prompt_lazy_align="${DYNAMIC_PROMPT_LAZY_ALIGN}" \
	    dynamic_prompt_packing_buffer_size="${DYNAMIC_PROMPT_PACKING_BUFFER_SIZE}" \
	    dynamic_prompt_packing_log_interval="${DYNAMIC_PROMPT_PACKING_LOG_INTERVAL}" \
	    dynamic_prompt_packing_global_shuffle="${DYNAMIC_PROMPT_PACKING_GLOBAL_SHUFFLE}" \
	    dynamic_prompt_packing_prefetch_buffers="${DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS}" \
	    dynamic_prompt_packing_carryover_packs="${DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS}" \
	    cutoff_len="${CUTOFF_LEN}" \
	    "${packing_args[@]}" \
	    ignore_data_skip="${IGNORE_DATA_SKIP}" \
	    create_new_adapter="${CREATE_NEW_ADAPTER}" \
	    funaudiochat_full_audio_tuning="${FUNAUDIOCHAT_FULL_AUDIO_TUNING}" \
	    eval_dataset="${EVAL_DATASET}" \
    eval_strategy=steps \
    eval_steps="${EVAL_STEPS}" \
    per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE}" \
    predict_with_generate=true \
    compute_wer_cer=true \
    eval_num_samples="${EVAL_NUM_SAMPLES}" \
    eval_loss_on_full_dataset=false \
    do_sample=false \
    temperature=0.0 \
    top_p=1.0 \
    num_beams=1 \
    max_new_tokens="${EVAL_MAX_NEW_TOKENS}" \
    overwrite_cache=false \
    preprocessing_num_workers="${PREPROCESSING_NUM_WORKERS}" \
    dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
    finetuning_type=lora \
    lora_rank=8 \
    lora_alpha=16 \
    lora_dropout=0.05 \
    "${adapter_arg[@]}" \
    output_dir="${OUTPUT_DIR}" \
    overwrite_output_dir=false \
    learning_rate="${LEARNING_RATE}" \
    lr_scheduler_type="${LR_SCHEDULER_TYPE}" \
    warmup_ratio="${WARMUP_RATIO}" \
    "${max_steps_arg[@]}" \
    per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}" \
    logging_steps="${LOGGING_STEPS}" \
    save_strategy=steps \
    save_steps="${SAVE_STEPS}" \
    save_total_limit="${SAVE_TOTAL_LIMIT}" \
    load_best_model_at_end="${LOAD_BEST_MODEL_AT_END}" \
    metric_for_best_model="${METRIC_FOR_BEST_MODEL}" \
    "${greater_is_better_arg[@]}" \
    2>&1 | tee -a "${train_log}"
  local rc="${PIPESTATUS[0]}"
  set -e

  if [[ "${rc}" -eq 0 ]]; then
    log "INFO" "Training finished successfully (run_id=${run_id})."
    return 0
  fi

  log "ERROR" "Training exited with code ${rc} (run_id=${run_id})."
  if is_oom_log "${train_log}"; then
    maybe_backoff_bs_on_oom
  fi
  return "${rc}"
}

main() {
  log "INFO" "FunAudioChat S2T watchdog started."
  log "INFO" "Logs: ${LOG_DIR}"

  local restarts=0
  while [[ "${restarts}" -lt "${MAX_RESTARTS}" ]]; do
    if run_once; then
      exit 0
    fi
    restarts=$((restarts + 1))
    log "WARN" "Restarting training in ${RESTART_SLEEP_SECONDS}s (restart #${restarts}/${MAX_RESTARTS})..."
    sleep "${RESTART_SLEEP_SECONDS}"
  done

  log "ERROR" "Reached MAX_RESTARTS=${MAX_RESTARTS}, exiting."
  exit 2
}

main "$@"
