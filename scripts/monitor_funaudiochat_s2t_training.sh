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
ENABLE_EVAL="${ENABLE_EVAL:-true}"
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
# Override the auto-sharding heuristic used by dynamic prompt packing.
# - 0 means auto (default behavior)
DYNAMIC_PROMPT_PACKING_NUM_SHARDS="${DYNAMIC_PROMPT_PACKING_NUM_SHARDS:-0}"

# Sharded parquet backend (optional; avoids building huge HF map-style JSONL index).
# Requires `packing=true` and `max_steps>0`.
#
# SHARDED_DATASET_BACKEND: Backend type for sharded dataset loading.
#   - off (default): Disable sharded backend, use standard HuggingFace datasets.
#   - polars_parquet_shards: Use Polars to stream-read Parquet shards.
#
# SHARDED_MANIFEST_PATH: Path to the JSONL manifest file listing all shards.
#   Each line should contain {"path": "/path/to/shard.parquet", "num_rows": N}.
#   Required when SHARDED_DATASET_BACKEND != off.
#
# SHARDED_INPUT_ALIGNED: Whether samples are pre-aligned to shard boundaries.
#   - false (default): Samples may span across shards (requires cross-shard handling).
#   - true: Each sample is contained within a single shard (faster, requires pre-processing).
#
# SHARDED_SHUFFLE_SHARDS: Whether to shuffle the order of shards at epoch start.
#   - true (default): Randomize shard read order each epoch.
#   - false: Read shards in manifest order.
#
# SHARDED_ROW_SHUFFLE_BUFFER: Row-level shuffle buffer size (reservoir sampling).
#   - 0 (default): Disable row shuffle (deterministic, fastest, lowest memory).
#   - N > 0: Enable randomization within a buffer of N rows (higher memory usage).
#
# SHARDED_PARQUET_BATCH_ROWS: Number of rows to read per batch from Parquet.
#   - 8192 (default): Balanced for memory and throughput.
#   - Smaller (e.g., 4096): Use when memory is tight or rows are large.
#   - Larger (e.g., 16384): Use when memory is abundant for higher throughput.
#
# SHARDED_PREFETCH_NEXT_SHARD: Prefetch the next shard in a background thread (per dataloader worker).
#   - true (default): Helps hide stalls at shard boundaries in DDP.
#   - false: Disable next-shard prefetch.
#
# SHARDED_PREFETCH_QUEUE_BATCHES: Max number of parquet RecordBatches to prefetch for the next shard.
#   - 1 (default): Minimal extra RAM, still hides open/first-read stalls.
#   - 2-4: More aggressive prefetch; use only if RAM allows.
#
# SHARDED_PREFETCH_LOG: Log shard prefetch events (rank0) for debugging.
#
SHARDED_DATASET_BACKEND="${SHARDED_DATASET_BACKEND:-off}" # off | polars_parquet_shards
SHARDED_MANIFEST_PATH="${SHARDED_MANIFEST_PATH:-}"
SHARDED_INPUT_ALIGNED="${SHARDED_INPUT_ALIGNED:-false}"
SHARDED_SHUFFLE_SHARDS="${SHARDED_SHUFFLE_SHARDS:-true}"
SHARDED_ROW_SHUFFLE_BUFFER="${SHARDED_ROW_SHUFFLE_BUFFER:-0}"
SHARDED_PARQUET_BATCH_ROWS="${SHARDED_PARQUET_BATCH_ROWS:-8192}"
SHARDED_PREFETCH_NEXT_SHARD="${SHARDED_PREFETCH_NEXT_SHARD:-true}"
SHARDED_PREFETCH_QUEUE_BATCHES="${SHARDED_PREFETCH_QUEUE_BATCHES:-1}"
SHARDED_PREFETCH_LOG="${SHARDED_PREFETCH_LOG:-false}"
#
# SHARDED_RESUME_MODE: coarse resume for sharded parquet backend.
#   - off (default): disable shard-boundary resume state.
#   - shard_boundary: persist per-rank/worker shard cursors; restart skips completed shards (may repeat within last shard).
# Note: if enabled, prefer `IGNORE_DATA_SKIP=true` to avoid Trainer-level skipping + double-skips.
#
# SHARDED_RESUME_STATE_DIR: where to write resume state json files (default: <output_dir>/shard_resume_state).
# SHARDED_RESUME_PREFER_CHECKPOINT: prefer loading state from <checkpoint_dir>/shard_resume_state if present.
# SHARDED_RESUME_LOG: enable rank0 resume logging.
#
SHARDED_RESUME_MODE="${SHARDED_RESUME_MODE:-off}" # off | shard_boundary
SHARDED_RESUME_STATE_DIR="${SHARDED_RESUME_STATE_DIR:-}"
SHARDED_RESUME_PREFER_CHECKPOINT="${SHARDED_RESUME_PREFER_CHECKPOINT:-true}"
SHARDED_RESUME_LOG="${SHARDED_RESUME_LOG:-false}"

# Perf logging (optional; disabled by default).
LLAMAFACTORY_PERF_LOG="${LLAMAFACTORY_PERF_LOG:-0}"
LLAMAFACTORY_DATALOADER_PERF_LOG="${LLAMAFACTORY_DATALOADER_PERF_LOG:-0}"
LLAMAFACTORY_DYNAMIC_PACKING_PERF_META="${LLAMAFACTORY_DYNAMIC_PACKING_PERF_META:-0}"
LLAMAFACTORY_PRESERVE_AUDIO_META="${LLAMAFACTORY_PRESERVE_AUDIO_META:-0}"
LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS="${LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS:-0}"

# Optional: bypass fuse mounts (e.g. /mnt/asr-audio-data/...) by mapping to tos:// URIs and using S3-compatible APIs.
LLAMAFACTORY_TOS_SDK_FOR_MOUNT="${LLAMAFACTORY_TOS_SDK_FOR_MOUNT:-0}"
LLAMAFACTORY_TOS_MOUNT_MAP="${LLAMAFACTORY_TOS_MOUNT_MAP:-}"
LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS="${LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS:-}"
LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS="${LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS:-}"

# Target ~60GB VRAM on H20; adjust if you see frequent OOMs.
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.02}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"

# LoRA knobs (keep defaults aligned with prior behavior).
LORA_TARGET="${LORA_TARGET:-}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"

# Regularization knobs (optional; omitted when empty).
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
LABEL_SMOOTHING_FACTOR="${LABEL_SMOOTHING_FACTOR:-}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"

CUTOFF_LEN="${CUTOFF_LEN:-2048}"
PACKING="${PACKING:-false}"
NEAT_PACKING="${NEAT_PACKING:-false}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"

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

# Token throughput stats (requires Transformers support).
INCLUDE_NUM_INPUT_TOKENS_SEEN="${INCLUDE_NUM_INPUT_TOKENS_SEEN:-true}"
INCLUDE_TOKENS_PER_SECOND="${INCLUDE_TOKENS_PER_SECOND:-true}"

# Audio epoch logging (optional; can be expensive to pre-scan durations on huge datasets).
# - When set to "true"/"false", passed as `log_audio_epochs=...` to LLaMA-Factory.
# - When unset, falls back to config/default (`DataArguments.log_audio_epochs`).
LOG_AUDIO_EPOCHS="${LOG_AUDIO_EPOCHS:-}"

# When IGNORE_DATA_SKIP=true, changing data_seed on watchdog restarts helps avoid
# repeating the same initial batches after resuming from a checkpoint.
AUTO_DATA_SEED_ON_RESTART="${AUTO_DATA_SEED_ON_RESTART:-true}"
DATA_SEED_BASE="${DATA_SEED_BASE:-}"
DATA_SEED_STEP="${DATA_SEED_STEP:-1}"
SCRIPT_START_EPOCH="$(date +%s)"

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

ENABLE_SYS_MONITOR="${ENABLE_SYS_MONITOR:-false}"
SYS_MONITOR_INTERVAL_SEC="${SYS_MONITOR_INTERVAL_SEC:-30}"
SYS_MONITOR_TOP_N="${SYS_MONITOR_TOP_N:-20}"
# Default to matching the unique `output_dir=...` argument so concurrent watchdogs don't mix process trees.
SYS_MONITOR_MATCH="${SYS_MONITOR_MATCH:-output_dir=${OUTPUT_DIR}}"

start_sys_monitor() {
  local run_id="$1"
  local mem_log="${LOG_DIR}/mem_${run_id}.log"
  (
    echo "[start] $(date -Is)"
    echo "interval_sec=${SYS_MONITOR_INTERVAL_SEC} top_n=${SYS_MONITOR_TOP_N} match=${SYS_MONITOR_MATCH}"
    while true; do
      echo "=== $(date -Is) ==="
      free -h || true
      echo "--- nvidia-smi ---"
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
      echo "--- rss summary (match) ---"
      # Match against full command line. Keep the computation separate from the top-N listing.
      local rss_total_kb
      rss_total_kb="$(ps -eo rss,cmd --no-headers | rg "${SYS_MONITOR_MATCH}" | awk '{sum+=$1} END{print sum+0}')"
      local proc_count
      proc_count="$(ps -eo rss,cmd --no-headers | rg "${SYS_MONITOR_MATCH}" | wc -l | tr -d ' ')"
      local rss_total_gib
      rss_total_gib="$(awk -v x="${rss_total_kb}" 'BEGIN{printf "%.2f", (x/1024.0/1024.0)}')"
      echo "procs=${proc_count} rss_total_kb=${rss_total_kb} rss_total_gib=${rss_total_gib}"
      echo "--- top rss (${SYS_MONITOR_TOP_N}) ---"
      ps -eo pid,rss,cmd --sort=-rss | rg -n "${SYS_MONITOR_MATCH}" | head -n "${SYS_MONITOR_TOP_N}" || true
      sleep "${SYS_MONITOR_INTERVAL_SEC}"
    done
  ) >>"${mem_log}" 2>&1 &
  echo "$!"
}

is_oom_log() {
  local file="$1"
  rg -n "(CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED)" "${file}" >/dev/null 2>&1
}

is_integer() {
  [[ "$1" =~ ^-?[0-9]+$ ]]
}

maybe_backoff_bs_on_oom() {
  if [[ "${PER_DEVICE_TRAIN_BATCH_SIZE}" -le "${MIN_PER_DEVICE_TRAIN_BATCH_SIZE}" ]]; then
    return 0
  fi
  local new_bs
  new_bs="$(python3 - <<PY
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
  local restart_idx="${1:-0}"
  local run_id
  run_id="$(date +%Y%m%d_%H%M%S)"
  local train_log="${LOG_DIR}/train_${run_id}.log"

  local mon_pid=""
  _cleanup_run_once() {
    if [[ -n "${mon_pid}" ]]; then
      kill "${mon_pid}" 2>/dev/null || true
      wait "${mon_pid}" 2>/dev/null || true
      log "INFO" "Stopped sys monitor pid=${mon_pid}."
      mon_pid=""
    fi
  }
  trap _cleanup_run_once RETURN

  if [[ "${ENABLE_SYS_MONITOR}" == "true" ]]; then
    mon_pid="$(start_sys_monitor "${run_id}")"
    log "INFO" "Started sys monitor pid=${mon_pid} (ENABLE_SYS_MONITOR=true)."
  fi

  local sharded_args=()
  if [[ "${SHARDED_DATASET_BACKEND}" != "off" ]]; then
    if [[ -z "${SHARDED_MANIFEST_PATH}" ]]; then
      log "ERROR" "SHARDED_DATASET_BACKEND=${SHARDED_DATASET_BACKEND} but SHARDED_MANIFEST_PATH is empty."
      return 2
    fi
    sharded_args+=("sharded_dataset_backend=${SHARDED_DATASET_BACKEND}")
    sharded_args+=("sharded_manifest_path=${SHARDED_MANIFEST_PATH}")
    sharded_args+=("sharded_input_aligned=${SHARDED_INPUT_ALIGNED}")
    sharded_args+=("sharded_shuffle_shards=${SHARDED_SHUFFLE_SHARDS}")
    sharded_args+=("sharded_row_shuffle_buffer=${SHARDED_ROW_SHUFFLE_BUFFER}")
    sharded_args+=("sharded_parquet_batch_rows=${SHARDED_PARQUET_BATCH_ROWS}")
    sharded_args+=("sharded_prefetch_next_shard=${SHARDED_PREFETCH_NEXT_SHARD}")
    sharded_args+=("sharded_prefetch_queue_batches=${SHARDED_PREFETCH_QUEUE_BATCHES}")
    sharded_args+=("sharded_prefetch_log=${SHARDED_PREFETCH_LOG}")
    sharded_args+=("sharded_resume_mode=${SHARDED_RESUME_MODE}")
    if [[ -n "${SHARDED_RESUME_STATE_DIR}" ]]; then
      sharded_args+=("sharded_resume_state_dir=${SHARDED_RESUME_STATE_DIR}")
    fi
    sharded_args+=("sharded_resume_prefer_checkpoint=${SHARDED_RESUME_PREFER_CHECKPOINT}")
    sharded_args+=("sharded_resume_log=${SHARDED_RESUME_LOG}")
  fi

  local data_seed_arg=()
  if [[ "${AUTO_DATA_SEED_ON_RESTART}" == "true" && "${IGNORE_DATA_SKIP}" == "true" ]]; then
    local base="${DATA_SEED_BASE}"
    if [[ -z "${base}" ]]; then
      base="${SCRIPT_START_EPOCH}"
    fi
    if ! is_integer "${base}"; then
      log "WARN" "Invalid DATA_SEED_BASE=${base}, fallback to script start epoch."
      base="${SCRIPT_START_EPOCH}"
    fi

    local step="${DATA_SEED_STEP}"
    if ! is_integer "${step}" || [[ "${step}" -le 0 ]]; then
      log "WARN" "Invalid DATA_SEED_STEP=${step}, fallback to 1."
      step=1
    fi

    local data_seed=$((base + restart_idx * step))
    data_seed_arg=(data_seed="${data_seed}")
    log "INFO" "Using data_seed=${data_seed} (base=${base} step=${step} restart=${restart_idx})"
  fi

  local token_stats_args=()
  if [[ "${INCLUDE_NUM_INPUT_TOKENS_SEEN}" == "true" ]]; then
    token_stats_args+=(include_num_input_tokens_seen=true)
  fi
  if [[ "${INCLUDE_TOKENS_PER_SECOND}" == "true" ]]; then
    token_stats_args+=(include_tokens_per_second=true)
  fi

  local audio_epoch_args=()
  if [[ -n "${LOG_AUDIO_EPOCHS}" ]]; then
    audio_epoch_args+=(log_audio_epochs="${LOG_AUDIO_EPOCHS}")
  fi

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

  local eval_args=()
  local load_best_model_at_end_arg="${LOAD_BEST_MODEL_AT_END}"
  if [[ "${ENABLE_EVAL}" == "true" ]]; then
    if [[ -z "${EVAL_DATASET}" ]]; then
      log "WARN" "ENABLE_EVAL=true but EVAL_DATASET is empty; disabling eval."
      load_best_model_at_end_arg="false"
    else
      eval_args+=("eval_dataset=${EVAL_DATASET}")
      eval_args+=("eval_strategy=steps")
      eval_args+=("eval_steps=${EVAL_STEPS}")
      eval_args+=("per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}")
      eval_args+=("predict_with_generate=true")
      eval_args+=("compute_wer_cer=true")
      eval_args+=("eval_num_samples=${EVAL_NUM_SAMPLES}")
      eval_args+=("eval_loss_on_full_dataset=false")
      eval_args+=("do_sample=false")
      eval_args+=("temperature=0.0")
      eval_args+=("top_p=1.0")
      eval_args+=("num_beams=1")
      eval_args+=("max_new_tokens=${EVAL_MAX_NEW_TOKENS}")
    fi
  else
    if [[ "${LOAD_BEST_MODEL_AT_END}" == "true" ]]; then
      log "WARN" "ENABLE_EVAL=false: forcing load_best_model_at_end=false."
    fi
    load_best_model_at_end_arg="false"
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

  local lora_target_arg=()
  if [[ -n "${LORA_TARGET}" ]]; then
    lora_target_arg=(lora_target="${LORA_TARGET}")
  fi

  local reg_args=()
  if [[ -n "${WEIGHT_DECAY}" ]]; then
    reg_args+=(weight_decay="${WEIGHT_DECAY}")
  fi
  if [[ -n "${LABEL_SMOOTHING_FACTOR}" ]]; then
    reg_args+=(label_smoothing_factor="${LABEL_SMOOTHING_FACTOR}")
  fi
  if [[ -n "${MAX_GRAD_NORM}" ]]; then
    reg_args+=(max_grad_norm="${MAX_GRAD_NORM}")
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
	      LLAMAFACTORY_PERF_LOG=${LLAMAFACTORY_PERF_LOG} LLAMAFACTORY_DATALOADER_PERF_LOG=${LLAMAFACTORY_DATALOADER_PERF_LOG} \\
	      LLAMAFACTORY_DYNAMIC_PACKING_PERF_META=${LLAMAFACTORY_DYNAMIC_PACKING_PERF_META} LLAMAFACTORY_PRESERVE_AUDIO_META=${LLAMAFACTORY_PRESERVE_AUDIO_META} \\
	      LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS=${LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS} \\
	      LLAMAFACTORY_TOS_SDK_FOR_MOUNT=${LLAMAFACTORY_TOS_SDK_FOR_MOUNT} LLAMAFACTORY_TOS_MOUNT_MAP=${LLAMAFACTORY_TOS_MOUNT_MAP} \\
	      LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS=${LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS} LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS=${LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS} \\
	      PYTHONPATH=${PYTHONPATH_OVERRIDE} PYTHONNOUSERSITE=1 DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false \\
	      llamafactory-cli train ${CONFIG_FILE} \\
	        dataset=${DATASET} dataset_dir=${DATASET_DIR} dynamic_prompt_sampling=true dynamic_prompt_lazy_align=${DYNAMIC_PROMPT_LAZY_ALIGN} \\
	        dynamic_prompt_packing_buffer_size=${DYNAMIC_PROMPT_PACKING_BUFFER_SIZE} dynamic_prompt_packing_log_interval=${DYNAMIC_PROMPT_PACKING_LOG_INTERVAL} \\
	        dynamic_prompt_packing_global_shuffle=${DYNAMIC_PROMPT_PACKING_GLOBAL_SHUFFLE} \\
	        dynamic_prompt_packing_prefetch_buffers=${DYNAMIC_PROMPT_PACKING_PREFETCH_BUFFERS} \\
	        dynamic_prompt_packing_carryover_packs=${DYNAMIC_PROMPT_PACKING_CARRYOVER_PACKS} \\
	        dynamic_prompt_packing_num_shards=${DYNAMIC_PROMPT_PACKING_NUM_SHARDS} \\
	        ${sharded_args[*]} \\
	        cutoff_len=${CUTOFF_LEN} ${packing_args[*]} \\
	        ignore_data_skip=${IGNORE_DATA_SKIP} \\
	        ${data_seed_arg[*]} \\
	        ${audio_epoch_args[*]} \\
	        ${token_stats_args[*]} \\
	        create_new_adapter=${CREATE_NEW_ADAPTER} funaudiochat_full_audio_tuning=${FUNAUDIOCHAT_FULL_AUDIO_TUNING} \\
	        ${eval_args[*]} \\
        overwrite_cache=false preprocessing_num_workers=${PREPROCESSING_NUM_WORKERS} \\
        dataloader_num_workers=${DATALOADER_NUM_WORKERS} dataloader_prefetch_factor=${DATALOADER_PREFETCH_FACTOR} \\
        dataloader_pin_memory=${DATALOADER_PIN_MEMORY} dataloader_persistent_workers=${DATALOADER_PERSISTENT_WORKERS} \\
        finetuning_type=lora ${lora_target_arg[*]} lora_rank=${LORA_RANK} lora_alpha=${LORA_ALPHA} lora_dropout=${LORA_DROPOUT} ${reg_args[*]} ${adapter_arg[*]} \\
        output_dir=${OUTPUT_DIR} overwrite_output_dir=false \\
        learning_rate=${LEARNING_RATE} lr_scheduler_type=${LR_SCHEDULER_TYPE} warmup_ratio=${WARMUP_RATIO} \\
        ${max_steps_arg[*]} \\
        per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \\
        logging_steps=${LOGGING_STEPS} \\
        save_strategy=steps save_steps=${SAVE_STEPS} save_total_limit=${SAVE_TOTAL_LIMIT} \\
        load_best_model_at_end=${load_best_model_at_end_arg} metric_for_best_model=${METRIC_FOR_BEST_MODEL} ${greater_is_better_arg[*]}
EOF

  cd "${WORK_DIR}"
  set +e
  stdbuf -oL -eL conda run -n "${CONDA_ENV_NAME}" --no-capture-output \
    env CUDA_VISIBLE_DEVICES="${GPUS}" \
    FORCE_TORCHRUN=1 \
    NPROC_PER_NODE="${NPROC_PER_NODE}" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    LLAMAFACTORY_PERF_LOG="${LLAMAFACTORY_PERF_LOG}" \
    LLAMAFACTORY_DATALOADER_PERF_LOG="${LLAMAFACTORY_DATALOADER_PERF_LOG}" \
    LLAMAFACTORY_DYNAMIC_PACKING_PERF_META="${LLAMAFACTORY_DYNAMIC_PACKING_PERF_META}" \
    LLAMAFACTORY_PRESERVE_AUDIO_META="${LLAMAFACTORY_PRESERVE_AUDIO_META}" \
    LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS="${LLAMAFACTORY_SHARDED_PREFETCH_LOG_RANKS}" \
    LLAMAFACTORY_TOS_SDK_FOR_MOUNT="${LLAMAFACTORY_TOS_SDK_FOR_MOUNT}" \
    LLAMAFACTORY_TOS_MOUNT_MAP="${LLAMAFACTORY_TOS_MOUNT_MAP}" \
    LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS="${LLAMAFACTORY_TOS_MAX_POOL_CONNECTIONS}" \
    LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS="${LLAMAFACTORY_S3_MAX_POOL_CONNECTIONS}" \
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
		    dynamic_prompt_packing_num_shards="${DYNAMIC_PROMPT_PACKING_NUM_SHARDS}" \
		    "${sharded_args[@]}" \
		    cutoff_len="${CUTOFF_LEN}" \
		    "${packing_args[@]}" \
		    ignore_data_skip="${IGNORE_DATA_SKIP}" \
		    "${data_seed_arg[@]}" \
		    "${audio_epoch_args[@]}" \
		    "${token_stats_args[@]}" \
		    create_new_adapter="${CREATE_NEW_ADAPTER}" \
	    funaudiochat_full_audio_tuning="${FUNAUDIOCHAT_FULL_AUDIO_TUNING}" \
        "${eval_args[@]}" \
    overwrite_cache=false \
    preprocessing_num_workers="${PREPROCESSING_NUM_WORKERS}" \
    dataloader_num_workers="${DATALOADER_NUM_WORKERS}" \
    dataloader_prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
    dataloader_pin_memory="${DATALOADER_PIN_MEMORY}" \
    dataloader_persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
    finetuning_type=lora \
    "${lora_target_arg[@]}" \
    lora_rank="${LORA_RANK}" \
    lora_alpha="${LORA_ALPHA}" \
    lora_dropout="${LORA_DROPOUT}" \
    "${reg_args[@]}" \
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
    load_best_model_at_end="${load_best_model_at_end_arg}" \
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
    if run_once "${restarts}"; then
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
