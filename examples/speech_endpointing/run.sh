#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Speech endpointing runner (single entrypoint).

Train (single GPU):
  GPU_ID=0 \
  CFG=examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_neat_packing.yaml \
    bash examples/speech_endpointing/run.sh train \
      dataset_dir=/path/to/your/dataset_dir \
      output_dir=/path/to/output_dir

Export (merge LoRA):
  CFG=examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_export.yaml \
    bash examples/speech_endpointing/run.sh export \
      adapter_name_or_path=/path/to/lora_adapter_dir \
      export_dir=/path/to/export_dir

Notes:
  - Append Hydra-style overrides (key=value) after the subcommand.
  - If CONDA_ENV is set, commands run via: conda run -n $CONDA_ENV ...
EOF
}

run_in_env() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    conda run -n "${CONDA_ENV}" "$@"
  else
    "$@"
  fi
}

cmd="${1:-train}"
shift || true

case "${cmd}" in
  -h|--help|help)
    usage
    ;;
  train)
    GPU_ID="${GPU_ID:-0}"
    CFG="${CFG:-examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_neat_packing.yaml}"

    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"

    echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    echo "[INFO] train_config=${CFG}"

    run_in_env llamafactory-cli train "${CFG}" "$@"
    ;;
  export)
    CFG="${CFG:-examples/speech_endpointing/qwen2_5/3b/qwen2_5_3b_speech_endpointing_lora_export.yaml}"

    echo "[INFO] export_config=${CFG}"

    run_in_env llamafactory-cli export "${CFG}" "$@"
    ;;
  *)
    echo "[ERROR] Unknown subcommand: ${cmd}" >&2
    echo >&2
    usage >&2
    exit 2
    ;;
esac

