# FunAudioChat (S2T) in LLaMA-Factory

This repo includes a built-in integration for the **FunAudioChat** checkpoint (`model_type=funaudiochat`) with **S2T-only** fine-tuning.

- Template: `template: funaudiochat`
- MM plugin: expands audio placeholders into variable-length `<|AUDIO|>` tokens and provides:
  - `speech_ids` / `speech_attention_mask` (discrete 25Hz frames, no CosyVoice required)
  - `input_features` / `feature_attention_mask` (continuous waveform features)
  - `feature_exist_mask` (which audios have waveform features)
- Speech generation/loss is disabled automatically during training (`disable_speech=True`).

## Dataset format (ShareGPT)

Use a ShareGPT-style dataset with an `audios` column (mapped to your JSONL field via `dataset_info.json`).

- `messages[*].content` should contain an audio placeholder. Recommended (FunAudio style):
  - `<|audio_bos|><|AUDIO|><|audio_eos|>`
- The `audios` field is a list whose length matches the number of placeholders in the messages.

## Data conversion (normalized vs original)

If your ASR dataset contains both normalized text (lowercased / no punctuation) and original text (cased / punctuated),
see `examples/funaudiochat/DATA_CONVERSION_zh.md` for a practical conversion workflow and how to keep both targets via
`prompt_pool` + `dynamic_prompt_sampling: true`.

Each audio item can be either:

1) A plain path string: `"/abs/path/to.wav"`
2) A JSON string (FunAudio compatible):

```json
{"path": "/abs/path/to.wav", "token": "<|audio_pad|><|audio_pad|>..."}
```

If `token` is omitted/empty, the plugin will infer the 25Hz frame count from the waveform duration and build a pad-only token sequence.

## Example config

See `examples/funaudiochat/funaudiochat_s2t_sft_full.yaml`.

## Packing training (reference)

For long-running experiments, we recommend launching via the watchdog script:
`scripts/monitor_funaudiochat_s2t_training.sh`.

- If training exits (OOM / disconnect / crash), it restarts automatically.
- With `overwrite_output_dir=false`, LLaMA-Factory resumes from the latest checkpoint in `output_dir`.
- The script saves reproducibility artifacts into `output_dir`:
  - `training_command.txt` (the exact command line)
  - `config_base.yaml` (a copy of the base YAML config file)

### Neat packing reference command

```bash
export OUTPUT_DIR="/path/to/llamafactory_saves/funaudiochat/s2t_lora_neatpack_run"
# Optional: initialize from an existing LoRA adapter checkpoint.
export INIT_ADAPTER_NAME_OR_PATH="/path/to/prev_adapter/checkpoint-XXXXX"

GPUS=0,1,2,3,4,5 NPROC_PER_NODE=6 \
PACKING=true NEAT_PACKING=true \
DYNAMIC_PROMPT_LAZY_ALIGN=true DYNAMIC_PROMPT_PACKING_BUFFER_SIZE=1024 \
PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=4 \
MAX_STEPS=60000 EVAL_STEPS=2000 SAVE_STEPS=2000 \
DATALOADER_NUM_WORKERS=6 PREPROCESSING_NUM_WORKERS=32 DATALOADER_PREFETCH_FACTOR=4 \
EVAL_MAX_NEW_TOKENS=512 \
bash scripts/monitor_funaudiochat_s2t_training.sh
```

Notes:
- When packing is enabled, epoch semantics may not match “full dataset passes”; prefer `MAX_STEPS` for scheduling.
- If `output_dir` already has checkpoints, the script skips `INIT_ADAPTER_NAME_OR_PATH` and resumes from the latest checkpoint.

## Attention implementation (recommended: `fa2`)

LLaMA-Factory supports 3 attention implementations for FunAudioChat via `flash_attn`:

- `fa2`: Transformers `flash_attention_2` (requires the `flash-attn` package)
- `sdpa`: torch SDPA (`torch.nn.functional.scaled_dot_product_attention`)
- `disabled`: eager attention

**Recommendation:** use `flash_attn: fa2` when available (fastest). If you cannot install it, use `sdpa`.

### Benchmarks (single GPU)

These numbers are from a controlled local benchmark on **NVIDIA H20 (sm90)**, **single GPU**, **LoRA + bf16**,
**S2T-only**, `per_device_train_batch_size=23`, `max_steps=100`, `audio_padding=max_length`, with a fixed-length
~8.9s audio sample repeated (so padding and memory are stable).

| `flash_attn` | train steps/s | peak VRAM |
| --- | ---: | ---: |
| `fa2` | **0.358** | 45,250 MiB |
| `sdpa` | 0.189 | **36,965 MiB** |
| `disabled` | 0.112 | 68,109 MiB |

Notes:
- Speed and memory will vary with audio duration distribution, cutoff length, gradient checkpointing, etc.
- `fa2` is fastest in our tests; `sdpa` may be more memory-efficient depending on your workload.

### Benchmarks (neat packing vs no packing)

In a timed **25-minute** run on **2× NVIDIA H20**, we compared **neat packing** vs **no packing** for FunAudioChat S2T
using the same training datasets and tracked **effective tokens** (non-ignored labels, excluding audio tokens):

- `neat_packing=true` + dynamic prompt packing (`per_device_train_batch_size=2`): ~**1.00M** effective tokens/GPU/25min (≈**669 tok/s/GPU**)
- `packing=false` (`per_device_train_batch_size=8`): ~**0.71M** effective tokens/GPU/25min (≈**475 tok/s/GPU**)

Overall, neat packing delivered ~**1.41×** more effective tokens in the same wall-clock window (steady-state throughput in logs was ~**1.77×** higher).

### Install FlashAttention-2 (`fa2`)

#### 1) Quick install (recommended to try first)

Inside your training Python environment:

```bash
python -m pip install -U --no-build-isolation flash-attn
```

If your GPU is H20/H100 (sm90), you can speed up compilation by restricting the build arch:

```bash
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=8
export NVCC_THREADS=4
python -m pip install -U --no-build-isolation flash-attn
```

Verify:

```bash
python -c "from transformers.utils import is_flash_attn_2_available; print(is_flash_attn_2_available())"
```

You should also see this in training logs:
`Using FlashAttention-2 for faster training and inference.`

#### 2) If pip build fails: build a wheel then install

This is a robust fallback that avoids some PEP517 edge cases:

```bash
set -euo pipefail
mkdir -p /tmp/flash_attn_build && cd /tmp/flash_attn_build
python -m pip download --no-deps --no-binary :all: flash-attn==2.8.3
tar -xzf flash_attn-2.8.3.tar.gz
cd flash_attn-2.8.3

export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=90
export MAX_JOBS=8
export NVCC_THREADS=4

python setup.py bdist_wheel -v
python -m pip install --no-deps dist/flash_attn-*.whl
```
