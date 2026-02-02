# Qwen3-ASR in LLaMA-Factory

This repo includes an integration for **Qwen3-ASR** checkpoints (`model_type=qwen3_asr`), e.g.:

- `Qwen/Qwen3-ASR-0.6B`
- `Qwen/Qwen3-ASR-1.7B`

## Requirements

Qwen3-ASR may not be in upstream `transformers` yet. LLaMA-Factory supports loading it via either:

1) Install the official package:

```bash
python -m pip install -U qwen-asr
```

2) Or use the vendored submodule (recommended for reproducibility in this repo):

```bash
git submodule update --init --recursive third_party/qwen3-asr
```

## Template / audio placeholder

- Template: `template: qwen3_asr`
- Audio placeholder: use `<audio>` inside `messages[*].content`
- Audio files: provide an `audios` list; its length must match the number of `<audio>` placeholders.

## Dataset format (OpenAI messages + audios)

Each JSONL row should look like:

```json
{
  "system": "",
  "messages": [
    { "role": "user", "content": "<audio>" },
    { "role": "assistant", "content": "language English<asr_text>This is a test sentence." }
  ],
  "audios": ["/abs/path/to.wav"]
}
```

Notes:
- For best compatibility with Qwen3-ASR, keep the target format recommended upstream:
  - `language English<asr_text>...`
  - `language None<asr_text>...` (if unknown)

## Packing (on-the-fly)

Qwen3-ASR supports:

- `packing: true`
- `dynamic_prompt_packing: true` (buffered knapsack packing at training time)

See `examples/qwen3_asr/qwen3_asr_sft_full.yaml` for a ready-to-use config.

## FlashAttention 2 (FA2)

Set `flash_attn: fa2` and use `bf16: true` (or `fp16`) if your environment has FlashAttention 2 installed.

## Converting upstream JSONL (audio/text) to LLaMA-Factory format

If your data is in Qwen3-ASR finetuning JSONL format (fields: `audio`, `text`, optional `prompt`), convert it to
LLaMA-Factory OpenAI+audio JSONL with:

```bash
python scripts/convert_mm_data/convert_qwen3_asr_jsonl_to_openai_audio.py \
  --input /path/to/train.jsonl \
  --output /path/to/train_openai_audio.jsonl
```

## Example configs

- Full finetune: `examples/qwen3_asr/qwen3_asr_sft_full.yaml`
- LoRA finetune: `examples/qwen3_asr/qwen3_asr_sft_lora.yaml`

