# Voxtral in LLaMA-Factory

This repo includes an integration for **Voxtral** checkpoints (`model_type=voxtral`), e.g. `mistralai/Voxtral-Mini-3B-2507`.

- 中文版说明：`examples/voxtral/README_zh.md`

- Template: `template: voxtral`
- MM plugin: provides `input_features` (Whisper-style log-mel) chunked into 30s segments, matching Voxtral’s `[AUDIO]` token expansion.

## Requirements

Voxtral uses `mistral-common` under the hood for chat templating.

```bash
python -m pip install mistral-common
```

## Prompting modes

LLaMA-Factory supports two Voxtral prompting modes:

1) **Default (recommended for ASR)**: the official *transcription request* prefix (no chat messages).
   - Requires `voxtral_transcription_language` (or a per-sample `lang`/`language` column).
   - Expects exactly **one** audio per sample.

2) **Chat mode**: ShareGPT-style messages with `<audio>` placeholders.
   - Enable via `voxtral_chat_template: true`.

## Dataset format (ShareGPT, chat mode)

Use a ShareGPT-style dataset with an `audios` column (mapped via `dataset_info.json`).

- `messages[*].content` should contain one or more audio placeholders: `<audio>`
- `audios` is a list whose length matches the number of `<audio>` placeholders.

Example JSONL row:

```json
{
  "messages": [
    { "role": "user", "content": "<audio> Please transcribe the audio." },
    { "role": "assistant", "content": "..." }
  ],
  "audios": ["/abs/path/to.wav"]
}
```

Notes:
- Voxtral tokenization expands each `<audio>` placeholder into `375 * ceil(duration/30s)` `[AUDIO]` tokens.
- `cutoff_len` must be large enough to keep **all** `[AUDIO]` placeholders (audio tokens cannot be truncated).

## Packing

Voxtral supports `packing: true` and `neat_packing: true` for SFT. Long examples exceeding `cutoff_len` are dropped.

## Converting manifests

If you have a NeMo-style JSONL manifest (e.g. `audio_filepath` + `text` + optional `lang/language`), you can convert it
into a LLaMA-Factory ShareGPT audio dataset with:

```bash
python scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py \
  --input /path/to/train.manifest \
  --output /path/to/train_voxtral.jsonl \
  --audio-key audio_filepath \
  --text-key text \
  --lang-key lang
```

Notes:
- The converter now writes `lang` (when present), which Voxtral uses for the transcription request template.
- Alternatively, set a global `voxtral_transcription_language: xx` in your training YAML.

## Example config

See `examples/voxtral/voxtral_sft.yaml`.
