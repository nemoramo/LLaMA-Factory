# 在 LLaMA-Factory 中微调 Voxtral

本仓库已集成 **Voxtral**（`model_type=voxtral`），例如：`mistralai/Voxtral-Mini-3B-2507`。

- 模板：`template: voxtral`
- 多模态插件：自动抽取音频 `input_features`（Whisper log-mel），并按 **30 秒**切块，与 Voxtral 的 `[AUDIO]` token 展开对齐。

## 依赖

Voxtral 的 chat template 依赖 `mistral-common`：

```bash
python -m pip install mistral-common
```

## Voxtral 的两种 Prompt 模式

LLaMA-Factory 支持两种 Voxtral 提示方式：

### 1) 默认（推荐 ASR）：官方转写请求模板

默认情况下，`template: voxtral` 会使用 **Voxtral 官方 transcription request** 的前缀形式：

`<s>[INST][BEGIN_AUDIO]...[AUDIO]...[/INST] lang:xx [TRANSCRIBE]`

特点：
- **不走 ShareGPT 的 chat 拼接**，而是构造与官方一致的转写前缀。
- 需要语言信息：优先使用样本中的 `lang`/`language` 字段；否则使用全局参数 `voxtral_transcription_language`。
- 每条样本要求 **恰好 1 段音频**（`audios` 长度为 1）。

### 2) Chat 模式：ShareGPT messages + `<audio>`

如需与官方聊天对齐（messages 方式），请显式打开：

```yaml
voxtral_chat_template: true
```

特点：
- 按 ShareGPT messages 拼接对话，并支持 `<audio>` 占位符插入。
- `messages[*].content` 中 `<audio>` 的数量必须与 `audios` 列表长度一致。

## 数据格式（ShareGPT JSONL）

无论默认转写模板还是 chat 模式，都建议使用 ShareGPT 风格 JSONL（通过 `dataset_info.json` 映射）。

一条样本示例：

```json
{
  "messages": [
    { "role": "user", "content": "<audio> 请转写这段音频。" },
    { "role": "assistant", "content": "..." }
  ],
  "audios": ["/abs/path/to.wav"],
  "lang": "en"
}
```

说明：
- 默认转写模板下：`messages[0].content` 不参与构造最终 prompt，但建议仍包含 `<audio>` 以便后续切换 chat 模式不需要重新做数据。
- `cutoff_len` 必须足够大，以容纳完整的音频占位 token（音频 token 不能被截断；否则样本会被丢弃/报错）。

## Packing

Voxtral 支持 `packing: true` 与 `neat_packing: true`（SFT）。超过 `cutoff_len` 的样本会被丢弃。

## 从 manifest 转换（推荐）

如果你手头是 NeMo / ASR 常见的 JSONL manifest（例如包含 `audio_filepath`、`text`、可选 `lang/language`），可以用脚本转换成 LLaMA-Factory 可训练的数据：

```bash
python scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py \
  --input /path/to/train.manifest \
  --output /path/to/train_voxtral.jsonl \
  --audio-key audio_filepath \
  --text-key text \
  --lang-key lang
```

说明：
- 如果 manifest 中有 `lang/language`，转换后会写入 `lang` 字段，供默认转写模板使用。
- 如果你的数据是单语种，也可以不写 `lang`，转而在训练 YAML 里设置 `voxtral_transcription_language: xx`。

## 训练示例

参考配置：`examples/voxtral/voxtral_sft.yaml`。

关键参数（默认转写模板）：

```yaml
template: voxtral
voxtral_transcription_language: en  # 或者数据里每条带 lang/language
```

如需 chat 模式：

```yaml
voxtral_chat_template: true
```

