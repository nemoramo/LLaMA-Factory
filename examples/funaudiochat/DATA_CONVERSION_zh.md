# FunAudioChat S2T 数据转换说明（含 normalized / original 两种文本）

这份文档说明如何把现有的 ASR ShareGPT 数据（`messages` + `audios` + `prompt_pool`）转换为本 repo 的 FunAudioChat（S2T-only）训练格式，并解释 **normalized text** 与 **original text** 两种文本如何在训练时同时生效。

## 1. 背景与目标

你的源数据（如 `sw_v5_ha_v9.jsonl`）是 LLaMA-Factory 的 ShareGPT ASR 格式：

- `messages`: user/assistant
- `audios`: 音频路径列表（通常 1 条）
- `prompt_pool`: 用于区分 *normalized* 与 *verbatim/original* 的提示词后缀与目标文本（带权重）

FunAudioChat 训练侧需要：

- 文本中使用 FunAudioChat 的音频占位模板：`<|audio_bos|><|AUDIO|><|audio_eos|>`
- `audio` 字段为 **JSON 字符串列表**（官方 training 也是这样做的）

此外，我们希望：

- **S2T-only**（只训练输出文本）
- 同时利用两种转写目标：
  - `text`：lowercased / normalized / no punctuation（常用于 WER/CER）
  - `original_text`：尽可能保留大小写与标点（如果源数据提供）

## 2. 转换脚本

脚本位置：

- `scripts/convert_mm_data/convert_sharegpt_audio_to_funaudiochat_s2t.py`

它会把每条样本重建成：

- `system`: 默认 `You are asked to generate text tokens.`（S2T）
- `messages`: 只有一轮（user + assistant）
  - user prompt 中插入 `<|audio_bos|><|AUDIO|><|audio_eos|>`
  - assistant 默认填入 *normalized*（作为 fallback）
- `audio`: 形如 `["{\"path\": \"...wav\", \"token\": \"<|audio_pad|>...\", \"text\": \"...\", \"ref_text\": \"...\"}"]`
  - 注意：列表元素是 **JSON 字符串**，不是 dict
- `text` / `original_text`: 便于检索或后处理（训练本身主要靠 `prompt_pool`）
- `prompt_pool`: **默认保留原始 `prompt_pool`**（见第 3 节）

## 3. 两种文本如何“都用于训练”

关键点：**要让两种文本都参与训练，需要打开 `dynamic_prompt_sampling: true`**。

原因：

- 当 `dynamic_prompt_sampling=true` 且 `stage=sft` 时，训练集会被包装成 `DynamicPromptDataset`，每次 `__getitem__` 会从 `prompt_pool` 按权重随机采样一个 entry。
- 被采样的 entry 会同时影响：
  1) **Prompt 区分**：entry 的 `text`（suffix）会被追加到 system prompt（区分 “normalized” vs “verbatim/original”，以及语言提示等）。
  2) **Label 匹配**：如果 entry 带 `completion`，会覆盖 assistant 的目标文本，保证 loss 对齐到该风格的转写目标。

因此，训练过程中会按 `prompt_pool.weight` 在不同风格之间混合：

- 抽到 normalized entry → 用 normalized 的 suffix + normalized 的 completion
- 抽到 original entry → 用 verbatim/original 的 suffix + original 的 completion

如果你关闭 `dynamic_prompt_sampling`，那么训练只会使用 `messages` 里写死的 assistant 内容（默认是 normalized），original 分支不会被用到。

## 4. original_text 可能与 text 完全一样，怎么办？

是的，这种情况会出现：某些样本的 `original_text` 本身也没有大小写/标点，导致：

- `original_text == text`
- `prompt_pool` 里 “verbatim/original” entry 的 `completion` 也可能与 normalized 的 `completion` 相同

这不会导致训练报错；只是 **original 分支在该样本上没有额外监督信号**（因为目标文本相同）。

对训练效果的影响：

- 如果大量样本都没有标点/大小写信息，那么模型本身也很难学会“按指令生成标点/大小写”，这是数据本身的信息上限。
- 对于少量 `original==normalized` 的样本，它们更多是在做“不同指令 → 同一输出”的对齐，通常影响很小。

如果你非常在意 prompt 的一致性（例如不希望出现“请保留标点”但输出没有标点的样本），可以考虑：

- 在数据侧过滤：对 `original_text == text` 的样本，把 original 分支权重置 0 或移除对应 entry
- 或者干脆只训练 normalized（ASR 常见做法），把标点恢复作为后处理/另一个模型

当前转换脚本的策略是：**尽量保留原始信息，不做强行去重/裁剪**，便于你后续按需求做数据治理。

## 5. token 字段与大规模数据的建议（notoken 方案）

官方 FunAudioChat 的 processor 期望 `audio` JSON 里包含 `token`（离散 25Hz token 串）。但对 880 万条数据来说，把 `<|audio_pad|>` 重复 N 次写进文件会非常大且转换慢。

因此我们推荐使用 `--no-token` 生成 notoken 版本：

- 文件更小、写盘更快
- 本 repo 的 `funaudiochat` 插件会自动根据音频时长推断 token 长度
  - 优先从文件名 `*_segXXXX_<start>-<end>.wav` 推断时长（无需读 wav）
  - 否则才回退到读音频计算时长

## 6. 调整“带语言提示 prompt”的采样比例（6:4）

如果你的 `prompt_pool` 同时包含：

- 不带语言提示：`Just provide ...`
- 带语言提示：`Just provide ...\nThe language is xx.`

并且你希望 **带语言提示 : 不带语言提示 = 6 : 4**（同时保持每组 prompt 的总权重不变），可以用脚本对 JSONL 做一次流式后处理：

- `scripts/convert_mm_data/adjust_prompt_pool_lang_ratio.py`

它会按 *base prompt*（移除 `The language is ...` 行后的文本）分组，在每组内重分配权重。例如：

- `0.63 + 0.07 = 0.70` → `0.28 + 0.42 = 0.70`
- `0.27 + 0.03 = 0.30` → `0.12 + 0.18 = 0.30`

在你的环境里，我们已经把这个 6:4 调整 **直接写回** 到全量 JSONL 中（不额外产出新文件名）。

## 7. 本机数据落盘位置（你的环境）

本次全量转换输出（notoken + 保留 prompt_pool）：

- `/data2/mayufeng/manifests/llama_data/funaudiochat_data/sw_v5_ha_v9_funaudiochat_s2t_promptpool.notoken.jsonl`

并已在：

- `/data2/mayufeng/manifests/llama_data/dataset_info.json`

注册为 `funaudiochat_asr_v9_train`（字段映射 `system/messages/audio`）。

## 8. 训练侧最小配置提示

你需要（至少）：

- `template: funaudiochat`
- `dataset: funaudiochat_asr_v9_train`
- `dataset_dir: /data2/mayufeng/manifests/llama_data/`
- `dynamic_prompt_sampling: true`
