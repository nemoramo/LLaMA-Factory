# Gemma 3n（E2B）ASR 训练教程（从 NeMo manifest 开始）

本教程以“本地音频 + 转写文本”的 ASR 场景为例，介绍如何从 **NeMo manifest** 构建数据、转换为 LLaMA-Factory 可训练格式，并用 `examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora.yaml` 启动训练（含 WER/CER 展示、best by loss）。

> 适用：单个超大 jsonl / 本地音频路径 / 需要动态 prompt_pool（训练时在 normalized/original 之间切换）  
> 约定：**eval 只输出 normalized 文本（lowercase、no punctuation）**

---

## 0. 环境准备

1) **Hugging Face 登录（Gemma 权限）**

Gemma 系列通常需要你在 Hugging Face 上接受协议并登录后才能拉取：

```bash
huggingface-cli login
# 或者临时：export HF_TOKEN=...
```

2) **音频解码工具**

如果你看到 `pydub` 提示找不到 `ffmpeg`，需要安装系统 ffmpeg（否则某些音频格式可能读不了）：

```bash
ffmpeg -version
```

---

## 1. 构建 NeMo manifest（训练 / 评估）

NeMo manifest 是 **jsonl**，每行一个样本。最少包含：

- `audio_filepath`: 音频绝对路径（建议用绝对路径，分布式时每台机器都要能访问到）
- `text`: 作为训练/评估标签的“规范化转写”（推荐：**lowercase + no punctuation**）

可选字段：

- `original_text`: 原始/未规范化文本（可能含大小写、标点）；用于训练时做动态采样（prompt_pool）
- `lang` / `language`: 语种提示（可选）

示例（每行一条）：

```json
{"audio_filepath":"/data2/mayufeng/wavs/0001.wav","text":"ah shi cutan sanyi","original_text":"Ah, shi cutan sanyi.","lang":"ha"}
```

### 关于 `text` vs `original_text` 的重要提醒

- 如果某些数据的 `original_text` **本来就没有大小写/标点**，它可能和 `text` 完全一致。  
  这种情况下“original 分支”不会带来信息增量，后面生成 `prompt_pool` 时会自动忽略 original 的概率（避免 per-sample 概率无意义）。
- **eval 强制 normalized 输出**：建议 eval 的 `text` 就是你期望评估 WER/CER 的 normalized 版本（lowercase、no punctuation），不要用带标点/大小写的标签来算“normalized WER”。

---

## 2. 转换为 LLaMA-Factory（ShareGPT + audios jsonl）

使用脚本：`scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py`

### 2.1 训练集转换（支持 prompt_pool）

```bash
python3 scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py \
  --input /data2/mayufeng/manifests/hasua/train_v9.manifest \
  --output /data2/mayufeng/manifests/llama_data/hausa/train_v9.jsonl \
  --prompt "Transcribe the audio. Only output the text: <audio>" \
  --original-prob 0.2 \
  --normalized-suffix "Output only the text (lowercase, no punctuation)." \
  --original-suffix "Please transcribe verbatim, preserving casing and punctuation."
```

说明：

- 输出每条样本包含：
  - `messages`: OpenAI/ShareGPT 对话格式
  - `audios`: `[<absolute_audio_path>]`
  - `prompt_pool`（可选）：用于训练时动态采样 suffix/target（需要 `dynamic_prompt_sampling: true`）
- 若 `original_text == text`，脚本会自动不生成 original 分支（`original-prob` 对该样本等价于 0）。

### 2.2 评估集转换（强制 normalized 输出）

eval 的关键是：**prompt 要明确要求输出 normalized 文本**，并且标签也用 normalized（`text` 字段）。

如果你已经有一个 `test_youtube.jsonl`（但 prompt 不够“严格 normalized”），推荐直接生成一个
“只改 prompt”的派生版本（样本内容不变，避免重新全量转换）：

```bash
python3 - <<'PY'
import json

src = "/data2/mayufeng/manifests/llama_data/hausa/test_youtube.jsonl"
dst = "/data2/mayufeng/manifests/llama_data/hausa/test_youtube_norm_text.jsonl"
prompt = "Transcribe the audio. Output only the text (lowercase, no punctuation): <audio>"

with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        obj = json.loads(line)
        obj["messages"][0]["content"] = prompt
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print("wrote:", dst)
PY
```

如果你还没有评估集 jsonl，可以从 manifest 直接转换：

```bash
python3 scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py \
  --input /data2/mayufeng/manifests/hasua/test_youtube.manifest \
  --output /data2/mayufeng/manifests/llama_data/hausa/test_youtube_norm_text.jsonl \
  --prompt "Transcribe the audio. Output only the text (lowercase, no punctuation): <audio>" \
  --disable-prompt-pool
```

---

## 3. 注册数据集（dataset_dir + dataset_info.json）

在你选择的 `dataset_dir` 下放置：

- `dataset_info.json`
- `hausa/train_v9.jsonl`
- `hausa/test_youtube_norm_text.jsonl`

例如（与你当前 YAML 一致）：

- `dataset_dir`: `/data2/mayufeng/manifests/llama_data/`
- `dataset_info.json`: `/data2/mayufeng/manifests/llama_data/dataset_info.json`

`dataset_info.json` 最小示例：

```json
{
  "gemma3n_asr_v9_train": {
    "file_name": "hausa/train_v9.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "audios": "audios"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
  },
  "gemma3n_asr_hausa_youtube_test_norm_text": {
    "file_name": "hausa/test_youtube_norm_text.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "audios": "audios"},
    "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
  }
}
```

---

## 4. 训练 YAML（已对齐你的数据名）

你现在用的配置：`examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora.yaml`

关键字段（重点理解这几个）：

- `dataset / eval_dataset / dataset_dir`: 指向上面注册的名字和目录
- `train_on_prompt: false`: **prompt 部分不计入 loss**（ASR 通常需要这个）
- `dynamic_prompt_sampling: true`: 训练时启用 `prompt_pool` 动态采样（如果数据里有 `prompt_pool`）
- `predict_with_generate: true` + `compute_wer_cer: true`: eval 时做生成，展示 WER/CER
- `eval_num_samples: 10`: 生成评估只抽样 10 条（加速）
- `eval_loss_on_full_dataset: true`: loss 用 **全量 eval**；WER/CER 用抽样子集（推荐）
- `load_best_model_at_end: true` + `metric_for_best_model: loss`: **按 loss 选 best checkpoint**

> 分布式（8 卡）下：`eval_num_samples=10` 是“全局 10 条”，会被 DDP sampler 分到各 rank 执行，不是每张卡 10 条。

---

## 5. 启动训练

### 5.1 单卡

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true \
llamafactory-cli train examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora.yaml
```

### 5.2 8 卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 WANDB_DISABLED=true \
llamafactory-cli train examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora.yaml
```

### 5.3 Unsloth（单机版 dynamic prompt）

如果你希望用 Unsloth 加速（并保留 `dynamic_prompt_sampling: true` 的在线 prompt_pool 采样），使用：

- 配置：`examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora_unsloth.yaml`
- 关键：需要先安装 `unsloth`，并在 YAML 里设置 `use_unsloth: true`

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_DISABLED=true \
llamafactory-cli train examples/gemma3n/gemma3n_e2b_asr_nemo_debug_lora_unsloth.yaml
```

说明：

- LLaMA-Factory 会在多 GPU 时自动走 `torchrun`（或你也可以 `FORCE_TORCHRUN=1` 强制）。
- 如果你曾遇到 Ctrl+C 后 GPU 进程未退出：新版 launcher 会尽量杀掉 `torchrun` 及其子进程；必要时可手动：
  - `pkill -f torchrun`
  - 或根据 `nvidia-smi` / `ps -ef | grep python` 定位并结束残留进程

---

## 6. 常见问题排查（ASR 场景）

1) **KeyError: 'from' / 格式不对**

确认你的 jsonl 是 OpenAI messages 格式（`messages=[{role,content}, ...]`），并在 `dataset_info.json` 里用 `formatting: sharegpt` 且 `role_tag/content_tag` 对齐。

2) **评估输出不是 normalized**

eval 这条强依赖 prompt：请使用 `*_norm_text.jsonl`（prompt 明确要求 lowercase/no punctuation），并保证 eval 标签 `text` 也是 normalized。
