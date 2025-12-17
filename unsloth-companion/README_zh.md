# Unsloth Companion（不依赖 LLaMA-Factory Trainer）

这个目录提供一个**完全独立**的 Gemma3n-Audio SFT 训练脚本：

- 直接使用 **Unsloth + TRL**（类似官方 notebook：`Gemma3N_(4B)-Audio.ipynb`）
- 支持 LLaMA-Factory 数据转换脚本产出的 **ShareGPT-Audio jsonl**（`messages` + `audios` + 可选 `prompt_pool`）
- 在 `DataCollator` 内实现 **dynamic prompt（prompt_pool 在线采样）**，不依赖 LLaMA-Factory 的 dataset/Trainer

## 1) 依赖

参考 notebook 的版本组合（示例）：

```bash
pip install unsloth
pip install transformers==4.56.2
pip install --no-deps trl==0.22.2
pip install "datasets==4.3.0"
pip install torchcodec timm
```

> 说明：你的环境如果已经能运行 Unsloth notebook，可直接复用现有环境。

## 2) 数据格式（jsonl）

每行一个样本，最小字段如下（与 `scripts/convert_mm_data/convert_nemo_manifest_to_sharegpt_audio.py` 一致）：

```json
{
  "messages": [
    {"role": "user", "content": "Transcribe the audio. Output only the text: <audio>"},
    {"role": "assistant", "content": "normalized text"}
  ],
  "audios": ["/abs/path/to.wav"],
  "prompt_pool": [
    {"text": "Output only the text (lowercase, no punctuation).", "completion": "normalized text", "weight": 0.8},
    {"text": "Please transcribe verbatim, preserving casing and punctuation.", "completion": "Original Text.", "weight": 0.2}
  ]
}
```

注意：

- 当前脚本**仅支持每条样本 1 个音频**（`audios` 长度必须为 1）。
- `messages[0].content` 里使用的是 `"<audio>"` 占位符；脚本会在 collate 时替换为 Gemma3nProcessor 的 `audio_token`。

## 3) 启动训练

```bash
python unsloth-companion/train_gemma3n_audio_dynamic_prompt.py \
  --train_jsonl /data2/mayufeng/manifests/llama_data/hausa/train_v9.jsonl \
  --output_dir /data2/mayufeng/unsloth_outputs/gemma3n_asr_dynamic_prompt \
  --model_name unsloth/gemma-3n-E4B-it \
  --max_seq_length 2048 \
  --load_in_4bit \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --dynamic_prompt
```

如果需要固定每条样本的 pool 选择（便于复现/断点续训稳定），加：

```bash
--dynamic_prompt_deterministic
```

## 4) Eval 生成样例（generate）

脚本支持在评估时（或训练结束后）用 `model.generate` 打印/保存一些样本的解码输出：

- 传入评估集：`--eval_jsonl ...`
- 开启生成样例：`--eval_generate_samples 8`（每次 eval 抽 8 条；0 表示关闭）
- 评估频率：
  - `--eval_strategy steps --eval_steps 500`（训练中周期性 eval 并生成样例）
  - 或保持默认 `--eval_strategy no`，训练结束后生成一次 `eval_generations_final.jsonl`

示例（训练中每 500 step 生成 8 条）：

```bash
python unsloth-companion/train_gemma3n_audio_dynamic_prompt.py \
  --train_jsonl /path/train.jsonl \
  --eval_jsonl /path/dev.jsonl \
  --output_dir /path/out \
  --load_in_4bit \
  --dynamic_prompt \
  --eval_strategy steps --eval_steps 500 \
  --eval_generate_samples 8 --eval_generate_max_new_tokens 128
```

生成结果会写到 `output_dir` 下：

- `eval_generations_step<global_step>.jsonl`（周期性 eval）
- `eval_generations_final.jsonl`（当 `eval_strategy=no` 时的训练结束生成）
