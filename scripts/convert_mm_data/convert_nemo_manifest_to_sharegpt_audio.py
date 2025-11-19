#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 NeMo manifest (audio_filepath + text) 转成给 LLaMA-Factory 训练用的
“OpenAI messages + audios” 格式：

输入 (jsonl，每行 NeMo manifest)：
  {"audio_filepath": "...", "text": "..."}

输出 (jsonl，每行)：
  {
    "messages": [
      {"role": "user", "content": "指令 + <audio>"},
      {"role": "assistant", "content": "转写文本"}
    ],
    "audios": ["..."]
  }

然后在 dataset_info.json 里注册：
  formatting: "sharegpt"
  columns: {"messages": "messages", "audios": "audios"}
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="NeMo manifest jsonl 路径")
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 OpenAI messages+audios jsonl 路径",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="请逐字转写下面这段语音，不要额外说明，只输出文本：<audio>",
        help="user 侧的指令模板，必须包含 <audio> 占位符",
    )
    p.add_argument(
        "--audio-key",
        type=str,
        default="audio_filepath",
        help="NeMo manifest 中表示音频路径的字段名",
    )
    p.add_argument(
        "--text-key",
        type=str,
        default="text",
        help="NeMo manifest 中表示转写文本的字段名",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多转换多少条（用于 debug），None 表示全部",
    )
    p.add_argument(
        "--s3-prefix",
        type=str,
        default=None,
        help="S3 前缀，用于追加在manifest中的audio_filepath字段值前",
    )
    return p.parse_args()


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def convert_manifest(
    input_path: str,
    output_path: str,
    prompt: str,
    audio_key: str = "audio_filepath",
    text_key: str = "text",
    max_samples: Optional[int] = None,
    s3_prefix: Optional[str] = None,
) -> None:
    if "<audio>" not in prompt:
        raise ValueError("prompt 中必须包含 <audio> 占位符，否则多模态对不上。")

    ensure_parent_dir(output_path)

    n_in = 0
    n_out = 0

    prefix = "" if s3_prefix is None else str(s3_prefix)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            n_in += 1
            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 跳过无法解析的行 {n_in}: {e}")
                continue

            raw_audio_path = obj.get(audio_key)
            audio_path = raw_audio_path.strip() if isinstance(raw_audio_path, str) else None
            text = obj.get(text_key)

            if not audio_path or not text:
                # 没有音频或没有文本的样本直接丢弃
                continue

            audio_path = prefix + audio_path

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": text,
                    },
                ],
                "audios": [audio_path],
            }

            # 简单校验：<audio> 数量必须等于 audios 数量
            placeholder_count = prompt.count("<audio>")
            if placeholder_count != len(sample["audios"]):
                print(
                    f"[WARN] line {n_in}: <audio> 个数({placeholder_count}) != audios({len(sample['audios'])})，跳过",
                )
                continue

            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_out += 1

            if max_samples is not None and n_out >= max_samples:
                break

    print(f"[OK] 读取 NeMo manifest {n_in} 行，生成 OpenAI+audio 样本 {n_out} 条 -> {output_path}")


def main() -> None:
    args = parse_args()
    if not args.s3_prefix:
        print("[WARN] s3_prefix is empty!")
        s3_prefix = ""
    else:
        s3_prefix = args.s3_prefix
    convert_manifest(
        input_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        audio_key=args.audio_key,
        text_key=args.text_key,
        max_samples=args.max_samples,
        s3_prefix=s3_prefix,
    )


if __name__ == "__main__":
    main()
