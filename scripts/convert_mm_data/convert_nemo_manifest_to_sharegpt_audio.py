# Copyright 2025 the LlamaFactory team.
# Additional author: ramos.ma (GitHub: nemoramo).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""将 NeMo manifest (audio_filepath + text/original_text) 转为 LLaMA-Factory 训练用的 “OpenAI messages + audios” 格式.

并可选生成 `prompt_pool` 以支持训练时动态采样：

输入 (jsonl，每行 NeMo manifest)：
  {"audio_filepath": "...", "text": "...", "original_text": "...", "lang": "..."}

输出 (jsonl，每行)：
  {
    "messages": [
      {"role": "user", "content": "base prompt + <audio>"},
      {"role": "assistant", "content": "默认 completion（通常为 text）"}
    ],
    "audios": ["..."],
    "prompt_pool": [
      {"text": "<suffix>", "completion": "<text>", "weight": 0.8},
      {"text": "<suffix>", "completion": "<original_text>", "weight": 0.2}
    ]
  }

`prompt_pool.text` 会在训练时被追加到最后一个 user 消息后；如果 entry 带有
`completion`，则会覆盖 assistant 的目标文本，从而实现 original/text 的动态切换。

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
        default="Transcribe the audio verbatim. Do not add any commentary. Only output the text: <audio>",
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
        "--original-text-key",
        type=str,
        default="original_text",
        help="NeMo manifest 中表示原始/未规范化文本的字段名（可缺省）",
    )
    p.add_argument(
        "--lang-key",
        type=str,
        default="lang",
        help="NeMo manifest 中表示语种的字段名（也会自动尝试 language）",
    )
    p.add_argument(
        "--disable-prompt-pool",
        action="store_true",
        help="不输出 prompt_pool（保持旧行为）",
    )
    p.add_argument(
        "--original-prob",
        type=float,
        default=0.2,
        help="抽到 original_text 作为目标 completion 的概率（0-1），无 original_text 时自动忽略",
    )
    p.add_argument(
        "--lang-hint-prob",
        type=float,
        default=0.1,
        help="抽到带语言提示 suffix 的概率（0-1），无 lang 时自动忽略",
    )
    p.add_argument(
        "--normalized-suffix",
        type=str,
        default="Please provide a clean, normalized transcription (lowercase, no punctuation).",
        help=(
            "Suffix for normalized(text) targets (typically no capitalization/punctuation); "
            "supports {lang}/{has_digits} placeholders."
        ),
    )
    p.add_argument(
        "--original-suffix",
        type=str,
        default="Please transcribe verbatim, preserving casing and punctuation.",
        help="Suffix for original_text targets; supports {lang}/{has_digits} placeholders.",
    )
    p.add_argument(
        "--lang-hint-template",
        type=str,
        default="The language is {lang}.",
        help="Language-hint suffix template appended when lang_hint_prob hits.",
    )
    # Digit-related hints are left empty by default. We recommend handling digit hints online
    # in the dynamic sampling dataset if needed.
    p.add_argument(
        "--digits-hint-template",
        type=str,
        default="",
        help="(Optional, offline) Extra digits hint for normalized targets; default empty.",
    )
    p.add_argument(
        "--digits-original-hint-template",
        type=str,
        default="",
        help="(Optional, offline) Extra digits hint for original targets; default empty.",
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


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def _has_digits(*texts: Optional[str]) -> bool:
    for t in texts:
        if not t:
            continue
        if any(ch.isdigit() for ch in t):
            return True
    return False


def _format_suffix(
    base_suffix: str,
    *,
    lang: Optional[str],
    has_digits: bool,
    lang_hint: Optional[str],
    digits_hint: Optional[str],
) -> str:
    parts = []
    if base_suffix:
        parts.append(base_suffix)
    if lang_hint:
        parts.append(lang_hint)
    if has_digits and digits_hint:
        parts.append(digits_hint)
    return "\n".join([p for p in parts if p])


def convert_manifest(
    input_path: str,
    output_path: str,
    prompt: str,
    audio_key: str = "audio_filepath",
    text_key: str = "text",
    original_text_key: str = "original_text",
    lang_key: str = "lang",
    disable_prompt_pool: bool = False,
    original_prob: float = 0.2,
    lang_hint_prob: float = 0.1,
    normalized_suffix: str = "Please provide a clean, normalized transcription (lowercase, no punctuation).",
    original_suffix: str = "Please transcribe verbatim, preserving casing and punctuation.",
    lang_hint_template: str = "The language is {lang}.",
    digits_hint_template: str = "",
    digits_original_hint_template: str = "",
    max_samples: Optional[int] = None,
    s3_prefix: Optional[str] = None,
) -> None:
    if "<audio>" not in prompt:
        raise ValueError("prompt 中必须包含 <audio> 占位符，否则多模态对不上。")

    ensure_parent_dir(output_path)

    n_in = 0
    n_out = 0

    prefix = "" if s3_prefix is None else str(s3_prefix)

    with open(input_path, encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
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
            normalized_text = obj.get(text_key)
            original_text = obj.get(original_text_key) or normalized_text

            if not audio_path or not (normalized_text or original_text):
                # 没有音频或没有文本的样本直接丢弃
                continue

            audio_path = prefix + audio_path

            normalized_text = normalized_text or original_text or ""
            original_text = original_text or normalized_text or ""

            lang = obj.get(lang_key) or obj.get("language") or None

            has_digits = obj.get("has_digits")
            if has_digits is None:
                has_digits = _has_digits(normalized_text, original_text)
            else:
                has_digits = _coerce_bool(has_digits)

            pool = []
            if not disable_prompt_pool:
                p_orig = min(max(float(original_prob), 0.0), 1.0)
                p_lang = min(max(float(lang_hint_prob), 0.0), 1.0)

                has_lang = bool(lang) and bool(lang_hint_template)
                lang_hint = None
                if has_lang:
                    try:
                        lang_hint = lang_hint_template.format(lang=lang, has_digits=has_digits)
                    except Exception:  # noqa: BLE001
                        lang_hint = str(lang_hint_template).replace("{lang}", str(lang))

                def fmt_base(s: str) -> str:
                    if not s:
                        return ""
                    try:
                        return s.format(lang=lang or "", has_digits=has_digits)
                    except Exception:  # noqa: BLE001
                        return s

                digits_hint = fmt_base(digits_hint_template)
                digits_orig_hint = fmt_base(digits_original_hint_template)

                norm_total = 1.0
                orig_total = 0.0
                if original_text and original_text != normalized_text and p_orig > 0:
                    orig_total = p_orig
                    norm_total = 1.0 - p_orig

                # normalized entries
                if norm_total > 0:
                    base_norm_suffix = fmt_base(normalized_suffix)
                    if has_lang and p_lang > 0:
                        w_lang = norm_total * p_lang
                        w_nolang = norm_total * (1.0 - p_lang)
                        if w_nolang > 0:
                            pool.append(
                                {
                                    "text": _format_suffix(
                                        base_norm_suffix,
                                        lang=lang,
                                        has_digits=bool(has_digits),
                                        lang_hint=None,
                                        digits_hint=digits_hint,
                                    ),
                                    "completion": normalized_text,
                                    "weight": w_nolang,
                                }
                            )
                        if w_lang > 0:
                            pool.append(
                                {
                                    "text": _format_suffix(
                                        base_norm_suffix,
                                        lang=lang,
                                        has_digits=bool(has_digits),
                                        lang_hint=lang_hint,
                                        digits_hint=digits_hint,
                                    ),
                                    "completion": normalized_text,
                                    "weight": w_lang,
                                }
                            )
                    else:
                        pool.append(
                            {
                                "text": _format_suffix(
                                    base_norm_suffix,
                                    lang=lang,
                                    has_digits=bool(has_digits),
                                    lang_hint=None,
                                    digits_hint=digits_hint,
                                ),
                                "completion": normalized_text,
                                "weight": norm_total,
                            }
                        )

                # original entries
                if orig_total > 0:
                    base_orig_suffix = fmt_base(original_suffix)
                    if has_lang and p_lang > 0:
                        w_lang = orig_total * p_lang
                        w_nolang = orig_total * (1.0 - p_lang)
                        if w_nolang > 0:
                            pool.append(
                                {
                                    "text": _format_suffix(
                                        base_orig_suffix,
                                        lang=lang,
                                        has_digits=bool(has_digits),
                                        lang_hint=None,
                                        digits_hint=digits_orig_hint,
                                    ),
                                    "completion": original_text,
                                    "weight": w_nolang,
                                }
                            )
                        if w_lang > 0:
                            pool.append(
                                {
                                    "text": _format_suffix(
                                        base_orig_suffix,
                                        lang=lang,
                                        has_digits=bool(has_digits),
                                        lang_hint=lang_hint,
                                        digits_hint=digits_orig_hint,
                                    ),
                                    "completion": original_text,
                                    "weight": w_lang,
                                }
                            )
                    else:
                        pool.append(
                            {
                                "text": _format_suffix(
                                    base_orig_suffix,
                                    lang=lang,
                                    has_digits=bool(has_digits),
                                    lang_hint=None,
                                    digits_hint=digits_orig_hint,
                                ),
                                "completion": original_text,
                                "weight": orig_total,
                            }
                        )

            sample = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": normalized_text or original_text,
                    },
                ],
                "audios": [audio_path],
            }
            if pool:
                sample["prompt_pool"] = pool

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
        original_text_key=args.original_text_key,
        lang_key=args.lang_key,
        disable_prompt_pool=args.disable_prompt_pool,
        original_prob=args.original_prob,
        lang_hint_prob=args.lang_hint_prob,
        normalized_suffix=args.normalized_suffix,
        original_suffix=args.original_suffix,
        lang_hint_template=args.lang_hint_template,
        digits_hint_template=args.digits_hint_template,
        digits_original_hint_template=args.digits_original_hint_template,
        max_samples=args.max_samples,
        s3_prefix=s3_prefix,
    )


if __name__ == "__main__":
    main()
