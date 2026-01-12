# Copyright 2025 the LlamaFactory team.
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

r"""Adjust prompt_pool weights for language-hint prompts in a JSONL dataset.

Author: yufeng.ma

Many ASR datasets use `prompt_pool` entries like:

  - "Just provide a lowercased, normalized transcription ..."
  - "Just provide a lowercased, normalized transcription ...\nThe language is xx."

This script groups pool entries by their *base prompt text* (the prompt text with any
"The language is ..." lines removed). For each group that contains both:
  - entries WITH a language line
  - entries WITHOUT a language line

it redistributes the group's total weight so that:

  sum(with_lang) : sum(without_lang) = WITH : WITHOUT

while keeping the group total unchanged. Within each subset (with/without), weights are
scaled proportionally to their original weights (or equally if all are zero).

This is intended for long-running S2T training where you want more language-conditioned
prompts without changing the overall normalized-vs-verbatim mixture.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LANG_LINE_PREFIX = "the language is "


def _is_language_line(line: str) -> bool:
    return line.strip().lower().startswith(LANG_LINE_PREFIX)


def _split_prompt_text(text: Any) -> tuple[str, bool]:
    """Return (base_text, has_language_line)."""
    if not isinstance(text, str):
        return "", False

    lines = text.splitlines()
    has_lang = any(_is_language_line(line) for line in lines)
    base_lines = [line for line in lines if not _is_language_line(line)]
    base_text = "\n".join(base_lines).strip()
    return base_text, has_lang


def _sanitize_weight(value: Any) -> float:
    try:
        w = float(value)
    except Exception:  # noqa: BLE001
        w = 1.0

    if not math.isfinite(w) or w < 0.0:
        return 0.0

    return w


def _parse_ratio(ratio: str) -> tuple[float, float]:
    s = str(ratio).strip()
    if ":" in s:
        left, right = s.split(":", 1)
    elif "," in s:
        left, right = s.split(",", 1)
    else:
        raise ValueError("Invalid ratio format. Use '6:4' or '6,4'.")

    a = float(left.strip())
    b = float(right.strip())
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0 or b <= 0:
        raise ValueError("Ratio values must be positive finite numbers.")
    return a, b


def _redistribute_subset(
    pool: list[Any],
    indices: list[int],
    target_total: float,
) -> None:
    if not indices:
        return

    orig: list[float] = []
    for i in indices:
        item = pool[i]
        if isinstance(item, dict):
            orig.append(_sanitize_weight(item.get("weight", 1.0)))
        else:
            orig.append(1.0)
    orig_sum = float(sum(orig))

    if orig_sum > 0:
        scale = float(target_total) / orig_sum
        for i, ow in zip(indices, orig):
            if isinstance(pool[i], dict):
                pool[i]["weight"] = float(ow) * scale
    else:
        per = float(target_total) / float(len(indices))
        for i in indices:
            if isinstance(pool[i], dict):
                pool[i]["weight"] = per


def adjust_prompt_pool_language_ratio(
    prompt_pool: Any,
    *,
    with_ratio: float,
    without_ratio: float,
) -> tuple[Any, bool]:
    """Return (new_prompt_pool, modified)."""
    if not isinstance(prompt_pool, list) or len(prompt_pool) == 0:
        return prompt_pool, False

    # Only consider dict entries with a "text" field; leave other schemas untouched.
    bases: list[str] = []
    has_lang_flags: list[bool] = []
    valid = True
    for item in prompt_pool:
        if not isinstance(item, dict):
            valid = False
            break
        base, has_lang = _split_prompt_text(item.get("text"))
        bases.append(base)
        has_lang_flags.append(has_lang)

    if not valid:
        return prompt_pool, False

    # Group by base prompt text.
    groups: dict[str, list[int]] = {}
    for idx, base in enumerate(bases):
        key = base if base else str(prompt_pool[idx].get("text", ""))
        groups.setdefault(key, []).append(idx)

    modified = False
    total_ratio = float(with_ratio + without_ratio)
    with_frac = float(with_ratio) / total_ratio
    without_frac = float(without_ratio) / total_ratio

    for indices in groups.values():
        with_idx = [i for i in indices if has_lang_flags[i]]
        without_idx = [i for i in indices if not has_lang_flags[i]]
        if not with_idx or not without_idx:
            continue

        group_total = 0.0
        for i in indices:
            group_total += _sanitize_weight(prompt_pool[i].get("weight", 1.0))

        # If the group has no positive weights, keep it as-is (sampling will be uniform anyway).
        if group_total <= 0.0:
            continue

        target_with = group_total * with_frac
        target_without = group_total * without_frac
        _redistribute_subset(prompt_pool, with_idx, target_with)
        _redistribute_subset(prompt_pool, without_idx, target_without)
        modified = True

    return prompt_pool, modified


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True, help="Input JSONL path.")
    p.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    p.add_argument(
        "--ratio",
        type=str,
        default="6:4",
        help="Desired ratio for (with_lang : without_lang). Default: 6:4.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=200_000,
        help="Print progress every N lines.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with_ratio, without_ratio = _parse_ratio(args.ratio)

    in_path = args.input
    out_path = args.output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    n_modified = 0

    with open(in_path, encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if args.log_every and n_in % int(args.log_every) == 0:
                print(f"[..] processed {n_in} lines, wrote {n_out}, modified={n_modified}")

            try:
                obj = json.loads(line)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Skip unparsable line {n_in}: {e}")
                continue

            pool = obj.get("prompt_pool")
            new_pool, changed = adjust_prompt_pool_language_ratio(
                pool,
                with_ratio=float(with_ratio),
                without_ratio=float(without_ratio),
            )
            if changed:
                obj["prompt_pool"] = new_pool
                n_modified += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"[OK] Read {n_in} lines, wrote {n_out} -> {out_path}. Modified {n_modified} samples.")


if __name__ == "__main__":
    main()
