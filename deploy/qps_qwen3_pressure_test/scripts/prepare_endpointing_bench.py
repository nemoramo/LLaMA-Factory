#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from transformers import AutoTokenizer


def build_system_prompt(lang: str) -> str:
    return (
        "You are an endpointing classifier for spoken dialog.\n"
        f"Language: {lang}\n\n"
        "Task: Given the conversation history and the current user ASR transcript, "
        "output EXACTLY ONE token:\n"
        "<EOU> | <CONT_USER> | <UNADDRESSED>\n\n"
        "Definitions:\n"
        "- <EOU>: user finished speaking; assistant should respond now.\n"
        "- <CONT_USER>: user will continue speaking / ASR is partial; wait.\n"
        "- <UNADDRESSED>: speech is not addressed to the assistant or is unrelated.\n\n"
        "Output constraints: no spaces, no punctuation, no explanation."
    )


def extract_context(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("context"), str):
        return sample["context"]

    messages = sample.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if (
                isinstance(msg, dict)
                and msg.get("role") == "user"
                and isinstance(msg.get("content"), str)
            ):
                return msg["content"]

    raise ValueError("Sample has no usable 'context' or user message in 'messages'.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw jsonl")
    parser.add_argument("--output", required=True, help="Output bench jsonl")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer/model path")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If >0, only process first N samples",
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=0,
        help="If >0, drop prompts whose token length exceeds this value",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    kept = 0
    dropped = 0
    token_lens: List[int] = []

    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            lang = sample.get("lang", "en")
            if not isinstance(lang, str):
                lang = "en"

            system_prompt = build_system_prompt(lang)
            context = extract_context(sample)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            token_lens.append(prompt_len)

            processed += 1
            if args.max_prompt_len > 0 and prompt_len > args.max_prompt_len:
                dropped += 1
            else:
                record = {
                    "prompt": prompt,
                    "dialogue_id": sample.get("dialogue_id"),
                    "turn": sample.get("turn"),
                    "lang": lang,
                    "label": sample.get("label"),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

            if args.max_samples > 0 and processed >= args.max_samples:
                break

    if token_lens:
        p95_idx = max(0, int(len(token_lens) * 0.95) - 1)
        sorted_lens = sorted(token_lens)
        print(
            "Token length stats:",
            f"min={sorted_lens[0]}",
            f"avg={mean(sorted_lens):.2f}",
            f"p95={sorted_lens[p95_idx]}",
            f"max={sorted_lens[-1]}",
        )
    print(
        f"Done. processed={processed}, kept={kept}, dropped={dropped}, output={output_path}"
    )


if __name__ == "__main__":
    main()
