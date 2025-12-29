#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert TorchTune speech_endpointing *.manifest (JSONL) to LLaMA-Factory OpenAI-style JSONL.

Why:
  - TorchTune endpointing datasets are typically JSONL with fields like:
      dialogue_id, turn, lang, label, context, messages, meta, ...
  - LLaMA-Factory can directly consume OpenAI-style messages:
      {"messages": [{"role": "...", "content": "..."}, ...]}
    via dataset_info.json with formatting="openai".

Notes:
  - LLaMA-Factory neat packing happens during tokenization (packing/neat_packing) and does NOT
    require pre-packing in the dataset file. This script only normalizes the sample format.
  - This script is stdlib-only (no transformers dependency).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from typing import Any


DEFAULT_LABELS = ["<EOU>", "<CONT_USER>", "<UNADDRESSED>"]
DEFAULT_SYSTEM = (
    "You are a turn-taking judge. Decide whether the LAST user utterance is complete (<EOU>), "
    "likely to continue (<CONT_USER>), or not addressed to the assistant (<UNADDRESSED>). "
    "Output EXACTLY one tag and nothing else."
)


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}, got {type(obj).__name__}")
            yield obj


def _normalize_label(label: Any) -> str:
    if label is None:
        return ""
    if not isinstance(label, str):
        return str(label)
    return label.strip()


def _as_openai_message(role: Any, content: Any) -> dict[str, str]:
    return {"role": str(role), "content": "" if content is None else str(content)}


def _validate_openai_messages(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list) or not messages:
        return "messages must be a non-empty list"
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            return f"messages[{i}] is not an object"
        if "role" not in m or "content" not in m:
            return f"messages[{i}] missing 'role' or 'content'"

    # LLaMA-Factory openai converter expects (after optional leading system):
    # user/assistant alternating, ending with assistant.
    idx = 0
    if messages and str(messages[0].get("role")) == "system":
        idx = 1
    core = messages[idx:]
    if len(core) < 2 or (len(core) % 2 != 0):
        return "after optional system, message count must be even and >= 2 (user/assistant pairs)"
    for j, m in enumerate(core):
        role = str(m.get("role"))
        if j % 2 == 0 and role != "user":
            return f"invalid role order: core[{j}] expected 'user', got {role!r}"
        if j % 2 == 1 and role != "assistant":
            return f"invalid role order: core[{j}] expected 'assistant', got {role!r}"
    return ""


def build_messages(
    record: dict[str, Any],
    *,
    labels: list[str],
    messages_field: str,
    label_field: str,
    context_field: str,
    system_text: str,
    add_system_if_missing: bool,
    overwrite_assistant_label: bool,
) -> tuple[list[dict[str, str]], int]:
    """Returns: (messages, num_fixes).

    - num_fixes counts how many times we inserted/fixed assistant label.
    """
    allowed = set(labels)
    num_fixes = 0

    label = _normalize_label(record.get(label_field))

    raw_messages = record.get(messages_field)
    messages: list[dict[str, str]] = []
    if isinstance(raw_messages, list) and raw_messages:
        # Normalize existing messages.
        for m in raw_messages:
            if isinstance(m, dict):
                messages.append(_as_openai_message(m.get("role"), m.get("content")))
            else:
                # Unexpected structure; fall back to string repr.
                messages.append(_as_openai_message("user", str(m)))
    else:
        # Build messages from context + label
        context = record.get(context_field)
        if context is None:
            context = record.get("text")
        context = "" if context is None else str(context)
        if system_text and add_system_if_missing:
            messages.append(_as_openai_message("system", system_text))
        messages.append(_as_openai_message("user", context))
        messages.append(_as_openai_message("assistant", label))
        num_fixes += 1

    # Ensure leading system exists if requested
    if add_system_if_missing and system_text:
        if not messages or messages[0]["role"] != "system":
            messages.insert(0, _as_openai_message("system", system_text))

    # Ensure we end with an assistant label
    if label:
        if messages and messages[-1]["role"] == "assistant":
            cur = (messages[-1].get("content") or "").strip()
            if (not cur) or overwrite_assistant_label:
                messages[-1]["content"] = label
                num_fixes += 1
            elif (cur not in allowed) and (label in allowed):
                # If current assistant is not a tag but record.label is a tag, prefer label.
                messages[-1]["content"] = label
                num_fixes += 1
        else:
            messages.append(_as_openai_message("assistant", label))
            num_fixes += 1

    return messages, num_fixes


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--input", required=True, help="TorchTune *.manifest (JSONL) path")
    ap.add_argument("--output", required=True, help="Output JSONL for LLaMA-Factory (OpenAI messages)")
    ap.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated label tokens (must match training config add_special_tokens)",
    )
    ap.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Default system prompt inserted if missing (set to empty to disable)",
    )
    ap.add_argument("--messages-field", default="messages", help="Field name for messages list")
    ap.add_argument("--label-field", default="label", help="Field name for label string")
    ap.add_argument("--context-field", default="context", help="Field name for context string (fallback if no messages)")
    ap.add_argument(
        "--add-system-if-missing",
        action="store_true",
        default=True,
        help="Insert `--system` as the first message if no leading system message exists",
    )
    ap.add_argument(
        "--no-add-system-if-missing",
        dest="add_system_if_missing",
        action="store_false",
        help="Do not insert a system message",
    )
    ap.add_argument(
        "--overwrite-assistant-label",
        action="store_true",
        default=False,
        help="Always overwrite the last assistant content with record.label when present",
    )
    ap.add_argument(
        "--keep-fields",
        default="dialogue_id,turn,lang,label",
        help=(
            "Comma-separated fields to keep from the original record (in addition to messages). "
            "Empty means keep nothing except messages."
        ),
    )
    ap.add_argument("--max-samples", type=int, default=0, help="Process at most N samples (0 = all)")
    ap.add_argument("--strict", action="store_true", help="Drop records that fail OpenAI message validation")
    ap.add_argument(
        "--print-dataset-info",
        action="store_true",
        help="Print a dataset_info.json snippet (formatting=openai) to stdout",
    )
    ap.add_argument(
        "--dataset-key",
        default="speech_endpointing_train",
        help="Dataset key used in dataset_info.json snippet (only with --print-dataset-info)",
    )
    args = ap.parse_args()

    labels = [s.strip() for s in str(args.labels).split(",") if s.strip()]
    if len(labels) < 2:
        raise SystemExit("[ERROR] --labels must contain at least 2 tokens.")

    keep_fields = [s.strip() for s in str(args.keep_fields).split(",") if s.strip()]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    n_in = 0
    n_out = 0
    n_fixed = 0
    n_dropped = 0
    label_counts: dict[str, int] = {}

    with open(args.output, "w", encoding="utf-8") as w:
        for record in _iter_jsonl(args.input):
            n_in += 1
            messages, fixes = build_messages(
                record,
                labels=labels,
                messages_field=args.messages_field,
                label_field=args.label_field,
                context_field=args.context_field,
                system_text=str(args.system or ""),
                add_system_if_missing=bool(args.add_system_if_missing),
                overwrite_assistant_label=bool(args.overwrite_assistant_label),
            )
            n_fixed += fixes

            err = _validate_openai_messages(messages)
            if err:
                if args.strict:
                    n_dropped += 1
                    continue
                else:
                    # Keep record but still warn.
                    print(f"[WARN] {args.input}: record#{n_in}: {err}", file=sys.stderr)

            # Update label stats from last assistant content
            last = next((m for m in reversed(messages) if m.get("role") == "assistant"), None)
            if last is not None:
                lab = (last.get("content") or "").strip()
                if lab:
                    label_counts[lab] = label_counts.get(lab, 0) + 1

            out: dict[str, Any] = {"messages": messages}
            for k in keep_fields:
                if k in record:
                    out[k] = record[k]
            w.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_out += 1

            if args.max_samples and n_out >= args.max_samples:
                break

    print(f"[done] input={args.input} output={args.output}")
    print(f"[stats] read={n_in} wrote={n_out} dropped={n_dropped} label_fixes={n_fixed}")
    if label_counts:
        top = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("[labels] top:", ", ".join([f"{k}={v}" for k, v in top]))

    if args.print_dataset_info:
        # Note: openai formatting needs explicit tags because defaults are ShareGPT-like.
        snippet = {
            args.dataset_key: {
                "file_name": os.path.basename(args.output),
                "formatting": "openai",
                "columns": {"messages": "messages"},
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant",
                    "system_tag": "system",
                    "observation_tag": "observation",
                    "function_tag": "function",
                },
            }
        }
        print("\n# dataset_info.json snippet")
        print(json.dumps(snippet, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

