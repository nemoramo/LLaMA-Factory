#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TAGS = ("<EOU>", "<CONT_USER>", "<UNADDRESSED>")
MERGED_TAGS = ("<EOU>", "<CONT_USER>")
TAG_RE = re.compile(r"<EOU>|<CONT_USER>|<UNADDRESSED>")
PROBE_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a turn-taking judge. Decide whether the LAST user utterance is complete (<EOU>), "
            "likely to continue (<CONT_USER>), or not addressed to the assistant (<UNADDRESSED>). "
            "Output EXACTLY one tag and nothing else."
        ),
    },
    {"role": "user", "content": "[CONTEXT]\nUser: okay thanks\n[/CONTEXT]"},
]


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {path}")
            items.append(obj)
    return items


def _normalize_tag(text: Optional[str]) -> str:
    if text is None:
        return "NULL"
    s = str(text).strip()
    m = TAG_RE.search(s)
    if m:
        return m.group(0)
    return s if s else "EMPTY"


def _normalize_lang(lang: Optional[str]) -> str:
    if lang is None:
        return "UNKNOWN_LANG"
    s = str(lang).strip()
    if not s:
        return "UNKNOWN_LANG"
    lowered = s.lower()
    if lowered in {"ha", "hausa"}:
        return "hausa"
    return s


def _merge_unaddressed_as_eou(tag: str) -> str:
    return "<EOU>" if tag == "<UNADDRESSED>" else tag


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _coerce_turn(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.upper() == "N/A":
            return None
        try:
            return int(s)
        except Exception:
            return None
    return None


def _is_first_turn(turn: Any) -> bool:
    t = _coerce_turn(turn)
    return t is None or t <= 1


def _init_confusion(tags: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {gold: {pred: 0 for pred in tags} for gold in tags}


def _summarize_confusion(
    confusion: dict[str, dict[str, int]], row_totals: dict[str, int], tags: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float]] | float]:
    total = sum(row_totals.values())
    correct = sum(confusion[tag][tag] for tag in tags)
    col_totals = {pred: sum(confusion[gold][pred] for gold in tags) for pred in tags}

    per_label: dict[str, dict[str, float]] = {}
    macro_f1 = 0.0
    for tag in tags:
        tp = confusion[tag][tag]
        precision = _safe_div(tp, col_totals[tag])
        recall = _safe_div(tp, row_totals[tag])
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_label[tag] = {"precision": precision, "recall": recall, "f1": f1}
        macro_f1 += f1

    return {
        "accuracy": _safe_div(correct, total),
        "macro_f1": macro_f1 / float(len(tags)) if tags else 0.0,
        "per_label": per_label,
    }


def _compute_tag_confusion(results: list[dict[str, Any]]) -> list[list[int]]:
    tag_to_i = {t: i for i, t in enumerate(TAGS)}
    mat = [[0 for _ in TAGS] for _ in TAGS]
    for r in results:
        y = r.get("label")
        yhat = r.get("pred")
        if y not in tag_to_i or yhat not in tag_to_i:
            continue
        mat[tag_to_i[y]][tag_to_i[yhat]] += 1
    return mat


def _summarize_tag_confusion(mat: list[list[int]]) -> dict[str, Any]:
    row_sum = [sum(row) for row in mat]
    tag_confusion = {TAGS[i]: {TAGS[j]: mat[i][j] for j in range(len(TAGS))} for i in range(len(TAGS))}
    row_totals = {TAGS[i]: row_sum[i] for i in range(len(TAGS))}
    summary = _summarize_confusion(tag_confusion, row_totals, TAGS)

    eou_i = 0
    cont_i = 1
    unad_i = 2
    return {
        "total": int(sum(row_sum)),
        "correct": int(sum(mat[i][i] for i in range(len(TAGS)))),
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "confusion": tag_confusion,
        "per_label": summary["per_label"],
        "kpi": {
            "FAR_unad": _safe_div(row_sum[unad_i] - mat[unad_i][unad_i], row_sum[unad_i]),
            "Interrupt": _safe_div(mat[cont_i][eou_i], row_sum[cont_i]),
            "Delay": _safe_div(mat[eou_i][cont_i], row_sum[eou_i]),
            "Missed": _safe_div(mat[eou_i][unad_i], row_sum[eou_i]),
        },
        "_row_sum": row_totals,
    }


def _compute_merged_tag_confusion(results: list[dict[str, Any]]) -> list[list[int]]:
    tag_to_i = {t: i for i, t in enumerate(MERGED_TAGS)}
    mat = [[0 for _ in MERGED_TAGS] for _ in MERGED_TAGS]
    for r in results:
        gold = _merge_unaddressed_as_eou(str(r.get("label")))
        pred = _merge_unaddressed_as_eou(str(r.get("pred")))
        if gold not in tag_to_i or pred not in tag_to_i:
            continue
        mat[tag_to_i[gold]][tag_to_i[pred]] += 1
    return mat


def _summarize_merged_tag_confusion(mat: list[list[int]]) -> dict[str, Any]:
    row_sum = [sum(row) for row in mat]
    tag_confusion = {
        MERGED_TAGS[i]: {MERGED_TAGS[j]: mat[i][j] for j in range(len(MERGED_TAGS))}
        for i in range(len(MERGED_TAGS))
    }
    row_totals = {MERGED_TAGS[i]: row_sum[i] for i in range(len(MERGED_TAGS))}
    summary = _summarize_confusion(tag_confusion, row_totals, MERGED_TAGS)

    eou_i = 0
    cont_i = 1
    return {
        "total": int(sum(row_sum)),
        "correct": int(sum(mat[i][i] for i in range(len(MERGED_TAGS)))),
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "confusion": tag_confusion,
        "per_label": summary["per_label"],
        "kpi": {
            "Interrupt": _safe_div(mat[cont_i][eou_i], row_sum[cont_i]),
            "Delay": _safe_div(mat[eou_i][cont_i], row_sum[eou_i]),
        },
        "_row_sum": row_totals,
    }


def _print_eval_block(title: str, tag_summary: dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print("Confusion (gold -> pred):")
    row_sum = tag_summary.get("_row_sum", {})
    conf = tag_summary["confusion"]
    for gold in TAGS:
        row = [conf[gold].get(pred, 0) for pred in TAGS]
        print(f"{gold:<14} {row} sum={int(row_sum.get(gold, sum(row))):>5d}")

    print(f"Accuracy: {tag_summary['accuracy']:.4f} | Macro-F1: {tag_summary['macro_f1']:.4f}")
    for tag in TAGS:
        metric = tag_summary["per_label"][tag]
        print(f"  {tag:<14} P={metric['precision']:.4f} R={metric['recall']:.4f} F1={metric['f1']:.4f}")

    kpi = tag_summary["kpi"]
    print(
        f"[KPI] FAR_unad={kpi['FAR_unad']:.4f} | Interrupt={kpi['Interrupt']:.4f} | "
        f"Delay={kpi['Delay']:.4f} | Missed={kpi['Missed']:.4f}"
    )


def _print_eval_block_merged(title: str, tag_summary: dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print("Confusion (gold -> pred):")
    row_sum = tag_summary.get("_row_sum", {})
    conf = tag_summary["confusion"]
    for gold in MERGED_TAGS:
        row = [conf[gold].get(pred, 0) for pred in MERGED_TAGS]
        print(f"{gold:<14} {row} sum={int(row_sum.get(gold, sum(row))):>5d}")

    print(f"Accuracy: {tag_summary['accuracy']:.4f} | Macro-F1: {tag_summary['macro_f1']:.4f}")
    for tag in MERGED_TAGS:
        metric = tag_summary["per_label"][tag]
        print(f"  {tag:<14} P={metric['precision']:.4f} R={metric['recall']:.4f} F1={metric['f1']:.4f}")

    kpi = tag_summary["kpi"]
    print(f"[KPI] Interrupt={kpi['Interrupt']:.4f} | Delay={kpi['Delay']:.4f}")


@dataclass(frozen=True)
class Sample:
    idx: int
    dialogue_id: Optional[str]
    turn: Optional[int]
    lang: Optional[str]
    label: str
    prompt_messages: list[dict[str, str]]


def _extract_sample(obj: dict[str, Any], idx: int) -> Sample:
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("missing/invalid `messages`")

    assistant_indices = [i for i, m in enumerate(messages) if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistant_indices:
        raise ValueError("no assistant message found to use as label")

    label = obj.get("label")
    if not isinstance(label, str) or not label.strip():
        label = str(messages[assistant_indices[-1]].get("content", "")).strip()

    prompt_messages = messages[: assistant_indices[-1]]
    prompt_messages = [
        {"role": str(m.get("role")), "content": str(m.get("content", ""))}
        for m in prompt_messages
        if isinstance(m, dict)
    ]

    return Sample(
        idx=idx,
        dialogue_id=obj.get("dialogue_id"),
        turn=_coerce_turn(obj.get("turn")),
        lang=obj.get("lang"),
        label=_normalize_tag(label),
        prompt_messages=prompt_messages,
    )


def _build_tag_eval_per_lang(
    all_results: list[dict[str, Any]], valid_tag_results: list[dict[str, Any]]
) -> dict[str, Any]:
    lang_total: dict[str, int] = {}
    for r in all_results:
        lang = _normalize_lang(r.get("lang"))
        lang_total[lang] = lang_total.get(lang, 0) + 1

    lang_groups: dict[str, list[dict[str, Any]]] = {}
    for r in valid_tag_results:
        lang_groups.setdefault(_normalize_lang(r.get("lang")), []).append(r)

    out: dict[str, Any] = {}
    for lang, group in sorted(lang_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        summary = _summarize_tag_confusion(_compute_tag_confusion(group))
        out[lang] = {
            "valid": len(group),
            "invalid": max(lang_total.get(lang, 0) - len(group), 0),
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "confusion_matrix": summary["confusion"],
            "per_label": summary["per_label"],
            "kpi": summary["kpi"],
        }
    return out


def _build_tag_eval_merge_unad_as_eou_per_lang(valid_tag_results: list[dict[str, Any]]) -> dict[str, Any]:
    lang_groups: dict[str, list[dict[str, Any]]] = {}
    for r in valid_tag_results:
        lang_groups.setdefault(_normalize_lang(r.get("lang")), []).append(r)

    out: dict[str, Any] = {}
    for lang, group in sorted(lang_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        summary = _summarize_merged_tag_confusion(_compute_merged_tag_confusion(group))
        out[lang] = {
            "valid": len(group),
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "confusion_matrix": summary["confusion"],
            "per_label": summary["per_label"],
            "kpi": summary["kpi"],
        }
    return out


def _dtype_from_str(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def _load_model_and_tokenizer(
    base_model: str, adapter_path: Optional[str], *, dtype: torch.dtype, attn_impl: str
) -> tuple[Any, Any]:
    tokenizer_src = adapter_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if attn_impl == "flash2":
        load_kwargs["attn_implementation"] = "flash_attention_2"
    elif attn_impl == "sdpa":
        load_kwargs["attn_implementation"] = "sdpa"
    elif attn_impl == "eager":
        load_kwargs["attn_implementation"] = "eager"
    elif attn_impl != "auto":
        raise ValueError(f"Unknown attn_impl: {attn_impl}")

    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def _resolve_tag_token_ids(tokenizer: Any) -> dict[str, int]:
    tag_token_ids: dict[str, int] = {}
    missing_tags: list[str] = []
    for tag in TAGS:
        token_id = tokenizer.convert_tokens_to_ids(tag)
        round_trip = tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None
        decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False) if token_id is not None else None
        if token_id is None or int(token_id) < 0 or (round_trip != tag and decoded != tag):
            missing_tags.append(tag)
            continue
        tag_token_ids[tag] = int(token_id)

    if missing_tags:
        raise ValueError(f"Tokenizer is missing endpointing label tokens: {', '.join(missing_tags)}")

    return tag_token_ids


def _iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _device_of(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise RuntimeError("Model has no parameters.") from exc


def _token_debug_text(tokenizer: Any, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(token, str) and token:
        return token
    return tokenizer.decode([int(token_id)], skip_special_tokens=False)


def _summarize_topk(logits: torch.Tensor, tokenizer: Any, k: int) -> list[dict[str, Any]]:
    topk = min(max(1, int(k)), int(logits.shape[-1]))
    scores, token_ids = torch.topk(logits, k=topk, dim=-1)
    items: list[dict[str, Any]] = []
    for rank, (score, token_id) in enumerate(zip(scores.tolist(), token_ids.tolist(), strict=True), start=1):
        token_text = _token_debug_text(tokenizer, int(token_id))
        items.append(
            {
                "rank": rank,
                "token_id": int(token_id),
                "token": token_text,
                "normalized": _normalize_tag(token_text),
                "logit": float(score),
            }
        )
    return items


def _run_export_prompt_probe(
    tokenizer: Any,
    model: Any,
    *,
    tag_token_ids: dict[str, int],
    max_length: int,
    topk: int,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(PROBE_MESSAGES, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    device = _device_of(model)
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        logits = model(**enc).logits[0, -1].float().detach().cpu()

    topk_items = _summarize_topk(logits, tokenizer, k=topk)
    top3_tokens = topk_items[:3]
    top3_normalized = {_normalize_tag(item["token"]) for item in top3_tokens}
    tag_ranks = {
        tag: int((logits > logits[token_id]).sum().item()) + 1 for tag, token_id in tag_token_ids.items()
    }
    tag_logits = {tag: float(logits[token_id].item()) for tag, token_id in tag_token_ids.items()}
    tag_prob_tensor = torch.softmax(
        torch.tensor([tag_logits[tag] for tag in TAGS], dtype=torch.float32),
        dim=-1,
    )
    tag_probs = {tag: float(prob.item()) for tag, prob in zip(TAGS, tag_prob_tensor, strict=True)}
    passed = top3_normalized == set(TAGS)

    warning = None
    if not passed:
        warning = (
            "Export prompt probe failed: the three endpointing label tokens are not the full-vocab top-3 next-token "
            "candidates for the canonical endpointing prompt. This usually means the export, template override, "
            "special-token resize, or prompt format is inconsistent."
        )

    return {
        "passed": passed,
        "warning": warning,
        "prompt_messages": PROBE_MESSAGES,
        "topk": topk_items,
        "top3": top3_tokens,
        "tag_ranks": tag_ranks,
        "tag_logits": tag_logits,
        "tag_probs": tag_probs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True, help="Base HF model name or local merged model path.")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA checkpoint dir. If omitted, evaluate the base model as-is (for merged exports).",
    )
    parser.add_argument("--dataset", required=True, help="Eval JSONL in OpenAI messages format.")
    parser.add_argument("--out-dir", required=True, help="Directory to write outputs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn-impl", default="flash2", choices=["auto", "flash2", "sdpa", "eager"])
    parser.add_argument("--limit", type=int, default=0, help="Optional: only run first N samples (0 = all).")
    parser.add_argument("--topk", type=int, default=10, help="How many full-vocab candidates to keep for debug output.")
    parser.add_argument(
        "--skip-export-probe",
        action="store_true",
        help="Skip the canonical prompt check that requires the three label tokens to occupy full-vocab top-3.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, "pred.jsonl")
    summary_path = os.path.join(args.out_dir, "summary.json")

    items = _read_jsonl(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]

    samples = [_extract_sample(obj, idx=i) for i, obj in enumerate(items)]
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    dtype = _dtype_from_str(args.dtype)
    tokenizer, model = _load_model_and_tokenizer(args.base_model, args.adapter, dtype=dtype, attn_impl=args.attn_impl)
    tag_token_ids = _resolve_tag_token_ids(tokenizer)
    print(f"Resolved tag token ids: {tag_token_ids}")

    export_probe = None
    if not args.skip_export_probe:
        export_probe = _run_export_prompt_probe(
            tokenizer,
            model,
            tag_token_ids=tag_token_ids,
            max_length=args.max_length,
            topk=args.topk,
        )
        print("\n--- Export Prompt Probe ---")
        for item in export_probe["top3"]:
            print(
                f"top{item['rank']}: token={item['token']!r} token_id={item['token_id']} "
                f"logit={item['logit']:.4f}"
            )
        if export_probe["warning"] is not None:
            print(f"[WARN] {export_probe['warning']}")
        else:
            print("[OK] The three endpointing label tokens occupy the full-vocab top-3 positions.")

    device = _device_of(model)
    tag_id_tensor = torch.tensor([tag_token_ids[tag] for tag in TAGS], device=device, dtype=torch.long)

    results: list[dict[str, Any]] = []
    t0 = time.time()
    for batch in _iter_batches(samples, args.batch_size):
        prompts = [tokenizer.apply_chat_template(s.prompt_messages, tokenize=False, add_generation_prompt=True) for s in batch]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.inference_mode():
            next_logits = model(**enc).logits[:, -1, :].float()

        tag_logits = next_logits.index_select(dim=1, index=tag_id_tensor)
        tag_probs = torch.softmax(tag_logits, dim=-1)
        topk_scores, topk_ids = torch.topk(next_logits, k=min(args.topk, int(next_logits.shape[-1])), dim=-1)
        pred_indices = torch.argmax(tag_logits, dim=-1)

        for row_idx, sample in enumerate(batch):
            pred_tag = TAGS[int(pred_indices[row_idx].item())]
            row_topk = []
            for rank, (score, token_id) in enumerate(
                zip(topk_scores[row_idx].tolist(), topk_ids[row_idx].tolist(), strict=True),
                start=1,
            ):
                token_text = _token_debug_text(tokenizer, int(token_id))
                row_topk.append(
                    {
                        "rank": rank,
                        "token_id": int(token_id),
                        "token": token_text,
                        "normalized": _normalize_tag(token_text),
                        "logit": float(score),
                    }
                )

            tag_prob_row = {tag: float(tag_probs[row_idx, i].item()) for i, tag in enumerate(TAGS)}
            tag_logit_row = {tag: float(tag_logits[row_idx, i].item()) for i, tag in enumerate(TAGS)}
            results.append(
                {
                    "idx": sample.idx,
                    "dialogue_id": sample.dialogue_id,
                    "turn": sample.turn,
                    "lang": sample.lang,
                    "lang_norm": _normalize_lang(sample.lang),
                    "label": sample.label,
                    "pred": pred_tag,
                    "raw_top1": row_topk[0]["token"] if row_topk else None,
                    "raw_top1_normalized": row_topk[0]["normalized"] if row_topk else None,
                    "raw_topk": row_topk,
                    "tag_probs": tag_prob_row,
                    "tag_logits": tag_logit_row,
                    "ok": pred_tag == sample.label,
                }
            )

    elapsed_s = max(1e-6, time.time() - t0)
    print(f"\nInference finished in {elapsed_s:.2f}s ({len(results) / elapsed_s:.2f} samples/s)")

    valid = [r for r in results if r.get("label") in TAGS and r.get("pred") in TAGS]
    invalid = len(results) - len(valid)
    all_summary = _summarize_tag_confusion(_compute_tag_confusion(valid))
    merged_all = _summarize_merged_tag_confusion(_compute_merged_tag_confusion(valid))

    first_valid = [r for r in valid if _is_first_turn(r.get("turn"))]
    multi_valid = [r for r in valid if not _is_first_turn(r.get("turn"))]

    _print_eval_block("HF backend | ALL", all_summary)
    _print_eval_block("Turn: FIRST (User's 1st msg)", _summarize_tag_confusion(_compute_tag_confusion(first_valid)))
    _print_eval_block("Turn: MULTI (User's >1 msg)", _summarize_tag_confusion(_compute_tag_confusion(multi_valid)))
    _print_eval_block_merged("HF backend | ALL | merge <UNADDRESSED> as <EOU>", merged_all)
    _print_eval_block_merged(
        "Turn: FIRST | merge <UNADDRESSED> as <EOU>",
        _summarize_merged_tag_confusion(_compute_merged_tag_confusion(first_valid)),
    )
    _print_eval_block_merged(
        "Turn: MULTI | merge <UNADDRESSED> as <EOU>",
        _summarize_merged_tag_confusion(_compute_merged_tag_confusion(multi_valid)),
    )

    for lang, group in sorted(
        ((lang, [r for r in valid if _normalize_lang(r.get("lang")) == lang]) for lang in {_normalize_lang(r.get("lang")) for r in valid}),
        key=lambda kv: (-len(kv[1]), kv[0]),
    ):
        _print_eval_block(f"Language: {lang}", _summarize_tag_confusion(_compute_tag_confusion(group)))
        _print_eval_block_merged(
            f"Language: {lang} | merge <UNADDRESSED> as <EOU>",
            _summarize_merged_tag_confusion(_compute_merged_tag_confusion(group)),
        )

    raw_top1_non_tag = sum(1 for r in results if r.get("raw_top1_normalized") not in TAGS)
    pred_counts: dict[str, int] = {}
    for row in results:
        pred = str(row.get("pred"))
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

    summary = {
        "dataset": args.dataset,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_invalid": invalid,
        "seconds": float(elapsed_s),
        "samples_per_second": float(len(results) / elapsed_s),
        "pred_counts": pred_counts,
        "raw_top1_non_tag_count": raw_top1_non_tag,
        "export_prompt_probe": export_probe,
        "tag_eval": {
            "valid": len(valid),
            "invalid": invalid,
            "accuracy": all_summary["accuracy"],
            "macro_f1": all_summary["macro_f1"],
            "confusion_matrix": all_summary["confusion"],
            "per_label": all_summary["per_label"],
            "kpi": all_summary["kpi"],
            "per_lang": _build_tag_eval_per_lang(results, valid),
        },
        "tag_eval_merge_unad_as_eou": {
            "valid": len(valid),
            "invalid": invalid,
            "accuracy": merged_all["accuracy"],
            "macro_f1": merged_all["macro_f1"],
            "confusion_matrix": merged_all["confusion"],
            "per_label": merged_all["per_label"],
            "kpi": merged_all["kpi"],
            "per_lang": _build_tag_eval_merge_unad_as_eou_per_lang(valid),
        },
        "predictions_jsonl": pred_path,
    }

    with open(pred_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Wrote: {pred_path}")
    print(f"[OK] Wrote: {summary_path}")
    if export_probe and export_probe["warning"]:
        print(f"[WARN] {export_probe['warning']}")


if __name__ == "__main__":
    main()
