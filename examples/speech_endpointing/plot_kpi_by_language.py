#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any


TAGS = ("<EOU>", "<CONT_USER>", "<UNADDRESSED>")
MERGED_TAGS = ("<EOU>", "<CONT_USER>")  # merge <UNADDRESSED> into <EOU>


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _normalize_lang(lang: Any) -> str:
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


def _load_lang_map(dataset_jsonl: str) -> list[str]:
    lang_map: list[str] = []
    with open(dataset_jsonl, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {dataset_jsonl}: {e}") from e
            lang_map.append(_normalize_lang(obj.get("lang")))
    return lang_map


def _extract_index(obj: dict[str, Any]) -> int:
    # Support both infer_sglang format (`i`) and eval_sglang_endpointing format (`idx`).
    if "idx" in obj:
        return int(obj["idx"])
    if "i" in obj:
        return int(obj["i"])
    raise KeyError("Prediction record missing index key: expected `idx` or `i`.")


def _extract_gold_pred(obj: dict[str, Any]) -> tuple[str, str]:
    if "label" in obj:
        gold = str(obj.get("label", "")).strip()
    else:
        gold = str(obj.get("gold", "")).strip()
    pred = str(obj.get("pred", "")).strip()
    return gold, pred


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    interrupt: float
    delay: float


def _compute_metrics_merge_unad_as_eou(pairs: list[tuple[str, str]]) -> Metrics:
    # confusion: rows=gold [EOU, CONT_USER], cols=pred [EOU, CONT_USER]
    t2i = {t: i for i, t in enumerate(MERGED_TAGS)}
    mat = [[0, 0], [0, 0]]
    for gold, pred in pairs:
        g = _merge_unaddressed_as_eou(gold)
        p = _merge_unaddressed_as_eou(pred)
        if g not in t2i or p not in t2i:
            continue
        mat[t2i[g]][t2i[p]] += 1

    row_sum = [sum(mat[i]) for i in range(2)]
    col_sum = [mat[0][j] + mat[1][j] for j in range(2)]
    correct = mat[0][0] + mat[1][1]
    total = row_sum[0] + row_sum[1]

    # Per-class F1
    f1s: list[float] = []
    for i in range(2):
        tp = mat[i][i]
        p = _safe_div(tp, col_sum[i])
        r = _safe_div(tp, row_sum[i])
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / 2.0

    eou_i = 0
    cont_i = 1
    interrupt = _safe_div(mat[cont_i][eou_i], row_sum[cont_i])
    delay = _safe_div(mat[eou_i][cont_i], row_sum[eou_i])

    return Metrics(
        n=int(total),
        accuracy=_safe_div(correct, total),
        macro_f1=float(macro_f1),
        interrupt=float(interrupt),
        delay=float(delay),
    )


def _ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)


def _plot(metrics_by_lang: dict[str, Metrics], *, title: str, out_path: str) -> None:
    # Lazy imports: keep script importable without matplotlib, but plotting requires it.
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore

    rows = [
        (lang, m)
        for lang, m in metrics_by_lang.items()
        if lang and not math.isnan(m.macro_f1)
    ]
    rows.sort(key=lambda kv: (-kv[1].macro_f1, -kv[1].accuracy, kv[0]))

    langs = [lang for lang, _ in rows]
    acc = [m.accuracy for _, m in rows]
    macro_f1 = [m.macro_f1 for _, m in rows]
    interrupt = [m.interrupt for _, m in rows]
    delay = [m.delay for _, m in rows]

    y = np.arange(len(langs))
    h = 0.18
    offsets = [-1.5 * h, -0.5 * h, 0.5 * h, 1.5 * h]

    fig_h = max(4.0, 0.55 * len(langs) + 1.8)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    bars = []
    bars.append(
        ax.barh(
            y + offsets[0],
            acc,
            height=h,
            color="#1f77b4",
            label="Accuracy (higher better)",
        )
    )
    bars.append(
        ax.barh(
            y + offsets[1],
            macro_f1,
            height=h,
            color="#ff7f0e",
            label="Macro-F1 (higher better)",
        )
    )
    bars.append(
        ax.barh(
            y + offsets[2],
            interrupt,
            height=h,
            color="#2ca02c",
            label="Interrupt (lower better)",
        )
    )
    bars.append(
        ax.barh(
            y + offsets[3],
            delay,
            height=h,
            color="#d62728",
            label="Delay (lower better)",
        )
    )

    # Value labels
    for bset, values in zip(bars, [acc, macro_f1, interrupt, delay], strict=True):
        for rect, v in zip(bset, values, strict=True):
            ax.text(
                rect.get_width() + 0.01,
                rect.get_y() + rect.get_height() / 2,
                f"{v:.4f}",
                va="center",
                ha="left",
                fontsize=9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels([f"{i+1}. {lang}" for i, lang in enumerate(langs)])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Metric value")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    _ensure_parent_dir(out_path)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="TorchTune testset converted to OpenAI messages JSONL with `lang` field.",
    )
    ap.add_argument(
        "--pred",
        required=True,
        help="Prediction JSONL (infer_sglang*.pred.jsonl OR eval_sglang_endpointing pred_*.jsonl).",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="Plot title (default: derived from pred filename).",
    )
    ap.add_argument("--out", required=True, help="Output image path (.png).")
    ap.add_argument(
        "--out-metrics-json",
        default=None,
        help="Optional path to write per-language metrics as JSON.",
    )
    args = ap.parse_args()

    lang_map = _load_lang_map(args.dataset)
    lang_counts = Counter(lang_map)
    print("[INFO] Languages in dataset (after normalization):")
    for lang, c in sorted(lang_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {lang}: {c}")

    preds = _read_jsonl(args.pred)
    by_lang_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    missing_idx = 0
    for obj in preds:
        idx = _extract_index(obj)
        if idx < 0 or idx >= len(lang_map):
            missing_idx += 1
            lang = "UNKNOWN_LANG"
        else:
            lang = lang_map[idx]

        gold, pred = _extract_gold_pred(obj)
        if gold not in TAGS or pred not in TAGS:
            continue
        by_lang_pairs[lang].append((gold, pred))

    if missing_idx:
        print(f"[WARN] {missing_idx} prediction rows have idx out of dataset range.")

    metrics_by_lang: dict[str, Metrics] = {}
    for lang, pairs in by_lang_pairs.items():
        metrics_by_lang[lang] = _compute_metrics_merge_unad_as_eou(pairs)

    title = args.title or f"Combined KPI by Language (sorted by Macro-F1)\n{os.path.basename(args.pred)}"
    _plot(metrics_by_lang, title=title, out_path=args.out)
    print(f"[OK] Wrote plot: {args.out}")

    if args.out_metrics_json:
        out_obj = {
            "dataset": args.dataset,
            "pred": args.pred,
            "metrics": {
                lang: {
                    "n": m.n,
                    "accuracy": m.accuracy,
                    "macro_f1": m.macro_f1,
                    "interrupt": m.interrupt,
                    "delay": m.delay,
                }
                for lang, m in sorted(metrics_by_lang.items(), key=lambda kv: (-kv[1].macro_f1, kv[0]))
            },
        }
        _ensure_parent_dir(args.out_metrics_json)
        with open(args.out_metrics_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote metrics JSON: {args.out_metrics_json}")


if __name__ == "__main__":
    main()

