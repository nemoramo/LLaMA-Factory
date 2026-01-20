#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any


METRIC_KEYS = ("accuracy", "macro_f1", "interrupt", "delay")
METRIC_PREFER_HIGHER = {
    "accuracy": True,
    "macro_f1": True,
    "interrupt": False,  # lower is better
    "delay": False,  # lower is better
}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


@dataclass(frozen=True)
class Metrics:
    n: int
    accuracy: float
    macro_f1: float
    interrupt: float
    delay: float


def _load_metrics(path: str) -> dict[str, Metrics]:
    obj = _read_json(path)
    raw = obj.get("metrics")
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid metrics JSON: missing dict field `metrics`: {path}")

    out: dict[str, Metrics] = {}
    for lang, m in raw.items():
        if not isinstance(lang, str) or not lang:
            continue
        if not isinstance(m, dict):
            continue
        out[lang] = Metrics(
            n=_safe_int(m.get("n")),
            accuracy=_safe_float(m.get("accuracy")),
            macro_f1=_safe_float(m.get("macro_f1")),
            interrupt=_safe_float(m.get("interrupt")),
            delay=_safe_float(m.get("delay")),
        )
    if not out:
        raise ValueError(f"No per-language metrics found in: {path}")
    return out


def _derive_label(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".json"):
        base = base[: -len(".json")]
    return base


def _build_rows(
    metrics_a: dict[str, Metrics],
    metrics_b: dict[str, Metrics],
    *,
    min_n: int,
    sort_by: str,
) -> list[dict[str, Any]]:
    langs = sorted(set(metrics_a) & set(metrics_b))
    rows: list[dict[str, Any]] = []
    for lang in langs:
        a = metrics_a[lang]
        b = metrics_b[lang]
        n = min(a.n, b.n) if (a.n and b.n) else max(a.n, b.n)
        if n < min_n:
            continue

        row: dict[str, Any] = {"lang": lang, "n": n}
        for k in METRIC_KEYS:
            va = getattr(a, k)
            vb = getattr(b, k)
            delta = vb - va if (math.isfinite(va) and math.isfinite(vb)) else float("nan")
            prefer_higher = bool(METRIC_PREFER_HIGHER[k])
            score = delta if prefer_higher else (-delta)
            row[f"{k}_a"] = va
            row[f"{k}_b"] = vb
            row[f"delta_{k}"] = delta
            row[f"improve_{k}"] = score
        rows.append(row)

    if sort_by not in METRIC_KEYS:
        raise ValueError(f"Invalid --sort-by={sort_by}; choose from: {', '.join(METRIC_KEYS)}")

    def sort_key(r: dict[str, Any]) -> tuple[float, int, str]:
        s = r.get(f"improve_{sort_by}")
        s = float(s) if isinstance(s, (int, float)) and math.isfinite(float(s)) else float("-inf")
        return (s, int(r.get("n", 0)), str(r.get("lang", "")))

    rows.sort(key=sort_key, reverse=True)
    return rows


def _plot_delta(
    rows: list[dict[str, Any]],
    *,
    label_a: str,
    label_b: str,
    title: str,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import numpy as np  # type: ignore
    from matplotlib.patches import Patch  # type: ignore

    if not rows:
        raise ValueError("No rows to plot.")

    y_labels = [f"{r['lang']} (n={r['n']})" for r in rows]
    y = np.arange(len(rows))

    fig_h = max(4.2, 0.55 * len(rows) + 2.0)
    fig_w = 14.0
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), sharey=True)
    axes = axes.flatten()

    panels = [
        ("accuracy", "Accuracy (higher better)"),
        ("macro_f1", "Macro-F1 (higher better)"),
        ("interrupt", "Interrupt (lower better)"),
        ("delay", "Delay (lower better)"),
    ]

    green = "#2ca02c"
    red = "#d62728"

    for ax, (k, panel_title) in zip(axes, panels, strict=True):
        deltas = [float(r.get(f"delta_{k}", float("nan"))) for r in rows]
        prefer_higher = bool(METRIC_PREFER_HIGHER[k])
        colors = []
        for d in deltas:
            if not math.isfinite(d):
                colors.append("#7f7f7f")
                continue
            better = (d >= 0.0) if prefer_higher else (d <= 0.0)
            colors.append(green if better else red)

        ax.barh(y, deltas, color=colors)
        ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(y_labels)
        ax.invert_yaxis()
        ax.set_title(f"{panel_title}\nΔ = {label_b} - {label_a}")
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        finite = [abs(d) for d in deltas if math.isfinite(d)]
        max_abs = max(finite) if finite else 1.0
        if max_abs == 0.0:
            max_abs = 1.0
        ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)

        for yi, d in enumerate(deltas):
            if not math.isfinite(d):
                continue
            ha = "left" if d >= 0 else "right"
            dx = 0.01 if d >= 0 else -0.01
            ax.text(d + dx, yi, f"{d:+.4f}", va="center", ha=ha, fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.legend(
        handles=[
            Patch(facecolor=green, edgecolor="none", label=f"{label_b} better"),
            Patch(facecolor=red, edgecolor="none", label=f"{label_a} better"),
            Patch(facecolor="#7f7f7f", edgecolor="none", label="missing/invalid"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-a", required=True, help="Per-language metrics JSON (as produced by plot_kpi_by_language.py).")
    ap.add_argument("--metrics-b", required=True, help="Per-language metrics JSON (as produced by plot_kpi_by_language.py).")
    ap.add_argument("--label-a", default=None, help="Label for model A (default: derived from filename).")
    ap.add_argument("--label-b", default=None, help="Label for model B (default: derived from filename).")
    ap.add_argument("--min-n", type=int, default=1, help="Filter languages with fewer valid samples.")
    ap.add_argument("--sort-by", default="macro_f1", choices=list(METRIC_KEYS), help="Sort languages by this metric's improvement.")
    ap.add_argument("--title", default=None, help="Plot title (default: auto).")
    ap.add_argument("--out", required=True, help="Output image path (.png).")
    ap.add_argument("--out-delta-json", default=None, help="Optional path to write per-language deltas as JSON.")
    args = ap.parse_args()

    metrics_a = _load_metrics(args.metrics_a)
    metrics_b = _load_metrics(args.metrics_b)

    label_a = args.label_a or _derive_label(args.metrics_a)
    label_b = args.label_b or _derive_label(args.metrics_b)

    rows = _build_rows(metrics_a, metrics_b, min_n=args.min_n, sort_by=args.sort_by)

    title = args.title or f"Per-language KPI delta (merged <UNADDRESSED> as <EOU>)\nSorted by {args.sort_by} improvement"
    _plot_delta(rows, label_a=label_a, label_b=label_b, title=title, out_path=args.out)
    print(f"[OK] Wrote plot: {args.out}")

    if args.out_delta_json:
        out_obj = {
            "metrics_a": args.metrics_a,
            "metrics_b": args.metrics_b,
            "label_a": label_a,
            "label_b": label_b,
            "sort_by": args.sort_by,
            "min_n": args.min_n,
            "rows": rows,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out_delta_json)) or ".", exist_ok=True)
        with open(args.out_delta_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print(f"[OK] Wrote delta JSON: {args.out_delta_json}")


if __name__ == "__main__":
    main()

