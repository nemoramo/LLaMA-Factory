#!/usr/bin/env python3

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List


def load_result(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def scenario_order(name: str) -> int:
    if name.startswith("warm"):
        return 10
    if name.startswith("burst"):
        return 20
    if name.startswith("cold"):
        return 30
    if name.startswith("sweep"):
        return 40
    return 99


def latency_key(metric: str) -> str:
    return f"{metric}_ms"


def main() -> None:
    default_results_dir = str(Path(__file__).resolve().parent.parent / "results")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=default_results_dir,
        help="Directory containing *.json benchmark results",
    )
    parser.add_argument(
        "--latency-metric",
        choices=["p50", "p90", "p99"],
        default="p90",
        help="Latency metric used to select the best scenario. Default: p90",
    )
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=60.0,
        help="Maximum allowed latency in milliseconds for the selected metric. Default: 60",
    )
    args = parser.parse_args()

    files: List[str] = sorted(glob.glob(os.path.join(args.results_dir, "*.json")))
    if not files:
        print(f"No json files found under: {args.results_dir}")
        return

    rows = []
    for fp in files:
        d = load_result(fp)
        rows.append(
            {
                "scenario": os.path.basename(fp),
                "req_s": d.get("request_throughput", 0.0),
                "goodput_s": d.get("request_goodput", 0.0),
                "p50_ms": d.get("p50_e2el_ms", 0.0),
                "p90_ms": d.get("p90_e2el_ms", 0.0),
                "p99_ms": d.get("p99_e2el_ms", 0.0),
                "failed": d.get("failed", 0),
                "duration_s": d.get("duration", 0.0),
            }
        )

    rows.sort(key=lambda x: (scenario_order(x["scenario"]), x["scenario"]))

    print("| scenario | req/s | goodput(req/s) | p50(ms) | p90(ms) | p99(ms) | failed | duration(s) |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['scenario']} | {r['req_s']:.2f} | {r['goodput_s']:.2f} | "
            f"{r['p50_ms']:.2f} | {r['p90_ms']:.2f} | {r['p99_ms']:.2f} | "
            f"{r['failed']} | {r['duration_s']:.2f} |"
        )

    selected_latency_key = latency_key(args.latency_metric)
    valid_rows = [
        r
        for r in rows
        if r[selected_latency_key] <= args.latency_threshold_ms and r["failed"] == 0
    ]
    if valid_rows:
        best = max(valid_rows, key=lambda x: x["req_s"])
        print(f"\nBest ({args.latency_metric}<={args.latency_threshold_ms:.1f}ms):")
        print(
            f"scenario={best['scenario']}, req/s={best['req_s']:.2f}, "
            f"{args.latency_metric}={best[selected_latency_key]:.2f}ms, goodput={best['goodput_s']:.2f}"
        )
    else:
        print(f"\nNo scenario satisfies {args.latency_metric}<={args.latency_threshold_ms:.1f}ms.")


if __name__ == "__main__":
    main()
