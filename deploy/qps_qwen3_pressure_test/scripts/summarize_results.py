#!/usr/bin/env python3

import argparse
import glob
import json
import os
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="/home/sagemaker-user/qps_qwen3_pressure_test/results",
        help="Directory containing *.json benchmark results",
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

    valid_rows = [r for r in rows if r["p90_ms"] <= 60 and r["failed"] == 0]
    if valid_rows:
        best = max(valid_rows, key=lambda x: x["req_s"])
        print("\nBest (p90<=60ms):")
        print(
            f"scenario={best['scenario']}, req/s={best['req_s']:.2f}, "
            f"p90={best['p90_ms']:.2f}ms, goodput={best['goodput_s']:.2f}"
        )
    else:
        print("\nNo scenario satisfies p90<=60ms.")


if __name__ == "__main__":
    main()
