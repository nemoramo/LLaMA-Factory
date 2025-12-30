#!/usr/bin/env python3
import argparse
import datetime as _dt
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional


TAGS = ("<EOU>", "<CONT_USER>", "<UNADDRESSED>")
TAG_RE = re.compile(r"<EOU>|<CONT_USER>|<UNADDRESSED>")


def _now_ts() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {e}") from e
    return items


def _normalize_tag(text: Optional[str]) -> str:
    if text is None:
        return "NULL"
    text = str(text).strip()
    m = TAG_RE.search(text)
    if m:
        return m.group(0)
    return text if text else "EMPTY"


def _post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


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

    assistant_indices = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    if not assistant_indices:
        raise ValueError("no assistant message found to use as label")

    label = obj.get("label")
    if not isinstance(label, str) or not label.strip():
        label = str(messages[assistant_indices[-1]].get("content", "")).strip()

    prompt_messages = messages[: assistant_indices[-1]]
    prompt_messages = [
        {"role": str(m.get("role")), "content": str(m.get("content", ""))} for m in prompt_messages
    ]

    return Sample(
        idx=idx,
        dialogue_id=obj.get("dialogue_id"),
        turn=obj.get("turn"),
        lang=obj.get("lang"),
        label=_normalize_tag(label),
        prompt_messages=prompt_messages,
    )


def _coerce_turn(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is int-like in Python; treat as invalid here.
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
    # Follow the previous infer_sglang convention: unknown turn treated as FIRST.
    return t is None or t <= 1


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


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
    n = len(TAGS)
    assert n == 3, "KPI definitions assume 3-way tag classification."

    row_sum = [sum(mat[i]) for i in range(n)]
    col_sum = [sum(mat[i][j] for i in range(n)) for j in range(n)]
    correct = sum(mat[i][i] for i in range(n))
    total = sum(row_sum)

    per_label: dict[str, dict[str, float]] = {}
    for i, tag in enumerate(TAGS):
        tp = mat[i][i]
        p = _safe_div(tp, col_sum[i])
        r = _safe_div(tp, row_sum[i])
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_label[tag] = {"precision": p, "recall": r, "f1": f1}

    # KPI definitions (matched to infer_sglang logs)
    # - FAR_unad: gold <UNADDRESSED> predicted as (<EOU> or <CONT_USER>)
    # - Interrupt: gold <CONT_USER> predicted as <EOU>
    # - Delay: gold <EOU> predicted as <CONT_USER>
    # - Missed: gold <EOU> predicted as <UNADDRESSED>
    eou_i = 0
    cont_i = 1
    unad_i = 2
    far_unad = _safe_div(row_sum[unad_i] - mat[unad_i][unad_i], row_sum[unad_i])
    interrupt = _safe_div(mat[cont_i][eou_i], row_sum[cont_i])
    delay = _safe_div(mat[eou_i][cont_i], row_sum[eou_i])
    missed = _safe_div(mat[eou_i][unad_i], row_sum[eou_i])

    return {
        "total": total,
        "correct": correct,
        "accuracy": _safe_div(correct, total),
        "confusion": {TAGS[i]: {TAGS[j]: mat[i][j] for j in range(n)} for i in range(n)},
        "per_label": per_label,
        "kpi": {
            "FAR_unad": far_unad,
            "Interrupt": interrupt,
            "Delay": delay,
            "Missed": missed,
        },
        "_row_sum": {TAGS[i]: row_sum[i] for i in range(n)},
    }


def _print_eval_block(title: str, tag_summary: dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print("Confusion (gold -> pred):")

    row_sum = tag_summary.get("_row_sum", {})
    conf = tag_summary["confusion"]
    for gold in TAGS:
        row = [conf[gold].get(pred, 0) for pred in TAGS]
        s = int(row_sum.get(gold, sum(row)))
        print(f"{gold:<14} {row} sum={s:>5d}")

    print(f"Accuracy: {tag_summary['accuracy']:.4f}")
    print("Per-label precision/recall/f1:")
    for tag in TAGS:
        m = tag_summary["per_label"][tag]
        print(f"  {tag:<14} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")

    k = tag_summary["kpi"]
    print(
        f"[KPI] FAR_unad={k['FAR_unad']:.4f} | Interrupt={k['Interrupt']:.4f} | Delay={k['Delay']:.4f} | Missed={k['Missed']:.4f}"
    )


def _infer_one(
    sample: Sample,
    base_url: str,
    model: str,
    timeout_s: float,
    max_retries: int,
    retry_sleep_s: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": sample.prompt_messages,
        "temperature": 0,
        "max_tokens": 1,
        "skip_special_tokens": False,
    }
    last_error: Optional[str] = None
    for attempt in range(max_retries + 1):
        t0 = time.time()
        try:
            resp = _post_json(url, payload, timeout_s=timeout_s)
            dt_ms = int((time.time() - t0) * 1000)
            raw_pred = (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            pred = _normalize_tag(raw_pred)
            return {
                "idx": sample.idx,
                "dialogue_id": sample.dialogue_id,
                "turn": sample.turn,
                "lang": sample.lang,
                "label": sample.label,
                "pred": pred,
                "raw_pred": raw_pred,
                "ok": pred == sample.label,
                "latency_ms": dt_ms,
                "error": None,
                "usage": resp.get("usage"),
                "response_id": resp.get("id"),
            }
        except Exception as e:
            dt_ms = int((time.time() - t0) * 1000)
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(retry_sleep_s)
                continue
            return {
                "idx": sample.idx,
                "dialogue_id": sample.dialogue_id,
                "turn": sample.turn,
                "lang": sample.lang,
                "label": sample.label,
                "pred": "ERROR",
                "raw_pred": None,
                "ok": False,
                "latency_ms": dt_ms,
                "error": last_error,
                "usage": None,
                "response_id": None,
            }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="/data2/mayufeng/llama_data/speech_endpointing/speech_endpointing_eval.jsonl",
        help="Eval JSONL in LLaMA-Factory OpenAI messages format.",
    )
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:30010",
        help="SGLang server base URL.",
    )
    ap.add_argument(
        "--model",
        default="gemma3-270m-endpointing",
        help="served_model_name in SGLang.",
    )
    ap.add_argument(
        "--out-dir",
        default="/data2/mayufeng/llama_output/speech_endpointing/sglang_eval",
        help="Directory to write outputs.",
    )
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-sleep", type=float, default=0.2)
    ap.add_argument(
        "--no-per-lang",
        dest="per_lang",
        action="store_false",
        default=True,
        help="Do not print per-language breakdown blocks.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    run_id = _now_ts()
    pred_path = os.path.join(args.out_dir, f"pred_{args.model}_{run_id}.jsonl")
    summary_path = os.path.join(args.out_dir, f"summary_{args.model}_{run_id}.json")

    raw = _read_jsonl(args.input)
    samples: list[Sample] = []
    skipped = 0
    for i, obj in enumerate(raw):
        try:
            s = _extract_sample(obj, idx=i)
            # Preserve original `turn` as-is in output, but use a normalized int for bucketing.
            s = Sample(
                idx=s.idx,
                dialogue_id=s.dialogue_id,
                turn=_coerce_turn(obj.get("turn")),
                lang=s.lang,
                label=s.label,
                prompt_messages=s.prompt_messages,
            )
            samples.append(s)
        except Exception:
            skipped += 1

    turn_counter = Counter()
    first_cnt = 0
    multi_cnt = 0
    for s in samples:
        k = "N/A" if s.turn is None else str(s.turn)
        turn_counter[k] += 1
        if _is_first_turn(s.turn):
            first_cnt += 1
        else:
            multi_cnt += 1
    print(f"[DEBUG] Distribution of 'turn' field values: {dict(turn_counter)}")
    print(f"[DEBUG] Classified as First Turn: {first_cnt}, Multi Turn: {multi_cnt}")

    print(f"Loaded {len(raw)} records, using {len(samples)}, skipped {skipped}.")
    print(f"Server: {args.base_url} model={args.model} workers={args.max_workers}")
    print(f"Writing: {pred_path}")

    results: list[Optional[dict[str, Any]]] = [None] * len(samples)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(
                _infer_one,
                s,
                args.base_url,
                args.model,
                args.timeout,
                args.max_retries,
                args.retry_sleep,
            ): i
            for i, s in enumerate(samples)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0 or done == len(samples):
                elapsed = time.time() - t0
                rps = done / max(elapsed, 1e-6)
                print(f"Progress: {done}/{len(samples)} ({rps:.1f} req/s)")

    # Tag-mode evaluation (match infer_sglang: only count predictions that land in one of 3 tags).
    valid = [r for r in results if r and r.get("label") in TAGS and r.get("pred") in TAGS]
    invalid = len(results) - len(valid)

    print(f"\n--- OpenAI backend | ALL | N={len(samples)}, R=1, mode=tags ---")
    print(f"Valid(tag)={len(valid)} Invalid/other={invalid}")
    all_mat = _compute_tag_confusion(valid)
    all_summary = _summarize_tag_confusion(all_mat)
    _print_eval_block("OpenAI backend | ALL", all_summary)

    first_valid = [r for r in valid if _is_first_turn(r.get("turn"))]
    multi_valid = [r for r in valid if not _is_first_turn(r.get("turn"))]
    _print_eval_block("Turn: FIRST (User's 1st msg)", _summarize_tag_confusion(_compute_tag_confusion(first_valid)))
    _print_eval_block("Turn: MULTI (User's >1 msg)", _summarize_tag_confusion(_compute_tag_confusion(multi_valid)))

    if args.per_lang:
        lang_groups: dict[str, list[dict[str, Any]]] = {}
        for r in valid:
            lang = r.get("lang") or "unk"
            lang_groups.setdefault(lang, []).append(r)
        for lang, group in sorted(lang_groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            _print_eval_block(f"Language: {lang}", _summarize_tag_confusion(_compute_tag_confusion(group)))

    # Latency / throughput (reported on all requests, regardless of validity)
    latency_ms = [r["latency_ms"] for r in results if r and isinstance(r.get("latency_ms"), int)]
    latency_ms.sort()
    latency_s = [x / 1000.0 for x in latency_ms]

    def pct_s(p: float) -> float:
        if not latency_s:
            return 0.0
        k = int(round((p / 100.0) * (len(latency_s) - 1)))
        return float(latency_s[max(0, min(len(latency_s) - 1, k))])

    mean_s = (sum(latency_s) / len(latency_s)) if latency_s else 0.0
    elapsed_s = time.time() - t0
    throughput = len(results) / max(elapsed_s, 1e-6)
    print(
        f"Latency (s): p50={pct_s(50):.4f} p90={pct_s(90):.4f} p95={pct_s(95):.4f} p99={pct_s(99):.4f} mean={mean_s:.4f}"
    )
    print(f"Wall throughput: {throughput:.2f} req/s")

    # Write predictions in sample order.
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            assert r is not None
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summaries
    total = len(results)
    correct_all = sum(1 for r in results if r and r.get("ok"))
    errors = sum(1 for r in results if r and r.get("pred") == "ERROR")

    # Full confusion matrix (includes non-tag outputs) for debugging.
    cm_all: dict[str, dict[str, int]] = {}
    for r in results:
        assert r is not None
        y = str(r.get("label"))
        yhat = str(r.get("pred"))
        cm_all.setdefault(y, {})
        cm_all[y][yhat] = cm_all[y].get(yhat, 0) + 1

    per_lang_all: dict[str, dict[str, int]] = {}
    for r in results:
        assert r is not None
        lang = r.get("lang") or "UNKNOWN_LANG"
        per_lang_all.setdefault(lang, {"total": 0, "correct": 0, "errors": 0})
        per_lang_all[lang]["total"] += 1
        per_lang_all[lang]["correct"] += 1 if r.get("ok") else 0
        per_lang_all[lang]["errors"] += 1 if r.get("pred") == "ERROR" else 0

    summary = {
        "run_id": run_id,
        "input": args.input,
        "base_url": args.base_url,
        "model": args.model,
        "total": total,
        "correct": correct_all,
        "accuracy": (correct_all / total) if total else None,
        "errors": errors,
        "latency_ms_p50": int(round(pct_s(50) * 1000)) if latency_s else None,
        "latency_ms_p90": int(round(pct_s(90) * 1000)) if latency_s else None,
        "latency_ms_p95": int(round(pct_s(95) * 1000)) if latency_s else None,
        "latency_ms_p99": int(round(pct_s(99) * 1000)) if latency_s else None,
        "latency_ms_max": latency_ms[-1] if latency_ms else None,
        "latency_s_p50": pct_s(50) if latency_s else None,
        "latency_s_p90": pct_s(90) if latency_s else None,
        "latency_s_p95": pct_s(95) if latency_s else None,
        "latency_s_p99": pct_s(99) if latency_s else None,
        "latency_s_mean": mean_s if latency_s else None,
        "wall_throughput_rps": throughput,
        "confusion_matrix": cm_all,
        "per_lang": per_lang_all,
        "tag_eval": {
            "valid": len(valid),
            "invalid": invalid,
            "accuracy": all_summary["accuracy"],
            "confusion_matrix": all_summary["confusion"],
            "per_label": all_summary["per_label"],
            "kpi": all_summary["kpi"],
        },
        "predictions_jsonl": pred_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done. accuracy_all={summary['accuracy']:.4f} accuracy_tags={all_summary['accuracy']:.4f} total={total}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
