#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import sys

_EXT_DIR = Path(__file__).resolve().parent
if str(_EXT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXT_DIR))


params = {
    "display_name": "Speech Endpointing",
    "is_tab": True,
}

_ROOT_DIR = _EXT_DIR.parents[2]
_DATA_DIR = _ROOT_DIR / "user_data" / "endpointing_tool"
_EXPORT_DIR = _DATA_DIR / "exports"
_DATA_DIR.mkdir(parents=True, exist_ok=True)
_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
_CONFIG_PATH = _DATA_DIR / "config.json"

_DEFAULTS = {
    "grpc_target": "127.0.0.1:50051",
    "timeout_s": 15,
    "lang": "en-US",
    "treat_unaddressed_as_eou": True,
    "eou_threshold": 0.6,
    "concurrency": 8,
}

if _CONFIG_PATH.exists():
    try:
        with _CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            _DEFAULTS.update(cfg)
    except Exception:
        pass


def _lazy_import_grpc():
    try:
        import grpc  # type: ignore
        import endpointing_pb2  # type: ignore
        import endpointing_pb2_grpc  # type: ignore
        return grpc, endpointing_pb2, endpointing_pb2_grpc, None
    except Exception as e:
        return None, None, None, str(e)


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _normalize_role(role: Any) -> str:
    if role is None:
        return ""
    r = str(role).strip().lower()
    if r in ("user", "u", "human"):
        return "user"
    if r in ("assistant", "a", "bot", "assistant"):
        return "assistant"
    return r


def _rows_to_history(rows: Any) -> List[Dict[str, str]]:
    if not rows:
        return []
    history: List[Dict[str, str]] = []
    for row in rows:
        if not row or len(row) < 2:
            continue
        role = _normalize_role(row[0])
        text = "" if row[1] is None else str(row[1])
        if role in ("user", "assistant") and text.strip():
            history.append({"role": role, "text": text})
    return history


def _history_to_rows(history: Any) -> List[List[str]]:
    rows: List[List[str]] = []
    if not history:
        return rows
    for h in history:
        role = _normalize_role(h.get("role")) if isinstance(h, dict) else ""
        text = "" if not isinstance(h, dict) else str(h.get("text", ""))
        rows.append([role, text])
    return rows


def _history_to_text(history: Any) -> str:
    if not history:
        return ""
    lines: List[str] = []
    for h in history:
        if not isinstance(h, dict):
            continue
        role = _normalize_role(h.get("role"))
        text = str(h.get("text", ""))
        if not text:
            continue
        if role == "user":
            prefix = "User"
        elif role == "assistant":
            prefix = "Assistant"
        else:
            prefix = role or "Role"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


def _build_request(
    endpointing_pb2,
    sample: Dict[str, Any],
    default_lang: str,
    default_treat_unaddressed: bool,
    default_eou_threshold: float,
):
    request_id = sample.get("request_id") or str(uuid.uuid4())
    session_id = sample.get("session_id", "") or ""
    lang = sample.get("lang") or default_lang

    asr_text = ""
    asr = sample.get("asr")
    if isinstance(asr, dict):
        asr_text = str(asr.get("text", ""))
    elif isinstance(sample.get("asr_text"), str):
        asr_text = sample.get("asr_text", "")

    history = sample.get("history")
    if not isinstance(history, list):
        history = []

    options = sample.get("options") or {}
    treat = options.get("treat_unaddressed_as_eou", default_treat_unaddressed)
    threshold = options.get("eou_threshold", default_eou_threshold)

    opt = endpointing_pb2.Options(treat_unaddressed_as_eou=bool(treat))
    if threshold is not None:
        try:
            opt.eou_threshold = float(threshold)
        except Exception:
            pass

    req = endpointing_pb2.EndpointingRequest(
        request_id=request_id,
        session_id=session_id,
        lang=lang,
        asr=endpointing_pb2.Asr(text=asr_text),
        options=opt,
    )

    for t in history:
        if not isinstance(t, dict):
            continue
        role = _normalize_role(t.get("role"))
        text = str(t.get("text", ""))
        if not text.strip():
            continue
        if role == "user":
            role_enum = endpointing_pb2.USER
        elif role == "assistant":
            role_enum = endpointing_pb2.ASSISTANT
        else:
            continue
        req.history.append(endpointing_pb2.HistoryTurn(role=role_enum, text=text))

    return req


def _predict_one(
    target: str,
    timeout_s: float,
    sample: Dict[str, Any],
    default_lang: str,
    default_treat_unaddressed: bool,
    default_eou_threshold: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    grpc, endpointing_pb2, endpointing_pb2_grpc, err = _lazy_import_grpc()
    if err:
        return None, f"缺少依赖: {err}"

    channel = grpc.insecure_channel(target)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        stub = endpointing_pb2_grpc.EndpointingServiceStub(channel)
        req = _build_request(endpointing_pb2, sample, default_lang, default_treat_unaddressed, default_eou_threshold)
        resp = stub.Predict(req, timeout=timeout_s)
        return {
            "label": resp.label,
            "confidence": float(resp.confidence),
            "p_eou": float(getattr(resp, "p_eou", 0.0)),
            "p_cont_user": float(getattr(resp, "p_cont_user", 0.0)),
            "p_unaddressed": float(getattr(resp, "p_unaddressed", 0.0)),
            "latency_ms": int(resp.latency_ms),
            "model": resp.model,
            "request_id": resp.request_id,
        }, None
    except Exception as e:
        return None, str(e)
    finally:
        try:
            channel.close()
        except Exception:
            pass


def _apply_pred(sample: Dict[str, Any], pred: Optional[Dict[str, Any]], err: Optional[str]) -> Dict[str, Any]:
    if pred:
        sample["pred_label"] = pred.get("label")
        sample["confidence"] = pred.get("confidence")
        sample["p_eou"] = pred.get("p_eou")
        sample["p_cont_user"] = pred.get("p_cont_user")
        sample["p_unaddressed"] = pred.get("p_unaddressed")
        sample["latency_ms"] = pred.get("latency_ms")
        sample["model"] = pred.get("model")
        sample["error"] = ""
    else:
        sample["error"] = err or "unknown error"
    return sample


def _sample_from_form(
    request_id: str,
    session_id: str,
    lang: str,
    asr_text: str,
    history_rows: List[List[str]],
    treat_unaddressed_as_eou: bool,
    eou_threshold: float,
    human_label: str,
    note: str,
) -> Dict[str, Any]:
    sample: Dict[str, Any] = {
        "request_id": request_id or str(uuid.uuid4()),
        "session_id": session_id or "",
        "lang": lang or "",
        "asr": {"text": asr_text or ""},
        "history": _rows_to_history(history_rows),
        "options": {
            "treat_unaddressed_as_eou": bool(treat_unaddressed_as_eou),
            "eou_threshold": float(eou_threshold),
        },
    }
    if human_label:
        sample["human_label"] = human_label
    if note:
        sample["note"] = note
    return sample


def _sample_to_form(sample: Dict[str, Any]) -> Tuple[str, str, str, str, List[List[str]], bool, float, str, str]:
    request_id = sample.get("request_id", "")
    session_id = sample.get("session_id", "")
    lang = sample.get("lang", "")
    asr_text = ""
    asr = sample.get("asr")
    if isinstance(asr, dict):
        asr_text = str(asr.get("text", ""))
    history_rows = _history_to_rows(sample.get("history") or [])
    options = sample.get("options") or {}
    treat = bool(options.get("treat_unaddressed_as_eou", False))
    threshold = float(options.get("eou_threshold", 0.6))
    human_label = sample.get("human_label", "")
    note = sample.get("note", "")
    return request_id, session_id, lang, asr_text, history_rows, treat, threshold, human_label, note


def _sample_to_pred(sample: Dict[str, Any]) -> Tuple[str, float, float, float, float, int, str]:
    return (
        sample.get("pred_label", ""),
        float(sample.get("confidence", 0.0) or 0.0),
        float(sample.get("p_eou", 0.0) or 0.0),
        float(sample.get("p_cont_user", 0.0) or 0.0),
        float(sample.get("p_unaddressed", 0.0) or 0.0),
        int(sample.get("latency_ms", 0) or 0),
        sample.get("error", "") or "",
    )


def _format_idx(idx: int, total: int) -> str:
    if total <= 0:
        return "0 / 0"
    return f"{idx + 1} / {total}"


def _save_jsonl(dataset: List[Dict[str, Any]], path: Path) -> str:
    with path.open("w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return str(path)


def _export_excel(dataset: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    if not dataset:
        return None, "没有可导出的数据"
    try:
        import pandas as pd
    except Exception as e:
        return None, f"缺少依赖 pandas: {e}"

    rows: List[Dict[str, Any]] = []
    for s in dataset:
        history = s.get("history") or []
        row = {
            "request_id": s.get("request_id", ""),
            "session_id": s.get("session_id", ""),
            "lang": s.get("lang", ""),
            "asr_text": (s.get("asr") or {}).get("text", ""),
            "history_text": _history_to_text(history),
            "history_json": json.dumps(history, ensure_ascii=False),
            "treat_unaddressed_as_eou": (s.get("options") or {}).get("treat_unaddressed_as_eou", ""),
            "eou_threshold": (s.get("options") or {}).get("eou_threshold", ""),
            "pred_label": s.get("pred_label", ""),
            "confidence": s.get("confidence", ""),
            "p_eou": s.get("p_eou", ""),
            "p_cont_user": s.get("p_cont_user", ""),
            "p_unaddressed": s.get("p_unaddressed", ""),
            "latency_ms": s.get("latency_ms", ""),
            "model": s.get("model", ""),
            "error": s.get("error", ""),
            "human_label": s.get("human_label", ""),
            "note": s.get("note", ""),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = _EXPORT_DIR / f"endpointing_{_now_tag()}.xlsx"
    df.to_excel(out_path, index=False)
    return str(out_path), f"已导出到 {out_path}"


def _load_jsonl(file_path: str) -> Tuple[List[Dict[str, Any]], str]:
    if not file_path:
        return [], "请选择 JSONL 文件"
    dataset: List[Dict[str, Any]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                dataset.append(obj)
        msg = f"已加载 {len(dataset)} 条"
        return dataset, msg
    except Exception as e:
        return [], f"加载失败: {e}"


def _predict_single_action(
    dataset: List[Dict[str, Any]],
    idx: int,
    target: str,
    timeout_s: float,
    request_id: str,
    session_id: str,
    lang: str,
    asr_text: str,
    history_rows: List[List[str]],
    treat_unaddressed_as_eou: bool,
    eou_threshold: float,
    human_label: str,
    note: str,
):
    sample = _sample_from_form(
        request_id,
        session_id,
        lang,
        asr_text,
        history_rows,
        treat_unaddressed_as_eou,
        eou_threshold,
        human_label,
        note,
    )

    pred, err = _predict_one(target, timeout_s, sample, lang, treat_unaddressed_as_eou, eou_threshold)
    _apply_pred(sample, pred, err)

    if dataset:
        if idx < 0 or idx >= len(dataset):
            idx = 0
        dataset[idx] = sample

    pred_label = sample.get("pred_label", "")
    confidence = sample.get("confidence", 0.0)
    p_eou = sample.get("p_eou", 0.0)
    p_cont = sample.get("p_cont_user", 0.0)
    p_un = sample.get("p_unaddressed", 0.0)
    latency_ms = sample.get("latency_ms", 0)
    error = sample.get("error", "")

    return (
        dataset,
        sample.get("request_id", ""),
        pred_label,
        confidence,
        p_eou,
        p_cont,
        p_un,
        latency_ms,
        error,
    )


def _save_label_action(
    dataset: List[Dict[str, Any]],
    idx: int,
    request_id: str,
    session_id: str,
    lang: str,
    asr_text: str,
    history_rows: List[List[str]],
    treat_unaddressed_as_eou: bool,
    eou_threshold: float,
    human_label: str,
    note: str,
):
    if not dataset:
        return dataset, "当前没有批量数据，已忽略"

    if idx < 0 or idx >= len(dataset):
        return dataset, "索引超出范围"

    sample = _sample_from_form(
        request_id,
        session_id,
        lang,
        asr_text,
        history_rows,
        treat_unaddressed_as_eou,
        eou_threshold,
        human_label,
        note,
    )

    # 保留已有预测字段
    for k in ["pred_label", "confidence", "p_eou", "p_cont_user", "p_unaddressed", "latency_ms", "model", "error"]:
        if k in dataset[idx]:
            sample[k] = dataset[idx][k]

    dataset[idx] = sample
    autosave_path = _DATA_DIR / "autosave.jsonl"
    _save_jsonl(dataset, autosave_path)
    return dataset, f"已保存（自动备份：{autosave_path}）"


def _goto_idx(dataset: List[Dict[str, Any]], idx: int) -> Tuple[int, str, str, str, str, str, List[List[str]], bool, float, str, str, str, float, float, float, float, int, str]:
    total = len(dataset)
    if total == 0:
        return 0, "0 / 0", "", "", "", "", [], False, 0.6, "", "", "", 0.0, 0.0, 0.0, 0.0, 0, ""
    if idx < 0:
        idx = 0
    if idx >= total:
        idx = total - 1
    sample = dataset[idx]
    req_id, session_id, lang, asr_text, history_rows, treat, threshold, human_label, note = _sample_to_form(sample)
    pred_label, confidence, p_eou, p_cont, p_un, latency_ms, error = _sample_to_pred(sample)
    return (
        idx,
        _format_idx(idx, total),
        req_id,
        session_id,
        lang,
        asr_text,
        history_rows,
        treat,
        threshold,
        human_label,
        note,
        pred_label,
        confidence,
        p_eou,
        p_cont,
        p_un,
        latency_ms,
        error,
    )


def _load_jsonl_action(file_path: str):
    dataset, msg = _load_jsonl(file_path)
    idx = 0
    if dataset:
        req_id, session_id, lang, asr_text, history_rows, treat, threshold, human_label, note = _sample_to_form(dataset[0])
        pred_label, confidence, p_eou, p_cont, p_un, latency_ms, error = _sample_to_pred(dataset[0])
        return (
            dataset,
            idx,
            _format_idx(idx, len(dataset)),
            req_id,
            session_id,
            lang,
            asr_text,
            history_rows,
            treat,
            threshold,
            human_label,
            note,
            pred_label,
            confidence,
            p_eou,
            p_cont,
            p_un,
            latency_ms,
            error,
            msg,
        )
    return dataset, idx, "0 / 0", "", "", "", "", [], False, 0.6, "", "", "", 0.0, 0.0, 0.0, 0.0, 0, "", msg


def _export_jsonl_action(dataset: List[Dict[str, Any]]):
    if not dataset:
        return None, "没有可导出的数据"
    out_path = _EXPORT_DIR / f"endpointing_{_now_tag()}.jsonl"
    _save_jsonl(dataset, out_path)
    return str(out_path), f"已导出到 {out_path}"


def _export_excel_action(dataset: List[Dict[str, Any]]):
    return _export_excel(dataset)


def _batch_predict_action(
    dataset: List[Dict[str, Any]],
    target: str,
    timeout_s: float,
    default_lang: str,
    default_treat: bool,
    default_threshold: float,
    concurrency: int,
    progress=gr.Progress(),
):
    if not dataset:
        return dataset, "没有可评测的数据"

    grpc, endpointing_pb2, endpointing_pb2_grpc, err = _lazy_import_grpc()
    if err:
        return dataset, f"缺少依赖: {err}"

    concurrency = max(1, int(concurrency))
    total = len(dataset)
    progress(0, desc="批量评测中...")

    channel = grpc.insecure_channel(target)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
    except Exception as e:
        return dataset, f"连接失败: {e}"

    stub = endpointing_pb2_grpc.EndpointingServiceStub(channel)

    def _task(i: int) -> Tuple[int, Optional[Dict[str, Any]], Optional[str]]:
        sample = dataset[i]
        req = _build_request(endpointing_pb2, sample, default_lang, default_treat, default_threshold)
        try:
            resp = stub.Predict(req, timeout=timeout_s)
            pred = {
                "label": resp.label,
                "confidence": float(resp.confidence),
                "p_eou": float(getattr(resp, "p_eou", 0.0)),
                "p_cont_user": float(getattr(resp, "p_cont_user", 0.0)),
                "p_unaddressed": float(getattr(resp, "p_unaddressed", 0.0)),
                "latency_ms": int(resp.latency_ms),
                "model": resp.model,
                "request_id": resp.request_id,
            }
            return i, pred, None
        except Exception as e:
            return i, None, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_task, i) for i in range(total)]
        for fut in as_completed(futures):
            i, pred, err = fut.result()
            dataset[i] = _apply_pred(dataset[i], pred, err)
            done += 1
            progress(done / total, desc=f"已完成 {done}/{total}")

    autosave_path = _DATA_DIR / "autosave.jsonl"
    _save_jsonl(dataset, autosave_path)
    return dataset, f"批量完成，已自动备份到 {autosave_path}"


def _test_connection(target: str, timeout_s: float) -> str:
    grpc, _, _, err = _lazy_import_grpc()
    if err:
        return f"缺少依赖: {err}"

    channel = grpc.insecure_channel(target)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        return "连接成功"
    except Exception as e:
        return f"连接失败: {e}"
    finally:
        try:
            channel.close()
        except Exception:
            pass


def ui():
    with gr.Tab("Speech Endpointing", elem_id="endpointing-tab"):
        gr.Markdown("""
        **Speech Endpointing 评测/标注工具**（gRPC）
        - 连接后可进行单条评测或批量评测
        - 支持填写人工标签并导出 Excel
        """)

        dataset_state = gr.State([])
        idx_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=3):
                target = gr.Textbox(label="gRPC 地址", value=str(_DEFAULTS.get("grpc_target", "127.0.0.1:50051")))
            with gr.Column(scale=1):
                timeout_s = gr.Number(label="超时(s)", value=float(_DEFAULTS.get("timeout_s", 15)))
            with gr.Column(scale=1):
                test_btn = gr.Button("测试连接")

        conn_status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=2):
                lang = gr.Textbox(label="语言", value=str(_DEFAULTS.get("lang", "en-US")))
            with gr.Column(scale=2):
                session_id = gr.Textbox(label="Session ID", value="")
            with gr.Column(scale=2):
                request_id = gr.Textbox(label="Request ID", value="", interactive=True)

        with gr.Row():
            with gr.Column(scale=6):
                history_df = gr.Dataframe(
                    headers=["role", "text"],
                    datatype=["str", "str"],
                    row_count=(3, "dynamic"),
                    col_count=(2, "fixed"),
                    label="历史对话（role: user/assistant）",
                    value=[],
                )
            with gr.Column(scale=4):
                asr_text = gr.Textbox(label="ASR 文本", value="", lines=6)

        with gr.Row():
            with gr.Column(scale=2):
                treat_unaddressed_as_eou = gr.Checkbox(
                    label="Unaddressed 归并到 EOU",
                    value=bool(_DEFAULTS.get("treat_unaddressed_as_eou", True)),
                )
            with gr.Column(scale=2):
                eou_threshold = gr.Slider(
                    label="EOU 阈值",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=float(_DEFAULTS.get("eou_threshold", 0.6)),
                )
            with gr.Column(scale=2):
                human_label = gr.Radio(
                    label="人工标签",
                    choices=["", "<EOU>", "<CONT_USER>", "<UNADDRESSED>"],
                    value="",
                )
            with gr.Column(scale=4):
                note = gr.Textbox(label="备注", value="", lines=2)

        with gr.Row():
            predict_btn = gr.Button("单条预测")
            save_label_btn = gr.Button("保存标注")

        with gr.Row():
            with gr.Column(scale=2):
                pred_label = gr.Textbox(label="预测标签", value="", interactive=False)
            with gr.Column(scale=2):
                confidence = gr.Number(label="confidence", value=0.0, interactive=False)
            with gr.Column(scale=2):
                latency_ms = gr.Number(label="latency_ms", value=0, interactive=False)

        with gr.Row():
            with gr.Column(scale=2):
                p_eou = gr.Number(label="P(<EOU>)", value=0.0, interactive=False)
            with gr.Column(scale=2):
                p_cont = gr.Number(label="P(<CONT_USER>)", value=0.0, interactive=False)
            with gr.Column(scale=2):
                p_un = gr.Number(label="P(<UNADDRESSED>)", value=0.0, interactive=False)

        error_msg = gr.Markdown("")

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                jsonl_file = gr.File(label="导入 JSONL", file_types=[".jsonl"], type="filepath")
            with gr.Column(scale=1):
                load_btn = gr.Button("加载 JSONL")
            with gr.Column(scale=2):
                idx_label = gr.Textbox(label="当前样本", value="0 / 0", interactive=False)
            with gr.Column(scale=1):
                prev_btn = gr.Button("上一条")
            with gr.Column(scale=1):
                next_btn = gr.Button("下一条")

        batch_status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=2):
                concurrency = gr.Slider(
                    label="批量并发",
                    minimum=1,
                    maximum=64,
                    step=1,
                    value=int(_DEFAULTS.get("concurrency", 8)),
                )
            with gr.Column(scale=2):
                batch_btn = gr.Button("批量评测")
            with gr.Column(scale=2):
                export_jsonl_btn = gr.Button("导出 JSONL")
            with gr.Column(scale=2):
                export_excel_btn = gr.Button("导出 Excel")

        export_jsonl_file = gr.File(label="JSONL 导出结果")
        export_excel_file = gr.File(label="Excel 导出结果")

        # Actions
        test_btn.click(_test_connection, [target, timeout_s], conn_status)

        predict_btn.click(
            _predict_single_action,
            [
                dataset_state,
                idx_state,
                target,
                timeout_s,
                request_id,
                session_id,
                lang,
                asr_text,
                history_df,
                treat_unaddressed_as_eou,
                eou_threshold,
                human_label,
                note,
            ],
            [
                dataset_state,
                request_id,
                pred_label,
                confidence,
                p_eou,
                p_cont,
                p_un,
                latency_ms,
                error_msg,
            ],
        )

        save_label_btn.click(
            _save_label_action,
            [
                dataset_state,
                idx_state,
                request_id,
                session_id,
                lang,
                asr_text,
                history_df,
                treat_unaddressed_as_eou,
                eou_threshold,
                human_label,
                note,
            ],
            [dataset_state, batch_status],
        )

        load_btn.click(
            _load_jsonl_action,
            [jsonl_file],
            [
                dataset_state,
                idx_state,
                idx_label,
                request_id,
                session_id,
                lang,
                asr_text,
                history_df,
                treat_unaddressed_as_eou,
                eou_threshold,
                human_label,
                note,
                pred_label,
                confidence,
                p_eou,
                p_cont,
                p_un,
                latency_ms,
                error_msg,
                batch_status,
            ],
        )

        prev_btn.click(
            lambda dataset, idx: _goto_idx(dataset, idx - 1),
            [dataset_state, idx_state],
            [
                idx_state,
                idx_label,
                request_id,
                session_id,
                lang,
                asr_text,
                history_df,
                treat_unaddressed_as_eou,
                eou_threshold,
                human_label,
                note,
                pred_label,
                confidence,
                p_eou,
                p_cont,
                p_un,
                latency_ms,
                error_msg,
            ],
        )

        next_btn.click(
            lambda dataset, idx: _goto_idx(dataset, idx + 1),
            [dataset_state, idx_state],
            [
                idx_state,
                idx_label,
                request_id,
                session_id,
                lang,
                asr_text,
                history_df,
                treat_unaddressed_as_eou,
                eou_threshold,
                human_label,
                note,
                pred_label,
                confidence,
                p_eou,
                p_cont,
                p_un,
                latency_ms,
                error_msg,
            ],
        )

        batch_btn.click(
            _batch_predict_action,
            [dataset_state, target, timeout_s, lang, treat_unaddressed_as_eou, eou_threshold, concurrency],
            [dataset_state, batch_status],
        )

        export_jsonl_btn.click(
            _export_jsonl_action,
            [dataset_state],
            [export_jsonl_file, batch_status],
        )

        export_excel_btn.click(
            _export_excel_action,
            [dataset_state],
            [export_excel_file, batch_status],
        )
