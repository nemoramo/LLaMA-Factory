#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shlex
import signal
import subprocess
import time
from concurrent import futures
from math import exp, isfinite, log
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import grpc
import httpx

import endpointing_pb2
import endpointing_pb2_grpc


LABELS: List[str] = ["<EOU>", "<CONT_USER>", "<UNADDRESSED>"]


def _logsumexp(logps: List[float]) -> float:
    finite = [x for x in logps if isfinite(x)]
    if not finite:
        return float("-inf")
    m = max(finite)
    return m + log(sum(exp(x - m) for x in finite))


def _normalize_label(text: Optional[str]) -> str:
    if not text:
        return ""
    s = text.strip().lower()
    t = s.replace(" ", "")
    if "<eou>" in t or t == "eou":
        return "<EOU>"
    if "<cont_user>" in t or t in ("cont_user", "cont"):
        return "<CONT_USER>"
    if "<unaddressed>" in t or t in ("unaddressed", "unrelated", "unadd"):
        return "<UNADDRESSED>"
    return ""


def _first_token_from_logprobs(choice: Dict[str, Any]) -> str:
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    if not content:
        return ""
    tok0 = content[0].get("token")
    return (tok0 or "").strip()


def _extract_label_probs(choice: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract label probabilities from OpenAI-style top_logprobs and renormalize over LABELS only.
    Missing labels get probability 0.
    """
    lp = choice.get("logprobs") or {}
    content = lp.get("content") or []
    top = (content[0].get("top_logprobs") if content else None) or []

    logps_by_label: Dict[str, List[float]] = {l: [] for l in LABELS}
    for item in top:
        tok = item.get("token")
        lprob = item.get("logprob")
        if tok is None or lprob is None:
            continue
        lab = _normalize_label(tok)
        if lab in logps_by_label and isinstance(lprob, (int, float)):
            logps_by_label[lab].append(float(lprob))

    merged_logps = {l: _logsumexp(v) for l, v in logps_by_label.items()}
    denom = _logsumexp(list(merged_logps.values()))
    if not isfinite(denom):
        return {l: 0.0 for l in LABELS}
    return {l: (exp(merged_logps[l] - denom) if isfinite(merged_logps[l]) else 0.0) for l in LABELS}


def _apply_options(probs: Dict[str, float], treat_unaddressed_as_eou: bool) -> Dict[str, float]:
    pe = float(probs.get("<EOU>", 0.0))
    pc = float(probs.get("<CONT_USER>", 0.0))
    pu = float(probs.get("<UNADDRESSED>", 0.0))

    if treat_unaddressed_as_eou:
        pe = pe + pu
        pu = 0.0

    total = pe + pc + pu
    if total > 0:
        pe /= total
        pc /= total
        pu /= total
    return {"<EOU>": pe, "<CONT_USER>": pc, "<UNADDRESSED>": pu}


def _decide_label(
    probs: Dict[str, float], eou_threshold: float, treat_unaddressed_as_eou: bool
) -> Tuple[str, float]:
    pe = float(probs.get("<EOU>", 0.0))
    pc = float(probs.get("<CONT_USER>", 0.0))
    pu = float(probs.get("<UNADDRESSED>", 0.0))

    if pe >= eou_threshold and pe >= pc and pe >= pu:
        return "<EOU>", pe
    if (not treat_unaddressed_as_eou) and pu >= pc:
        return "<UNADDRESSED>", pu
    return "<CONT_USER>", pc


def _build_messages(req: endpointing_pb2.EndpointingRequest) -> List[Dict[str, str]]:
    system = (
        "You are an endpointing classifier for spoken dialog.\n"
        f"Language: {req.lang}\n\n"
        "Task: Given the conversation history and the current user ASR transcript, "
        "output EXACTLY ONE token:\n"
        "<EOU> | <CONT_USER> | <UNADDRESSED>\n\n"
        "Definitions:\n"
        "- <EOU>: user finished speaking; assistant should respond now.\n"
        "- <CONT_USER>: user will continue speaking / ASR is partial; wait.\n"
        "- <UNADDRESSED>: speech is not addressed to the assistant or is unrelated.\n\n"
        "Output constraints: no spaces, no punctuation, no explanation."
    )

    msgs: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for t in req.history:
        if t.role == endpointing_pb2.USER:
            role = "user"
        elif t.role == endpointing_pb2.ASSISTANT:
            role = "assistant"
        else:
            continue
        if t.text:
            msgs.append({"role": role, "content": t.text})

    msgs.append({"role": "user", "content": req.asr.text})
    return msgs


def _parse_logit_bias_json(s: str) -> Dict[str, int]:
    obj = json.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("LOGIT_BIAS_JSON must be a JSON object like {\"123\": 100}")
    return {str(k): int(v) for k, v in obj.items()}


def _compute_logit_bias_from_tokenizer(tokenizer_dir: str, bias_value: int) -> Dict[str, int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    mapping: Dict[str, int] = {}
    for t in LABELS:
        ids = tok.encode(t, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Token {t!r} is not a single token in tokenizer_dir={tokenizer_dir!r}: ids={ids}")
        mapping[str(ids[0])] = int(bias_value)
    return mapping


class ServiceConfig:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        model_alias: str,
        timeout_s: float,
        top_logprobs: int,
        default_eou_threshold: float,
        logit_bias: Optional[Dict[str, int]],
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.model_alias = model_alias
        self.timeout_s = timeout_s
        self.top_logprobs = top_logprobs
        self.default_eou_threshold = default_eou_threshold
        self.logit_bias = logit_bias


def _resolve_first_model_id(http: httpx.Client, base_url: str, api_key: str) -> str:
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = http.get(url, headers=headers)
    r.raise_for_status()
    data = r.json()
    models = data.get("data") or []
    if not models:
        raise RuntimeError("No models returned by /v1/models.")
    return models[0].get("id") or ""


def _wait_vllm_ready(http: httpx.Client, base_url: str, api_key: str, max_wait_s: float) -> None:
    t0 = time.perf_counter()
    last_err: Optional[str] = None
    while True:
        try:
            _ = _resolve_first_model_id(http, base_url, api_key)
            return
        except Exception as e:
            last_err = str(e)
        if time.perf_counter() - t0 >= max_wait_s:
            raise RuntimeError(f"vLLM not ready after {max_wait_s:.1f}s: {last_err}")
        time.sleep(0.5)


def _start_vllm(cmd: str) -> subprocess.Popen:
    argv = shlex.split(cmd)
    if not argv:
        raise ValueError("Empty --vllm-cmd.")
    return subprocess.Popen(argv, preexec_fn=os.setsid)


class EndpointingServicer(endpointing_pb2_grpc.EndpointingServiceServicer):
    def __init__(self, cfg: ServiceConfig, http: httpx.Client):
        self.cfg = cfg
        self.http = http

    def Predict(self, request, context):  # noqa: N802 (generated signature)
        t0 = time.perf_counter()
        try:
            eou_threshold = self.cfg.default_eou_threshold
            if request.options.HasField("eou_threshold"):
                eou_threshold = float(request.options.eou_threshold)
            if not (0.0 <= float(eou_threshold) <= 1.0):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "options.eou_threshold must be in [0, 1].")

            treat_unaddressed = bool(request.options.treat_unaddressed_as_eou)

            payload: Dict[str, Any] = {
                "model": self.cfg.model,
                "messages": _build_messages(request),
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": self.cfg.top_logprobs,
            }
            if self.cfg.logit_bias:
                payload["logit_bias"] = self.cfg.logit_bias

            url = self.cfg.base_url.rstrip("/") + "/chat/completions"
            headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
            r = self.http.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

            choices = data.get("choices") or []
            if not choices:
                context.abort(grpc.StatusCode.UNAVAILABLE, "Empty choices from vLLM.")
            choice0 = choices[0]

            msg = (choice0.get("message") or {}) if isinstance(choice0, dict) else {}
            txt = ((msg.get("content") or "") if isinstance(msg, dict) else "").strip()
            if not txt:
                txt = _first_token_from_logprobs(choice0)
            pred_from_text = _normalize_label(txt)

            probs = _extract_label_probs(choice0)
            probs = _apply_options(probs, treat_unaddressed_as_eou=treat_unaddressed)
            if sum(probs.values()) <= 0.0:
                if pred_from_text in LABELS:
                    probs = {l: (1.0 if l == pred_from_text else 0.0) for l in LABELS}
                else:
                    probs = {"<EOU>": 0.0, "<CONT_USER>": 1.0, "<UNADDRESSED>": 0.0}
                probs = _apply_options(probs, treat_unaddressed_as_eou=treat_unaddressed)

            label, conf = _decide_label(
                probs, eou_threshold=float(eou_threshold), treat_unaddressed_as_eou=treat_unaddressed
            )

            t1 = time.perf_counter()
            return endpointing_pb2.EndpointingResponse(
                request_id=request.request_id,
                label=label,
                confidence=float(conf),
                model=self.cfg.model_alias,
                latency_ms=int(round((t1 - t0) * 1000)),
                p_eou=float(probs.get("<EOU>", 0.0)),
                p_cont_user=float(probs.get("<CONT_USER>", 0.0)),
                p_unaddressed=float(probs.get("<UNADDRESSED>", 0.0)),
            )

        except grpc.RpcError:
            raise
        except httpx.HTTPStatusError as e:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                f"vLLM HTTP {e.response.status_code}: {e.response.text[:2000]}",
            )
        except Exception as e:
            context.abort(grpc.StatusCode.INTERNAL, str(e))


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "50051")))

    ap.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:30000/v1"))
    ap.add_argument("--api-key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.getenv("VLLM_MODEL", ""))
    ap.add_argument("--model-alias", default=os.getenv("MODEL_ALIAS", "endpointing-judge-v1"))

    ap.add_argument("--timeout", type=float, default=float(os.getenv("TIMEOUT_S", "30")))
    ap.add_argument("--top-logprobs", type=int, default=int(os.getenv("TOP_LOGPROBS", "20")))
    ap.add_argument("--default-eou-threshold", type=float, default=float(os.getenv("EOU_THRESHOLD", "0.6")))

    ap.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "64")))

    ap.add_argument(
        "--vllm-cmd",
        default=os.getenv("VLLM_CMD", ""),
        help='Optional: command to launch vLLM server (e.g. "vllm serve /models/model --host 0.0.0.0 --port 30000 ...").',
    )
    ap.add_argument("--wait-ready", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ready-timeout", type=float, default=float(os.getenv("READY_TIMEOUT_S", "120")))

    # logit_bias to strongly restrict output to LABELS only.
    ap.add_argument("--logit-bias-json", default=os.getenv("LOGIT_BIAS_JSON", ""))
    ap.add_argument("--logit-bias-value", type=int, default=int(os.getenv("LOGIT_BIAS_VALUE", "100")))
    ap.add_argument(
        "--tokenizer-dir",
        default=os.getenv("TOKENIZER_DIR", ""),
        help="Tokenizer/model dir used to compute <EOU>/<CONT_USER>/<UNADDRESSED> token ids for logit_bias.",
    )
    ap.add_argument("--require-logit-bias", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    vllm_proc: Optional[subprocess.Popen] = None
    if args.vllm_cmd:
        vllm_proc = _start_vllm(args.vllm_cmd)

    http = httpx.Client(timeout=args.timeout)
    try:
        if args.wait_ready:
            _wait_vllm_ready(http, args.base_url, args.api_key, max_wait_s=args.ready_timeout)

        model_id = args.model
        if not model_id:
            model_id = _resolve_first_model_id(http, args.base_url, args.api_key)

        logit_bias: Optional[Dict[str, int]] = None
        if args.logit_bias_json:
            logit_bias = _parse_logit_bias_json(args.logit_bias_json)
        else:
            tokenizer_dir = args.tokenizer_dir
            if not tokenizer_dir:
                default_dir = "/models/model"
                if Path(default_dir).exists():
                    tokenizer_dir = default_dir
            if tokenizer_dir:
                logit_bias = _compute_logit_bias_from_tokenizer(tokenizer_dir, args.logit_bias_value)
            elif args.require_logit_bias:
                raise RuntimeError(
                    "logit_bias is required but cannot be constructed: provide LOGIT_BIAS_JSON or mount TOKENIZER_DIR."
                )

        cfg = ServiceConfig(
            base_url=args.base_url,
            api_key=args.api_key,
            model=model_id,
            model_alias=args.model_alias,
            timeout_s=float(args.timeout),
            top_logprobs=int(args.top_logprobs),
            default_eou_threshold=float(args.default_eou_threshold),
            logit_bias=logit_bias,
        )

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.max_workers))
        endpointing_pb2_grpc.add_EndpointingServiceServicer_to_server(EndpointingServicer(cfg, http), server)
        server.add_insecure_port(f"{args.host}:{args.port}")

        print(f"[grpc] listening on {args.host}:{args.port}")
        print(f"[vllm] base_url={args.base_url} model={cfg.model} alias={cfg.model_alias}")
        if cfg.logit_bias:
            print(f"[vllm] logit_bias={cfg.logit_bias}")
        server.start()
        server.wait_for_termination()

    finally:
        http.close()
        if vllm_proc is not None:
            try:
                os.killpg(os.getpgid(vllm_proc.pid), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    main()
