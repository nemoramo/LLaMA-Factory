# Copyright 2025 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from transformers.utils import is_nltk_available

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.misc import numpify
from ...extras.packages import is_jieba_available, is_jiwer_available, is_rouge_available
from ..metric_utils import compute_error_rate, has_cjk, normalize_text


if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizer


if is_jieba_available():
    import jieba  # type: ignore


if is_jiwer_available():
    import jiwer  # type: ignore


if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore


if is_rouge_available():
    from rouge_chinese import Rouge  # type: ignore


logger = logging.get_logger(__name__)

ENDPOINTING_TAGS = ("<EOU>", "<CONT_USER>", "<UNADDRESSED>")
MERGED_ENDPOINTING_TAGS = ("<EOU>", "<CONT_USER>")
ENDPOINTING_TAG_NAME_MAP = {
    "<EOU>": "eou",
    "<CONT_USER>": "cont_user",
    "<UNADDRESSED>": "unaddressed",
}


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    r"""Compute the token with the largest likelihood to reduce memory footprint."""
    if isinstance(logits, (list, tuple)):
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]
        else:  # moe models have aux loss
            logits = logits[1]

    if logits.dim() != 3:
        raise ValueError("Cannot process the logits.")

    return torch.argmax(logits, dim=-1)


@dataclass
class ComputeAccuracy:
    r"""Compute accuracy and support `batch_eval_metrics`."""

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        setattr(self, "_printed_examples", False)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))

        if compute_result:
            return self._dump()


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _merge_unaddressed_as_eou(tag: str) -> str:
    return "<EOU>" if tag == "<UNADDRESSED>" else tag


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


@dataclass
class ComputeEndpointingMetrics:
    r"""Compute endpointing label metrics on the first supervised tag token."""

    tokenizer: "PreTrainedTokenizer"

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "token_accuracy"):
            label_summary = _summarize_confusion(self.label_confusion, self.label_row_totals, ENDPOINTING_TAGS)
            merged_summary = _summarize_confusion(
                self.merged_label_confusion,
                self.merged_label_row_totals,
                MERGED_ENDPOINTING_TAGS,
            )
            result = {
                "accuracy": float(np.mean(self.token_accuracy)) if self.token_accuracy else 0.0,
                "label_acc": float(label_summary["accuracy"]),
                "label_macro_f1": float(label_summary["macro_f1"]),
                "label_valid_ratio": _safe_div(self.label_total - self.label_invalid_count, self.label_total),
                "label_far_unad": _safe_div(
                    self.label_row_totals["<UNADDRESSED>"] - self.label_confusion["<UNADDRESSED>"]["<UNADDRESSED>"],
                    self.label_row_totals["<UNADDRESSED>"],
                ),
                "label_interrupt": _safe_div(
                    self.label_confusion["<CONT_USER>"]["<EOU>"],
                    self.label_row_totals["<CONT_USER>"],
                ),
                "label_delay": _safe_div(
                    self.label_confusion["<EOU>"]["<CONT_USER>"],
                    self.label_row_totals["<EOU>"],
                ),
                "label_missed": _safe_div(
                    self.label_confusion["<EOU>"]["<UNADDRESSED>"],
                    self.label_row_totals["<EOU>"],
                ),
                "merged_label_acc": float(merged_summary["accuracy"]),
                "merged_label_macro_f1": float(merged_summary["macro_f1"]),
                "merged_label_interrupt": _safe_div(
                    self.merged_label_confusion["<CONT_USER>"]["<EOU>"],
                    self.merged_label_row_totals["<CONT_USER>"],
                ),
                "merged_label_delay": _safe_div(
                    self.merged_label_confusion["<EOU>"]["<CONT_USER>"],
                    self.merged_label_row_totals["<EOU>"],
                ),
            }

            for tag, short_name in ENDPOINTING_TAG_NAME_MAP.items():
                per_label = label_summary["per_label"][tag]
                result[f"label_precision_{short_name}"] = float(per_label["precision"])
                result[f"label_recall_{short_name}"] = float(per_label["recall"])
                result[f"label_f1_{short_name}"] = float(per_label["f1"])

            for tag in MERGED_ENDPOINTING_TAGS:
                short_name = ENDPOINTING_TAG_NAME_MAP[tag]
                per_label = merged_summary["per_label"][tag]
                result[f"merged_label_precision_{short_name}"] = float(per_label["precision"])
                result[f"merged_label_recall_{short_name}"] = float(per_label["recall"])
                result[f"merged_label_f1_{short_name}"] = float(per_label["f1"])

        self.token_accuracy: list[float] = []
        self.label_total = 0
        self.label_invalid_count = 0
        self.label_confusion = _init_confusion(ENDPOINTING_TAGS)
        self.label_row_totals = {tag: 0 for tag in ENDPOINTING_TAGS}
        self.merged_label_confusion = _init_confusion(MERGED_ENDPOINTING_TAGS)
        self.merged_label_row_totals = {tag: 0 for tag in MERGED_ENDPOINTING_TAGS}
        return result

    def __post_init__(self):
        self.tag_token_ids = {}
        missing_tags = []
        for tag in ENDPOINTING_TAGS:
            token_id = self.tokenizer.convert_tokens_to_ids(tag)
            round_trip = self.tokenizer.convert_ids_to_tokens(token_id) if token_id is not None else None
            if token_id is None or int(token_id) < 0 or round_trip != tag:
                missing_tags.append(tag)
                continue
            self.tag_token_ids[int(token_id)] = tag

        if missing_tags:
            raise ValueError(f"Endpointing metrics require tokenizer to contain tags: {', '.join(missing_tags)}.")

        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)
        for i in range(len(preds)):
            pred, label = preds[i, :-1], labels[i, 1:]
            label_mask = label != IGNORE_INDEX
            if not np.any(label_mask):
                continue

            self.token_accuracy.append(np.mean(pred[label_mask] == label[label_mask]))

            first_label_pos = int(np.flatnonzero(label_mask)[0])
            gold_tag = self.tag_token_ids.get(int(label[first_label_pos]))
            if gold_tag not in ENDPOINTING_TAGS:
                continue

            pred_tag = self.tag_token_ids.get(int(pred[first_label_pos]), "OTHER")
            self.label_total += 1
            self.label_row_totals[gold_tag] += 1
            if pred_tag in ENDPOINTING_TAGS:
                self.label_confusion[gold_tag][pred_tag] += 1
            else:
                self.label_invalid_count += 1

            merged_gold_tag = _merge_unaddressed_as_eou(gold_tag)
            self.merged_label_row_totals[merged_gold_tag] += 1
            if pred_tag in ENDPOINTING_TAGS:
                merged_pred_tag = _merge_unaddressed_as_eou(pred_tag)
                self.merged_label_confusion[merged_gold_tag][merged_pred_tag] += 1

        if compute_result:
            return self._dump()


@dataclass
class ComputeSimilarity:
    r"""Compute text similarity scores and support `batch_eval_metrics`.

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.
    """

    tokenizer: "PreTrainedTokenizer"
    compute_wer_cer: bool = False

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {}
            for k, v in self.score_dict.items():
                if len(v) > 0:
                    result[k] = float(np.mean(v))

        # Reset metric containers. Only allocate WER/CER when they are enabled.
        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        if getattr(self, "compute_wer_cer", False):
            self.score_dict["wer"] = []
            self.score_dict["cer"] = []

        self._printed_examples = False
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Log a few sample prediction/label pairs during evaluation for debugging.
        if self.compute_wer_cer and not getattr(self, "_printed_examples", False):
            num_samples = min(10, len(decoded_preds))
            if num_samples > 0:
                sample_indices = np.random.choice(len(decoded_preds), size=num_samples, replace=False)
                logger.info_rank0("Sample predictions for WER/CER evaluation:")
                for rank, sample_idx in enumerate(sample_indices):
                    logger.info_rank0(f"[sample {rank}] pred : {decoded_preds[sample_idx]}")
                    logger.info_rank0(f"[sample {rank}] label: {decoded_labels[sample_idx]}")
                self._printed_examples = True

        for pred, label in zip(decoded_preds, decoded_labels):
            if is_jieba_available() and (has_cjk(pred) or has_cjk(label)):
                hypothesis = [t for t in jieba.cut(pred) if t.strip()]
                reference = [t for t in jieba.cut(label) if t.strip()]
            else:
                hypothesis = pred.split()
                reference = label.split()

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}
            else:
                rouge = Rouge()
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
                result = scores[0]

            for k, v in result.items():
                self.score_dict[k].append(round(v["f"] * 100, 4))

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))

            if self.compute_wer_cer:
                pred_norm = normalize_text(pred)
                label_norm = normalize_text(label)

                if is_jieba_available() and (has_cjk(pred_norm) or has_cjk(label_norm)):
                    hypothesis_wer = [t for t in jieba.cut(pred_norm) if t.strip()]
                    reference_wer = [t for t in jieba.cut(label_norm) if t.strip()]
                else:
                    hypothesis_wer = pred_norm.split()
                    reference_wer = label_norm.split()

                if is_jiwer_available():
                    wer = jiwer.wer(" ".join(reference_wer), " ".join(hypothesis_wer))
                    cer = jiwer.cer(" ".join(list(label_norm)), " ".join(list(pred_norm)))
                else:
                    # Word Error Rate (WER) based on segmented tokens
                    wer = compute_error_rate(reference_wer, hypothesis_wer)
                    # Character Error Rate (CER) based on raw characters
                    cer = compute_error_rate(list(label_norm), list(pred_norm))

                self.score_dict["wer"].append(round(wer * 100, 4))
                self.score_dict["cer"].append(round(cer * 100, 4))

        if compute_result:
            return self._dump()
