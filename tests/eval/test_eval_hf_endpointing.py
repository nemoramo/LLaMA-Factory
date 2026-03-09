import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = Path(__file__).resolve().parents[2] / "examples" / "speech_endpointing" / "eval_hf_endpointing.py"
SPEC = importlib.util.spec_from_file_location("eval_hf_endpointing", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class _FakeTokenizer:
    def __init__(self, vocab_size: int) -> None:
        self._vocab_size = vocab_size
        self.pad_token = None
        self.eos_token = "</s>"
        self.padding_side = "right"
        self.truncation_side = "right"

    def __len__(self) -> int:
        return self._vocab_size


class _FakeModel:
    def __init__(self, vocab_size: int = 100, hidden_size: int = 8) -> None:
        self._hidden_size = hidden_size
        self._input_embeddings = torch.nn.Embedding(vocab_size, hidden_size)
        self._output_embeddings = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.resize_calls: list[tuple[int, int | None]] = []

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self._input_embeddings

    def get_output_embeddings(self) -> torch.nn.Linear:
        return self._output_embeddings

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: int | None = None) -> torch.nn.Embedding:
        self.resize_calls.append((new_num_tokens, pad_to_multiple_of))
        resized_num_tokens = new_num_tokens
        if pad_to_multiple_of is not None:
            resized_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        self._input_embeddings = torch.nn.Embedding(resized_num_tokens, self._hidden_size)
        self._output_embeddings = torch.nn.Linear(self._hidden_size, resized_num_tokens, bias=False)
        self.config.vocab_size = resized_num_tokens
        return self._input_embeddings

    def eval(self) -> "_FakeModel":
        return self


def test_load_model_and_tokenizer_resizes_before_loading_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = _FakeTokenizer(vocab_size=103)
    model = _FakeModel(vocab_size=100)
    peft_load_state: dict[str, int] = {}

    monkeypatch.setattr(MODULE.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(MODULE.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
    monkeypatch.setattr(MODULE, "_infer_adapter_embedding_size", lambda adapter_path: 128)

    def _fake_peft_from_pretrained(base_model: _FakeModel, adapter_path: str) -> _FakeModel:
        peft_load_state["vocab_size_before_peft_load"] = base_model.get_input_embeddings().weight.size(0)
        return base_model

    monkeypatch.setattr(MODULE.PeftModel, "from_pretrained", staticmethod(_fake_peft_from_pretrained))

    _, loaded_model = MODULE._load_model_and_tokenizer(
        "Qwen/Qwen3-0.6B",
        "/tmp/fake-adapter",
        dtype=torch.float16,
        attn_impl="auto",
    )

    assert loaded_model is model
    assert model.resize_calls == [(128, None)]
    assert peft_load_state["vocab_size_before_peft_load"] == 128


def test_validate_tag_token_ids_fit_model_raises_for_out_of_range_ids() -> None:
    model = _FakeModel(vocab_size=100)

    with pytest.raises(ValueError, match="Tokenizer label token ids exceed model vocab size"):
        MODULE._validate_tag_token_ids_fit_model(model, {"<EOU>": 100, "<CONT_USER>": 101, "<UNADDRESSED>": 102})


def test_gather_tag_logits_uses_logits_device_and_tag_order() -> None:
    next_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=torch.float32)
    tag_logits = MODULE._gather_tag_logits(
        next_logits,
        {"<EOU>": 4, "<CONT_USER>": 1, "<UNADDRESSED>": 3},
    )

    assert tag_logits.device == next_logits.device
    assert torch.allclose(tag_logits, torch.tensor([[0.5, 0.2, 0.4]], dtype=torch.float32))
