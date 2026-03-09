# Copyright 2025 the LlamaFactory team.
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

import pytest


@pytest.mark.runs_on(["cpu"])
def test_funaudiochat_registration_template_plugin():
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES
    from llamafactory.model.funaudiochat.register import register_funaudiochat

    register_funaudiochat()
    register_funaudiochat()  # idempotent

    cfg = AutoConfig.for_model("funaudiochat")
    assert getattr(cfg, "model_type", None) == "funaudiochat"
    assert type(cfg) in AutoModelForSeq2SeqLM._model_mapping.keys()
    if hasattr(AutoProcessor, "_processor_mapping"):
        assert type(cfg) in AutoProcessor._processor_mapping.keys()
    else:  # transformers>=4.56 moves processor mapping to module-level constant
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

        assert type(cfg) in PROCESSOR_MAPPING.keys()

    assert "funaudiochat" in TEMPLATES

    plugin = get_mm_plugin(name="funaudiochat", audio_token="<|AUDIO|>")
    assert plugin.audio_token == "<|AUDIO|>"


@pytest.mark.runs_on(["cpu"])
def test_funaudiochat_freeze_audio_tower_forbidden_modules():
    from transformers import AutoConfig

    from llamafactory.hparams import FinetuningArguments
    from llamafactory.model.funaudiochat.register import register_funaudiochat
    from llamafactory.model.model_utils.visual import get_forbidden_modules

    register_funaudiochat()

    cfg = AutoConfig.for_model("funaudiochat")
    args = FinetuningArguments()
    args.funaudiochat_freeze_audio_tower = True
    forbidden = get_forbidden_modules(cfg, args)

    assert "continuous_audio_tower" in forbidden
    assert "audio_tower" in forbidden
    assert "audio_invert_tower" in forbidden


@pytest.mark.runs_on(["cpu"])
def test_funaudiochat_tie_weights_accepts_recompute_mapping():
    from types import SimpleNamespace

    from llamafactory.model.funaudiochat.modeling_funaudiochat import FunAudioChatForConditionalGeneration

    tie_calls = []

    class DummyLegacyLanguageModel:
        def tie_weights(self):
            tie_calls.append(("legacy", (), {}))
            return "legacy-ok"

    fake_model = SimpleNamespace(
        audio_invert_tower=None,
        audio_tower=SimpleNamespace(embed_tokens="embed"),
        language_model=DummyLegacyLanguageModel(),
        _tie_or_clone_weights=lambda *args: None,
    )

    result = FunAudioChatForConditionalGeneration.tie_weights(fake_model, recompute_mapping=False)

    assert result == "legacy-ok"
    assert tie_calls == [("legacy", (), {})]


@pytest.mark.runs_on(["cpu"])
def test_funaudiochat_tie_or_clone_weights_compat():
    from types import SimpleNamespace

    from torch import nn

    from llamafactory.model.funaudiochat.modeling_funaudiochat import FunAudioChatPreTrainedModel

    fake_model = SimpleNamespace(config=SimpleNamespace(torchscript=False))
    output_embeddings = nn.Linear(4, 3, bias=True)
    input_embeddings = nn.Embedding(3, 4)

    FunAudioChatPreTrainedModel._tie_or_clone_weights(fake_model, output_embeddings, input_embeddings)

    assert output_embeddings.weight is input_embeddings.weight
    assert output_embeddings.out_features == input_embeddings.num_embeddings


@pytest.mark.runs_on(["cpu"])
def test_funaudiochat_tie_weights_forwards_supported_kwargs():
    from types import SimpleNamespace

    from llamafactory.model.funaudiochat.modeling_funaudiochat import FunAudioChatForConditionalGeneration

    tied_modules = []
    tie_calls = []

    class DummyModernLanguageModel:
        def tie_weights(self, *args, **kwargs):
            tie_calls.append((args, kwargs))
            return "modern-ok"

    fake_model = SimpleNamespace(
        audio_invert_tower=SimpleNamespace(lm_head="lm_head"),
        audio_tower=SimpleNamespace(embed_tokens="embed"),
        language_model=DummyModernLanguageModel(),
        _tie_or_clone_weights=lambda *args: tied_modules.append(args),
    )

    result = FunAudioChatForConditionalGeneration.tie_weights(fake_model, recompute_mapping=False)

    assert result == "modern-ok"
    assert tied_modules == [("lm_head", "embed")]
    assert tie_calls == [((), {"recompute_mapping": False})]
