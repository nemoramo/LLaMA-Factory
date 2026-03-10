# Copyright 2026 the LlamaFactory team.
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
def test_qwen3_asr_registration_template_plugin():
    from transformers import AutoConfig, AutoModel, AutoProcessor

    from llamafactory.data.mm_plugin import get_mm_plugin
    from llamafactory.data.template import TEMPLATES
    from llamafactory.model.qwen3_asr.register import register_qwen3_asr

    register_qwen3_asr()
    register_qwen3_asr()  # idempotent

    cfg = AutoConfig.for_model("qwen3_asr")
    assert getattr(cfg, "model_type", None) == "qwen3_asr"
    assert type(cfg) in AutoModel._model_mapping.keys()
    if hasattr(AutoProcessor, "_processor_mapping"):
        assert type(cfg) in AutoProcessor._processor_mapping.keys()
    else:  # transformers>=4.56 moves processor mapping to module-level constant
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING

        assert type(cfg) in PROCESSOR_MAPPING.keys()

    assert "qwen3_asr" in TEMPLATES

    plugin = get_mm_plugin(name="qwen3_asr", audio_token="<|audio_pad|>")
    assert plugin.audio_token == "<|audio_pad|>"


@pytest.mark.runs_on(["cpu"])
def test_qwen3_5_template_registered():
    from llamafactory.data.template import TEMPLATES

    assert "qwen3_5_nothink" in TEMPLATES


@pytest.mark.runs_on(["cpu"])
def test_qwen3_asr_default_rope_type_supported():
    import torch

    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import Qwen3ASRTextConfig
    from llamafactory.model.qwen3_asr.modeling_qwen3_asr import Qwen3ASRThinkerTextRotaryEmbedding

    cfg = Qwen3ASRTextConfig(
        hidden_size=96,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=12,
        max_position_embeddings=128,
        rope_theta=1000000,
        rope_scaling={"rope_type": "default", "type": "default", "mrope_section": [2, 2, 2]},
    )

    rotary = Qwen3ASRThinkerTextRotaryEmbedding(cfg)
    cos, sin = rotary(torch.zeros(1, 2, 12), torch.arange(2).unsqueeze(0))

    assert rotary.rope_type == "default"
    assert cos.shape == sin.shape == (1, 2, 12)


@pytest.mark.runs_on(["cpu"])
def test_qwen3_asr_unknown_rope_type_raises():
    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import Qwen3ASRTextConfig
    from llamafactory.model.qwen3_asr.modeling_qwen3_asr import Qwen3ASRThinkerTextRotaryEmbedding

    cfg = Qwen3ASRTextConfig(
        hidden_size=96,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=12,
        max_position_embeddings=128,
        rope_theta=1000000,
        rope_scaling={"rope_type": "definitely_not_supported", "mrope_section": [2, 2, 2]},
    )

    with pytest.raises(ValueError, match="Unsupported rope_type"):
        Qwen3ASRThinkerTextRotaryEmbedding(cfg)


@pytest.mark.runs_on(["cpu"])
def test_qwen3_asr_thinker_config_inherits_pad_token_id():
    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import Qwen3ASRTextConfig, Qwen3ASRThinkerConfig

    text_config = Qwen3ASRTextConfig(pad_token_id=151643)
    thinker_config = Qwen3ASRThinkerConfig(text_config=text_config)

    assert thinker_config.pad_token_id == 151643
    assert thinker_config.text_config.pad_token_id == 151643


@pytest.mark.runs_on(["cpu"])
def test_qwen3_asr_create_causal_mask_compat_accepts_input_embeds(monkeypatch):
    import torch

    from llamafactory.model.qwen3_asr import modeling_qwen3_asr as qwen3_asr_modeling
    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import Qwen3ASRTextConfig, Qwen3ASRThinkerConfig

    captured = {}

    def fake_create_causal_mask(
        *,
        config,
        input_embeds,
        attention_mask,
        cache_position,
        past_key_values,
        position_ids=None,
    ):
        captured["config"] = config
        captured["input_embeds"] = input_embeds
        captured["attention_mask"] = attention_mask
        captured["cache_position"] = cache_position
        captured["past_key_values"] = past_key_values
        captured["position_ids"] = position_ids
        return torch.ones(1, 1, 2, 2)

    monkeypatch.setattr(qwen3_asr_modeling, "create_causal_mask", fake_create_causal_mask)

    thinker_config = Qwen3ASRThinkerConfig(
        text_config=Qwen3ASRTextConfig(
            hidden_size=96,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=8,
            head_dim=12,
            max_position_embeddings=128,
        )
    )
    embeds = torch.randn(1, 2, 96)
    position_ids = torch.arange(2).unsqueeze(0)
    cache_position = torch.arange(2)

    mask = qwen3_asr_modeling._create_causal_mask_compat(
        config=thinker_config,
        inputs_embeds=embeds,
        attention_mask=None,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )

    assert torch.equal(mask, torch.ones(1, 1, 2, 2))
    assert captured["config"] is thinker_config
    assert captured["input_embeds"] is embeds
    assert captured["attention_mask"] is None
    assert captured["cache_position"] is cache_position
    assert captured["past_key_values"] is None
    assert captured["position_ids"] is position_ids
