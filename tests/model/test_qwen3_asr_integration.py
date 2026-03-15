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


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen3_asr_audio_encoder_batched_forward_matches_serial():
    import torch

    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
    from llamafactory.model.qwen3_asr.modeling_qwen3_asr import Qwen3ASRAudioEncoder, _get_feat_extract_output_lengths

    torch.manual_seed(0)
    config = Qwen3ASRAudioEncoderConfig(
        num_mel_bins=8,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        max_source_positions=64,
        n_window=4,
        n_window_infer=8,
        output_dim=16,
        downsample_hidden_size=8,
    )
    setattr(config, "_attn_implementation", "eager")
    encoder = Qwen3ASRAudioEncoder(config)
    encoder.eval()

    feature_lens = torch.tensor([12, 16], dtype=torch.long)
    input_features = torch.randn(2, config.num_mel_bins, int(feature_lens.max()))
    expected_lens = _get_feat_extract_output_lengths(feature_lens)

    serial_outputs = []
    for i, feature_len in enumerate(feature_lens.tolist()):
        serial_outputs.append(
            encoder(
                input_features[i, :, :feature_len],
                feature_lens=torch.tensor([feature_len], dtype=torch.long),
            ).last_hidden_state
        )

    serial_hidden = torch.cat(serial_outputs, dim=0)
    batched_hidden = encoder(input_features, feature_lens=feature_lens).last_hidden_state

    assert batched_hidden.shape == (int(expected_lens.sum().item()), config.output_dim)
    assert torch.allclose(batched_hidden, serial_hidden, atol=1e-5, rtol=1e-5)


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen3_asr_get_audio_features_batches_training_audio_tower(monkeypatch):
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
    )
    from llamafactory.model.qwen3_asr.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
        _get_feat_extract_output_lengths,
    )

    thinker_config = Qwen3ASRThinkerConfig(
        audio_config=Qwen3ASRAudioEncoderConfig(
            num_mel_bins=8,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
            n_window=4,
            n_window_infer=8,
            output_dim=16,
            downsample_hidden_size=8,
        ),
        text_config=Qwen3ASRTextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
        ),
    )
    model = Qwen3ASRThinkerForConditionalGeneration(thinker_config)
    model.train()
    setattr(model.audio_tower.config, "_attn_implementation", "flash_attention_2")

    feature_lens = torch.tensor([12, 16], dtype=torch.long)
    input_features = torch.randn(2, thinker_config.audio_config.num_mel_bins, int(feature_lens.max()))
    feature_attention_mask = torch.zeros(2, int(feature_lens.max()), dtype=torch.long)
    for i, feature_len in enumerate(feature_lens.tolist()):
        feature_attention_mask[i, :feature_len] = 1

    calls = []

    def fake_forward(input_features, feature_lens=None, aftercnn_lens=None):
        calls.append(
            {
                "shape": tuple(input_features.shape),
                "feature_lens": None if feature_lens is None else feature_lens.clone(),
                "aftercnn_lens": None if aftercnn_lens is None else aftercnn_lens.clone(),
            }
        )
        assert input_features.ndim == 3
        assert feature_lens is not None
        token_lens = _get_feat_extract_output_lengths(feature_lens)
        total_tokens = int(token_lens.sum().item())
        hidden = torch.arange(
            total_tokens * thinker_config.audio_config.output_dim,
            dtype=torch.float32,
        ).view(total_tokens, thinker_config.audio_config.output_dim)
        return BaseModelOutput(last_hidden_state=hidden)

    monkeypatch.setattr(model.audio_tower, "forward", fake_forward)

    audio_features, audio_feature_token_lens = model.get_audio_features(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    expected_token_lens = _get_feat_extract_output_lengths(feature_lens).tolist()
    assert len(calls) == 1
    assert calls[0]["shape"] == tuple(input_features.shape)
    assert torch.equal(calls[0]["feature_lens"], feature_lens)
    assert calls[0]["aftercnn_lens"] is None
    assert audio_feature_token_lens == expected_token_lens
    assert audio_features.shape == (sum(expected_token_lens), thinker_config.audio_config.output_dim)


@pytest.mark.runs_on(["cpu", "mps"])
def test_qwen3_asr_get_audio_features_keeps_serial_training_path_without_fa2(monkeypatch):
    import torch
    from transformers.modeling_outputs import BaseModelOutput

    from llamafactory.model.qwen3_asr.configuration_qwen3_asr import (
        Qwen3ASRAudioEncoderConfig,
        Qwen3ASRTextConfig,
        Qwen3ASRThinkerConfig,
    )
    from llamafactory.model.qwen3_asr.modeling_qwen3_asr import (
        Qwen3ASRThinkerForConditionalGeneration,
        _get_feat_extract_output_lengths,
    )

    thinker_config = Qwen3ASRThinkerConfig(
        audio_config=Qwen3ASRAudioEncoderConfig(
            num_mel_bins=8,
            encoder_layers=2,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            d_model=32,
            max_source_positions=64,
            n_window=4,
            n_window_infer=8,
            output_dim=16,
            downsample_hidden_size=8,
        ),
        text_config=Qwen3ASRTextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=0,
        ),
    )
    model = Qwen3ASRThinkerForConditionalGeneration(thinker_config)
    model.train()
    setattr(model.audio_tower.config, "_attn_implementation", "eager")

    feature_lens = torch.tensor([12, 16], dtype=torch.long)
    input_features = torch.randn(2, thinker_config.audio_config.num_mel_bins, int(feature_lens.max()))
    feature_attention_mask = torch.zeros(2, int(feature_lens.max()), dtype=torch.long)
    for i, feature_len in enumerate(feature_lens.tolist()):
        feature_attention_mask[i, :feature_len] = 1

    calls = []

    def fake_forward(input_features, feature_lens=None, aftercnn_lens=None):
        calls.append(
            {
                "shape": tuple(input_features.shape),
                "feature_lens": None if feature_lens is None else feature_lens.clone(),
                "aftercnn_lens": None if aftercnn_lens is None else aftercnn_lens.clone(),
            }
        )
        assert input_features.ndim == 2
        assert feature_lens is not None
        token_lens = _get_feat_extract_output_lengths(feature_lens)
        total_tokens = int(token_lens.sum().item())
        hidden = torch.zeros(total_tokens, thinker_config.audio_config.output_dim)
        return BaseModelOutput(last_hidden_state=hidden)

    monkeypatch.setattr(model.audio_tower, "forward", fake_forward)

    audio_features, audio_feature_token_lens = model.get_audio_features(
        input_features=input_features,
        feature_attention_mask=feature_attention_mask,
    )

    expected_token_lens = _get_feat_extract_output_lengths(feature_lens).tolist()
    assert len(calls) == 2
    assert calls[0]["shape"] == (thinker_config.audio_config.num_mel_bins, 12)
    assert calls[1]["shape"] == (thinker_config.audio_config.num_mel_bins, 16)
    assert torch.equal(calls[0]["feature_lens"], torch.tensor([12]))
    assert torch.equal(calls[1]["feature_lens"], torch.tensor([16]))
    assert calls[0]["aftercnn_lens"] is None
    assert calls[1]["aftercnn_lens"] is None
    assert audio_feature_token_lens == expected_token_lens
    assert audio_features.shape == (sum(expected_token_lens), thinker_config.audio_config.output_dim)


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
