"""fairseq wav2vec 2.0 names map onto transformers' Wav2Vec2Model, every key matching."""

from __future__ import annotations

import pytest
import torch

from deeptrace_bench.models._fairseq import check_config, fairseq_to_hf, hf_name


def _tiny_config(stable: bool):
    from transformers import Wav2Vec2Config

    return Wav2Vec2Config(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_stride=(5, 2),
        conv_kernel=(10, 3),
        conv_bias=stable,
        num_conv_pos_embeddings=4,
        num_conv_pos_embedding_groups=2,
        feat_extract_norm="layer" if stable else "group",
        do_stable_layer_norm=stable,
    )


def _fairseq_state(config, stable: bool) -> dict[str, torch.Tensor]:
    """A state dict named the way fairseq 0.12's Wav2Vec2Model names it."""
    h, f = config.hidden_size, config.intermediate_size
    state: dict[str, torch.Tensor] = {}
    channels = 1
    for i, (dim, kernel) in enumerate(zip(config.conv_dim, config.conv_kernel, strict=True)):
        state[f"feature_extractor.conv_layers.{i}.0.weight"] = torch.randn(dim, channels, kernel)
        if config.conv_bias:
            state[f"feature_extractor.conv_layers.{i}.0.bias"] = torch.randn(dim)
        if stable:  # layer norm on every conv layer, inside TransposeLast wrappers
            state[f"feature_extractor.conv_layers.{i}.2.1.weight"] = torch.randn(dim)
            state[f"feature_extractor.conv_layers.{i}.2.1.bias"] = torch.randn(dim)
        elif i == 0:  # group norm on the first layer only
            state[f"feature_extractor.conv_layers.{i}.2.weight"] = torch.randn(dim)
            state[f"feature_extractor.conv_layers.{i}.2.bias"] = torch.randn(dim)
        channels = dim
    c = config.conv_dim[-1]
    state |= {
        "layer_norm.weight": torch.randn(c),
        "layer_norm.bias": torch.randn(c),
        "post_extract_proj.weight": torch.randn(h, c),
        "post_extract_proj.bias": torch.randn(h),
        "mask_emb": torch.randn(h),
        "encoder.pos_conv.0.bias": torch.randn(h),
        "encoder.pos_conv.0.weight_g": torch.randn(1, 1, config.num_conv_pos_embeddings),
        "encoder.pos_conv.0.weight_v": torch.randn(
            h, h // config.num_conv_pos_embedding_groups, config.num_conv_pos_embeddings
        ),
        "encoder.layer_norm.weight": torch.randn(h),
        "encoder.layer_norm.bias": torch.randn(h),
        # pretraining-only parts, which the conversion drops
        "quantizer.vars": torch.randn(1, 4, 8),
        "quantizer.weight_proj.weight": torch.randn(4, c),
        "project_q.weight": torch.randn(8, 8),
        "final_proj.weight": torch.randn(8, h),
    }
    for layer in range(config.num_hidden_layers):
        p = f"encoder.layers.{layer}."
        for proj in ("q", "k", "v", "out"):
            state[f"{p}self_attn.{proj}_proj.weight"] = torch.randn(h, h)
            state[f"{p}self_attn.{proj}_proj.bias"] = torch.randn(h)
        for norm in ("self_attn_layer_norm", "final_layer_norm"):
            state[f"{p}{norm}.weight"] = torch.randn(h)
            state[f"{p}{norm}.bias"] = torch.randn(h)
        state[f"{p}fc1.weight"], state[f"{p}fc1.bias"] = torch.randn(f, h), torch.randn(f)
        state[f"{p}fc2.weight"], state[f"{p}fc2.bias"] = torch.randn(h, f), torch.randn(h)
    return state


@pytest.mark.parametrize("stable", [True, False], ids=["xls_r_layer_norm", "base_group_norm"])
def test_fairseq_state_loads_into_wav2vec2_with_every_key(stable):
    from transformers import Wav2Vec2Model

    config = _tiny_config(stable)
    fairseq = {f"ssl_model.model.{k}": v for k, v in _fairseq_state(config, stable).items()}
    fairseq["LL.weight"] = torch.randn(4, 16)
    converted = fairseq_to_hf(fairseq, prefix="ssl_model.model.", new_prefix="ssl.")
    assert converted["LL.weight"] is fairseq["LL.weight"]  # the detector head passes through
    backbone = {k.removeprefix("ssl."): v for k, v in converted.items() if k.startswith("ssl.")}
    check_config(config, backbone)
    model = Wav2Vec2Model(config)
    model.load_state_dict(backbone, strict=True)
    # the values arrive where fairseq had them
    torch.testing.assert_close(
        model.encoder.layers[1].feed_forward.output_dense.weight,
        fairseq["ssl_model.model.encoder.layers.1.fc2.weight"],
    )


def test_unknown_fairseq_names_fail_loudly():
    with pytest.raises(KeyError, match="encoder.layers.0.adapter"):
        hf_name("encoder.layers.0.adapter.weight")


def test_config_mismatch_is_named():
    config = _tiny_config(True)
    state = fairseq_to_hf(_fairseq_state(config, True), prefix="")
    config.num_hidden_layers = 3
    with pytest.raises(ValueError, match="num_hidden_layers"):
        check_config(config, state)
