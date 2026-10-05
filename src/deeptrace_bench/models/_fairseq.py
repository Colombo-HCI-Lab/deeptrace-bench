"""fairseq wav2vec 2.0 weights in transformers' ``Wav2Vec2Model``, without fairseq.

Several audio detectors fine-tune a fairseq wav2vec 2.0 or XLS-R front end (XLS-R + AASIST,
XLS-R + SLS, XLSR-Mamba, the AntiDeepfake family) and ship the result as one state dict
with fairseq's parameter names. fairseq doesn't install on Python 3.12, but its network is
the one transformers implements as ``Wav2Vec2Model``: same layers, same order, other names.
``fairseq_to_hf`` renames a fairseq state dict so it loads into ``Wav2Vec2Model`` with every
key matching, and ``hf_config_for`` checks the transformers config agrees with the weights'
shapes. The parity check against the original fairseq code is what proves the two compute
the same thing; this module only gets the weights across.

The names, from fairseq 0.12's ``wav2vec2.py``:

- ``feature_extractor.conv_layers.{i}.0`` is the convolution. Its norm sits at ``.2`` (a
  group norm on layer 0 only, ``extractor_mode="default"``) or at ``.2.1`` (a layer norm
  between two transposes on every layer, ``extractor_mode="layer_norm"``, as in XLS-R).
- ``layer_norm`` normalises the conv features, ``post_extract_proj`` projects them.
- ``encoder.pos_conv.0`` is the weight-normed positional convolution (``weight_g``,
  ``weight_v``); torch now calls those ``parametrizations.weight.original0`` / ``original1``.
- ``encoder.layers.{i}``: ``self_attn.{q,k,v,out}_proj``, ``self_attn_layer_norm``, ``fc1``,
  ``fc2``, ``final_layer_norm``; ``encoder.layer_norm`` closes (or opens) the stack.
- ``mask_emb`` is transformers' ``masked_spec_embed``. ``quantizer``, ``project_q``,
  ``final_proj`` and ``label_embs_concat`` exist only for pretraining and are dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

_PRETRAINING_ONLY = ("quantizer.", "project_q.", "final_proj.", "label_embs_concat")

_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^feature_extractor\.conv_layers\.(\d+)\.0\.(weight|bias)$"),
        r"feature_extractor.conv_layers.\1.conv.\2",
    ),
    (
        re.compile(r"^feature_extractor\.conv_layers\.(\d+)\.2\.1\.(weight|bias)$"),
        r"feature_extractor.conv_layers.\1.layer_norm.\2",
    ),
    (
        re.compile(r"^feature_extractor\.conv_layers\.(\d+)\.2\.(weight|bias)$"),
        r"feature_extractor.conv_layers.\1.layer_norm.\2",
    ),
    (re.compile(r"^layer_norm\.(weight|bias)$"), r"feature_projection.layer_norm.\1"),
    (re.compile(r"^post_extract_proj\.(weight|bias)$"), r"feature_projection.projection.\1"),
    (re.compile(r"^mask_emb$"), "masked_spec_embed"),
    (
        re.compile(r"^encoder\.pos_conv\.0\.weight_g$"),
        "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
    ),
    (
        re.compile(r"^encoder\.pos_conv\.0\.weight_v$"),
        "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
    ),
    (re.compile(r"^encoder\.pos_conv\.0\.bias$"), "encoder.pos_conv_embed.conv.bias"),
    (re.compile(r"^encoder\.layer_norm\.(weight|bias)$"), r"encoder.layer_norm.\1"),
    (
        re.compile(r"^encoder\.layers\.(\d+)\.self_attn\.(q|k|v|out)_proj\.(weight|bias)$"),
        r"encoder.layers.\1.attention.\2_proj.\3",
    ),
    (
        re.compile(r"^encoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$"),
        r"encoder.layers.\1.layer_norm.\2",
    ),
    (
        re.compile(r"^encoder\.layers\.(\d+)\.fc1\.(weight|bias)$"),
        r"encoder.layers.\1.feed_forward.intermediate_dense.\2",
    ),
    (
        re.compile(r"^encoder\.layers\.(\d+)\.fc2\.(weight|bias)$"),
        r"encoder.layers.\1.feed_forward.output_dense.\2",
    ),
    (
        re.compile(r"^encoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$"),
        r"encoder.layers.\1.final_layer_norm.\2",
    ),
]


def hf_name(key: str) -> str | None:
    """The transformers name of one fairseq wav2vec 2.0 parameter, None if it's dropped.

    Raises:
        KeyError: for a name that is neither known nor pretraining-only, so a checkpoint
            with an unexpected layout fails loudly instead of loading half its weights.
    """
    if key.startswith(_PRETRAINING_ONLY):
        return None
    for pattern, replacement in _RULES:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    raise KeyError(f"unknown fairseq wav2vec 2.0 parameter {key!r}")


def fairseq_to_hf(state: Mapping[str, Any], prefix: str, new_prefix: str = "") -> dict[str, Any]:
    """Rename the wav2vec 2.0 part of a state dict; other keys pass through unchanged.

    Args:
        state: a detector's whole state dict.
        prefix: where the fairseq model sits in it (``"ssl_model.model."`` for XLS-R +
            AASIST); keys under it are renamed, pretraining-only ones dropped.
        new_prefix: where the ``Wav2Vec2Model`` sits in the adapter's network.
    """
    out: dict[str, Any] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            out[key] = value
            continue
        name = hf_name(key.removeprefix(prefix))
        if name is not None:
            out[new_prefix + name] = value
    return out


def check_config(config: Any, state: Mapping[str, Any], prefix: str = "") -> None:
    """Fail unless ``config`` describes the network ``state`` holds.

    Checks the layer count, hidden size, feed-forward size and conv stack, the things a
    wrong backbone config gets wrong; a strict load would catch them too, but later and
    with a less useful message.
    """
    layers = {
        int(m.group(1))
        for k in state
        if (m := re.match(re.escape(prefix) + r"encoder\.layers\.(\d+)\.", k))
    }
    hidden = state[prefix + "encoder.layer_norm.weight"].shape[0]
    ffn = state[prefix + "encoder.layers.0.feed_forward.intermediate_dense.weight"].shape[0]
    convs = len(
        {
            k
            for k in state
            if re.match(
                re.escape(prefix) + r"feature_extractor\.conv_layers\.\d+\.conv\.weight$", k
            )
        }
    )
    found = {
        "num_hidden_layers": len(layers),
        "hidden_size": hidden,
        "intermediate_size": ffn,
        "num_feat_extract_layers": convs,
    }
    wrong = {k: (getattr(config, k), v) for k, v in found.items() if getattr(config, k) != v}
    if wrong:
        raise ValueError(f"backbone config disagrees with the weights (config, weights): {wrong}")
