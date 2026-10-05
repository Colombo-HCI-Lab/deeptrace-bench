"""The pure-PyTorch mamba-ssm stand-ins match transformers' Mamba mixer and the reference norms."""

from __future__ import annotations

import torch

from deeptrace_bench.models._mamba import Mamba, RMSNorm, rms_norm_fn, stand_in_modules


def test_the_mixer_matches_transformers_mamba():
    from transformers import MambaConfig
    from transformers.models.mamba.modeling_mamba import MambaMixer

    torch.manual_seed(0)
    config = MambaConfig(
        hidden_size=16,
        state_size=4,
        expand=2,
        conv_kernel=4,
        time_step_rank=3,
        use_bias=False,
        use_conv_bias=True,
    )
    theirs = MambaMixer(config, layer_idx=0).eval()
    ours = Mamba(16, d_state=4, d_conv=4, expand=2, dt_rank=3).eval()
    ours.load_state_dict(theirs.state_dict(), strict=True)  # the same names and shapes
    x = torch.randn(2, 11, 16)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), theirs.slow_forward(x), rtol=1e-4, atol=1e-5)


def test_rms_norm_prenorm_returns_the_normalised_sum_and_the_sum():
    x, residual = torch.randn(3, 8), torch.randn(3, 8)
    weight = torch.rand(8) + 0.5
    out, summed = rms_norm_fn(x, weight, None, residual=residual, prenorm=True, eps=1e-5)
    s = x + residual
    torch.testing.assert_close(summed, s)
    torch.testing.assert_close(
        out, s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5) * weight
    )
    norm = RMSNorm(8)
    assert norm.bias is None and norm.weight.shape == (8,)


def test_stand_in_modules_expose_what_upstreams_import():
    modules = stand_in_modules()
    simple = modules["mamba_ssm.modules.mamba_simple"]
    assert simple.Mamba is Mamba and hasattr(simple, "Block")
    assert modules["mamba_ssm"].ops.triton.layernorm.RMSNorm is RMSNorm
