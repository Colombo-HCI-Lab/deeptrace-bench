"""Pure-PyTorch stand-ins for the parts of ``mamba-ssm`` 1.1.x that upstream detectors import.

``mamba-ssm`` needs compiled CUDA kernels (and ``causal-conv1d``), which the one environment
doesn't have. Its kernels compute documented functions, so this module writes those functions
out in plain PyTorch with the same module and parameter names, and an upstream file that
imports ``mamba_ssm`` can be loaded against them with ``_upstream.scoped_modules``:

- ``Mamba``: the selective state-space mixer of ``mamba_ssm/modules/mamba_simple.py``
  (``in_proj``, depthwise causal ``conv1d``, ``x_proj``, ``dt_proj``, ``A_log``, ``D``,
  ``out_proj``), with the scan run step by step as ``selective_scan_ref`` defines it.
- ``Block``: the pre-norm residual wrapper of the same file, both its plain and its fused
  add-norm paths.
- ``RMSNorm``, ``rms_norm_fn``, ``layer_norm_fn``: ``mamba_ssm/ops/triton/layernorm.py``'s
  reference semantics, including ``prenorm`` (return the normalised sum and the sum).

Inference only: no inference cache, no training-time initialisation. The parity check against
the CUDA kernels is what licenses the substitution.
"""

from __future__ import annotations

import math
import types
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None,
    eps: float,
    prenorm: bool,
    residual_in_fp32: bool,
    rms: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    dtype = x.dtype
    if residual is not None:
        x = (x.float() + residual.float()).to(dtype)
    summed = x.float() if residual_in_fp32 else x
    xf = x.float()
    if rms:
        normed = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    else:
        normed = F.layer_norm(xf, xf.shape[-1:], eps=eps)
    out = normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    out = out.to(dtype)
    return (out, summed) if prenorm else out


def rms_norm_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None = None,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    eps: float = 1e-6,
) -> Tensor | tuple[Tensor, Tensor]:
    """RMS-normalise ``x + residual``; with ``prenorm``, also return the sum."""
    return _norm(x, weight, bias, residual, eps, prenorm, residual_in_fp32, rms=True)


def layer_norm_fn(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    residual: Tensor | None = None,
    eps: float = 1e-6,
    prenorm: bool = False,
    residual_in_fp32: bool = False,
    is_rms_norm: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Layer-normalise (or RMS-normalise) ``x + residual``; with ``prenorm``, also the sum."""
    return _norm(x, weight, bias, residual, eps, prenorm, residual_in_fp32, rms=is_rms_norm)


class RMSNorm(nn.Module):
    """RMS normalisation with a weight and no bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, device: Any = None, dtype: Any = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.register_parameter("bias", None)

    def forward(
        self,
        x: Tensor,
        residual: Tensor | None = None,
        prenorm: bool = False,
        residual_in_fp32: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        return rms_norm_fn(x, self.weight, self.bias, residual, prenorm, residual_in_fp32, self.eps)


def selective_scan(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor,
    z: Tensor,
    delta_bias: Tensor,
) -> Tensor:
    """The selective scan, step by step (``selective_scan_ref`` with ``delta_softplus``).

    Shapes: ``u``, ``delta``, ``z`` (batch, d_inner, length); ``A`` (d_inner, d_state);
    ``B``, ``C`` (batch, d_state, length); ``D``, ``delta_bias`` (d_inner,).
    """
    dtype = u.dtype
    u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
    delta = F.softplus(delta + delta_bias[..., None].float())
    delta_a = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    delta_b_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    state = u.new_zeros(u.shape[0], u.shape[1], A.shape[1])
    ys = []
    for i in range(u.shape[2]):
        state = delta_a[:, :, i] * state + delta_b_u[:, :, i]
        ys.append(torch.einsum("bdn,bn->bd", state, C[:, :, i]))
    y = torch.stack(ys, dim=2) + u * D[..., None].float()
    return (y * F.silu(z.float())).to(dtype)


class Mamba(nn.Module):
    """The Mamba (S6) mixer, same names and shapes as ``mamba_simple.Mamba``."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: int | None = None,
        device: Any = None,
        dtype: Any = None,
    ):
        super().__init__()
        kw = {"device": device, "dtype": dtype}
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, **kw)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
            **kw,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False, **kw)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **kw)
        a = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **kw)

    def forward(self, hidden_states: Tensor, inference_params: Any = None) -> Tensor:
        if inference_params is not None:
            raise NotImplementedError("step-by-step decoding isn't supported")
        length = hidden_states.shape[1]
        xz = self.in_proj(hidden_states).transpose(1, 2)  # (batch, 2 * d_inner, length)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        dbl = self.x_proj(x.transpose(1, 2))  # (batch, length, dt_rank + 2 * d_state)
        dt, B, C = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = (dt @ self.dt_proj.weight.T).transpose(1, 2)  # bias goes in as delta_bias
        A = -torch.exp(self.A_log.float())
        y = selective_scan(
            x, dt, A, B.transpose(1, 2), C.transpose(1, 2), self.D, z, self.dt_proj.bias
        )
        return self.out_proj(y.transpose(1, 2))


class Block(nn.Module):
    """``mamba_simple.Block``: add the residual, normalise, mix; returns (mixed, residual)."""

    def __init__(
        self,
        dim: int,
        mixer_cls: Any,
        norm_cls: Any = nn.LayerNorm,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

    def forward(
        self, hidden_states: Tensor, residual: Tensor | None = None, inference_params: Any = None
    ) -> tuple[Tensor, Tensor]:
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = _norm(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual,
                self.norm.eps,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                rms=isinstance(self.norm, RMSNorm),
            )
        return self.mixer(hidden_states, inference_params=inference_params), residual


def stand_in_modules() -> dict[str, types.ModuleType]:
    """``mamba_ssm`` modules backed by this file, for ``_upstream.scoped_modules``."""
    names = [
        "mamba_ssm",
        "mamba_ssm.modules",
        "mamba_ssm.modules.mamba_simple",
        "mamba_ssm.ops",
        "mamba_ssm.ops.triton",
        "mamba_ssm.ops.triton.layernorm",
    ]
    modules = {name: types.ModuleType(name) for name in names}
    simple = modules["mamba_ssm.modules.mamba_simple"]
    simple.Mamba, simple.Block = Mamba, Block  # type: ignore[attr-defined]
    norms = modules["mamba_ssm.ops.triton.layernorm"]
    norms.RMSNorm = RMSNorm  # type: ignore[attr-defined]
    norms.rms_norm_fn = rms_norm_fn  # type: ignore[attr-defined]
    norms.layer_norm_fn = layer_norm_fn  # type: ignore[attr-defined]
    for name, module in modules.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)
    return modules
