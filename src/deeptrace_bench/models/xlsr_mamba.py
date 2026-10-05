"""XLSR-Mamba (Xiao and Das, 2024, swagshaw/XLSR-Mamba, MIT): Tak et al.'s fine-tuned XLS-R
300M front end with a bidirectional Mamba back end, trained on ASVspoof 2019 LA.

Upstream needs fairseq and ``mamba-ssm`` 1.1.4's compiled CUDA kernels. The adapter is the
XLS-R family's (``ssl_xlsr.py``: fairseq stood in by transformers) plus pure-PyTorch
stand-ins for the ``mamba_ssm`` names upstream's ``mamba_blocks.py`` imports
(``_mamba.py``). Upstream's own ``MixerModel`` then runs unchanged, including its fused
add-norm ending, which normalises the back end's input plus each direction's residual stream
rather than the last block's output; the stand-in norm functions keep that behaviour, as the
kernels do.

``Model(args, device)`` reads ``args.emb_size`` (144) and ``args.num_encoders`` (12, so six
Mamba blocks per direction) from ``input``. The released checkpoint is safetensors with the
front end under ``ssl_model.model.``. Input: 64,600-sample windows (upstream's evaluation
cut). Output: two logits, bona fide at index 1 as upstream scores ``batch_out[:, 1]``.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

from ._mamba import stand_in_modules
from ._upstream import load_module
from .ssl_xlsr import SSLXLSRDetector


class XLSRMambaDetector(SSLXLSRDetector):
    """XLSR-Mamba, its Mamba kernels replaced by pure PyTorch."""

    def extra_stubs(self) -> dict[str, Any]:
        stubs: dict[str, Any] = dict(stand_in_modules())
        assert self.upstream_dir is not None
        # model.py imports its sibling as a top-level module; load it against the stand-ins
        # under that name, for as long as the stubs are in place.
        import sys

        saved = {name: sys.modules.get(name) for name in stubs}
        sys.modules.update(stubs)
        try:
            stubs["mamba_blocks"] = load_module(
                Path(self.upstream_dir) / "mamba_blocks.py", f"{self.module_name}_mamba_blocks"
            )
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
        return stubs

    def model_args(self) -> Any:
        return types.SimpleNamespace(
            emb_size=int(self.config.input["emb_size"]),
            num_encoders=int(self.config.input["num_encoders"]),
        )
