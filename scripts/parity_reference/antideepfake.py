"""Reference scores for the AntiDeepfake checkpoints, with real fairseq.

Runs in the same throwaway Python 3.9 virtualenv as ``ssl_xlsr.py`` (torch 1.13, fairseq
0.12.2, plus safetensors), not in this repo's environment. It imports upstream's own
``models/W2V.py`` from the checkout, whose ``SSLModel`` builds fairseq's ``Wav2Vec2Model``
from ``W2V_configs.py``'s settings for ``input.variant``, and loads the Hub's
``model.safetensors`` (the download, not the converted file) with every key matching.

Each window is normalised as upstream's loader normalises an utterance
(``layer_norm(wav, wav.shape)``); upstream labels real 1 and fake 0, so P(fake) is the
softmax at ``input.fake_index`` (0).

Usage: <fairseq venv>/bin/python scripts/parity_reference/antideepfake.py <run dir>
"""

import json
import platform
import sys
from pathlib import Path

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
inputs = spec["input"]
sys.path.insert(0, spec["checkout"])
from models.W2V import Model  # noqa: E402  (upstream's package, from the checkout)

model = Model(inputs["variant"])
weights = Path(spec["weights_dir"])
checkpoint = next(p for p in weights.glob("*.safetensors") if p.name != "model.safetensors")
model.load_state_dict(load_file(str(checkpoint)))
model.eval()

fake_index = int(inputs.get("fake_index", 0))
units = np.load(run / "inputs.npz")["units"]
scores = []
with torch.no_grad():
    for unit in units:
        wav = torch.from_numpy(unit).float()
        wav = F.layer_norm(wav, wav.shape)
        out = model(wav.unsqueeze(0))
        scores.append(torch.softmax(out, dim=-1)[0, fake_index].item())
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": f"upstream models/W2V.py ({inputs['variant']}) and the Hub's "
            f"{checkpoint.name}, one window at a time",
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "fairseq": fairseq.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(units)} windows in {run}")
