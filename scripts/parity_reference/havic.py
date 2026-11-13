"""Reference scores for HAVIC, in upstream's own environment.

Runs in a virtualenv with upstream's ``requirements.txt`` versions (torch,
torchvision and torchaudio 2.3.1 on CPU, timm 0.4.5, numpy 1.24.4, pillow 10.4.0, einops 0.8.0; Python 3.10, which upstream doesn't name),
not in this repo's environment, and doesn't import this package. It follows
``evaluation/sliding_window_infer.py`` from the face crops on: ``HAVIC_FT()`` loaded as it loads
it (``strict=False``, the missing and unexpected keys recorded), each exported face crop through
``T.Resize((224, 224))`` and ``T.ToTensor()`` on a PIL image, the frames trimmed so windows of 16
at stride 2 tile them, the audio cut per window with the script's own arithmetic, centred, and
turned into 128-bin Kaldi filterbanks stretched to 1,024 frames and normalised. P(fake) is the
sigmoid of each window's logit (upstream averages the logits per video).

timm 0.4.5 still takes ``qk_scale``, so the adapter's first patch isn't needed here. At this
commit ``HAVIC_FT.forward`` passes ``use_mask=False`` to encoders that don't accept it, which stops
upstream's own inference; the encoders here drop that argument, as the adapter's second patch
does (masking is already off with ``ids_keep=None``). The crops and the audio come from the
adapter, so this checks the network, weights and transforms, not upstream's RetinaFace crops.

Usage: <havic venv>/bin/python scripts/parity_reference/havic.py <run dir>
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np
import timm
import torch
import torchaudio
import torchvision.transforms as T
from PIL import Image

WINDOW, STRIDE = 16, 2

run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
sys.path.insert(0, spec["checkout"])
from src.models.HAVIC import HAVIC_FT  # noqa: E402

model = HAVIC_FT()
state_dict = torch.load(Path(spec["weights_dir"]) / "best_ft_model.pth", map_location="cpu", weights_only=True)
missing, unexpected = model.load_state_dict(state_dict, strict=False)
for encoder in (model.visual_encoder, model.audio_encoder):
    encoder.forward = lambda *a, use_mask=None, _forward=encoder.forward, **k: _forward(*a, **k)
model.eval()

tensor_process = T.Compose([T.Resize(size=(224, 224)), T.ToTensor()])
saved = np.load(run / "inputs.npz")


def faces_of(i):
    if f"{i}__faces__n" in saved:
        return [saved[f"{i}__faces__{j}"] for j in range(int(saved[f"{i}__faces__n"]))]
    return list(saved[f"{i}__faces"])


scores = []
for i in range(len(spec["items"])):
    face_frames_list = [tensor_process(Image.fromarray(f).convert("RGB")) for f in faces_of(i)]
    remainder = (len(face_frames_list) - WINDOW) % STRIDE
    if remainder != 0:
        face_frames_list = face_frames_list[:-remainder]
    frames_tensor = torch.stack(face_frames_list)
    n_windows = (len(face_frames_list) - WINDOW) // STRIDE + 1
    video_batch = torch.stack(
        [frames_tensor[s : s + WINDOW].permute(1, 0, 2, 3) for s in range(0, len(frames_tensor) - WINDOW + 1, STRIDE)]
    )

    waveform = torch.from_numpy(saved[f"{i}__audio"].astype(np.float32))[None]
    sr = int(saved[f"{i}__sample_rate"])
    total_frames = int(saved[f"{i}__n_sampled"])
    total_audio_len = int((len(face_frames_list) / total_frames) * waveform.size(1))
    segment_len = int((total_audio_len * WINDOW) / (WINDOW + STRIDE * (n_windows - 1)))
    stride = int((STRIDE / WINDOW) * segment_len)
    fbank_list = []
    for w in range(n_windows):
        start, end = w * stride, w * stride + segment_len
        if end > total_audio_len:
            end = total_audio_len
            start = max(0, end - segment_len)
        segment = waveform[:, int(start) : int(end)]
        segment = segment - segment.mean()
        fbank = torchaudio.compliance.kaldi.fbank(
            segment, htk_compat=True, sample_frequency=sr, use_energy=False,
            window_type="hanning", num_mel_bins=128, dither=0.0, frame_shift=10,
        )
        fbank = torch.nn.functional.interpolate(
            fbank.unsqueeze(0).transpose(1, 2), size=(1024,), mode="linear", align_corners=False
        ).transpose(1, 2).squeeze(0)
        fbank_list.append((fbank - (-6.9960)) / (3.1205))
    audio_batch = torch.stack(fbank_list)

    with torch.inference_mode():
        output = model(audio=audio_batch, video=video_batch, is_training=False)
    scores.append(torch.sigmoid(output.float()).reshape(-1).numpy())

np.save(run / "reference.npy", np.concatenate(scores).astype(np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "upstream HAVIC_FT and best_ft_model.pth as sliding_window_infer.py loads "
            "them, its resize and audio windows; encoders drop use_mask",
            "missing_keys": len(missing),
            "unexpected_keys": len(unexpected),
            "env": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torchaudio": torchaudio.__version__,
                "timm": timm.__version__,
            },
        },
        indent=2,
    )
)
print(f"reference scores for {len(spec['items'])} items in {run}; missing {len(missing)}, unexpected {len(unexpected)}")
