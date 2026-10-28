"""Reference scores for AVH-Align, on the real AV-HuBERT and fairseq.

Runs in a virtualenv with AV-HuBERT's stack, not in this repo's environment, and doesn't
import this package: Python 3.8, torch 1.10.0 on CPU, the fairseq fork pinned as a submodule of
facebookresearch/av_hubert (installed editable), omegaconf 2.0.6, hydra-core 1.0.7,
python_speech_features, scikit-image and opencv (docs/src/content/docs/models.md says how to make
it). ``AVHUBERT_DIR`` names the clone's ``avhubert/`` folder.

It does what ``deepfake_feature_extraction.py`` and ``eval.py`` do, from the exported inputs
instead of files: AV-HuBERT loaded by fairseq's ``load_model_ensemble_and_task`` (its
``encoder.w2v_model``, the checkpoint fine-tuned for lip reading), each item's mouth frames
through AV-HuBERT's own ``utils.Normalize(0, 255)``, ``CenterCrop(88)`` and
``Normalize(mean, std)`` with the task's settings, the audio through the body of upstream's
``load_audio`` (26 log filterbanks, stacked four at a time, layer-normalised), both cut to the
shorter, ``extract_finetune`` once with only the audio and once with only the video, then
upstream's ``FusionModel`` on the L2-normalised features and ``logsumexp(-output)`` over frames,
through the sigmoid the adapter applies to make it a probability.
Both checkpoints are checked against their pinned sha256 before they are unpickled.

``DTB_PATH_MAP="<from>=<to>,..."`` rewrites the path prefixes in ``parity.json`` when the run
directory was written on another machine.

Usage: AVHUBERT_DIR=<av_hubert>/avhubert <av_hubert venv>/bin/python \
    scripts/parity_reference/avh_align.py <run dir>
"""

import hashlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

np.float = np.float64  # upstream's own compatibility shim (deepfake_feature_extraction.py)
np.int = np.int_


def mapped(path):
    for pair in filter(None, os.environ.get("DTB_PATH_MAP", "").split(",")):
        old, new = pair.split("=", 1)
        if path.startswith(old):
            return new + path[len(old) :]
    return path


run = Path(sys.argv[1]).resolve()
spec = json.loads((run / "parity.json").read_text())
weights_dir = Path(mapped(spec["weights_dir"]))
for name in ("self_large_vox_433h.pt", "AVH-Align_AV1M.pt"):
    if hashlib.sha256((weights_dir / name).read_bytes()).hexdigest() != spec["weights"][name]:
        sys.exit(f"{name} doesn't match its pinned sha256")

avhubert_dir = os.environ["AVHUBERT_DIR"]
sys.path.insert(0, avhubert_dir)
cwd = os.getcwd()
os.chdir(avhubert_dir)
# hubert.py imports its siblings by plain names only when the command line is empty (its own
# debug branch, the one that works outside a package); hide the run directory while it imports
argv, sys.argv = sys.argv, sys.argv[:1]
import hubert  # noqa: E402,F401
import hubert_asr  # noqa: E402,F401
import hubert_pretraining  # noqa: E402,F401  (registers AV-HuBERT's task and model with fairseq)
import utils as avhubert_utils  # noqa: E402

sys.argv = argv
from fairseq import checkpoint_utils  # noqa: E402
from python_speech_features import logfbank  # noqa: E402

models, _, task = checkpoint_utils.load_model_ensemble_and_task(
    [str(weights_dir / "self_large_vox_433h.pt")]
)
os.chdir(cwd)
model = models[0]
if hasattr(model, "decoder"):
    model = model.encoder.w2v_model
model.eval()
transform = avhubert_utils.Compose(
    [
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
    ]
)

module_spec = importlib.util.spec_from_file_location(
    "avh_align_model", Path(mapped(spec["checkout"])) / "model.py"
)
upstream = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(upstream)
fusion = upstream.FusionModel()
fusion.load_state_dict(
    torch.load(weights_dir / "AVH-Align_AV1M.pt", map_location="cpu")["state_dict"]
)
fusion.eval()


def load_audio(wav_data, sample_rate, stack_order_audio=4):
    """The body of upstream's load_audio, from a decoded wave (no starting silence)."""
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
    if len(audio_feats) % stack_order_audio != 0:
        pad = stack_order_audio - len(audio_feats) % stack_order_audio
        audio_feats = np.concatenate(
            [audio_feats, np.zeros((pad, audio_feats.shape[1]), dtype=audio_feats.dtype)]
        )
    audio_feats = audio_feats.reshape(-1, stack_order_audio * audio_feats.shape[1])
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    return audio_feats


data = np.load(run / "inputs.npz")
scores = []
with torch.no_grad():
    for i in range(len(spec["items"])):
        frames = transform(data[f"{i}__mouths"])
        frames = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0)
        audio = load_audio(data[f"{i}__audio"], int(data[f"{i}__sample_rate"]))
        audio = audio[None, :, :].transpose(1, 2)
        min_len = min(frames.shape[2], audio.shape[-1])
        frames, audio = frames[:, :, :min_len], audio[:, :, :min_len]
        f_audio, _ = model.extract_finetune({"video": None, "audio": audio}, None, None)
        f_video, _ = model.extract_finetune({"video": frames, "audio": None}, None, None)
        visual = f_video.squeeze(0)
        sound = f_audio.squeeze(0)
        visual = visual / torch.linalg.norm(visual, ord=2, dim=-1, keepdim=True)
        sound = sound / torch.linalg.norm(sound, ord=2, dim=-1, keepdim=True)
        output = fusion(visual, sound)
        score = torch.logsumexp(-output, dim=0).squeeze()
        scores.append(float(torch.sigmoid(score.double())))  # the adapter's monotone map
np.save(run / "reference.npy", np.asarray(scores, dtype=np.float64))
(run / "reference.json").write_text(
    json.dumps(
        {
            "method": "fairseq AV-HuBERT extract_finetune (audio-only, video-only), FusionModel",
            "env": {"python": platform.python_version(), "torch": torch.__version__},
        },
        indent=2,
    )
)
print(f"reference scores for {len(scores)} items in {run}")
