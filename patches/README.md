# Patches

Compatibility patches for upstream detector code, applied by `scripts/setup_models.py` right after checkout. One folder per upstream repo (named after the repo, e.g. `DeepfakeBench/`), patches applied in name order with `git apply`.

Keep patches to what running on current PyTorch needs (`torch.load` defaults, removed numpy aliases, hard-coded paths). Anything that changes model behaviour must be caught by the parity check (see `docs/models.md`). A patch added after a checkout exists only applies to a fresh checkout: delete `third_party/<repo>-<commit>/` and re-run.
