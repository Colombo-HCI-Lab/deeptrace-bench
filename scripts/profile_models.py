"""Measure each model's size and inference cost: parameters, FLOPs, latency and memory.

Examples:
    uv run scripts/profile_models.py --device cuda             # every model with an adapter
    uv run scripts/profile_models.py aasist gend --device cuda --items 4

Works in the smoke namespace on sampled items of one evalset per modality, so the inputs are
real and go through the same loaders as scoring. Times only the detector's ``score`` call
(warm, after two calls), counts FLOPs on one item with torch's FlopCounterMode, and writes
``results/profile/<model>.json``: aggregates only, no item ids. Loader time is recorded
separately, since face detection and file reads depend on caches and storage, not the model.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from datetime import UTC, datetime

import torch
from torch.utils.flop_counter import FlopCounterMode

from deeptrace_bench import evalset as evalsets
from deeptrace_bench.models.base import load_detector, resolve_device, set_tf32
from deeptrace_bench.paths import PUBLISHED_RESULTS_DIR, use_namespace
from deeptrace_bench.preprocess import PreprocessError
from deeptrace_bench.registry import Modality, ModelConfig, Registry
from deeptrace_bench.score import build_loader

log = logging.getLogger("profile")

# One evalset per modality, each with a smoke sample in the store.
PROFILE_EVALSET = {
    Modality.AUDIO: "urdu_csalt",
    Modality.VIDEO: "hidf_video",
    Modality.AUDIO_VIDEO: "unidatapro_av",
}


def parameters(detector: object) -> list[torch.nn.Parameter]:
    """Parameters of every torch module the adapter holds, each once."""
    seen: dict[int, torch.nn.Parameter] = {}
    for value in vars(detector).values():
        for module in value if isinstance(value, list | tuple) else [value]:
            if isinstance(module, torch.nn.Module):
                seen.update({id(p): p for p in module.parameters()})
    return list(seen.values())


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def profile(model: ModelConfig, registry: Registry, device: str, n_items: int, repeats: int):
    """Profile one model; returns the summary dict."""
    evalset = registry.evalset(PROFILE_EVALSET[model.modality])
    loader, _ = build_loader(registry, model, device)
    items = evalsets.load_items(evalset)
    detector = load_detector(model)
    base = 0
    if device == "cuda":
        # The previous model's tensors may linger until collected; measure above them.
        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    detector.load(device)
    inputs, load_ms = [], []
    for _, row in items.iterrows():
        start = time.perf_counter()
        try:
            inputs.append(loader(row))
        except PreprocessError:
            continue
        load_ms.append(1000 * (time.perf_counter() - start))
        if len(inputs) == n_items:
            break
    if hasattr(loader, "close"):
        loader.close()
    if not inputs:
        raise RuntimeError(f"no item of {evalset.id} could be preprocessed")

    for _ in range(2):
        detector.score(inputs[0])
    item_ms, unit_ms, units = [], [], []
    for x in inputs:
        for _ in range(repeats):
            _sync(device)
            start = time.perf_counter()
            n = len(detector.score(x))
            _sync(device)
            ms = 1000 * (time.perf_counter() - start)
            item_ms.append(ms)
            unit_ms.append(ms / n)
        units.append(n)
    params = parameters(detector)
    # FlopCounterMode hooks tensors that require grad, which fails under inference_mode.
    for p in params:
        p.requires_grad_(False)
    with FlopCounterMode(display=False) as counter:
        n0 = len(detector.score(inputs[0]))
    flops = counter.get_total_flops()

    if model.modality == Modality.AUDIO:
        unit = "4 s audio window"
    elif model.input.get("inputs") == "mouths":
        unit = "mouth clip"
    else:
        unit = "clip" if max(units) == 1 else "face frame"
    gpu = None
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        gpu = f"{round(props.total_memory / 2**30)} GB GPU"
    return {
        "model": model.id,
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": device,
        "gpu": gpu,
        "precision": "tf32" if registry.eval.compute["tf32"] else "fp32",
        "torch": torch.__version__,
        "evalset": evalset.id,
        "items": len(inputs),
        "repeats": repeats,
        "unit": unit,
        "units_per_item": statistics.mean(units),
        "params": sum(p.numel() for p in params),
        "gflops_per_unit": flops / n0 / 1e9 if flops else None,
        "ms_per_unit": statistics.median(unit_ms),
        "ms_per_item": statistics.median(item_ms),
        "load_ms_per_item": statistics.median(load_ms),
        "peak_memory_gb": (
            (torch.cuda.max_memory_allocated() - base) / 2**30 if device == "cuda" else None
        ),
    }


def main() -> int:
    """Parse arguments and profile each model in turn."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", help="model ids (default: all with an adapter)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--items", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    use_namespace("smoke")

    registry = Registry.load()
    device = resolve_device(args.device)
    set_tf32(registry.eval.compute["tf32"])
    ids = args.models or [
        m.id for m in registry.models.values() if m.adapter and m.status not in ("test", "blocked")
    ]
    out = PUBLISHED_RESULTS_DIR / "profile"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'model':<24} {'params':>12} {'GFLOPs':>9} {'ms/unit':>9} {'unit'}")
    failed = []
    for model_id in ids:
        model = registry.model(model_id)
        try:
            summary = profile(model, registry, device, args.items, args.repeats)
        except Exception:
            log.exception("could not profile %s", model_id)
            failed.append(model_id)
            continue
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()
        # Write and rename: over a CIFS mount a rewrite in place can keep the old file's tail.
        tmp = out / f".{model_id}.json.tmp"
        tmp.write_text(json.dumps(summary, indent=2) + "\n")
        tmp.replace(out / f"{model_id}.json")
        gf = summary["gflops_per_unit"]
        print(
            f"{model_id:<24} {summary['params']:>12,} {gf if gf is None else round(gf, 1):>9} "
            f"{summary['ms_per_unit']:>9.2f} {summary['unit']}"
        )
    if failed:
        print("failed:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
