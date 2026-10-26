"""What a scoring session cost: time, memory, GPU load and energy, and the machine it ran on.

``Session`` wraps one invocation of ``scripts/score.py``. A background thread samples the GPU
through NVML once a second (utilisation, memory, power, temperature); energy comes from the
GPU's own counter where it has one, and from integrating the power samples otherwise. Peak
memory comes from torch's allocator and the process's peak RSS. Without NVML (a Mac, a CPU
run) the GPU fields stay empty and everything else is still recorded.

Each session writes ``telemetry/<prefix>-<utc>.json`` in the run directory, one file per
session, so shards that finish together never write the same file. ``evaluate.py`` sums them
into the summary's ``compute`` block.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SAMPLE_EVERY_S = 1.0


def _nvml():
    """The pynvml module, initialised, or None where there is no NVIDIA driver."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:  # ImportError, or NVMLError on a machine without the driver
        return None
    return pynvml


def _nvml_index(cuda_index: int) -> int:
    """NVML numbers every GPU on the machine; CUDA only the visible ones."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [v.strip() for v in visible.split(",") if v.strip()]
    if ids and all(v.isdigit() for v in ids) and cuda_index < len(ids):
        return int(ids[cuda_index])
    return cuda_index


def _cuda_index(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    return int(device.split(":")[1]) if ":" in device else 0


def environment(device: str) -> dict[str, Any]:
    """The software and hardware a session ran on (no host name: published runs carry this)."""
    import torch

    env: dict[str, Any] = {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "arch": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "gpu": None,
        "gpu_memory_gb": None,
        "driver": None,
    }
    index = _cuda_index(device)
    if index is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(index)
        env["gpu"] = props.name
        env["gpu_memory_gb"] = round(props.total_memory / 1e9, 1)
        env["compute_capability"] = f"{props.major}.{props.minor}"
        nvml = _nvml()
        if nvml is not None:
            env["driver"] = nvml.nvmlSystemGetDriverVersion()
    elif device == "mps":
        env["gpu"] = "Apple GPU (MPS)"
    return env


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def peak_rss_gb() -> float:
    """The process's peak resident memory (ru_maxrss is bytes on macOS, KiB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1e9 if sys.platform == "darwin" else 1e6), 3)


class GpuSampler(threading.Thread):
    """Samples one GPU through NVML until stopped; ``stats()`` summarises what it saw."""

    def __init__(self, nvml, index: int, every_s: float = SAMPLE_EVERY_S):
        super().__init__(daemon=True)
        self.nvml = nvml
        self.handle = nvml.nvmlDeviceGetHandleByIndex(index)
        self.every_s = every_s
        self.samples: list[tuple[float, float, float, float, float]] = []
        self._stop = threading.Event()
        self.energy_start_mj = self._energy_mj()

    def _energy_mj(self) -> int | None:
        try:
            return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
        except Exception:  # pre-Volta GPUs have no counter
            return None

    def run(self) -> None:
        nvml = self.nvml
        while not self._stop.is_set():
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                watts = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
                temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
                self.samples.append((time.monotonic(), util.gpu, mem.used / 1e9, watts, temp))
            except Exception as exc:  # a sampling hiccup must never stop scoring
                log.debug("nvml sample failed: %s", exc)
            self._stop.wait(self.every_s)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=5)

    def stats(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        columns = np.array(self.samples, dtype=float).T
        t, util, used, watts, temp = columns
        end = self._energy_mj()
        if self.energy_start_mj is not None and end is not None:
            energy_wh, source = (end - self.energy_start_mj) / 3.6e6, "counter"
        else:
            energy_wh = float(np.trapezoid(watts, t)) / 3600 if len(t) > 1 else 0.0
            source = "integrated"
        return {
            "samples": len(t),
            "gpu_util_mean": round(float(util.mean()), 1),
            "gpu_util_p95": round(float(np.percentile(util, 95)), 1),
            "gpu_mem_used_max_gb": round(float(used.max()), 3),
            "power_mean_w": round(float(watts.mean()), 1),
            "power_max_w": round(float(watts.max()), 1),
            "temp_max_c": round(float(temp.max()), 1),
            "energy_wh": round(energy_wh, 3),
            "energy_source": source,
        }


class Session:
    """Measures one scoring session; use as a context manager around loading and scoring.

    ``mark(name)`` records how long the phase since the previous mark took (``setup`` for
    everything before the first mark, then e.g. ``load_model``, ``score``).
    """

    def __init__(self, directory: Path, device: str, prefix: str):
        self.directory = directory
        self.device = device
        self.prefix = prefix
        self.started = datetime.now(UTC)
        self.phases: dict[str, float] = {}
        self.items = 0
        self.sampler: GpuSampler | None = None

    def __enter__(self) -> Session:
        self._t0 = self._last = time.perf_counter()
        index = _cuda_index(self.device)
        nvml = _nvml() if index is not None else None
        if nvml is not None:
            try:
                self.sampler = GpuSampler(nvml, _nvml_index(index))
                self.sampler.start()
            except Exception as exc:
                log.warning("no GPU sampling: %s", exc)
                self.sampler = None
        if index is not None:
            import torch

            torch.cuda.reset_peak_memory_stats(index)
        return self

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.phases[name] = round(now - self._last, 3)
        self._last = now

    def __exit__(self, exc_type, exc, tb) -> None:
        wall = time.perf_counter() - self._t0
        if self.sampler is not None:
            self.sampler.stop()
        record: dict[str, Any] = {
            "started_utc": self.started.isoformat(timespec="seconds"),
            "ended_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "completed": exc_type is None,
            "error": None if exc is None else f"{exc_type.__name__}: {exc}",
            "prefix": self.prefix,
            "wall_s": round(wall, 3),
            "phases_s": self.phases,
            "items": self.items,
            "peak_rss_gb": peak_rss_gb(),
            "env": environment(self.device),
            "gpu": self.sampler.stats() if self.sampler else {},
        }
        index = _cuda_index(self.device)
        if index is not None:
            import torch

            record["cuda_max_allocated_gb"] = round(torch.cuda.max_memory_allocated(index) / 1e9, 3)
            record["cuda_max_reserved_gb"] = round(torch.cuda.max_memory_reserved(index) / 1e9, 3)
        try:
            write_session(self.directory, record)
        except OSError as err:  # never let bookkeeping hide the scoring error
            log.warning("couldn't write telemetry: %s", err)


def write_session(directory: Path, record: dict[str, Any]) -> Path:
    """Write one session's telemetry next to the run's score parts."""
    folder = directory / "telemetry"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = record["started_utc"].replace(":", "").replace("-", "").replace("+0000", "Z")
    path = folder / f"{record['prefix']}-{stamp}-{os.getpid()}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(path)
    return path


def read_sessions(directory: Path) -> list[dict[str, Any]]:
    """Every session's telemetry in a run directory, oldest first."""
    folder = directory / "telemetry"
    if not folder.exists():
        return []
    sessions = [json.loads(p.read_text()) for p in folder.glob("*.json")]
    return sorted(sessions, key=lambda s: s["started_utc"])


def _ms(values: pd.Series) -> dict[str, float | None]:
    values = values.dropna()
    if values.empty:
        return {"mean": None, "p50": None, "p95": None}
    ms = values.to_numpy(dtype=float) * 1000
    return {
        "mean": round(float(ms.mean()), 2),
        "p50": round(float(np.percentile(ms, 50)), 2),
        "p95": round(float(np.percentile(ms, 95)), 2),
    }


def summarize_compute(
    scores: pd.DataFrame, sessions: list[dict[str, Any]], window_s: float | None
) -> dict[str, Any]:
    """The summary's ``compute`` block: what scoring this run cost, over all its sessions.

    Args:
        scores: the run's score parts (``load_s`` and ``infer_s`` are missing on parts
            written before timing existed; those items simply aren't timed).
        sessions: ``read_sessions`` of the run directory.
        window_s: seconds of audio per window for an audio model, None for video (where a
            window is one face frame).
    """
    timed = scores.reindex(columns=["status", "n_windows", "load_s", "infer_s"])
    ok = timed[timed["status"] == "ok"]
    windows = int(ok["n_windows"].sum()) if len(ok) else 0
    infer_total = float(ok["infer_s"].sum()) if ok["infer_s"].notna().any() else None
    score_s = sum(s["phases_s"].get("score", 0.0) for s in sessions)
    session_items = sum(s.get("items", 0) for s in sessions)
    gpu = [s["gpu"] for s in sessions if s.get("gpu")]
    weights = [g["samples"] for g in gpu]

    def weighted(key: str) -> float | None:
        if not gpu:
            return None
        return round(float(np.average([g[key] for g in gpu], weights=weights)), 1)

    def peak(key: str, source: list[dict[str, Any]] = sessions) -> float | None:
        values = [s[key] for s in source if s.get(key) is not None]
        return max(values) if values else None

    energy = sum(g["energy_wh"] for g in gpu) if gpu else None
    phases: dict[str, float] = {}
    for s in sessions:
        for name, seconds in s["phases_s"].items():
            phases[name] = round(phases.get(name, 0.0) + seconds, 3)
    media_s = windows * window_s if window_s else None
    return {
        "measured": bool(sessions),
        "sessions": len(sessions),
        "sessions_completed": sum(bool(s.get("completed")) for s in sessions),
        "wall_s": round(sum(s["wall_s"] for s in sessions), 1) if sessions else None,
        "phases_s": phases,
        "items_timed": int(timed["load_s"].notna().sum()),
        "items_per_s": round(session_items / score_s, 3) if score_s else None,
        "load_ms": _ms(timed["load_s"]),
        "infer_ms": _ms(ok["infer_s"]),
        "windows": windows,
        "window_unit": f"{window_s:.2f} s audio window" if window_s else "face frame",
        "windows_per_s": round(windows / infer_total, 2) if infer_total else None,
        # seconds of audio scored per second of scoring; above 1 is faster than real time
        "realtime_factor": round(media_s / score_s, 2) if media_s and score_s else None,
        "cuda_max_allocated_gb": peak("cuda_max_allocated_gb"),
        "cuda_max_reserved_gb": peak("cuda_max_reserved_gb"),
        "gpu_mem_used_max_gb": peak("gpu_mem_used_max_gb", gpu),
        "peak_rss_gb": peak("peak_rss_gb"),
        "gpu_util_mean": weighted("gpu_util_mean"),
        "power_mean_w": weighted("power_mean_w"),
        "temp_max_c": peak("temp_max_c", gpu),
        "energy_wh": round(energy, 3) if energy is not None else None,
        "energy_wh_per_1k_items": (
            round(energy / session_items * 1000, 3) if energy and session_items else None
        ),
        "env": sessions[-1]["env"] if sessions else None,
    }
