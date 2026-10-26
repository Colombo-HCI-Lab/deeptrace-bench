"""Telemetry records a session on any machine and sums sessions into the compute block."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from deeptrace_bench import telemetry
from deeptrace_bench.telemetry import Session, read_sessions, summarize_compute


def test_a_cpu_session_records_phases_without_a_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "_nvml", lambda: None)
    with Session(tmp_path, "cpu", "part-000of001") as session:
        session.mark("load_model")
        session.items = 3
        session.mark("score")
    [record] = read_sessions(tmp_path)
    assert record["completed"] and record["items"] == 3
    assert set(record["phases_s"]) == {"load_model", "score"}
    assert record["gpu"] == {} and record["env"]["device"] == "cpu"
    assert "host" not in json.dumps(record["env"])


def test_a_cuda_session_survives_cuda_never_starting(tmp_path, monkeypatch):
    # The first SPARK run crashed resetting peak memory before CUDA was initialised.
    monkeypatch.setattr(telemetry, "_nvml", lambda: None)
    with Session(tmp_path, "cuda", "p") as session:
        session.mark("score")
    [record] = read_sessions(tmp_path)
    assert record["completed"]
    assert "cuda_max_allocated_gb" in record


def test_a_failed_session_is_still_written(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "_nvml", lambda: None)
    try:
        with Session(tmp_path, "cpu", "p"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    [record] = read_sessions(tmp_path)
    assert not record["completed"] and "boom" in record["error"]


def test_sampler_stats_from_samples():
    sampler = telemetry.GpuSampler.__new__(telemetry.GpuSampler)
    sampler.samples = [(0.0, 50, 2.0, 100.0, 60), (3600.0, 100, 4.0, 300.0, 70)]
    sampler.energy_start_mj = None
    sampler._energy_mj = lambda: None
    stats = sampler.stats()
    assert stats["gpu_util_mean"] == 75 and stats["gpu_mem_used_max_gb"] == 4.0
    # an hour at an average of 200 W
    assert stats["energy_wh"] == 200.0 and stats["energy_source"] == "integrated"


def _session(items, score_s, energy_wh, samples, peak):
    return {
        "started_utc": "2026-10-27T00:00:00+00:00",
        "completed": True,
        "wall_s": score_s + 1,
        "phases_s": {"load_model": 1.0, "score": score_s},
        "items": items,
        "peak_rss_gb": 3.0,
        "cuda_max_allocated_gb": peak,
        "env": {"gpu": "NVIDIA GeForce RTX 3090"},
        "gpu": {
            "samples": samples,
            "gpu_util_mean": 50.0,
            "gpu_mem_used_max_gb": peak + 1,
            "power_mean_w": 200.0,
            "temp_max_c": 70.0,
            "energy_wh": energy_wh,
        },
    }


def test_compute_sums_sessions():
    scores = pd.DataFrame(
        {
            "status": ["ok", "ok", "ok", "too_short"],
            "n_windows": [2, 2, 1, 0],
            "load_s": [0.01, 0.02, 0.03, 0.01],
            "infer_s": [0.1, 0.1, 0.05, np.nan],
        }
    )
    sessions = [_session(2, 10.0, 1.0, 10, 1.5), _session(2, 10.0, 3.0, 30, 2.5)]
    out = summarize_compute(scores, sessions, window_s=4.0375)
    assert out["sessions"] == 2 and out["items_per_s"] == 0.2
    assert out["windows"] == 5 and out["windows_per_s"] == 20.0
    assert out["energy_wh"] == 4.0 and out["energy_wh_per_1k_items"] == 1000.0
    assert out["cuda_max_allocated_gb"] == 2.5 and out["phases_s"]["score"] == 20.0
    assert out["realtime_factor"] == round(5 * 4.0375 / 20, 2)
    assert out["env"]["gpu"].endswith("3090")


def test_compute_without_telemetry_is_marked_unmeasured():
    scores = pd.DataFrame({"status": ["ok"], "n_windows": [1], "score": [0.3]})
    out = summarize_compute(scores, [], window_s=None)
    assert not out["measured"] and out["items_timed"] == 0 and out["energy_wh"] is None
