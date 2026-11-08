import numpy as np

from deeptrace_bench.eval import shortcut


def _tone(seconds: float, amp: float, rate: int = 16_000, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return amp * np.sin(2 * np.pi * freq * t)


def test_features_read_length_loudness_silence_and_spectrum():
    wave = np.concatenate([_tone(1.0, 0.5), np.zeros(16_000)])
    f = shortcut.trivial_features(wave, 16_000, 44_100)
    assert set(f) == set(shortcut.FEATURES)
    assert f["duration_s"] == 2.0 and f["original_sample_rate"] == 44_100
    assert -12.5 < f["rms_db"] < -11.5  # a 0.5 sine is -9 dB; half silence halves its power
    assert 0.4 < f["silence_ratio"] < 0.6
    assert 300 < f["spectral_rolloff_hz"] < 600  # a pure 440 Hz tone
    assert f["spectral_bandwidth_hz"] < 200


def test_short_and_silent_clips_still_get_features():
    assert shortcut.trivial_features(np.zeros(100), 16_000, 16_000)["rms_db"] == -120.0
    assert shortcut.trivial_features(_tone(0.01, 0.1), 16_000, 8_000)["duration_s"] == 0.01


def test_probe_finds_a_planted_shortcut_and_not_noise():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 200)
    noise = rng.normal(size=(400, len(shortcut.FEATURES)))
    assert abs(shortcut.probe_auc(noise, y) - 0.5) < 0.1
    planted = noise.copy()
    planted[:, -1] = np.where(y == 1, 16_000, 44_100)  # fakes at another sample rate
    assert shortcut.probe_auc(planted, y) > 0.99
    assert shortcut.feature_aucs(planted, y)["original_sample_rate"] == 1.0
