from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TremorDataset
from .signal_processing import bandpass_filter, compute_fft_features, magnitude


def extract_features_from_signal(signal: np.ndarray, sampling_rate: float) -> dict[str, float]:
    filtered = bandpass_filter(signal, sampling_rate)
    features: dict[str, float] = {}

    channel_names = _channel_names(signal.shape[1])
    for idx, name in enumerate(channel_names):
        channel = filtered[:, idx]
        features[f"{name}_mean"] = float(np.mean(channel))
        features[f"{name}_std"] = float(np.std(channel))
        features[f"{name}_rms"] = float(np.sqrt(np.mean(np.square(channel))))
        features[f"{name}_max_abs"] = float(np.max(np.abs(channel)))

    if signal.shape[1] >= 3:
        acc_mag = magnitude(filtered[:, :3])
        features.update(_spectral_features(acc_mag, sampling_rate, "acc_mag"))

    if signal.shape[1] >= 6:
        gyro_mag = magnitude(filtered[:, 3:6])
        features.update(_spectral_features(gyro_mag, sampling_rate, "gyro_mag"))

    return features


def build_feature_table(dataset: TremorDataset) -> tuple[pd.DataFrame, np.ndarray]:
    rows = [extract_features_from_signal(signal, dataset.sampling_rate) for signal in dataset.signals]
    return pd.DataFrame(rows), dataset.labels


def _spectral_features(signal_1d: np.ndarray, sampling_rate: float, prefix: str) -> dict[str, float]:
    freqs, amps = compute_fft_features(signal_1d, sampling_rate)
    if len(amps) <= 1:
        dominant_freq = 0.0
    else:
        dominant_idx = int(np.argmax(amps[1:]) + 1)
        dominant_freq = float(freqs[dominant_idx])

    tremor_band = (freqs >= 4.0) & (freqs <= 6.0)
    return {
        f"{prefix}_mean": float(np.mean(signal_1d)),
        f"{prefix}_std": float(np.std(signal_1d)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(signal_1d)))),
        f"{prefix}_dominant_frequency": dominant_freq,
        f"{prefix}_tremor_band_power": float(np.sum(np.square(amps[tremor_band]))),
        f"{prefix}_spectral_energy": float(np.sum(np.square(amps))),
    }


def _channel_names(num_channels: int) -> list[str]:
    known = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]
    if num_channels <= len(known):
        return known[:num_channels]
    return [f"channel_{idx}" for idx in range(num_channels)]
