from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt

from .config import SIGNAL_CONFIG


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low_cut_hz: float = SIGNAL_CONFIG.low_cut_hz,
    high_cut_hz: float = SIGNAL_CONFIG.high_cut_hz,
    order: int = SIGNAL_CONFIG.filter_order,
) -> np.ndarray:
    nyquist = 0.5 * sampling_rate
    low = low_cut_hz / nyquist
    high = high_cut_hz / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)


def magnitude(signal: np.ndarray) -> np.ndarray:
    return np.linalg.norm(signal, axis=1)


def compute_fft_features(signal_1d: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(signal_1d)
    freqs = np.fft.rfftfreq(len(signal_1d), d=1.0 / sampling_rate)
    amplitudes = np.abs(spectrum)
    return freqs, amplitudes

