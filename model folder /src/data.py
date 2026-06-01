from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TremorDataset:
    signals: list[np.ndarray]
    labels: np.ndarray
    sampling_rate: float
    groups: np.ndarray


def load_dataset(
    path: Path,
    sampling_rate: float = 100.0,
    window_size: int = 150,
    step_size: int = 75,
    segment_size: int = 300,
) -> TremorDataset:
    df = pd.read_csv(path)
    columns = set(df.columns)

    if {"signal_id", "x", "y", "z", "label"}.issubset(columns):
        return load_grouped_signal_dataset(df, sampling_rate)

    if {"xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro", "label"}.issubset(columns):
        return load_windowed_imu_dataset(df, sampling_rate, window_size, step_size, segment_size)

    raise ValueError(
        "Unsupported dataset format. Expected either grouped signal columns "
        "['signal_id', 'x', 'y', 'z', 'label'] or IMU columns "
        "['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro', 'label']."
    )


def load_grouped_signal_dataset(df: pd.DataFrame, sampling_rate: float) -> TremorDataset:
    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    for signal_id, group in df.groupby("signal_id"):
        signal = group[["x", "y", "z"]].to_numpy(dtype=float)
        label = int(group["label"].iloc[0])
        signals.append(signal)
        labels.append(label)
        groups.append(f"signal_{signal_id}")

    return TremorDataset(
        signals=signals,
        labels=np.asarray(labels, dtype=int),
        sampling_rate=sampling_rate,
        groups=np.asarray(groups),
    )


def load_windowed_imu_dataset(
    df: pd.DataFrame,
    sampling_rate: float,
    window_size: int,
    step_size: int,
    segment_size: int,
) -> TremorDataset:
    imu_cols = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]
    signals: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []

    if segment_size < window_size:
        raise ValueError("segment_size must be greater than or equal to window_size")

    block_ids = (df["label"] != df["label"].shift()).cumsum()
    for block_index, (_, block) in enumerate(df.groupby(block_ids)):
        label = int(block["label"].iloc[0])
        values = block[imu_cols].to_numpy(dtype=float)
        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, step_size):
            window = values[start : start + window_size]
            signals.append(window)
            labels.append(label)
            segment_index = start // segment_size
            groups.append(f"block_{block_index}_segment_{segment_index}")

    return TremorDataset(
        signals=signals,
        labels=np.asarray(labels, dtype=int),
        sampling_rate=sampling_rate,
        groups=np.asarray(groups),
    )
