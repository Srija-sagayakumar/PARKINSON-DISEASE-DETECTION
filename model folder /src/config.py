from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalConfig:
    sampling_rate: float = 50.0
    low_cut_hz: float = 3.0
    high_cut_hz: float = 7.0
    filter_order: int = 4


SIGNAL_CONFIG = SignalConfig()
