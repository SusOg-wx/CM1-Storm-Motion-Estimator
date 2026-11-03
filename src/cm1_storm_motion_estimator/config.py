from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CM1_COORD_NAMES = {
    "xh", "yh", "zh", "xf", "yf", "zf", "time",
}

CM1_VARIABLE_NAMES = {
    "u", "v", "w", "th", "prs", "vort",
}

DEFAULT_TRACKING = {
    "updraft_threshold": 1.0,
    "vorticity_threshold": 0.005,
    "min_region_pixels": 10,
    "max_displacement_m": 2000.0,
    "smoothing_window": 1,
}

@dataclass
class SimulationConfig:
    updraft_threshold: float = 1.0
    vorticity_threshold: float = 0.005
    min_region_pixels: int = 10
    max_displacement_m: float = 2000.0
    smoothing_window: int = 1
    selected_altitude_min: Optional[float] = None
    selected_altitude_max: Optional[float] = None
    analysis_start_time: Optional[float] = None
    analysis_end_time: Optional[float] = None


@dataclass
class FeatureTrack:
    times: List[float] = field(default_factory=list)
    x_positions: List[float] = field(default_factory=list)
    y_positions: List[float] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
