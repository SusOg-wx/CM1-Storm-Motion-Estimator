from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def estimate_motion(times: Sequence[float], positions: Sequence[float], method: str = "linear") -> dict:
    """Return motion estimate and fit quality for a one-dimensional track."""
    if len(times) < 2 or len(positions) != len(times):
        raise ValueError("times and positions must have matching lengths and at least 2 points")

    t = np.asarray(times, dtype=float)
    x = np.asarray(positions, dtype=float)

    if method == "linear":
        slope, intercept = np.polyfit(t, x, 1)
        y_fit = slope * t + intercept
        residual = x - y_fit
        rms = float(np.sqrt(np.mean(residual ** 2)))
        return {"slope": float(slope), "intercept": float(intercept), "rms": rms, "fit": y_fit.tolist()}

    if method == "rolling":
        if len(t) < 3:
            raise ValueError("rolling regression requires at least 3 points")
        slopes = []
        for i in range(0, len(t) - 2):
            window_t = t[i : i + 3]
            window_x = x[i : i + 3]
            slope, _ = np.polyfit(window_t, window_x, 1)
            slopes.append(float(slope))
        return {"slope": float(np.nanmean(slopes)), "rolling_slopes": slopes}

    if method == "displacement":
        dt = float(t[-1] - t[0])
        if dt == 0:
            raise ValueError("time span must be nonzero")
        return {"slope": float((x[-1] - x[0]) / dt), "displacement": float(x[-1] - x[0])}

    raise ValueError(f"Unsupported motion method: {method}")
