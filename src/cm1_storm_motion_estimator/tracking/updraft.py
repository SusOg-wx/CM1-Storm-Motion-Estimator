from __future__ import annotations

import numpy as np


def detect_updraft_regions(w_field: np.ndarray, threshold: float = 1.0, min_region_pixels: int = 10) -> list[np.ndarray]:
    """Return connected positive updraft regions in a 2D field."""
    if w_field.ndim != 2:
        raise ValueError("w_field must be a 2D array")

    mask = w_field > threshold
    labeled, num = _connected_components(mask)
    regions: list[np.ndarray] = []
    for label in range(1, num + 1):
        region = np.argwhere(labeled == label)
        if region.shape[0] >= min_region_pixels:
            regions.append(region)
    return regions


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    labels, num = ndimage.label(mask)
    return labels, num


def weighted_region_center(region: np.ndarray, field: np.ndarray) -> tuple[float, float]:
    """Compute a weighted centroid of a region in a 2D field."""
    if region.size == 0:
        raise ValueError("Region cannot be empty")

    values = field[region[:, 0], region[:, 1]]
    total = float(np.sum(values))
    if total <= 0:
        raise ValueError("Region total must be positive")

    y = np.average(region[:, 0], weights=values)
    x = np.average(region[:, 1], weights=values)
    return float(x), float(y)
