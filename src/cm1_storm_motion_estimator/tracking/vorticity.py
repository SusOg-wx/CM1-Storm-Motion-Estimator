from __future__ import annotations

import numpy as np


def compute_relative_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute vertical relative vorticity ζ = dv/dx - du/dy for a 2D field."""
    if u.shape != v.shape:
        raise ValueError("u and v arrays must have matching shape")

    dudy = np.gradient(u, axis=0) / dy
    dvdx = np.gradient(v, axis=1) / dx
    return dvdx - dudy


def detect_vorticity_regions(vorticity: np.ndarray, threshold: float = 0.005, min_region_pixels: int = 10, sign: int = 1) -> list[np.ndarray]:
    """Return connected regions of a given vorticity sign."""
    if vorticity.ndim != 2:
        raise ValueError("vorticity must be a 2D array")

    if sign == 1:
        mask = vorticity > threshold
    elif sign == -1:
        mask = vorticity < -threshold
    else:
        raise ValueError("sign must be 1 or -1")

    labels, num = _connected_components(mask)
    regions: list[np.ndarray] = []
    for label in range(1, num + 1):
        region = np.argwhere(labels == label)
        if region.shape[0] >= min_region_pixels:
            regions.append(region)
    return regions


def _connected_components(mask: np.ndarray):
    from scipy import ndimage

    return ndimage.label(mask)
