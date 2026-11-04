from __future__ import annotations

import numpy as np


def _ensure_1d(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def interpolate_staggered_to_mass(field: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray, target_x: np.ndarray, target_y: np.ndarray) -> np.ndarray:
    """Interpolate a staggered-grid field onto a mass-grid point set.

    The implementation is regular-grid aware and maintains physically meaningful
    coordinates rather than assuming the field shares the same grid as the target.
    """
    field_array = np.asarray(field, dtype=float)
    x = _ensure_1d("x_coords", x_coords)
    y = _ensure_1d("y_coords", y_coords)
    tx = _ensure_1d("target_x", target_x)
    ty = _ensure_1d("target_y", target_y)

    if field_array.shape != (y.size, x.size):
        raise ValueError(
            f"Field shape {field_array.shape} does not match coordinate sizes {(y.size, x.size)}"
        )

    from scipy.interpolate import RegularGridInterpolator

    interpolator = RegularGridInterpolator((y, x), field_array, method="linear", bounds_error=False, fill_value=np.nan)
    points = np.stack(np.meshgrid(ty, tx, indexing="ij"), axis=-1).reshape(-1, 2)
    values = interpolator(points).reshape(ty.size, tx.size)
    return values


def compute_relative_vorticity(
    u: np.ndarray,
    v: np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    *,
    u_x: np.ndarray | None = None,
    u_y: np.ndarray | None = None,
    v_x: np.ndarray | None = None,
    v_y: np.ndarray | None = None,
    mass_x: np.ndarray | None = None,
    mass_y: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ζ = dv/dx - du/dy using a physically consistent staggered-grid strategy.

    If staggered wind coordinates are supplied, the routine interpolates each wind
    component onto the common mass grid before differentiating. Otherwise, the
    inputs are treated as already colocated on the same grid.
    """
    if u_x is not None and u_y is not None and v_x is not None and v_y is not None and mass_x is not None and mass_y is not None:
        u_grid = interpolate_staggered_to_mass(u, u_x, u_y, mass_x, mass_y)
        v_grid = interpolate_staggered_to_mass(v, v_x, v_y, mass_x, mass_y)
    else:
        if u.shape != v.shape:
            raise ValueError("u and v arrays must have matching shape when already colocated")
        u_grid = np.asarray(u, dtype=float)
        v_grid = np.asarray(v, dtype=float)

    if dx is None:
        raise ValueError("dx must be supplied for vorticity differentiation")
    if dy is None:
        raise ValueError("dy must be supplied for vorticity differentiation")

    dudy = np.gradient(u_grid, axis=0, edge_order=1) / dy
    dvdx = np.gradient(v_grid, axis=1, edge_order=1) / dx
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
    for label_index in range(1, num + 1):
        region = np.argwhere(labels == label_index)
        if region.shape[0] >= min_region_pixels:
            regions.append(region)
    return regions


def _connected_components(mask: np.ndarray):
    from scipy import ndimage

    return ndimage.label(mask)
