from __future__ import annotations

import numpy as np

from cm1_storm_motion_estimator.data.netcdf_inspector import CM1DataInspector
from cm1_storm_motion_estimator.tracking.updraft import detect_updraft_regions, weighted_region_center
from cm1_storm_motion_estimator.tracking.vorticity import compute_relative_vorticity, detect_vorticity_regions


def test_detect_updraft_regions_and_centroid() -> None:
    field = np.zeros((10, 10), dtype=float)
    field[3:7, 3:7] = 5.0
    regions = detect_updraft_regions(field, threshold=1.0, min_region_pixels=10)
    assert len(regions) >= 1
    region = regions[0]
    center = weighted_region_center(region, field)
    assert center[0] > 0
    assert center[1] > 0


def test_relative_vorticity_and_vorticity_regions() -> None:
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 10, 20)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u = yy
    v = xx
    vort = compute_relative_vorticity(u, v, dx=1.0, dy=1.0)
    assert vort.shape == u.shape
    regions = detect_vorticity_regions(vort, threshold=0.05, min_region_pixels=10, sign=1)
    assert isinstance(regions, list)


def test_grid_spacing_detection() -> None:
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    assert CM1DataInspector.detect_grid_spacing(arr) == 1.0
