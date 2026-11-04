from __future__ import annotations

import numpy as np
import xarray as xr

from cm1_storm_motion_estimator.data.netcdf_inspector import CM1DataInspector
from cm1_storm_motion_estimator.tracking.updraft import detect_updraft_regions, weighted_region_center
from cm1_storm_motion_estimator.tracking.vorticity import compute_relative_vorticity


def test_detect_updraft_regions_and_centroid() -> None:
    field = np.zeros((10, 10), dtype=float)
    field[3:7, 3:7] = 5.0
    regions = detect_updraft_regions(field, threshold=1.0, min_region_pixels=10)
    assert len(regions) >= 1
    region = regions[0]
    center = weighted_region_center(region, field)
    assert center[0] > 0
    assert center[1] > 0


def test_relative_vorticity_uses_physical_coordinates() -> None:
    x = np.linspace(0.0, 10.0, 20)
    y = np.linspace(0.0, 10.0, 20)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    u = xr.DataArray(
        np.zeros_like(yy), dims=("yh", "xh"),
        coords={
            "yh": xr.DataArray(y, dims="yh", attrs={"units": "m"}),
            "xh": xr.DataArray(x, dims="xh", attrs={"units": "m"}),
        }, attrs={"units": "m s-1"},
    )
    v = xr.DataArray(
        2.0 * xx, dims=("yh", "xh"),
        coords={
            "yh": xr.DataArray(y, dims="yh", attrs={"units": "m"}),
            "xh": xr.DataArray(x, dims="xh", attrs={"units": "m"}),
        }, attrs={"units": "m s-1"},
    )
    vort = compute_relative_vorticity(u, v, mass_x=u.xh, mass_y=u.yh)
    np.testing.assert_allclose(vort.values, 2.0, atol=1e-12)
    assert vort.attrs["units"] == "s-1"


def test_grid_spacing_detection() -> None:
    arr = np.array([0.0, 1.0, 2.0, 3.0])
    assert CM1DataInspector.detect_grid_spacing(arr) == 1.0
