from __future__ import annotations

import numpy as np
import xarray as xr

from cm1_storm_motion_estimator.tracking.vorticity import compute_relative_vorticity


def test_reversed_dimension_order_preserves_x_y_axis_association():
    # u is stored as (xh, yh), while v is stored as (yf, xh). The values are
    # generated from physical coordinate values, so a silent transpose or axis
    # swap changes the expected sign/magnitude and fails this assertion.
    xh = np.array([0.0, 1.0, 3.0, 6.0])
    xf = np.array([-1.0, 0.5, 2.0, 4.5, 7.0])
    yh = np.array([0.0, 2.0, 5.0])
    yf = np.array([-1.0, 1.0, 3.5, 6.0])
    a, b = 0.004, 0.011

    u_values = np.broadcast_to(a * yh[None, :], (xh.size, yh.size)).copy()
    v_values = np.broadcast_to(b * xh[None, :], (yf.size, xh.size)).copy()
    u = xr.DataArray(
        u_values,
        dims=("xh", "yh"),
        coords={
            "xh": xr.DataArray(xh, dims="xh", attrs={"units": "m"}),
            "yh": xr.DataArray(yh, dims="yh", attrs={"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )
    v = xr.DataArray(
        v_values,
        dims=("yf", "xh"),
        coords={
            "xh": xr.DataArray(xh, dims="xh", attrs={"units": "m"}),
            "yf": xr.DataArray(yf, dims="yf", attrs={"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )
    mass_x = xr.DataArray(xh, dims="xh", attrs={"units": "m"}, name="xh")
    mass_y = xr.DataArray(yh, dims="yh", attrs={"units": "m"}, name="yh")

    zeta = compute_relative_vorticity(u, v, mass_x=mass_x, mass_y=mass_y)
    np.testing.assert_allclose(zeta.values, b - a, rtol=1e-12, atol=1e-14)
    assert zeta.dims == ("yh", "xh")


def test_swapped_coordinate_dimension_association_is_rejected():
    x = np.arange(4.0)
    y = np.arange(3.0)
    u = xr.DataArray(
        np.zeros((3, 4)),
        dims=("yh", "xh"),
        coords={
            "xh": xr.DataArray(x, dims="yh", attrs={"units": "m"}),
            "yh": xr.DataArray(y, dims="xh", attrs={"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )
    v = xr.DataArray(
        np.zeros((3, 4)),
        dims=("yh", "xh"),
        coords={
            "xh": xr.DataArray(x, dims="xh", attrs={"units": "m"}),
            "yh": xr.DataArray(y, dims="yh", attrs={"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )
    mass_x = xr.DataArray(x, dims="xh", attrs={"units": "m"}, name="xh")
    mass_y = xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh")
    try:
        compute_relative_vorticity(u, v, mass_x=mass_x, mass_y=mass_y)
    except ValueError as exc:
        assert "dimensions" in str(exc) or "shape" in str(exc)
    else:
        raise AssertionError("Mismatched coordinate-to-dimension association must be rejected")
