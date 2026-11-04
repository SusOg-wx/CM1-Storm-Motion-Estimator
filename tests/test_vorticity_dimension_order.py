from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cm1_storm_motion_estimator.tracking.vorticity import compute_relative_vorticity


def _wind(u_or_v: str, dims: tuple[str, str], x_dim: str, y_dim: str, x, y, values):
    x_name = "xh" if u_or_v == "x" else "xh"
    y_name = "yh"
    return xr.DataArray(
        values,
        dims=dims,
        coords={
            x_name: xr.DataArray(x, dims=(x_dim,), attrs={"units": "m"}),
            y_name: xr.DataArray(y, dims=(y_dim,), attrs={"units": "m"}),
        },
        attrs={"units": "m s-1"},
    )


def test_reversed_dimension_order_preserves_x_y_axis_association():
    xh = np.array([0.0, 1.0, 3.0, 6.0])
    xf = np.array([-1.0, 0.5, 2.0, 4.5, 7.0])
    yh = np.array([0.0, 2.0, 5.0])
    yf = np.array([-1.0, 1.0, 3.5, 6.0])
    a, b = 0.004, 0.011
    u_values = np.broadcast_to(a * yh[None, :], (xh.size, yh.size)).copy()
    v_values = np.broadcast_to(b * xh[None, :], (yf.size, xh.size)).copy()
    u = xr.DataArray(u_values, dims=("xh", "yh"), coords={
        "xh": xr.DataArray(xh, dims="xh", attrs={"units": "m"}),
        "yh": xr.DataArray(yh, dims="yh", attrs={"units": "m"}),
    }, attrs={"units": "m s-1"})
    v = xr.DataArray(v_values, dims=("yf", "xh"), coords={
        "xh": xr.DataArray(xh, dims="xh", attrs={"units": "m"}),
        "yf": xr.DataArray(yf, dims="yf", attrs={"units": "m"}),
    }, attrs={"units": "m s-1"})
    mass_x = xr.DataArray(xh, dims="xh", attrs={"units": "m"}, name="xh")
    mass_y = xr.DataArray(yh, dims="yh", attrs={"units": "m"}, name="yh")
    zeta = compute_relative_vorticity(u, v, mass_x=mass_x, mass_y=mass_y)
    np.testing.assert_allclose(zeta.values, b - a, rtol=1e-12, atol=1e-14)
    assert zeta.dims == ("yh", "xh")


def test_swapped_coordinate_dimension_association_is_rejected_by_application():
    x = np.arange(4.0)
    y = np.arange(3.0)
    u = xr.DataArray(np.zeros((4, 3)), dims=("yh", "xh"), coords={
        "xh": xr.DataArray(x, dims="yh", attrs={"units": "m"}),
        "yh": xr.DataArray(y, dims="xh", attrs={"units": "m"}),
    }, attrs={"units": "m s-1"})
    v = xr.DataArray(np.zeros((3, 4)), dims=("yh", "xh"), coords={
        "xh": xr.DataArray(x, dims="xh", attrs={"units": "m"}),
        "yh": xr.DataArray(y, dims="yh", attrs={"units": "m"}),
    }, attrs={"units": "m s-1"})
    mass_x = xr.DataArray(x, dims="xh", attrs={"units": "m"}, name="xh")
    mass_y = xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh")
    with pytest.raises(ValueError, match="Incompatible coordinate association"):
        compute_relative_vorticity(u, v, mass_x=mass_x, mass_y=mass_y)


def test_opaque_dimension_names_allow_valid_axis_mapping_and_reversed_order():
    x = np.array([0.0, 1.0, 3.0, 6.0])
    y = np.array([0.0, 2.0, 5.0])
    a, b = 0.004, 0.011
    # Data dimensions are intentionally named independently of axis names and
    # appear as (column, row), not conventional (y, x).
    u = xr.DataArray(
        np.broadcast_to(a * y[None, :], (x.size, y.size)).copy(),
        dims=("column", "row"),
        coords={
            "xh": xr.DataArray(x, dims="column", attrs={"units": "m"}),
            "yh": xr.DataArray(y, dims="row", attrs={"units": "m"}),
        }, attrs={"units": "m s-1"},
    )
    v = xr.DataArray(
        np.broadcast_to(b * x[:, None], (x.size, y.size)).T.copy(),
        dims=("row", "column"),
        coords={
            "xh": xr.DataArray(x, dims="column", attrs={"units": "m"}),
            "yh": xr.DataArray(y, dims="row", attrs={"units": "m"}),
        }, attrs={"units": "m s-1"},
    )
    mass_x = xr.DataArray(x, dims="x_mass", attrs={"units": "m"}, name="xh")
    mass_y = xr.DataArray(y, dims="y_mass", attrs={"units": "m"}, name="yh")
    zeta = compute_relative_vorticity(u, v, mass_x=mass_x, mass_y=mass_y)
    np.testing.assert_allclose(zeta.values, b - a, rtol=1e-12, atol=1e-14)
