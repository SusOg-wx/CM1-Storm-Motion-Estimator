from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from cm1_storm_motion_estimator.tracking.vorticity import compute_relative_vorticity


def _field(values, x_name, x_dim, x, y_name, y_dim, y, units="m s-1"):
    return xr.DataArray(
        values,
        dims=(y_dim, x_dim),
        coords={
            y_name: xr.DataArray(y, dims=(y_dim,), attrs={"units": "m"}),
            x_name: xr.DataArray(x, dims=(x_dim,), attrs={"units": "m"}),
        },
        attrs={"units": units},
    )


def test_uniform_winds_have_zero_vorticity():
    x = np.arange(5.0)
    y = np.arange(4.0)
    u = _field(np.full((4, 5), 7.0), "xh", "xh", x, "yh", "yh", y)
    v = _field(np.full((4, 5), -2.0), "xh", "xh", x, "yh", "yh", y)
    result = compute_relative_vorticity(
        u, v,
        mass_x=xr.DataArray(x, dims="xh", attrs={"units": "m"}, name="xh"),
        mass_y=xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh"),
    )
    np.testing.assert_allclose(result.values, 0.0, atol=1e-14)
    assert result.attrs["units"] == "s-1"


def test_linear_wind_field_matches_analytic_vorticity_on_nonuniform_grid():
    x = np.array([0.0, 1.0, 3.0, 6.0, 10.0])
    y = np.array([0.0, 2.0, 5.0, 9.0])
    xx, yy = np.meshgrid(x, y)
    a, b = 0.003, 0.007
    u = _field(a * yy, "xh", "xh", x, "yh", "yh", y)
    v = _field(b * xx, "xh", "xh", x, "yh", "yh", y)
    result = compute_relative_vorticity(
        u, v,
        mass_x=xr.DataArray(x, dims="xh", attrs={"units": "m"}, name="xh"),
        mass_y=xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh"),
    )
    np.testing.assert_allclose(result.values, b - a, rtol=1e-12, atol=1e-14)


def test_staggered_components_interpolate_and_preserve_mass_coordinates():
    xh = np.array([1.0, 3.0, 6.0, 10.0])
    xf = np.array([0.0, 2.0, 4.5, 8.0, 12.0])
    yh = np.array([1.0, 4.0, 8.0])
    yf = np.array([0.0, 2.5, 6.0, 10.0])
    a, b = 0.002, 0.009
    u = _field(a * np.meshgrid(xf, yh)[1], "xf", "xf", xf, "yh", "yh", yh)
    v = _field(b * np.meshgrid(xh, yf)[0], "xh", "xh", xh, "yf", "yf", yf)
    target_x = xr.DataArray(xh, dims="xh", attrs={"units": "m"}, name="xh")
    target_y = xr.DataArray(yh, dims="yh", attrs={"units": "m"}, name="yh")
    result = compute_relative_vorticity(u, v, mass_x=target_x, mass_y=target_y)
    np.testing.assert_allclose(result.values, b - a, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(result.xh, xh)
    np.testing.assert_array_equal(result.yh, yh)
    assert result.dims == ("yh", "xh")


def test_coordinates_can_be_descending_and_unit_converted():
    x = np.array([4.0, 2.0, 0.0])
    y = np.array([3.0, 1.0, 0.0])
    xx, yy = np.meshgrid(x * 1000.0, y * 1000.0)
    u = _field(0.004 * yy, "xh", "xh", x, "yh", "yh", y)
    u.xh.attrs["units"] = "km"
    u.yh.attrs["units"] = "km"
    v = _field(0.010 * xx, "xh", "xh", x, "yh", "yh", y)
    v.xh.attrs["units"] = "km"
    v.yh.attrs["units"] = "km"
    mx = xr.DataArray(x, dims="xh", attrs={"units": "km"}, name="xh")
    my = xr.DataArray(y, dims="yh", attrs={"units": "km"}, name="yh")
    result = compute_relative_vorticity(u, v, mass_x=mx, mass_y=my)
    np.testing.assert_allclose(result.values, 0.006, atol=1e-14)


def test_missing_coordinate_units_rejected():
    x = np.arange(4.0)
    y = np.arange(3.0)
    u = _field(np.zeros((3, 4)), "xh", "xh", x, "yh", "yh", y)
    del u.xh.attrs["units"]
    v = _field(np.zeros((3, 4)), "xh", "xh", x, "yh", "yh", y)
    with pytest.raises(ValueError, match="Units are missing"):
        compute_relative_vorticity(
            u, v,
            mass_x=xr.DataArray(x, dims="xh", attrs={"units": "m"}, name="xh"),
            mass_y=xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh"),
        )


def test_nonmonotonic_and_missing_winds_rejected():
    x = np.array([0.0, 2.0, 1.0])
    y = np.arange(3.0)
    u = _field(np.zeros((3, 3)), "xh", "xh", x, "yh", "yh", y)
    v = _field(np.zeros((3, 3)), "xh", "xh", np.arange(3.0), "yh", "yh", y)
    mx = xr.DataArray(np.arange(3.0), dims="xh", attrs={"units": "m"}, name="xh")
    my = xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh")
    with pytest.raises(ValueError, match="strictly monotonic"):
        compute_relative_vorticity(u, v, mass_x=mx, mass_y=my)
    u = _field(np.zeros((3, 3)), "xh", "xh", np.arange(3.0), "yh", "yh", y)
    u.values[1, 1] = np.nan
    with pytest.raises(ValueError, match="missing or non-finite"):
        compute_relative_vorticity(u, v, mass_x=mx, mass_y=my)


def test_incompatible_target_grid_and_ambiguous_axes_rejected():
    x, y = np.arange(4.0), np.arange(3.0)
    u = _field(np.zeros((3, 4)), "xf", "xf", x, "yh", "yh", y)
    v = _field(np.zeros((3, 4)), "xh", "xh", x, "yf", "yf", y)
    mx = xr.DataArray(np.array([-1.0, 0.0, 1.0]), dims="xh", attrs={"units": "m"}, name="xh")
    my = xr.DataArray(y, dims="yh", attrs={"units": "m"}, name="yh")
    with pytest.raises(ValueError, match="extends beyond"):
        compute_relative_vorticity(u, v, mass_x=mx, mass_y=my)

    ambiguous = u.assign_coords(xh=("xf", np.arange(4.0)))
    with pytest.raises(ValueError, match="Cannot unambiguously identify"):
        compute_relative_vorticity(ambiguous, v, mass_x=mx, mass_y=my)
