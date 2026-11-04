from __future__ import annotations

import numpy as np
import xarray as xr

from cm1_storm_motion_estimator.data.netcdf_source import CM1DataSource


def _write_frame(path, time, reverse_dims=False):
    # One time, two heights, three y points, four x points.
    values = np.arange(1 * 2 * 3 * 4, dtype=float).reshape(1, 2, 3, 4)
    dims = ("time", "zh", "yh", "xh")
    if reverse_dims:
        dims = ("time", "zh", "xh", "yh")
        values = values.transpose(0, 1, 3, 2)
    ds = xr.Dataset(
        {"w": (dims, values, {"units": "m s-1"})},
        coords={
            "time": [time],
            "zh": ("zh", [500.0, 1500.0], {"units": "m"}),
            "yh": ("yh", [100.0, 200.0, 300.0], {"units": "m"}),
            "xh": ("xh", [10.0, 20.0, 30.0, 40.0], {"units": "m"}),
        },
    )
    ds.to_netcdf(path)


def test_read_plane_uses_physical_coordinates_and_arbitrary_dimension_order(tmp_path):
    path = tmp_path / "frame.nc"
    _write_frame(path, 12.0, reverse_dims=True)
    with CM1DataSource([path]) as source:
        plane = source.read_horizontal_field("w", 0, height=1490.0)
    assert plane.values.shape == (3, 4)
    np.testing.assert_array_equal(plane.x, [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(plane.y, [100.0, 200.0, 300.0])
    assert plane.selected_height == 1500.0
    assert plane.units == "m s-1"
    assert plane.time == 12.0


def test_multifile_frames_are_sorted_by_time(tmp_path):
    later = tmp_path / "a_later.nc"
    earlier = tmp_path / "z_earlier.nc"
    _write_frame(later, 20.0)
    _write_frame(earlier, 10.0)
    with CM1DataSource([later, earlier]) as source:
        assert source.times == [10.0, 20.0]
        first = source.read_horizontal_field("w", 0, height=500.0)
        assert first.time == 10.0


def test_vertical_selection_must_be_explicit(tmp_path):
    path = tmp_path / "frame.nc"
    _write_frame(path, 0.0)
    with CM1DataSource([path]) as source:
        try:
            source.read_horizontal_field("w", 0)
        except ValueError as exc:
            assert "provide height" in str(exc)
        else:
            raise AssertionError("Expected an explicit-height error")
