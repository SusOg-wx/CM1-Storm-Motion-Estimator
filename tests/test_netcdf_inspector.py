from __future__ import annotations

import numpy as np

from cm1_storm_motion_estimator.data.netcdf_inspector import CM1DataInspector


def test_inspect_file_metadata() -> None:
    import xarray as xr

    ds = xr.Dataset(
        data_vars={
            "u": (("time", "y", "x"), np.ones((2, 3, 4), dtype=float)),
            "v": (("time", "y", "x"), np.ones((2, 3, 4), dtype=float)),
            "w": (("time", "y", "x"), np.ones((2, 3, 4), dtype=float)),
            "prs": (("time", "y", "x"), np.ones((2, 3, 4), dtype=float)),
        },
        coords={
            "time": [0.0, 10.0],
            "xh": np.linspace(0.0, 30.0, 4),
            "yh": np.linspace(0.0, 20.0, 3),
        },
    )

    path = "./tmp_test.nc"
    ds.to_netcdf(path)
    info = CM1DataInspector.inspect_file(path)
    assert info["dimensions"]["time"] == 2
    assert "xh" in info["coordinate_names"]
    assert "u" in info["common_variables"]
    assert info["grid_spacing_x"] is not None
