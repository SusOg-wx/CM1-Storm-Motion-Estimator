from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from cm1_storm_motion_estimator.data.netcdf_inspector import CM1DataInspector


def test_inspect_file_metadata(tmp_path: Path) -> None:
    ds = xr.Dataset(
        data_vars={
            "u": (("time", "yh", "xh"), np.ones((2, 3, 4), dtype=float)),
            "v": (("time", "yh", "xh"), np.ones((2, 3, 4), dtype=float)),
            "w": (("time", "yh", "xh"), np.ones((2, 3, 4), dtype=float)),
            "prs": (("time", "yh", "xh"), np.ones((2, 3, 4), dtype=float)),
        },
        coords={
            "time": [0.0, 10.0],
            "xh": ("xh", np.linspace(0.0, 30.0, 4)),
            "yh": ("yh", np.linspace(0.0, 20.0, 3)),
        },
    )
    path = tmp_path / "metadata.nc"
    ds.to_netcdf(path)

    info = CM1DataInspector.inspect_file(path)
    assert info["dimensions"]["time"] == 2
    assert "xh" in info["coordinate_names"]
    assert "u" in info["common_variables"]
    assert info["grid_spacing_x"] is not None
