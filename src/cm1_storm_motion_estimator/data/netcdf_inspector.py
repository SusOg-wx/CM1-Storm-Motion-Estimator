from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _extract_dim_sizes(dataset: Any) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for name, dim in dataset.dimensions.items():
        dims[name] = len(dim)
    return dims


def _detect_coordinate_names(variable_names: Iterable[str]) -> List[str]:
    names = set(variable_names)
    recognized = [name for name in sorted(names) if name in {"xh", "yh", "zh", "xf", "yf", "zf", "time"}]
    return recognized


def _detect_common_variables(variable_names: Iterable[str]) -> List[str]:
    names = set(variable_names)
    recognized = [name for name in sorted(names) if name in {"u", "v", "w", "th", "prs", "vort"}]
    return recognized


class CM1DataInspector:
    """Inspects CM1 NetCDF output and returns metadata needed by the tracking engine."""

    COMMON_COORD_NAMES = {"xh", "yh", "zh", "xf", "yf", "zf", "time"}
    COMMON_VARIABLE_NAMES = {"u", "v", "w", "th", "prs", "vort"}

    @staticmethod
    def inspect_file(path: str | Path) -> Dict[str, Any]:
        import xarray as xr

        dataset = xr.open_dataset(path, decode_times=False)
        dims = _extract_dim_sizes(dataset)
        variable_names = list(dataset.data_vars)
        coord_names = _detect_coordinate_names(dataset.coords)
        common_vars = _detect_common_variables(variable_names)

        time_values = None
        if "time" in dataset.coords:
            time_values = np.asarray(dataset["time"].values, dtype=float)

        grid_spacing_x = None
        grid_spacing_y = None
        if "xh" in dataset.coords and "xf" in dataset.coords:
            xh = np.asarray(dataset["xh"].values, dtype=float)
            xf = np.asarray(dataset["xf"].values, dtype=float)
            if xh.size > 1:
                grid_spacing_x = float(np.median(np.diff(xh)))
            elif xf.size > 1:
                grid_spacing_x = float(np.median(np.diff(xf)))

        if "yh" in dataset.coords and "yf" in dataset.coords:
            yh = np.asarray(dataset["yh"].values, dtype=float)
            yf = np.asarray(dataset["yf"].values, dtype=float)
            if yh.size > 1:
                grid_spacing_y = float(np.median(np.diff(yh)))
            elif yf.size > 1:
                grid_spacing_y = float(np.median(np.diff(yf)))

        metadata = {
            "path": str(path),
            "dimensions": dims,
            "variables": variable_names,
            "coordinate_names": coord_names,
            "common_variables": common_vars,
            "attrs": {key: value for key, value in dataset.attrs.items()},
            "time": time_values,
            "grid_spacing_x": grid_spacing_x,
            "grid_spacing_y": grid_spacing_y,
        }

        dataset.close()
        return metadata

    @staticmethod
    def inspect_files(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
        return [CM1DataInspector.inspect_file(path) for path in paths]

    @staticmethod
    def detect_grid_spacing(values: np.ndarray) -> float | None:
        arr = np.asarray(values, dtype=float)
        if arr.size < 2:
            return None
        diffs = np.diff(arr)
        if np.allclose(diffs, diffs[0]):
            return float(diffs[0])
        return float(np.median(np.abs(diffs)))
