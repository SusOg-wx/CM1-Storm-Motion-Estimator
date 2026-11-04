from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _extract_dim_sizes(dataset: Any) -> Dict[str, int]:
    """Return dimension lengths from an xarray Dataset."""
    return {name: int(size) for name, size in dataset.sizes.items()}


def _detect_coordinate_names(variable_names: Iterable[str]) -> List[str]:
    names = set(variable_names)
    return [name for name in sorted(names) if name in {"xh", "yh", "zh", "xf", "yf", "zf", "time"}]


def _detect_common_variables(variable_names: Iterable[str]) -> List[str]:
    names = set(variable_names)
    return [name for name in sorted(names) if name in {"u", "v", "w", "th", "prs", "vort"}]


class CM1DataInspector:
    """Inspects CM1 NetCDF output using xarray's Dataset API."""

    COMMON_COORD_NAMES = {"xh", "yh", "zh", "xf", "yf", "zf", "time"}
    COMMON_VARIABLE_NAMES = {"u", "v", "w", "th", "prs", "vort"}

    @staticmethod
    def inspect_file(path: str | Path) -> Dict[str, Any]:
        import xarray as xr

        with xr.open_dataset(path, decode_times=False) as dataset:
            dims = _extract_dim_sizes(dataset)
            variable_names = list(dataset.data_vars)
            coord_names = _detect_coordinate_names(dataset.coords)
            common_vars = _detect_common_variables(variable_names)

            time_values = None
            if "time" in dataset.coords:
                time_values = np.asarray(dataset["time"].values, dtype=float)

            grid_spacing_x = None
            if "xh" in dataset.coords:
                grid_spacing_x = CM1DataInspector.detect_grid_spacing(dataset["xh"].values)
            elif "xf" in dataset.coords:
                grid_spacing_x = CM1DataInspector.detect_grid_spacing(dataset["xf"].values)

            grid_spacing_y = None
            if "yh" in dataset.coords:
                grid_spacing_y = CM1DataInspector.detect_grid_spacing(dataset["yh"].values)
            elif "yf" in dataset.coords:
                grid_spacing_y = CM1DataInspector.detect_grid_spacing(dataset["yf"].values)

            return {
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

    @staticmethod
    def inspect_files(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
        return [CM1DataInspector.inspect_file(path) for path in paths]

    @staticmethod
    def detect_grid_spacing(values: np.ndarray) -> float | None:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 1 or arr.size < 2 or not np.all(np.isfinite(arr)):
            return None
        diffs = np.diff(arr)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            return None
        return float(np.median(np.abs(diffs)))
