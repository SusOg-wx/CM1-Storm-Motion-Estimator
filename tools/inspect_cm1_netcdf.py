#!/usr/bin/env python3
"""Read-only summary of CM1-style NetCDF coordinates and wind-grid layout.

Usage: python tools/inspect_cm1_netcdf.py FILE.nc [FILE2.nc ...]

This utility does not write to or alter input files. Coordinate arrays are
printed in full by default because they are normally short grid axes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

AXES = {
    "x": ("xh", "xf"),
    "y": ("yh", "yf"),
    "z": ("zh", "zf"),
    "time": ("time",),
}
WIND_NAMES = ("u", "v", "w")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _associated_coordinate(dataset: xr.Dataset, variable: xr.DataArray, axis: str) -> str | None:
    matches = []
    for name in AXES[axis]:
        if name in dataset.variables:
            coord = dataset[name]
            if coord.ndim == 1 and coord.dims[0] in variable.dims:
                matches.append(name)
    if len(matches) == 1:
        return matches[0]
    return None


def _spacing(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        return {"monotonic": False, "spacing": None, "uniform": None}
    diff = np.diff(values)
    monotonic = bool(np.all(diff > 0) or np.all(diff < 0))
    uniform = bool(monotonic and np.allclose(diff, diff[0], rtol=1e-6, atol=1e-10))
    return {
        "monotonic": monotonic,
        "spacing": float(np.median(np.abs(diff))) if monotonic else None,
        "uniform": uniform,
        "minimum_spacing": float(np.min(np.abs(diff))) if monotonic else None,
        "maximum_spacing": float(np.max(np.abs(diff))) if monotonic else None,
    }


def inspect_file(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"path": str(path), "error": None}
    try:
        with xr.open_dataset(path, decode_times=False, mask_and_scale=False) as ds:
            report["global_attributes"] = {key: _json_value(value) for key, value in ds.attrs.items()}
            report["dimensions"] = dict(ds.sizes)
            report["variables"] = {
                name: {
                    "dimensions": list(var.dims),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "attributes": {key: _json_value(value) for key, value in var.attrs.items()},
                }
                for name, var in ds.variables.items()
            }
            report["coordinates"] = {}
            for name, coord in ds.variables.items():
                if name in ds.coords or name in sum((list(v) for v in AXES.values()), []):
                    values = np.asarray(coord.values)
                    entry: dict[str, Any] = {
                        "dimensions": list(coord.dims),
                        "shape": list(coord.shape),
                        "units": coord.attrs.get("units"),
                        "attributes": {key: _json_value(value) for key, value in coord.attrs.items()},
                        "values": _json_value(values),
                    }
                    if values.ndim == 1 and values.dtype.kind in "iuf":
                        entry["grid_spacing"] = _spacing(values)
                    report["coordinates"][name] = entry

            wind_layout = {}
            for name in WIND_NAMES:
                if name not in ds.data_vars:
                    continue
                var = ds[name]
                associated = {axis: _associated_coordinate(ds, var, axis) for axis in ("x", "y", "z", "time")}
                labels = []
                for axis in ("x", "y"):
                    coord = associated[axis]
                    if coord is None:
                        labels.append(f"{axis}: ambiguous or not found")
                    elif coord.endswith("f"):
                        labels.append(f"{axis}: face/staggered candidate ({coord})")
                    else:
                        labels.append(f"{axis}: mass-grid candidate ({coord})")
                wind_layout[name] = {
                    "dimensions": list(var.dims),
                    "units": var.attrs.get("units"),
                    "associated_coordinates": associated,
                    "layout_assessment": "; ".join(labels),
                }
            report["wind_layout"] = wind_layout
            report["time_and_vertical"] = {
                axis: {
                    name: {
                        "dimensions": list(ds[name].dims),
                        "units": ds[name].attrs.get("units"),
                        "values": _json_value(np.asarray(ds[name].values)),
                    }
                    for name in candidates
                    if name in ds.variables
                }
                for axis, candidates in (("time", AXES["time"]), ("vertical", AXES["z"]))
            }
    except Exception as exc:  # include file-level errors in the machine-readable output
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="CM1 NetCDF output file(s), read only")
    args = parser.parse_args()
    print(json.dumps([inspect_file(path) for path in args.files], indent=2, default=_json_value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
