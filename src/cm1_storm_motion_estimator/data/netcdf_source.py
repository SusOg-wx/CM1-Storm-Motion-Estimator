from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import xarray as xr


_COORDINATE_NAMES = {
    "x": ("xh", "xf"),
    "y": ("yh", "yf"),
    "z": ("zh", "zf"),
    "time": ("time",),
}


@dataclass(frozen=True)
class HorizontalField:
    """A 2-D field in native grid locations, ordered as (y, x).

    Coordinates are returned alongside values so callers cannot accidentally
    treat an x-face or y-face field as if it were on the mass grid.
    """

    values: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_coordinate: str
    y_coordinate: str
    units: str | None
    variable: str
    time: float | None
    time_units: str | None
    vertical_coordinate: str | None
    selected_height: float | None
    dimensions: tuple[str, str]


@dataclass(frozen=True)
class _Frame:
    dataset_index: int
    time_index: int | None
    time: float | None
    time_units: str | None


class CM1DataSource:
    """Lazy reader for one or more CM1 NetCDF output files.

    Files remain open until :meth:`close` is called. Fields are read only when
    requested. Multiple files are ordered by their numeric time coordinate if
    one is present; files without time coordinates retain the supplied order.
    The reader deliberately returns native staggered-grid coordinates rather
    than silently interpolating winds to mass points.
    """

    def __init__(self, paths: Iterable[str | Path]):
        paths = [Path(path) for path in paths]
        if not paths:
            raise ValueError("At least one NetCDF path is required")
        self._datasets: list[xr.Dataset] = []
        self.paths = paths
        try:
            for path in paths:
                self._datasets.append(
                    xr.open_dataset(path, decode_times=False, mask_and_scale=True)
                )
            self._frames = self._build_frame_index()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for dataset in self._datasets:
            dataset.close()
        self._datasets.clear()

    def __enter__(self) -> "CM1DataSource":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    @property
    def times(self) -> list[float | None]:
        return [frame.time for frame in self._frames]

    @property
    def variable_names(self) -> list[str]:
        names: set[str] = set()
        for dataset in self._datasets:
            names.update(dataset.data_vars)
        return sorted(names)

    def inspect(self) -> list[dict[str, Any]]:
        """Return per-file dimensions, variables, and variable attributes."""
        results = []
        for path, dataset in zip(self.paths, self._datasets):
            results.append({
                "path": str(path),
                "dimensions": dict(dataset.sizes),
                "variables": {
                    name: {
                        "dimensions": tuple(variable.dims),
                        "shape": tuple(variable.shape),
                        "attrs": dict(variable.attrs),
                    }
                    for name, variable in dataset.variables.items()
                },
                "global_attributes": dict(dataset.attrs),
            })
        return results

    def read_horizontal_field(
        self,
        variable_name: str,
        frame_index: int,
        *,
        height: float | None = None,
    ) -> HorizontalField:
        """Read one horizontal plane, selecting the nearest requested height.

        ``height`` uses the units of the dataset's vertical coordinate. If the
        variable has a vertical dimension and height is omitted, the method
        raises instead of choosing an undocumented level.
        """
        try:
            frame = self._frames[frame_index]
        except IndexError as exc:
            raise IndexError(f"Frame index {frame_index} is out of range") from exc
        dataset = self._datasets[frame.dataset_index]
        if variable_name not in dataset.variables:
            raise KeyError(f"Variable {variable_name!r} is absent from {self.paths[frame.dataset_index]}")

        variable = dataset[variable_name]
        selection: dict[str, int] = {}
        remaining_dims = list(variable.dims)

        time_dim = self._coordinate_dim(dataset, "time")
        if time_dim in remaining_dims:
            if frame.time_index is None:
                raise ValueError(f"Cannot determine time index for variable {variable_name!r}")
            selection[time_dim] = frame.time_index
            remaining_dims.remove(time_dim)

        vertical_dim, vertical_name = self._find_axis(dataset, remaining_dims, "z")
        selected_height = None
        if vertical_dim is not None:
            if height is None:
                raise ValueError("Variable has a vertical dimension; provide height to select a level")
            z_values = np.asarray(dataset[vertical_name].values, dtype=float)
            if z_values.ndim != 1:
                raise ValueError(f"Vertical coordinate {vertical_name!r} is not one-dimensional")
            level = int(np.nanargmin(np.abs(z_values - height)))
            selection[vertical_dim] = level
            selected_height = float(z_values[level])
            remaining_dims.remove(vertical_dim)

        x_dim, x_name = self._find_axis(dataset, remaining_dims, "x")
        y_dim, y_name = self._find_axis(dataset, remaining_dims, "y")
        if x_dim is None or y_dim is None:
            raise ValueError(
                f"Cannot identify both horizontal axes for {variable_name!r}; dimensions are {variable.dims}"
            )

        other_dims = [dim for dim in remaining_dims if dim not in (x_dim, y_dim)]
        if other_dims:
            raise ValueError(f"Unselected dimensions remain for {variable_name!r}: {other_dims}")

        selected = variable.isel(selection) if selection else variable
        values = np.asarray(selected.transpose(y_dim, x_dim).values, dtype=float)
        x = np.asarray(dataset[x_name].values, dtype=float)
        y = np.asarray(dataset[y_name].values, dtype=float)
        if values.shape != (y.size, x.size):
            raise ValueError(
                f"Coordinate/field shape mismatch for {variable_name!r}: "
                f"field {values.shape}, coordinates {(y.size, x.size)}"
            )
        return HorizontalField(
            values=values,
            x=x,
            y=y,
            x_coordinate=x_name,
            y_coordinate=y_name,
            units=variable.attrs.get("units"),
            variable=variable_name,
            time=frame.time,
            time_units=frame.time_units,
            vertical_coordinate=vertical_name,
            selected_height=selected_height,
            dimensions=(y_dim, x_dim),
        )

    def _build_frame_index(self) -> list[_Frame]:
        frames: list[_Frame] = []
        has_numeric_time = False
        for dataset_index, dataset in enumerate(self._datasets):
            time_name = self._coordinate_name(dataset, "time")
            if time_name is None:
                frames.append(_Frame(dataset_index, None, None, None))
                continue
            coordinate = dataset[time_name]
            if coordinate.ndim == 0:
                times = [float(coordinate.values)]
                time_dim = None
            elif coordinate.ndim == 1:
                times = np.asarray(coordinate.values, dtype=float).tolist()
                time_dim = coordinate.dims[0]
            else:
                raise ValueError(f"Time coordinate {time_name!r} must be scalar or one-dimensional")
            for time_index, value in enumerate(times):
                frames.append(_Frame(dataset_index, time_index if time_dim else None, float(value), coordinate.attrs.get("units")))
                has_numeric_time = True
        if has_numeric_time:
            if any(frame.time is None for frame in frames):
                raise ValueError("Cannot combine files with and without time coordinates")
            frames.sort(key=lambda frame: float(frame.time))
            times = [float(frame.time) for frame in frames]
            if len(set(times)) != len(times):
                raise ValueError("Duplicate time values found across input files")
        return frames

    @staticmethod
    def _coordinate_name(dataset: xr.Dataset, axis: str) -> str | None:
        for name in _COORDINATE_NAMES[axis]:
            if name in dataset.variables:
                return name
        return None

    @classmethod
    def _coordinate_dim(cls, dataset: xr.Dataset, axis: str) -> str | None:
        name = cls._coordinate_name(dataset, axis)
        if name is None or dataset[name].ndim != 1:
            return None
        return dataset[name].dims[0]

    @classmethod
    def _find_axis(
        cls, dataset: xr.Dataset, dimensions: list[str], axis: str
    ) -> tuple[str | None, str | None]:
        for name in _COORDINATE_NAMES[axis]:
            if name not in dataset.variables:
                continue
            coordinate = dataset[name]
            if coordinate.ndim == 1 and coordinate.dims[0] in dimensions:
                return coordinate.dims[0], name
        return None, None
