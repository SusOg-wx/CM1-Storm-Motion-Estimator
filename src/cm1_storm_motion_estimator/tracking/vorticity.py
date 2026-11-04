from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True)
class _Axis:
    values: np.ndarray
    units: str
    name: str
    dim: str


_LENGTH_TO_METERS = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "km": 1000.0,
    "cm": 0.01,
}
_WIND_TO_M_S = {
    "m s-1": 1.0,
    "m/s": 1.0,
    "ms-1": 1.0,
    "m s^-1": 1.0,
    "km h-1": 1.0 / 3.6,
    "km/h": 1.0 / 3.6,
}


def _unit_key(value: Any) -> str:
    if value is None:
        raise ValueError("Units are missing; coordinate and wind units must be explicit")
    return " ".join(str(value).strip().lower().replace("**", "^").split())


def _length_scale(units: Any, label: str) -> float:
    key = _unit_key(units)
    if key not in _LENGTH_TO_METERS:
        raise ValueError(f"Unsupported {label} units {units!r}; use m, km, or cm")
    return _LENGTH_TO_METERS[key]


def _wind_scale(units: Any, label: str) -> tuple[float, str]:
    key = _unit_key(units)
    if key not in _WIND_TO_M_S:
        raise ValueError(f"Unsupported {label} units {units!r}; expected a recognized speed unit")
    return _WIND_TO_M_S[key], "m s-1"


def _validate_axis(values: Any, name: str, units: Any, dim: str) -> _Axis:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"Coordinate {name!r} must be a one-dimensional array with at least two points")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Coordinate {name!r} contains non-finite values")
    delta = np.diff(array)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise ValueError(f"Coordinate {name!r} must be strictly monotonic with no duplicate points")
    _length_scale(units, name)
    return _Axis(array, str(units), name, dim)


def _native_axis(field: xr.DataArray, axis: str) -> _Axis:
    candidates = ("xh", "xf") if axis == "x" else ("yh", "yf")
    matches = []
    for name in candidates:
        if name not in field.coords:
            continue
        coordinate = field.coords[name]
        if coordinate.ndim == 1 and coordinate.dims[0] in field.dims:
            matches.append((name, coordinate))
    if len(matches) != 1:
        names = [name for name, _ in matches]
        raise ValueError(
            f"Cannot unambiguously identify {axis}-coordinate for {field.name or 'wind field'}; "
            f"found {names}. Attach exactly one native coordinate ({', '.join(candidates)})."
        )
    name, coordinate = matches[0]
    return _validate_axis(coordinate.values, name, coordinate.attrs.get("units"), coordinate.dims[0])


def _target_axis(value: Any, name: str, units: Any, dim: str | None) -> _Axis:
    if isinstance(value, xr.DataArray):
        if value.ndim != 1:
            raise ValueError(f"Target coordinate {name!r} must be one-dimensional")
        target_dim = value.dims[0]
        target_units = units if units is not None else value.attrs.get("units")
        target_name = value.name or name
        values = value.values
    else:
        if units is None:
            raise ValueError(f"Units for target coordinate {name!r} must be supplied")
        values = value
        target_dim = dim or name
        target_units = units
        target_name = name
    return _validate_axis(values, target_name, target_units, target_dim)


def _orient(field: xr.DataArray, y_axis: _Axis, x_axis: _Axis) -> np.ndarray:
    if field.ndim != 2 or set(field.dims) != {y_axis.dim, x_axis.dim}:
        raise ValueError(
            f"{field.name or 'Wind field'} dimensions {field.dims} do not match its native "
            f"coordinate dimensions {(y_axis.dim, x_axis.dim)}; select a single time and height first"
        )
    values = np.asarray(field.transpose(y_axis.dim, x_axis.dim).values, dtype=float)
    if values.shape != (y_axis.values.size, x_axis.values.size):
        raise ValueError(f"{field.name or 'Wind field'} shape is incompatible with its coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{field.name or 'Wind field'} contains missing or non-finite values")
    return values


def _ascending(axis: _Axis, values: np.ndarray, *, x_axis: bool) -> tuple[np.ndarray, np.ndarray]:
    if axis.values[0] < axis.values[-1]:
        return axis.values, values
    return axis.values[::-1], values[:, ::-1] if x_axis else values[::-1, :]


def _interpolate_to_mass(
    field: np.ndarray, source_y: _Axis, source_x: _Axis, target_y: _Axis, target_x: _Axis
) -> np.ndarray:
    sx = source_x.values * _length_scale(source_x.units, source_x.name)
    sy = source_y.values * _length_scale(source_y.units, source_y.name)
    tx = target_x.values * _length_scale(target_x.units, target_x.name)
    ty = target_y.values * _length_scale(target_y.units, target_y.name)
    # Convert source field axes to ascending order for RegularGridInterpolator.
    if sy[0] > sy[-1]:
        sy = sy[::-1]
        field = field[::-1, :]
    if sx[0] > sx[-1]:
        sx = sx[::-1]
        field = field[:, ::-1]
    interpolator = RegularGridInterpolator(
        (sy, sx), field, method="linear", bounds_error=False, fill_value=np.nan
    )
    yy, xx = np.meshgrid(ty, tx, indexing="ij")
    result = interpolator(np.column_stack((yy.ravel(), xx.ravel()))).reshape(ty.size, tx.size)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            "The requested mass grid extends beyond a wind component's native grid; "
            "linear extrapolation is not performed. Use the common in-domain grid."
        )
    return result


def compute_relative_vorticity(
    u: xr.DataArray,
    v: xr.DataArray,
    *,
    mass_x: xr.DataArray | np.ndarray,
    mass_y: xr.DataArray | np.ndarray,
    mass_x_units: str | None = None,
    mass_y_units: str | None = None,
    mass_x_dim: str | None = None,
    mass_y_dim: str | None = None,
) -> xr.DataArray:
    """Compute CM1 vertical relative vorticity on the supplied mass grid.

    Parameters
    ----------
    u, v:
        Two-dimensional xarray wind fields from one time and height. Their
        native horizontal coordinates must be named xh/xf and yh/yf and carry
        explicit coordinate units. Wind data must have recognized speed units.
    mass_x, mass_y:
        Target mass-grid coordinates, as one-dimensional DataArrays with units,
        or arrays accompanied by ``mass_x_units``/``mass_y_units``. Target
        coordinates must lie inside both wind fields' domains.

    Components are linearly interpolated in physical space to the target grid
    when their native coordinates differ from the target coordinates. Derivatives
    use NumPy's second-order nonuniform finite-difference formula (first order
    only when an axis has exactly two points). Coordinates are converted to metres
    and wind speeds to m s-1 internally. The returned DataArray is in s-1.
    Missing values are rejected rather than filled or silently masked.
    """
    if not isinstance(u, xr.DataArray) or not isinstance(v, xr.DataArray):
        raise TypeError("u and v must be xarray.DataArray objects carrying dimensions and coordinates")
    ux, uy = _native_axis(u, "x"), _native_axis(u, "y")
    vx, vy = _native_axis(v, "x"), _native_axis(v, "y")
    mx = _target_axis(mass_x, "mass_x", mass_x_units, mass_x_dim)
    my = _target_axis(mass_y, "mass_y", mass_y_units, mass_y_dim)

    u_values = _orient(u, uy, ux)
    v_values = _orient(v, vy, vx)
    u_scale, _ = _wind_scale(u.attrs.get("units"), "u")
    v_scale, _ = _wind_scale(v.attrs.get("units"), "v")
    u_values = u_values * u_scale
    v_values = v_values * v_scale

    u_mass = _interpolate_to_mass(u_values, uy, ux, my, mx)
    v_mass = _interpolate_to_mass(v_values, vy, vx, my, mx)

    x_m = mx.values * _length_scale(mx.units, mx.name)
    y_m = my.values * _length_scale(my.units, my.name)
    edge_order_x = 2 if x_m.size >= 3 else 1
    edge_order_y = 2 if y_m.size >= 3 else 1
    dvdx = np.gradient(v_mass, x_m, axis=1, edge_order=edge_order_x)
    dudy = np.gradient(u_mass, y_m, axis=0, edge_order=edge_order_y)
    zeta = dvdx - dudy

    return xr.DataArray(
        zeta,
        dims=(my.dim, mx.dim),
        coords={
            my.name: xr.DataArray(my.values, dims=(my.dim,), attrs={"units": my.units}),
            mx.name: xr.DataArray(mx.values, dims=(mx.dim,), attrs={"units": mx.units}),
        },
        name="vertical_relative_vorticity",
        attrs={
            "long_name": "vertical relative vorticity",
            "standard_name": "atmosphere_relative_vorticity",
            "units": "s-1",
            "definition": "zeta = dv/dx - du/dy",
            "horizontal_grid": "supplied common mass grid",
            "interpolation": "linear interpolation in physical coordinates; no extrapolation",
            "differentiation": "second-order nonuniform finite differences; first-order for two-point axes",
            "wind_input_units": f"u: {u.attrs.get('units')}; v: {v.attrs.get('units')}",
            "coordinate_input_units": f"x: {mx.units}; y: {my.units}",
        },
    )
