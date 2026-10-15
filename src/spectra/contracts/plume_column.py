"""Contract A: plume columns exported by an LES processing chain.

One file holds one case, one tracer and one source as a time series of
vertical columns on the native, uniform LES grid. The contract is model
agnostic; the MicroHH processing chain is one producer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from spectra.contracts.errors import ContractError

PLUME_COLUMN_CONTRACT = "spectra.PlumeColumn"
PLUME_COLUMN_VERSION = "1.0"
COLUMN_UNITS = "kg m-2"
EMISSION_RATE_UNITS = "kg s-1"
CONCENTRATION_UNITS = "kg m-3"
WIND_UNITS = "m s-1"
COLUMN_DIMS = ("time", "y", "x")
REQUIRED_ATTRS = (
    "gas",
    "les_model",
    "case_id",
    "tracer",
    "source_id",
    "source_x",
    "source_y",
    "source_z",
    "emission_rate_units",
    "dx",
    "dy",
    "reference_time",
    "column_convention",
)
_SPACING_RTOL = 1e-6


@dataclass(frozen=True)
class PlumeColumnInfo:
    """Summary of a validated plume column.

    Attributes:
        case_id: LES case identifier.
        tracer: Tracer name.
        source_id: Source identifier.
        gas: Gas name used for unit conversion.
        dx: Cell size along x in metres.
        dy: Cell size along y in metres.
        n_time: Number of snapshots.
        has_partial: Whether partial columns below altitudes are present.
        has_3d: Whether the three-dimensional concentration is present.
        source_xy: Source position in the model frame in metres.
        source_z: Source height above ground in metres.
        emission_rate_is_constant: Whether the emission rate is constant in time.
        support_bounds_m: Cell-edge bounds of the union of nonzero cells.
    """

    case_id: str
    tracer: str
    source_id: str
    gas: str
    dx: float
    dy: float
    n_time: int
    has_partial: bool
    has_3d: bool
    source_xy: tuple[float, float]
    source_z: float
    emission_rate_is_constant: bool
    support_bounds_m: tuple[float, float, float, float]


def _validate_axis(dataset: xr.Dataset, name: str) -> float:
    """Validate a uniform, increasing one-dimensional metre axis and return its step."""
    if name not in dataset.dims or name not in dataset.coords:
        raise ContractError(f"PlumeColumn must have a '{name}' dimension coordinate.")
    coordinate = dataset.coords[name]
    if coordinate.dims != (name,) or coordinate.size < 2:
        raise ContractError(
            f"PlumeColumn '{name}' must be one-dimensional with at least two cells."
        )
    values = np.asarray(coordinate.values, dtype=float)
    steps = np.diff(values)
    if not np.all(np.isfinite(values)) or np.any(steps <= 0):
        raise ContractError(f"PlumeColumn '{name}' must be finite and increasing.")
    if not np.allclose(steps, steps[0], rtol=_SPACING_RTOL, atol=1e-9):
        raise ContractError(f"PlumeColumn '{name}' must be uniformly spaced.")
    return float(steps[0])


def _require_units(variable: xr.DataArray, name: str, units: str) -> None:
    found = variable.attrs.get("units")
    if found != units:
        raise ContractError(
            f"PlumeColumn '{name}' must use units {units!r}; got {found!r}."
        )


def _require_dims(variable: xr.DataArray, name: str, dims: tuple[str, ...]) -> None:
    if tuple(variable.dims) != dims:
        raise ContractError(
            f"PlumeColumn '{name}' must have dimensions {dims}; got "
            f"{tuple(variable.dims)}."
        )


def plume_support_bounds(
    dataset: xr.Dataset, *, threshold: float = 0.0, variable: str = "column"
) -> tuple[float, float, float, float]:
    """Return cell-edge bounds of cells whose column exceeds ``threshold`` at any time.

    Args:
        dataset: Plume column dataset.
        threshold: Column value above which a cell belongs to the support.
        variable: Column variable name.

    Returns:
        ``(xmin, ymin, xmax, ymax)`` in model-frame metres.
    """
    column = dataset[variable]
    leading = [dim for dim in column.dims if dim not in {"x", "y"}]
    support = column.fillna(0.0) > threshold
    if leading:
        support = support.any(dim=leading)
    mask = np.asarray(support.transpose("y", "x").values, dtype=bool)
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ContractError(f"PlumeColumn '{variable}' has empty support.")
    x = np.asarray(dataset.coords["x"].values, dtype=float)
    y = np.asarray(dataset.coords["y"].values, dtype=float)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return (
        float(x[indices[:, 1].min()] - dx / 2),
        float(y[indices[:, 0].min()] - dy / 2),
        float(x[indices[:, 1].max()] + dx / 2),
        float(y[indices[:, 0].max()] + dy / 2),
    )


def validate_plume_column(
    dataset: xr.Dataset,
    *,
    require_partial: bool = False,
    require_3d: bool = False,
) -> PlumeColumnInfo:
    """Validate a dataset against the ``PlumeColumn`` contract.

    Args:
        dataset: Candidate dataset.
        require_partial: Require ``column_below_altitude``.
        require_3d: Require ``concentration``.

    Returns:
        Summary of the validated column.

    Raises:
        ContractError: If the dataset violates the contract.
    """
    contract = dataset.attrs.get("contract")
    if contract is not None and contract != PLUME_COLUMN_CONTRACT:
        raise ContractError(f"Dataset declares contract {contract!r}.")
    dx = _validate_axis(dataset, "x")
    dy = _validate_axis(dataset, "y")
    if "time" not in dataset.dims:
        raise ContractError("PlumeColumn must have a 'time' dimension.")
    time = dataset.coords.get("time")
    if time is None or not np.issubdtype(time.dtype, np.datetime64):
        raise ContractError("PlumeColumn 'time' must be a datetime64 coordinate.")
    if "simulation_time" not in dataset.coords:
        raise ContractError("PlumeColumn must carry a 'simulation_time' coordinate.")
    _require_dims(dataset["simulation_time"], "simulation_time", ("time",))

    missing = [name for name in REQUIRED_ATTRS if name not in dataset.attrs]
    if missing:
        raise ContractError(f"PlumeColumn is missing attributes: {missing}.")
    attrs = dataset.attrs
    if attrs["column_convention"] != "vertical":
        raise ContractError("PlumeColumn column_convention must be 'vertical'.")
    if attrs["emission_rate_units"] != EMISSION_RATE_UNITS:
        raise ContractError(
            f"PlumeColumn emission_rate_units must be {EMISSION_RATE_UNITS!r}."
        )
    for name, step in (("dx", dx), ("dy", dy)):
        if not np.isclose(float(attrs[name]), step, rtol=_SPACING_RTOL):
            raise ContractError(
                f"PlumeColumn attribute {name}={attrs[name]} disagrees with the "
                f"coordinate spacing {step}."
            )

    if "column" not in dataset:
        raise ContractError("PlumeColumn must contain a 'column' variable.")
    column = dataset["column"]
    _require_dims(column, "column", COLUMN_DIMS)
    _require_units(column, "column", COLUMN_UNITS)
    if not np.issubdtype(column.dtype, np.floating):
        raise ContractError("PlumeColumn 'column' must be floating point.")
    values = column.values
    if not np.all(np.isfinite(values)):
        raise ContractError("PlumeColumn 'column' must be finite everywhere.")
    if np.any(values < 0):
        raise ContractError("PlumeColumn 'column' must be non-negative.")

    if "emission_rate" not in dataset:
        raise ContractError("PlumeColumn must contain 'emission_rate'.")
    rate = dataset["emission_rate"]
    _require_dims(rate, "emission_rate", ("time",))
    _require_units(rate, "emission_rate", EMISSION_RATE_UNITS)
    rate_values = np.asarray(rate.values, dtype=float)
    if not np.all(np.isfinite(rate_values)) or np.any(rate_values < 0):
        raise ContractError("PlumeColumn 'emission_rate' must be finite and >= 0.")

    has_partial = "column_below_altitude" in dataset
    if require_partial and not has_partial:
        raise ContractError("PlumeColumn lacks the required 'column_below_altitude'.")
    if has_partial:
        partial = dataset["column_below_altitude"]
        _require_dims(partial, "column_below_altitude", ("time", "altitude", "y", "x"))
        _require_units(partial, "column_below_altitude", COLUMN_UNITS)
        if "altitude" not in dataset.coords:
            raise ContractError("PlumeColumn lacks the 'altitude' coordinate.")
        altitude = np.asarray(dataset.coords["altitude"].values, dtype=float)
        if np.any(np.diff(altitude) <= 0):
            raise ContractError("PlumeColumn 'altitude' must be strictly increasing.")

    has_3d = "concentration" in dataset
    if require_3d and not has_3d:
        raise ContractError("PlumeColumn lacks the required 'concentration'.")
    if has_3d:
        concentration = dataset["concentration"]
        _require_dims(concentration, "concentration", ("time", "z", "y", "x"))
        _require_units(concentration, "concentration", CONCENTRATION_UNITS)
        if "z" not in dataset.coords:
            raise ContractError("PlumeColumn 'concentration' needs a 'z' coordinate.")

    if ("wind_u" in dataset) != ("wind_v" in dataset):
        raise ContractError("PlumeColumn wind_u and wind_v must occur together.")
    if "wind_u" in dataset:
        for name in ("wind_u", "wind_v"):
            wind = dataset[name]
            if tuple(wind.dims) not in (("time",), COLUMN_DIMS):
                raise ContractError(
                    f"PlumeColumn '{name}' must have dimensions ('time',) or "
                    f"{COLUMN_DIMS}."
                )
            _require_units(wind, name, WIND_UNITS)

    support = plume_support_bounds(dataset)
    return PlumeColumnInfo(
        case_id=str(attrs["case_id"]),
        tracer=str(attrs["tracer"]),
        source_id=str(attrs["source_id"]),
        gas=str(attrs["gas"]),
        dx=dx,
        dy=dy,
        n_time=int(dataset.sizes["time"]),
        has_partial=has_partial,
        has_3d=has_3d,
        source_xy=(float(attrs["source_x"]), float(attrs["source_y"])),
        source_z=float(attrs["source_z"]),
        emission_rate_is_constant=bool(
            attrs.get("emission_rate_is_constant", bool(np.ptp(rate_values) == 0))
        ),
        support_bounds_m=support,
    )


def select_time(dataset: xr.Dataset, time_index: int) -> xr.Dataset:
    """Return one snapshot of a plume column with its time recorded in attributes.

    Args:
        dataset: Plume column dataset.
        time_index: Zero-based snapshot index.

    Returns:
        Dataset without the ``time`` dimension; ``time`` and
        ``simulation_time`` remain as scalar coordinates.
    """
    n_time = int(dataset.sizes["time"])
    if not 0 <= time_index < n_time:
        raise ContractError(f"time_index {time_index} outside [0, {n_time}).")
    snapshot = dataset.isel(time=time_index)
    snapshot.attrs.update(
        {
            "time_index": int(time_index),
            "simulation_time_s": float(snapshot["simulation_time"].values),
            "snapshot_time": np.datetime_as_string(
                snapshot["time"].values.astype("datetime64[s]"), unit="s"
            ),
        }
    )
    return snapshot


def scale_emission(dataset: xr.Dataset, factor: float) -> xr.Dataset:
    """Scale the plume by a positive factor, as a linear passive tracer allows.

    Args:
        dataset: Plume column dataset or snapshot.
        factor: Multiplicative factor applied to columns, concentrations and
            the emission rate.

    Returns:
        Scaled copy recording the cumulative ``emission_scale`` attribute.
    """
    if not np.isfinite(factor) or factor <= 0:
        raise ContractError("factor must be a positive finite number.")
    if dataset.attrs.get("passive_tracer") in (0, False, "false", "False"):
        raise ContractError("Only passive (linear) tracers can be scaled.")
    scaled = dataset.copy()
    for name in ("column", "column_below_altitude", "concentration", "emission_rate"):
        if name in scaled:
            attrs = dict(scaled[name].attrs)
            scaled[name] = scaled[name] * factor
            scaled[name].attrs = attrs
    previous = float(dataset.attrs.get("emission_scale", 1.0))
    scaled.attrs["emission_scale"] = previous * float(factor)
    return scaled


def plume_column_attrs(**values: object) -> dict[str, object]:
    """Return contract identification attributes merged with ``values``."""
    attrs: dict[str, object] = {
        "contract": PLUME_COLUMN_CONTRACT,
        "contract_version": PLUME_COLUMN_VERSION,
        "column_convention": "vertical",
        "emission_rate_units": EMISSION_RATE_UNITS,
        "grid_frame": "model",
    }
    attrs.update(values)
    return attrs


__all__ = [
    "COLUMN_DIMS",
    "COLUMN_UNITS",
    "EMISSION_RATE_UNITS",
    "PLUME_COLUMN_CONTRACT",
    "PLUME_COLUMN_VERSION",
    "REQUIRED_ATTRS",
    "PlumeColumnInfo",
    "plume_column_attrs",
    "plume_support_bounds",
    "scale_emission",
    "select_time",
    "validate_plume_column",
]
