"""Synthetic plume columns and sensor scenes for tests.

The generators build small, deterministic datasets that satisfy the SPECTRA
contracts without external data. They back the toy sensor adapter and the
adapter conformance suite.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from scipy import ndimage

import spectra
from spectra.contracts.plume_column import (
    COLUMN_UNITS,
    EMISSION_RATE_UNITS,
    WIND_UNITS,
    plume_column_attrs,
)
from spectra.contracts.sensor_scene import (
    PLATFORMS,
    RADIANCE_UNITS,
    SENSOR_SCENE_CONTRACT,
    SENSOR_SCENE_VERSION,
    SPATIAL_DIMS,
)
from spectra.geometry.crs import (
    projected_metre_crs,
    spatial_ref_coordinate,
    to_lonlat,
)
from spectra.types import Platform

DEFAULT_SCENE_CRS = "EPSG:32632"
DEFAULT_SCENE_ORIGIN_XY = (500000.0, 5200000.0)
UNIT_ABSORPTION_UNITS = "ppm-1 m-1"
SCENE_COORDINATE_NAMES = ("lat", "lon", "x", "y", "spatial_ref", "time")
# Gaussian absorption lines of the toy unit absorption spectrum: centre (nm),
# width (nm) and relative depth. The values mimic the 2.3 um CH4 band shape.
_TOY_LINES: tuple[tuple[float, float, float], ...] = (
    (2210.0, 12.0, 0.35),
    (2250.0, 10.0, 0.45),
    (2290.0, 14.0, 0.75),
    (2320.0, 10.0, 1.0),
    (2350.0, 12.0, 0.65),
    (2380.0, 16.0, 0.4),
)


def scene_seed(scene_id: str) -> int:
    """Return a stable, process-independent integer seed derived from a scene id."""
    digest = hashlib.blake2b(scene_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _to_datetime64(
    value: str | datetime | np.datetime64 | pd.Timestamp,
) -> np.datetime64:
    """Return a naive UTC ``datetime64[ns]`` from a string, datetime or timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_datetime64().astype("datetime64[ns]")


def _smooth_texture(
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    sigma: float,
    amplitude: float,
) -> np.ndarray:
    """Return a positive multiplicative texture ``1 + amplitude * smooth noise``."""
    noise = ndimage.gaussian_filter(rng.standard_normal(shape), sigma, mode="wrap")
    scale = float(np.std(noise))
    if scale > 0:
        noise = noise / scale
    lower, upper = 1.0 - 2.5 * amplitude, 1.0 + 2.5 * amplitude
    return np.clip(1.0 + amplitude * noise, lower, upper)


def _wind_components(speed: float, direction_deg: float) -> tuple[float, float]:
    """Return ``(u, v)`` of a wind from ``direction_deg`` (clockwise from north)."""
    radians = np.deg2rad(direction_deg)
    u = float(-speed * np.sin(radians))
    v = float(-speed * np.cos(radians))
    return (0.0 if abs(u) < 1e-12 else u, 0.0 if abs(v) < 1e-12 else v)


def _exit_distance(
    origin: tuple[float, float],
    direction: tuple[float, float],
    extent: tuple[float, float],
) -> float:
    """Return the distance from ``origin`` along ``direction`` to the domain edge."""
    distance = np.inf
    for position, component, upper in zip(origin, direction, extent, strict=True):
        if component > 0:
            distance = min(distance, (upper - position) / component)
        elif component < 0:
            distance = min(distance, -position / component)
    return float(distance)


def make_plume_column(
    *,
    nx: int = 256,
    ny: int = 256,
    dx: float = 2.0,
    dy: float | None = None,
    n_time: int = 3,
    dt_s: float = 60.0,
    source_xy: tuple[float, float] | None = None,
    source_z: float = 5.0,
    emission_rate_kg_s: float = 1.0,
    wind_ms: float = 3.0,
    wind_dir_deg: float = 270.0,
    sigma0_m: float = 6.0,
    spread: float = 0.15,
    case_id: str = "toy_case",
    tracer: str = "ch4",
    reference_time: str | datetime | np.datetime64 = "2026-01-01T12:00:00",
    seed: int = 0,
) -> xr.Dataset:
    """Return a synthetic ``PlumeColumn`` dataset of a Gaussian plume.

    Snapshot ``k`` holds the plume emitted during the first ``(k + 1) * dt_s``
    seconds: a steady Gaussian plume whose cross-wind width grows linearly
    with the downwind distance, advected by a constant wind and multiplied by
    a smooth, seeded texture. Every snapshot is normalised so that
    ``column.sum() * dx * dy`` equals the analytic mass of the plume segment
    inside the domain, recorded per snapshot in ``toy_total_mass_kg``.

    Args:
        nx: Number of cells along x.
        ny: Number of cells along y.
        dx: Cell size along x in metres.
        dy: Cell size along y in metres; defaults to ``dx``.
        n_time: Number of snapshots.
        dt_s: Time between snapshots in seconds.
        source_xy: Source position in the model frame in metres; defaults to a
            quarter of the domain along x and the domain centre along y.
        source_z: Source height above ground in metres.
        emission_rate_kg_s: Constant emission rate in kg s-1.
        wind_ms: Wind speed in m s-1.
        wind_dir_deg: Meteorological wind direction in degrees (direction the
            wind blows from, clockwise from north); 270 blows towards +x.
        sigma0_m: Cross-wind standard deviation at the source in metres.
        spread: Growth of the cross-wind standard deviation per metre travelled.
        case_id: LES case identifier.
        tracer: Tracer name.
        reference_time: UTC time of the LES start.
        seed: Seed of the plume texture.

    Returns:
        Dataset passing ``validate_plume_column``.

    Raises:
        ValueError: If the arguments do not describe a plume inside the domain.
    """
    dy = dx if dy is None else dy
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must be at least 2.")
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be positive.")
    if n_time < 1 or dt_s <= 0:
        raise ValueError("n_time must be at least 1 and dt_s positive.")
    if wind_ms <= 0 or sigma0_m <= 0 or spread < 0:
        raise ValueError("wind_ms and sigma0_m must be positive and spread >= 0.")
    if emission_rate_kg_s <= 0:
        raise ValueError("emission_rate_kg_s must be positive.")
    extent = (nx * dx, ny * dy)
    x0, y0 = (0.25 * extent[0], 0.5 * extent[1]) if source_xy is None else source_xy
    if not (0.0 <= x0 <= extent[0] and 0.0 <= y0 <= extent[1]):
        raise ValueError("source_xy must lie inside the domain.")

    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    grid_x, grid_y = np.meshgrid(x, y)
    u, v = _wind_components(wind_ms, wind_dir_deg)
    ex, ey = u / wind_ms, v / wind_ms
    # Plume frame: ``along`` is the downwind distance, ``cross`` the lateral offset.
    along = (grid_x - x0) * ex + (grid_y - y0) * ey
    cross = (grid_y - y0) * ex - (grid_x - x0) * ey
    sigma = sigma0_m + spread * np.clip(along, 0.0, None)
    profile = np.exp(-0.5 * (cross / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
    # A steady plume carries emission_rate / wind_speed kilograms per metre.
    line_density = emission_rate_kg_s / wind_ms
    rng = np.random.default_rng(seed)
    texture = _smooth_texture(rng, (ny, nx), sigma=3.0, amplitude=0.2)
    exit_distance = _exit_distance((x0, y0), (ex, ey), extent)

    simulation_time = (np.arange(n_time) + 1.0) * dt_s
    column = np.zeros((n_time, ny, nx), dtype=np.float32)
    masses: list[float] = []
    lengths: list[float] = []
    for index, time_s in enumerate(simulation_time):
        front = wind_ms * float(time_s)
        inside = (along >= 0.0) & (along <= front)
        field = np.where(inside, profile * line_density * texture, 0.0)
        # Zero the far Gaussian tails so the support stays off the domain edge.
        field[field < 1e-4 * field.max()] = 0.0
        gridded = float(field.astype(np.float32).astype(np.float64).sum() * dx * dy)
        if gridded <= 0.0:
            raise ValueError("The plume covers no cell; increase dt_s or wind_ms.")
        length = min(front, exit_distance)
        mass = line_density * length
        snapshot = field.astype(np.float32).astype(np.float64) * (mass / gridded)
        column[index] = snapshot.astype(np.float32)
        masses.append(float(mass))
        lengths.append(float(length))

    reference = _to_datetime64(reference_time)
    times = reference + (simulation_time * 1e9).astype("timedelta64[ns]")
    dataset = xr.Dataset(
        {
            "column": (
                ("time", "y", "x"),
                column,
                {"units": COLUMN_UNITS, "long_name": "vertical column of the tracer"},
            ),
            "emission_rate": (
                ("time",),
                np.full(n_time, float(emission_rate_kg_s)),
                {"units": EMISSION_RATE_UNITS, "long_name": "emission rate"},
            ),
            "wind_u": (
                ("time",),
                np.full(n_time, u),
                {"units": WIND_UNITS, "height": 10.0, "long_name": "eastward wind"},
            ),
            "wind_v": (
                ("time",),
                np.full(n_time, v),
                {"units": WIND_UNITS, "height": 10.0, "long_name": "northward wind"},
            ),
        },
        coords={
            "x": ("x", x, {"units": "m", "long_name": "x in the model frame"}),
            "y": ("y", y, {"units": "m", "long_name": "y in the model frame"}),
            "time": ("time", times, {"long_name": "time [UTC]"}),
            "simulation_time": (
                "time",
                simulation_time,
                {"units": "s", "long_name": "seconds since the LES start"},
            ),
        },
        attrs=plume_column_attrs(
            gas="CH4",
            les_model="toy",
            les_model_version=spectra.__version__,
            case_id=case_id,
            tracer=tracer,
            source_id=f"{case_id}:{tracer}",
            source_x=float(x0),
            source_y=float(y0),
            source_z=float(source_z),
            source_diameter=float(2.0 * sigma0_m),
            dx=float(dx),
            dy=float(dy),
            reference_time=np.datetime_as_string(reference, unit="s"),
            passive_tracer=True,
            emission_rate_is_constant=True,
            producer="spectra.testing",
            producer_version=spectra.__version__,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            toy_total_mass_kg=masses,
            toy_plume_length_m=lengths,
            toy_wind_ms=float(wind_ms),
            toy_wind_dir_deg=float(wind_dir_deg),
            toy_seed=int(seed),
        ),
    )
    return dataset


def scene_pixel_centres(
    *,
    n_across: int = 64,
    n_along: int = 128,
    pixel_across_m: float = 10.0,
    pixel_along_m: float | None = None,
    origin_xy: tuple[float, float] = DEFAULT_SCENE_ORIGIN_XY,
    grid_rotation_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return projected pixel-centre ``x`` and ``y`` on ``(across, along)``.

    The along-track axis runs along the flight direction, rotated
    counterclockwise by ``grid_rotation_deg`` from +x; the across-track axis
    points to the right of the flight direction, mirroring the AVIRIS-4
    flight-line layout.
    """
    pixel_along_m = pixel_across_m if pixel_along_m is None else pixel_along_m
    if n_across < 2 or n_along < 2:
        raise ValueError("n_across and n_along must be at least 2.")
    if pixel_across_m <= 0 or pixel_along_m <= 0:
        raise ValueError("Pixel sizes must be positive.")
    across = (np.arange(n_across) + 0.5) * pixel_across_m
    along = (np.arange(n_along) + 0.5) * pixel_along_m
    v, u = np.meshgrid(across, along, indexing="ij")
    theta = np.deg2rad(grid_rotation_deg)
    cos, sin = float(np.cos(theta)), float(np.sin(theta))
    x = origin_xy[0] + u * cos + v * sin
    y = origin_xy[1] + u * sin - v * cos
    return x, y


def make_sensor_scene(
    *,
    n_across: int = 64,
    n_along: int = 128,
    pixel_across_m: float = 10.0,
    pixel_along_m: float | None = None,
    wavelengths: npt.ArrayLike | None = None,
    fwhm_nm: float | npt.ArrayLike = 8.0,
    radiance_level: float = 0.05,
    sza: float = 30.0,
    vza: float = 0.0,
    saa: float = 150.0,
    vaa: float = 90.0,
    sensor_altitude: float = 3000.0,
    surface_elevation: float = 100.0,
    crs: str | int = DEFAULT_SCENE_CRS,
    origin_xy: tuple[float, float] = DEFAULT_SCENE_ORIGIN_XY,
    grid_rotation_deg: float = 0.0,
    noise_sigma: float = 0.0,
    sensor: str = "toy",
    scene_id: str = "toy-0001",
    platform: Platform = "airborne",
    acquisition_time: str | datetime | np.datetime64 = "2026-06-01T10:30:00",
    seed: int = 0,
    lazy: bool = True,
) -> xr.Dataset:
    """Return a synthetic ``SensorScene`` dataset on a rotated regular lattice.

    The radiance is ``radiance_level`` times a smooth, seeded albedo texture,
    a gently sloping spectral shape and optional multiplicative Gaussian noise.

    Args:
        n_across: Number of across-track pixels.
        n_along: Number of along-track pixels.
        pixel_across_m: Across-track pixel size in metres.
        pixel_along_m: Along-track pixel size in metres; defaults to the
            across-track size.
        wavelengths: Band centre wavelengths in nanometres, strictly
            increasing; defaults to 2100-2500 nm in 7.5 nm steps.
        fwhm_nm: Band widths in nanometres, scalar or per band.
        radiance_level: Typical radiance in W m-2 nm-1 sr-1.
        sza: Solar zenith angle in degrees.
        vza: Viewing zenith angle in degrees.
        saa: Solar azimuth angle in degrees.
        vaa: Viewing azimuth angle in degrees.
        sensor_altitude: Sensor altitude in metres above sea level.
        surface_elevation: Surface elevation in metres above sea level.
        crs: Projected metre CRS of the pixel centres.
        origin_xy: Projected coordinates of the lattice origin.
        grid_rotation_deg: Counterclockwise rotation of the flight direction
            from +x in degrees.
        noise_sigma: Relative standard deviation of the multiplicative noise.
        sensor: Sensor name written to the attributes.
        scene_id: Scene identifier.
        platform: Airborne or spaceborne.
        acquisition_time: UTC acquisition time.
        seed: Seed of the albedo texture and the noise.
        lazy: Back the radiance by a dask array.

    Returns:
        Dataset passing ``validate_sensor_scene(require_lazy=lazy)``.

    Raises:
        ValueError: If the arguments are inconsistent.
    """
    pixel_along_m = pixel_across_m if pixel_along_m is None else pixel_along_m
    bands = (
        np.arange(2100.0, 2500.0, 7.5)
        if wavelengths is None
        else np.asarray(wavelengths, dtype=float)
    )
    if bands.ndim != 1 or bands.size < 2 or np.any(np.diff(bands) <= 0):
        raise ValueError("wavelengths must be a strictly increasing 1-D array.")
    fwhm = np.broadcast_to(np.asarray(fwhm_nm, dtype=float), bands.shape).copy()
    if np.any(fwhm <= 0):
        raise ValueError("fwhm_nm must be positive.")
    if radiance_level <= 0 or noise_sigma < 0:
        raise ValueError("radiance_level must be positive and noise_sigma >= 0.")
    if platform not in PLATFORMS:
        raise ValueError(f"platform must be one of {PLATFORMS}.")

    crs_object = projected_metre_crs(crs)
    x, y = scene_pixel_centres(
        n_across=n_across,
        n_along=n_along,
        pixel_across_m=pixel_across_m,
        pixel_along_m=pixel_along_m,
        origin_xy=origin_xy,
        grid_rotation_deg=grid_rotation_deg,
    )
    lon, lat = to_lonlat(crs_object, x, y)
    epsg = crs_object.to_epsg()
    projected_attrs: dict[str, Any] = {"units": "m"}
    if epsg is not None:
        projected_attrs["epsg"] = int(epsg)

    rng = np.random.default_rng(seed)
    albedo = _smooth_texture(rng, (n_across, n_along), sigma=4.0, amplitude=0.2)
    span = max(float(bands[-1] - bands[0]), 1.0)
    spectral = 1.0 - 0.3 * (bands - bands[0]) / span
    radiance = radiance_level * albedo[:, :, None] * spectral[None, None, :]
    if noise_sigma > 0:
        radiance = radiance * (1.0 + noise_sigma * rng.standard_normal(radiance.shape))
    radiance32 = radiance.astype(np.float32)

    time = _to_datetime64(acquisition_time)
    pixel_area = float(pixel_across_m * pixel_along_m)
    spatial_shape = (n_across, n_along)
    scene = xr.Dataset(
        {
            "radiance": (
                ("across", "along", "band"),
                radiance32,
                {"units": RADIANCE_UNITS, "long_name": "at-sensor radiance"},
            ),
            "sza": ((), float(sza), {"units": "degree", "long_name": "solar zenith"}),
            "saa": ((), float(saa), {"units": "degree", "long_name": "solar azimuth"}),
            "vza": ((), float(vza), {"units": "degree", "long_name": "view zenith"}),
            "vaa": ((), float(vaa), {"units": "degree", "long_name": "view azimuth"}),
            "surface_elevation": (
                SPATIAL_DIMS,
                np.full(spatial_shape, float(surface_elevation)),
                {"units": "m", "long_name": "surface elevation above sea level"},
            ),
            "sensor_altitude": (
                (),
                float(sensor_altitude),
                {"units": "m", "long_name": "sensor altitude above sea level"},
            ),
            "valid": (
                SPATIAL_DIMS,
                np.ones(spatial_shape, dtype=bool),
                {"long_name": "usable pixel"},
            ),
            "pixel_area": (
                SPATIAL_DIMS,
                np.full(spatial_shape, pixel_area),
                {"units": "m2", "long_name": "pixel area"},
            ),
            "albedo": (
                SPATIAL_DIMS,
                albedo.astype(np.float32),
                {"units": "1", "long_name": "relative surface albedo"},
            ),
        },
        coords={
            "wavelength": (
                "band",
                bands,
                {"units": "nm", "long_name": "band centre wavelength"},
            ),
            "fwhm": ("band", fwhm, {"units": "nm", "long_name": "band width"}),
            "x": (SPATIAL_DIMS, x, projected_attrs),
            "y": (SPATIAL_DIMS, y, projected_attrs),
            "lon": (
                SPATIAL_DIMS,
                lon,
                {"units": "degrees_east", "standard_name": "longitude"},
            ),
            "lat": (
                SPATIAL_DIMS,
                lat,
                {"units": "degrees_north", "standard_name": "latitude"},
            ),
            "spatial_ref": spatial_ref_coordinate(crs_object),
            "time": ((), time, {"long_name": "time [UTC]"}),
        },
        attrs={
            "contract": SENSOR_SCENE_CONTRACT,
            "contract_version": SENSOR_SCENE_VERSION,
            "sensor": sensor,
            "platform": platform,
            "scene_id": scene_id,
            "acquisition_time": np.datetime_as_string(time, unit="s"),
            "geometry_space": "sensor",
            "adapter_version": spectra.__version__,
            "pixel_size_m": float(np.sqrt(pixel_area)),
            "toy_grid_rotation_deg": float(grid_rotation_deg),
            "toy_seed": int(seed),
        },
    )
    if lazy:
        scene["radiance"] = scene["radiance"].chunk(
            {"across": 16, "along": 32, "band": -1}
        )
    return scene


def scene_spatial_coords(scene: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return the georeference coordinates of a scene to copy onto a product."""
    return {
        name: scene[name]
        for name in SCENE_COORDINATE_NAMES
        if name in scene.coords or name in scene.data_vars
    }


def make_unit_absorption_spectrum(
    wavelengths: npt.ArrayLike,
    *,
    peak: float = -5e-5,
    gas: str = "CH4",
    lines: tuple[tuple[float, float, float], ...] = _TOY_LINES,
) -> xr.Dataset:
    """Return a toy ``UnitAbsorptionSpectrum`` of Gaussian absorption lines.

    Args:
        wavelengths: Strictly increasing wavelengths in nanometres.
        peak: Unit absorption of the strongest line in ppm-1 m-1 (negative).
        gas: Gas name written to the attributes.
        lines: ``(centre_nm, width_nm, relative_depth)`` per absorption line.

    Returns:
        Dataset with ``unit_abs_spect(wavelength)`` in ppm-1 m-1.

    Raises:
        ValueError: If ``peak`` is not negative or the wavelengths are invalid.
    """
    grid = np.asarray(wavelengths, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("wavelengths must be a strictly increasing 1-D array.")
    if not peak < 0:
        raise ValueError("peak must be negative: absorption lowers the radiance.")
    shape = np.zeros_like(grid)
    for centre, width, depth in lines:
        shape += depth * np.exp(-0.5 * ((grid - centre) / width) ** 2)
    spectrum = peak * shape
    return xr.Dataset(
        {
            "unit_abs_spect": (
                ("wavelength",),
                spectrum,
                {
                    "units": UNIT_ABSORPTION_UNITS,
                    "long_name": "unit absorption spectrum per vertical path amount",
                },
            )
        },
        coords={"wavelength": ("wavelength", grid, {"units": "nm"})},
        attrs={
            "gas": gas,
            "provider": "toy",
            "provider_version": spectra.__version__,
            "description": f"Toy unit absorption spectrum for {gas}",
        },
    )


__all__ = [
    "DEFAULT_SCENE_CRS",
    "DEFAULT_SCENE_ORIGIN_XY",
    "SCENE_COORDINATE_NAMES",
    "UNIT_ABSORPTION_UNITS",
    "make_plume_column",
    "make_sensor_scene",
    "make_unit_absorption_spectrum",
    "scene_pixel_centres",
    "scene_seed",
    "scene_spatial_coords",
]
