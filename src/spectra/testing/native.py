"""Synthetic native sensor files and storage-level comparison helpers.

The writers reproduce the on-disk schema of real products (names, groups,
dtypes, fill values, attributes and storage layout) at a small size, so the
importers and the copy-and-patch exporter can be tested without real data:

- :func:`write_aviris4_files`: AVIRIS-4 L1 radiance and PARGE metadata NetCDF
  as written for the 2025 campaigns (contiguous, int64 index coordinates) or
  by the newer converter of the 2026 campaigns (chunked, int32 coordinates);
- :func:`write_emit_files`: EMIT L1B RAD and OBS granules (root radiance plus
  ``sensor_band_parameters`` and ``location`` groups);
- :func:`write_aviris3_files`: AVIRIS-3 L1B RDN and OBS granules of the ORNL
  DAAC (band-major radiance in a ``radiance`` group, root geolocation and
  grid mapping, ``geolocation_lookup_table`` and ``observation_parameters``
  groups).

:func:`native_snapshot` and :func:`assert_native_preserved` compare two files
at the HDF5 level (dtype, layout, chunks, filters, fill value, storage offset,
attributes with their HDF5 types and the raw bytes), which is what "the output
is the input file with only the patched pixels changed" means. They need
``h5py``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import netCDF4
import numpy as np
import pyproj

if TYPE_CHECKING:
    import h5py

AVIRIS4_SCHEMAS = ("2025", "2026")
EMIT_BUILDS = ("010635", "010617")
OBS_BAND_NAMES = (
    "Path length (sensor-to-ground in meters)",
    "To-sensor azimuth (0 to 360 degrees CW from N)",
    "To-sensor zenith (0 to 90 degrees from zenith)",
    "To-sun azimuth (0 to 360 degrees CW from N)",
    "To-sun zenith (0 to 90 degrees from zenith)",
    "Solar phase (degrees between to-sensor and to-sun vectors in principal plane)",
    "Slope (local surface slope as derived from DEM in degrees)",
    "Aspect (local surface aspect 0 to 360 degrees clockwise from N)",
    "Cosine(i) (apparent local illumination factor based on DEM slope and aspect "
    "and to sun vector)",
    "UTC Time (decimal hours for mid-line pixels)",
    "Earth-sun distance (AU)",
)
# Attributes HDF5 maintains for dimension scales; they change whenever a new
# variable is attached to an existing dimension and are not netCDF metadata.
_SCALE_BOOKKEEPING = ("REFERENCE_LIST",)
_HASH_BLOCK_BYTES = 64 * 2**20


def _swath_lonlat(
    n_across: int,
    n_along: int,
    *,
    centre_lonlat: tuple[float, float],
    pixel_m: float,
    heading_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pyproj.CRS]:
    """Return ``(x, y, lon, lat, crs)`` of a straight swath on ``(across, along)``."""
    lon0, lat0 = centre_lonlat
    zone = int((lon0 + 180.0) // 6.0) + 1
    crs = pyproj.CRS.from_epsg((32600 if lat0 >= 0 else 32700) + zone)
    to_xy = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_ll = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x0, y0 = to_xy.transform(lon0, lat0)
    heading = np.deg2rad(heading_deg)
    along = (np.arange(n_along) - (n_along - 1) / 2.0) * pixel_m
    across = (np.arange(n_across) - (n_across - 1) / 2.0) * pixel_m
    grid_across, grid_along = np.meshgrid(across, along, indexing="ij")
    # Along-track points to the heading (clockwise from north), across-track
    # to its right.
    x = x0 + grid_along * np.sin(heading) + grid_across * np.cos(heading)
    y = y0 + grid_along * np.cos(heading) - grid_across * np.sin(heading)
    lon, lat = to_ll.transform(x, y)
    return x, y, np.asarray(lon), np.asarray(lat), crs


def _wavelengths(n_band: int) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.linspace(380.0, 2500.0, n_band, dtype=np.float32)
    fwhm = np.full(n_band, 8.5, dtype=np.float32)
    return wavelength, fwhm


def _radiance_cube(
    shape_spatial: tuple[int, int], wavelength: np.ndarray, *, scale: float, seed: int
) -> np.ndarray:
    """Return a positive, textured radiance cube on ``(*shape_spatial, band)``."""
    rng = np.random.default_rng(seed)
    spectrum = np.exp(-(((wavelength - 900.0) / 700.0) ** 2)) + 0.05
    albedo = 0.2 + 0.1 * rng.random(shape_spatial)
    cube = albedo[..., None] * spectrum[None, None, :] * scale
    return cube.astype(np.float32)


def write_aviris4_files(
    directory: str | Path,
    *,
    schema: Literal["2025", "2026"] = "2025",
    stem: str = "M000_250101_TST_Methane_Line_0001_120000_000",
    n_across: int = 20,
    n_along: int = 24,
    n_band: int = 40,
    centre_lonlat: tuple[float, float] = (19.5, 50.1),
    pixel_m: float = 2.0,
    heading_deg: float = 347.0,
    time: str = "2024-09-28T07:29:27",
    sun_zenith: float = 55.0,
    sun_azimuth: float = 140.0,
    fill_pixels: Iterable[tuple[int, int]] = ((0, 0),),
    seed: int = 0,
) -> dict[str, Path]:
    """Write a small AVIRIS-4 radiance and metadata pair with the real schema.

    Args:
        directory: Output directory.
        schema: ``2025`` (contiguous, int64 coordinates, ``days since`` time)
            or ``2026`` (chunked, int32 coordinates, epoch-seconds time, pixel
            sizes and along-track time in the metadata).
        stem: File-name stem; the names carry no meaning for SPECTRA.
        n_across: Across-track pixels.
        n_along: Along-track lines.
        n_band: Spectral bands.
        centre_lonlat: Swath centre in degrees.
        pixel_m: Pixel size in metres.
        heading_deg: Flight heading, clockwise from north.
        time: Acquisition time (UTC).
        sun_zenith: Solar zenith angle stored on ``cast_shadow``.
        sun_azimuth: Solar azimuth angle stored on ``cast_shadow``.
        fill_pixels: ``(across, along)`` pixels whose radiance is NaN.
        seed: Seed of the radiance texture.

    Returns:
        ``{"radiance": path, "metadata": path}``.
    """
    if schema not in AVIRIS4_SCHEMAS:
        raise ValueError(f"schema must be one of {AVIRIS4_SCHEMAS}.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, y, lon, lat, crs = _swath_lonlat(
        n_across,
        n_along,
        centre_lonlat=centre_lonlat,
        pixel_m=pixel_m,
        heading_deg=heading_deg,
    )
    wavelength, fwhm = _wavelengths(n_band)
    radiance = _radiance_cube((n_across, n_along), wavelength, scale=0.1, seed=seed)
    for across, along in fill_pixels:
        radiance[across, along, :] = np.nan
    stamp = np.datetime64(time, "s")
    paths = {
        "radiance": directory / f"{stem}_rdn.nc",
        "metadata": directory / f"{stem}_rdn_metadata.nc",
    }
    new = schema == "2026"
    index_type = "i4" if new else "i8"
    cube_chunks = (1, max(1, (n_along + 1) // 2), n_band) if new else None

    def write_time(ds: netCDF4.Dataset) -> None:
        if new:
            variable = ds.createVariable("time", "f8", ())
            variable.long_name = "overpass time [UTC]"
            variable.units = "seconds since 1970-01-01 00:00:00"
            variable.calendar = "proleptic_gregorian"
            variable.assignValue(float(stamp.astype("int64")))
        else:
            variable = ds.createVariable("time", "i8", ())
            variable.long_name = "time [UTC]"
            variable.units = f"days since {str(stamp).replace('T', ' ')}"
            variable.calendar = "proleptic_gregorian"
            variable.assignValue(0)

    def write_index(ds: netCDF4.Dataset, name: str, size: int) -> None:
        variable = ds.createVariable(name, index_type, (name,))
        variable[:] = np.arange(size)

    with netCDF4.Dataset(paths["radiance"], "w", format="NETCDF4") as ds:
        ds.createDimension("across", n_across)
        ds.createDimension("along", n_along)
        ds.createDimension("band", n_band)
        if new:
            for name, size in (("across", n_across), ("along", n_along)):
                write_index(ds, name, size)
            write_index(ds, "band", n_band)
            write_time(ds)
        cube = ds.createVariable(
            "radiance",
            "f4",
            ("across", "along", "band"),
            fill_value=np.float32(np.nan),
            contiguous=not new,
            chunksizes=cube_chunks,
        )
        cube.units = "W m-2 nm-1 sr-1"
        cube.coordinates = "time"
        cube[:] = radiance
        for name, values in (("wavelength", wavelength), ("fwhm", fwhm)):
            if new:
                variable = ds.createVariable(name, "f4", ("band",))
                variable.units = "nanometers"
            else:
                variable = ds.createVariable(
                    name, "f4", ("band",), fill_value=np.float32(np.nan)
                )
                variable.units = "nm"
                variable.coordinates = "time"
            variable[:] = values
        if not new:
            for name, size in (("across", n_across), ("along", n_along)):
                write_index(ds, name, size)
            write_index(ds, "band", n_band)
            write_time(ds)
        ds.setncattr("masked pixel noise", 0.0493343035868973)

    epsg = np.int64(crs.to_epsg() or 0)
    elevation = 300.0 + 0.5 * np.arange(n_along)[None, :] + np.zeros((n_across, 1))
    scan = np.linspace(-20.0, 20.0, n_across)[:, None] + np.zeros((1, n_along))
    metadata_vars: dict[str, tuple[str, np.ndarray, dict[str, Any]]] = {
        "aspect": ("i2", np.full((n_across, n_along), 90), {"units!": "°"}),
        "dem": ("f4", elevation - 41.5, {"units": "m"}),
        "x": ("f8", x, {"units": "m", "epsg": epsg}),
        "y": ("f8", y, {"units": "m", "epsg": epsg}),
        "elevation": ("f8", elevation, {"units": "m"}),
        "illumination": ("f4", np.full((n_across, n_along), 0.8), {"units": "-"}),
        "scan_zenith": ("i2", np.round(scan * 100), {"units!": "° * 100"}),
        "absolute_azimuth": (
            "i2",
            np.full((n_across, n_along), round(heading_deg * 10) % 3600),
            {"units!": "° * 10"},
        ),
        "aircraft_altitude": ("i2", np.full((n_across, n_along), 2175), {"units": "m"}),
        "cast_shadow": (
            "u1",
            np.zeros((n_across, n_along)),
            {"units": "-", "azimuth": sun_azimuth, "zenith": sun_zenith},
        ),
        "skyview_factor": ("u1", np.full((n_across, n_along), 97), {"units": "%"}),
        "slope": ("u1", np.full((n_across, n_along), 3), {"units!": "°"}),
        "lon": (
            "f4" if new else "f8",
            lon,
            {
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "longitude",
                "epsg": np.int64(4326),
            },
        ),
        "lat": (
            "f4" if new else "f8",
            lat,
            {
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "latitude",
                "epsg": np.int64(4326),
            },
        ),
    }
    if new:
        metadata_vars["pixel_size_across_track"] = (
            "f4",
            np.full((n_across, n_along), pixel_m),
            {"units": "m"},
        )
        metadata_vars["pixel_size_along_track"] = (
            "f4",
            np.full((n_across, n_along), pixel_m),
            {"units": "m"},
        )
    nan_filled = {"dem", "x", "y", "elevation", "illumination", "lon", "lat"}
    with netCDF4.Dataset(paths["metadata"], "w", format="NETCDF4") as ds:
        ds.createDimension("across", n_across)
        ds.createDimension("along", n_along)
        for name, size in (("across", n_across), ("along", n_along)):
            write_index(ds, name, size)
        write_time(ds)
        for name, (dtype, values, attrs) in metadata_vars.items():
            fill = None
            if name in nan_filled and (not new or name in ("lon", "lat")):
                fill = np.dtype(dtype).type(np.nan)
            variable = ds.createVariable(
                name,
                dtype,
                ("across", "along"),
                fill_value=fill,
                contiguous=not new,
                chunksizes=(1, n_along) if new else None,
            )
            for key, value in attrs.items():
                if key.endswith("!"):
                    # Non-ASCII units are NC_STRING attributes in the real files.
                    variable.setncattr_string(key[:-1], value)
                else:
                    variable.setncattr(key, value)
            if not new:
                variable.coordinates = "time"
            variable[:] = np.asarray(values).astype(dtype)
        if new:
            along_time = ds.createVariable("along_track_time", "f8", ("along",))
            along_time.long_name = "precise along-track acquisition time [UTC]"
            along_time.units = "seconds since 1970-01-01 00:00:00"
            along_time[:] = float(stamp.astype("int64")) + 0.004 * np.arange(n_along)
    return paths


def write_emit_files(
    directory: str | Path,
    *,
    stem: str = "EMIT_L1B_{kind}_001_20260819T180247_2623112_032",
    n_downtrack: int = 24,
    n_crosstrack: int = 20,
    n_band: int = 40,
    centre_lonlat: tuple[float, float] = (-102.9, 32.2),
    pixel_m: float = 60.0,
    heading_deg: float = 200.0,
    time: str = "2026-08-19T18:02:47",
    build: Literal["010635", "010617"] = "010635",
    fill_pixels: Iterable[tuple[int, int]] = ((0, 0),),
    seed: int = 0,
) -> dict[str, Path]:
    """Write a small EMIT L1B RAD and OBS pair with the real schema.

    Args:
        directory: Output directory.
        stem: File-name stem with a ``{kind}`` placeholder (RAD/OBS).
        n_downtrack: Downtrack lines.
        n_crosstrack: Crosstrack samples.
        n_band: Radiance bands.
        centre_lonlat: Granule centre in degrees.
        pixel_m: Pixel size in metres.
        heading_deg: Downtrack direction, clockwise from north.
        time: ``time_coverage_start`` (UTC).
        build: Software build; ``010635`` writes the GLT fill value 0 and
            ``CF-1.13``, older builds -9999 and ``CF-1.63``.
        fill_pixels: ``(downtrack, crosstrack)`` pixels whose radiance is the
            -9999 fill value.
        seed: Seed of the radiance texture.

    Returns:
        ``{"radiance": path, "obs": path}``.
    """
    if build not in EMIT_BUILDS:
        raise ValueError(f"build must be one of {EMIT_BUILDS}.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, y, lon_ac, lat_ac, _ = _swath_lonlat(
        n_crosstrack,
        n_downtrack,
        centre_lonlat=centre_lonlat,
        pixel_m=pixel_m,
        heading_deg=heading_deg,
    )
    del x, y
    lon, lat = lon_ac.T, lat_ac.T  # (downtrack, crosstrack)
    wavelength, fwhm = _wavelengths(n_band)
    radiance = _radiance_cube(
        (n_downtrack, n_crosstrack), wavelength, scale=10.0, seed=seed
    )
    for downtrack, crosstrack in fill_pixels:
        radiance[downtrack, crosstrack, :] = -9999.0
    elevation = (
        900.0 + 2.0 * np.arange(n_crosstrack)[None, :] + np.zeros((n_downtrack, 1))
    )
    stamp = np.datetime64(time, "s")
    start = str(stamp).replace(" ", "T") + "+0000"
    end = str(stamp + np.timedelta64(12, "s")).replace(" ", "T") + "+0000"
    old = build != "010635"
    ortho = (n_downtrack + 5, n_crosstrack + 7)
    west, east = float(lon.min()), float(lon.max())
    south, north = float(lat.min()), float(lat.max())
    resolution = (east - west) / ortho[1]
    globals_common: dict[str, Any] = {
        "ncei_template_version": "NCEI_NetCDF_Swath_Template_v2.0",
        "keywords": "Imaging Spectroscopy, minerals, EMIT, dust, radiative forcing",
        "sensor": "EMIT (Earth Surface Mineral Dust Source Investigation)",
        "instrument": "EMIT",
        "platform": "ISS",
        "Conventions": "CF-1.63" if old else "CF-1.13",
        "institution": (
            "NASA Jet Propulsion Laboratory/California Institute of Technology"
        ),
        "naming_authority": "LPDAAC",
        "date_created": "2026-08-23T06:24:47Z",
        "flight_line": "emit20260819t180247_o23112_s001",
        "time_coverage_start": start,
        "time_coverage_end": end,
        "software_build_version": build,
        "software_delivery_version": build,
        "product_version": "V001",
        "history": "PGE Run Command: {...}",
        "easternmost_longitude": east,
        "northernmost_latitude": north,
        "westernmost_longitude": west,
        "southernmost_latitude": south,
        "spatialResolution": resolution,
        "spatial_ref": 'GEOGCS["WGS 84",AUTHORITY["EPSG","4326"]]',
        "geotransform": np.array([west, resolution, -0.0, north, -0.0, -resolution]),
        "day_night_flag": "Day",
    }
    paths = {
        "radiance": directory / f"{stem.format(kind='RAD')}.nc",
        "obs": directory / f"{stem.format(kind='OBS')}.nc",
    }
    obs = np.zeros((n_downtrack, n_crosstrack, len(OBS_BAND_NAMES)), dtype=np.float32)
    vza = np.abs(np.linspace(-6.0, 6.0, n_crosstrack))[None, :] + np.zeros(
        (n_downtrack, 1)
    )
    obs[..., 0] = 420_000.0 / np.cos(np.deg2rad(vza))
    obs[..., 1] = 110.0
    obs[..., 2] = vza
    obs[..., 3] = 150.0
    obs[..., 4] = 25.0 + 0.01 * np.arange(n_downtrack)[:, None]
    obs[..., 5] = 30.0
    obs[..., 6] = 2.0
    obs[..., 7] = 180.0
    obs[..., 8] = 0.9
    obs[..., 9] = 18.05
    obs[..., 10] = 1.012

    def write_location(ds: netCDF4.Dataset) -> None:
        group = ds.createGroup("location")
        for name, values, long_name, units in (
            ("lon", lon, "Longitude (WGS-84)", "degrees east"),
            ("lat", lat, "Latitude (WGS-84)", "degrees north"),
            ("elev", elevation, "Surface Elevation", "m"),
        ):
            variable = group.createVariable(
                name,
                "f8",
                ("downtrack", "crosstrack"),
                fill_value=-9999.0,
                contiguous=True,
            )
            variable.long_name = long_name
            variable.units = units
            variable[:] = values
        glt_fill = -9999 if old else 0
        glt = np.zeros(ortho, dtype=np.int32)
        glt[:n_downtrack, :n_crosstrack] = np.arange(1, n_crosstrack + 1)[None, :]
        for name, long_name in (
            ("glt_x", "GLT Sample Lookup"),
            ("glt_y", "GLT Line Lookup"),
        ):
            variable = group.createVariable(
                name,
                "i4",
                ("ortho_y", "ortho_x"),
                fill_value=glt_fill,
                zlib=True,
                complevel=9,
                shuffle=True,
                chunksizes=(max(1, ortho[0] // 2), max(1, ortho[1] // 2)),
            )
            variable.long_name = long_name
            variable.units = "pixel location"
            variable[:] = glt

    def write_globals(ds: netCDF4.Dataset, summary: str, title: str) -> None:
        ds.setncattr_string("summary", summary)
        for key, value in globals_common.items():
            ds.setncattr(key, value)
        ds.title = title

    for kind, path in paths.items():
        with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
            n_bands = n_band if kind == "radiance" else len(OBS_BAND_NAMES)
            for name, size in (
                ("downtrack", n_downtrack),
                ("crosstrack", n_crosstrack),
                ("bands", n_bands),
                ("ortho_y", ortho[0]),
                ("ortho_x", ortho[1]),
            ):
                ds.createDimension(name, size)
            if kind == "radiance":
                cube = ds.createVariable(
                    "radiance",
                    "f4",
                    ("downtrack", "crosstrack", "bands"),
                    fill_value=np.float32(-9999.0),
                    contiguous=True,
                )
                cube.long_name = "Radiance Data"
                cube.units = "uW/cm^2/SR/nm"
                cube[:] = radiance
                flat = ds.createVariable(
                    "flat_field_update",
                    "f4",
                    ("crosstrack", "bands"),
                    fill_value=np.float32(-9999.0),
                    contiguous=True,
                )
                flat.long_name = "Flat Field Update"
                flat[:] = np.ones((n_crosstrack, n_band), dtype=np.float32)
                write_globals(
                    ds,
                    "This file contains L1B at-sensor calibrated radiances (°).",
                    "EMIT L1B At-Sensor Calibrated Radiance Data 60 m V001",
                )
                ds.crosstrack_orientation = "as seen on ground"
                bands = ds.createGroup("sensor_band_parameters")
                for name, values, long_name in (
                    ("wavelengths", wavelength, "Wavelength Centers"),
                    ("fwhm", fwhm, "Full Width at Half Max"),
                ):
                    variable = bands.createVariable(
                        name, "f4", ("bands",), fill_value=np.float32(-9999.0)
                    )
                    variable.long_name = long_name
                    variable.units = "nm"
                    variable[:] = values
            else:
                cube = ds.createVariable(
                    "obs",
                    "f4",
                    ("downtrack", "crosstrack", "bands"),
                    fill_value=np.float32(-9999.0),
                    contiguous=True,
                )
                cube.long_name = "Observation Data"
                cube[:] = obs
                write_globals(
                    ds,
                    "This file contains L1B geometric information (°).",
                    "EMIT L1B Observation Data 60 m V001",
                )
                bands = ds.createGroup("sensor_band_parameters")
                names = bands.createVariable(
                    "observation_bands", str, ("bands",), fill_value="-9999.0"
                )
                names.long_name = "Observation Band Names"
                for index, name in enumerate(OBS_BAND_NAMES):
                    names[index] = name
            write_location(ds)
    return paths


def write_aviris3_files(
    directory: str | Path,
    *,
    stem: str = "AV320231006t181013_005_L1B_{kind}",
    n_lines: int = 24,
    n_samples: int = 20,
    n_band: int = 40,
    centre_lonlat: tuple[float, float] = (-104.03, 32.02),
    pixel_m: float = 4.0,
    heading_deg: float = 40.0,
    time: str = "2023-10-06T18:13:29",
    flight_line: str = "AV320231006t181013",
    scene: str = "005",
    fill_pixels: Iterable[tuple[int, int]] = ((0, 0),),
    seed: int = 0,
) -> dict[str, Path]:
    """Write a small AVIRIS-3 L1B RDN and OBS pair with the real schema.

    Args:
        directory: Output directory.
        stem: File-name stem with a ``{kind}`` placeholder; the names carry
            no meaning for SPECTRA.
        n_lines: Lines (along track).
        n_samples: Samples (across track).
        n_band: Radiance bands.
        centre_lonlat: Granule centre in degrees.
        pixel_m: Pixel size in metres.
        heading_deg: Flight heading, clockwise from north.
        time: ``time_coverage_start`` (UTC).
        flight_line: ``flight_line`` attribute of both granules.
        scene: ``scene`` attribute of both granules.
        fill_pixels: ``(line, sample)`` pixels whose radiance is the -9999
            fill value.
        seed: Seed of the radiance texture.

    Returns:
        ``{"radiance": path, "obs": path}``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x_ac, y_ac, lon_ac, lat_ac, crs = _swath_lonlat(
        n_samples,
        n_lines,
        centre_lonlat=centre_lonlat,
        pixel_m=pixel_m,
        heading_deg=heading_deg,
    )
    x, y, lon, lat = x_ac.T, y_ac.T, lon_ac.T, lat_ac.T  # (lines, samples)
    wavelength, fwhm = _wavelengths(n_band)
    radiance = np.moveaxis(
        _radiance_cube((n_lines, n_samples), wavelength, scale=1.0, seed=seed), -1, 0
    )
    for line, sample in fill_pixels:
        radiance[:, line, sample] = -9999.0
    elevation = 880.0 + 1.5 * np.arange(n_samples)[None, :] + np.zeros((n_lines, 1))
    stamp = np.datetime64(time, "s")
    start = str(stamp) + "Z"
    end = str(stamp + np.timedelta64(36, "s")) + "Z"
    paths = {
        "radiance": directory / f"{stem.format(kind='RDN_cbeae6f8_RDN')}.nc",
        "obs": directory / f"{stem.format(kind='ORT_3b50f254_OBS')}.nc",
    }
    spatial_chunks = (max(1, n_lines // 2), max(1, n_samples // 2))

    # The geolocation lookup table on a north-up UTM grid of the pixel size:
    # 1-based raw line and sample of every orthocorrected cell, 0 elsewhere.
    west = np.floor(x.min() / pixel_m) * pixel_m
    north = np.ceil(y.max() / pixel_m) * pixel_m
    column = np.rint((x - west) / pixel_m).astype(int)
    row = np.rint((north - y) / pixel_m).astype(int)
    easting = west + pixel_m * np.arange(column.max() + 1)
    northing = north - pixel_m * np.arange(row.max() + 1)
    glt_sample = np.zeros((northing.size, easting.size), dtype=np.float32)
    glt_line = np.zeros_like(glt_sample)
    lines_index, samples_index = np.indices((n_lines, n_samples))
    glt_sample[row, column] = samples_index + 1
    glt_line[row, column] = lines_index + 1
    cf = crs.to_cf()
    wkt = crs.to_wkt("WKT1_GDAL")
    geotransform = (
        f"{west - pixel_m / 2} {pixel_m} 0.0 {north + pixel_m / 2} 0.0 {-pixel_m}"
    )
    summary = (
        "The Airborne Visible / Infrared Imaging Spectrometer-3 (AVIRIS-3) is a "
        "NASA JPL airborne imaging spectrometer."
    )
    globals_common: dict[str, Any] = {
        "Conventions": "CF-1.6",
        "summary": summary,
        "keywords": "Imaging Spectroscopy, AVIRIS, AVIRIS-3",
        "sensor": "Airborne Visible / Infrared Imaging Spectrometer 3",
        "instrument": "AVIRIS-3",
        "platform": "Beechcraft Super King Air 200",
        "institution": (
            "NASA Jet Propulsion Laboratory/California Institute of Technology"
        ),
        "keywords_vocabulary": (
            "NASA Global Change Master Directory (GCMD) Science Keywords"
        ),
        "creator_name": "Jet Propulsion Laboratory/California Institute of Technology",
        "creator_url": "aviris.jpl.nasa.gov",
        "publisher_type": "institution",
        "publisher_institution": (
            "Oak Ridge National Laboratory Distributed Active Archive Center "
            "(ORNL DAAC)"
        ),
        "identifier_product_doi_authority": "https://doi.org",
        "flight_line": flight_line,
        "scene": scene,
        "software_build_version": "001",
        "ncei_template_version": "NCEI_NetCDF_Swath_Template_v2.0",
    }
    globals_tail: dict[str, Any] = {
        "processing_level": "L1B",
        "time_coverage_start": start,
        "time_coverage_end": end,
        "product_version": "1",
    }
    location_attrs: dict[str, dict[str, Any]] = {
        "lat": {
            "_CoordinateAxisType": "Lat",
            "long_name": "Latitude coordinate",
            "standard_name": "latitude",
            "units": "degrees_north",
            "description": "Latitude for center of grid cell",
        },
        "lon": {
            "_CoordinateAxisType": "Lon",
            "long_name": "Longitude coordinate",
            "standard_name": "longitude",
            "units": "degrees_east",
            "description": "Longitude for center of grid cell",
        },
        "elev": {
            "standard_name": "height_above_mean_sea_level",
            "long_name": "Surface elevation",
            "units": "m",
        },
    }
    vza = np.abs(np.linspace(-21.0, 21.0, n_samples))[None, :] + np.zeros((n_lines, 1))
    hours = (stamp - stamp.astype("datetime64[D]")) / np.timedelta64(1, "h")
    observations: dict[str, tuple[np.ndarray, dict[str, str]]] = {
        "path_length": (
            (8400.0 - elevation) / np.cos(np.deg2rad(vza)),
            {
                "units": "m",
                "long_name": "Path length",
                "description": "Sensor-to-ground distance",
            },
        ),
        "to_sensor_azimuth": (
            np.where(
                np.arange(n_samples)[None, :] < n_samples // 2,
                (heading_deg + 270.0) % 360.0,
                (heading_deg + 90.0) % 360.0,
            )
            + np.zeros((n_lines, 1)),
            {
                "standard_name": "sensor_azimuth_angle",
                "units": "degrees",
                "long_name": "To-sensor-azimuth",
                "description": "0 to 360 degrees clockwise from N",
            },
        ),
        "to_sensor_zenith": (
            vza,
            {
                "standard_name": "sensor_zenith_angle",
                "units": "degrees",
                "long_name": "To-sensor-zenith",
                "description": "0 to 90 degrees from zenith",
            },
        ),
        "to_sun_azimuth": (
            np.full((n_lines, n_samples), 166.2),
            {
                "standard_name": "solar_azimuth_angle",
                "units": "degrees",
                "long_name": "To-sun-azimuth",
                "description": "0 to 360 degrees clockwise from N",
            },
        ),
        "to_sun_zenith": (
            38.1 + 0.001 * np.arange(n_lines)[:, None] + np.zeros((1, n_samples)),
            {
                "standard_name": "solar_zenith_angle",
                "units": "degrees",
                "long_name": "To-sun-zenith",
                "description": "0 to 360 degrees clockwise from N",
            },
        ),
        "solar_phase": (
            np.full((n_lines, n_samples), 40.0),
            {
                "units": "degrees",
                "long_name": "Solar phase",
                "description": (
                    "Degrees between to-sensor and to-sun vectors in principal plane"
                ),
            },
        ),
        "slope": (
            np.full((n_lines, n_samples), 2.0),
            {
                "standard_name": "ground_slope_angle",
                "units": "degrees",
                "long_name": "Slope",
                "description": "Local surface slope as derived from DEM in degrees",
            },
        ),
        "aspect": (
            np.full((n_lines, n_samples), 180.0),
            {
                "standard_name": "ground_slope_direction",
                "units": "degrees",
                "long_name": "Aspect",
                "description": "Local surface aspect 0 to 360 degrees clockwise from N",
            },
        ),
        "cosine_i": (
            np.full((n_lines, n_samples), 0.8),
            {
                "units": "unitless",
                "long_name": "Cosine i",
                "description": (
                    "Apparent local illumination factor based on DEM slope and "
                    "aspect and to sun vector, -1 to 1"
                ),
            },
        ),
        "utc_time": (
            hours + 1e-4 * np.arange(n_lines)[:, None] + np.zeros((1, n_samples)),
            {
                "standard_name": "time",
                "units": f"hours since {str(stamp)[:10]} 00:00:00Z",
                "long_name": "UTC time",
                "description": "Decimal hours for mid-line pixels",
            },
        ),
        "earth_sun_distance": (
            np.full((n_lines, n_samples), 0.99976),
            {
                "units": "AU",
                "long_name": "Earth sun distance",
                "description": "Earth-sun distance",
            },
        ),
    }

    def write_root(ds: netCDF4.Dataset, title: str, created: str) -> None:
        ds.createDimension("lines", n_lines)
        ds.createDimension("samples", n_samples)
        for name, values in (("lat", lat), ("lon", lon), ("elev", elevation)):
            variable = ds.createVariable(
                name,
                "f4",
                ("lines", "samples"),
                fill_value=np.float32(-9999.0),
                chunksizes=spatial_chunks,
            )
            variable.setncatts(location_attrs[name])
            variable[:] = values
        mapping = ds.createVariable("transverse_mercator", "S1", ())
        for key in (
            "grid_mapping_name",
            "longitude_of_central_meridian",
            "false_easting",
            "false_northing",
            "latitude_of_projection_origin",
            "scale_factor_at_central_meridian",
        ):
            mapping.setncattr(key, cf[key])
        mapping.long_name = "CRS definition"
        mapping.longitude_of_prime_meridian = 0.0
        mapping.semi_major_axis = cf["semi_major_axis"]
        mapping.inverse_flattening = cf["inverse_flattening"]
        mapping.spatial_ref = wkt
        mapping.crs_wkt = wkt
        mapping.GeoTransform = geotransform
        mapping.units = "m"
        ds.setncattr("Conventions", globals_common["Conventions"])
        ds.setncattr("date_created", created)
        for key, value in globals_common.items():
            if key != "Conventions":
                ds.setncattr(key, value)
        ds.setncattr("title", f"{title} (flight line: {flight_line}, scene: {scene})")
        for key, value in globals_tail.items():
            ds.setncattr(key, value)
        glt = ds.createGroup("geolocation_lookup_table")
        glt.createDimension("easting", easting.size)
        glt.createDimension("northing", northing.size)
        for name, values, axis in (
            ("easting", easting, "X"),
            ("northing", northing, "Y"),
        ):
            variable = glt.createVariable(name, "f8", (name,))
            variable.description = (
                f"UTM {name} coordinate for center of grid cell for orthocorrected "
                "pixel location"
            )
            variable.standard_name = f"projection_{axis.lower()}_coordinate"
            variable.long_name = f"{axis} coordinate of projection"
            variable.units = "m"
            variable.axis = axis
            variable[:] = values
        for name, values, long_name, direction in (
            ("sample", glt_sample, "Sample lookup table", "across-track"),
            ("line", glt_line, "Line lookup table", "along-track"),
        ):
            variable = glt.createVariable(
                name,
                "f4",
                ("northing", "easting"),
                fill_value=np.float32(0.0),
                chunksizes=(
                    max(1, northing.size // 2),
                    max(1, easting.size // 2),
                ),
            )
            variable.long_name = long_name
            variable.grid_mapping = "transverse_mercator"
            variable.description = (
                f"Pixel identifier in the {direction} direction within the "
                "orthocorrected grid."
            )
            variable[:] = values
        glt.long_name = "Geolocation lookup table"
        glt.description = (
            "Orthocorrected product with a fixed pixel size projected into a UTM "
            "system; positive values are real pixels, negative values "
            "nearest-neighbor infill."
        )

    with netCDF4.Dataset(paths["radiance"], "w", format="NETCDF4") as ds:
        write_root(ds, "AVIRIS-3 L1B Calibrated Radiance", "2024-10-01T00:15:29Z")
        group = ds.createGroup("radiance")
        group.createDimension("wavelength", n_band)
        cube = group.createVariable(
            "radiance",
            "f4",
            ("wavelength", "lines", "samples"),
            fill_value=np.float32(-9999.0),
            chunksizes=(max(1, n_band // 3), *spatial_chunks),
        )
        cube.standard_name = "surface_upwelling_radiance_per_unit_wavelength_in_air"
        cube.long_name = "Calibrated radiance"
        cube.units = "uW nm-1 cm-2 sr-1"
        cube.coordinates = "lat lon"
        cube[:] = radiance
        centres = group.createVariable("wavelength", "f4", ("wavelength",))
        centres.standard_name = "radiation_wavelength"
        centres.long_name = "Wavelength centers"
        centres.units = "nm"
        centres[:] = wavelength
        widths = group.createVariable("fwhm", "f4", ("wavelength",))
        widths.long_name = "Full width at half maximum"
        widths.units = "nm"
        widths[:] = fwhm

    with netCDF4.Dataset(paths["obs"], "w", format="NETCDF4") as ds:
        write_root(ds, "AVIRIS-3 L1B Observation Parameters", "2024-10-01T00:12:15Z")
        group = ds.createGroup("observation_parameters")
        for name, (values, attrs) in observations.items():
            variable = group.createVariable(
                name,
                "f4",
                ("lines", "samples"),
                fill_value=np.float32(-9999.0),
                chunksizes=spatial_chunks,
            )
            variable.setncatts({**attrs, "coordinates": "lat lon"})
            variable[:] = values
        group.description = (
            "Observation parameters in the raw spatial format; matches the "
            "corresponding unorthocorrected radiance image."
        )
    return paths


# -- storage-level comparison -----------------------------------------------------


def _attribute_record(
    attrs: h5py.AttributeManager,
) -> dict[str, tuple[str, str, Any, Any]]:
    import h5py

    record = {}
    for name in attrs:
        try:
            value = repr(attrs[name])
        except (OSError, TypeError, ValueError) as error:  # pragma: no cover
            value = f"<unreadable: {error}>"
        attribute = h5py.h5a.open(attrs._id, name.encode())
        type_id = attribute.get_type()
        variable_string = isinstance(type_id, h5py.h5t.TypeStringID) and bool(
            type_id.is_variable_str()
        )
        size = "vlen" if variable_string else type_id.get_size()
        record[name] = (value, type(type_id).__name__, size, attribute.shape)
    return record


def _hash_dataset(dataset: h5py.Dataset, mask: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    if dataset.dtype.kind == "O":
        digest.update(repr(dataset[()]).encode())
        return digest.hexdigest()
    if dataset.ndim == 0:
        digest.update(np.ascontiguousarray(dataset[()]).tobytes())
        return digest.hexdigest()
    row_bytes = max(1, int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize)
    step = max(1, _HASH_BLOCK_BYTES // row_bytes)
    for start in range(0, dataset.shape[0], step):
        data = np.array(dataset[start : start + step])
        if mask is not None:
            data[mask[start : start + step]] = 0
        digest.update(np.ascontiguousarray(data).view(np.uint8).tobytes())
    return digest.hexdigest()


def native_snapshot(
    path: str | Path, *, masks: Mapping[str, np.ndarray] | None = None
) -> dict[str, dict[str, Any]]:
    """Return a storage-level description of every group and dataset of a file.

    Args:
        path: NetCDF4/HDF5 file.
        masks: Dataset path to boolean mask (native order) of elements excluded
            from the data digest, for elements a patch is allowed to change.

    Returns:
        Mapping of object path to its properties.
    """
    import h5py

    masks = dict(masks or {})
    snapshot: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        snapshot["/"] = {"kind": "group", "attrs": _attribute_record(handle.attrs)}

        def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Group):
                snapshot[name] = {
                    "kind": "group",
                    "attrs": _attribute_record(obj.attrs),
                }
                return
            plist = obj.id.get_create_plist()
            snapshot[name] = {
                "kind": "dataset",
                "dtype": obj.dtype.str,
                "shape": obj.shape,
                "chunks": obj.chunks,
                "compression": obj.compression,
                "compression_opts": obj.compression_opts,
                "shuffle": obj.shuffle,
                "fillvalue": repr(obj.fillvalue),
                "layout": plist.get_layout(),
                "offset": obj.id.get_offset(),
                "storage": obj.id.get_storage_size(),
                "attrs": _attribute_record(obj.attrs),
                "sha256": _hash_dataset(obj, masks.get(name)),
            }

        handle.visititems(visit)
    return snapshot


def assert_native_preserved(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    new_objects: Iterable[str] = (),
    new_root_attrs: Iterable[str] = (),
) -> None:
    """Assert that every object of ``before`` is unchanged in ``after``.

    Dimension-scale ``REFERENCE_LIST`` attributes may change (HDF5 updates them
    when a variable is attached to an existing dimension). Objects and root
    attributes not in ``before`` must be listed in ``new_objects`` and
    ``new_root_attrs``.

    Raises:
        AssertionError: Naming the first difference.
    """
    for name, old in before.items():
        assert name in after, f"{name} disappeared"
        new = after[name]
        for key, value in old.items():
            if key == "attrs":
                for attribute, record in value.items():
                    assert attribute in new["attrs"], f"{name}:{attribute} disappeared"
                    if attribute in _SCALE_BOOKKEEPING:
                        continue
                    assert new["attrs"][attribute] == record, (
                        f"{name}:{attribute} changed: {record} -> "
                        f"{new['attrs'][attribute]}"
                    )
                added = set(new["attrs"]) - set(value) - set(_SCALE_BOOKKEEPING)
                unexpected = added - set(new_root_attrs) if name == "/" else added
                assert not unexpected, f"{name} gained attributes {sorted(unexpected)}"
            else:
                assert new[key] == value, f"{name}.{key} changed: {value} -> {new[key]}"
    added_objects = set(after) - set(before)
    unexpected_objects = added_objects - set(new_objects)
    assert not unexpected_objects, (
        f"unexpected new objects {sorted(unexpected_objects)}"
    )


__all__ = [
    "AVIRIS4_SCHEMAS",
    "EMIT_BUILDS",
    "OBS_BAND_NAMES",
    "assert_native_preserved",
    "native_snapshot",
    "write_aviris3_files",
    "write_aviris4_files",
    "write_emit_files",
]
