"""EMIT sensor adapter over the L1B RAD and OBS NetCDF granules.

A scene is two files, given by role:

- ``radiance``: an EMIT L1B RAD granule, ``radiance(downtrack, crosstrack,
  bands)`` in uW cm-2 sr-1 nm-1 with the -9999 fill value, band centres and
  widths in ``sensor_band_parameters`` and per-pixel ``lon``/``lat``/``elev``
  in ``location``;
- ``obs``: the matching L1B OBS granule, whose ``obs`` bands hold the path
  length and the view and solar angles.

The radiance stays in sensor geometry and native units: downtrack maps to
``along``, crosstrack to ``across``. Projected ``x``/``y`` are computed in the
UTM zone of the granule centre. Nothing is derived from file names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.embedded import check_not_embedded
from spectra.contracts.errors import ContractError
from spectra.geometry.area import pixel_area_from_centres
from spectra.geometry.crs import from_lonlat, spatial_ref_coordinate, utm_crs
from spectra.sensors.native import (
    coverage_start,
    global_attrs,
    open_native_radiance,
    resolve_files,
    to_scene_dims,
)
from spectra.types import NativeLayout, Platform

ROLES: dict[str, bool] = {"radiance": True, "obs": True}
LAYOUT = NativeLayout(
    "radiance",
    "radiance",
    {"downtrack": "along", "crosstrack": "across", "bands": "band"},
)
# The radiance is stored contiguously, downtrack-major: chunk along downtrack.
DEFAULT_CHUNKS: dict[str, int] = {"along": 32, "across": -1, "band": -1}
# OBS band index and the start of its name in ``observation_bands``.
OBS_BANDS: dict[str, tuple[int, str]] = {
    "path_length": (0, "Path length"),
    "vaa": (1, "To-sensor azimuth"),
    "vza": (2, "To-sensor zenith"),
    "saa": (3, "To-sun azimuth"),
    "sza": (4, "To-sun zenith"),
}
# Global attributes the RAD and OBS granules of one scene share.
GRANULE_ATTRS = ("flight_line", "time_coverage_start")


class EmitAdapter:
    """Sensor adapter for EMIT L1B granules.

    Args:
        chunks: Dask chunks of the radiance in common dimension names.
    """

    name = "emit"
    platform: Platform = "spaceborne"
    ROLES: Mapping[str, bool] = ROLES
    LAYOUT: NativeLayout = LAYOUT

    def __init__(self, chunks: Mapping[str, int] | None = None) -> None:
        """Store the radiance chunks."""
        self.chunks = dict(DEFAULT_CHUNKS if chunks is None else chunks)

    def open_scene(
        self,
        files: Mapping[str, str | Path],
        *,
        chunks: Mapping[str, int] | None = None,
    ) -> xr.Dataset:
        """Open a granule lazily as a ``SensorScene`` dataset.

        Raises:
            ContractError: If a file is missing or already embedded, the OBS
                granule does not match the RAD granule, or its bands differ
                from the EMIT layout.
        """
        paths = resolve_files(self.ROLES, files)
        check_not_embedded(paths["radiance"])
        radiance = open_native_radiance(
            paths["radiance"], self.LAYOUT, chunks=dict(chunks or self.chunks)
        )
        attrs = global_attrs(paths["radiance"], GRANULE_ATTRS)
        obs_attrs = global_attrs(paths["obs"], GRANULE_ATTRS)
        if attrs != obs_attrs:
            raise ContractError(
                f"{paths['obs']} is not the OBS granule of {paths['radiance']}: "
                f"{obs_attrs} differ from {attrs}."
            )
        time = coverage_start(attrs["time_coverage_start"], paths["radiance"])
        with xr.open_dataset(
            paths["radiance"], group="sensor_band_parameters"
        ) as bands:
            wavelength = np.asarray(bands["wavelengths"].values)
            fwhm = np.asarray(bands["fwhm"].values)
        with xr.open_dataset(paths["radiance"], group="location") as location:
            lon = to_scene_dims(location["lon"], self.LAYOUT).load()
            lat = to_scene_dims(location["lat"], self.LAYOUT).load()
            elevation = to_scene_dims(location["elev"], self.LAYOUT).load()
        with xr.open_dataset(paths["obs"], group="sensor_band_parameters") as names:
            band_names = [str(name) for name in names["observation_bands"].values]
        for key, (index, prefix) in OBS_BANDS.items():
            if index >= len(band_names) or not band_names[index].startswith(prefix):
                raise ContractError(
                    f"OBS band {index} of {paths['obs']} is not {prefix!r} ({key})."
                )
        # The OBS granule is small; its bands are read eagerly and the file is
        # closed. Keeping it open lazily while it is opened again crashes HDF5
        # 1.14 on the string fill value of observation_bands.
        bands_needed = [index for index, _ in OBS_BANDS.values()]
        with xr.open_dataset(paths["obs"]) as observations:
            obs = observations["obs"].isel(bands=bands_needed).load()
        if (obs.sizes["downtrack"], obs.sizes["crosstrack"]) != (
            radiance.sizes["along"],
            radiance.sizes["across"],
        ):
            raise ContractError(
                f"{paths['obs']} and {paths['radiance']} differ in size."
            )
        angles = {
            key: to_scene_dims(obs.isel(bands=position, drop=True), self.LAYOUT)
            for position, key in enumerate(OBS_BANDS)
        }

        valid_location = np.isfinite(lon.values) & np.isfinite(lat.values)
        if not valid_location.any():
            raise ContractError(f"{paths['radiance']} has no finite geolocation.")
        crs = utm_crs(
            float(np.median(lon.values[valid_location])),
            float(np.median(lat.values[valid_location])),
        )
        x_values, y_values = from_lonlat(crs, lon.values, lat.values)
        n_across, n_along = radiance.sizes["across"], radiance.sizes["along"]
        index = {
            "across": np.arange(n_across),
            "along": np.arange(n_along),
            "band": np.arange(radiance.sizes["band"]),
        }
        x = xr.DataArray(x_values, dims=("across", "along"), attrs={"units": "m"})
        y = xr.DataArray(y_values, dims=("across", "along"), attrs={"units": "m"})
        scene = xr.Dataset(
            {"radiance": radiance.assign_coords(index)},
            coords={
                "wavelength": ("band", wavelength, {"units": "nm"}),
                "fwhm": ("band", fwhm, {"units": "nm"}),
                "lon": lon.drop_vars(list(lon.coords)),
                "lat": lat.drop_vars(list(lat.coords)),
                "x": x,
                "y": y,
                "spatial_ref": spatial_ref_coordinate(crs),
                "time": time,
            },
        )
        for key in ("sza", "saa", "vza", "vaa"):
            scene[key] = angles[key].assign_attrs(units="degree")
        surface = elevation.drop_vars(list(elevation.coords))
        scene["surface_elevation"] = surface.assign_attrs(units="m")
        # The OBS path length runs from the sensor to the ground along the view.
        altitude = angles["path_length"] * np.cos(np.deg2rad(angles["vza"])) + surface
        scene["sensor_altitude"] = altitude.assign_attrs(units="m")
        scene["pixel_area"] = pixel_area_from_centres(x, y)
        mid_band = scene.sizes["band"] // 2
        valid = scene["radiance"].isel(band=mid_band).notnull() & xr.DataArray(
            valid_location, dims=("across", "along")
        )
        scene["valid"] = valid.drop_vars(
            [name for name in valid.coords if name not in ("across", "along")]
        )
        scene.attrs.update(
            {
                "sensor": self.name,
                "platform": self.platform,
                "scene_id": attrs["flight_line"] or paths["radiance"].stem,
                "acquisition_time": np.datetime_as_string(time, unit="s"),
                "geometry_space": "sensor",
                "crs": crs.to_string(),
                "source_files": json.dumps({k: str(v) for k, v in paths.items()}),
                "adapter_version": spectra.__version__,
            }
        )
        # Radiance and geometry come from different files and chunkings.
        return scene.unify_chunks()


__all__ = [
    "DEFAULT_CHUNKS",
    "GRANULE_ATTRS",
    "LAYOUT",
    "OBS_BANDS",
    "ROLES",
    "EmitAdapter",
]
