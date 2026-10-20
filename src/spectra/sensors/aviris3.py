"""AVIRIS-3 sensor adapter over the ORNL DAAC L1B RDN and OBS NetCDF granules.

A scene is two files, given by role:

- ``radiance``: an AVIRIS-3 L1B calibrated radiance granule (``AV3_L1B_RDN``),
  ``radiance/radiance(wavelength, lines, samples)`` in uW nm-1 cm-2 sr-1 with
  the -9999 fill value, band centres and widths in ``radiance/wavelength`` and
  ``radiance/fwhm``, per-pixel ``lon``/``lat``/``elev`` and the
  ``transverse_mercator`` grid mapping of the granule in the root group;
- ``obs``: the matching L1B observation-parameters granule, whose
  ``observation_parameters`` group holds the path length and the view and
  solar angles on the same raw grid.

The radiance stays in sensor geometry and native units: ``lines`` maps to
``along``, ``samples`` to ``across``. Projected ``x``/``y`` are computed in the
granule's own UTM grid mapping. Nothing is derived from file names.
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
from spectra.geometry.crs import (
    from_lonlat,
    projected_metre_crs,
    spatial_ref_coordinate,
)
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
    "radiance/radiance",
    {"wavelength": "band", "lines": "along", "samples": "across"},
)
# The radiance is stored in (50, 256, 256) chunks, band-major.
DEFAULT_CHUNKS: dict[str, int] = {"across": 256, "along": 256, "band": -1}
OBS_GROUP = "observation_parameters"
# Scene variable -> variable of the OBS group.
OBS_VARIABLES: dict[str, str] = {
    "path_length": "path_length",
    "vaa": "to_sensor_azimuth",
    "vza": "to_sensor_zenith",
    "saa": "to_sun_azimuth",
    "sza": "to_sun_zenith",
}
# Global attributes the RDN and OBS granules of one scene share.
GRANULE_ATTRS = ("flight_line", "scene", "time_coverage_start")


def _grid_mapping_wkt(root: xr.Dataset, path: Path) -> str:
    """Return the WKT of the single CF grid mapping in the root group.

    Raises:
        ContractError: If there is none, more than one, or it has no WKT.
    """
    mappings = [
        variable
        for variable in root.variables.values()
        if "grid_mapping_name" in variable.attrs
    ]
    if len(mappings) != 1:
        raise ContractError(
            f"{path} must have one grid-mapping variable in its root group; "
            f"found {len(mappings)}."
        )
    attrs = mappings[0].attrs
    for key in ("crs_wkt", "spatial_ref"):
        if key in attrs:
            return str(attrs[key])
    raise ContractError(f"The grid mapping of {path} has no 'crs_wkt'.")


class Aviris3Adapter:
    """Sensor adapter for AVIRIS-3 L1B granules.

    Args:
        chunks: Dask chunks of the radiance in common dimension names; they
            are rounded up to the storage chunks.
    """

    name = "aviris3"
    platform: Platform = "airborne"
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

        Only the radiance stays lazy; the geolocation and the observation
        geometry are read and the files closed.

        Raises:
            ContractError: If a file is missing or already embedded, the OBS
                granule does not match the RDN granule, or a variable or the
                grid mapping is missing.
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
        with xr.open_dataset(paths["radiance"], group="radiance") as bands:
            wavelength = np.asarray(bands["wavelength"].values)
            fwhm = np.asarray(bands["fwhm"].values)
        with xr.open_dataset(paths["radiance"]) as root:
            lon, lat, elevation = (
                to_scene_dims(root[name], self.LAYOUT).load()
                for name in ("lon", "lat", "elev")
            )
            wkt = _grid_mapping_wkt(root, paths["radiance"])
        with xr.open_dataset(paths["obs"], group=OBS_GROUP) as obs:
            missing = sorted(
                name for name in OBS_VARIABLES.values() if name not in obs.variables
            )
            if missing:
                raise ContractError(
                    f"The {OBS_GROUP!r} group of {paths['obs']} lacks {missing}."
                )
            angles = {
                key: to_scene_dims(obs[name], self.LAYOUT).load()
                for key, name in OBS_VARIABLES.items()
            }
        sizes = (radiance.sizes["across"], radiance.sizes["along"])
        if angles["sza"].shape != sizes:
            raise ContractError(
                f"{paths['obs']} and {paths['radiance']} differ in size."
            )

        valid_location = np.isfinite(lon.values) & np.isfinite(lat.values)
        if not valid_location.any():
            raise ContractError(f"{paths['radiance']} has no finite geolocation.")
        crs = projected_metre_crs(wkt)
        x_values, y_values = from_lonlat(crs, lon.values, lat.values)
        n_across, n_along = sizes
        index = {
            "across": np.arange(n_across),
            "along": np.arange(n_along),
            "band": np.arange(radiance.sizes["band"]),
        }
        x = xr.DataArray(x_values, dims=("across", "along"), attrs={"units": "m"})
        y = xr.DataArray(y_values, dims=("across", "along"), attrs={"units": "m"})
        # The renamed ``wavelength`` dimension coordinate is replaced by the
        # band index; wavelengths are their own coordinate.
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
        geometry = {
            key: value.drop_vars(list(value.coords)) for key, value in angles.items()
        }
        for key in ("sza", "saa", "vza", "vaa"):
            scene[key] = geometry[key].assign_attrs(units="degree")
        surface = elevation.drop_vars(list(elevation.coords))
        scene["surface_elevation"] = surface.assign_attrs(units="m")
        # The OBS path length runs from the sensor to the ground along the view.
        altitude = (
            geometry["path_length"] * np.cos(np.deg2rad(geometry["vza"])) + surface
        )
        scene["sensor_altitude"] = altitude.assign_attrs(units="m")
        scene["pixel_area"] = pixel_area_from_centres(x, y)
        mid_band = scene.sizes["band"] // 2
        valid = scene["radiance"].isel(band=mid_band).notnull() & xr.DataArray(
            valid_location, dims=("across", "along")
        )
        scene["valid"] = valid.drop_vars(
            [name for name in valid.coords if name not in ("across", "along")]
        )
        scene_id = "_".join(
            part for part in (attrs["flight_line"], attrs["scene"]) if part
        )
        scene.attrs.update(
            {
                "sensor": self.name,
                "platform": self.platform,
                "scene_id": scene_id or paths["radiance"].stem,
                "acquisition_time": np.datetime_as_string(time, unit="s"),
                "geometry_space": "sensor",
                "crs": crs.to_string(),
                "source_files": json.dumps({k: str(v) for k, v in paths.items()}),
                "adapter_version": spectra.__version__,
            }
        )
        return scene


__all__ = [
    "DEFAULT_CHUNKS",
    "GRANULE_ATTRS",
    "LAYOUT",
    "OBS_GROUP",
    "OBS_VARIABLES",
    "ROLES",
    "Aviris3Adapter",
]
