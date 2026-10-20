"""AVIRIS-4 sensor adapter over the L1 radiance and PARGE metadata NetCDF files.

A scene is two files, given by role:

- ``radiance``: ``radiance(across, along, band)`` with ``wavelength`` and
  ``fwhm`` data variables and a scalar ``time``;
- ``metadata``: projected ``x``/``y`` (``epsg`` attributes), ``lon``/``lat``,
  ``elevation``, the signed scan zenith (degrees x 100), the view azimuth
  (degrees x 10), the aircraft altitude and ``cast_shadow`` whose ``zenith``
  and ``azimuth`` attributes are the solar angles of the flight line.

Both the files of the 2024/2025 campaigns (contiguous, ``days since`` time)
and the newer converter output (chunked, epoch-seconds time, pixel sizes) are
read. Nothing is derived from file names.
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
from spectra.geometry.crs import projected_crs_from_xy, spatial_ref_coordinate
from spectra.sensors.native import aligned_chunks, open_native_radiance, resolve_files
from spectra.types import NativeLayout, Platform

ROLES: dict[str, bool] = {"radiance": True, "metadata": True}
LAYOUT = NativeLayout(
    "radiance", "radiance", {"across": "across", "along": "along", "band": "band"}
)
DEFAULT_CHUNKS: dict[str, int] = {"across": 16, "along": 512, "band": -1}
METADATA_CHUNKS: dict[str, int] = {"across": 64, "along": -1}


def _scalar_time(dataset: xr.Dataset, path: Path) -> np.datetime64:
    """Return the scalar acquisition time of an AVIRIS-4 file.

    Raises:
        ContractError: If the file has no scalar datetime ``time``.
    """
    if "time" not in dataset.variables or dataset["time"].ndim != 0:
        raise ContractError(f"{path} has no scalar 'time' variable.")
    value = np.asarray(dataset["time"].values)
    if not np.issubdtype(value.dtype, np.datetime64):
        raise ContractError(f"The 'time' of {path} does not decode to a datetime.")
    return np.datetime64(value.astype("datetime64[ns]").reshape(-1)[0])


class Aviris4Adapter:
    """Sensor adapter for AVIRIS-4 flight lines.

    Args:
        chunks: Dask chunks of the radiance in common dimension names; they
            are rounded up to the storage chunks of chunked files.
    """

    name = "aviris4"
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
        """Open a flight line lazily as a ``SensorScene`` dataset.

        Raises:
            ContractError: If a file is missing, already embedded, or lacks
                the time, geolocation or solar angles.
        """
        paths = resolve_files(self.ROLES, files)
        check_not_embedded(paths["radiance"])
        radiance = open_native_radiance(
            paths["radiance"], self.LAYOUT, chunks=dict(chunks or self.chunks)
        )
        with xr.open_dataset(paths["radiance"]) as header:
            time = _scalar_time(header, paths["radiance"])
            wavelength = np.asarray(header["wavelength"].values)
            fwhm = np.asarray(header["fwhm"].values)
        metadata = xr.open_dataset(
            paths["metadata"],
            chunks=aligned_chunks(paths["metadata"], "x", METADATA_CHUNKS),
        )
        if (
            "time" in metadata.variables
            and _scalar_time(metadata, paths["metadata"]) != time
        ):
            raise ContractError(
                f"{paths['metadata']} belongs to another acquisition than "
                f"{paths['radiance']} (different 'time')."
            )
        metadata = metadata.reset_coords(drop=True)
        for dim in ("across", "along"):
            if metadata.sizes.get(dim) != radiance.sizes[dim]:
                raise ContractError(
                    f"{paths['metadata']} has {metadata.sizes.get(dim)} pixels "
                    f"along {dim!r}; the radiance has {radiance.sizes[dim]}."
                )
        attrs = metadata["cast_shadow"].attrs if "cast_shadow" in metadata else {}
        if "zenith" not in attrs or "azimuth" not in attrs:
            raise ContractError(
                f"{paths['metadata']} lacks the solar angles (cast_shadow zenith "
                "and azimuth attributes)."
            )
        crs = projected_crs_from_xy(metadata["x"], metadata["y"])

        scene = xr.Dataset(
            {"radiance": radiance},
            coords={
                "wavelength": ("band", wavelength, {"units": "nm"}),
                "fwhm": ("band", fwhm, {"units": "nm"}),
                **{name: metadata[name] for name in ("lon", "lat", "x", "y")},
                "spatial_ref": spatial_ref_coordinate(crs),
                "time": time,
            },
        )
        scene["sza"] = xr.DataArray(float(attrs["zenith"]), attrs={"units": "degree"})
        scene["saa"] = xr.DataArray(float(attrs["azimuth"]), attrs={"units": "degree"})
        # PARGE stores a signed scan zenith (left/right of nadir); AMFs need |vza|.
        scene["vza"] = (abs(metadata["scan_zenith"]) / 100.0).assign_attrs(
            units="degree"
        )
        scene["vaa"] = (metadata["absolute_azimuth"] / 10.0).assign_attrs(
            units="degree"
        )
        scene["surface_elevation"] = metadata["elevation"].assign_attrs(units="m")
        scene["sensor_altitude"] = metadata["aircraft_altitude"].assign_attrs(units="m")
        if (
            "pixel_size_across_track" in metadata
            and "pixel_size_along_track" in metadata
        ):
            area = (
                metadata["pixel_size_across_track"] * metadata["pixel_size_along_track"]
            )
            scene["pixel_area"] = area.assign_attrs(units="m2")
        else:
            scene["pixel_area"] = pixel_area_from_centres(metadata["x"], metadata["y"])
        mid_band = scene.sizes["band"] // 2
        valid = (
            scene["radiance"].isel(band=mid_band).notnull() & metadata["x"].notnull()
        )
        scene["valid"] = valid.drop_vars(
            [name for name in valid.coords if name not in ("across", "along")]
        )
        scene.attrs.update(
            {
                "sensor": self.name,
                "platform": self.platform,
                # An opaque label for messages; SPECTRA never parses it.
                "scene_id": paths["radiance"].stem,
                "acquisition_time": np.datetime_as_string(time, unit="s"),
                "geometry_space": "sensor",
                "crs": crs.to_string(),
                "source_files": json.dumps({k: str(v) for k, v in paths.items()}),
                "adapter_version": spectra.__version__,
            }
        )
        # Radiance and geometry come from different files and chunkings.
        return scene.unify_chunks()


__all__ = ["DEFAULT_CHUNKS", "LAYOUT", "ROLES", "Aviris4Adapter"]
