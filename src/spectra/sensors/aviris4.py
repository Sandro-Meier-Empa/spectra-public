"""AVIRIS-4 sensor adapter over the NetCDF products written by ``aviris4_ch4``.

Every scene is given as explicit files (a scene table or command-line paths):
the L1 radiance cube (``radiance(across, along, band)`` with ``wavelength`` and
``fwhm`` stored as data variables), the PARGE metadata file (projected ``x``/
``y`` with ``epsg`` attributes, ``lon``, ``lat``, terrain, viewing geometry),
and optionally the shadow index, the cloud mask and the matched-filter
product. The adapter derives nothing from directory layouts or file names.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import xarray as xr

import spectra
from spectra.contracts.column_product import convert_column_units
from spectra.contracts.errors import ContractError, MissingExtraError
from spectra.contracts.sensor_scene import RADIANCE_UNITS
from spectra.geometry.crs import projected_crs_from_xy, spatial_ref_coordinate
from spectra.sensors.table import filter_rows, read_scene_table
from spectra.types import Platform, SceneGeometry, SceneRecord

DEFAULT_CHUNKS: dict[str, int] = {"across": 8, "along": -1, "band": -1}
# Roles of the scene table / command line: role -> required.
ROLES: dict[str, bool] = {
    "radiance": True,
    "metadata": True,
    "shadow": False,
    "cloud": False,
    "product": False,
}


def _aviris4_misc() -> ModuleType:
    """Import ``aviris4_ch4.misc`` lazily."""
    try:
        import aviris4_ch4.misc as misc
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise MissingExtraError("aviris4", "aviris4_ch4", error) from error
    return misc


def _scene_time(radiance: xr.Dataset, path: Path) -> np.datetime64:
    """Return the scalar acquisition time of a radiance file."""
    if "time" in radiance and radiance["time"].ndim == 0:
        value = radiance["time"].values
        if np.issubdtype(np.asarray(value).dtype, np.datetime64):
            return np.datetime64(
                np.asarray(value).astype("datetime64[ns]").reshape(-1)[0]
            )
    misc = _aviris4_misc()
    stamp = misc.extract_datetime(path, tz="UTC")
    return np.datetime64(stamp.tz_convert("UTC").tz_localize(None))


def _isfinite(array: xr.DataArray) -> xr.DataArray:
    """Return a dask-aware finite mask as a DataArray."""
    return xr.apply_ufunc(np.isfinite, array, dask="allowed")


def _pixel_area(x: xr.DataArray, y: xr.DataArray) -> xr.DataArray:
    """Estimate pixel areas from neighbouring centre spacings (dask-aware).

    The area is the parallelogram spanned by the across- and along-track
    centre spacing vectors, so skew between the two directions is accounted
    for; this matches the polygon areas of the regridders to the lattice's
    local curvature.
    """
    across, along = (str(dim) for dim in x.dims)
    # Work without index coordinates: padding would duplicate index values.
    plain_x = xr.DataArray(x.data, dims=x.dims)
    plain_y = xr.DataArray(y.data, dims=y.dims)
    dxa = plain_x.diff(across).pad({across: (0, 1)}, mode="edge")
    dya = plain_y.diff(across).pad({across: (0, 1)}, mode="edge")
    dxl = plain_x.diff(along).pad({along: (0, 1)}, mode="edge")
    dyl = plain_y.diff(along).pad({along: (0, 1)}, mode="edge")
    area = abs(dxa * dyl - dya * dxl).transpose(across, along)
    area = area.assign_coords({across: x[across], along: x[along]})
    area.attrs = {"units": "m2", "long_name": "pixel area from centre spacing"}
    return area


def mf_product_to_column_product(
    product: xr.Dataset,
    *,
    attrs: Mapping[str, Any],
    geometry: xr.Dataset | None = None,
    gas: str = "ch4",
) -> xr.Dataset:
    """Convert an ``aviris4_ch4`` matched-filter product to a ``ColumnProduct``.

    Args:
        product: Matched-filter output with ``CH4`` and ``uncertainty`` in
            ppm-m and optional ``sensitivity``, ``retrieval_valid``,
            ``background_valid``.
        attrs: Scene attributes (``sensor``, ``scene_id``, ``background_group``).
        geometry: Metadata dataset supplying ``x``, ``y``, ``lon``, ``lat`` on
            the same grid, if available.
        gas: Lower-case gas name of the output variables.

    Returns:
        Sensor-space column product in kg m-2.
    """
    label = gas.upper()
    if label not in product or "uncertainty" not in product:
        raise ContractError(
            f"Matched-filter product must contain '{label}' and 'uncertainty'."
        )
    output = xr.Dataset(coords={"across": product["across"], "along": product["along"]})
    output[f"{gas}_mass"] = convert_column_units(product[label], "kg m-2", gas=label)
    output[f"{gas}_mass"].attrs.update(
        {"native_units": product[label].attrs.get("units"), "native_variable": label}
    )
    output[f"{gas}_mass_uncertainty"] = convert_column_units(
        product["uncertainty"], "kg m-2", gas=label
    )
    for name in ("sensitivity", "background_valid"):
        if name in product:
            output[name] = product[name]
    if "sensitivity" in output:
        # The matched filter labels the dimensionless factor "-"; prism uses "1".
        output["sensitivity"].attrs["units"] = "1"
    # A retrieval without a finite, positive uncertainty is not usable (the
    # noise model is undefined for negative radiances), whatever the filter's
    # own validity flag says.
    finite = (
        _isfinite(output[f"{gas}_mass"])
        & _isfinite(output[f"{gas}_mass_uncertainty"])
        & (output[f"{gas}_mass_uncertainty"] > 0)
    )
    if "retrieval_valid" in product:
        valid = product["retrieval_valid"].astype(bool) & finite
    else:
        valid = finite
    output["retrieval_valid"] = valid.astype(bool)
    if geometry is not None:
        coords = {
            name: geometry[name]
            for name in ("lon", "lat", "x", "y")
            if name in geometry and geometry[name].dims == ("across", "along")
        }
        output = output.assign_coords(coords)
        if "x" in coords and "y" in coords:
            output = output.assign_coords(
                spatial_ref=spatial_ref_coordinate(
                    projected_crs_from_xy(coords["x"], coords["y"])
                )
            )
            output["pixel_area"] = _pixel_area(coords["x"], coords["y"])
    if "time" in product.coords:
        output = output.assign_coords(time=product["time"])
    output.attrs.update(product.attrs)
    output.attrs.update(dict(attrs))
    output.attrs.update(
        {
            "geometry_space": "sensor",
            "retriever": output.attrs.get("retriever", "aviris4_mf"),
            "gas": label,
            "contract": "spectra.ColumnProduct",
        }
    )
    return output


class Aviris4Adapter:
    """Sensor adapter for AVIRIS-4 flight lines processed with ``aviris4_ch4``.

    Scenes are specified explicitly: a scene table (CSV/YAML, see
    :mod:`spectra.sensors.table`) or :meth:`record_from_files`. Roles are
    ``radiance`` and ``metadata`` (required), ``shadow``, ``cloud`` and
    ``product`` (optional).

    Args:
        chunks: Dask chunks used to open radiance cubes.
        shadow_threshold: Shadow index above which a pixel is shadowed.
    """

    name: str = "aviris4"
    platform: Platform = "airborne"
    ROLES: Mapping[str, bool] = ROLES

    def __init__(
        self,
        *,
        chunks: Mapping[str, int] | None = None,
        shadow_threshold: float = 0.25,
    ) -> None:
        """Store the opening options."""
        self.chunks = dict(chunks or DEFAULT_CHUNKS)
        self.shadow_threshold = float(shadow_threshold)

    # -- discovery ---------------------------------------------------------
    def _record(
        self,
        scene_id: str,
        uris: Mapping[str, str],
        background_group: str | None,
        extra: Mapping[str, str] | None = None,
    ) -> SceneRecord:
        """Build a record from resolved files, reading only file headers."""
        with xr.open_dataset(uris["radiance"]) as radiance:
            time = _scene_time(radiance, Path(uris["radiance"]))
        with xr.open_dataset(uris["metadata"]) as metadata:
            lon = metadata["lon"][::64, ::256].values
            lat = metadata["lat"][::64, ::256].values
        centre = (float(np.nanmean(lon)), float(np.nanmean(lat)))
        bbox = (
            float(np.nanmin(lon)),
            float(np.nanmin(lat)),
            float(np.nanmax(lon)),
            float(np.nanmax(lat)),
        )
        return SceneRecord(
            sensor=self.name,
            scene_id=scene_id,
            platform="airborne",
            acquisition_time=datetime.fromtimestamp(
                float(time.astype("datetime64[s]").astype("int64")), tz=UTC
            ),
            background_group=background_group or f"{self.name}:{scene_id}",
            uris=dict(uris),
            center_lonlat=centre,
            bbox_lonlat=bbox,
            has_radiance=True,
            has_product="product" in uris,
            extra=dict(extra or {}),
        )

    def list_scenes(
        self,
        scenes: str | Path | Sequence[Mapping[str, object]],
        **filters: object,
    ) -> list[SceneRecord]:
        """Return the scenes of a scene table.

        Args:
            scenes: CSV/YAML scene table or in-memory rows with the columns
                ``scene_id`` (optional), ``radiance``, ``metadata``,
                ``shadow``, ``cloud``, ``product``, ``background_group``
                (optional); extra columns are kept in ``SceneRecord.extra``.
            **filters: ``scene_id`` (string or iterable) or equality filters
                on extra columns.

        Returns:
            Records sorted by acquisition time.
        """
        rows = filter_rows(read_scene_table(scenes, roles=self.ROLES), **filters)
        records = [
            self._record(row.scene_id, row.uris, row.background_group, row.extra)
            for row in rows
        ]
        return sorted(records, key=lambda item: item.acquisition_time)

    def record_from_files(
        self,
        radiance: str | Path,
        metadata: str | Path,
        *,
        shadow: str | Path | None = None,
        cloud: str | Path | None = None,
        product: str | Path | None = None,
        scene_id: str | None = None,
        background_group: str | None = None,
    ) -> SceneRecord:
        """Build one record from explicit file paths."""
        row = {
            "scene_id": scene_id or "",
            "radiance": str(radiance),
            "metadata": str(metadata),
            "shadow": "" if shadow is None else str(shadow),
            "cloud": "" if cloud is None else str(cloud),
            "product": "" if product is None else str(product),
            "background_group": background_group or "",
        }
        return self.list_scenes([row])[0]

    def background_group(self, record: SceneRecord) -> str:
        """Return the leakage group recorded for a scene."""
        return record.background_group

    # -- opening -----------------------------------------------------------
    def _metadata(self, record: SceneRecord) -> xr.Dataset:
        return xr.open_dataset(record.uris["metadata"])

    def scene_geometry(self, record: SceneRecord) -> SceneGeometry:
        """Return the projected pixel geometry without opening the radiance."""
        metadata = self._metadata(record)
        return SceneGeometry(
            x=metadata["x"],
            y=metadata["y"],
            crs=projected_crs_from_xy(metadata["x"], metadata["y"]),
            lon=metadata["lon"],
            lat=metadata["lat"],
            source=record.scene_id,
        )

    def open_scene(
        self, record: SceneRecord, *, chunks: Mapping[str, int] | None = None
    ) -> xr.Dataset:
        """Open a flight line lazily as a ``SensorScene`` dataset."""
        radiance = xr.open_dataset(
            record.uris["radiance"], chunks=dict(chunks or self.chunks)
        )
        radiance = radiance.set_coords(
            [name for name in ("wavelength", "fwhm") if name in radiance.data_vars]
        )
        metadata = self._metadata(record)
        scene = radiance.copy()
        scene["radiance"].attrs.setdefault("units", RADIANCE_UNITS)
        coords = {name: metadata[name] for name in ("lon", "lat", "x", "y")}
        scene = scene.assign_coords(coords)
        crs = projected_crs_from_xy(metadata["x"], metadata["y"])
        scene = scene.assign_coords(spatial_ref=spatial_ref_coordinate(crs))
        time = _scene_time(radiance, Path(record.uris["radiance"]))
        scene = scene.assign_coords(time=time)

        cast = metadata["cast_shadow"].attrs if "cast_shadow" in metadata else {}
        if "zenith" in cast:
            sza = float(cast["zenith"])
            saa = float(cast.get("azimuth", np.nan))
        else:
            misc = _aviris4_misc()
            lat = float(metadata["lat"][::64, ::256].mean())
            lon = float(metadata["lon"][::64, ::256].mean())
            sza = float(misc.calculate_solar_zenith_angle(lat, lon, time))
            saa = float("nan")
        scene["sza"] = xr.DataArray(sza, attrs={"units": "degrees"})
        scene["saa"] = xr.DataArray(saa, attrs={"units": "degrees"})
        # PARGE stores a signed scan zenith (left/right of nadir); AMFs need |vza|.
        scene["vza"] = (abs(metadata["scan_zenith"]) / 100.0).assign_attrs(
            units="degrees"
        )
        scene["vaa"] = (metadata["absolute_azimuth"] / 10.0).assign_attrs(
            units="degrees"
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
            scene["pixel_area"] = _pixel_area(metadata["x"], metadata["y"])

        mid_band = scene.sizes["band"] // 2
        valid = _isfinite(scene["radiance"].isel(band=mid_band)) & _isfinite(
            metadata["x"]
        )
        scene["valid"] = valid.astype(bool).drop_vars(
            [name for name in ("wavelength", "fwhm", "band") if name in valid.coords]
        )
        if "shadow" in record.uris:
            with xr.open_dataset(record.uris["shadow"]) as shadow:
                index = shadow["shadow_index"].load()
            scene["shadow_index"] = index
            scene["shadow"] = (index > self.shadow_threshold).assign_attrs(
                threshold=self.shadow_threshold
            )
        if "cloud" in record.uris:
            with xr.open_dataset(record.uris["cloud"]) as cloud:
                for name in ("cloud_mask", "opaque_cloud", "cirrus"):
                    if name in cloud:
                        scene["cloud"] = cloud[name].astype(bool).load()
                        break

        scene.attrs.update(
            {
                "sensor": self.name,
                "platform": self.platform,
                "scene_id": record.scene_id,
                "acquisition_time": np.datetime_as_string(time, unit="s"),
                "background_group": record.background_group,
                "geometry_space": "sensor",
                "source_uris": json.dumps(dict(record.uris)),
                "adapter_version": spectra.__version__,
                "crs": crs.to_string(),
            }
        )
        return scene

    def quality_masks(self, scene: xr.Dataset, record: SceneRecord) -> xr.Dataset:
        """Return the boolean masks carried by an opened scene."""
        names = [
            name for name in ("valid", "shadow", "cloud", "water") if name in scene
        ]
        return scene[names]

    def open_product(self, record: SceneRecord) -> xr.Dataset:
        """Open the matched-filter product of a flight line as a ``ColumnProduct``."""
        if "product" not in record.uris:
            raise ContractError(
                f"Scene {record.scene_id} has no matched-filter product."
            )
        product = xr.open_dataset(record.uris["product"])
        metadata = self._metadata(record)
        attrs = {
            "sensor": self.name,
            "platform": self.platform,
            "scene_id": record.scene_id,
            "background_group": record.background_group,
            "source_uris": json.dumps(dict(record.uris)),
        }
        return mf_product_to_column_product(product, attrs=attrs, geometry=metadata)


__all__ = ["ROLES", "Aviris4Adapter", "mf_product_to_column_product"]
