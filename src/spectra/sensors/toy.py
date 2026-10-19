"""Synthetic sensor adapter producing deterministic toy scenes."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import xarray as xr

from spectra.contracts.sensor_scene import PLATFORMS, SPATIAL_DIMS
from spectra.geometry.crs import projected_metre_crs, to_lonlat
from spectra.testing.synthetic import (
    DEFAULT_SCENE_CRS,
    make_sensor_scene,
    scene_pixel_centres,
    scene_seed,
)
from spectra.types import Platform, SceneRecord

_FIRST_ACQUISITION = datetime(2026, 6, 1, 10, 30, tzinfo=UTC)
_LATTICE_OPTIONS = frozenset(
    {
        "n_across",
        "n_along",
        "pixel_across_m",
        "pixel_along_m",
        "origin_xy",
        "grid_rotation_deg",
    }
)
_RESERVED_OPTIONS = frozenset(
    {
        "scene_id",
        "seed",
        "sensor",
        "acquisition_time",
        "background_group",
        "lazy",
    }
)
_SCENE_OPTIONS = frozenset(inspect.signature(make_sensor_scene).parameters)


class ToySensorAdapter:
    """Sensor adapter serving synthetic scenes from :mod:`spectra.testing`.

    Implements ``SensorAdapter``. Scenes are generated
    on demand from ``make_sensor_scene`` with a seed derived from the scene
    id, so every record opens to the same data in every process.

    Args:
        **scene_options: Keyword arguments forwarded to ``make_sensor_scene``
            for every scene, for example ``n_across`` or ``noise_sigma``.
            ``scene_id``, ``seed``, ``sensor``, ``acquisition_time``,
            ``background_group`` and ``lazy`` are set per record and cannot
            be given.

    Raises:
        ValueError: If an option is reserved or unknown.
    """

    name: str = "toy"
    platform: Platform = "airborne"

    def __init__(self, **scene_options: object) -> None:
        reserved = _RESERVED_OPTIONS.intersection(scene_options)
        if reserved:
            raise ValueError(f"Options {sorted(reserved)} are set per scene record.")
        unknown = set(scene_options).difference(_SCENE_OPTIONS)
        if unknown:
            raise ValueError(
                f"Unknown make_sensor_scene options {sorted(unknown)}; choose from "
                f"{sorted(_SCENE_OPTIONS - _RESERVED_OPTIONS)}."
            )
        options: dict[str, Any] = dict(scene_options)
        platform = options.pop("platform", self.platform)
        if platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}.")
        self.platform = platform
        self.scene_options = options

    def _footprint_lonlat(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
        """Return the centre and bounds of the configured lattice in degrees."""
        lattice = {
            key: value
            for key, value in self.scene_options.items()
            if key in _LATTICE_OPTIONS
        }
        x, y = scene_pixel_centres(**lattice)
        crs = projected_metre_crs(self.scene_options.get("crs", DEFAULT_SCENE_CRS))
        lon, lat = to_lonlat(crs, x, y)
        centre = (float(lon.mean()), float(lat.mean()))
        bounds = (
            float(lon.min()),
            float(lat.min()),
            float(lon.max()),
            float(lat.max()),
        )
        return centre, bounds

    def list_scenes(
        self,
        scenes: str | Path | Sequence[Mapping[str, object]] | None = None,
        n: int = 3,
        **filters: object,
    ) -> list[SceneRecord]:
        """Return ``n`` synthetic scene records ``toy-0001`` to ``toy-000n``.

        Args:
            scenes: Ignored; the scenes need no files.
            n: Number of records.
            **filters: Only ``scene_id`` (a string or an iterable of strings)
                is supported.

        Returns:
            Records with acquisition times one day apart.

        Raises:
            ValueError: If an unsupported filter is given.
        """
        wanted = filters.pop("scene_id", None)
        if filters:
            raise ValueError(f"Unsupported toy scene filters: {sorted(filters)}.")
        selected: set[str] | None
        if wanted is None:
            selected = None
        elif isinstance(wanted, str):
            selected = {wanted}
        elif isinstance(wanted, Iterable):
            selected = {str(item) for item in wanted}
        else:
            raise ValueError("scene_id filter must be a string or iterable of strings.")
        centre, bounds = self._footprint_lonlat()
        records: list[SceneRecord] = []
        for index in range(int(n)):
            scene_id = f"toy-{index + 1:04d}"
            if selected is not None and scene_id not in selected:
                continue
            records.append(
                SceneRecord(
                    sensor=self.name,
                    scene_id=scene_id,
                    platform=self.platform,
                    acquisition_time=_FIRST_ACQUISITION + timedelta(days=index),
                    background_group=self.background_group_of(scene_id),
                    uris={},
                    center_lonlat=centre,
                    bbox_lonlat=bounds,
                    has_radiance=True,
                    extra={"scenes": str(scenes), "index": index},
                )
            )
        return records

    @staticmethod
    def background_group_of(scene_id: str) -> str:
        """Return the leakage group of a scene id."""
        return f"toy:{scene_id}"

    def background_group(self, record: SceneRecord) -> str:
        """Return the leakage group of a record."""
        return self.background_group_of(record.scene_id)

    def open_scene(
        self, record: SceneRecord, *, chunks: Mapping[str, int] | None = None
    ) -> xr.Dataset:
        """Generate the scene of a record as a lazy ``SensorScene`` dataset."""
        scene = make_sensor_scene(
            scene_id=record.scene_id,
            seed=scene_seed(record.scene_id),
            sensor=self.name,
            platform=self.platform,
            acquisition_time=record.acquisition_time,
            background_group=self.background_group(record),
            lazy=True,
            **self.scene_options,
        )
        if chunks:
            scene["radiance"] = scene["radiance"].chunk(dict(chunks))
        return scene

    def quality_masks(self, scene: xr.Dataset, record: SceneRecord) -> xr.Dataset:
        """Return the ``valid`` mask of the scene and empty cloud and shadow masks."""
        del record
        valid = scene["valid"].transpose(*SPATIAL_DIMS).astype(bool)
        flagged = xr.zeros_like(valid, dtype=bool)
        return xr.Dataset(
            {
                "valid": valid,
                "cloud": flagged.assign_attrs(long_name="cloud flagged"),
                "shadow": flagged.assign_attrs(long_name="shadow flagged"),
            }
        )


__all__ = ["ToySensorAdapter"]
