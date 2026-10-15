"""Table schema, status lifecycle and thresholds of the scene catalogue."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

import spectra
from spectra.types import Platform, SceneRecord

CATALOG_VERSION = "1.0"

Status = Literal[
    "discovered",
    "qa_pending",
    "qa_done",
    "eligible",
    "rejected",
    "quarantined",
    "manual_review",
]
STATUSES: tuple[Status, ...] = (
    "discovered",
    "qa_pending",
    "qa_done",
    "eligible",
    "rejected",
    "quarantined",
    "manual_review",
)
OUTCOME_STATUSES: tuple[Status, ...] = (
    "eligible",
    "rejected",
    "quarantined",
    "manual_review",
)
# Lifecycle: discovered -> qa_pending -> qa_done -> outcome. QA may be re-run
# from any QA-done state (back to ``qa_done``) and outcomes may be overridden
# manually among each other; ``discovered`` scenes cannot skip QA.
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "discovered": frozenset({"qa_pending", "qa_done"}),
    "qa_pending": frozenset({"qa_done"}),
    "qa_done": frozenset({"qa_done", "qa_pending", *OUTCOME_STATUSES}),
    **{
        status: frozenset({"qa_done", "qa_pending", *OUTCOME_STATUSES})
        for status in OUTCOME_STATUSES
    },
}
AUTOMATIC_ACTOR = "spectra.catalog"

STRING = "string"
FLOAT = "float64"
INT = "Int64"
BOOL = "bool"
TIMESTAMP = "datetime64[ns, UTC]"

# Column name to pandas dtype, in table order. ``scene_id`` is the primary key.
COLUMNS: dict[str, str] = {
    "sensor": STRING,
    "scene_id": STRING,
    "platform": STRING,
    "background_group": STRING,
    "acquisition_time": TIMESTAMP,
    "center_lat": FLOAT,
    "center_lon": FLOAT,
    "bbox_w": FLOAT,
    "bbox_s": FLOAT,
    "bbox_e": FLOAT,
    "bbox_n": FLOAT,
    "n_across": INT,
    "n_along": INT,
    "n_band": INT,
    "pixel_size_m_median": FLOAT,
    "uris": STRING,
    "has_radiance": BOOL,
    "has_product": BOOL,
    "status": STRING,
    "status_reason": STRING,
    "status_set_by": STRING,
    "status_set_utc": TIMESTAMP,
    "qa_version": STRING,
    "qa_utc": TIMESTAMP,
    "valid_fraction": FLOAT,
    "cloud_fraction": FLOAT,
    "shadow_fraction": FLOAT,
    "water_fraction": FLOAT,
    "saturation_fraction": FLOAT,
    "bg_median_kgm2": FLOAT,
    "bg_std_kgm2": FLOAT,
    "bg_mad_kgm2": FLOAT,
    "uncertainty_median_kgm2": FLOAT,
    "n_detections": INT,
    "max_detection_snr": FLOAT,
    "detection_uri": STRING,
    "annotation_uri": STRING,
    "annotation_status": STRING,
    "stats_uri": STRING,
    "catalog_version": STRING,
    "spectra_version": STRING,
    # Sensor-specific record metadata (JSON) so rows round-trip to records.
    "extra": STRING,
}
PRIMARY_KEY = "scene_id"
QA_METRIC_COLUMNS: tuple[str, ...] = (
    "valid_fraction",
    "cloud_fraction",
    "shadow_fraction",
    "water_fraction",
    "saturation_fraction",
    "bg_median_kgm2",
    "bg_std_kgm2",
    "bg_mad_kgm2",
    "uncertainty_median_kgm2",
    "n_detections",
    "max_detection_snr",
)

NAN = float("nan")


class CatalogError(ValueError):
    """The catalogue was used inconsistently (unknown scene, illegal transition)."""


@dataclass(frozen=True)
class SceneRow:
    """One row of the catalogue table; fields match :data:`COLUMNS` in order.

    Missing floats are NaN; missing integers, strings and timestamps are
    ``None`` (``pd.NA``/``NaT`` in the table).
    """

    sensor: str
    scene_id: str
    platform: str
    background_group: str
    acquisition_time: datetime
    center_lat: float
    center_lon: float
    bbox_w: float = NAN
    bbox_s: float = NAN
    bbox_e: float = NAN
    bbox_n: float = NAN
    n_across: int | None = None
    n_along: int | None = None
    n_band: int | None = None
    pixel_size_m_median: float = NAN
    uris: str = "{}"
    has_radiance: bool = True
    has_product: bool = False
    status: str = "discovered"
    status_reason: str | None = None
    status_set_by: str | None = None
    status_set_utc: datetime | None = None
    qa_version: str | None = None
    qa_utc: datetime | None = None
    valid_fraction: float = NAN
    cloud_fraction: float = NAN
    shadow_fraction: float = NAN
    water_fraction: float = NAN
    saturation_fraction: float = NAN
    bg_median_kgm2: float = NAN
    bg_std_kgm2: float = NAN
    bg_mad_kgm2: float = NAN
    uncertainty_median_kgm2: float = NAN
    n_detections: int | None = None
    max_detection_snr: float = NAN
    detection_uri: str | None = None
    annotation_uri: str | None = None
    annotation_status: str | None = None
    stats_uri: str | None = None
    catalog_version: str = CATALOG_VERSION
    spectra_version: str = spectra.__version__
    extra: str = "{}"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> SceneRow:
        """Build a row from a table record (for example a ``pandas.Series``).

        Pandas missing values become ``None`` and NumPy scalars become
        Python scalars; columns absent from ``mapping`` take their defaults.
        """
        values: dict[str, Any] = {}
        for item in fields(cls):
            if item.name not in mapping:
                continue
            values[item.name] = _to_python(mapping[item.name])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        """Return the row as a column-name mapping."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True)
class QaThresholds:
    """Automatic eligibility thresholds applied to QA metrics.

    Attributes:
        min_valid_fraction: Minimum fraction of valid pixels.
        max_cloud_fraction: Maximum fraction of cloudy pixels.
        max_detections: Maximum number of plume-like detections; scenes above
            it need manual review.
        max_bg_std_kgm2: Maximum background column standard deviation, or
            ``None`` to skip the check.
    """

    min_valid_fraction: float = 0.9
    max_cloud_fraction: float = 0.05
    max_detections: int = 0
    max_bg_std_kgm2: float | None = None


def _to_python(value: object) -> object:
    """Convert pandas/NumPy scalars to plain Python values (missing -> None)."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.datetime64):
        return pd.Timestamp(value).to_pydatetime()
    if isinstance(value, float | np.floating):
        return float(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.str_):
        return str(value)
    return value


def as_utc(value: datetime | np.datetime64 | pd.Timestamp | str) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime (naive input is UTC)."""
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp):
        raise CatalogError(f"{value!r} is not a valid timestamp.")
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    return stamp.to_pydatetime()


def record_to_row(record: SceneRecord, *, status: Status = "discovered") -> SceneRow:
    """Convert a scene record to a catalogue row without QA metrics."""
    bbox = record.bbox_lonlat
    west, south, east, north = (
        (NAN, NAN, NAN, NAN) if bbox is None else tuple(float(v) for v in bbox)
    )
    return SceneRow(
        sensor=record.sensor,
        scene_id=record.scene_id,
        platform=record.platform,
        background_group=record.background_group,
        acquisition_time=as_utc(record.acquisition_time),
        center_lat=float(record.center_lonlat[1]),
        center_lon=float(record.center_lonlat[0]),
        bbox_w=west,
        bbox_s=south,
        bbox_e=east,
        bbox_n=north,
        uris=json.dumps(dict(record.uris), sort_keys=True),
        has_radiance=bool(record.has_radiance),
        has_product=bool(record.has_product),
        status=status,
        extra=json.dumps(dict(record.extra), sort_keys=True, default=str),
    )


def row_to_record(row: SceneRow | Mapping[str, Any]) -> SceneRecord:
    """Rebuild the scene record stored in a catalogue row."""
    if not isinstance(row, SceneRow):
        row = SceneRow.from_mapping(row)
    bbox_values = (row.bbox_w, row.bbox_s, row.bbox_e, row.bbox_n)
    bbox = (
        None
        if any(math.isnan(v) for v in bbox_values)
        else cast(tuple[float, float, float, float], bbox_values)
    )
    return SceneRecord(
        sensor=row.sensor,
        scene_id=row.scene_id,
        platform=cast(Platform, row.platform),
        acquisition_time=as_utc(row.acquisition_time).astimezone(UTC),
        background_group=row.background_group,
        uris=json.loads(row.uris or "{}"),
        center_lonlat=(row.center_lon, row.center_lat),
        bbox_lonlat=bbox,
        has_radiance=bool(row.has_radiance),
        has_product=bool(row.has_product),
        extra=json.loads(row.extra or "{}"),
    )


__all__ = [
    "ALLOWED_TRANSITIONS",
    "AUTOMATIC_ACTOR",
    "CATALOG_VERSION",
    "COLUMNS",
    "OUTCOME_STATUSES",
    "PRIMARY_KEY",
    "QA_METRIC_COLUMNS",
    "STATUSES",
    "CatalogError",
    "QaThresholds",
    "SceneRow",
    "Status",
    "as_utc",
    "record_to_row",
    "row_to_record",
]
