"""Scene catalogue persisted as Parquet with a status lifecycle."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import spectra
from spectra.catalog.qa import QaResult
from spectra.catalog.schema import (
    ALLOWED_TRANSITIONS,
    AUTOMATIC_ACTOR,
    BOOL,
    CATALOG_VERSION,
    COLUMNS,
    PRIMARY_KEY,
    STATUSES,
    TIMESTAMP,
    CatalogError,
    QaThresholds,
    SceneRow,
    Status,
    record_to_row,
    row_to_record,
)
from spectra.protocols import SensorAdapter
from spectra.types import SceneRecord

SCENES_FILE = "scenes.parquet"
META_FILE = "catalog.json"
DETECTIONS_DIR = "detections"


def _timestamp(value: datetime | np.datetime64 | str | pd.Timestamp) -> pd.Timestamp:
    """Return ``value`` as a UTC timestamp (naive input is UTC)."""
    stamp = pd.Timestamp(value)
    if not isinstance(stamp, pd.Timestamp):
        raise CatalogError(f"{value!r} is not a valid timestamp.")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _now() -> pd.Timestamp:
    return _timestamp(datetime.now(tz=UTC))


def _to_utc(value: object) -> pd.Timestamp:
    if not isinstance(value, datetime | np.datetime64 | str):
        raise CatalogError(f"Date bound {value!r} must be a datetime or ISO string.")
    return _timestamp(value)


def _number(key: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise CatalogError(f"Filter {key!r} needs a numeric bound; got {value!r}.")
    return float(value)


def _as_bool(series: pd.Series) -> pd.Series:
    """Return a boolean mask with missing comparisons counted as ``False``."""
    return series.fillna(False).astype(bool)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Return one column as a Series (column names are unique)."""
    column = frame[name]
    if not isinstance(column, pd.Series):
        raise CatalogError(f"Column {name!r} is not unique.")
    return column


def _row_mapping(series: pd.Series) -> dict[str, Any]:
    """Return one table row as a column-name mapping."""
    return {str(name): value for name, value in series.items()}


def _version_key(version: str) -> tuple[tuple[int, str], ...]:
    """Sort key comparing dotted numeric components, then text."""
    parts = re.split(r"[.\-_]", version.strip())
    return tuple((int(p), "") if p.isdigit() else (-1, p) for p in parts)


def coerce_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` with exactly the catalogue columns and dtypes."""
    output = pd.DataFrame(index=pd.RangeIndex(len(frame)))
    for name, dtype in COLUMNS.items():
        if name in frame.columns:
            series = pd.Series(frame[name].to_numpy(dtype=object), index=output.index)
        else:
            series = pd.Series([None] * len(frame), index=output.index, dtype=object)
        if dtype == TIMESTAMP:
            column = pd.to_datetime(series, utc=True).astype(dtype)
        elif dtype == BOOL:
            column = series.fillna(False).astype(bool)
        else:
            column = series.astype(dtype)
        output[name] = column
    return output


def rows_to_frame(rows: Iterable[SceneRow]) -> pd.DataFrame:
    """Build a typed catalogue table from rows."""
    records = [row.to_dict() for row in rows]
    frame = pd.DataFrame.from_records(records, columns=list(COLUMNS))
    return coerce_frame(frame)


class SceneCatalog:
    """Table of scenes with QA metrics and a status per scene.

    The table is a :class:`pandas.DataFrame` with the columns of
    :data:`spectra.catalog.schema.COLUMNS`; ``scene_id`` is unique. Status is
    the only field changed in place; QA metrics are append-only by
    ``qa_version``.

    Args:
        frame: Table with at least the record columns; missing columns are
            added, dtypes are coerced.
        directory: Catalogue directory used for detection masks and
            ``write`` without a path.
    """

    def __init__(
        self, frame: pd.DataFrame | None = None, *, directory: str | Path | None = None
    ) -> None:
        """Wrap and validate a catalogue table."""
        self._frame = coerce_frame(
            frame if frame is not None else pd.DataFrame(columns=list(COLUMNS))
        )
        duplicates = self._frame[PRIMARY_KEY][self._frame[PRIMARY_KEY].duplicated()]
        if len(duplicates):
            raise CatalogError(
                f"scene_id must be unique; duplicated: {sorted(set(duplicates))}."
            )
        unknown = set(self._frame["status"].dropna()) - set(STATUSES)
        if unknown:
            raise CatalogError(f"Unknown statuses {sorted(unknown)}.")
        self.directory = None if directory is None else Path(directory)

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        adapter: SensorAdapter,
        scenes: str | Path | Sequence[Mapping[str, object]] | None,
        *,
        records: Sequence[SceneRecord] | None = None,
        inspect: bool = True,
        subsample: int = 16,
        directory: str | Path | None = None,
        **filters: object,
    ) -> SceneCatalog:
        """Catalogue the scenes of an explicit scene table as ``discovered``.

        Args:
            adapter: Sensor adapter.
            scenes: Scene table (CSV/YAML path or rows) passed to
                ``adapter.list_scenes``; ``None`` only with ``records``.
            records: Records to catalogue instead of calling ``list_scenes``.
            inspect: Open every scene to fill ``n_across``/``n_along``/
                ``n_band`` and ``pixel_size_m_median``; disable for quick
                cataloguing of large campaigns.
            subsample: Stride of the pixel-area sample used for the pixel size.
            directory: Catalogue directory (see :class:`SceneCatalog`).
            **filters: Filters forwarded to ``adapter.list_scenes``.

        Returns:
            The catalogue.
        """
        if records is None:
            if scenes is None:
                raise CatalogError("build() needs a scene table or records.")
            records = adapter.list_scenes(scenes, **filters)
        rows: list[SceneRow] = []
        for record in records:
            row = record_to_row(record)
            if inspect:
                row = SceneRow(
                    **{**row.to_dict(), **_inspect(adapter, record, subsample)}
                )
            rows.append(row)
        return cls(rows_to_frame(rows), directory=directory)

    @classmethod
    def read(cls, path: str | Path) -> SceneCatalog:
        """Read a catalogue directory written by :meth:`write`."""
        directory = Path(path)
        table = directory / SCENES_FILE if directory.is_dir() else directory
        if not table.exists():
            raise FileNotFoundError(f"No catalogue table at {table}.")
        meta_path = table.parent / META_FILE
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            major = str(meta.get("catalog_version", CATALOG_VERSION)).split(".")[0]
            if major != CATALOG_VERSION.split(".")[0]:
                raise CatalogError(
                    f"Catalogue version {meta.get('catalog_version')} is "
                    f"incompatible with {CATALOG_VERSION}."
                )
        frame = pd.read_parquet(table)
        return cls(frame, directory=table.parent)

    def write(self, path: str | Path | None = None) -> Path:
        """Write ``scenes.parquet`` and ``catalog.json`` into a directory.

        Args:
            path: Catalogue directory; defaults to ``self.directory``.

        Returns:
            The catalogue directory.
        """
        directory = Path(path) if path is not None else self.directory
        if directory is None:
            raise CatalogError("No catalogue directory given.")
        directory.mkdir(parents=True, exist_ok=True)
        self._frame.to_parquet(directory / SCENES_FILE, index=False)
        meta = {
            "catalog_version": CATALOG_VERSION,
            "spectra_version": spectra.__version__,
            "written_utc": _now().isoformat(),
            "n_scenes": len(self._frame),
            "scenes_file": SCENES_FILE,
            "statuses": {
                k: int(v) for k, v in self._frame["status"].value_counts().items()
            },
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True))
        self.directory = directory
        return directory

    # -- access ------------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        """Return the underlying table (treat as read-only)."""
        return self._frame

    def __len__(self) -> int:
        """Return the number of scenes."""
        return len(self._frame)

    def __contains__(self, scene_id: object) -> bool:
        """Return whether a scene id is catalogued."""
        return bool((self._frame[PRIMARY_KEY] == scene_id).any())

    def __repr__(self) -> str:
        """Summarise the catalogue."""
        counts = ", ".join(
            f"{status}={count}"
            for status, count in self._frame["status"].value_counts().items()
        )
        return f"SceneCatalog({len(self)} scenes; {counts})"

    def _index_of(self, scene_id: str) -> int:
        matches = np.flatnonzero((self._frame[PRIMARY_KEY] == scene_id).to_numpy())
        if matches.size == 0:
            raise CatalogError(f"Unknown scene_id {scene_id!r}.")
        return int(matches[0])

    def row(self, scene_id: str) -> SceneRow:
        """Return the row of one scene."""
        return SceneRow.from_mapping(
            _row_mapping(self._frame.iloc[self._index_of(scene_id)])
        )

    def records(
        self, status: Status | str | Sequence[str] | None = None
    ) -> list[SceneRecord]:
        """Return scene records, optionally restricted to one or more statuses."""
        frame = self._frame if status is None else self.filter(status=status)
        return [
            row_to_record(SceneRow.from_mapping(_row_mapping(row)))
            for _, row in frame.iterrows()
        ]

    def filter(self, **criteria: object) -> pd.DataFrame:
        """Return the rows matching all criteria.

        Column names select rows by equality (scalars) or membership
        (sequences); ``date_min``/``date_max`` bound ``acquisition_time``;
        ``max_<column>``/``min_<column>`` are inclusive bounds on numeric
        columns (missing values never match).
        """
        frame = self._frame
        mask = pd.Series(True, index=frame.index, dtype=bool)
        for key, value in criteria.items():
            if key == "date_min":
                mask &= _as_bool(_column(frame, "acquisition_time") >= _to_utc(value))
            elif key == "date_max":
                mask &= _as_bool(_column(frame, "acquisition_time") <= _to_utc(value))
            elif key in COLUMNS:
                if isinstance(value, Sequence | set | frozenset) and not isinstance(
                    value, str | bytes
                ):
                    mask &= _as_bool(_column(frame, key).isin(list(value)))
                else:
                    mask &= _as_bool(_column(frame, key) == value)
            elif key.startswith("max_") and key[4:] in COLUMNS:
                mask &= _as_bool(_column(frame, key[4:]) <= _number(key, value))
            elif key.startswith("min_") and key[4:] in COLUMNS:
                mask &= _as_bool(_column(frame, key[4:]) >= _number(key, value))
            else:
                raise CatalogError(f"Unknown filter criterion {key!r}.")
        return frame.loc[mask].copy()

    def summary(self) -> pd.DataFrame:
        """Return scene counts per sensor (rows) and status (columns)."""
        frame = self._frame
        counts = pd.crosstab(frame["sensor"], frame["status"])
        counts = counts.reindex(columns=list(STATUSES), fill_value=0)
        counts.columns = pd.Index([str(c) for c in counts.columns], name="status")
        counts.index = pd.Index([str(i) for i in counts.index], name="sensor")
        counts["total"] = counts.sum(axis=1)
        counts.loc["total"] = counts.sum(axis=0)
        return counts.astype("int64")

    # -- mutation ----------------------------------------------------------
    def set_status(
        self, scene_id: str, status: Status | str, *, reason: str, by: str
    ) -> None:
        """Move a scene along the lifecycle, recording who did it and why.

        Raises:
            CatalogError: If the transition is not in ``ALLOWED_TRANSITIONS``.
        """
        if status not in STATUSES:
            raise CatalogError(f"Unknown status {status!r}; choose from {STATUSES}.")
        index = self._index_of(scene_id)
        current = str(self._frame.at[index, "status"])
        if status not in ALLOWED_TRANSITIONS[current]:
            raise CatalogError(
                f"Transition {current!r} -> {status!r} is not allowed for "
                f"{scene_id!r}; allowed: {sorted(ALLOWED_TRANSITIONS[current])}."
            )
        self._frame.at[index, "status"] = status
        self._frame.at[index, "status_reason"] = reason
        self._frame.at[index, "status_set_by"] = by
        self._frame.at[index, "status_set_utc"] = _now()

    def update_qa(
        self,
        scene_id: str,
        result: QaResult,
        *,
        qa_version: str,
        force: bool = False,
    ) -> None:
        """Store QA metrics of a newer ``qa_version`` and mark the scene ``qa_done``.

        The detection mask, when present and a catalogue directory is known,
        is written to ``<directory>/detections/<scene_id>.nc`` and referenced
        by ``detection_uri``.

        Args:
            scene_id: Scene to update.
            result: Metrics from :func:`spectra.catalog.qa.compute_scene_qa`.
            qa_version: Version label of the QA run; must be newer than the
                stored one.
            force: Overwrite metrics of the same or an older version.

        Raises:
            CatalogError: If ``qa_version`` is not newer and ``force`` is off.
        """
        index = self._index_of(scene_id)
        existing = self._frame.at[index, "qa_version"]
        if (
            not force
            and existing is not pd.NA
            and _version_key(str(qa_version)) <= _version_key(str(existing))
        ):
            raise CatalogError(
                f"QA metrics of {scene_id!r} are append-only: version "
                f"{qa_version!r} is not newer than {existing!r}."
            )
        for name, value in result.metrics().items():
            self._frame.at[index, name] = pd.NA if value is None else value
        detection_uri: str | None = None
        if result.detections is not None and self.directory is not None:
            target = self.directory / DETECTIONS_DIR / f"{scene_id}.nc"
            target.parent.mkdir(parents=True, exist_ok=True)
            result.detections.to_netcdf(target)
            detection_uri = str(target)
        self._frame.at[index, "detection_uri"] = (
            pd.NA if detection_uri is None else detection_uri
        )
        self._frame.at[index, "qa_version"] = str(qa_version)
        self._frame.at[index, "qa_utc"] = _now()
        self.set_status(
            scene_id,
            "qa_done",
            reason=f"QA {qa_version} ({result.detection_method})",
            by=AUTOMATIC_ACTOR,
        )

    def apply_thresholds(
        self, thresholds: QaThresholds | None = None, *, by: str = AUTOMATIC_ACTOR
    ) -> dict[str, str]:
        """Classify every ``qa_done`` scene and return the new status per scene.

        Scenes with an annotation are ``quarantined``; scenes failing the
        valid/cloud/background-std thresholds are ``rejected``; scenes with
        more detections than allowed (or without a valid fraction) need
        ``manual_review``; the rest are ``eligible``. Detection counts of
        ``None`` (detection disabled) count as zero.
        """
        thresholds = thresholds or QaThresholds()
        outcomes: dict[str, str] = {}
        for scene_id in self.filter(status="qa_done")[PRIMARY_KEY]:
            row = self.row(str(scene_id))
            status, reason = _classify(row, thresholds)
            self.set_status(str(scene_id), status, reason=reason, by=by)
            outcomes[str(scene_id)] = status
        return outcomes


def _classify(row: SceneRow, thresholds: QaThresholds) -> tuple[Status, str]:
    if row.annotation_uri is not None:
        return "quarantined", "annotation present"
    if np.isnan(row.valid_fraction):
        return "manual_review", "valid_fraction unknown"
    if row.valid_fraction < thresholds.min_valid_fraction:
        return (
            "rejected",
            f"valid_fraction {row.valid_fraction:.3f} < "
            f"{thresholds.min_valid_fraction}",
        )
    if row.cloud_fraction > thresholds.max_cloud_fraction:
        return (
            "rejected",
            f"cloud_fraction {row.cloud_fraction:.3f} > "
            f"{thresholds.max_cloud_fraction}",
        )
    if (
        thresholds.max_bg_std_kgm2 is not None
        and row.bg_std_kgm2 > thresholds.max_bg_std_kgm2
    ):
        return (
            "rejected",
            f"bg_std_kgm2 {row.bg_std_kgm2:.3e} > {thresholds.max_bg_std_kgm2:.3e}",
        )
    n_detections = row.n_detections or 0
    if n_detections > thresholds.max_detections:
        return (
            "manual_review",
            f"n_detections {n_detections} > {thresholds.max_detections}",
        )
    return "eligible", "passed thresholds"


def _inspect(
    adapter: SensorAdapter, record: SceneRecord, subsample: int
) -> Mapping[str, Any]:
    """Open a scene and read its sizes and median pixel size."""
    scene = adapter.open_scene(record)
    try:
        values: dict[str, Any] = {
            "n_across": int(scene.sizes["across"]),
            "n_along": int(scene.sizes["along"]),
            "n_band": int(scene.sizes["band"]),
        }
        if "pixel_area" in scene:
            area = scene["pixel_area"]
            step = max(1, int(subsample))
            sample = area.isel({str(dim): slice(None, None, step) for dim in area.dims})
            sample_values = np.asarray(sample.values, dtype=float)
            if np.isfinite(sample_values).any():
                values["pixel_size_m_median"] = float(
                    np.sqrt(np.nanmedian(sample_values))
                )
    finally:
        _close(scene)
    return values


def _close(scene: xr.Dataset) -> None:
    with contextlib.suppress(Exception):  # in-memory datasets have nothing to close
        scene.close()


__all__ = [
    "DETECTIONS_DIR",
    "META_FILE",
    "SCENES_FILE",
    "SceneCatalog",
    "coerce_frame",
    "rows_to_frame",
]
