"""Pairings table of a generated dataset (prism Contract C, SPECTRA side).

prism's ``prism-compose-dataset`` reads a CSV with the required columns
``sample_id, product_uri, sensor, background_group, les_case, source_id,
split``; SPECTRA adds the injected source pixel and the provenance columns
that become part of the dataset identity (prism's manifest fingerprint
excludes only ``*_uri`` columns).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr

import spectra
from spectra.contracts.errors import ContractError

if TYPE_CHECKING:
    from spectra.recipe.params import SampleParams
    from spectra.recipe.schema import Recipe

PRISM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "product_uri",
    "background_group",
    "les_case",
    "source_id",
    "split",
    "sensor",
)
PAIRINGS_COLUMNS: tuple[str, ...] = (
    *PRISM_REQUIRED_COLUMNS,
    "source_y",
    "source_x",
    "scene_id",
    "tracer",
    "time_index",
    "simulation_time_s",
    "emission_rate_kg_h",
    "emission_scale",
    "rotation_deg",
    "flip",
    "anchor_corner",
    "anchor_along",
    "anchor_across",
    "window_across_start",
    "window_along_start",
    "embedding_method",
    "retriever",
    "unit_absorption_provider",
    "regridder",
    "mf_statistics_scope",
    "uncertainty_source",
    "column_convention",
    "pixel_size_m",
    "pair_id",
    "plume_uri",
    "scene_uri",
    "recipe_hash",
    "params_hash",
    "spectra_version",
)
GROUP_COLUMNS: tuple[str, ...] = ("background_group", "les_case")
_STRING_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "product_uri",
    "background_group",
    "les_case",
    "source_id",
    "split",
    "sensor",
    "scene_id",
    "tracer",
    "flip",
    "anchor_corner",
    "embedding_method",
    "retriever",
    "unit_absorption_provider",
    "regridder",
    "mf_statistics_scope",
    "uncertainty_source",
    "column_convention",
    "pair_id",
    "plume_uri",
    "scene_uri",
    "recipe_hash",
    "params_hash",
    "spectra_version",
)


def _optional_pixel(attrs: Mapping[str, Any], key: str) -> float:
    """Return an integer attribute as float, NaN when absent."""
    value = attrs.get(key)
    if value is None:
        return math.nan
    return float(int(value))


def pairings_row(
    product: xr.Dataset, params: SampleParams, recipe: Recipe, uri: str | Path
) -> dict[str, Any]:
    """Return the pairings row of one written product.

    Args:
        product: The prism product as written (its attributes supply the
            retriever, uncertainty source, pixel size, source pixel, scene
            URI and recipe hash).
        params: Sample parameters.
        recipe: Recipe the sample belongs to.
        uri: Path of the written product.

    Returns:
        Mapping with exactly :data:`PAIRINGS_COLUMNS`.
    """
    attrs = product.attrs
    retrieval = recipe.retrieval
    provider = recipe.embedding.unit_absorption
    retriever = attrs.get("retriever")
    if retriever is None:
        retriever = retrieval.retriever if retrieval is not None else ""
    row: dict[str, Any] = {
        "sample_id": params.sample_id,
        "product_uri": str(uri),
        "background_group": params.background_group,
        "les_case": params.les_case,
        "source_id": params.source_id,
        "split": params.split,
        "sensor": params.sensor,
        "source_y": _optional_pixel(attrs, "injected_source_y"),
        "source_x": _optional_pixel(attrs, "injected_source_x"),
        "scene_id": params.scene_id,
        "tracer": params.tracer,
        "time_index": int(params.time_index),
        "simulation_time_s": float(params.simulation_time_s),
        "emission_rate_kg_h": float(params.emission_rate_kg_s) * 3600.0,
        "emission_scale": float(params.emission_scale),
        "rotation_deg": int(params.placement.rotation_deg),
        "flip": params.placement.flip,
        "anchor_corner": params.placement.anchor_corner,
        "anchor_along": int(params.placement.anchor_along),
        "anchor_across": int(params.placement.anchor_across),
        "window_across_start": int(params.window.across.start),
        "window_along_start": int(params.window.along.start),
        "embedding_method": str(attrs.get("embedding_method", recipe.method)),
        "retriever": str(retriever),
        "unit_absorption_provider": provider.provider
        if provider is not None and recipe.method == "radiance"
        else "",
        "regridder": recipe.embedding.regridder,
        "mf_statistics_scope": retrieval.statistics_scope
        if retrieval is not None
        else "",
        "uncertainty_source": str(attrs.get("uncertainty_source", "")),
        "column_convention": recipe.plumes.column_convention,
        "pixel_size_m": float(attrs.get("pixel_size_m", math.nan)),
        "pair_id": params.pair_id,
        "plume_uri": params.plume_uri,
        "scene_uri": str(attrs.get("scene_uri", "")),
        "recipe_hash": str(attrs.get("recipe_hash", "")),
        "params_hash": params.params_hash,
        "spectra_version": spectra.__version__,
    }
    return {name: row[name] for name in PAIRINGS_COLUMNS}


def pairings_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Return rows as a table in :data:`PAIRINGS_COLUMNS` order."""
    records = [dict(row) for row in rows]
    extra = sorted({key for row in records for key in row} - set(PAIRINGS_COLUMNS))
    frame = pd.DataFrame.from_records(records, columns=[*PAIRINGS_COLUMNS, *extra])
    for name in _STRING_COLUMNS:
        frame[name] = frame[name].astype("string")
    return frame


def write_pairings(rows: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Write pairings rows as CSV."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pairings_frame(rows).to_csv(target, index=False)
    return target


def read_pairings(path: str | Path) -> pd.DataFrame:
    """Read a pairings CSV, keeping identifiers as strings.

    Raises:
        ContractError: If a prism-required column is missing.
    """
    frame = pd.read_csv(
        path, dtype=dict.fromkeys(_STRING_COLUMNS, "string"), keep_default_na=True
    )
    missing = [name for name in PRISM_REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ContractError(f"{path} lacks pairings columns {missing}.")
    for name in _STRING_COLUMNS:
        if name in frame.columns:
            frame[name] = frame[name].fillna("").astype("string")
    return frame


def assert_group_independence(frame: pd.DataFrame) -> None:
    """Apply prism's leakage rule: groups never cross splits or sensors.

    Raises:
        ContractError: If a ``background_group`` or ``les_case`` occurs in
            more than one split, or a ``background_group`` spans sensors.
    """
    for column in GROUP_COLUMNS:
        counts = frame.groupby(column, dropna=False)["split"].nunique()
        crossing = [str(name) for name, count in counts.items() if count > 1]
        if crossing:
            raise ContractError(
                f"{column} values cross train/val/test boundaries: {crossing}"
            )
    sensors = frame.groupby("background_group", dropna=False)["sensor"].nunique()
    mixed = [str(name) for name, count in sensors.items() if count > 1]
    if mixed:
        raise ContractError(
            f"background_group values span more than one sensor: {mixed}"
        )


def scene_uri(uris: Mapping[str, str]) -> str:
    """Return the scene URIs of a record as one JSON string for the table."""
    return json.dumps(dict(uris), sort_keys=True)


__all__ = [
    "GROUP_COLUMNS",
    "PAIRINGS_COLUMNS",
    "PRISM_REQUIRED_COLUMNS",
    "assert_group_independence",
    "pairings_frame",
    "pairings_row",
    "read_pairings",
    "scene_uri",
    "write_pairings",
]
