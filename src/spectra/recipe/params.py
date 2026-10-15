"""Deterministic per-sample parameters of a recipe.

``build_context`` resolves what a recipe refers to (eligible scenes, plume
files, group-to-split assignment) and ``derive_sample_params`` draws the
parameters of sample ``i`` from ``SeedSequence(seed, spawn_key=(i,))`` in a
fixed order, so generating sample ``i`` needs only its own row of the
parameter table and never depends on the other samples or the process.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr

from spectra.contracts.errors import ContractError
from spectra.protocols import SensorAdapter
from spectra.recipe.schema import (
    SPLITS,
    EmissionSpec,
    Recipe,
    RotationSpec,
    TimePolicy,
    recipe_hash,
)
from spectra.types import (
    AnchorCorner,
    Flip,
    Placement,
    Platform,
    SceneGeometry,
    SceneRecord,
    Window,
)

CONTEXT_FILE = "context.json"
PARAMS_FILE = "params.parquet"
NEGATIVE_SOURCE_ID = "none"
NEGATIVE_CASE_PREFIX = "none:"
PARAMS_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "index",
    "split",
    "sensor",
    "scene_id",
    "background_group",
    "plume_uri",
    "les_case",
    "tracer",
    "source_id",
    "time_index",
    "simulation_time_s",
    "emission_rate_kg_s",
    "emission_scale",
    "anchor_along",
    "anchor_across",
    "anchor_corner",
    "rotation_deg",
    "flip",
    "window_across_start",
    "window_across_stop",
    "window_along_start",
    "window_along_stop",
    "is_negative",
    "pair_id",
    "params_hash",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- scene records as JSON -------------------------------------------------
def record_to_dict(record: SceneRecord) -> dict[str, Any]:
    """Return a scene record as a JSON-serialisable mapping."""
    return {
        "sensor": record.sensor,
        "scene_id": record.scene_id,
        "platform": record.platform,
        "acquisition_time": record.acquisition_time.isoformat(),
        "background_group": record.background_group,
        "uris": dict(record.uris),
        "center_lonlat": list(record.center_lonlat),
        "bbox_lonlat": None if record.bbox_lonlat is None else list(record.bbox_lonlat),
        "has_radiance": bool(record.has_radiance),
        "has_product": bool(record.has_product),
        "extra": json.loads(json.dumps(dict(record.extra), default=str)),
    }


def record_from_dict(mapping: Mapping[str, Any]) -> SceneRecord:
    """Rebuild a scene record written by :func:`record_to_dict`."""
    bbox = mapping.get("bbox_lonlat")
    return SceneRecord(
        sensor=str(mapping["sensor"]),
        scene_id=str(mapping["scene_id"]),
        platform=cast(Platform, str(mapping["platform"])),
        acquisition_time=datetime.fromisoformat(str(mapping["acquisition_time"])),
        background_group=str(mapping["background_group"]),
        uris=dict(mapping.get("uris", {})),
        center_lonlat=(
            float(mapping["center_lonlat"][0]),
            float(mapping["center_lonlat"][1]),
        ),
        bbox_lonlat=None
        if bbox is None
        else (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
        has_radiance=bool(mapping.get("has_radiance", True)),
        has_product=bool(mapping.get("has_product", False)),
        extra=dict(mapping.get("extra", {})),
    )


# -- plume files -------------------------------------------------------------
@dataclass(frozen=True)
class PlumeEntry:
    """Summary of one plume-column file used by the draws.

    Attributes:
        uri: File path.
        case_id: LES case identifier (the ``les_case`` group).
        tracer: Tracer name.
        source_id: Source identifier.
        gas: Gas name.
        les_model: LES model that produced the file.
        n_time: Number of snapshots.
        simulation_time_s: Seconds since the LES start per snapshot.
        emission_rate_kg_s: Emission rate per snapshot.
        column_max_kg_m2: Largest column value per snapshot.
        source_xy: Source position in the model frame.
        has_wind: Whether ``wind_u``/``wind_v`` are present.
    """

    uri: str
    case_id: str
    tracer: str
    source_id: str
    gas: str
    les_model: str
    n_time: int
    simulation_time_s: tuple[float, ...]
    emission_rate_kg_s: tuple[float, ...]
    column_max_kg_m2: tuple[float, ...]
    source_xy: tuple[float, float]
    has_wind: bool

    @classmethod
    def from_dataset(cls, uri: str, dataset: xr.Dataset) -> PlumeEntry:
        """Summarise a validated ``PlumeColumn`` dataset."""
        attrs = dataset.attrs
        column = dataset["column"]
        return cls(
            uri=str(uri),
            case_id=str(attrs["case_id"]),
            tracer=str(attrs["tracer"]),
            source_id=str(attrs["source_id"]),
            gas=str(attrs["gas"]),
            les_model=str(attrs.get("les_model", "")),
            n_time=int(dataset.sizes["time"]),
            simulation_time_s=tuple(
                float(v) for v in np.asarray(dataset["simulation_time"].values)
            ),
            emission_rate_kg_s=tuple(
                float(v) for v in np.asarray(dataset["emission_rate"].values)
            ),
            column_max_kg_m2=tuple(
                float(v) for v in np.asarray(column.max(dim=("y", "x")).values)
            ),
            source_xy=(float(attrs["source_x"]), float(attrs["source_y"])),
            has_wind="wind_u" in dataset and "wind_v" in dataset,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the entry as a JSON-serialisable mapping."""
        return {
            "uri": self.uri,
            "case_id": self.case_id,
            "tracer": self.tracer,
            "source_id": self.source_id,
            "gas": self.gas,
            "les_model": self.les_model,
            "n_time": self.n_time,
            "simulation_time_s": list(self.simulation_time_s),
            "emission_rate_kg_s": list(self.emission_rate_kg_s),
            "column_max_kg_m2": list(self.column_max_kg_m2),
            "source_xy": list(self.source_xy),
            "has_wind": self.has_wind,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> PlumeEntry:
        """Rebuild an entry written by :meth:`to_dict`."""
        return cls(
            uri=str(mapping["uri"]),
            case_id=str(mapping["case_id"]),
            tracer=str(mapping["tracer"]),
            source_id=str(mapping["source_id"]),
            gas=str(mapping["gas"]),
            les_model=str(mapping.get("les_model", "")),
            n_time=int(mapping["n_time"]),
            simulation_time_s=tuple(float(v) for v in mapping["simulation_time_s"]),
            emission_rate_kg_s=tuple(float(v) for v in mapping["emission_rate_kg_s"]),
            column_max_kg_m2=tuple(float(v) for v in mapping["column_max_kg_m2"]),
            source_xy=(float(mapping["source_xy"][0]), float(mapping["source_xy"][1])),
            has_wind=bool(mapping.get("has_wind", False)),
        )


def resolve_plume_files(patterns: Iterable[str]) -> list[str]:
    """Expand globs and return the sorted, de-duplicated plume file paths.

    Raises:
        ContractError: If a pattern matches nothing or a file is missing.
    """
    files: set[str] = set()
    for pattern in patterns:
        if glob.has_magic(pattern):
            # Absolute patterns with wildcards in directory parts need glob.glob.
            matches = sorted(glob.glob(pattern))  # noqa: PTH207
            if not matches:
                raise ContractError(
                    f"plumes.files pattern {pattern!r} matches nothing."
                )
            files.update(matches)
        else:
            if not Path(pattern).exists():
                raise ContractError(f"plumes.files entry {pattern!r} does not exist.")
            files.add(pattern)
    return sorted(files)


# -- split assignment --------------------------------------------------------
def assign_splits(
    groups: Iterable[str], ratios: Mapping[str, float], seed: int
) -> dict[str, str]:
    """Assign indivisible groups to splits deterministically.

    The groups are ordered by the SHA-256 of ``"<seed>:<group>"`` (a
    process-independent shuffle), the split sizes follow the cumulative
    ratios with largest-remainder rounding, and, like prism's
    ``assign_groups``, every split with a positive ratio receives at least
    one group when there are enough groups.

    Args:
        groups: Group identifiers (duplicates are ignored).
        ratios: Fraction per split name in :data:`SPLITS`.
        seed: Seed mixed into the ordering hash.

    Returns:
        Split name per group.
    """
    unique = sorted(set(groups))
    if not unique:
        raise ContractError("assign_splits needs at least one group.")
    order = sorted(unique, key=lambda group: (_sha256(f"{seed}:{group}"), group))
    raw = [float(ratios.get(name, 0.0)) * len(order) for name in SPLITS]
    counts = [math.floor(value) for value in raw]
    remainders = [value - count for value, count in zip(raw, counts, strict=True)]
    extra = len(order) - sum(counts)
    for index in sorted(range(len(SPLITS)), key=lambda i: (-remainders[i], i))[:extra]:
        counts[index] += 1
    positive = [i for i, name in enumerate(SPLITS) if ratios.get(name, 0.0) > 0]
    if len(order) >= len(positive):
        for empty in [i for i in positive if counts[i] == 0]:
            donors = [i for i in positive if counts[i] > 1]
            if not donors:
                break
            donor = max(donors, key=lambda i: (counts[i], -i))
            counts[donor] -= 1
            counts[empty] += 1
    assignment: dict[str, str] = {}
    offset = 0
    for name, count in zip(SPLITS, counts, strict=True):
        for group in order[offset : offset + count]:
            assignment[group] = name
        offset += count
    return assignment


# -- context -----------------------------------------------------------------
@dataclass(frozen=True)
class GenerationContext:
    """Everything a recipe refers to, resolved once and persisted.

    Attributes:
        recipe_hash: Hash of the recipe the context was built for.
        records: Eligible background scenes.
        plumes: Plume files.
        background_splits: Split per ``background_group``.
        case_splits: Split per ``les_case``.
        context_hash: Hash of the resolved content.
    """

    recipe_hash: str
    records: tuple[SceneRecord, ...]
    plumes: tuple[PlumeEntry, ...]
    background_splits: dict[str, str]
    case_splits: dict[str, str]
    context_hash: str

    def record(self, scene_id: str) -> SceneRecord:
        """Return the record of one scene."""
        for record in self.records:
            if record.scene_id == scene_id:
                return record
        raise ContractError(f"Scene {scene_id!r} is not in the generation context.")

    def scenes_in_split(self, split: str) -> list[SceneRecord]:
        """Return the eligible scenes whose background group is in ``split``."""
        return [
            record
            for record in self.records
            if self.background_splits.get(record.background_group) == split
        ]

    def plumes_in_split(self, split: str) -> list[PlumeEntry]:
        """Return the plume files whose LES case is in ``split``."""
        return [
            plume
            for plume in self.plumes
            if self.case_splits.get(plume.case_id) == split
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return the context as a JSON-serialisable mapping."""
        return {
            "recipe_hash": self.recipe_hash,
            "context_hash": self.context_hash,
            "records": [record_to_dict(record) for record in self.records],
            "plumes": [plume.to_dict() for plume in self.plumes],
            "background_splits": dict(self.background_splits),
            "case_splits": dict(self.case_splits),
        }

    def write(self, path: str | Path) -> Path:
        """Write the context as ``context.json``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return target

    @classmethod
    def read(cls, path: str | Path) -> GenerationContext:
        """Read a context written by :meth:`write`."""
        mapping = json.loads(Path(path).read_text())
        return cls(
            recipe_hash=str(mapping["recipe_hash"]),
            records=tuple(record_from_dict(item) for item in mapping["records"]),
            plumes=tuple(PlumeEntry.from_dict(item) for item in mapping["plumes"]),
            background_splits=dict(mapping["background_splits"]),
            case_splits=dict(mapping["case_splits"]),
            context_hash=str(mapping["context_hash"]),
        )


def _context_hash(
    records: Sequence[SceneRecord],
    plumes: Sequence[PlumeEntry],
    background_splits: Mapping[str, str],
    case_splits: Mapping[str, str],
) -> str:
    payload = {
        "records": [record_to_dict(record) for record in records],
        "plumes": [plume.to_dict() for plume in plumes],
        "background_splits": dict(background_splits),
        "case_splits": dict(case_splits),
    }
    return _sha256(json.dumps(payload, sort_keys=True, default=str))


def _eligible_records(
    recipe: Recipe, adapter: SensorAdapter | None
) -> list[SceneRecord]:
    """Return the background scenes the recipe selects."""
    if recipe.catalog.path is None:
        adapter = adapter or _adapter(recipe)
        assert recipe.sensor.scenes is not None
        records = list(
            adapter.list_scenes(recipe.sensor.scenes, **recipe.catalog.filters)
        )
    else:
        from spectra.catalog.schema import SceneRow, row_to_record
        from spectra.catalog.store import SceneCatalog

        catalog = SceneCatalog.read(recipe.catalog.path)
        frame = catalog.filter(
            status=list(recipe.catalog.status), **recipe.catalog.filters
        )
        records = [
            row_to_record(SceneRow.from_mapping({str(k): v for k, v in row.items()}))
            for _, row in frame.iterrows()
        ]
    records = [record for record in records if record.sensor == recipe.sensor.name]
    if recipe.method == "product":
        records = [record for record in records if record.has_product]
    if not records:
        raise ContractError(
            f"No eligible scene of sensor {recipe.sensor.name!r} matches "
            f"catalog.status={list(recipe.catalog.status)} and "
            f"catalog.filters={recipe.catalog.filters}."
        )
    return sorted(records, key=lambda record: record.scene_id)


def _adapter(recipe: Recipe) -> SensorAdapter:
    from spectra.registry import get_sensor

    return get_sensor(recipe.sensor.name, **recipe.sensor.options)


def build_context(
    recipe: Recipe, *, adapter: SensorAdapter | None = None
) -> GenerationContext:
    """Resolve the scenes, plume files and split assignment of a recipe.

    Args:
        recipe: Validated recipe.
        adapter: Sensor adapter; built from the registry when omitted.

    Returns:
        The context.

    Raises:
        ContractError: If nothing is eligible, a plume file is invalid, or a
            split with a positive ratio has no scene or no plume file.
    """
    from spectra.plume.io import open_plume_column

    records = _eligible_records(recipe, adapter)
    plumes: list[PlumeEntry] = []
    for uri in resolve_plume_files(recipe.plumes.files):
        dataset = open_plume_column(uri)
        try:
            plumes.append(PlumeEntry.from_dataset(uri, dataset))
        finally:
            dataset.close()
    ratios = recipe.split.ratios
    background_splits = assign_splits(
        (record.background_group for record in records), ratios, recipe.split_seed
    )
    case_splits = assign_splits(
        (plume.case_id for plume in plumes), ratios, recipe.split_seed + 1
    )
    context = GenerationContext(
        recipe_hash=recipe_hash(recipe),
        records=tuple(records),
        plumes=tuple(plumes),
        background_splits=background_splits,
        case_splits=case_splits,
        context_hash=_context_hash(records, plumes, background_splits, case_splits),
    )
    for split in SPLITS:
        if ratios.get(split, 0.0) <= 0:
            continue
        if not context.scenes_in_split(split):
            raise ContractError(
                f"Split {split!r} has no eligible scene: {len(records)} scene(s) in "
                f"{len(background_splits)} background group(s) cannot fill "
                f"split.ratios={ratios}. Add scenes or change the ratios."
            )
        if not context.plumes_in_split(split):
            raise ContractError(
                f"Split {split!r} has no plume file: {len(plumes)} file(s) in "
                f"{len(case_splits)} LES case(s) cannot fill split.ratios={ratios}. "
                "Add plume cases or change the ratios."
            )
    return context


# -- resources used by the draws ----------------------------------------------
class ContextResources:
    """Lazily opened scene geometries and plume columns of a context.

    Args:
        recipe: Recipe the context belongs to.
        context: Generation context.
        adapter: Sensor adapter; built from the registry when omitted.
    """

    def __init__(
        self,
        recipe: Recipe,
        context: GenerationContext,
        *,
        adapter: SensorAdapter | None = None,
    ) -> None:
        self.recipe = recipe
        self.context = context
        self._adapter = adapter
        self._geometries: dict[str, SceneGeometry] = {}
        self._plumes: dict[str, xr.Dataset] = {}

    @property
    def adapter(self) -> SensorAdapter:
        """Return the sensor adapter, loading it on first use."""
        if self._adapter is None:
            self._adapter = _adapter(self.recipe)
        return self._adapter

    def geometry(self, scene_id: str) -> SceneGeometry:
        """Return the projected pixel geometry of a scene (cached)."""
        if scene_id not in self._geometries:
            record = self.context.record(scene_id)
            adapter = self.adapter
            scene_geometry = getattr(adapter, "scene_geometry", None)
            if callable(scene_geometry):
                geometry = cast(SceneGeometry, scene_geometry(record))
            else:
                chunks = self.recipe.compute.chunks or None
                geometry = SceneGeometry.from_scene(
                    adapter.open_scene(record, chunks=chunks)
                )
            self._geometries[scene_id] = geometry
        return self._geometries[scene_id]

    def plume(self, uri: str) -> xr.Dataset:
        """Return an opened plume column (cached)."""
        if uri not in self._plumes:
            from spectra.plume.io import open_plume_column

            self._plumes[uri] = open_plume_column(uri)
        return self._plumes[uri]

    def snapshot(self, uri: str, time_index: int) -> xr.DataArray:
        """Return the unscaled ``(y, x)`` column of one snapshot."""
        from spectra.pipeline import snapshot_column

        return snapshot_column(self.plume(uri), time_index=time_index)

    def close(self) -> None:
        """Close the opened plume files."""
        for dataset in self._plumes.values():
            dataset.close()
        self._plumes.clear()


# -- sample parameters ---------------------------------------------------------
@dataclass(frozen=True)
class SampleParams:
    """All parameters of one sample.

    Attributes:
        sample_id: ``<recipe name>-<index:06d>``.
        index: Sample index.
        split: Split the sample belongs to.
        sensor: Sensor adapter name.
        scene_id: Background scene.
        background_group: Leakage group of the scene.
        plume_uri: Plume file (for negatives: the file defining the window).
        les_case: LES case (``none:<sample_id>`` for negatives).
        tracer: Tracer name (``none`` for negatives).
        source_id: Source identifier (``none`` for negatives).
        time_index: Snapshot index.
        simulation_time_s: Snapshot time since the LES start.
        emission_rate_kg_s: Embedded emission rate (0 for negatives).
        emission_scale: Factor applied to the column (0 for negatives).
        placement: Resolved placement with the anchor indices.
        window: Sample window on the sensor grid.
        is_negative: Whether the sample carries no plume.
        pair_id: Identity of the physical draw, shared by method twins.
        params_hash: Hash of recipe, context and index.
    """

    sample_id: str
    index: int
    split: str
    sensor: str
    scene_id: str
    background_group: str
    plume_uri: str
    les_case: str
    tracer: str
    source_id: str
    time_index: int
    simulation_time_s: float
    emission_rate_kg_s: float
    emission_scale: float
    placement: Placement
    window: Window
    is_negative: bool
    pair_id: str
    params_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return the flattened parameters in :data:`PARAMS_COLUMNS` order."""
        values: dict[str, Any] = {
            "sample_id": self.sample_id,
            "index": int(self.index),
            "split": self.split,
            "sensor": self.sensor,
            "scene_id": self.scene_id,
            "background_group": self.background_group,
            "plume_uri": self.plume_uri,
            "les_case": self.les_case,
            "tracer": self.tracer,
            "source_id": self.source_id,
            "time_index": int(self.time_index),
            "simulation_time_s": float(self.simulation_time_s),
            "emission_rate_kg_s": float(self.emission_rate_kg_s),
            "emission_scale": float(self.emission_scale),
            "anchor_along": int(self.placement.anchor_along),
            "anchor_across": int(self.placement.anchor_across),
            "anchor_corner": self.placement.anchor_corner,
            "rotation_deg": int(self.placement.rotation_deg),
            "flip": self.placement.flip,
            "window_across_start": int(self.window.across.start),
            "window_across_stop": int(self.window.across.stop),
            "window_along_start": int(self.window.along.start),
            "window_along_stop": int(self.window.along.stop),
            "is_negative": bool(self.is_negative),
            "pair_id": self.pair_id,
            "params_hash": self.params_hash,
        }
        return {name: values[name] for name in PARAMS_COLUMNS}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> SampleParams:
        """Rebuild the parameters from a flattened row (``pandas`` row or dict)."""
        missing = [name for name in PARAMS_COLUMNS if name not in mapping]
        if missing:
            raise ContractError(f"Parameter row lacks columns {missing}.")
        negative = bool(mapping["is_negative"])
        emission_scale = float(mapping["emission_scale"])
        placement = Placement(
            mode="anchor",
            anchor_along=int(mapping["anchor_along"]),
            anchor_across=int(mapping["anchor_across"]),
            anchor_corner=cast(AnchorCorner, str(mapping["anchor_corner"])),
            rotation_deg=int(mapping["rotation_deg"]),
            flip=cast(Flip, str(mapping["flip"])),
            emission_scale=1.0 if negative else emission_scale,
            time_index=int(mapping["time_index"]),
        )
        window = Window(
            across=slice(
                int(mapping["window_across_start"]), int(mapping["window_across_stop"])
            ),
            along=slice(
                int(mapping["window_along_start"]), int(mapping["window_along_stop"])
            ),
        )
        return cls(
            sample_id=str(mapping["sample_id"]),
            index=int(mapping["index"]),
            split=str(mapping["split"]),
            sensor=str(mapping["sensor"]),
            scene_id=str(mapping["scene_id"]),
            background_group=str(mapping["background_group"]),
            plume_uri=str(mapping["plume_uri"]),
            les_case=str(mapping["les_case"]),
            tracer=str(mapping["tracer"]),
            source_id=str(mapping["source_id"]),
            time_index=int(mapping["time_index"]),
            simulation_time_s=float(mapping["simulation_time_s"]),
            emission_rate_kg_s=float(mapping["emission_rate_kg_s"]),
            emission_scale=emission_scale,
            placement=placement,
            window=window,
            is_negative=negative,
            pair_id=str(mapping["pair_id"]),
            params_hash=str(mapping["params_hash"]),
        )


def sample_rng(seed: int, index: int) -> np.random.Generator:
    """Return the generator of sample ``index``: ``SeedSequence(seed, (index,))``."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(index,)))


def _draw_time_index(
    policy: TimePolicy, plume: PlumeEntry, rng: np.random.Generator
) -> int:
    if policy.mode == "uniform":
        # Ramped releases have snapshots without emission (before the release
        # starts); they carry no plume and cannot be scaled, so skip them.
        candidates = [
            i
            for i, time_s in enumerate(plume.simulation_time_s)
            if time_s >= policy.min_simulation_time_s
            and plume.emission_rate_kg_s[i] > 0
        ]
        if not candidates:
            raise ContractError(
                f"{plume.uri} has no snapshot with a positive emission rate at or "
                f"after plumes.time_policy.min_simulation_time_s="
                f"{policy.min_simulation_time_s} s."
            )
        return candidates[int(rng.integers(len(candidates)))]
    assert policy.indices is not None
    invalid = [i for i in policy.indices if i >= plume.n_time]
    if invalid:
        raise ContractError(
            f"plumes.time_policy.indices {invalid} exceed the {plume.n_time} "
            f"snapshots of {plume.uri}."
        )
    if policy.mode == "fixed":
        return int(policy.indices[0])
    return int(policy.indices[int(rng.integers(len(policy.indices)))])


def _draw_emission_kg_s(spec: EmissionSpec, rng: np.random.Generator) -> float:
    if spec.distribution == "uniform":
        assert spec.low is not None and spec.high is not None
        value = float(rng.uniform(spec.low, spec.high))
    elif spec.distribution == "loguniform":
        assert spec.low is not None and spec.high is not None
        value = float(np.exp(rng.uniform(np.log(spec.low), np.log(spec.high))))
    elif spec.distribution == "list":
        assert spec.values is not None
        value = float(spec.values[int(rng.integers(len(spec.values)))])
    else:
        assert spec.values is not None
        value = float(spec.values[0])
    return value * spec.to_kg_s


def _draw_rotation(spec: RotationSpec, rng: np.random.Generator) -> int:
    if spec.mode == "uniform_int":
        return int(rng.integers(spec.low, spec.high + 1))
    if spec.mode == "list":
        return int(spec.values[int(rng.integers(len(spec.values)))])
    return int(spec.value)


def _draw_flip(weights: Mapping[str, float], rng: np.random.Generator) -> Flip:
    axes = sorted(weights)
    total = float(sum(weights[axis] for axis in axes))
    probabilities = [float(weights[axis]) / total for axis in axes]
    return cast(Flip, str(rng.choice(np.asarray(axes, dtype=object), p=probabilities)))


def grow_window(
    window: Window, shape: tuple[int, int], scene_shape: tuple[int, int]
) -> Window:
    """Expand a window to at least ``shape`` (``n_across``, ``n_along``).

    The window grows symmetrically around the current one and is shifted
    back into the scene where it would cross the edge.

    Raises:
        ContractError: If ``shape`` exceeds the scene.
    """
    slices: list[slice] = []
    for current, size, limit in zip(
        (window.across, window.along), shape, scene_shape, strict=True
    ):
        if size > limit:
            raise ContractError(
                f"placement.window_shape {shape} exceeds the scene shape {scene_shape}."
            )
        length = current.stop - current.start
        if length >= size:
            slices.append(current)
            continue
        start = current.start - (size - length) // 2
        start = max(0, min(start, limit - size))
        slices.append(slice(start, start + size))
    return Window(across=slices[0], along=slices[1])


def _pair_id(values: Mapping[str, Any]) -> str:
    return _sha256(json.dumps(dict(values), sort_keys=True))[:16]


def derive_sample_params(
    recipe: Recipe,
    context: GenerationContext,
    index: int,
    *,
    resources: ContextResources | None = None,
) -> SampleParams:
    """Draw the parameters of sample ``index``.

    Draws happen in the fixed order split -> negative -> scene -> plume file
    -> time index -> emission -> rotation -> flip -> anchor corner -> anchor
    pixel, each from ``SeedSequence(recipe.seed, spawn_key=(index,))``.
    Negatives draw everything (so the stream is identical) and then drop the
    plume: zero emission, ``les_case="none:<sample_id>"``.

    Args:
        recipe: Validated recipe.
        context: Context from :func:`build_context`.
        index: Sample index in ``[0, recipe.n_samples)``.
        resources: Cached geometries and plume columns.

    Returns:
        The sample parameters.
    """
    from spectra.geometry.swath import place_in_scene

    if not 0 <= index < recipe.n_samples:
        raise ContractError(f"index {index} is outside [0, {recipe.n_samples}).")
    resources = resources or ContextResources(recipe, context)
    rng = sample_rng(recipe.seed, index)
    sample_id = f"{recipe.name}-{index:06d}"

    probabilities = [float(recipe.split.ratios[name]) for name in SPLITS]
    split = str(rng.choice(np.asarray(SPLITS, dtype=object), p=probabilities))
    is_negative = bool(rng.random() < recipe.negatives.fraction)
    scenes = context.scenes_in_split(split)
    if not scenes:
        raise ContractError(f"Split {split!r} has no eligible scene.")
    record = scenes[int(rng.integers(len(scenes)))]
    plumes = context.plumes_in_split(split)
    if not plumes:
        raise ContractError(f"Split {split!r} has no plume file.")
    plume = plumes[int(rng.integers(len(plumes)))]
    time_index = _draw_time_index(recipe.plumes.time_policy, plume, rng)
    rate = plume.emission_rate_kg_s[time_index]
    target_kg_s = _draw_emission_kg_s(recipe.plumes.emission, rng)
    if rate <= 0:
        raise ContractError(
            f"{plume.uri} has emission rate {rate} kg s-1 at time index "
            f"{time_index}; it cannot be scaled to a target rate."
        )
    emission_scale = target_kg_s / rate
    rotation = _draw_rotation(recipe.placement.rotations_deg, rng)
    flip = _draw_flip(recipe.placement.flips, rng)
    corner = cast(
        AnchorCorner,
        str(rng.choice(np.asarray(recipe.placement.anchor_corners, dtype=object))),
    )

    geometry = resources.geometry(record.scene_id)
    placed = place_in_scene(
        resources.snapshot(plume.uri, time_index),
        geometry,
        anchor_corner=corner,
        rotation_deg=rotation,
        flip=flip,
        min_outside_fraction=recipe.placement.min_outside_fraction,
        max_outside_fraction=recipe.placement.max_outside_fraction,
        rng=rng,
        max_attempts=recipe.placement.max_attempts,
    )
    placement = replace(
        placed.placement,
        emission_scale=1.0 if is_negative else emission_scale,
        time_index=time_index,
    )
    window = placed.window
    if recipe.placement.window_shape is not None:
        window = grow_window(window, recipe.placement.window_shape, geometry.shape)

    pair_id = _pair_id(
        {
            "sensor": record.sensor,
            "scene_id": record.scene_id,
            "plume_uri": "" if is_negative else plume.uri,
            "time_index": time_index,
            "emission_rate_kg_s": 0.0 if is_negative else round(target_kg_s, 12),
            "rotation_deg": placement.rotation_deg,
            "flip": placement.flip,
            "anchor_corner": placement.anchor_corner,
            "anchor_along": placement.anchor_along,
            "anchor_across": placement.anchor_across,
            "is_negative": is_negative,
        }
    )
    return SampleParams(
        sample_id=sample_id,
        index=index,
        split=split,
        sensor=record.sensor,
        scene_id=record.scene_id,
        background_group=record.background_group,
        plume_uri=plume.uri,
        les_case=f"{NEGATIVE_CASE_PREFIX}{sample_id}" if is_negative else plume.case_id,
        tracer="none" if is_negative else plume.tracer,
        source_id=NEGATIVE_SOURCE_ID if is_negative else plume.source_id,
        time_index=time_index,
        simulation_time_s=float(plume.simulation_time_s[time_index]),
        emission_rate_kg_s=0.0 if is_negative else float(target_kg_s),
        emission_scale=0.0 if is_negative else float(emission_scale),
        placement=placement,
        window=window,
        is_negative=is_negative,
        pair_id=pair_id,
        params_hash=_sha256(f"{context.recipe_hash}:{context.context_hash}:{index}"),
    )


def build_parameter_table(
    recipe: Recipe,
    context: GenerationContext,
    indices: Iterable[int] | None = None,
    *,
    resources: ContextResources | None = None,
) -> pd.DataFrame:
    """Return one row of flattened :class:`SampleParams` per sample."""
    resources = resources or ContextResources(recipe, context)
    selected = range(recipe.n_samples) if indices is None else list(indices)
    rows = [
        derive_sample_params(recipe, context, index, resources=resources).to_dict()
        for index in selected
    ]
    return pd.DataFrame.from_records(rows, columns=list(PARAMS_COLUMNS))


def read_parameter_table(path: str | Path) -> pd.DataFrame:
    """Read ``params.parquet`` and check its columns."""
    frame = pd.read_parquet(path)
    missing = [name for name in PARAMS_COLUMNS if name not in frame.columns]
    if missing:
        raise ContractError(f"{path} lacks parameter columns {missing}.")
    return frame


def params_from_table(frame: pd.DataFrame, index: int) -> SampleParams:
    """Return the parameters of sample ``index`` from a parameter table."""
    matches = frame.loc[frame["index"] == index]
    if len(matches) != 1:
        raise ContractError(f"Parameter table holds {len(matches)} rows for {index}.")
    row = matches.iloc[0]
    return SampleParams.from_dict({str(k): v for k, v in row.items()})


__all__ = [
    "CONTEXT_FILE",
    "NEGATIVE_CASE_PREFIX",
    "NEGATIVE_SOURCE_ID",
    "PARAMS_COLUMNS",
    "PARAMS_FILE",
    "ContextResources",
    "GenerationContext",
    "PlumeEntry",
    "SampleParams",
    "assign_splits",
    "build_context",
    "build_parameter_table",
    "derive_sample_params",
    "grow_window",
    "params_from_table",
    "read_parameter_table",
    "record_from_dict",
    "record_to_dict",
    "resolve_plume_files",
    "sample_rng",
]
